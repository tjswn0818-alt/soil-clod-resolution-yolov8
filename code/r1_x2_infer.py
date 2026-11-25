"""R1 추가 실험 2 추론 (추가 시험셋 20장) — review_infer_261026.infer_cell 을 import 해 그대로 사용 (수정 없음).
  리사이즈 INTER_LINEAR(이미지)/INTER_NEAREST(GT), predict retina_masks, iou 0.7, conf 0.001, max_det 1000.
  셀 = (집합, 모델, 추론 해상도, seed). 저장은 P0 평가·KS 에 필요한 값만:
    score > 0.5 & 장축 >= 4 cm 예측의 score, area(평가 격자), IoU(vs GT>=4cm), ED_cm(원본 격자 NEAREST 업샘플)
    GT(>=4cm) ED_cm, n_pred_raw(전체), n_pred_4cm_conf(> 0.5)
집합: new20 (추가 시험셋 20장, config.EXTRA_TEST_ROOT, 전 조건 8 seed) / test20 (기준선, seed 0 만)
조건: MR 6개 범위 x 추론 448/512/736/1024, SR-448/512/736/1024 (같은 해상도), 보조 SR-1600@1600
사용: python r1_x2_infer.py base | main
"""
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config                             # noqa: E402  code/config.py (경로 설정)
import review_infer_261026 as RI          # noqa: E402  (패치 적용)

U, F, S = RI.U, RI.F, RI.S
M = U.M
OUT = config.R1_EXTRA_DIR
CACHE = OUT / "cache" / "x2"
CACHE.mkdir(parents=True, exist_ok=True)
DIRS = {"new20": config.EXTRA_TEST_ROOT, "test20": F.DATASETS["test"]}
T = 0.5
SEEDS = list(range(8))
EVALS = [448, 512, 736, 1024]
MR = [("MR[128-1600]", "_128_1600"), ("MR[384-1600]", "_384_1600"), ("MR[448-1600]", "_448_1600"),
      ("MR[448-1280]", "_448_1280"), ("MR[448-1024]", "_448_1024"), ("MR[448-736]", "_448_736")]
SR = [("SR-448", 448, 448), ("SR-512", 512, 512), ("SR-736", 720, 736), ("SR-1024", 1024, 1024)]
SR_AUX = [("SR-1600", 1600, 1600)]


def jobs(sets, seeds):
    """(모델 라벨, ckpt, [(set, eval_res), ...], seed)"""
    out = []
    for sd in seeds:
        for lbl, tag in MR:
            out.append((lbl, M.ckpt_of(tag, sd), [(s, ev) for s in sets for ev in EVALS], sd))
        for lbl, tr, ev in SR + SR_AUX:
            out.append((lbl, S.ckpt_of(tr, sd), [(s, ev) for s in sets], sd))
    return out


def cpath(st, lbl, ev, sd):
    return CACHE / f"{st}_{lbl.replace('[', '').replace(']', '')}_{ev}_s{sd}.pkl"


def load_set(st):
    gt = F.load_gt(DIRS[st])
    return dict(gt=gt, img={f: cv2.imread(str(DIRS[st] / f)) for f in sorted(gt)})


def compact(cell, dd):
    res = {}
    for f, im in cell.items():
        g = dd["gt"][f]
        H, W, cpp = g["H"], g["W"], g["cpp"]
        kept = [im["preds"][i] for i in im["kept_idx"]]
        sel = np.array([p["score"] > T for p in kept], bool)
        ed = []
        for p, s in zip(kept, sel):
            if s:
                m = cv2.resize(coco_mask.decode(p["rle"]), (W, H), interpolation=U.INTERP_MASK)
                ed.append(F.shape_stats(m, cpp)[1])
        res[f] = dict(score=np.array([p["score"] for p in kept])[sel] if len(kept) else np.zeros(0),
                      area=np.array([p["area"] for p in kept], float)[sel] if len(kept) else np.zeros(0),
                      iou=im["iou"][sel] if len(kept) else np.zeros((0, im["n_gt"])),
                      ed=np.array(ed), n_gt=im["n_gt"],
                      gt_ed=np.array([F.shape_stats(m, cpp)[1] for m in g["masks"]]),
                      n_raw=len(im["preds"]), n_raw_conf=int(sum(p["score"] > T for p in im["preds"])))
    return res


def main(mode):
    sets, seeds = {"base": (["test20"], [0]), "main": (["new20"], SEEDS)}[mode]
    D = {s: load_set(s) for s in sets}
    for s in sets:
        print(s, len(D[s]["gt"]), "images, GT>=4cm", sum(len(v["masks"]) for v in D[s]["gt"].values()),
              flush=True)
    t00 = time.time()
    for lbl, ck, cells, sd in jobs(sets, seeds):
        todo = [(s, ev) for s, ev in cells if not cpath(s, lbl, ev, sd).exists()]
        if not todo:
            continue
        assert ck.exists(), ck
        model = YOLO(str(ck))
        t0 = time.time()
        for s, ev in todo:
            cell = RI.infer_cell(model, D[s], ev)
            p = cpath(s, lbl, ev, sd)
            with open(str(p) + ".tmp", "wb") as fh:
                pickle.dump(dict(set=s, model=lbl, eval_res=ev, seed=sd, ckpt=config.rel(ck),
                                 images=compact(cell, D[s])), fh)
            Path(str(p) + ".tmp").replace(p)
        del model
        torch.cuda.empty_cache()
        print(f"  {lbl:>13} s{sd} {len(todo)} cells {time.time() - t0:6.1f}s (total {time.time() - t00:7.1f}s)",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
