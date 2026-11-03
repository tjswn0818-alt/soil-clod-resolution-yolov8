"""
리뷰 대응 STEP 0 — SR matched 72모델 저-conf 재추론 (results_review_261026/).

평가 프로토콜은 unified_4cm_260914.py 를 import 해 그대로 쓴다.
  리사이즈        U.INTERP_IMG (INTER_LINEAR) / U.INTERP_MASK (INTER_NEAREST)
  GT             U.load_data() -> F.load_gt (원본 장축>=4cm) -> 평가 격자 NEAREST
  예측 필터       F.major_axis_px(m) >= F.MIN_MAJOR_FRAC * r32
  IoU            coco_mask.iou (F.match 내부와 동일 호출)
추론은 기존과 같은 model.predict(retina_masks=True, iou=U.IOU_NMS) 이고 다음만 바꾼다.
  conf 0.5 -> 0.001, max_det 300 -> 1000
  process_mask_native 를 64개 단위 청크로 실행 (마스크별 독립 연산 -> 결과 동일, 메모리 절약)
  NMS 에 return_idxs=True 를 걸어 앵커 인덱스 -> P3/P4/P5 레벨 귀속

저장 (셀 = dataset x res x seed, pickle 1개)
  예측마다 score, level, major_px, area, keep4cm, RLE
  이미지마다 4cm 통과 예측 x GT IoU 행렬
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
import ultralytics.utils.nms as unms
import ultralytics.models.yolo.segment.predict as segpred
from ultralytics.utils import ops

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import unified_4cm_260914 as U          # noqa: E402
import config  # noqa: E402  code/config.py (경로 설정)
F, S = U.F, U.S

OUT = config.REVIEW_DIR
PRED = OUT / "preds"
CONF_LOW = 0.001
MAX_DET = 1000
CHUNK = 64

# ── 패치 1: NMS 앵커 인덱스 저장 ──
_orig_nms = unms.non_max_suppression
_STASH = {}


def _nms_idx(*a, **k):
    k["return_idxs"] = True
    out, keepi = _orig_nms(*a, **k)
    _STASH["idx"] = [x.view(-1).long().cpu() for x in keepi]
    return out


unms.non_max_suppression = _nms_idx

# ── 패치 2: 마스크 청크 처리 + 빈 마스크 keep 벡터 저장 ──
_orig_pmn = ops.process_mask_native


def _pmn_chunked(protos, masks_in, bboxes, shape):
    outs = [_orig_pmn(protos, masks_in[i:i + CHUNK], bboxes[i:i + CHUNK], shape)
            for i in range(0, masks_in.shape[0], CHUNK)]
    m = torch.cat(outs, 0)
    _STASH["keep"] = (m.amax((-2, -1)) > 0).cpu()
    return m


segpred.ops.process_mask_native = _pmn_chunked


def level_of(idx, r32):
    n3, n4 = (r32 // 8) ** 2, (r32 // 16) ** 2
    return np.where(idx < n3, 3, np.where(idx < n3 + n4, 4, 5)).astype(np.int8)


def infer_cell(model, dd, r32, conf=CONF_LOW, max_det=MAX_DET):
    thr = F.MIN_MAJOR_FRAC * r32
    out = {}
    for f in sorted(dd["gt"]):
        g = dd["gt"][f]
        img_r = cv2.resize(dd["img"][f], (r32, r32), interpolation=U.INTERP_IMG)
        gts_r = [cv2.resize(m, (r32, r32), interpolation=U.INTERP_MASK)
                 for m in g["masks"]]
        _STASH.clear()
        r = model.predict(img_r, imgsz=r32, conf=conf, iou=U.IOU_NMS,
                          max_det=max_det, retina_masks=True, verbose=False,
                          device=0)[0]
        recs = []
        if r.masks is not None:
            idx = _STASH["idx"][0].numpy()
            if "keep" in _STASH and not bool(_STASH["keep"].all()):
                idx = idx[_STASH["keep"].numpy()]
            lv = level_of(idx, r32)
            sc = r.boxes.conf.cpu().numpy().astype(np.float32)
            md = r.masks.data
            assert len(idx) == len(sc) == md.shape[0]
            for c0 in range(0, md.shape[0], CHUNK):
                blk = md[c0:c0 + CHUNK].cpu().numpy()
                for j, pm in enumerate(blk):
                    m = (pm > 0.5).astype(np.uint8)          # 기존 eval_cell 과 동일
                    if m.shape != (r32, r32):
                        m = cv2.resize(m, (r32, r32), interpolation=U.INTERP_MASK)
                    a = int(m.sum())
                    if a == 0:
                        continue
                    mj = F.major_axis_px(m)
                    k = c0 + j
                    recs.append(dict(score=float(sc[k]), level=int(lv[k]),
                                     major_px=mj, area=a, keep4cm=bool(mj >= thr),
                                     rle=coco_mask.encode(np.asfortranarray(m))))
            del md
        kept = [x for x in recs if x["keep4cm"]]
        n = len(gts_r)
        if kept and n:
            ious = coco_mask.iou([x["rle"] for x in kept],
                                 [coco_mask.encode(np.asfortranarray(m)) for m in gts_r],
                                 [0] * n)
            ious = np.asarray(ious, dtype=np.float64).reshape(len(kept), n)
        else:
            ious = np.zeros((len(kept), n))
        out[f] = dict(n_gt=n, preds=recs, kept_idx=[i for i, x in enumerate(recs)
                                                    if x["keep4cm"]], iou=ious,
                      gt_rle=[coco_mask.encode(np.asfortranarray(m)) for m in gts_r])
    return out


def main():
    PRED.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:]          # 예: 512:0  (디버그용)
    D = U.load_data()
    for tr in U.TRAIN_RES:
        r32 = int(np.ceil(tr / 32) * 32)
        for sd in U.SEEDS:
            if only and f"{tr}:{sd}" not in only:
                continue
            todo = [ds for ds in U.DATASETS
                    if not (PRED / f"{ds}_{tr}_s{sd}.pkl").exists()]
            if not todo:
                continue
            model = YOLO(str(S.ckpt_of(tr, sd)))
            t0 = time.time()
            for ds in todo:
                cell = infer_cell(model, D[ds], r32)
                tmp = PRED / f"{ds}_{tr}_s{sd}.pkl.tmp"
                with open(tmp, "wb") as fh:
                    pickle.dump(dict(dataset=ds, train_res=tr, r32=r32, seed=sd,
                                     ckpt=str(S.ckpt_of(tr, sd)), images=cell), fh)
                tmp.replace(PRED / f"{ds}_{tr}_s{sd}.pkl")
            del model
            torch.cuda.empty_cache()
            npred = sum(len(v["preds"]) for v in cell.values())
            print(f"  R{tr} s{sd} | {todo} {time.time() - t0:6.1f}s "
                  f"(마지막 셋 예측 {npred}개)", flush=True)


if __name__ == "__main__":
    main()
