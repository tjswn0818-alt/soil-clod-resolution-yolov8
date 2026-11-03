"""
리뷰 대응 추가 — NMS IoU 임계값(0.5 / 0.6 / 0.7)에 따른 검출 지표와 해상도 순위.

raw 가로채기: review_prenms_level_261026.py 와 같은 방식 (NMS 직전 prediction 복제,
             construct_results 직전 proto 복제). 이미지당 forward 1회.
평가 경로: U.eval_cell 과 동일 —
  predict(conf=U.CONF 0.5, iou=0.7, retina_masks, max_det 기본 300) 를 실제로 호출하고,
  가로챈 raw 에 같은 NMS 인자(conf 0.5, iou t, classes None, agnostic False, max_det 300,
  nc 1)를 다시 건 뒤 construct_result 와 같은 scale_boxes -> process_mask_native ->
  빈 마스크 제거 -> (pm > 0.5) -> 장축 >= 4cm 필터 -> F.match.
검증: t = 0.7 재구성 마스크 == predict 출력 마스크 (이미지마다), 그리고
      이미지별 TP/FN/FP == unified_4cm_perimage_260914 (SR, test).
대상: SR matched 9 해상도 x 8 seed, test 20장.
요약: seed 별 Recall/Precision -> 8 seed 평균 ± SD(n-1), FN/FP per image (Table 3 규약).
bootstrap (참고): B=10,000, rng 2026, 이미지·seed 2단계, 전 해상도·전 임계값 같은 인덱스.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import ultralytics.utils.nms as unms
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import ops

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import review_analysis_261026 as R      # noqa: E402
import review_common_261026 as C        # noqa: E402

U, F, S = C.U, C.F, C.U.S
OUT = C.OUT
SFX = "271026"
DS = "test"
TS = [0.5, 0.6, 0.7]
MAX_DET = 300                        # predict 기본값 (eval_cell 은 지정하지 않음)
RAW = OUT / f"nms_iou_perimage_{SFX}.csv"
HDR = ["dataset", "train_res", "eval_res", "seed", "nms_iou", "file", "n_gt",
       "n_pred_raw", "n_pred_4cm", "TP", "FN", "FP"]
B = 10000
PAIRS = [(1600, 512), (1600, 1024), (1280, 512), (1024, 512), (448, 512), (736, 512)]

# ── raw 가로채기 (review_prenms_level_261026 과 동일) ──
_STASH = {}
_orig_nms = unms.non_max_suppression


def _nms_stash(prediction, *a, **k):
    p = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
    _STASH["raw"] = p.detach().clone()
    return _orig_nms(prediction, *a, **k)


unms.non_max_suppression = _nms_stash
_orig_cr = SegmentationPredictor.construct_results


def _cr_stash(self, preds, img, orig_imgs, protos):
    _STASH["protos"] = protos[0].detach().clone()
    return _orig_cr(self, preds, img, orig_imgs, protos)


SegmentationPredictor.construct_results = _cr_stash


def masks_at(raw, protos, r32, t):
    """construct_result(retina) 와 같은 경로로 NMS(iou=t) 후 이진 마스크 목록."""
    pred = _orig_nms(raw[None].clone(), U.CONF, t, None, False, max_det=MAX_DET,
                     nc=1, end2end=False, rotated=False)[0]
    if pred.shape[0] == 0:
        return []
    pred[:, :4] = ops.scale_boxes((r32, r32), pred[:, :4], (r32, r32, 3))
    m = ops.process_mask_native(protos, pred[:, 6:], pred[:, :4], (r32, r32))
    m = m[m.amax((-2, -1)) > 0]
    return list(m.cpu().numpy())


def binarize(pms, r32):
    out = []
    for pm in pms:                                   # eval_cell 과 동일
        m = (pm > 0.5).astype(np.uint8)
        if m.shape != (r32, r32):
            m = cv2.resize(m, (r32, r32), interpolation=U.INTERP_MASK)
        if m.sum():
            out.append(m)
    return out


def run():
    done = {(int(r["train_res"]), int(r["seed"])) for r in U.load_rows(RAW)}
    D = U.load_data()[DS]
    ref = {(int(r["eval_res"]), int(r["seed"]), r["file"]): r
           for r in U.load_rows(U.OUT / "unified_4cm_perimage_260914.csv")
           if r["dataset"] == DS and r["family"] == "SR"}
    nbad_mask = nbad_ref = 0
    for tr in U.TRAIN_RES:
        r32 = C.GRID[tr]
        thr = F.MIN_MAJOR_FRAC * r32
        for sd in U.SEEDS:
            if (tr, sd) in done:
                continue
            model = YOLO(str(S.ckpt_of(tr, sd)))
            t0 = time.time()
            rows = []
            for f in sorted(D["gt"]):
                g = D["gt"][f]
                img_r = cv2.resize(D["img"][f], (r32, r32), interpolation=U.INTERP_IMG)
                gts_r = [cv2.resize(m, (r32, r32), interpolation=U.INTERP_MASK)
                         for m in g["masks"]]
                _STASH.clear()
                r = model.predict(img_r, imgsz=r32, conf=U.CONF, iou=U.IOU_NMS,
                                  retina_masks=True, verbose=False, device=0)[0]
                pipe = binarize(r.masks.data.cpu().numpy(), r32) \
                    if r.masks is not None else []
                raw, protos = _STASH["raw"][0], _STASH["protos"]
                for t in TS:
                    allm = binarize(masks_at(raw, protos, r32, t), r32)
                    if t == U.IOU_NMS:                       # 검증 1: 파이프라인 재현
                        same = len(allm) == len(pipe) and all(
                            np.array_equal(a, b) for a, b in zip(allm, pipe))
                        nbad_mask += not same
                    preds = [m for m in allm if F.major_axis_px(m) >= thr]
                    _, _, tp = F.match(preds, gts_r)
                    row = [DS, tr, r32, sd, t, f, len(gts_r), len(allm), len(preds),
                           tp, len(gts_r) - tp, len(preds) - tp]
                    if t == U.IOU_NMS:                       # 검증 2: unified 이미지별
                        q = ref[(r32, sd, f)]
                        nbad_ref += [int(q["TP"]), int(q["FN"]), int(q["FP"])] != row[-3:]
                    rows.append(row)
            U.append_rows(RAW, HDR, rows)
            del model
            torch.cuda.empty_cache()
            print(f"  R{tr} s{sd} {time.time() - t0:5.1f}s | 누적 불일치: 마스크 "
                  f"{nbad_mask}, unified TP/FN/FP {nbad_ref}", flush=True)
    return nbad_mask, nbad_ref


def summarize():
    rows = U.load_rows(RAW)
    files = sorted({r["file"] for r in rows})
    RES, SD, nI = list(U.TRAIN_RES), list(U.SEEDS), len(files)
    fi = {f: i for i, f in enumerate(files)}
    shp = (len(TS), len(RES), len(SD), nI)
    TP, FN, FP = np.zeros(shp), np.zeros(shp), np.zeros(shp)
    for r in rows:
        k = (TS.index(float(r["nms_iou"])), RES.index(int(r["train_res"])),
             SD.index(int(r["seed"])), fi[r["file"]])
        TP[k], FN[k], FP[k] = int(r["TP"]), int(r["FN"]), int(r["FP"])
    assert len(rows) == np.prod(shp)

    def met(tp, fn, fp, n):
        with np.errstate(invalid="ignore", divide="ignore"):
            return dict(Recall=tp / (tp + fn),
                        Precision=np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0),
                        FN_per_img=fn / n, FP_per_img=fp / n)
    # 점추정 (seed 별 -> 평균 ± SD)
    ps = met(TP.sum(-1), FN.sum(-1), FP.sum(-1), nI)          # (T,R,S)
    # Table 3 대조 (t = 0.7)
    t3 = {int(r["train_res"]): r for r in U.load_rows(
        U.OUT / "unified_4cm_sr_summary_260914.csv") if r["dataset"] == DS}
    summ, chk = [], []
    for a, t in enumerate(TS):
        rec = ps["Recall"][a].mean(-1)
        order = np.argsort(-rec)
        rank = {RES[j]: k + 1 for k, j in enumerate(order)}
        for j, tr in enumerate(RES):
            row = [DS, t, tr, C.GRID[tr], rank[tr]]
            for k in ("Recall", "Precision", "FN_per_img", "FP_per_img"):
                v = ps[k][a, j]
                row += [round(float(v.mean()), 6), round(float(v.std(ddof=1)), 6)]
                if t == U.IOU_NMS:
                    old = float(t3[tr][f"{k}_mean"])
                    chk.append([tr, k, round(float(v.mean()), 6), old,
                                int(abs(v.mean() - old) < 5e-7)])
            row += [round(float(FP[a, j].sum(-1).mean() + TP[a, j].sum(-1).mean()), 2)]
            summ.append(row)
    R.wcsv(f"nms_iou_summary_{SFX}.csv",
           ["dataset", "nms_iou", "train_res", "eval_res", "recall_rank",
            "Recall_mean", "Recall_sd", "Precision_mean", "Precision_sd",
            "FN_per_img_mean", "FN_per_img_sd", "FP_per_img_mean", "FP_per_img_sd",
            "n_pred_4cm_mean"], summ)
    R.wcsv(f"nms_iou_check_table3_{SFX}.csv",
           ["train_res", "metric", "new_iou07", "table3", "equal_1e-6"], chk)

    # ── bootstrap: 2단계(이미지+seed), 전 해상도·임계값 같은 인덱스 ──
    rng = np.random.default_rng(R.RNG_SEED)
    II = rng.integers(0, nI, (B, nI))
    SS = rng.integers(0, len(SD), (B, len(SD)))
    W = np.zeros((B, nI))
    np.add.at(W, (np.repeat(np.arange(B), nI), II.ravel()), 1)
    sb = lambda X: X @ W.T                                    # (T,R,S,B)
    bm = met(sb(TP), sb(FN), sb(FP), nI)
    two = {k: np.nanmean(v[:, :, SS.T, np.arange(B)], 2) for k, v in bm.items()}
    one = {k: np.nanmean(v, 2) for k, v in bm.items()}         # (T,R,B)
    ci = lambda x: np.nanpercentile(x, [2.5, 97.5])
    prow = []
    for a, t in enumerate(TS):
        for hi, lo in PAIRS:
            j1, j0 = RES.index(hi), RES.index(lo)
            for k in ("Recall", "Precision", "FN_per_img", "FP_per_img"):
                d = ps[k][a, j1].mean() - ps[k][a, j0].mean()
                d2 = two[k][a, j1] - two[k][a, j0]
                d1 = one[k][a, j1] - one[k][a, j0]
                c2, c1 = ci(d2), ci(d1)
                prow.append([DS, t, f"{hi}-{lo}", k, round(float(d), 5),
                             round(c2[0], 5), round(c2[1], 5),
                             int(c2[0] > 0 or c2[1] < 0), round(c1[0], 5),
                             round(c1[1], 5), round(float(np.mean(d2 > 0)), 4),
                             int(np.isnan(d2).sum())])
    # 임계값 간 변화 (같은 해상도): t - 0.7
    for a, t in enumerate(TS[:-1]):
        for j, tr in enumerate(RES):
            for k in ("Recall", "Precision", "FN_per_img", "FP_per_img"):
                d = ps[k][a, j].mean() - ps[k][-1, j].mean()
                d2 = two[k][a, j] - two[k][-1, j]
                d1 = one[k][a, j] - one[k][-1, j]
                c2, c1 = ci(d2), ci(d1)
                prow.append([DS, f"{t}-0.7", f"{tr}", k, round(float(d), 5),
                             round(c2[0], 5), round(c2[1], 5),
                             int(c2[0] > 0 or c2[1] < 0), round(c1[0], 5),
                             round(c1[1], 5), round(float(np.mean(d2 > 0)), 4),
                             int(np.isnan(d2).sum())])
    R.wcsv(f"nms_iou_pairdiff_{SFX}.csv",
           ["dataset", "nms_iou", "contrast", "metric", "diff", "CI2_lo", "CI2_hi",
            "CI2_excl0", "CI1_lo", "CI1_hi", "P_diff_gt0", "n_rep_nan"], prow)

    # 순위 안정성: 해상도별 Recall 순위 분포 (2단계 재표집)
    rk = []
    for a, t in enumerate(TS):
        rb = two["Recall"][a]                                  # (R,B)
        ranks = (-rb).argsort(0).argsort(0) + 1
        for j, tr in enumerate(RES):
            rk.append([DS, t, tr, int(summ[a * len(RES) + j][4]),
                       round(float(np.median(ranks[j])), 1),
                       int(np.percentile(ranks[j], 2.5)),
                       int(np.percentile(ranks[j], 97.5)),
                       round(float(np.mean(ranks[j] == len(RES))), 4)])
    R.wcsv(f"nms_iou_rank_{SFX}.csv",
           ["dataset", "nms_iou", "train_res", "rank_point", "rank_boot_median",
            "rank_CI_lo", "rank_CI_hi", "P_last"], rk)
    return summ, chk, prow


if __name__ == "__main__":
    if "--report" not in sys.argv:
        print("불일치(마스크, unified):", run())
    s, c, _ = summarize()
    print("Table 3 대조 불일치:", sum(1 - x[-1] for x in c), "/", len(c))
    for r in s:
        print(r)
