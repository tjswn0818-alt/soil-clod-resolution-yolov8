"""
인식 -> 파쇄도 평가 연결 분석 (YOLOv8m-seg, matched 조건).

목적
  분할 성능(Recall/mIoU) 이 아니라, 실제 농학적 산출물인 '파쇄도 지표'를
  예측이 얼마나 재현하는지 본다.

대상
  - 데이터셋: test(18.v1i, 20장) 및 val(pixel_split/val, 24장)
  - clod 기준: 장축 >= 4cm  (MIN_MAJOR_FRAC = 176/2200 = 0.08, 프레임 50cm)
  - 모델: SR-512 @ eval 512, SR-736 @ eval 736  (matched: train=eval)
  - 예측 conf >= 0.5, 매칭 IoU >= 0.5, 예측 면적 내림차순 greedy
    (eval_pixel_final.py::match 와 동일)

분석
  1) 이미지별 clod 개수  GT vs Pred   -> MAE / RMSE / R2 / bias
  2) 등가직경 ED = 2*sqrt(area/pi) 분포 -> KS test, Wasserstein distance
  3) 4cm+ clod 면적비 (프레임 대비, 전체 clod 면적 대비) -> MAE / RMSE / R2
     = 파쇄도(큰 덩어리 비율) 프록시

면적/직경은 모두 원본 이미지 격자로 되돌린 뒤 cm 환산 (cm/px = 50 / max(H,W)).

출력: results/frag/  (CSV 5종 + PNG 4종)
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask
from ultralytics import YOLO

import config  # noqa: E402  code/config.py (경로 설정)
RS_ROOT = config.RS_ROOT
PIXEL_ROOT = config.PIXEL_ROOT
WEIGHTS = RS_ROOT / "weights"
OUT = RS_ROOT / "results" / "frag"
FIG = RS_ROOT / "figures"

DATASETS = {
    "test": PIXEL_ROOT / "18.v1i.coco-segmentation" / "train",
    "val": PIXEL_ROOT / "pixel_split" / "val",
}
MODELS = [("SR-512", 512), ("SR-736", 736)]   # (라벨, 학습 해상도) — eval = ceil(r/32)*32

FRAME_CM = 50.0
PX_PER_CM = 44.0
MIN_MAJOR_PX = 176.0
REF_SIDE = PX_PER_CM * FRAME_CM               # 2200
MIN_MAJOR_FRAC = MIN_MAJOR_PX / REF_SIDE      # 0.08
IOU_THR = 0.5
CONF = 0.5


# ─────────────────────────── 공통 유틸 ───────────────────────────
def seg_to_mask(seg, h, w):
    if isinstance(seg, list):
        rle = coco_mask.merge(coco_mask.frPyObjects(seg, h, w))
    elif isinstance(seg.get("counts"), list):
        rle = coco_mask.frPyObjects(seg, h, w)
    else:
        rle = seg
    return coco_mask.decode(rle).astype(np.uint8)


def major_axis_px(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    if len(c) >= 5:
        try:
            return float(max(cv2.fitEllipse(c)[1]))
        except Exception:
            pass
    r = cv2.minAreaRect(c)
    return float(max(r[1])) if r[1][0] > 0 and r[1][1] > 0 else 0.0


def match(preds, gts):
    """eval_pixel_final.py::match 와 동일 — 예측 면적 내림차순 greedy, IoU>=0.5."""
    n = len(gts)
    taken = [False] * n
    pairs = {}
    if not preds or n == 0:
        return taken, pairs, 0
    ious = coco_mask.iou([coco_mask.encode(np.asfortranarray(m)) for m in preds],
                         [coco_mask.encode(np.asfortranarray(m)) for m in gts],
                         [0] * n)
    areas = [int(m.sum()) for m in preds]
    tp = 0
    for di in sorted(range(len(preds)), key=lambda i: areas[i], reverse=True):
        gi, best = -1, IOU_THR
        for g in range(n):
            if not taken[g] and ious[di, g] >= best:
                best, gi = ious[di, g], g
        if gi >= 0:
            taken[gi] = True
            pairs[gi] = (di, float(ious[di, gi]))
            tp += 1
    return taken, pairs, tp


def load_gt(d):
    """원본 격자에서 GT 로드. 4cm 필터 전/후 모두 보관."""
    coco = json.load(open(d / "_annotations.coco.json", encoding="utf-8"))
    ann_by = {}
    for a in coco["annotations"]:
        ann_by.setdefault(a["image_id"], []).append(a)
    out = {}
    for im in coco["images"]:
        H, W = im["height"], im["width"]
        side = max(H, W)
        thr = MIN_MAJOR_FRAC * side
        cpp = FRAME_CM / side
        big, all_area = [], 0.0
        for a in ann_by.get(im["id"], []):
            m = seg_to_mask(a["segmentation"], H, W).astype(np.uint8)
            ar = int(m.sum())
            if ar == 0:
                continue
            all_area += ar * cpp ** 2
            if major_axis_px(m) >= thr:
                big.append(m)
        out[im["file_name"]] = dict(masks=big, H=H, W=W, side=side, cpp=cpp,
                                    total_clod_cm2=all_area)
    return out


def shape_stats(mask, cpp):
    """원본 격자 마스크 -> (면적 cm2, 등가직경 cm, 장축 cm)."""
    a_px = float(mask.sum())
    a_cm2 = a_px * cpp ** 2
    ed = 2.0 * np.sqrt(a_cm2 / np.pi)
    return a_cm2, ed, major_axis_px(mask) * cpp


# ─────────────────────────── 추론 ───────────────────────────
def run_model(ck, gt_by_file, img_dir, eval_r, interp=cv2.INTER_AREA, model=None,
              img_cache=None):
    """matched 조건 추론. 예측 마스크는 원본 격자로 되돌려 반환.

    선택 인자 (기본값은 종전 동작과 완전히 동일 — 기존 호출부 영향 없음)
      interp    : 이미지 리사이즈 보간. 기본 INTER_AREA.
                  MR 인식 평가(eval_mr_seeds_260825)와 맞추려면 INTER_LINEAR.
                  마스크는 종전대로 항상 INTER_NEAREST.
      model     : 미리 로드한 YOLO 객체. 주면 재사용하고 해제하지 않는다
                  (같은 가중치를 여러 eval 해상도로 돌릴 때 로드 비용 절약).
      img_cache : {file_name: 원본 BGR ndarray} 캐시. 주면 imread 를 건너뛴다.
    """
    r32 = int(np.ceil(eval_r / 32) * 32)
    thr_eval = MIN_MAJOR_FRAC * r32
    own = model is None
    if own:
        model = YOLO(str(ck))
    res_all = {}
    for f, g in gt_by_file.items():
        img = img_cache[f] if img_cache is not None else cv2.imread(str(img_dir / f))
        img_r = cv2.resize(img, (r32, r32), interpolation=interp)
        gts_r = [cv2.resize(m, (r32, r32), interpolation=cv2.INTER_NEAREST)
                 for m in g["masks"]]
        r = model.predict(img_r, imgsz=r32, conf=CONF, retina_masks=True,
                          verbose=False, device=0)[0]
        preds_r, preds_all_r = [], []
        if r.masks is not None:
            for pm in r.masks.data.cpu().numpy():
                m = (pm > 0.5).astype(np.uint8)
                if m.shape != (r32, r32):
                    m = cv2.resize(m, (r32, r32), interpolation=cv2.INTER_NEAREST)
                if m.sum() == 0:
                    continue
                preds_all_r.append(m)
                if major_axis_px(m) >= thr_eval:
                    preds_r.append(m)
        flags, pairs, tp = match(preds_r, gts_r)
        # 원본 격자로 복원 (면적/직경은 원본 해상도 기준 cm 환산)
        H, W = g["H"], g["W"]
        up = lambda m: cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        res_all[f] = dict(preds=[up(m) for m in preds_r],
                          pred_all_area_cm2=sum(float(up(m).sum()) * g["cpp"] ** 2
                                                for m in preds_all_r),
                          flags=flags, pairs=pairs, tp=tp,
                          n_pred_all=len(preds_all_r))
    if own:
        del model
        torch.cuda.empty_cache()
    return res_all, r32


# ─────────────────────────── 지표 ───────────────────────────
def agree(y_gt, y_pr):
    y_gt, y_pr = np.asarray(y_gt, float), np.asarray(y_pr, float)
    d = y_pr - y_gt
    sse = float((d ** 2).sum())
    sst = float(((y_gt - y_gt.mean()) ** 2).sum())
    r2_11 = 1 - sse / sst if sst > 0 else float("nan")     # 1:1 라인 기준
    r = float(np.corrcoef(y_gt, y_pr)[0, 1]) if len(y_gt) > 2 else float("nan")
    sl, ic = np.polyfit(y_gt, y_pr, 1) if len(y_gt) > 2 else (np.nan, np.nan)
    nz = np.abs(y_gt) > 1e-9                      # GT=0 인 이미지는 MAPE 정의 불가
    mape = float(np.mean(np.abs(d[nz]) / np.abs(y_gt[nz])) * 100) if nz.any() else np.nan
    return dict(n=len(y_gt), n_mape=int(nz.sum()),
                gt_mean=float(y_gt.mean()), pred_mean=float(y_pr.mean()),
                bias=float(d.mean()), MAE=float(np.abs(d).mean()),
                RMSE=float(np.sqrt((d ** 2).mean())), R2_1to1=r2_11,
                r_pearson=r, R2_fit=r * r if np.isfinite(r) else float("nan"),
                slope=float(sl), intercept=float(ic), MAPE=mape)


# ─────────────────────────── 본체 ───────────────────────────
def main():
    from scipy import stats
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="test,val")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    use = [s.strip() for s in args.sets.split(",")]

    per_img, inst, paired = [], [], []
    print("=" * 108)
    print("추론 (matched: train=eval, conf>=0.5, IoU>=0.5, clod 장축>=4cm)")
    print("=" * 108)
    for ds in use:
        d = DATASETS[ds]
        gt = load_gt(d)
        files = sorted(gt)
        ngt = sum(len(gt[f]["masks"]) for f in files)
        print(f"\n[{ds}] {d}\n  이미지 {len(files)}장 | GT clod(4cm+) {ngt}개")
        for lbl, tr in MODELS:
            ck = WEIGHTS / "sr" / f"R{tr}.pt"
            if not ck.exists():
                print(f"  [skip] {ck}")
                continue
            R, r32 = run_model(ck, gt, d, tr)
            TP = sum(R[f]["tp"] for f in files)
            NP = sum(len(R[f]["preds"]) for f in files)
            print(f"  {lbl} (eval {r32}) | GT {ngt} Pred {NP} TP {TP} "
                  f"| Recall {TP/ngt:.3f} Precision {TP/NP if NP else 0:.3f}")
            for f in files:
                g, r = gt[f], R[f]
                cpp = g["cpp"]
                gs = [shape_stats(m, cpp) for m in g["masks"]]
                ps = [shape_stats(m, cpp) for m in r["preds"]]
                frame_cm2 = FRAME_CM ** 2
                ga = sum(x[0] for x in gs)
                pa = sum(x[0] for x in ps)
                per_img.append([ds, lbl, f, len(gs), len(ps),
                                round(ga, 3), round(pa, 3),
                                round(ga / frame_cm2, 5), round(pa / frame_cm2, 5),
                                round(g["total_clod_cm2"], 3),
                                round(r["pred_all_area_cm2"], 3),
                                round(ga / g["total_clod_cm2"], 5)
                                if g["total_clod_cm2"] > 0 else "",
                                round(pa / r["pred_all_area_cm2"], 5)
                                if r["pred_all_area_cm2"] > 0 else "",
                                r["tp"], len(gs) - r["tp"], len(ps) - r["tp"]])
                inv = {di: gi for gi, (di, _) in r["pairs"].items()}
                for i, (a, e, mj) in enumerate(gs):
                    inst.append([ds, lbl, f, "GT", i, round(a, 4), round(e, 4),
                                 round(mj, 4), int(r["flags"][i])])
                for i, (a, e, mj) in enumerate(ps):
                    inst.append([ds, lbl, f, "Pred", i, round(a, 4), round(e, 4),
                                 round(mj, 4), int(i in inv)])
                for gi, (di, iou) in r["pairs"].items():
                    paired.append([ds, lbl, f, gi, di, round(iou, 4),
                                   round(gs[gi][0], 4), round(ps[di][0], 4),
                                   round(gs[gi][1], 4), round(ps[di][1], 4),
                                   round(gs[gi][2], 4), round(ps[di][2], 4)])

    with open(OUT / "frag_per_image.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "file", "n_clod_GT", "n_clod_Pred",
                    "area4cm_GT_cm2", "area4cm_Pred_cm2",
                    "ratio_frame_GT", "ratio_frame_Pred",
                    "area_allclod_GT_cm2", "area_allclod_Pred_cm2",
                    "ratio_ofclod_GT", "ratio_ofclod_Pred", "TP", "FN", "FP"])
        w.writerows(per_img)
    with open(OUT / "frag_instances.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "file", "source", "idx", "area_cm2",
                    "ed_cm", "major_cm", "matched"])
        w.writerows(inst)
    with open(OUT / "frag_matched_pairs.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "file", "gt_idx", "pred_idx", "iou",
                    "area_GT_cm2", "area_Pred_cm2", "ed_GT_cm", "ed_Pred_cm",
                    "major_GT_cm", "major_Pred_cm"])
        w.writerows(paired)

    # ── 분석 1·3 : 이미지 단위 일치도 ──
    srows = []
    print("\n" + "=" * 108)
    print("=== 분석 1 — 이미지별 clod 개수 (4cm+) ===")
    print("=" * 108)
    hdr = (f"{'set':>6} {'model':>8} {'n':>4} {'GT평균':>8} {'Pred평균':>9} "
           f"{'bias':>8} {'MAE':>8} {'RMSE':>8} {'R2(1:1)':>9} {'R2(fit)':>9} "
           f"{'기울기':>8} {'MAPE%':>8}")
    print(hdr)

    def block(title, gi, pi, tag, scale=1.0):
        print(f"\n{title}")
        print(hdr)
        for ds in use:
            for lbl, _ in MODELS:
                rr = [r for r in per_img if r[0] == ds and r[1] == lbl]
                if not rr:
                    continue
                a = agree([float(r[gi]) * scale for r in rr],
                          [float(r[pi]) * scale for r in rr])
                print(f"{ds:>6} {lbl:>8} {a['n']:>4} {a['gt_mean']:>8.3f} "
                      f"{a['pred_mean']:>9.3f} {a['bias']:>+8.3f} {a['MAE']:>8.3f} "
                      f"{a['RMSE']:>8.3f} {a['R2_1to1']:>9.3f} {a['R2_fit']:>9.3f} "
                      f"{a['slope']:>8.3f} {a['MAPE']:>8.1f}")
                srows.append([tag, ds, lbl] + [round(a[k], 5) for k in
                                               ("n", "n_mape", "gt_mean", "pred_mean",
                                                "bias", "MAE", "RMSE", "R2_1to1",
                                                "r_pearson", "R2_fit", "slope",
                                                "intercept", "MAPE")])
        # pooled
        for lbl, _ in MODELS:
            rr = [r for r in per_img if r[1] == lbl]
            if len(use) > 1 and rr:
                a = agree([float(r[gi]) * scale for r in rr],
                          [float(r[pi]) * scale for r in rr])
                print(f"{'pooled':>6} {lbl:>8} {a['n']:>4} {a['gt_mean']:>8.3f} "
                      f"{a['pred_mean']:>9.3f} {a['bias']:>+8.3f} {a['MAE']:>8.3f} "
                      f"{a['RMSE']:>8.3f} {a['R2_1to1']:>9.3f} {a['R2_fit']:>9.3f} "
                      f"{a['slope']:>8.3f} {a['MAPE']:>8.1f}")
                srows.append([tag, "pooled", lbl] + [round(a[k], 5) for k in
                                                     ("n", "n_mape", "gt_mean",
                                                      "pred_mean", "bias", "MAE",
                                                      "RMSE", "R2_1to1", "r_pearson",
                                                      "R2_fit", "slope", "intercept",
                                                      "MAPE")])

    block("[개수]", 3, 4, "count")
    print("\n" + "=" * 108)
    print("=== 분석 3 — 4cm+ clod 면적비 (파쇄도 프록시) ===")
    print("=" * 108)
    block("[면적비 · 프레임(50x50cm) 대비, %]", 7, 8, "ratio_frame_pct", 100.0)
    block("[면적비 · 전체 clod 면적 대비, %]", 11, 12, "ratio_ofclod_pct", 100.0)

    with open(OUT / "frag_agreement.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "dataset", "model", "n", "n_for_MAPE", "GT_mean",
                    "Pred_mean", "bias", "MAE", "RMSE", "R2_1to1", "r_pearson",
                    "R2_fit", "slope", "intercept", "MAPE_pct"])
        w.writerows(srows)

    # ── 분석 2 : 크기 분포 ──
    print("\n" + "=" * 108)
    print("=== 분석 2 — 등가직경 / 면적 분포 유사도 ===")
    print("=" * 108)
    print(f"{'set':>6} {'model':>8} {'변수':>10} {'nGT':>5} {'nPred':>6} "
          f"{'GT평균':>8} {'Pred평균':>9} {'GT중앙':>8} {'Pred중앙':>9} "
          f"{'KS D':>8} {'KS p':>9} {'Wasser':>9}")
    drows = []
    for ds in list(use) + (["pooled"] if len(use) > 1 else []):
        for lbl, _ in MODELS:
            sel = [r for r in inst if r[1] == lbl and (ds == "pooled" or r[0] == ds)]
            if not sel:
                continue
            for col, nm, unit in [(6, "ED", "cm"), (5, "area", "cm2")]:
                G = np.array([float(r[col]) for r in sel if r[3] == "GT"])
                P = np.array([float(r[col]) for r in sel if r[3] == "Pred"])
                if len(G) < 3 or len(P) < 3:
                    continue
                ks = stats.ks_2samp(G, P)
                wd = float(stats.wasserstein_distance(G, P))
                print(f"{ds:>6} {lbl:>8} {nm+'('+unit+')':>10} {len(G):>5} {len(P):>6} "
                      f"{G.mean():>8.3f} {P.mean():>9.3f} {np.median(G):>8.3f} "
                      f"{np.median(P):>9.3f} {ks.statistic:>8.4f} {ks.pvalue:>9.4f} "
                      f"{wd:>9.4f}")
                drows.append([ds, lbl, nm, unit, len(G), len(P),
                              round(float(G.mean()), 4), round(float(P.mean()), 4),
                              round(float(np.median(G)), 4),
                              round(float(np.median(P)), 4),
                              round(float(G.std(ddof=1)), 4),
                              round(float(P.std(ddof=1)), 4),
                              round(float(ks.statistic), 4),
                              round(float(ks.pvalue), 6), round(wd, 4)])
    with open(OUT / "frag_distribution.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "variable", "unit", "n_GT", "n_Pred",
                    "GT_mean", "Pred_mean", "GT_median", "Pred_median",
                    "GT_sd", "Pred_sd", "KS_D", "KS_p", "wasserstein"])
        w.writerows(drows)

    # 매칭쌍 개체 단위 크기 오차 (참고)
    print("\n[매칭쌍 개체 단위 크기 오차 — 인식된 clod 한정]")
    print(f"{'set':>6} {'model':>8} {'n':>5} {'ED bias':>9} {'ED MAE':>9} "
          f"{'ED RMSE':>9} {'R2(1:1)':>9} {'면적 bias%':>11}")
    prows = []
    for ds in list(use) + (["pooled"] if len(use) > 1 else []):
        for lbl, _ in MODELS:
            sel = [r for r in paired if r[1] == lbl and (ds == "pooled" or r[0] == ds)]
            if len(sel) < 3:
                continue
            a = agree([float(r[8]) for r in sel], [float(r[9]) for r in sel])
            ab = np.mean([(float(r[7]) - float(r[6])) / float(r[6]) for r in sel]) * 100
            print(f"{ds:>6} {lbl:>8} {a['n']:>5} {a['bias']:>+9.4f} {a['MAE']:>9.4f} "
                  f"{a['RMSE']:>9.4f} {a['R2_1to1']:>9.4f} {ab:>+11.2f}")
            prows.append([ds, lbl, a["n"], round(a["bias"], 4), round(a["MAE"], 4),
                          round(a["RMSE"], 4), round(a["R2_1to1"], 4),
                          round(a["r_pearson"], 4), round(float(ab), 3)])
    with open(OUT / "frag_matched_size_error.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "n_pairs", "ED_bias_cm", "ED_MAE_cm",
                    "ED_RMSE_cm", "ED_R2_1to1", "ED_r", "area_bias_pct"])
        w.writerows(prows)

    plots(per_img, inst, paired, use)
    print(f"\nCSV: {OUT}")
    print(f"PNG: {FIG}")


# ─────────────────────────── 그림 ───────────────────────────
def plots(per_img, inst, paired, use):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    COL = {"SR-512": "#4C78A8", "SR-736": "#E45756"}
    MK = {"test": "o", "val": "s"}
    FIG.mkdir(parents=True, exist_ok=True)

    def scat(ax, gi, pi, lab, scale=1.0):
        lo, hi = 1e9, -1e9
        for lbl, _ in MODELS:
            for ds in use:
                rr = [r for r in per_img if r[0] == ds and r[1] == lbl]
                if not rr:
                    continue
                x = np.array([float(r[gi]) for r in rr]) * scale
                y = np.array([float(r[pi]) for r in rr]) * scale
                ax.scatter(x, y, s=46, alpha=.8, marker=MK[ds], color=COL[lbl],
                           edgecolor="w", lw=.6, label=f"{lbl} · {ds}")
                lo, hi = min(lo, x.min(), y.min()), max(hi, x.max(), y.max())
        pad = (hi - lo) * .08 + 1e-9
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="1:1 line")
        for lbl, _ in MODELS:
            rr = [r for r in per_img if r[1] == lbl]
            if len(rr) > 2:
                x = np.array([float(r[gi]) for r in rr]) * scale
                y = np.array([float(r[pi]) for r in rr]) * scale
                s, b = np.polyfit(x, y, 1)
                xs = np.array([lo, hi])
                ax.plot(xs, s * xs + b, "-", color=COL[lbl], lw=1.4, alpha=.65)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f"GT {lab}"); ax.set_ylabel(f"Predicted {lab}")
        ax.grid(alpha=.25, lw=.6); ax.set_axisbelow(True)
        ax.set_aspect("equal", adjustable="box")

    # Fig 1 — 개수 + 면적비 산점도
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4))
    scat(axes[0], 3, 4, "clod count (major axis >= 4 cm)")
    axes[0].set_title("(a) Clod count per image", fontsize=11.5)
    axes[0].legend(fontsize=8.5, loc="upper left")
    scat(axes[1], 7, 8, "coarse-clod area / frame area (%)", 100.0)
    axes[1].set_title("(b) Coarse-clod area ratio (frame basis)", fontsize=11.5)
    scat(axes[2], 11, 12, "coarse-clod area / total clod area (%)", 100.0)
    axes[2].set_title("(c) Coarse-clod area ratio (clod basis)", fontsize=11.5)
    fig.suptitle("From recognition to fragmentation assessment — "
                 "GT vs prediction (matched, conf>=0.5, IoU>=0.5)", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    p = FIG / "frag_scatter_260907.png"
    fig.savefig(p, dpi=170); plt.close(fig)
    print(f"저장: {p}")

    # Fig 2 — ED 분포 (히스토그램 + CDF)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    for j, (lbl, _) in enumerate(MODELS):
        sel = [r for r in inst if r[1] == lbl]
        G = np.array([float(r[6]) for r in sel if r[3] == "GT"])
        P = np.array([float(r[6]) for r in sel if r[3] == "Pred"])
        bins = np.linspace(min(G.min(), P.min()), max(G.max(), P.max()), 26)
        ax = axes[0, j]
        ax.hist(G, bins=bins, alpha=.55, color="#54A24B", label=f"GT (n={len(G)})",
                density=True, edgecolor="w", lw=.5)
        ax.hist(P, bins=bins, alpha=.55, color=COL[lbl],
                label=f"{lbl} (n={len(P)})", density=True, edgecolor="w", lw=.5)
        ax.set_xlabel("equivalent diameter (cm)"); ax.set_ylabel("density")
        ax.set_title(f"(a{j+1}) ED distribution — {lbl}", fontsize=11)
        ax.legend(fontsize=9); ax.grid(alpha=.25, lw=.6); ax.set_axisbelow(True)
        ax = axes[1, j]
        for v, c, nm in [(G, "#54A24B", "GT"), (P, COL[lbl], lbl)]:
            xs = np.sort(v)
            ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post",
                    color=c, lw=2, label=nm)
        ks = stats.ks_2samp(G, P)
        wd = stats.wasserstein_distance(G, P)
        ax.set_xlabel("equivalent diameter (cm)")
        ax.set_ylabel("cumulative probability")
        ax.set_title(f"(b{j+1}) ECDF — KS D={ks.statistic:.3f}, p={ks.pvalue:.3f}, "
                     f"W={wd:.3f} cm", fontsize=11)
        ax.legend(fontsize=9); ax.grid(alpha=.25, lw=.6); ax.set_axisbelow(True)
    fig.suptitle("Clod size distribution: ground truth vs prediction "
                 "(pooled test+val, major axis >= 4 cm)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = FIG / "frag_size_dist_260907.png"
    fig.savefig(p, dpi=170); plt.close(fig)
    print(f"저장: {p}")

    # Fig 3 — Bland-Altman (면적비, 프레임 기준)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (gi, pi, nm, sc) in zip(axes, [(3, 4, "clod count", 1.0),
                                           (7, 8, "area ratio (%)", 100.0)]):
        for lbl, _ in MODELS:
            rr = [r for r in per_img if r[1] == lbl]
            x = np.array([float(r[gi]) for r in rr]) * sc
            y = np.array([float(r[pi]) for r in rr]) * sc
            d = y - x
            ax.scatter((x + y) / 2, d, s=44, alpha=.8, color=COL[lbl],
                       edgecolor="w", lw=.6, label=lbl)
            ax.axhline(d.mean(), color=COL[lbl], ls="-", lw=1.4)
            ax.axhline(d.mean() + 1.96 * d.std(ddof=1), color=COL[lbl], ls=":", lw=1.1)
            ax.axhline(d.mean() - 1.96 * d.std(ddof=1), color=COL[lbl], ls=":", lw=1.1)
        ax.axhline(0, color="k", lw=1)
        ax.set_xlabel(f"mean of GT and prediction — {nm}")
        ax.set_ylabel(f"prediction - GT — {nm}")
        ax.set_title(f"Bland-Altman: {nm}", fontsize=11.5)
        ax.legend(fontsize=9); ax.grid(alpha=.25, lw=.6); ax.set_axisbelow(True)
    fig.suptitle("Agreement between predicted and reference fragmentation indices",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = FIG / "frag_blandaltman_260907.png"
    fig.savefig(p, dpi=170); plt.close(fig)
    print(f"저장: {p}")

    # Fig 4 — 매칭쌍 개체 크기 (GT ED vs Pred ED)
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    lo, hi = 1e9, -1e9
    for lbl, _ in MODELS:
        sel = [r for r in paired if r[1] == lbl]
        if not sel:
            continue
        x = np.array([float(r[8]) for r in sel])
        y = np.array([float(r[9]) for r in sel])
        ax.scatter(x, y, s=26, alpha=.55, color=COL[lbl], edgecolor="none",
                   label=f"{lbl} (n={len(x)})")
        lo, hi = min(lo, x.min(), y.min()), max(hi, x.max(), y.max())
    pad = (hi - lo) * .06
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.2, label="1:1 line")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("GT equivalent diameter (cm)")
    ax.set_ylabel("Predicted equivalent diameter (cm)")
    ax.set_title("Per-clod size fidelity (matched pairs, IoU >= 0.5)", fontsize=11.5)
    ax.legend(fontsize=9); ax.grid(alpha=.25, lw=.6); ax.set_axisbelow(True)
    fig.tight_layout()
    p = FIG / "frag_pair_size_260907.png"
    fig.savefig(p, dpi=170); plt.close(fig)
    print(f"저장: {p}")


if __name__ == "__main__":
    main()
