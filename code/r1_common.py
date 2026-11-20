"""리뷰 R1 (261116) 공용 — 기존 코드 import + 변형은 옵션으로만.

기존 캐시(results_review_261026/preds/*.pkl)에는 모든 예측(conf >= 0.001)의 RLE 가
있지만 IoU 행렬은 '4cm 통과 예측 x GT(>=4cm)' 뿐이다. P1 은 모든 GT(크기 무관)와
모든 예측이 필요하므로 GT 를 크기 무관하게 다시 읽고(F.load_gt 와 같은 디코딩·장축 함수),
같은 coco_mask.iou 호출로 IoU 를 다시 계산해 r1 캐시에 저장한다.
"""
import csv
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco_mask

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402  code/config.py (경로 설정)
import review_analysis_261026 as R          # noqa: E402
import review_common_261026 as C            # noqa: E402

U, F = R.U, R.F
OUT = config.R1_DIR
CACHE = OUT / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
RESL, SEEDS, LBL = R.RESL, R.SEEDS, R.LBL
FRAC4 = F.MIN_MAJOR_FRAC            # 0.08 = 4 cm
FRAC36 = F.MIN_MAJOR_FRAC * 0.9     # 0.072 = 3.6 cm (P2)


def wcsv(name, hdr, rows):
    p = OUT / name
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
    return p


def load_gt_all(d):
    """F.load_gt 와 같은 규칙(seg_to_mask, area>0, major_axis_px, thr = 0.08*side)으로
    GT 를 크기 무관하게 모두 반환. big 플래그 = F.load_gt 에 포함되는 GT."""
    coco = json.load(open(d / "_annotations.coco.json", encoding="utf-8"))
    ann_by = {}
    for a in coco["annotations"]:
        ann_by.setdefault(a["image_id"], []).append(a)
    out = {}
    for im in coco["images"]:
        H, W = im["height"], im["width"]
        side = max(H, W)
        thr = F.MIN_MAJOR_FRAC * side
        cpp = F.FRAME_CM / side
        masks, big, major_cm = [], [], []
        for a in ann_by.get(im["id"], []):
            m = F.seg_to_mask(a["segmentation"], H, W).astype(np.uint8)
            if int(m.sum()) == 0:
                continue
            mj = F.major_axis_px(m)
            masks.append(m)
            big.append(mj >= thr)
            major_cm.append(mj * cpp)
        out[im["file_name"]] = dict(masks=masks, big=np.array(big, bool),
                                    major_cm=np.array(major_cm), H=H, W=W,
                                    side=side, cpp=cpp)
    return out


_GT = {}


def gt_all(ds):
    if ds not in _GT:
        _GT[ds] = load_gt_all(F.DATASETS[ds])
    return _GT[ds]


def cell_all(ds, tr, sd):
    """셀 단위 r1 캐시: 예측 전부 x GT 전부 IoU (평가 격자)."""
    p = CACHE / f"iouall_{ds}_{tr}_s{sd}.pkl"
    if p.exists():
        with open(p, "rb") as fh:
            return pickle.load(fh)
    ims = C.load_cell(ds, tr, sd)["images"]
    r32 = C.GRID[tr]
    G = gt_all(ds)
    out = {}
    for f in sorted(ims):
        im, g = ims[f], G[f]
        gts_r = [cv2.resize(m, (r32, r32), interpolation=U.INTERP_MASK)
                 for m in g["masks"]]
        grle = [coco_mask.encode(np.asfortranarray(m)) for m in gts_r]
        preds = im["preds"]
        n, k = len(grle), len(preds)
        if n and k:
            iou = np.asarray(coco_mask.iou([q["rle"] for q in preds], grle, [0] * n),
                             np.float64).reshape(k, n)
        else:
            iou = np.zeros((k, n))
        # 기존 캐시 IoU(4cm 예측 x big GT)와 일치 확인
        big_idx = np.where(g["big"])[0]
        assert len(big_idx) == im["n_gt"], (ds, tr, sd, f)
        if len(im["kept_idx"]) and im["n_gt"]:
            assert np.allclose(iou[np.ix_(im["kept_idx"], big_idx)], im["iou"]), f
        out[f] = dict(
            score=np.array([q["score"] for q in preds], np.float64),
            area=np.array([q["area"] for q in preds], np.float64),
            major_px=np.array([q["major_px"] for q in preds], np.float64),
            level=np.array([q["level"] for q in preds], np.int8),
            iou=iou, big=g["big"].copy(), gt_major_cm=g["major_cm"].copy(),
            gt_area_r=np.array([int(m.sum()) for m in gts_r]), r32=r32)
    with open(p, "wb") as fh:
        pickle.dump(out, fh)
    return out


def match_sel(iou, area, sel, gcols):
    """기존 greedy(C.greedy) 를 선택된 예측(sel, bool) x GT 열(gcols) 에 적용.
    IoU >= 0.5 인 GT 가 없는 예측은 결과에 영향이 없으므로 제외하고 FP 로 센다
    (review_analysis_261026.match_fast 와 같은 최적화).
    반환: pred_to_gt (원래 GT 인덱스, -1), dict"""
    idx = np.where(sel)[0]
    p2g = np.full(len(area), -1, int)
    if not len(idx) or not len(gcols):
        return p2g
    sub = iou[np.ix_(idx, gcols)]
    cand = np.where((sub >= F.IOU_THR).any(1))[0]
    if not len(cand):
        return p2g
    gt_to, _ = C.greedy(sub[cand], list(area[idx[cand]]), len(gcols))
    for gi, d in enumerate(gt_to):
        if d >= 0:
            p2g[idx[cand[d]]] = gcols[gi]
    return p2g


def evaluate(P, t, proto, strict):
    """이미지 1장. proto: P0 / P1 / P2. 반환 dict(tp, fp, fn, n_gt, tp_smallgt,
    matched_scores, fn_gt_idx)."""
    sc, ar, mj, iou, big = P["score"], P["area"], P["major_px"], P["iou"], P["big"]
    r32 = P["r32"]
    above = sc > t if strict else sc >= t
    keep4 = mj >= FRAC4 * r32
    nbig = int(big.sum())
    if proto in ("P0", "P2"):
        keep = keep4 if proto == "P0" else (mj >= FRAC36 * r32)
        sel = above & keep
        p2g = match_sel(iou, ar, sel, np.where(big)[0])
        m = p2g >= 0
        tp = int(m.sum())
        matched_gt = set(p2g[m].tolist())
        return dict(tp=tp, fp=int(sel.sum()) - tp, fn=nbig - tp, n_gt=nbig,
                    tp_smallgt=0, npred=int(sel.sum()),
                    fn_gt=[g for g in np.where(big)[0] if g not in matched_gt],
                    mscore=sc[m])
    # P1: 크기 무관 매칭 후 집계
    sel = above
    p2g = match_sel(iou, ar, sel, np.arange(len(big)))
    m = p2g >= 0
    matched_gt = set(p2g[m].tolist())
    tp_gt = sum(1 for g in matched_gt if big[g])
    pk = sel & keep4
    tp_pred = int((pk & m).sum())
    tp_small = int((pk & m & ~big[np.clip(p2g, 0, None)]).sum()) if len(big) else 0
    return dict(tp=tp_gt, fp=int(pk.sum()) - tp_pred, fn=nbig - tp_gt, n_gt=nbig,
                tp_pred=tp_pred, tp_smallgt=tp_small, npred=int(pk.sum()),
                fn_gt=[g for g in np.where(big)[0] if g not in matched_gt],
                mscore=sc[m & big[np.clip(p2g, 0, None)]] if len(big) else sc[:0])
