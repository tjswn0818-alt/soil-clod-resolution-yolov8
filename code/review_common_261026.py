"""리뷰 대응 분석 공용 — 저장된 저-conf 예측에서 threshold 별 재매칭.

매칭은 F.match 와 같은 규칙(예측 면적 내림차순 greedy, IoU >= F.IOU_THR, 1:1)을
STEP 0 에서 저장한 IoU 행렬 위에서 수행한다. IoU 는 F.match 와 동일한
coco_mask.iou 호출로 이미 계산돼 있으므로, threshold 마다 마스크를 다시 인코딩하지
않는 것만 다르다. verify_match() 가 F.match 원본과의 결과 일치를 확인한다.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import unified_4cm_260914 as U          # noqa: E402
import config  # noqa: E402  code/config.py (경로 설정)
F = U.F

OUT = config.REVIEW_DIR
PRED = OUT / "preds"
SFX = "261026"
GRID = {r: int(np.ceil(r / 32) * 32) for r in U.TRAIN_RES}


def load_cell(ds, tr, sd):
    with open(PRED / f"{ds}_{tr}_s{sd}.pkl", "rb") as fh:
        return pickle.load(fh)


def greedy(ious, areas, n_gt):
    """F.match 와 동일한 greedy. 반환: (GT별 매칭 예측 인덱스 or -1, tp)."""
    taken = [False] * n_gt
    gt_to = [-1] * n_gt
    tp = 0
    if len(areas) == 0 or n_gt == 0:
        return gt_to, 0
    for di in sorted(range(len(areas)), key=lambda i: areas[i], reverse=True):
        gi, best = -1, F.IOU_THR
        for g in range(n_gt):
            if not taken[g] and ious[di, g] >= best:
                best, gi = ious[di, g], g
        if gi >= 0:
            taken[gi] = True
            gt_to[gi] = di
            tp += 1
    return gt_to, tp


def match_at(im, t):
    """이미지 1장, score >= t 인 4cm 통과 예측으로 매칭.
    반환: tp, fp, fn, [(score, level) of matched preds]"""
    kept = [im["preds"][i] for i in im["kept_idx"]]
    sel = [j for j, p in enumerate(kept) if p["score"] >= t]
    n = im["n_gt"]
    if not sel:
        return 0, 0, n, []
    sub = im["iou"][sel][:, :] if n else np.zeros((len(sel), 0))
    areas = [kept[j]["area"] for j in sel]
    gt_to, tp = greedy(sub, areas, n)
    m = [(kept[sel[d]]["score"], kept[sel[d]]["level"]) for d in gt_to if d >= 0]
    return tp, len(sel) - tp, n - tp, m


def verify_match(im, t):
    """F.match 원본(마스크 디코딩 후)과 tp 일치 확인."""
    kept = [im["preds"][i] for i in im["kept_idx"] if im["preds"][i]["score"] >= t]
    preds = [coco_mask.decode(p["rle"]) for p in kept]
    gts = [coco_mask.decode(r) for r in im["gt_rle"]]
    _, _, tp_ref = F.match(preds, gts)
    tp, _, _, _ = match_at(im, t)
    return tp == tp_ref, tp, tp_ref
