"""R1 추가 실험 1 공용: 경로, 기존 코드 import, 병합 파이프라인.

기존 코드는 import 만 하고 수정하지 않는다.
  r1_common (Q), r1_s7_tiling (T7: tiles_of, infer_img, eval_img), review_infer_261026 패치(RI).
병합 파이프라인 (사전 고정):
  1) 타일 = T7.tiles_of(H, W, "tile") (3x3, 프레임 절반, 50% 중첩), s x s INTER_LINEAR
  2) T7.infer_img 와 같은 predict 설정 (conf 0.001, iou 0.7, max_det 1000, retina), RLE 보존
  3) score > 0.5
  4) 타일 k (row = k // 3, col = k % 3) 를 2s x 2s 캔버스의 (col * s/2, row * s/2) 에 배치
  5) 내부 경계(프레임 가장자리가 아닌 변)에서 margin px 이내에 마스크 픽셀이 있으면 버림
     거리 = 왼쪽 변: x, 오른쪽 변: s - 1 - x (위/아래 동일). 거리 <= margin 이면 버림.
     margin 0 = 경계 열/행에 닿는 마스크.
  6) score 내림차순 greedy mask-NMS: 남긴 마스크와 IoU >= 0.5 이면 낮은 쪽 제거
     (동점 score 는 타일 번호, 타일 내 순서로 안정 정렬)
  7) 캔버스 장축 >= 0.08 * 2s (F.major_axis_px; 평행이동 불변이므로 타일 마스크에서 계산)
  8) GT = F.load_gt (원본 격자 >= 4 cm) 를 2s x 2s 로 INTER_NEAREST 리사이즈,
     Q.match_sel (IoU 0.5, 면적 내림차순 greedy)
전체 프레임: 타일 1개, 캔버스 = s, 내부 경계 없음.
"""
import csv
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco_mask

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config                  # noqa: E402  code/config.py (경로 설정)
import r1_common as Q          # noqa: E402
import r1_s7_tiling as T7      # noqa: E402

R, U, C, F = Q.R, Q.U, Q.C, Q.F
S = U.S
OUT = config.R1_EXTRA_DIR       # 출력 CSV
R1_DIR = config.R1_DIR          # r1_s1/r1_s7 출력 (기준선 대조용)
XC = OUT / "cache" / "x1"
XC.mkdir(parents=True, exist_ok=True)
SEEDS = Q.SEEDS
T_SCORE = 0.5

# (조건, 학습 해상도, 타일 크기 s, 전체 프레임 대응 추론 크기)
TILE_CONDS = [("A", 1600, 800, 1600), ("A", 1280, 640, 1280), ("A", 1024, 512, 1024),
              ("A", 512, 256, 512), ("A", 448, 224, 448),
              ("B", 512, 512, 1024), ("B", 256, 256, 512), ("B", 128, 128, 256)]
# 기본 + 민감도: (이름, margin(None = 경계 규칙 없음), NMS 기준, NMS 사용)
VARIANTS = [("base", 2, "iou", True), ("ios", 2, "ios", True),
            ("noborder", None, "iou", True), ("margin0", 0, "iou", True),
            ("margin4", 4, "iou", True)]


def wcsv(name, hdr, rows):
    p = OUT / name
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
    return p


def cache_path(kind, ds, tr, s, sd):
    return XC / f"{kind}_{ds}_{tr}_{s}_s{sd}.pkl"


def load_pkl(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def save_pkl(obj, p):
    tmp = Path(str(p) + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)


def infer_image(model, img, s, mode):
    """원본 BGR -> 타일별 예측 (RLE 포함)."""
    H, W = img.shape[:2]
    T = T7.tiles_of(H, W, mode)
    tiles = []
    for (x, y, w, h) in T:
        crop = img[y:y + h, x:x + w]
        tiles.append(T7.infer_img(model, cv2.resize(crop, (s, s), interpolation=U.INTERP_IMG), s))
    return dict(T=T, tiles=tiles, H=H, W=W, s=s, mode=mode)


# ------------------------------------------------------------------ 병합
def _edge_dist(rle, s):
    """마스크 픽셀의 (왼, 오른, 위, 아래) 변까지 최소 거리."""
    x0, y0, bw, bh = coco_mask.toBbox(rle)
    x0, y0, bw, bh = int(round(x0)), int(round(y0)), int(round(bw)), int(round(bh))
    return x0, s - (x0 + bw), y0, s - (y0 + bh)     # 오른쪽: s-1-xmax = s-(x0+bw)


def candidates(img_pred):
    """3)~5) 재료: score > 0.5 예측을 캔버스로 옮김. 반환 dict 배열."""
    s, ntile = img_pred["s"], len(img_pred["tiles"])
    full = ntile == 1
    cs = s if full else 2 * s
    out = dict(score=[], area=[], major=[], dist=[], rle=[], cs=cs, s=s)
    for k, recs in enumerate(img_pred["tiles"]):
        row, col = k // 3, k % 3
        ox, oy = (0, 0) if full else (col * s // 2, row * s // 2)
        internal = ((False,) * 4 if full else (col > 0, col < 2, row > 0, row < 2))
        for q in recs:
            if not q["score"] > T_SCORE:
                continue
            d = _edge_dist(q["rle"], s)
            dmin = min([dd for dd, it in zip(d, internal) if it], default=10 ** 9)
            if full:
                crle = q["rle"]
            else:
                m = coco_mask.decode(q["rle"])
                cm = np.zeros((cs, cs), np.uint8)
                cm[oy:oy + s, ox:ox + s] = m
                crle = coco_mask.encode(np.asfortranarray(cm))
            out["score"].append(q["score"])
            out["area"].append(q["area"])
            out["major"].append(q["major_px"])
            out["dist"].append(dmin)
            out["rle"].append(crle)
    for k in ("score", "area", "major", "dist"):
        out[k] = np.array(out[k], float)
    n = len(out["rle"])
    if n:
        iou = np.asarray(coco_mask.iou(out["rle"], out["rle"], [0] * n), float).reshape(n, n)
        a = out["area"]
        inter = iou * (a[:, None] + a[None, :]) / (1.0 + iou)
        ios = inter / np.minimum(a[:, None], a[None, :])
    else:
        iou = ios = np.zeros((0, 0))
    out["iou"], out["ios"] = iou, ios
    return out


def select(cd, margin, crit, use_nms):
    """5)~7) 적용 후 남는 인덱스."""
    n = len(cd["rle"])
    alive = np.ones(n, bool)
    if margin is not None:
        alive &= ~(cd["dist"] <= margin)
    order = [i for i in np.argsort(-cd["score"], kind="stable") if alive[i]]
    if use_nms:
        M = cd[crit]
        kept = []
        for i in order:
            if all(M[i, j] < 0.5 for j in kept):
                kept.append(i)
    else:
        kept = order
    kept = np.array(sorted(kept), int)
    if len(kept):
        kept = kept[cd["major"][kept] >= F.MIN_MAJOR_FRAC * cd["cs"]]
    return kept


_GTR = {}


def gt_rle(ds, f, gmasks, cs):
    key = (ds, f, cs)
    if key not in _GTR:
        _GTR[key] = [coco_mask.encode(np.asfortranarray(
            cv2.resize(m, (cs, cs), interpolation=U.INTERP_MASK))) for m in gmasks]
    return _GTR[key]


def match_count(cd, kept, grle):
    """8) 반환 (tp, npred, n_gt)."""
    n, k = len(grle), len(kept)
    if not n or not k:
        return 0, k, n
    iou = np.asarray(coco_mask.iou([cd["rle"][i] for i in kept], grle, [0] * n),
                     float).reshape(k, n)
    p2g = Q.match_sel(iou, cd["area"][kept], np.ones(k, bool), np.arange(n))
    return int((p2g >= 0).sum()), k, n
