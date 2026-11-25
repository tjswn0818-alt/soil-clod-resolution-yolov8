"""R1 추가 실험 2 평가 공용: P0 (score > 0.5, 장축 >= 4 cm 필터 후 r1_common.match_sel), KS D."""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r1_common as Q          # noqa: E402
import r1_x2_infer as I        # noqa: E402


def load(st, lbl, ev, sd):
    with open(I.cpath(st, lbl, ev, sd), "rb") as fh:
        return pickle.load(fh)["images"]


def per_image(cell):
    """파일 정렬순 (TP, FN, FP, nGT), 예측 ED 목록, GT ED 목록."""
    files = sorted(cell)
    X = np.zeros((len(files), 4), int)
    ped, ged = [], []
    for i, f in enumerate(cell and files):
        c = cell[f]
        n = c["n_gt"]
        k = len(c["area"])
        p2g = Q.match_sel(c["iou"], c["area"], np.ones(k, bool), np.arange(n))
        tp = int((p2g >= 0).sum())
        X[i] = [tp, n - tp, k - tp, n]
        ped.append(c["ed"])
        ged.append(c["gt_ed"])
    return files, X, ped, ged


def ks_d(ped, ged):
    P, G = np.concatenate(ped), np.concatenate(ged)
    if len(P) < 3 or len(G) < 3:
        return float("nan")
    return float(stats.ks_2samp(G, P).statistic)
