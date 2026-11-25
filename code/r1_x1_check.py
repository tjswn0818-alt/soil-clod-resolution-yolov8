"""기준선 확인.
(a) 타일 1개(전체 프레임) 파이프라인 (1-3, 7, 8 단계; 6 단계 없음)
    a1: 기존 캐시(preds/*.pkl, RLE) 8 seed  -> s1_seed_metrics.csv P0 t0.5 (TP, n_pred4) 와 일치
    a2: 새로 추론한 A 전체 프레임 seed 0 (fullAchk) -> 같은 값과 일치
    a3: 새로 추론한 B 전체 프레임 (fullB) 8 seed 평균 recall -> 기존 교차평가 행렬 값
(b) 새 타일 캐시를 GT 배정 방식(T7.eval_img, P0, t 0.5)으로 집계 -> s7_tiling_seed.csv 와 일치
사용: python r1_x1_check.py a1 | a2 | a3 | b
"""
import sys

import cv2
import numpy as np
from pycocotools import mask as coco_mask

import r1_x1_common as X

Q, U, F, C, T7 = X.Q, X.U, X.F, X.C, X.T7
R1 = X.R1_DIR
GT = {}


def gt(ds):
    if ds not in GT:
        GT[ds] = F.load_gt(F.DATASETS[ds])
    return GT[ds]


def pipe_full(cell, ds):
    """단일 타일 파이프라인 (6 단계 없음). 반환 (TP, NP, NGT)."""
    tp = npd = ng = 0
    for f in sorted(cell):
        cd = X.candidates(cell[f])
        kept = X.select(cd, None, "iou", False)
        a, b, c = X.match_count(cd, kept, X.gt_rle(ds, f, gt(ds)[f]["masks"], cd["cs"]))
        tp, npd, ng = tp + a, npd + b, ng + c
    return tp, npd, ng


def s1_ref():
    ref = {}
    for r in U.load_rows(R1 / "s1_seed_metrics.csv"):
        if r["protocol"] == "P0" and float(r["t"]) == 0.5:
            ref[(r["dataset"], int(r["train_res"]), int(r["seed"]))] = (
                int(r["TP_gt"]), int(r["n_pred4"]), int(r["n_GT4"]))
    return ref


def a1():
    ref, rows, bad = s1_ref(), [], 0
    for cond, tr, s, sfull in X.TILE_CONDS:
        if cond != "A":
            continue
        for ds in U.DATASETS:
            for sd in X.SEEDS:
                ims = C.load_cell(ds, tr, sd)["images"]
                cell = {f: dict(tiles=[ims[f]["preds"]], s=C.GRID[tr]) for f in ims}
                got = pipe_full(cell, ds)
                ok = got == ref[(ds, tr, sd)]
                bad += not ok
                rows.append(["a1_cache", ds, tr, C.GRID[tr], sd, *got, *ref[(ds, tr, sd)], int(ok)])
    print("a1 mismatches:", bad, "/", len(rows))
    return rows


def a2():
    ref, rows, bad = s1_ref(), [], 0
    for cond, tr, s, sfull in X.TILE_CONDS:
        if cond != "A":
            continue
        for ds in U.DATASETS:
            p = X.cache_path("fullAchk", ds, tr, sfull, 0)
            got = pipe_full(X.load_pkl(p), ds)
            ok = got == ref[(ds, tr, 0)]
            bad += not ok
            rows.append(["a2_reinfer", ds, tr, sfull, 0, *got, *ref[(ds, tr, 0)], int(ok)])
    print("a2 mismatches:", bad, "/", len(rows))
    return rows


def a3():
    rows = []
    for ds in U.DATASETS:
        cr = {r["model"]: r for r in U.load_rows(U.OUT / f"{U.PFX}_cross_recall_{ds}_{U.STAMP}.csv")}
        for cond, tr, s, sfull in X.TILE_CONDS:
            if cond != "B":
                continue
            rec = []
            for sd in X.SEEDS:
                tp, npd, ng = pipe_full(X.load_pkl(X.cache_path("fullB", ds, tr, sfull, sd)), ds)
                rec.append(tp / ng)
            refv = float(cr[U.SR_LABEL[tr]][f"eval_{sfull}"])
            rows.append(["a3_cross", ds, tr, sfull, "mean8", round(float(np.mean(rec)), 5),
                         "", "", round(refv, 5), "", "", int(abs(np.mean(rec) - refv) < 5e-4)])
    for r in rows:
        print(r)
    return rows


def b():
    ref = {}
    for r in U.load_rows(R1 / "s7_tiling_seed.csv"):
        if r["protocol"] == "P0":
            ref[(r["condition"], r["dataset"], int(r["train_res"]), int(r["infer_size"]),
                 int(r["seed"]))] = (int(r["TP"]), int(r["n_GT4_assigned"]))
    G = {ds: Q.gt_all(ds) for ds in U.DATASETS}
    rows, bad = [], 0
    name = {"A": "A_halfinput", "B": "B_sameinput"}
    for cond, tr, s, sfull in X.TILE_CONDS:
        for ds in U.DATASETS:
            for sd in X.SEEDS:
                p = X.cache_path("tile" + cond, ds, tr, s, sd)
                if not p.exists():
                    continue
                cell = X.load_pkl(p)
                tp = nb = 0
                for f in sorted(cell):
                    g = G[ds][f]
                    Dd = dict(assign=T7.assign(g, cell[f]["T"]), big=g["big"], s=s, mode="tile",
                              tiles=[tile_mat(cell[f], k, g, s) for k in range(9)])
                    e = T7.eval_img(Dd, 0.5, "P0")
                    tp, nb = tp + e[0], nb + e[1]
                rf = ref[(name[cond], ds, tr, s, sd)]
                ok = (tp, nb) == rf
                bad += not ok
                rows.append(["b_assign", ds, tr, s, sd, tp, "", nb, rf[0], "", rf[1], int(ok)])
    print("b mismatches:", bad, "/", len(rows))
    return rows


def tile_mat(ip, k, g, s):
    """T7.run_cell 과 같은 타일별 재료 (RLE 로 IoU 재계산)."""
    x, y, w, h = ip["T"][k]
    recs = ip["tiles"][k]
    gidx, grle = [], []
    for gi, m in enumerate(g["masks"]):
        c = m[y:y + h, x:x + w]
        if c.any():
            gidx.append(gi)
            grle.append(coco_mask.encode(np.asfortranarray(
                cv2.resize(c, (s, s), interpolation=U.INTERP_MASK))))
    n = len(grle)
    if recs and n:
        iou = np.asarray(coco_mask.iou([q["rle"] for q in recs], grle, [0] * n),
                         np.float64).reshape(len(recs), n)
    else:
        iou = np.zeros((len(recs), n))
    return dict(gidx=np.array(gidx, int), iou=iou,
                score=np.array([q["score"] for q in recs]),
                area=np.array([q["area"] for q in recs], float),
                major=np.array([q["major_px"] for q in recs]),
                level=np.array([q["level"] for q in recs], int))


HDR = ["check", "dataset", "train_res", "infer_size", "seed", "TP_or_recall", "n_pred",
       "n_GT", "ref_TP_or_recall", "ref_n_pred", "ref_n_GT", "match"]

if __name__ == "__main__":
    which = sys.argv[1]
    rows = {"a1": a1, "a2": a2, "a3": a3, "b": b}[which]()
    X.wcsv(f"x1_baseline_check_{which}.csv", HDR, rows)
