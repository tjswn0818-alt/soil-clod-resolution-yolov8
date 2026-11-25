"""실험 1 평가: 병합(기본 + 민감도) vs 전체 프레임, P0, t 0.5, 8 seeds.

전체 프레임 (쌍체 비교, 6단계 없음):
  A (학습 tr, 추론 tr)  : 기존 캐시 r1 iouall_* + Q.evaluate(P0, t 0.5, strict) (기준선 a1 에서 파이프라인과 일치 확인)
  B (학습 tr, 추론 2tr) : 새 추론 fullB 캐시 + 단일 타일 파이프라인 (6단계 없음)
참고 행 (fullframe_maskNMS): 전체 프레임 + 6단계(mask IoU >= 0.5). A 는 기존 preds RLE, B 는 fullB.
부트스트랩: 데이터셋별 rng(2026), II (B x nI) 이미지 리샘플 -> SS (B x 8) seed 리샘플, s7 과 같은 순서.
            모든 조건이 같은 II, SS 를 공유 -> 쌍체 차이.
"""
import multiprocessing as mp
import sys

import numpy as np

import r1_x1_common as X

Q, U, F, C, R = X.Q, X.U, X.F, X.C, X.R
B = R.B
METRICS = ["Recall", "Precision", "FN_per_img", "FP_per_img"]
_GT = {}


def gt(ds):
    if ds not in _GT:
        _GT[ds] = F.load_gt(F.DATASETS[ds])
    return _GT[ds]


def job_tile(args):
    cond, tr, s, ds, sd = args
    cell = X.load_pkl(X.cache_path("tile" + cond, ds, tr, s, sd))
    files = sorted(cell)
    out = {v[0]: np.zeros((len(files), 3), int) for v in X.VARIANTS}
    ncand = []
    for i, f in enumerate(files):
        cd = X.candidates(cell[f])
        ncand.append(len(cd["rle"]))
        grle = X.gt_rle(ds, f, gt(ds)[f]["masks"], cd["cs"])
        for name, margin, crit, use in X.VARIANTS:
            out[name][i] = X.match_count(cd, X.select(cd, margin, crit, use), grle)
    return ("merge", cond, tr, s, ds, sd), out, files


def job_full(args):
    cond, tr, sfull, ds, sd = args
    out = {"fullframe": None, "fullframe_maskNMS": None}
    if cond == "A":
        ims = Q.cell_all(ds, tr, sd)
        files = sorted(ims)
        a = np.zeros((len(files), 3), int)
        for i, f in enumerate(files):
            e = Q.evaluate(ims[f], X.T_SCORE, "P0", True)
            a[i] = (e["tp"], e["npred"], e["n_gt"])
        out["fullframe"] = a
        raw = C.load_cell(ds, tr, sd)["images"]
        cell = {f: dict(tiles=[raw[f]["preds"]], s=C.GRID[tr]) for f in raw}
    else:
        cell = X.load_pkl(X.cache_path("fullB", ds, tr, sfull, sd))
        files = sorted(cell)
    b = np.zeros((len(files), 3), int)
    c = np.zeros((len(files), 3), int)
    for i, f in enumerate(files):
        cd = X.candidates(cell[f])
        grle = X.gt_rle(ds, f, gt(ds)[f]["masks"], cd["cs"])
        b[i] = X.match_count(cd, X.select(cd, None, "iou", False), grle)
        c[i] = X.match_count(cd, X.select(cd, None, "iou", True), grle)
    if out["fullframe"] is None:
        out["fullframe"] = b
    else:
        assert (out["fullframe"] == b).all(), (cond, tr, ds, sd)    # 캐시 평가 == 파이프라인
    out["fullframe_maskNMS"] = c
    return ("full", cond, tr, sfull, ds, sd), out, files


def seed_metrics(a, n_img):
    tp, npd, ng = a[:, 0].sum(), a[:, 1].sum(), a[:, 2].sum()
    return [ng, tp, npd, npd - tp, ng - tp, tp / ng, tp / npd if npd else np.nan,
            (ng - tp) / n_img, (npd - tp) / n_img]


def boot_metrics(arr, II, SS):
    """arr (seed, img, 3) -> dict metric -> (B,) 부트스트랩 값."""
    tp, npd, ng = arr[..., 0].astype(float), arr[..., 1].astype(float), arr[..., 2].astype(float)
    nI = arr.shape[1]
    st = lambda x: x[:, II].sum(-1)                     # noqa: E731  (seed, B)
    TP, NP, NG = st(tp), st(npd), st(ng)
    with np.errstate(invalid="ignore", divide="ignore"):
        per = {"Recall": TP / NG, "Precision": TP / NP,
               "FN_per_img": (NG - TP) / nI, "FP_per_img": (NP - TP) / nI}
    cols = np.arange(II.shape[0])
    return {k: v[SS.T, cols].mean(0) for k, v in per.items()}


def main():
    jobs_t = [(c, tr, s, ds, sd) for c, tr, s, sf in X.TILE_CONDS for ds in U.DATASETS for sd in X.SEEDS]
    jobs_f = [(c, tr, sf, ds, sd) for c, tr, s, sf in X.TILE_CONDS for ds in U.DATASETS for sd in X.SEEDS]
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    with mp.Pool(nproc) as pool:
        res = pool.map(job_tile, jobs_t, chunksize=1) + pool.map(job_full, jobs_f, chunksize=1)
    X.save_pkl(res, X.OUT / "cache" / "x1_eval_raw.pkl")
    report(res)


def report(res):
    # key: (method, cond, tr, s_tile, s_full, ds, sd) -> (img, 3)
    sfull = {(c, tr): sf for c, tr, s, sf in X.TILE_CONDS}
    stile = {(c, tr): s for c, tr, s, sf in X.TILE_CONDS}
    A = {}
    files_of = {}
    for key, out, files in res:
        kind, cond, tr, s, ds, sd = key
        for meth, a in out.items():
            m = meth if kind == "full" else f"merge_{meth}"
            A[(m, cond, tr, ds, sd)] = a
            if ds in files_of:
                assert files_of[ds] == files, ds
            files_of[ds] = files
    methods = ["merge_base", "fullframe", "fullframe_maskNMS"] + [f"merge_{v[0]}" for v in X.VARIANTS[1:]]
    seed_rows = []
    for (m, cond, tr, ds, sd), a in sorted(A.items()):
        seed_rows.append([m, cond, tr, stile[(cond, tr)], sfull[(cond, tr)], ds, sd, a.shape[0]]
                         + [round(float(v), 5) if isinstance(v, float) else int(v)
                            for v in seed_metrics(a, a.shape[0])])
    X.wcsv("x1_merge_seed.csv",
           ["method", "condition", "train_res", "tile_size", "fullframe_size", "dataset", "seed", "n_img",
            "n_GT4", "TP", "n_pred", "FP", "FN", "Recall", "Precision", "FN_per_img", "FP_per_img"],
           seed_rows)

    s7 = {}
    for r in U.load_rows(X.R1_DIR / "s7_tiling_summary.csv"):
        if r["protocol"] == "P0" and r["condition"] != "full":
            s7[(r["condition"][0], int(r["train_res"]), r["dataset"])] = (
                float(r["Recall_t05_mean"]), float(r["Recall_t05_sd"]), int(r["n_GT4_unassigned"]))

    sum_rows, pd_rows, sens_rows = [], [], []
    for ds in U.DATASETS:
        nI = len(files_of[ds])
        rng = np.random.default_rng(R.RNG_SEED)
        II = rng.integers(0, nI, (B, nI))
        SS = rng.integers(0, len(X.SEEDS), (B, len(X.SEEDS)))
        for cond, tr, s, sf in X.TILE_CONDS:
            bt, pt = {}, {}
            for m in methods:
                arr = np.stack([A[(m, cond, tr, ds, sd)] for sd in X.SEEDS])
                bt[m] = boot_metrics(arr, II, SS)
                sm = np.array([seed_metrics(arr[k], nI)[5:] for k in range(len(X.SEEDS))], float)
                pt[m] = dict(zip(METRICS, sm.T))
            for m in methods:
                row = [m, cond, tr, s, sf, ds, len(X.SEEDS), nI, int(A[(m, cond, tr, ds, 0)][:, 2].sum())]
                for k in METRICS:
                    v = pt[m][k]
                    lo, hi = np.nanpercentile(bt[m][k], [2.5, 97.5])
                    row += [round(float(np.nanmean(v)), 4), round(float(np.nanstd(v, ddof=1)), 4),
                            round(float(lo), 4), round(float(hi), 4)]
                s7r = s7.get((cond, tr, ds), ("", "", ""))
                row += [s7r[0], s7r[1], s7r[2]] if m == "merge_base" else ["", "", ""]
                (sum_rows if m in ("merge_base", "fullframe", "fullframe_maskNMS") else sens_rows).append(row)
                # 쌍체 차이
                refs = ["fullframe"] + (["merge_base"] if m.startswith("merge_") and m != "merge_base" else [])
                if m == "fullframe_maskNMS":
                    refs = ["fullframe"]
                if m == "fullframe":
                    continue
                for ref in refs:
                    prow = [m, ref, cond, tr, s, sf, ds]
                    for k in METRICS:
                        d = bt[m][k] - bt[ref][k]
                        lo, hi = np.nanpercentile(d, [2.5, 97.5])
                        prow += [round(float(np.nanmean(pt[m][k]) - np.nanmean(pt[ref][k])), 4),
                                 round(float(lo), 4), round(float(hi), 4)]
                    pd_rows.append(prow)
    mh = []
    for k in METRICS:
        mh += [f"{k}_mean", f"{k}_sd", f"{k}_CI_lo", f"{k}_CI_hi"]
    hdr = (["method", "condition", "train_res", "tile_size", "fullframe_size", "dataset", "n_seeds",
            "n_img", "n_GT4"] + mh + ["s7_GTassign_Recall_mean", "s7_GTassign_Recall_sd",
                                      "s7_n_GT4_unassigned"])
    X.wcsv("x1_merge_summary.csv", hdr, sum_rows)
    X.wcsv("x1_merge_sensitivity.csv", hdr, sens_rows)
    ph = ["method", "minus", "condition", "train_res", "tile_size", "fullframe_size", "dataset"]
    for k in METRICS:
        ph += [f"d_{k}", f"d_{k}_CI_lo", f"d_{k}_CI_hi"]
    X.wcsv("x1_merge_pairdiff_ci.csv", ph, pd_rows)
    for r in sum_rows:
        print(r[:6], "R", r[9], r[11:13], "P", r[13], "FN", r[17], "FP", r[21], "s7", r[-3])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report(X.load_pkl(X.OUT / "cache" / "x1_eval_raw.pkl"))
    else:
        main()
