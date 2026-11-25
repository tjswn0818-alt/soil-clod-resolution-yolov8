"""R1 추가 실험 2 평가·부트스트랩·비교표 (추가 시험셋 new20 + 기존 test/val).
지표 (seed 별, 영상 합산): Recall = TP/GT, Precision = TP/(TP+FP), FN/img, FP/img, KS D(ED, 영상 풀링).
부트스트랩: rng 2026 에서 II (B, nI) -> SS (B, 8) 순서 생성, B = 10,000.
  재표본마다 영상 가중치로 seed 별 지표 -> 재표본 seed 평균 (CI2, 2단계: seed x 영상). CI1_img 는 영상만 재표본
  (seed 고정, seed 평균). 쌍체 차이는 seed s 끼리 짝지음 (review_bootstrap_metrics_271026.mr_vs_sr 과 같은 방식).
같은 절차를 기존 test/val 에도 unified_4cm per-image 원자료로 적용 (SR-1600 포함).
비교:
  main       MR[448-1024] - SR (같은 추론 해상도) @ 448/512/736/1024
  aux        MR[448-736] - MR[448-1024] @ 448/512/736/1024
  aux_sr1600 SR-512@512 - SR-1600@1600
출력 (config.R1_EXTRA_DIR): x2_mr_seed.csv, x2_mr_summary.csv (new20), x2_mr_pairdiff_ci.csv, x2_mr_compare.csv
"""
import csv
from collections import defaultdict

import numpy as np

import r1_x2_common as X
import r1_x2_infer as I

OUT = I.OUT
UN = I.U.OUT
B, RNG = 10000, 2026
SEEDS = I.SEEDS
MRL = [l for l, _ in I.MR]
SRM = {ev: l for l, _, ev in I.SR}
CONDS = [(l, ev) for l in MRL for ev in I.EVALS] + [(SRM[ev], ev) for ev in I.EVALS] + [("SR-1600", 1600)]
RANGE = {"MR[128-1600]": (128, 1600), "MR[384-1600]": (384, 1600), "MR[448-1600]": (448, 1600),
         "MR[448-1280]": (448, 1280), "MR[448-1024]": (448, 1024), "MR[448-736]": (448, 736)}
NAMES = ["Recall", "Precision", "FN_per_img", "FP_per_img"]


def wcsv(name, hdr, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)


def r6(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 6)


def mets(Xc, W):
    """Xc: (seed, nI, 4) [TP, FN, FP, GT]; W: (b, nI) -> (4, seed, b)"""
    nI = Xc.shape[1]
    tp, fn, fp, gt = (np.einsum("bi,si->sb", W, Xc[:, :, j].astype(float)) for j in range(4))
    wn = W.sum(1)[None, :]
    return np.stack([tp / gt, np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0),
                     fn / wn * 1.0, fp / wn * 1.0]) if nI else None


def boot(nI):
    rng = np.random.default_rng(RNG)
    II = rng.integers(0, nI, (B, nI))
    SS = rng.integers(0, len(SEEDS), (B, len(SEEDS)))
    Cw = np.zeros((B, nI))
    np.add.at(Cw, (np.repeat(np.arange(B), nI), II.ravel()), 1)
    return Cw, SS


def load_new(st):
    data, seedrows, files0 = {}, [], None
    for lbl, ev in CONDS:
        arr = []
        for sd in SEEDS:
            cell = X.load(st, lbl, ev, sd)
            files, A, ped, ged = X.per_image(cell)
            files0 = files0 or files
            assert files == files0
            arr.append(A)
            k = X.ks_d(ped, ged)
            tp, fn, fp, gt = A.sum(0)
            n = len(files)
            seedrows.append([lbl, "SR" if lbl.startswith("SR") else "MR", ev, sd, n, gt, tp, fn, fp, tp + fp,
                             r6(tp / gt), r6(tp / (tp + fp) if tp + fp else 0.0), r6(fn / n), r6(fp / n),
                             r6(k), sum(c["n_raw"] for c in cell.values())])
        data[(lbl, ev)] = np.stack(arr)
    return files0, data, seedrows


def load_old(ds):
    rows = [r for r in csv.DictReader(open(UN / "unified_4cm_perimage_260914.csv", encoding="utf-8-sig"))
            if r["dataset"] == ds]
    files = sorted({r["file"] for r in rows})
    fi = {f: i for i, f in enumerate(files)}
    data = {}
    for lbl, ev in CONDS:
        A = np.zeros((len(SEEDS), len(files), 4), int)
        seen = np.zeros(len(SEEDS), int)
        for r in rows:
            if r["model"] == lbl and int(r["eval_res"]) == ev:
                A[int(r["seed"]), fi[r["file"]]] = [int(r["TP"]), int(r["FN"]), int(r["FP"]), int(r["n_clod_GT"])]
                seen[int(r["seed"])] += 1
        assert (seen == len(files)).all(), (ds, lbl, ev, seen)
        data[(lbl, ev)] = A
    return files, data


def analyse(ds, data, nI):
    Cw, SS = boot(nI)
    one = np.ones((1, nI))
    P = {k: mets(v, one)[:, :, 0] for k, v in data.items()}         # (4, seed)
    BM = {k: mets(v, Cw) for k, v in data.items()}                  # (4, seed, B)
    two = lambda D: D[SS.T, np.arange(B)].mean(0)                   # (seed, B) -> (B,)
    pct = lambda x: np.percentile(x, [2.5, 97.5])
    pairs = ([("MR[448-1024] - SR matched", ("MR[448-1024]", ev), (SRM[ev], ev), "main") for ev in I.EVALS]
             + [("MR[448-736] - MR[448-1024]", ("MR[448-736]", ev), ("MR[448-1024]", ev), "aux") for ev in I.EVALS]
             + [("SR-512@512 - SR-1600@1600", ("SR-512", 512), ("SR-1600", 1600), "aux_sr1600")])
    prow = []
    for name, a, b, kind in pairs:
        for j, nm in enumerate(NAMES):
            d0 = P[a][j] - P[b][j]
            D = BM[a][j] - BM[b][j]
            d2, d1 = two(D), D.mean(0)
            l2, l1 = pct(d2), pct(d1)
            prow.append([ds, kind, name, a[1], nm, r6(P[a][j].mean()), r6(P[b][j].mean()), r6(d0.mean()),
                         r6(d0.std(ddof=1)), r6(l2[0]), r6(l2[1]), round(float((d2 > 0).mean()), 4),
                         int(not (l2[0] <= 0 <= l2[1])), r6(l1[0]), r6(l1[1]), int(not (l1[0] <= 0 <= l1[1]))])
    return prow


def main():
    files, data, seedrows = load_new("new20")
    print("new20 images:", len(files), "GT>=4cm:", int(data[CONDS[0]][0, :, 3].sum()), flush=True)
    wcsv("x2_mr_seed.csv", ["model", "family", "eval_res", "seed", "n_img", "n_GT", "TP", "FN", "FP", "n_pred",
                            "Recall", "Precision", "FN_per_img", "FP_per_img", "ED_KS_D", "n_pred_raw_conf0.001"],
         seedrows)
    by = defaultdict(list)
    for r in seedrows:
        by[(r[0], r[2])].append(r)
    srow = []
    for lbl, ev in CONDS:
        rr = by[(lbl, ev)]
        line = [lbl, ev, int(RANGE[lbl][0] <= ev <= RANGE[lbl][1]) if lbl in RANGE else 1, len(rr)]
        for c in (10, 11, 12, 13, 14):
            v = np.array([np.nan if x[c] == "" else float(x[c]) for x in rr])
            f = v[np.isfinite(v)]
            line += [r6(f.mean()) if len(f) else "", r6(f.std(ddof=1)) if len(f) > 1 else "", len(f)]
        srow.append(line)
    hdr = ["model", "eval_res", "eval_in_train_range", "n_seeds"]
    for m in ("Recall", "Precision", "FN_per_img", "FP_per_img", "ED_KS_D"):
        hdr += [f"{m}_mean", f"{m}_sd", f"{m}_n"]
    wcsv("x2_mr_summary.csv", hdr, srow)

    prow = []
    for ds, (fs, dd) in [("new20_260623", (files, data)), ("test", load_old("test")), ("val", load_old("val"))]:
        prow += analyse(ds, dd, len(fs))
        print(ds, "done", flush=True)
    wcsv("x2_mr_pairdiff_ci.csv",
         ["dataset", "comparison_type", "comparison", "eval_res", "metric", "A_point", "B_point", "diff_A_minus_B",
          "diff_seed_SD", "CI2_lo", "CI2_hi", "P2_gt0", "resolved_CI2_excludes0", "CI1_img_lo", "CI1_img_hi",
          "resolved_CI1_excludes0"], prow)

    # 비교표: new20 | test | val (test/val 은 unified_4cm 요약 CSV 값)
    oldsum = {}
    for fn in ("unified_4cm_mr_summary_260914.csv", "unified_4cm_sr_summary_260914.csv"):
        for r in csv.DictReader(open(UN / fn, encoding="utf-8-sig")):
            oldsum[(r["dataset"], r["model"], int(float(r["eval_res"])))] = r
    oks = {}
    for fn in ("unified_4cm_frag_mr_260914.csv", "unified_4cm_frag_sr_260914.csv"):
        for r in csv.DictReader(open(UN / fn, encoding="utf-8-sig")):
            oks[(r["dataset"], r["model"], int(float(r["eval_res"])))] = r
    crow = []
    for line in srow:
        lbl, ev = line[0], line[1]
        row = [lbl, ev, line[2], line[4], line[5], line[10], line[13], line[16]]
        for ds in ("test", "val"):
            o, k = oldsum[(ds, lbl, ev)], oks[(ds, lbl, ev)]
            row += [o["Recall_mean"], o["Recall_sd"], o["FN_per_img_mean"], o["FP_per_img_mean"], k["ED_KS_D_mean"]]
        crow.append(row)
    h = ["model", "eval_res", "eval_in_train_range"]
    for ds in ("new20", "test", "val"):
        h += [f"{ds}_Recall_mean", f"{ds}_Recall_sd", f"{ds}_FN_per_img", f"{ds}_FP_per_img", f"{ds}_KS_D"]
    wcsv("x2_mr_compare.csv", h, crow)


if __name__ == "__main__":
    main()
