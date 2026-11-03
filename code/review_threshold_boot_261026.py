"""
리뷰 대응 추가 — t = 0.25 / 0.75 에서 이미지별 TP/FN/FP 재집계 + paired bootstrap.

저장된 preds (review_infer_261026.py) 를 다시 매칭만 한다 (재추론 없음).
threshold 규약은 STEP 3 과 같다: score > t (ultralytics conf 와 같은 strict 부등호).
bootstrap 도 STEP 3 과 동일: B=10,000, rng seed 2026, 이미지 인덱스 -> seed 인덱스
순서로 생성하므로 재표집 표본이 STEP 3 과 같다. t = 0.5 를 함께 돌려
bootstrap_pairdiff_261026.csv 와 일치하는지 검산한다.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import review_analysis_261026 as R      # noqa: E402
import review_common_261026 as C        # noqa: E402

U = R.U
TS = [0.25, 0.5, 0.75]
PAIRS = [(512, 1280), (512, 1600), (1024, 1600)]


def main():
    A = {}
    ties = {t: 0 for t in TS}
    for tr in R.RESL:
        for sd in R.SEEDS:
            ims = C.load_cell("test", tr, sd)["images"]
            A[(tr, sd)] = {f: R.prep(ims[f]) for f in sorted(ims)}
            for t in TS:
                ties[t] += sum(p["score"] == np.float32(t) or p["score"] == t
                               for v in ims.values() for p in v["preds"])
    files = sorted(A[(R.RESL[0], 0)])
    nI, nR, nS = len(files), len(R.RESL), len(R.SEEDS)

    rng = np.random.default_rng(R.RNG_SEED)
    II = rng.integers(0, nI, (R.B, nI))
    SS = rng.integers(0, nS, (nR, R.B, nS))
    ci = lambda x: np.percentile(x, [2.5, 97.5])

    prow, crow = [], []
    for t in TS:
        TP = np.zeros((nR, nS, nI))
        FP = np.zeros_like(TP)
        GT = np.zeros(nI)
        rows = []
        for a, tr in enumerate(R.RESL):
            for s in R.SEEDS:
                for i, f in enumerate(files):
                    P = A[(tr, s)][f]
                    tp, fp, fn, _ = R.match_fast(P, t, strict=True)
                    TP[a, s, i], FP[a, s, i], GT[i] = tp, fp, P["n"]
                    rows.append(["test", tr, R.LBL[tr], s, f, P["n"], tp, fn, fp])
        if t != 0.5:
            tag = f"{int(round(t * 100)):03d}"
            R.wcsv(f"per_image_counts_conf{tag}_{R.SFX}.csv",
                   ["dataset", "train_res", "eval_res", "seed", "file", "n_GT",
                    "TP", "FN", "FP"], rows)

        gtb = GT[II].sum(1)
        tpb = TP[:, :, II].sum(-1)
        fpb = FP[:, :, II].sum(-1)
        rec_s = tpb / gtb
        pre_s = np.divide(tpb, tpb + fpb, out=np.zeros_like(tpb),
                          where=tpb + fpb > 0)
        rec_a, pre_a = rec_s.mean(1), pre_s.mean(1)
        take = lambda X: np.stack([X[a][SS[a].T, np.arange(R.B)].mean(0)
                                   for a in range(nR)])
        rec_b, pre_b = take(rec_s), take(pre_s)
        rec_pt = (TP.sum(-1) / GT.sum()).mean(1)
        pre_pt = (TP.sum(-1) / (TP.sum(-1) + FP.sum(-1))).mean(1)

        for tr in sorted({r for p in PAIRS for r in p}):
            a = R.RESL.index(tr)
            crow.append([t, tr, R.LBL[tr], round(rec_pt[a], 4),
                         *[round(v, 4) for v in ci(rec_a[a])],
                         *[round(v, 4) for v in ci(rec_b[a])],
                         round(pre_pt[a], 4)])
        for x, y in PAIRS:
            a, b = R.RESL.index(x), R.RESL.index(y)
            for metric, pt, ra, rb in [("Recall", rec_pt, rec_a, rec_b),
                                       ("Precision", pre_pt, pre_a, pre_b)]:
                da, db = ra[a] - ra[b], rb[a] - rb[b]
                la, lb = ci(da), ci(db)
                prow.append([t, metric, f"{R.LBL[x]}-{R.LBL[y]}",
                             round(pt[a] - pt[b], 4),
                             round(la[0], 4), round(la[1], 4),
                             round(float((da > 0).mean()), 4),
                             int(la[0] <= 0 <= la[1]),
                             round(lb[0], 4), round(lb[1], 4),
                             round(float((db > 0).mean()), 4),
                             int(lb[0] <= 0 <= lb[1])])

    R.wcsv(f"bootstrap_pairdiff_t025_t075_{R.SFX}.csv",
           ["t", "metric", "pair", "delta_point", "CI_img_lo", "CI_img_hi",
            "P_gt0_img", "CI_img_contains0", "CI_2stage_lo", "CI_2stage_hi",
            "P_gt0_2stage", "CI_2stage_contains0"], prow)
    R.wcsv(f"bootstrap_recall_ci_t025_t075_{R.SFX}.csv",
           ["t", "train_res", "eval_res", "Recall_8seed_mean", "CI_img_lo",
            "CI_img_hi", "CI_2stage_lo", "CI_2stage_hi", "Precision_8seed_mean"],
           crow)

    # 검산: t = 0.5 가 기존 bootstrap_pairdiff_261026.csv 와 같은지
    old = {(r["metric"], r["pair"]): r
           for r in U.load_rows(R.OUT / f"bootstrap_pairdiff_{R.SFX}.csv")}
    cols = ["delta_point", "CI_img_lo", "CI_img_hi", "P_gt0_img", "CI_2stage_lo",
            "CI_2stage_hi", "P_gt0_2stage"]
    n = bad = 0
    for r in prow:
        if r[0] != 0.5 or (r[1], r[2]) not in old:
            continue
        o = old[(r[1], r[2])]
        n += 1
        bad += any(abs(float(o[c]) - v) > 1e-9
                   for c, v in zip(cols, [r[3], r[4], r[5], r[6], r[8], r[9], r[10]]))
    print(f"검산 t=0.5: 기존 pairdiff 와 {n}행 대조, 불일치 {bad}")
    print("score == t 동점 수:", ties)


if __name__ == "__main__":
    main()
