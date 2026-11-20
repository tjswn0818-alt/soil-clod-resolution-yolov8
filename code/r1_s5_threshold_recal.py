"""5단계 — val 에서 임계값 선택 -> test 적용 (해상도·seed 별). score >= t, 격자 0.01..0.99
(review_analysis_261026 PR 곡선과 같은 격자·부등호). P0 기본, P1 도 함께 산출.
규칙: (a) val recall >= 0.85 인 가장 높은 t (없으면 0.01, flag)
      (b) 2·FN + FP 최소   (c) 3·FN + FP 최소   (d) val F1 최대
      동률이면 작은 t (기존 manuscript_check 규약과 같음).
부트스트랩: 1단계와 같은 교차 방식(B 10,000, rng 2026).
재현 확인: pr_curve CSV 로 계산한 test recall 평균 (a) 1600 0.826 / 512 0.914, (b) 1600 0.793 / 512 0.909.
"""
import numpy as np

import r1_common as Q

R, U = Q.R, Q.U
RULES = ["a_recall085", "b_2FN+FP", "c_3FN+FP", "d_F1max"]
PAIRS = [(512, 1600), (1024, 1600)]


def counts(imgs, t, pr):
    ev = [Q.evaluate(P, t, pr, False) for _, P in sorted(imgs.items())]
    return ev, dict(TP=sum(e["tp"] for e in ev), FN=sum(e["fn"] for e in ev),
                    FP=sum(e["fp"] for e in ev), NP=sum(e["npred"] for e in ev),
                    TPp=sum(e.get("tp_pred", e["tp"]) for e in ev))


def choose(curve):
    ts = sorted(curve)
    rec = {t: c["TP"] / (c["TP"] + c["FN"]) for t, c in curve.items()}
    f1 = {}
    for t, c in curve.items():
        p = c["TPp"] / c["NP"] if c["NP"] else 0.0
        f1[t] = 2 * p * rec[t] / (p + rec[t]) if p + rec[t] else 0.0
    ok = [t for t in ts if rec[t] >= 0.85]
    a = (max(ok), 0) if ok else (ts[0], 1)
    b = min(ts, key=lambda t: (2 * curve[t]["FN"] + curve[t]["FP"], t))
    c = min(ts, key=lambda t: (3 * curve[t]["FN"] + curve[t]["FP"], t))
    d = min(ts, key=lambda t: (-f1[t], t))
    return {"a_recall085": a, "b_2FN+FP": (b, 0), "c_3FN+FP": (c, 0), "d_F1max": (d, 0)}


def main():
    seed_rows, sum_rows, pair_rows = [], [], []
    files = None
    TPm = {}
    for pr in ("P0", "P1"):
        for tr in Q.RESL:
            for sd in Q.SEEDS:
                val = Q.cell_all("val", tr, sd)
                test = Q.cell_all("test", tr, sd)
                files = sorted(test)
                curve = {float(t): counts(val, float(t), pr)[1] for t in R.TS}
                sel = choose(curve)
                for rule, (t, flag) in sel.items():
                    ev, c = counts(test, t, pr)
                    nI = len(ev)
                    ties_v = sum(int(((P["score"] == t)
                                      | (P["score"] == float(np.float32(t)))).sum())
                                 for P in val.values())
                    ties_t = sum(int(((P["score"] == t)
                                      | (P["score"] == float(np.float32(t)))).sum())
                                 for P in test.values())
                    vr = curve[t]["TP"] / (curve[t]["TP"] + curve[t]["FN"])
                    seed_rows.append([pr, rule, tr, Q.LBL[tr], sd, t, flag, round(vr, 5),
                                      round(c["TP"] / (c["TP"] + c["FN"]), 5),
                                      round(c["TPp"] / c["NP"], 5) if c["NP"] else 0.0,
                                      round(c["FN"] / nI, 4), round(c["FP"] / nI, 4),
                                      ties_v, ties_t])
                    TPm[(pr, rule, tr, sd)] = np.array([e["tp"] for e in ev])
                    TPm[("GT",)] = np.array([e["n_gt"] for e in ev])
            print(f"{pr} R{tr} 완료", flush=True)
    Q.wcsv("s5_threshold_seed.csv",
           ["protocol", "rule", "train_res", "eval_res", "seed", "t_selected",
            "flag_no_t_meets_085", "val_recall_at_t", "test_Recall", "test_Precision",
            "test_FN_per_img", "test_FP_per_img", "n_val_pred_score_eq_t",
            "n_test_pred_score_eq_t"], seed_rows)
    for pr in ("P0", "P1"):
        for rule in RULES:
            for tr in Q.RESL:
                s = np.array([r[5:12] for r in seed_rows if r[0] == pr and r[1] == rule
                              and r[2] == tr], float)
                o = [pr, rule, tr, Q.LBL[tr]]
                for j in range(7):
                    o += [round(s[:, j].mean(), 4), round(s[:, j].std(ddof=1), 4)]
                sum_rows.append(o)
    Q.wcsv("s5_threshold_summary.csv",
           ["protocol", "rule", "train_res", "eval_res"]
           + [f"{k}_{m}" for k in ("t_selected", "flag", "val_recall", "test_Recall",
                                   "test_Precision", "test_FN_per_img", "test_FP_per_img")
              for m in ("mean", "sd")], sum_rows)

    GT = TPm[("GT",)].astype(float)
    nI, nR, nS = len(GT), len(Q.RESL), len(Q.SEEDS)
    for pr in ("P0", "P1"):
        for rule in RULES:
            rng = np.random.default_rng(R.RNG_SEED)
            II = rng.integers(0, nI, (R.B, nI))
            SS = rng.integers(0, nS, (nR, R.B, nS))
            TP = np.stack([np.stack([TPm[(pr, rule, tr, s)] for s in Q.SEEDS])
                           for tr in Q.RESL]).astype(float)
            rec_s = TP[:, :, II].sum(-1) / GT[II].sum(1)
            rec_b = np.stack([rec_s[a][SS[a].T, np.arange(R.B)].mean(0)
                              for a in range(nR)])
            pt = (TP.sum(-1) / GT.sum()).mean(1)
            for x, y in PAIRS:
                a, b = Q.RESL.index(x), Q.RESL.index(y)
                d = rec_b[a] - rec_b[b]
                lo, hi = np.percentile(d, [2.5, 97.5])
                pair_rows.append([pr, rule, f"{Q.LBL[x]}-{Q.LBL[y]}", round(pt[a] - pt[b], 4),
                                  round(lo, 4), round(hi, 4), round(float((d > 0).mean()), 4),
                                  int(lo <= 0 <= hi)])
    Q.wcsv("s5_threshold_pairdiff_bootstrap.csv",
           ["protocol", "rule", "pair", "delta_point", "CI_2stage_lo", "CI_2stage_hi",
            "P_gt0", "CI_contains0"], pair_rows)

    # 재현 확인 (pr_curve CSV 기반 사전 계산값)
    ref = {("a_recall085", 1600): .826, ("a_recall085", 512): .914,
           ("b_2FN+FP", 1600): .793, ("b_2FN+FP", 512): .909}
    for (rule, tr), want in ref.items():
        got = [r for r in sum_rows if r[0] == "P0" and r[1] == rule and r[2] == tr][0][10]
        print(f"재현 {rule} {tr}: {got} vs {want} -> {'OK' if round(got, 3) == want else 'DIFF'}")
    for r in sum_rows:
        if r[0] == "P0":
            print(r[1], r[3], "t", r[4], "R", r[10], "P", r[12], "FN/img", r[14], "FP/img", r[16])
    for r in pair_rows:
        print(r)


if __name__ == "__main__":
    main()
