"""1단계 — 크기 필터 민감도 (P0 / P1 / P2). 재추론 없음, 캐시 예측 재매칭.

부등호: t = 0.25/0.5/0.75 지표는 score > t, AP50·t 0.01 recall 은 score >= t (기존 PR 코드).
부트스트랩: review_analysis_261026.step3 와 같은 교차(이미지 -> seed 2단계) 방식,
B = 10,000, rng seed 2026, 같은 난수 생성 순서. P0 t 0.5 결과가 기존
bootstrap_pairdiff_261026.csv 와 같은지 검산한다.
"""
import numpy as np

import r1_common as Q

R, U = Q.R, Q.U
TS3 = [0.25, 0.5, 0.75]
PROTOS = ["P0", "P1", "P2"]
PAIRS = [(512, 1600), (1024, 1600), (512, 1280)]


def main():
    cells = {}
    for ds in U.DATASETS:
        for tr in Q.RESL:
            for sd in Q.SEEDS:
                cells[(ds, tr, sd)] = Q.cell_all(ds, tr, sd)
        print(f"[{ds}] IoU 캐시 완료", flush=True)

    # 0) 검산: P0 t 0.5 strict == 기존 match_fast
    A, _ = R.load_all()
    for k, imgs in cells.items():
        for f, P in imgs.items():
            a = Q.evaluate(P, 0.5, "P0", True)
            b = R.match_fast(A[k][f], 0.5, strict=True)
            assert (a["tp"], a["fp"], a["fn"]) == tuple(b[:3]), (k, f)
    print("P0 == 기존 match_fast (t 0.5) 전 셀 일치")

    # 1) t = 0.25/0.5/0.75 지표 + 동점 수
    seed_rows, sum_rows, tie_rows, small_rows = [], [], [], []
    per_img = {}       # (proto, ds, tr, sd, t) -> list of (tp, fn, fp, n)
    for (ds, tr, sd), imgs in sorted(cells.items()):
        nI = len(imgs)
        for t in TS3:
            tie_rows.append([ds, tr, Q.LBL[tr], sd, t,
                             sum(int(((P["score"] == t)
                                      | (P["score"] == float(np.float32(t)))).sum())
                                 for P in imgs.values())])
            for pr in PROTOS:
                ev = [Q.evaluate(P, t, pr, True) for _, P in sorted(imgs.items())]
                per_img[(pr, ds, tr, sd, t)] = ev
                TP = sum(e["tp"] for e in ev)
                FN = sum(e["fn"] for e in ev)
                NP = sum(e["npred"] for e in ev)
                FP = sum(e["fp"] for e in ev)
                TPp = sum(e.get("tp_pred", e["tp"]) for e in ev)
                SM = sum(e["tp_smallgt"] for e in ev)
                rec = TP / (TP + FN)
                pre = TPp / NP if NP else 0.0
                seed_rows.append([pr, ds, tr, Q.LBL[tr], sd, t, TP + FN, TP, FN, NP,
                                  TPp, FP, SM, round(rec, 5), round(pre, 5),
                                  round(FN / nI, 4), round(FP / nI, 4)])
                if pr == "P1":
                    small_rows.append([ds, tr, Q.LBL[tr], sd, t, NP, TPp, SM])
    Q.wcsv("s1_seed_metrics.csv",
           ["protocol", "dataset", "train_res", "eval_res", "seed", "t", "n_GT4",
            "TP_gt", "FN", "n_pred4", "TP_pred", "FP", "TPpred_on_smallGT",
            "Recall", "Precision", "FN_per_img", "FP_per_img"], seed_rows)
    Q.wcsv("s1_P1_pred4_matched_to_smallGT.csv",
           ["dataset", "train_res", "eval_res", "seed", "t", "n_pred4",
            "TP_pred4", "TP_pred4_on_GT_lt4cm"], small_rows)
    Q.wcsv("s1_score_ties.csv", ["dataset", "train_res", "eval_res", "seed", "t",
                                 "n_pred_score_eq_t"], tie_rows)
    for pr in PROTOS:
        for ds in U.DATASETS:
            for tr in Q.RESL:
                for t in TS3:
                    s = [r for r in seed_rows if r[0] == pr and r[1] == ds
                         and r[2] == tr and r[5] == t]
                    row = [pr, ds, tr, Q.LBL[tr], t]
                    for c in (13, 14, 15, 16, 12):
                        v = np.array([r[c] for r in s], float)
                        row += [round(v.mean(), 4), round(v.std(ddof=1), 4)]
                    sum_rows.append(row)
    Q.wcsv("s1_summary_t025_050_075.csv",
           ["protocol", "dataset", "train_res", "eval_res", "t", "Recall_mean",
            "Recall_sd", "Precision_mean", "Precision_sd", "FN_per_img_mean",
            "FN_per_img_sd", "FP_per_img_mean", "FP_per_img_sd",
            "TPpred_on_smallGT_mean", "TPpred_on_smallGT_sd"], sum_rows)

    # 2) AP50 / t 0.01 recall (score >= t), P0·P1
    ap_rows, ap_seed = [], []
    for pr in ("P0", "P1"):
        for ds in U.DATASETS:
            for tr in Q.RESL:
                v = []
                for sd in Q.SEEDS:
                    imgs = cells[(ds, tr, sd)]
                    Rr, Pp, r01 = [], [], None
                    for t in R.TS:
                        ev = [Q.evaluate(P, t, pr, False) for P in imgs.values()]
                        TP = sum(e["tp"] for e in ev)
                        FN = sum(e["fn"] for e in ev)
                        NP = sum(e["npred"] for e in ev)
                        TPp = sum(e.get("tp_pred", e["tp"]) for e in ev)
                        if abs(t - 0.01) < 1e-9:
                            r01 = TP / (TP + FN)
                        if NP:
                            Rr.append(TP / (TP + FN))
                            Pp.append(TPp / NP)
                    ap = R.ap101(Rr, Pp)
                    v.append([ap, r01])
                    ap_seed.append([pr, ds, tr, Q.LBL[tr], sd, round(ap, 5),
                                    round(r01, 5)])
                v = np.array(v)
                ap_rows.append([pr, ds, tr, Q.LBL[tr], round(v[:, 0].mean(), 4),
                                round(v[:, 0].std(ddof=1), 4),
                                round(v[:, 1].mean(), 4), round(v[:, 1].std(ddof=1), 4)])
        print(f"AP {pr} 완료", flush=True)
    Q.wcsv("s1_ap50_recall001_seed.csv", ["protocol", "dataset", "train_res",
                                          "eval_res", "seed", "AP50", "Recall_t001"],
           ap_seed)
    Q.wcsv("s1_ap50_recall001_summary.csv",
           ["protocol", "dataset", "train_res", "eval_res", "AP50_mean", "AP50_sd",
            "Recall_t001_mean", "Recall_t001_sd"], ap_rows)
    # 검산: P0 AP 가 기존 ap_seed_261026 와 같은지
    old = {(r["dataset"], int(r["train_res"]), int(r["seed"])): r
           for r in U.load_rows(R.OUT / f"ap_seed_{R.SFX}.csv")}
    bad = sum(abs(float(old[(r[1], r[2], r[4])]["AP50"]) - r[5]) > 1e-4
              or abs(float(old[(r[1], r[2], r[4])]["Recall_max_t0.01"]) - r[6]) > 1e-4
              for r in ap_seed if r[0] == "P0")
    print(f"검산 P0 AP50/R@0.01 vs ap_seed_261026: 불일치 {bad}")

    # 3) 교차 부트스트랩 (test, t 0.5): P0 (검산) + P1
    files = sorted(cells[("test", Q.RESL[0], 0)])
    nI, nR, nS = len(files), len(Q.RESL), len(Q.SEEDS)
    rng = np.random.default_rng(R.RNG_SEED)
    II = rng.integers(0, nI, (R.B, nI))
    SS = rng.integers(0, nS, (nR, R.B, nS))
    ci = lambda x: np.percentile(x, [2.5, 97.5])
    brow, prow = [], []
    for pr in ("P0", "P1"):
        TP = np.zeros((nR, nS, nI))
        GT = np.zeros(nI)
        for a, tr in enumerate(Q.RESL):
            for s in Q.SEEDS:
                for i, e in enumerate(per_img[(pr, "test", tr, s, 0.5)]):
                    TP[a, s, i], GT[i] = e["tp"], e["n_gt"]
        rec_s = TP[:, :, II].sum(-1) / GT[II].sum(1)
        rec_a = rec_s.mean(1)
        rec_b = np.stack([rec_s[a][SS[a].T, np.arange(R.B)].mean(0) for a in range(nR)])
        pt = (TP.sum(-1) / GT.sum()).mean(1)
        for a, tr in enumerate(Q.RESL):
            brow.append([pr, tr, Q.LBL[tr], round(pt[a], 4),
                         *np.round(ci(rec_b[a]), 4), *np.round(ci(rec_a[a]), 4)])
        for x, y in PAIRS:
            a, b = Q.RESL.index(x), Q.RESL.index(y)
            db, da = rec_b[a] - rec_b[b], rec_a[a] - rec_a[b]
            lb, la = ci(db), ci(da)
            prow.append([pr, f"{Q.LBL[x]}-{Q.LBL[y]}", round(pt[a] - pt[b], 4),
                         round(lb[0], 4), round(lb[1], 4), round(float((db > 0).mean()), 4),
                         int(lb[0] <= 0 <= lb[1]), round(la[0], 4), round(la[1], 4)])
    Q.wcsv("s1_bootstrap_recall_ci_t05.csv",
           ["protocol", "train_res", "eval_res", "Recall_8seed_mean", "CI_2stage_lo",
            "CI_2stage_hi", "CI_img_lo", "CI_img_hi"], brow)
    Q.wcsv("s1_bootstrap_pairdiff_t05.csv",
           ["protocol", "pair", "delta_point", "CI_2stage_lo", "CI_2stage_hi",
            "P_gt0_2stage", "CI_2stage_contains0", "CI_img_lo", "CI_img_hi"], prow)
    old = {r["pair"]: r for r in U.load_rows(R.OUT / f"bootstrap_pairdiff_{R.SFX}.csv")
           if r["metric"] == "Recall"}
    for r in prow:
        if r[0] == "P0" and r[1] in old:
            o = old[r[1]]
            print(f"  검산 P0 {r[1]}: 2stage CI {r[3]},{r[4]} vs 기존 "
                  f"{o['CI_2stage_lo']},{o['CI_2stage_hi']}")

    # 4) 진단: P0 FN 중 '장축 < 4cm 이라 제거된 예측이 IoU>=0.5, conf>0.5' 인 GT
    drows, dsum = [], []
    for ds in U.DATASETS:
        for tr in Q.RESL:
            nfn = nhit = 0
            for sd in Q.SEEDS:
                imgs = cells[(ds, tr, sd)]
                ev = per_img[("P0", ds, tr, sd, 0.5)]
                for (f, P), e in zip(sorted(imgs.items()), ev):
                    small = (P["major_px"] < Q.FRAC4 * P["r32"]) & (P["score"] > 0.5)
                    for g in e["fn_gt"]:
                        nfn += 1
                        hit = bool(small.any() and (P["iou"][small, g] >= 0.5).any())
                        nhit += hit
                        drows.append([ds, tr, Q.LBL[tr], sd, f, g,
                                      round(P["gt_major_cm"][g], 3), int(hit)])
            mj = np.array([r[6] for r in drows if r[0] == ds and r[1] == tr and r[7]])
            dsum.append([ds, tr, Q.LBL[tr], nfn, nhit,
                         round(nhit / nfn, 4) if nfn else "",
                         *(np.round(np.percentile(mj, [0, 25, 50, 75, 100]), 2)
                           if len(mj) else [""] * 5)])
    Q.wcsv("s1_diag_fn_removed_by_filter_detail.csv",
           ["dataset", "train_res", "eval_res", "seed", "file", "gt_idx",
            "gt_major_cm", "hit_by_pred_lt4cm"], drows)
    Q.wcsv("s1_diag_fn_removed_by_filter_summary.csv",
           ["dataset", "train_res", "eval_res", "n_FN_8seed", "n_FN_with_lt4cm_pred",
            "ratio", "gt_major_cm_min", "q25", "median", "q75", "max"], dsum)

    # 5) Figure 9 분해 (P0, P1): 1-R(>=0.01) / R(>=0.01) - R(>0.5)
    frows = []
    for pr in ("P0", "P1"):
        for ds in U.DATASETS:
            for tr in (512, 1024, 1600):
                a, b = [], []
                for sd in Q.SEEDS:
                    imgs = cells[(ds, tr, sd)]
                    e1 = [Q.evaluate(P, 0.01, pr, False) for P in imgs.values()]
                    r1 = sum(e["tp"] for e in e1) / sum(e["n_gt"] for e in e1)
                    e5 = per_img[(pr, ds, tr, sd, 0.5)]
                    r5 = sum(e["tp"] for e in e5) / sum(e["n_gt"] for e in e5)
                    a.append(1 - r1)
                    b.append(r1 - r5)
                frows.append([pr, ds, tr, Q.LBL[tr], round(np.mean(a), 4),
                              round(np.std(a, ddof=1), 4), round(np.mean(b), 4),
                              round(np.std(b, ddof=1), 4)])
    Q.wcsv("s1_fig9_miss_decomposition.csv",
           ["protocol", "dataset", "train_res", "eval_res",
            "miss_no_match_ge001_mean", "sd", "miss_matched_001_05_mean", "sd"], frows)
    print("1단계 완료")
    for r in sum_rows:
        if r[4] == 0.5 and r[1] == "test":
            print("  ", r[:4], "R", r[5], "P", r[7], "FN/img", r[9], "FP/img", r[11])


if __name__ == "__main__":
    main()
