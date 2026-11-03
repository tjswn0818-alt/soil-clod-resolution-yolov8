"""
리뷰 대응 STEP 1-3 — review_infer_261026.py 의 저-conf 예측으로 분석.

STEP 1  재현성: score > 0.5 (ultralytics conf 규약과 동일한 strict 부등호; 동점 개수도 보고)
        -> res x seed 별 TP/FP/FN/Recall/Precision 을 기존 값과 대조.
        기준: seed8_matched_linear_260825.csv 는 요약(평균·SD·min·max)만 담고 있어
        res x seed 값은 이를 만든 같은 실행의 cross9_recall_linear_260825.csv
        (train_res == eval_res 행) 에서 가져오고, 요약 통계도 따로 대조한다.
        불일치 시 STEP 2-3 을 진행하지 않는다.
        (두 기준 파일은 not included in this repository -> 이 저장소에서는 대조를 건너뛴다)
STEP 2  t = 0.01..0.99 PR, AP50 (단조 envelope + 101-point), 매칭 score 분포
STEP 3  이미지 단위 bootstrap (test, conf 0.5)

매칭 최적화: IoU >= 0.5 인 GT 가 하나도 없는 예측은 greedy 에서 어떤 GT 도 가져가지
못하고 다른 예측의 선택에도 영향을 주지 않으므로 FP 로만 센다 (결과 동일).
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import review_common_261026 as C       # noqa: E402

U, F = C.U, C.F
OUT, SFX = C.OUT, C.SFX
RESL = U.TRAIN_RES
SEEDS = U.SEEDS
TS = np.round(np.arange(1, 100) / 100, 2)
TOL = 1e-3
RNG_SEED = 2026
B = 10000
PAIRS = [(512, 448), (512, 736), (512, 1024), (512, 1280), (512, 1600),
         (1024, 1280), (1280, 1600)]
LBL = {r: str(C.GRID[r]) for r in RESL}           # 격자 라벨 (예: "736")


def wcsv(name, hdr, rows):
    p = OUT / name
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
    return p


def prep(im):
    """이미지 1장 -> 매칭에 필요한 압축 표현."""
    kept = [im["preds"][i] for i in im["kept_idx"]]
    sc = np.array([p["score"] for p in kept], float)
    lv = np.array([p["level"] for p in kept], int)
    ar = np.array([p["area"] for p in kept], float)
    n = im["n_gt"]
    cand = np.where((im["iou"] >= F.IOU_THR).any(1))[0] if n and len(kept) \
        else np.array([], int)
    return dict(n=n, sc=sc, lv=lv, ar=ar, cand=cand,
                iou=im["iou"][cand] if len(cand) else None)


def match_fast(P, t, strict=False):
    sel_all = (P["sc"] > t) if strict else (P["sc"] >= t)
    npred = int(sel_all.sum())
    if not len(P["cand"]):
        return 0, npred, P["n"], []
    cs = sel_all[P["cand"]]
    idx = np.where(cs)[0]
    if not len(idx):
        return 0, npred, P["n"], []
    gt_to, tp = C.greedy(P["iou"][idx], list(P["ar"][P["cand"][idx]]), P["n"])
    m = [(P["sc"][P["cand"][idx[d]]], P["lv"][P["cand"][idx[d]]])
         for d in gt_to if d >= 0]
    return tp, npred - tp, P["n"] - tp, m


def load_all():
    A = {}
    trunc = []
    for ds in U.DATASETS:
        for tr in RESL:
            for sd in SEEDS:
                c = C.load_cell(ds, tr, sd)
                ims = c["images"]
                A[(ds, tr, sd)] = {f: prep(ims[f]) for f in sorted(ims)}
                nmax = max(len(v["preds"]) for v in ims.values())
                ties = sum(p["score"] == 0.5 for v in ims.values() for p in v["preds"])
                trunc.append((ds, tr, sd, nmax, ties))
                if sd == 0 and ds == "test":
                    # F.match 원본 대조 (이미지 전부, t = 0.01/0.25/0.5)
                    for t in (0.01, 0.25, 0.5):
                        for f, im in ims.items():
                            ok, a, b = C.verify_match(im, t)
                            assert ok, (ds, tr, sd, f, t, a, b)
                            assert match_fast(A[(ds, tr, sd)][f], t)[0] == a
        print(f"  [{ds}] 로드 완료", flush=True)
    return A, trunc


# ══════════════ STEP 1 ══════════════
def step1(A, trunc):
    # 재현성 대조 (cross9_recall_linear_260825.csv, seed8_matched_linear_260825.csv 와 비교 ->
    # repro_check_cells/summary) 는 기준 파일이 not included in this repository 라 생략.
    wcsv(f"infer_meta_{SFX}.csv",
         ["dataset", "train_res", "seed", "max_preds_per_img_after_nms",
          "n_score_eq_0.5"], trunc)
    print(f"         max_det(1000) 도달 이미지가 있는 셀: "
          f"{sum(t[3] >= 1000 for t in trunc)}, score==0.5 동점: "
          f"{sum(t[4] for t in trunc)}")
    return True


# ══════════════ STEP 2 ══════════════
def ap101(R, P):
    """R, P: threshold 별 점 (예측 0개 점 제외). COCO 101-point, 단조 envelope."""
    o = np.argsort(R)
    r, p = np.asarray(R)[o], np.asarray(P)[o]
    env = np.maximum.accumulate(p[::-1])[::-1]
    q = np.zeros(101)
    for k, rt in enumerate(np.linspace(0, 1, 101)):
        j = np.searchsorted(r, rt, side="left")
        if j < len(r):
            q[k] = env[j]
    return float(q.mean())


def step2(A):
    pr_rows = {ds: [] for ds in U.DATASETS}
    cell = {}
    conf_rows = []
    for (ds, tr, sd), imgs in sorted(A.items()):
        R, Pp, recs = [], [], {}
        for t in TS:
            TP = FP = FN = 0
            for P in imgs.values():
                tp, fp, fn, m = match_fast(P, t)
                TP, FP, FN = TP + tp, FP + fp, FN + fn
                if abs(t - 0.01) < 1e-9:
                    conf_rows += [[ds, tr, LBL[tr], sd, round(s, 5), f"P{l}"]
                                  for s, l in m]
            r = TP / (TP + FN)
            p = TP / (TP + FP) if TP + FP else np.nan
            f1 = 2 * p * r / (p + r) if TP else 0.0
            recs[t] = (TP, FP, FN, p, r, f1)
            pr_rows[ds].append([ds, tr, LBL[tr], sd, t, TP, FP, FN,
                                "" if np.isnan(p) else round(p, 5), round(r, 5),
                                round(f1, 5)])
            if TP + FP:
                R.append(r)
                Pp.append(p)
        f1s = {t: v[5] for t, v in recs.items()}
        tbest = max(f1s, key=lambda t: (f1s[t], -t))
        cell[(ds, tr, sd)] = dict(AP50=ap101(R, Pp), Rmax=recs[0.01][4],
                                  R25=recs[0.25][4], R50=recs[0.5][4],
                                  P50=recs[0.5][3], F1max=f1s[tbest], tF1=tbest)
    hdr = ["dataset", "train_res", "eval_res", "seed", "t", "TP", "FP", "FN",
           "Precision", "Recall", "F1"]
    for ds in U.DATASETS:
        wcsv(f"pr_curve_{ds}_{SFX}.csv", hdr, pr_rows[ds])
    keys = ["AP50", "Rmax", "R25", "R50", "P50", "F1max", "tF1"]
    srows, crows = [], []
    for ds in U.DATASETS:
        for tr in RESL:
            row = [ds, tr, LBL[tr]]
            for k in keys:
                v = np.array([cell[(ds, tr, sd)][k] for sd in SEEDS], float)
                row += [round(v.mean(), 4), round(v.std(ddof=1), 4)]
            srows.append(row)
            for sd in SEEDS:
                crows.append([ds, tr, LBL[tr], sd] +
                             [round(cell[(ds, tr, sd)][k], 5) for k in keys])
    h = ["dataset", "train_res", "eval_res"]
    for k in ["AP50", "Recall_max_t0.01", "Recall_t0.25", "Recall_t0.5",
              "Precision_t0.5", "F1_max", "t_at_F1max"]:
        h += [f"{k}_mean", f"{k}_sd"]
    wcsv(f"ap_summary_{SFX}.csv", h, srows)
    wcsv(f"ap_seed_{SFX}.csv", ["dataset", "train_res", "eval_res", "seed"] +
         ["AP50", "Recall_max_t0.01", "Recall_t0.25", "Recall_t0.5",
          "Precision_t0.5", "F1_max", "t_at_F1max"], crows)
    wcsv(f"matched_conf_{SFX}.csv",
         ["dataset", "train_res", "eval_res", "seed", "score", "level"], conf_rows)
    # 매칭 score 분포 요약 (t=0.01, 8 seed pool)
    mrows = []
    for ds in U.DATASETS:
        for tr in RESL:
            s = np.array([r[4] for r in conf_rows if r[0] == ds and r[1] == tr])
            lv = np.array([r[5] for r in conf_rows if r[0] == ds and r[1] == tr])
            q1, md, q3 = np.percentile(s, [25, 50, 75])
            row = [ds, tr, LBL[tr], len(s), round(md, 4), round(q1, 4),
                   round(q3, 4), round(q3 - q1, 4),
                   round(float((s >= 0.5).mean()), 4)]
            for L in ("P3", "P4", "P5"):
                x = s[lv == L]
                row += [len(x), round(float(np.median(x)), 4) if len(x) else "",
                        round(float((x >= 0.5).mean()), 4) if len(x) else ""]
            mrows.append(row)
    wcsv(f"matched_conf_summary_{SFX}.csv",
         ["dataset", "train_res", "eval_res", "n_matched_pooled8", "median", "q25", "q75",
          "IQR", "frac_ge_0.5", "n_P3", "median_P3", "frac_ge_0.5_P3", "n_P4",
          "median_P4", "frac_ge_0.5_P4", "n_P5", "median_P5", "frac_ge_0.5_P5"],
         mrows)
    return cell, pr_rows, conf_rows, srows, mrows


# ══════════════ STEP 3 ══════════════
def step3(A):
    rows = []
    for (ds, tr, sd), imgs in sorted(A.items()):
        for f, P in imgs.items():
            tp, fp, fn, _ = match_fast(P, 0.5, strict=True)
            rows.append([ds, tr, LBL[tr], sd, f, P["n"], tp, fn, fp])
    wcsv(f"per_image_counts_conf05_{SFX}.csv",
         ["dataset", "train_res", "eval_res", "seed", "file", "n_GT", "TP", "FN",
          "FP"], rows)
    T = [r for r in rows if r[0] == "test"]
    files = sorted({r[4] for r in T})
    fi = {f: i for i, f in enumerate(files)}
    nI = len(files)
    TP = np.zeros((len(RESL), len(SEEDS), nI))
    FP = np.zeros_like(TP)
    GT = np.zeros(nI)
    for r in T:
        a, s, i = RESL.index(r[1]), r[3], fi[r[4]]
        TP[a, s, i], FP[a, s, i], GT[i] = r[6], r[8], r[5]
    wcsv(f"gt_per_image_test_{SFX}.csv", ["file", "n_GT"],
         [[f, int(GT[i])] for i, f in enumerate(files)])

    rng = np.random.default_rng(RNG_SEED)
    II = rng.integers(0, nI, (B, nI))                     # 이미지 재표집 (전 해상도 공유)
    SS = rng.integers(0, len(SEEDS), (len(RESL), B, len(SEEDS)))  # 2단계용 seed 재표집
    gtb = GT[II].sum(1)                                   # (B,)
    tpb = TP[:, :, II].sum(-1)                            # (res, seed, B)
    fpb = FP[:, :, II].sum(-1)
    rec_s = tpb / gtb                                     # seed 별 Recall
    pre_s = np.divide(tpb, tpb + fpb, out=np.zeros_like(tpb), where=tpb + fpb > 0)
    # (a) 이미지만
    rec_a, pre_a = rec_s.mean(1), pre_s.mean(1)           # (res, B)
    # (b) 2단계: seed 도 재표집
    take = lambda X: np.stack([X[a][SS[a].T, np.arange(B)].mean(0)
                               for a in range(len(RESL))])
    rec_b, pre_b = take(rec_s), take(pre_s)
    rec_pt = (TP.sum(-1) / GT.sum()).mean(1)
    pre_pt = (TP.sum(-1) / (TP.sum(-1) + FP.sum(-1))).mean(1)
    sd_seed = (TP.sum(-1) / GT.sum()).std(1, ddof=1)

    ci = lambda x: np.percentile(x, [2.5, 97.5])
    crow = []
    for a, tr in enumerate(RESL):
        ra, rb, pa, pb = ci(rec_a[a]), ci(rec_b[a]), ci(pre_a[a]), ci(pre_b[a])
        crow.append([tr, LBL[tr], round(rec_pt[a], 4), round(sd_seed[a], 4),
                     round(ra[0], 4), round(ra[1], 4), round(ra[1] - ra[0], 4),
                     round(rb[0], 4), round(rb[1], 4), round(rb[1] - rb[0], 4),
                     round(pre_pt[a], 4), round(pa[0], 4), round(pa[1], 4),
                     round(pb[0], 4), round(pb[1], 4)])
    wcsv(f"bootstrap_recall_ci_{SFX}.csv",
         ["train_res", "eval_res", "Recall_8seed_mean", "Recall_seed_SD",
          "Recall_CI_img_lo", "Recall_CI_img_hi", "Recall_CI_img_width",
          "Recall_CI_2stage_lo", "Recall_CI_2stage_hi", "Recall_CI_2stage_width",
          "Precision_8seed_mean", "Precision_CI_img_lo", "Precision_CI_img_hi",
          "Precision_CI_2stage_lo", "Precision_CI_2stage_hi"], crow)

    prow = []
    for x, y in PAIRS:
        a, b = RESL.index(x), RESL.index(y)
        for metric, pt, ra, rb in [("Recall", rec_pt, rec_a, rec_b),
                                   ("Precision", pre_pt, pre_a, pre_b)]:
            d = pt[a] - pt[b]
            da, db = ra[a] - ra[b], rb[a] - rb[b]
            la, lb = ci(da), ci(db)
            prow.append([metric, f"{LBL[x]}-{LBL[y]}", round(d, 4),
                         round(la[0], 4), round(la[1], 4),
                         round(float((da > 0).mean()), 4),
                         int(la[0] <= 0 <= la[1]),
                         round(lb[0], 4), round(lb[1], 4),
                         round(float((db > 0).mean()), 4),
                         int(lb[0] <= 0 <= lb[1])])
    wcsv(f"bootstrap_pairdiff_{SFX}.csv",
         ["metric", "pair", "delta_point", "CI_img_lo", "CI_img_hi", "P_gt0_img",
          "CI_img_contains0", "CI_2stage_lo", "CI_2stage_hi", "P_gt0_2stage",
          "CI_2stage_contains0"], prow)
    return crow, prow, GT, files


# ══════════════ 그림 ══════════════
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
       "#6250d6", "#e34948"]
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)


def figures(pr_rows, conf_rows, srows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    rep = [256, 512, 1024, 1280, 1600]
    for ds in U.DATASETS:
        fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=200)
        for k, tr in enumerate(rep):
            sub = [r for r in pr_rows[ds] if r[1] == tr]
            R = [np.mean([r[9] for r in sub if r[4] == t]) for t in TS]
            P = [np.nanmean([np.nan if r[8] == "" else r[8] for r in sub
                             if r[4] == t]) for t in TS]
            ax.plot(R, P, color=PAL[k], lw=2, label=f"{LBL[tr]} px")
            j = int(np.where(TS == 0.5)[0][0])
            ax.plot(R[j], P[j], "o", ms=7, color=PAL[k], mec="white", mew=1.5,
                    zorder=5)
        ax.set_xlabel("Recall", color=INK)
        ax.set_ylabel("Precision", color=INK)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"PR 곡선 ({ds}, 8 seed 평균, ● = t 0.5)", color=INK,
                     fontsize=11, loc="left")
        style(ax)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(OUT / f"pr_curve_{ds}_{SFX}.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    data = [np.array([r[4] for r in conf_rows if r[0] == "test" and r[1] == tr])
            for tr in RESL]
    vp = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.8)
    for b in vp["bodies"]:
        b.set_facecolor(PAL[0])
        b.set_edgecolor("white")
        b.set_alpha(0.75)
    vp["cmedians"].set_color(INK)
    ax.axhline(0.5, color=PAL[7], lw=1.2, ls="--")
    ax.text(0.55, 0.51, "conf 0.5", color=INK2, fontsize=8, va="bottom")
    ax.set_xticks(range(1, len(RESL) + 1), [LBL[r] for r in RESL])
    ax.set_xlabel("학습 = 추론 해상도 (px)", color=INK)
    ax.set_ylabel("매칭된 예측의 confidence", color=INK)
    ax.set_ylim(0, 1)
    ax.set_title("매칭 GT 의 confidence 분포 (test, t = 0.01, 8 seed 합산)",
                 color=INK, fontsize=11, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / f"matched_conf_violin_{SFX}.png")
    plt.close(fig)

    for ds in U.DATASETS:
        rr = [r for r in srows if r[0] == ds]
        x = np.arange(len(rr))
        fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=200)
        for k, (nm, col) in enumerate([("AP50", 3), ("최대 Recall (t 0.01)", 5),
                                        ("Recall@0.5", 9)]):
            m = np.array([r[col] for r in rr])
            s = np.array([r[col + 1] for r in rr])
            ax.fill_between(x, m - s, m + s, color=PAL[k], alpha=0.15, lw=0)
            ax.plot(x, m, "-o", color=PAL[k], lw=2, ms=6, mec="white", mew=1.2,
                    label=nm)
        ax.set_xticks(x, [r[2] for r in rr])
        ax.set_xlabel("학습 = 추론 해상도 (px)", color=INK)
        ax.set_ylabel("값 (8 seed 평균 ± SD)", color=INK)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"해상도별 AP50 · 최대 Recall · Recall@0.5 ({ds})",
                     color=INK, fontsize=11, loc="left")
        style(ax)
        ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
        fig.tight_layout()
        fig.savefig(OUT / f"ap_recall_vs_res_{ds}_{SFX}.png")
        plt.close(fig)


def main():
    A, trunc = load_all()
    print("  F.match 원본 대조 통과 (test seed 0, 9 해상도 x 3 threshold x 20장)")
    if not step1(A, trunc):
        print("STEP 1 불일치 -> 중단. repro_check_*_261026.csv 확인")
        return
    _, pr_rows, conf_rows, srows, mrows = step2(A)
    print("[STEP 2] 완료")
    step3(A)
    print("[STEP 3] 완료")
    figures(pr_rows, conf_rows, srows)
    print("그림 완료")


if __name__ == "__main__":
    main()
