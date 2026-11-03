"""
파쇄도 재현성 분석 — 9개 단일 해상도(SR) 모델 전체로 확장.

clod_frag_260907.py 와 동일한 방법·기준을 그대로 사용하고 모델만 늘린다.
  - matched 조건 (학습 해상도 = 추론 해상도, 격자 736)
  - conf >= 0.5, 매칭 IoU >= 0.5, 예측 면적 내림차순 greedy
  - clod 정의: 장축 >= 4cm (MIN_MAJOR_FRAC = 0.08, 프레임 50cm)
  - 면적/직경은 원본 이미지 격자로 복원 후 cm 환산, ED = 2*sqrt(area/pi)

저해상도(128/256)는 검출이 붕괴할 수 있으므로 표본이 부족한 지표는 NaN 으로
남기고 유효 표본 수(n)를 함께 기록한다.

출력: results/frag/  *_all.csv  +  figures/frag_res_summary_260909.png
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clod_frag_260907 as F  # noqa: E402

RS_ROOT = F.RS_ROOT
OUT = F.OUT
FIG = F.FIG
DATASETS = F.DATASETS
FRAME_CM = F.FRAME_CM

# (라벨, 학습 해상도) — 라벨은 실제 추론 격자(32 배수) 기준
TRAIN_RES = [128, 256, 384, 448, 512, 736, 1024, 1280, 1600]
MODELS = [(f"SR-{int(np.ceil(r/32)*32)}", r) for r in TRAIN_RES]
NAN = float("nan")

AGREE_KEYS = ("n", "n_mape", "gt_mean", "pred_mean", "bias", "MAE", "RMSE",
              "R2_1to1", "r_pearson", "R2_fit", "slope", "intercept", "MAPE")


def agree(y_gt, y_pr):
    """clod_frag_260907.agree 와 동일하되 표본 부족/무분산에서 NaN 안전 처리."""
    y_gt, y_pr = np.asarray(y_gt, float), np.asarray(y_pr, float)
    n = len(y_gt)
    if n == 0:
        return {k: NAN for k in AGREE_KEYS} | {"n": 0, "n_mape": 0}
    d = y_pr - y_gt
    sse = float((d ** 2).sum())
    sst = float(((y_gt - y_gt.mean()) ** 2).sum())
    r2_11 = 1 - sse / sst if sst > 0 else NAN
    if n > 2 and y_gt.std() > 0 and y_pr.std() > 0:
        r = float(np.corrcoef(y_gt, y_pr)[0, 1])
        sl, ic = [float(v) for v in np.polyfit(y_gt, y_pr, 1)]
    else:
        r = sl = ic = NAN
    nz = np.abs(y_gt) > 1e-9
    mape = float(np.mean(np.abs(d[nz]) / np.abs(y_gt[nz])) * 100) if nz.any() else NAN
    return dict(n=n, n_mape=int(nz.sum()), gt_mean=float(y_gt.mean()),
                pred_mean=float(y_pr.mean()), bias=float(d.mean()),
                MAE=float(np.abs(d).mean()), RMSE=float(np.sqrt((d ** 2).mean())),
                R2_1to1=r2_11, r_pearson=r,
                R2_fit=r * r if np.isfinite(r) else NAN,
                slope=sl, intercept=ic, MAPE=mape)


def rd(v, k=5):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(v, k)


def fm(v, w=9, p=3):
    return f"{'nan':>{w}}" if not np.isfinite(v) else f"{v:>{w}.{p}f}"


def main():
    from scipy import stats
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="test,val")
    ap.add_argument("--models", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    use = [s.strip() for s in args.sets.split(",")]
    models = MODELS
    if args.models:
        keep = {s.strip() for s in args.models.split(",")}
        models = [m for m in MODELS if m[0] in keep or str(m[1]) in keep]

    per_img, inst, paired, detect = [], [], [], []
    print("=" * 112)
    print("추론 — 9 해상도 x 2 데이터셋 (matched, conf>=0.5, IoU>=0.5, clod 장축>=4cm)")
    print("=" * 112)
    for ds in use:
        d = DATASETS[ds]
        gt = F.load_gt(d)
        files = sorted(gt)
        ngt = sum(len(gt[f]["masks"]) for f in files)
        print(f"\n[{ds}] 이미지 {len(files)}장 | GT clod(4cm+) {ngt}개")
        print(f"{'model':>9} {'eval':>6} {'Pred':>6} {'TP':>5} {'Recall':>8} "
              f"{'Precision':>10}")
        for lbl, tr in models:
            ck = F.WEIGHTS / "sr" / f"R{tr}.pt"
            if not ck.exists():
                print(f"  [skip] {ck}")
                continue
            R, r32 = F.run_model(ck, gt, d, tr)
            TP = sum(R[f]["tp"] for f in files)
            NP = sum(len(R[f]["preds"]) for f in files)
            rec = TP / ngt if ngt else NAN
            pre = TP / NP if NP else NAN
            print(f"{lbl:>9} {r32:>6} {NP:>6} {TP:>5} {fm(rec, 8, 3)} {fm(pre, 10, 3)}")
            detect.append([ds, lbl, tr, r32, len(files), ngt, NP, TP,
                           rd(rec), rd(pre)])
            for f in files:
                g, r = gt[f], R[f]
                cpp = g["cpp"]
                gs = [F.shape_stats(m, cpp) for m in g["masks"]]
                ps = [F.shape_stats(m, cpp) for m in r["preds"]]
                frame = FRAME_CM ** 2
                ga, pa = sum(x[0] for x in gs), sum(x[0] for x in ps)
                per_img.append([ds, lbl, tr, r32, f, len(gs), len(ps),
                                round(ga, 3), round(pa, 3),
                                round(ga / frame, 5), round(pa / frame, 5),
                                round(g["total_clod_cm2"], 3),
                                round(r["pred_all_area_cm2"], 3),
                                rd(ga / g["total_clod_cm2"]) if g["total_clod_cm2"] > 0
                                else "",
                                rd(pa / r["pred_all_area_cm2"])
                                if r["pred_all_area_cm2"] > 0 else "",
                                r["tp"], len(gs) - r["tp"], len(ps) - r["tp"]])
                inv = {di: gi for gi, (di, _) in r["pairs"].items()}
                for i, (a, e, mj) in enumerate(gs):
                    inst.append([ds, lbl, tr, f, "GT", i, round(a, 4), round(e, 4),
                                 round(mj, 4), int(r["flags"][i])])
                for i, (a, e, mj) in enumerate(ps):
                    inst.append([ds, lbl, tr, f, "Pred", i, round(a, 4), round(e, 4),
                                 round(mj, 4), int(i in inv)])
                for gi, (di, iou) in r["pairs"].items():
                    paired.append([ds, lbl, tr, f, gi, di, round(iou, 4),
                                   round(gs[gi][0], 4), round(ps[di][0], 4),
                                   round(gs[gi][1], 4), round(ps[di][1], 4),
                                   round(gs[gi][2], 4), round(ps[di][2], 4)])
    torch.cuda.empty_cache()

    labels = [l for l, _ in models]
    tr_of = {l: t for l, t in models}
    scopes = list(use) + (["pooled"] if len(use) > 1 else [])

    # ── 원자료 CSV ──
    with open(OUT / "frag_per_image_all.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "train_res", "eval_res", "file", "n_clod_GT",
                    "n_clod_Pred", "area4cm_GT_cm2", "area4cm_Pred_cm2",
                    "ratio_frame_GT", "ratio_frame_Pred", "area_allclod_GT_cm2",
                    "area_allclod_Pred_cm2", "ratio_ofclod_GT", "ratio_ofclod_Pred",
                    "TP", "FN", "FP"])
        w.writerows(per_img)
    with open(OUT / "frag_instances_all.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "train_res", "file", "source", "idx",
                    "area_cm2", "ed_cm", "major_cm", "matched"])
        w.writerows(inst)
    with open(OUT / "frag_matched_pairs_all.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "train_res", "file", "gt_idx", "pred_idx",
                    "iou", "area_GT_cm2", "area_Pred_cm2", "ed_GT_cm", "ed_Pred_cm",
                    "major_GT_cm", "major_Pred_cm"])
        w.writerows(paired)
    with open(OUT / "frag_detection_all.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "train_res", "eval_res", "n_images",
                    "n_GT_clod", "n_Pred_clod", "TP", "Recall", "Precision"])
        w.writerows(detect)

    # ── 분석 1·4 : 이미지 단위 일치도 ──
    QTY = [("count", "clod 개수", 5, 6, 1.0),
           ("ratio_frame_pct", "면적비·프레임 대비(%)", 9, 10, 100.0),
           ("ratio_ofclod_pct", "면적비·전체clod 대비(%)", 13, 14, 100.0)]
    srows = []
    for tag, name, gi, pi, sc in QTY:
        print("\n" + "=" * 112)
        print(f"=== [{name}] 이미지별 GT vs Pred ===")
        print("=" * 112)
        print(f"{'set':>7} {'model':>9} {'n':>4} {'GT평균':>9} {'Pred평균':>9} "
              f"{'bias':>9} {'MAE':>9} {'RMSE':>9} {'R2(1:1)':>9} {'R2(fit)':>9} "
              f"{'기울기':>8} {'MAPE%':>8}")
        for scope in scopes:
            for lbl in labels:
                rr = [r for r in per_img if r[1] == lbl
                      and (scope == "pooled" or r[0] == scope)]
                rr = [r for r in rr if r[gi] != "" and r[pi] != ""]
                if not rr:
                    continue
                a = agree([float(r[gi]) * sc for r in rr],
                          [float(r[pi]) * sc for r in rr])
                print(f"{scope:>7} {lbl:>9} {a['n']:>4} {fm(a['gt_mean'])} "
                      f"{fm(a['pred_mean'])} {fm(a['bias'])} {fm(a['MAE'])} "
                      f"{fm(a['RMSE'])} {fm(a['R2_1to1'])} {fm(a['R2_fit'])} "
                      f"{fm(a['slope'], 8)} {fm(a['MAPE'], 8, 1)}")
                srows.append([tag, name, scope, lbl, tr_of[lbl]]
                             + [rd(a[k]) for k in AGREE_KEYS])
            print()
    with open(OUT / "frag_agreement_all.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "quantity_label", "dataset", "model", "train_res",
                    "n", "n_for_MAPE", "GT_mean", "Pred_mean", "bias", "MAE", "RMSE",
                    "R2_1to1", "r_pearson", "R2_fit", "slope", "intercept",
                    "MAPE_pct"])
        w.writerows(srows)

    # ── 분석 2 : 매칭쌍 크기 오차 ──
    print("\n" + "=" * 112)
    print("=== [매칭쌍 개체 크기 오차] 인식된 clod 한정 ===")
    print("=" * 112)
    print(f"{'set':>7} {'model':>9} {'n쌍':>5} {'ED bias':>9} {'ED MAE':>9} "
          f"{'ED RMSE':>9} {'ED R2(1:1)':>11} {'ED r':>8} {'area bias%':>11} "
          f"{'IoU평균':>8}")
    prows = []
    for scope in scopes:
        for lbl in labels:
            sel = [r for r in paired if r[1] == lbl
                   and (scope == "pooled" or r[0] == scope)]
            if not sel:
                prows.append([scope, lbl, tr_of[lbl], 0] + [""] * 7)
                print(f"{scope:>7} {lbl:>9} {0:>5} " + " ".join(f"{'nan':>9}"
                                                                for _ in range(6)))
                continue
            a = agree([float(r[9]) for r in sel], [float(r[10]) for r in sel])
            ab = float(np.mean([(float(r[8]) - float(r[7])) / float(r[7])
                                for r in sel]) * 100)
            miou = float(np.mean([float(r[6]) for r in sel]))
            print(f"{scope:>7} {lbl:>9} {a['n']:>5} {fm(a['bias'], 9, 4)} "
                  f"{fm(a['MAE'], 9, 4)} {fm(a['RMSE'], 9, 4)} "
                  f"{fm(a['R2_1to1'], 11, 4)} {fm(a['r_pearson'], 8, 4)} "
                  f"{fm(ab, 11, 2)} {fm(miou, 8, 4)}")
            prows.append([scope, lbl, tr_of[lbl], a["n"], rd(a["bias"], 4),
                          rd(a["MAE"], 4), rd(a["RMSE"], 4), rd(a["R2_1to1"], 4),
                          rd(a["r_pearson"], 4), rd(ab, 3), rd(miou, 4)])
        print()
    with open(OUT / "frag_matched_size_error_all.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "train_res", "n_pairs", "ED_bias_cm",
                    "ED_MAE_cm", "ED_RMSE_cm", "ED_R2_1to1", "ED_r",
                    "area_bias_pct", "mean_IoU"])
        w.writerows(prows)

    # ── 분석 3 : 크기 분포 ──
    print("=" * 112)
    print("=== [크기 분포 유사도] GT vs Pred ===")
    print("=" * 112)
    print(f"{'set':>7} {'model':>9} {'변수':>10} {'nGT':>5} {'nPred':>6} "
          f"{'GT평균':>8} {'Pred평균':>9} {'GT중앙':>8} {'Pred중앙':>9} "
          f"{'KS D':>8} {'KS p':>9} {'Wasser':>9}")
    drows = []
    for scope in scopes:
        for lbl in labels:
            sel = [r for r in inst if r[1] == lbl
                   and (scope == "pooled" or r[0] == scope)]
            for col, nm, unit in [(7, "ED", "cm"), (6, "area", "cm2")]:
                G = np.array([float(r[col]) for r in sel if r[4] == "GT"])
                P = np.array([float(r[col]) for r in sel if r[4] == "Pred"])
                if len(G) < 3 or len(P) < 3:
                    print(f"{scope:>7} {lbl:>9} {nm+'('+unit+')':>10} {len(G):>5} "
                          f"{len(P):>6}" + "".join(f"{'nan':>9}" for _ in range(7)))
                    drows.append([scope, lbl, tr_of[lbl], nm, unit, len(G), len(P)]
                                 + [""] * 9)
                    continue
                ks = stats.ks_2samp(G, P)
                wd = float(stats.wasserstein_distance(G, P))
                print(f"{scope:>7} {lbl:>9} {nm+'('+unit+')':>10} {len(G):>5} "
                      f"{len(P):>6} {G.mean():>8.3f} {P.mean():>9.3f} "
                      f"{np.median(G):>8.3f} {np.median(P):>9.3f} "
                      f"{ks.statistic:>8.4f} {ks.pvalue:>9.4f} {wd:>9.4f}")
                drows.append([scope, lbl, tr_of[lbl], nm, unit, len(G), len(P),
                              rd(float(G.mean()), 4), rd(float(P.mean()), 4),
                              rd(float(np.median(G)), 4), rd(float(np.median(P)), 4),
                              rd(float(G.std(ddof=1)), 4), rd(float(P.std(ddof=1)), 4),
                              rd(float(ks.statistic), 4), rd(float(ks.pvalue), 6),
                              rd(wd, 4)])
        print()
    with open(OUT / "frag_distribution_all.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "train_res", "variable", "unit", "n_GT",
                    "n_Pred", "GT_mean", "Pred_mean", "GT_median", "Pred_median",
                    "GT_sd", "Pred_sd", "KS_D", "KS_p", "wasserstein"])
        w.writerows(drows)

    # ── 해상도축 요약 CSV (Origin 재작도용) ──
    ag = {(r[0], r[2], r[3]): r for r in srows}
    me = {(r[0], r[1]): r for r in prows}
    di = {(r[0], r[1], r[3]): r for r in drows}
    rows = []
    for scope in scopes:
        for lbl in labels:
            c = ag.get(("count", scope, lbl))
            f1 = ag.get(("ratio_frame_pct", scope, lbl))
            f2 = ag.get(("ratio_ofclod_pct", scope, lbl))
            m = me.get((scope, lbl))
            de = di.get((scope, lbl, "ED"))
            da = di.get((scope, lbl, "area"))
            det = [r for r in detect if r[1] == lbl
                   and (scope == "pooled" or r[0] == scope)]
            TPs = sum(r[7] for r in det)
            NGs = sum(r[5] for r in det)
            NPs = sum(r[6] for r in det)
            rows.append([scope, lbl, tr_of[lbl],
                         rd(TPs / NGs) if NGs else "", rd(TPs / NPs) if NPs else "",
                         *(c[9:12] if c else ["", "", ""]),       # bias MAE RMSE
                         *(c[12:13] if c else [""]),              # R2_1to1
                         *(c[14:15] if c else [""]),              # R2_fit
                         *(c[17:18] if c else [""]),              # MAPE
                         *(m[3:11] if m else [""] * 8),
                         *(de[13:16] if de else ["", "", ""]),
                         *(da[13:16] if da else ["", "", ""]),
                         *(f1[9:13] if f1 else [""] * 4),
                         *(f2[9:13] if f2 else [""] * 4)])
    with open(OUT / "frag_by_resolution.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "train_res", "Recall", "Precision",
                    "count_bias", "count_MAE", "count_RMSE", "count_R2_1to1",
                    "count_R2_fit", "count_MAPE",
                    "n_pairs", "ED_bias_cm", "ED_MAE_cm", "ED_RMSE_cm",
                    "ED_R2_1to1", "ED_r", "area_bias_pct", "mean_IoU",
                    "ED_KS_D", "ED_KS_p", "ED_wasserstein",
                    "area_KS_D", "area_KS_p", "area_wasserstein",
                    "ratioFrame_bias", "ratioFrame_MAE", "ratioFrame_RMSE",
                    "ratioFrame_R2_1to1",
                    "ratioClod_bias", "ratioClod_MAE", "ratioClod_RMSE",
                    "ratioClod_R2_1to1"])
        w.writerows(rows)

    verify(srows, prows, drows)
    plot(rows, labels, tr_of, scopes)
    print(f"\nCSV: {OUT}")


# ─────────────── 기존 512/736 결과와 대조 ───────────────
def verify(srows, prows, drows):
    print("\n" + "=" * 112)
    print("=== 기존 clod_frag_260907 결과(SR-512 / SR-736)와 재현성 대조 ===")
    print("=" * 112)
    checks, bad = 0, 0

    def cmp(name, new, old, tol=1e-4):
        nonlocal checks, bad
        checks += 1
        if old in ("", None):
            return
        try:
            o, n = float(old), float(new)
        except (TypeError, ValueError):
            return
        if abs(o - n) > tol:
            bad += 1
            print(f"  [불일치] {name}: 기존 {o} vs 신규 {n}")

    p = OUT / "frag_agreement.csv"
    if p.exists():
        old = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        newmap = {(r[0], r[2], r[3]): r for r in srows}
        for o in old:
            k = (o["quantity"], o["dataset"], o["model"])
            if k in newmap:
                n = newmap[k]
                for col, idx in [("MAE", 10), ("RMSE", 11), ("R2_1to1", 12),
                                 ("bias", 9), ("MAPE_pct", 17)]:
                    cmp(f"agreement {k} {col}", n[idx], o[col])
    p = OUT / "frag_matched_size_error.csv"
    if p.exists():
        old = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        newmap = {(r[0], r[1]): r for r in prows}
        for o in old:
            k = (o["dataset"], o["model"])
            if k in newmap:
                n = newmap[k]
                for col, idx in [("n_pairs", 3), ("ED_MAE_cm", 5),
                                 ("ED_RMSE_cm", 6), ("ED_R2_1to1", 7),
                                 ("area_bias_pct", 9)]:
                    cmp(f"size_error {k} {col}", n[idx], o[col])
    p = OUT / "frag_distribution.csv"
    if p.exists():
        old = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        newmap = {(r[0], r[1], r[3]): r for r in drows}
        for o in old:
            k = (o["dataset"], o["model"], o["variable"])
            if k in newmap:
                n = newmap[k]
                for col, idx in [("KS_D", 13), ("KS_p", 14), ("wasserstein", 15)]:
                    cmp(f"distribution {k} {col}", n[idx], o[col])
    print(f"  대조 {checks}개 항목 | 불일치 {bad}개 "
          f"-> {'완전 재현' if bad == 0 else '확인 필요'}")


# ─────────────────────────── 그림 ───────────────────────────
def plot(rows, labels, tr_of, scopes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"test": "#4C78A8", "val": "#E45756", "pooled": "#54A24B"}
    xs = [tr_of[l] for l in labels]

    def series(scope, col):
        d = {r[1]: r[col] for r in rows if r[0] == scope}
        return np.array([float(d[l]) if d.get(l) not in ("", None) else np.nan
                         for l in labels], float)

    PANELS = [
        (3, "Recall (clod, major axis >= 4 cm)", "(a) Detection recall", None),
        (8, "count R$^2$ (1:1 line)", "(b) Clod count agreement", (-1.05, 1.05)),
        (15, "ED R$^2$ (1:1 line)", "(c) Per-clod size fidelity", (-1.05, 1.05)),
        (17, "area bias (%)", "(d) Per-clod area bias", None),
        (20, "KS p (ED distribution)", "(e) Size-distribution similarity", None),
        (26, "coarse-clod area ratio MAE (%p, frame basis)",
         "(f) Fragmentation-index error", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.4))
    for ax, (col, ylab, ttl, ylim) in zip(axes.ravel(), PANELS):
        for scope in scopes:
            y = series(scope, col)
            ax.plot(xs, y, "o-", color=COL[scope], lw=1.9, ms=6.5, label=scope,
                    alpha=.9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], rotation=45, fontsize=8.5)
        ax.minorticks_off()
        ax.set_xlabel("training / inference resolution (px)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=11)
        if ylim:
            ax.set_ylim(*ylim)
        if col in (8, 15):
            ax.axhline(0, color="#888", ls=":", lw=1)
        if col == 17:
            ax.axhline(0, color="#888", ls=":", lw=1)
        if col == 20:
            ax.axhline(0.05, color="#888", ls="--", lw=1)
            ax.text(xs[0], 0.055, "p = 0.05", fontsize=8, color="#666")
        ax.grid(alpha=.25, lw=.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8.5)
    fig.suptitle("Fragmentation-assessment fidelity across single-resolution models "
                 "(matched condition, conf>=0.5, IoU>=0.5)", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    p = FIG / "frag_res_summary_260909.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print(f"저장: {p}")

    # 보조 그림 — 개수/면적비 오차의 해상도 의존
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
    for ax, (col, ylab, ttl) in zip(axes, [
            (6, "count MAE (clods / image)", "(a) Clod-count MAE"),
            (5, "count bias (clods / image)", "(b) Clod-count bias"),
            (30, "coarse-clod ratio MAE (%p, clod basis)",
             "(c) Coarse-fraction error")]):
        for scope in scopes:
            ax.plot(xs, series(scope, col), "o-", color=COL[scope], lw=1.9, ms=6.5,
                    label=scope, alpha=.9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], rotation=45, fontsize=8.5)
        ax.minorticks_off()
        ax.axhline(0, color="#888", ls=":", lw=1)
        ax.set_xlabel("training / inference resolution (px)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=11)
        ax.grid(alpha=.25, lw=.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8.5)
    fig.suptitle("Error of fragmentation indices vs resolution", fontsize=12.5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    p = FIG / "frag_res_error_260909.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print(f"저장: {p}")


if __name__ == "__main__":
    main()
