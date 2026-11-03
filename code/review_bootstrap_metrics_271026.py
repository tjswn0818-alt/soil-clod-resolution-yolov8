"""
리뷰 대응 — Table 3·4 지표의 2단계 bootstrap 95% CI (SR 일치 조건, test 20장).

평가 조건 (원고 2.4절 = unified_4cm_260914 와 동일)
  conf > 0.5 (strict), 예측·GT 모두 장축 >= 4cm (타원 장축), INTER_LINEAR,
  매칭 IoU >= 0.5 예측 면적 내림차순 greedy. 면적·ED 는 원본 격자 (예측은 NEAREST 업샘플).
  지표 정의는 clod_frag_seeds8_260910.metrics_one 과 같다
  (매칭쌍 < 3 이면 mIoU/ED R²/Area bias NaN, 입단 < 3 이면 KS D NaN, NaN seed 는 평균에서 제외).

1) 사전 집계 (한 번만, 단일 프로세스): results_review_261026/preds (review_infer_261026) 에서
   score > 0.5 의 4cm 통과 예측만 남겨 res x seed x image 통계 저장
     bootstrap_pre_image_271026.csv  이미지: TP/FP/FN, 굵은 입단 수·면적합·면적비
     bootstrap_pre_pairs_271026.csv  매칭쌍: IoU, 면적, ED
     bootstrap_pre_inst_271026.csv   굵은 입단 ED (GT / 예측)
2) bootstrap: 이미지 20장과 seed 8개 동시 복원추출 (B=10,000, rng 2026, 이미지 인덱스 ->
   seed 인덱스 순서로 생성, 모든 해상도 공유). 재표본마다 '재표본 seed 별 계산 -> seed 평균'.
   이미지 재표본은 이미지별 가중치(뽑힌 횟수)로 바꿔 합계에 곱한다 (중복 포함 계산과 동일).
   KS D 는 가중 경험분포의 최대 차이 (= 중복 포함 표본의 ks_2samp 통계량).
   1단계(참고): seed 고정, 이미지만 재표집, 8 seed 평균.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco_mask
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import review_analysis_261026 as R      # noqa: E402
import review_common_261026 as C        # noqa: E402

U, F, S = C.U, C.F, C.U.S
OUT = C.OUT
SFX = "271026"
CONF = 0.5
B, RNG_SEED = 10000, 2026
RESL = U.TRAIN_RES
SEEDS = U.SEEDS
FRAME = F.FRAME_CM ** 2
MIN_PAIRS, MIN_INST = S.MIN_PAIRS, S.MIN_INST
P_IMG = OUT / f"bootstrap_pre_image_{SFX}.csv"
P_PAR = OUT / f"bootstrap_pre_pairs_{SFX}.csv"
P_INS = OUT / f"bootstrap_pre_inst_{SFX}.csv"
METRICS = [("Precision", "Table 3", +1), ("FN_per_img", "Table 3", -1),
           ("FP_per_img", "Table 3", -1), ("mIoU", "Table 4", +1),
           ("ED_R2_1to1", "Table 4", +1), ("area_bias_pct", "Table 4", 0),
           ("ED_KS_D", "Table 4", -1), ("count_MAE", "Table 4", -1),
           ("areaRatio_MAE", "Table 4", -1)]
PAIRS = [(1600, 512), (1600, 1024), (1024, 512), (1280, 512), (448, 512), (736, 512)]
LBL = {r: str(C.GRID[r]) for r in RESL}


# ══════════════ 1) 사전 집계 ══════════════
def precompute():
    D = U.load_data()["test"]
    files = sorted(D["gt"])
    gst = {}
    for f in files:                                       # GT 는 원본 격자에서 한 번만
        g = D["gt"][f]
        gst[f] = [F.shape_stats(m, g["cpp"]) for m in g["masks"]]
    img_rows, par_rows, ins_rows = [], [], []
    for tr in RESL:
        for sd in SEEDS:
            ims = C.load_cell("test", tr, sd)["images"]
            for f in files:
                g, im = D["gt"][f], ims[f]
                H, W, cpp = g["H"], g["W"], g["cpp"]
                kept = [im["preds"][i] for i in im["kept_idx"]]
                sel = [j for j, p in enumerate(kept) if p["score"] > CONF]
                n = im["n_gt"]
                iou = im["iou"][sel] if n else np.zeros((len(sel), 0))
                gt_to, tp = C.greedy(iou, [kept[j]["area"] for j in sel], n)
                ps = []
                for j in sel:
                    m = coco_mask.decode(kept[j]["rle"])
                    up = cv2.resize(m, (W, H), interpolation=U.INTERP_MASK)
                    ps.append(F.shape_stats(up, cpp))
                ga = sum(x[0] for x in gst[f])
                pa = sum(x[0] for x in ps)
                img_rows.append([tr, LBL[tr], sd, f, tp, len(sel) - tp, n - tp,
                                 len(sel), n, pa, ga, pa / FRAME * 100,
                                 ga / FRAME * 100])
                for gi, d in enumerate(gt_to):
                    if d < 0:
                        continue
                    a_g, e_g, _ = gst[f][gi]
                    a_p, e_p, _ = ps[d]
                    par_rows.append([tr, LBL[tr], sd, f, gi, float(iou[d, gi]),
                                     a_p, a_g, e_p, e_g])
                ins_rows += [[tr, LBL[tr], sd, f, "GT", x[1]] for x in gst[f]]
                ins_rows += [[tr, LBL[tr], sd, f, "Pred", x[1]] for x in ps]
        print(f"  사전 집계 R{tr} 완료", flush=True)
    R.wcsv(P_IMG.name, ["train_res", "eval_res", "seed", "file", "TP", "FP", "FN",
                        "n_pred_4cm", "n_gt_4cm", "area_pred_cm2", "area_gt_cm2",
                        "ratio_frame_pred_pct", "ratio_frame_gt_pct"], img_rows)
    R.wcsv(P_PAR.name, ["train_res", "eval_res", "seed", "file", "gt_idx", "iou",
                        "area_pred_cm2", "area_gt_cm2", "ed_pred_cm", "ed_gt_cm"],
           par_rows)
    R.wcsv(P_INS.name, ["train_res", "eval_res", "seed", "file", "source", "ed_cm"],
           ins_rows)


# ══════════════ 2) 셀별 지표 (이미지 가중치 행렬 C: (B, nI)) ══════════════
class Cell:
    """한 (res, seed) 의 이미지별 합계. metrics(Cw) -> 지표별 (B,) 배열."""

    def __init__(self, img, par, ins, files):
        fi = {f: i for i, f in enumerate(files)}
        nI = len(files)
        z = lambda: np.zeros(nI)
        self.TP, self.FP, self.FN = z(), z(), z()
        self.dN, self.dR = z(), z()
        for r in img:
            i = fi[r["file"]]
            self.TP[i], self.FP[i], self.FN[i] = (int(r["TP"]), int(r["FP"]),
                                                  int(r["FN"]))
            self.dN[i] = abs(int(r["n_pred_4cm"]) - int(r["n_gt_4cm"]))
            self.dR[i] = abs(float(r["ratio_frame_pred_pct"])
                             - float(r["ratio_frame_gt_pct"]))
        self.np_, self.siou, self.srel = z(), z(), z()
        self.sg, self.sg2, self.sd2 = z(), z(), z()
        for r in par:
            i = fi[r["file"]]
            ap, ag = float(r["area_pred_cm2"]), float(r["area_gt_cm2"])
            ep, eg = float(r["ed_pred_cm"]), float(r["ed_gt_cm"])
            self.np_[i] += 1
            self.siou[i] += float(r["iou"])
            self.srel[i] += (ap - ag) / ag
            self.sg[i] += eg
            self.sg2[i] += eg * eg
            self.sd2[i] += (ep - eg) ** 2
        v = np.array([float(r["ed_cm"]) for r in ins])
        isp = np.array([r["source"] == "Pred" for r in ins])
        im = np.array([fi[r["file"]] for r in ins], int)
        o = np.argsort(v, kind="mergesort")
        self.kv, self.kp, self.ki = v[o], isp[o], im[o]
        self.klast = np.r_[self.kv[1:] != self.kv[:-1], True]   # 동률 묶음의 끝

    def metrics(self, Cw):
        nI = Cw.shape[1]
        s = lambda x: Cw @ x
        tp, fp = s(self.TP), s(self.FP)
        out = {"Precision": np.divide(tp, tp + fp, out=np.zeros_like(tp),
                                      where=tp + fp > 0),
               "FN_per_img": s(self.FN) / nI, "FP_per_img": s(self.FP) / nI,
               "count_MAE": s(self.dN) / nI, "areaRatio_MAE": s(self.dR) / nI}
        npair = s(self.np_)
        okp = npair >= MIN_PAIRS
        with np.errstate(invalid="ignore", divide="ignore"):
            out["mIoU"] = np.where(okp, s(self.siou) / npair, np.nan)
            out["area_bias_pct"] = np.where(okp, s(self.srel) / npair * 100, np.nan)
            sst = s(self.sg2) - s(self.sg) ** 2 / npair
            out["ED_R2_1to1"] = np.where(okp & (sst > 1e-12),
                                         1 - s(self.sd2) / sst, np.nan)
        # 가중 KS
        W = Cw[:, self.ki]
        wG, wP = W * ~self.kp, W * self.kp
        nG, nP = wG.sum(1), wP.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cG = np.cumsum(wG, 1) / nG[:, None]
            cP = np.cumsum(wP, 1) / nP[:, None]
            d = np.abs(cG - cP)[:, self.klast].max(1)
        out["ED_KS_D"] = np.where((nG >= MIN_INST) & (nP >= MIN_INST), d, np.nan)
        return out


def nanmean_rows(X, axis):
    ok = np.isfinite(X)
    n = ok.sum(axis)
    with np.errstate(invalid="ignore"):
        return np.where(n > 0, np.where(ok, X, 0).sum(axis) / np.maximum(n, 1),
                        np.nan)


def load_cells():
    img, par, ins = (U.load_rows(p) for p in (P_IMG, P_PAR, P_INS))
    files = sorted({r["file"] for r in img})
    cells = {}
    for tr in RESL:
        for sd in SEEDS:
            k = (str(tr), str(sd))
            cells[(tr, sd)] = Cell(
                [r for r in img if (r["train_res"], r["seed"]) == k],
                [r for r in par if (r["train_res"], r["seed"]) == k],
                [r for r in ins if (r["train_res"], r["seed"]) == k], files)
    return cells, files, img, par, ins


# ══════════════ 3) 검산 ══════════════
def check_points(cells, files, img, par, ins):
    nI = len(files)
    one = np.ones((1, nI))
    pt = {(tr, sd): {k: float(v[0]) for k, v in cells[(tr, sd)].metrics(one).items()}
          for tr in RESL for sd in SEEDS}
    # (a) 가중 KS 가 scipy ks_2samp 와 같은지 (가중치 1)
    worst_ks = 0.0
    for tr in RESL:
        for sd in SEEDS:
            sel = [r for r in ins if r["train_res"] == str(tr) and r["seed"] == str(sd)]
            G = [float(r["ed_cm"]) for r in sel if r["source"] == "GT"]
            P = [float(r["ed_cm"]) for r in sel if r["source"] == "Pred"]
            if len(G) >= MIN_INST and len(P) >= MIN_INST:
                worst_ks = max(worst_ks, abs(stats.ks_2samp(G, P).statistic
                                             - pt[(tr, sd)]["ED_KS_D"]))
    # (b) 이미지 단위 카운트가 unified per-image 원자료와 같은지
    uimg = {(int(r["eval_res"]), int(r["seed"]), r["file"]): r
            for r in U.load_rows(U.P_IMG) if r["dataset"] == "test"
            and r["family"] == "SR"}
    cnt_bad = 0
    for r in img:
        u = uimg[(C.GRID[int(r["train_res"])], int(r["seed"]), r["file"])]
        cnt_bad += [int(r["TP"]), int(r["FN"]), int(r["FP"]), int(r["n_pred_4cm"]),
                    int(r["n_gt_4cm"])] != [int(u["TP"]), int(u["FN"]), int(u["FP"]),
                                            int(u["n_clod_Pred"]), int(u["n_clod_GT"])]
    # (c) 원고 표 (unified 요약) 와 대조
    t3 = {(r["dataset"], int(r["train_res"])): r
          for r in U.load_rows(U.OUT / "unified_4cm_sr_summary_260914.csv")}
    t4 = {(r["dataset"], int(r["eval_res"])): r
          for r in U.load_rows(U.OUT / "unified_4cm_frag_sr_260914.csv")}
    ref = {"Precision": (t3, "Precision_mean"), "FN_per_img": (t3, "FN_per_img_mean"),
           "FP_per_img": (t3, "FP_per_img_mean"), "mIoU": (t4, "mean_IoU_mean"),
           "ED_R2_1to1": (t4, "ED_R2_1to1_mean"),
           "area_bias_pct": (t4, "area_bias_pct_mean"),
           "ED_KS_D": (t4, "ED_KS_D_mean"), "count_MAE": (t4, "count_MAE_mean"),
           "areaRatio_MAE": (t4, "ratioFrame_MAE_mean")}
    rows, worst = [], {}
    for tr in RESL:
        for k, _, _ in METRICS:
            v = np.array([pt[(tr, sd)][k] for sd in SEEDS])
            new = float(np.nanmean(v))
            tab, col = ref[k]
            key = ("test", tr if tab is t3 else C.GRID[tr])
            old = float(tab[key][col])
            diff = new - old
            worst[k] = max(worst.get(k, 0), abs(diff))
            rows.append([tr, LBL[tr], k, round(new, 6), round(old, 6), f"{diff:.2e}",
                         int(np.isfinite(v).sum()),
                         int(round(new + 1e-9, 3) == round(old + 1e-9, 3)),
                         int(round(new + 1e-9, 4) == round(old + 1e-9, 4))])
    R.wcsv(f"bootstrap_metrics_check_{SFX}.csv",
           ["train_res", "eval_res", "metric", "point_new", "table_unified",
            "diff", "n_seeds_finite", "equal_3dp", "equal_4dp"], rows)
    print(f"[검산] 가중 KS vs scipy 최대 차 {worst_ks:.2e} | 이미지 카운트 불일치 "
          f"{cnt_bad}/{len(img)}")
    for k, w in worst.items():
        print(f"   {k:>14}: 원고(unified) 대비 최대 |차| {w:.2e}")
    return pt, rows


# ══════════════ 4) bootstrap ══════════════
def bootstrap(cells, files, pt):
    nI, nS = len(files), len(SEEDS)
    rng = np.random.default_rng(RNG_SEED)
    II = rng.integers(0, nI, (B, nI))
    SS = rng.integers(0, nS, (B, nS))
    Cw = np.zeros((B, nI))
    np.add.at(Cw, (np.repeat(np.arange(B), nI), II.ravel()), 1)
    M = {}                                       # (tr, metric) -> (seed, B)
    for tr in RESL:
        per = [cells[(tr, sd)].metrics(Cw) for sd in SEEDS]
        for k, _, _ in METRICS:
            M[(tr, k)] = np.stack([p[k] for p in per])
        print(f"  bootstrap R{tr} 완료", flush=True)
    two = {key: nanmean_rows(X[SS.T, np.arange(B)], 0) for key, X in M.items()}
    one = {key: nanmean_rows(X, 0) for key, X in M.items()}
    pct = lambda x: np.percentile(x[np.isfinite(x)], [2.5, 97.5])

    rows = []
    for tr in RESL:
        for k, tab, _ in METRICS:
            v = np.array([pt[(tr, sd)][k] for sd in SEEDS])
            fin = v[np.isfinite(v)]
            a2, a1 = two[(tr, k)], one[(tr, k)]
            l2, l1 = pct(a2), pct(a1)
            rows.append([tr, LBL[tr], tab, k, round(fin.mean(), 6),
                         round(fin.std(ddof=1), 6) if len(fin) > 1 else "",
                         len(fin), round(l2[0], 6), round(l2[1], 6),
                         round(l2[1] - l2[0], 6), int((~np.isfinite(a2)).sum()),
                         int((~np.isfinite(M[(tr, k)])).any(0).sum()),
                         round(l1[0], 6), round(l1[1], 6), round(l1[1] - l1[0], 6),
                         int((~np.isfinite(a1)).sum())])
    R.wcsv(f"bootstrap_metrics_ci_{SFX}.csv",
           ["train_res", "eval_res", "table", "metric", "point_8seed_mean",
            "seed_SD", "n_seeds_finite", "CI2_lo", "CI2_hi", "CI2_width",
            "n_rep_excluded_2stage", "n_rep_with_any_nan_seed",
            "CI1_img_lo", "CI1_img_hi", "CI1_width", "n_rep_excluded_1stage"], rows)

    prow = []
    for a, b in PAIRS:
        for k, tab, _ in METRICS:
            pa = np.nanmean([pt[(a, s)][k] for s in SEEDS])
            pb = np.nanmean([pt[(b, s)][k] for s in SEEDS])
            d2, d1 = two[(a, k)] - two[(b, k)], one[(a, k)] - one[(b, k)]
            f2, f1 = d2[np.isfinite(d2)], d1[np.isfinite(d1)]
            l2, l1 = np.percentile(f2, [2.5, 97.5]), np.percentile(f1, [2.5, 97.5])
            prow.append([f"{LBL[a]}-{LBL[b]}", tab, k, round(pa - pb, 6),
                         round(l2[0], 6), round(l2[1], 6),
                         round(float((f2 > 0).mean()), 4), int(l2[0] <= 0 <= l2[1]),
                         B - len(f2), round(l1[0], 6), round(l1[1], 6),
                         round(float((f1 > 0).mean()), 4), int(l1[0] <= 0 <= l1[1]),
                         B - len(f1)])
    R.wcsv(f"bootstrap_metrics_pairdiff_{SFX}.csv",
           ["pair", "table", "metric", "diff_point", "CI2_lo", "CI2_hi", "P2_gt0",
            "CI2_contains0", "n_rep_excluded_2stage", "CI1_img_lo", "CI1_img_hi",
            "P1_gt0", "CI1_contains0", "n_rep_excluded_1stage"], prow)
    return II, SS, rows, prow


# ══════════════ 5) MR[448-1024] vs SR 일치 ══════════════
def mr_vs_sr(II, SS):
    """unified per-image 원자료(TP/FN/FP)로 계산. MR seed s 와 SR seed s 를 짝지음."""
    rows = U.load_rows(U.P_IMG)
    files = sorted({r["file"] for r in rows if r["dataset"] == "test"})
    fi = {f: i for i, f in enumerate(files)}
    nI = len(files)
    Cw = np.zeros((B, nI))
    np.add.at(Cw, (np.repeat(np.arange(B), nI), II.ravel()), 1)
    sr_of = {448: "SR-448", 512: "SR-512", 736: "SR-736", 1024: "SR-1024"}

    def arr(model, ev):
        X = np.zeros((4, len(SEEDS), nI))               # TP, FN, FP, GT
        seen = np.zeros(len(SEEDS), int)
        for r in rows:
            if (r["dataset"] == "test" and r["model"] == model
                    and int(r["eval_res"]) == ev):
                s, i = int(r["seed"]), fi[r["file"]]
                X[:, s, i] = [int(r["TP"]), int(r["FN"]), int(r["FP"]),
                              int(r["n_clod_GT"])]
                seen[s] += 1
        assert (seen == nI).all(), (model, ev, seen)
        return X

    def mets(X, W):                                     # W: (B, nI) -> (4, seed, B)
        tp, fn, fp, gt = (W @ X[j].T for j in range(4))  # (B, seed)
        return np.stack([tp / gt,
                         np.divide(tp, tp + fp, out=np.zeros_like(tp),
                                   where=tp + fp > 0),
                         fn / nI, fp / nI]).transpose(0, 2, 1)

    names = ["Recall", "Precision", "FN_per_img", "FP_per_img"]
    out = []
    for ev in (448, 512, 736, 1024):
        Xm, Xs = arr("MR[448-1024]", ev), arr(sr_of[ev], ev)
        P0 = mets(Xm, np.ones((1, nI)))[:, :, 0] - mets(Xs, np.ones((1, nI)))[:, :, 0]
        Mm, Ms = mets(Xm, Cw), mets(Xs, Cw)
        for j, nm in enumerate(names):
            D = Mm[j] - Ms[j]                               # (seed, B), 같은 seed 짝
            d2 = D[SS.T, np.arange(B)].mean(0)
            d1 = D.mean(0)
            l2, l1 = np.percentile(d2, [2.5, 97.5]), np.percentile(d1, [2.5, 97.5])
            out.append([ev, nm, round(float(mets(Xm, np.ones((1, nI)))[j].mean()), 6),
                        round(float(mets(Xs, np.ones((1, nI)))[j].mean()), 6),
                        round(float(P0[j].mean()), 6), round(float(P0[j].std(ddof=1)), 6),
                        round(l2[0], 6), round(l2[1], 6),
                        round(float((d2 > 0).mean()), 4), int(l2[0] <= 0 <= l2[1]),
                        round(l1[0], 6), round(l1[1], 6),
                        round(float((d1 > 0).mean()), 4), int(l1[0] <= 0 <= l1[1]),
                        "MR seed s 와 SR seed s 를 같은 인덱스로 짝지어 재표집 "
                        "(두 모델의 seed 번호 사이에 실제 대응 관계가 있다는 가정)"])
    R.wcsv(f"bootstrap_mr_vs_sr_{SFX}.csv",
           ["eval_res", "metric", "MR_point", "SR_point", "diff_MR_minus_SR",
            "diff_seed_SD", "CI2_lo", "CI2_hi", "P2_gt0", "CI2_contains0",
            "CI1_img_lo", "CI1_img_hi", "P1_gt0", "CI1_contains0", "pairing_note"],
           out)
    return out


def main():
    if not (P_IMG.exists() and P_PAR.exists() and P_INS.exists()):
        precompute()
    cells, files, img, par, ins = load_cells()
    pt, _ = check_points(cells, files, img, par, ins)
    if "--check-only" in sys.argv:
        return
    II, SS, _, _ = bootstrap(cells, files, pt)
    mr_vs_sr(II, SS)
    print("완료")


if __name__ == "__main__":
    main()
