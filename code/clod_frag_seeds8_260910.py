"""
파쇄도 재현성 분석 — 9개 단일 해상도(SR) x 8 seed 확장.

배경
  기존 clod_frag_all_260909.py 는 해상도별 단일 모델(seed 0) 결과였다.
  본문 인식 지표(Table 3)가 8-seed 평균±SD 이므로 파쇄도 지표도 동일하게 맞춘다.

규약 — clod_frag_260907.py 와 완전히 동일 (계산 커널을 그대로 import 해 사용)
  - matched 조건 (학습 해상도 = 추론 해상도, 격자 736)
  - conf >= 0.5, 매칭 IoU >= 0.5, 예측 면적 내림차순 greedy
  - clod 정의: 장축 >= 4cm (MIN_MAJOR_FRAC = 0.08, 프레임 50cm)
  - 면적/직경은 원본 격자로 복원 후 cm 환산, ED = 2*sqrt(A/pi)
  - 가중치: seed 0 = weights/sr/R{res}.pt, seed 1-7 = weights/sr_seed/R{res}_s{n}.pt
    (results/seed_manifest_260824.csv 와 동일한 매핑)

지표 — 각 seed 마다 아래 4군을 계산한 뒤 8-seed 평균±SD 로 집계
  1) clod 개수     : 이미지별 GT vs Pred -> MAE, RMSE, R2(1:1), bias (+ FN/img, FP/img)
  2) 매칭 clod 크기 : ED MAE, ED R2(1:1), area_bias_pct (+ ED bias/RMSE, mean IoU)
  3) 크기 분포      : ED 분포 GT vs Pred -> KS D, KS p, Wasserstein
  4) 조대 clod 면적비: 4cm+ 면적 / 프레임 -> MAE, bias (+ RMSE, R2(1:1))

KS 검정 방법 (둘 다 산출·명시)
  (A) per-seed : seed 별 예측 ED 분포 vs GT ED 분포로 각각 KS -> 8개의 D, p 를 평균±SD.
                 p 의 산술평균은 통계적 의미가 약하므로 중앙값과 p<0.05 인 seed 수를 함께 기록.
                 ** 요약표의 기본(primary) 값 **
  (B) pooled   : 8 seed 예측을 합친 분포(n 이 약 8배) vs GT 분포로 KS 를 1회.
                 표본이 8배라 D 가 같아도 p 가 작아지므로 참고용으로 별도 CSV 에만 기록.

유효 표본 처리 — 2단계
  (i) 표본 부족: 매칭쌍 < 3 또는 예측 인스턴스 < 3 인 지표는 NaN 으로 남기고,
      요약에 n_valid(그 지표가 유효했던 seed 수)를 기록한다.
  (ii) 검출 붕괴: 저해상도(특히 128)는 seed 에 따라 검출이 무너져 허위 예측이
      폭증한다. 요약할 지표와 독립적인 검출 기준으로 붕괴 seed 를 판정한다.
        Precision < 0.10  또는  Recall < 0.10   ->  collapsed = 1
      붕괴 seed 를 포함/제외한 두 벌의 요약을 모두 출력한다.
        frag_seeds8_summary_260910.csv         전체 8 seed (그대로)
        frag_seeds8_summary_robust_260910.csv  붕괴 seed 제외 (n_seeds 로 표본 명시)
      본문에는 robust 판을 쓰고 붕괴 seed 수를 각주로 밝히는 것을 권장한다.

출력 (results/frag/)
  frag_seeds8_per_image_260910.csv     이미지 단위 원자료 (기존 *_all.csv 구조 + seed)
  frag_seeds8_instances_260910.csv     개체 단위 원자료   (동일 + seed)
  frag_seeds8_pairs_260910.csv         매칭쌍 원자료      (동일 + seed)
  frag_seeds8_detection_260910.csv     모델 x seed 검출 요약
  frag_seeds8_metrics_260910.csv       seed 단위 지표 (해상도 9 x seed 8 = 72행)
  frag_seeds8_summary_260910.csv       해상도별 8-seed 평균±SD (wide, Origin 용)
  frag_seeds8_summary_long_260910.csv  동일 내용 long 포맷
  frag_seeds8_ks_pooled_260910.csv     KS 방법 (B) pooled 결과
  figures/frag_seeds8_260910.png       평균±SD 오차막대 그림
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clod_frag_260907 as F        # noqa: E402  계산 커널 (load_gt/run_model/shape_stats/match)
import clod_frag_all_260909 as A    # noqa: E402  NaN 안전 agree()

STAMP = "260910"
OUT, FIG = F.OUT, F.FIG
FRAME_CM = F.FRAME_CM
NAN = float("nan")

TRAIN_RES = [128, 256, 384, 448, 512, 736, 1024, 1280, 1600]
MODELS = [(f"SR-{int(np.ceil(r / 32) * 32)}", r) for r in TRAIN_RES]
SEEDS = list(range(8))
SEED_DIR = F.WEIGHTS / "sr_seed"

MIN_PAIRS = 3       # 매칭쌍 크기 지표 최소 표본
MIN_INST = 3        # 분포 검정 최소 표본
COLLAPSE_PREC = 0.10  # 검출 붕괴 판정 — Precision 하한
COLLAPSE_REC = 0.10   # 검출 붕괴 판정 — Recall 하한

P_IMG = OUT / f"frag_seeds8_per_image_{STAMP}.csv"
P_INS = OUT / f"frag_seeds8_instances_{STAMP}.csv"
P_PAR = OUT / f"frag_seeds8_pairs_{STAMP}.csv"
P_DET = OUT / f"frag_seeds8_detection_{STAMP}.csv"

H_IMG = ["dataset", "model", "train_res", "eval_res", "seed", "file",
         "n_clod_GT", "n_clod_Pred", "area4cm_GT_cm2", "area4cm_Pred_cm2",
         "ratio_frame_GT", "ratio_frame_Pred", "area_allclod_GT_cm2",
         "area_allclod_Pred_cm2", "ratio_ofclod_GT", "ratio_ofclod_Pred",
         "TP", "FN", "FP"]
H_INS = ["dataset", "model", "train_res", "seed", "file", "source", "idx",
         "area_cm2", "ed_cm", "major_cm", "matched"]
H_PAR = ["dataset", "model", "train_res", "seed", "file", "gt_idx", "pred_idx",
         "iou", "area_GT_cm2", "area_Pred_cm2", "ed_GT_cm", "ed_Pred_cm",
         "major_GT_cm", "major_Pred_cm"]
H_DET = ["dataset", "model", "train_res", "eval_res", "seed", "n_images",
         "n_GT_clod", "n_Pred_clod", "TP", "Recall", "Precision"]

# (키, 표시명) — seed 단위로 계산하고 8-seed 로 집계할 지표
METRICS = [
    ("Recall", "Recall"), ("Precision", "Precision"),
    # 1) clod 개수
    ("count_MAE", "clod 개수 MAE (개/장)"),
    ("count_RMSE", "clod 개수 RMSE (개/장)"),
    ("count_R2_1to1", "clod 개수 R2(1:1)"),
    ("count_bias", "clod 개수 bias (개/장)"),
    ("FN_per_img", "FN/장"), ("FP_per_img", "FP/장"),
    # 2) 매칭 clod 크기
    ("n_pairs", "매칭쌍 수"),
    ("ED_MAE_cm", "ED MAE (cm)"),
    ("ED_R2_1to1", "ED R2(1:1)"),
    ("area_bias_pct", "면적 bias (%)"),
    ("ED_bias_cm", "ED bias (cm)"), ("ED_RMSE_cm", "ED RMSE (cm)"),
    ("mean_IoU", "매칭 IoU 평균"),
    # 3) 크기 분포 (ED)
    ("n_pred_inst", "예측 개체 수"),
    ("ED_KS_D", "ED KS D"), ("ED_KS_p", "ED KS p"),
    ("ED_wasserstein", "ED Wasserstein (cm)"),
    # 4) 조대 clod 면적비 (프레임 기준, %)
    ("ratioFrame_MAE", "면적비 MAE (%p, 프레임)"),
    ("ratioFrame_bias", "면적비 bias (%p, 프레임)"),
    ("ratioFrame_RMSE", "면적비 RMSE (%p, 프레임)"),
    ("ratioFrame_R2_1to1", "면적비 R2(1:1), 프레임"),
    # 참고 — 전체 clod 면적 대비
    ("ratioClod_MAE", "면적비 MAE (%p, clod 기준)"),
    ("ratioClod_bias", "면적비 bias (%p, clod 기준)"),
]
MKEYS = [k for k, _ in METRICS]


# ────────────────────────── 유틸 ──────────────────────────
def ckpt_of(res, seed):
    return F.WEIGHTS / "sr" / f"R{res}.pt" if seed == 0 \
        else SEED_DIR / f"R{res}_s{seed}.pt"


def rd(v, k=5):
    if v is None or v == "":
        return ""
    v = float(v)
    return "" if not np.isfinite(v) else round(v, k)


def fm(v, w=9, p=3):
    v = float(v) if v not in ("", None) else NAN
    return f"{'nan':>{w}}" if not np.isfinite(v) else f"{v:>{w}.{p}f}"


def append_rows(path, hdr, rows):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(hdr)
        w.writerows(rows)


def load_rows(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def fnum(r, k):
    v = r.get(k, "")
    return NAN if v in ("", None) else float(v)


# ────────────────────────── 추론 ──────────────────────────
def run_all(use, models, seeds, force=False):
    if force:
        for p in (P_IMG, P_INS, P_PAR, P_DET):
            p.unlink(missing_ok=True)
    done = {(r["dataset"], r["model"], int(r["seed"])) for r in load_rows(P_DET)}
    todo = [(ds, lbl, sd) for ds in use for lbl, _ in models for sd in seeds
            if (ds, lbl, sd) not in done]
    total = len(use) * len(models) * len(seeds)
    print("=" * 110)
    print(f"추론 — {len(models)}해상도 x {len(seeds)}seed x {len(use)}set "
          f"| 총 {total}셀, 남은 {len(todo)}셀")
    print("matched(train=eval), conf>=0.5, IoU>=0.5, clod 장축>=4cm")
    print("=" * 110)
    if not todo:
        print("모든 셀이 이미 누적되어 있습니다 — 집계로 넘어갑니다.")
        return

    for ds in use:
        if not any(t[0] == ds for t in todo):
            continue
        d = F.DATASETS[ds]
        gt = F.load_gt(d)
        files = sorted(gt)
        ngt = sum(len(gt[f]["masks"]) for f in files)
        print(f"\n[{ds}] {d.name} | 이미지 {len(files)}장 | GT clod(4cm+) {ngt}개")
        print(f"{'model':>9} {'seed':>4} {'eval':>5} {'Pred':>5} {'TP':>4} "
              f"{'Recall':>8} {'Precision':>10} {'sec':>6}")
        for lbl, tr in models:
            for sd in seeds:
                if (ds, lbl, sd) in done:
                    continue
                ck = ckpt_of(tr, sd)
                if not ck.exists():
                    print(f"  [skip] 가중치 없음: {ck}")
                    continue
                t0 = time.time()
                R, r32 = F.run_model(ck, gt, d, tr)
                TP = sum(R[f]["tp"] for f in files)
                NP = sum(len(R[f]["preds"]) for f in files)
                rec = TP / ngt if ngt else NAN
                pre = TP / NP if NP else NAN
                pim, ins, par = [], [], []
                for f in files:
                    g, r = gt[f], R[f]
                    cpp = g["cpp"]
                    gs = [F.shape_stats(m, cpp) for m in g["masks"]]
                    ps = [F.shape_stats(m, cpp) for m in r["preds"]]
                    frame = FRAME_CM ** 2
                    ga, pa = sum(x[0] for x in gs), sum(x[0] for x in ps)
                    pim.append([ds, lbl, tr, r32, sd, f, len(gs), len(ps),
                                round(ga, 3), round(pa, 3),
                                round(ga / frame, 5), round(pa / frame, 5),
                                round(g["total_clod_cm2"], 3),
                                round(r["pred_all_area_cm2"], 3),
                                rd(ga / g["total_clod_cm2"])
                                if g["total_clod_cm2"] > 0 else "",
                                rd(pa / r["pred_all_area_cm2"])
                                if r["pred_all_area_cm2"] > 0 else "",
                                r["tp"], len(gs) - r["tp"], len(ps) - r["tp"]])
                    inv = {di: gi for gi, (di, _) in r["pairs"].items()}
                    for i, (a, e, mj) in enumerate(gs):
                        ins.append([ds, lbl, tr, sd, f, "GT", i, round(a, 4),
                                    round(e, 4), round(mj, 4), int(r["flags"][i])])
                    for i, (a, e, mj) in enumerate(ps):
                        ins.append([ds, lbl, tr, sd, f, "Pred", i, round(a, 4),
                                    round(e, 4), round(mj, 4), int(i in inv)])
                    for gi, (di, iou) in r["pairs"].items():
                        par.append([ds, lbl, tr, sd, f, gi, di, round(iou, 4),
                                    round(gs[gi][0], 4), round(ps[di][0], 4),
                                    round(gs[gi][1], 4), round(ps[di][1], 4),
                                    round(gs[gi][2], 4), round(ps[di][2], 4)])
                append_rows(P_IMG, H_IMG, pim)
                append_rows(P_INS, H_INS, ins)
                append_rows(P_PAR, H_PAR, par)
                append_rows(P_DET, H_DET, [[ds, lbl, tr, r32, sd, len(files), ngt,
                                            NP, TP, rd(rec), rd(pre)]])
                print(f"{lbl:>9} {sd:>4} {r32:>5} {NP:>5} {TP:>4} {fm(rec, 8, 3)} "
                      f"{fm(pre, 10, 3)} {time.time() - t0:>6.1f}", flush=True)
                torch.cuda.empty_cache()


# ────────────────────── seed 단위 지표 ──────────────────────
def metrics_one(rr, pp, gg, d0, **ident):
    """한 (dataset, model, [eval_res,] seed) 셀의 4지표를 계산한다.

    rr/pp/gg/d0 = 그 셀의 이미지행 / 매칭쌍행 / 개체행 / 검출요약행.
    ident 는 결과 dict 에 그대로 실리는 식별 컬럼(dataset, model, seed 등).
    MR 확장 스크립트가 같은 수식을 쓰도록 이 함수 하나만 공유한다.
    """
    from scipy import stats
    m = dict(ident)
    m.update(n_images=len(rr),
             n_GT_clod=fnum(d0, "n_GT_clod"),
             n_Pred_clod=fnum(d0, "n_Pred_clod"),
             TP=fnum(d0, "TP"),
             Recall=fnum(d0, "Recall"),
             Precision=fnum(d0, "Precision"))

    # 1) clod 개수 — 이미지 단위 GT vs Pred
    a = A.agree([fnum(r, "n_clod_GT") for r in rr],
                [fnum(r, "n_clod_Pred") for r in rr])
    for src, dst in [("MAE", "count_MAE"), ("RMSE", "count_RMSE"),
                     ("R2_1to1", "count_R2_1to1"), ("bias", "count_bias")]:
        m[dst] = a[src]
    m["FN_per_img"] = float(np.mean([fnum(r, "FN") for r in rr]))
    m["FP_per_img"] = float(np.mean([fnum(r, "FP") for r in rr]))

    # 2) 매칭 clod 크기 — ED / 면적
    m["n_pairs"] = len(pp)
    if len(pp) >= MIN_PAIRS:
        ae = A.agree([fnum(r, "ed_GT_cm") for r in pp],
                     [fnum(r, "ed_Pred_cm") for r in pp])
        m["ED_MAE_cm"] = ae["MAE"]
        m["ED_RMSE_cm"] = ae["RMSE"]
        m["ED_bias_cm"] = ae["bias"]
        m["ED_R2_1to1"] = ae["R2_1to1"]
        m["area_bias_pct"] = float(np.mean(
            [(fnum(r, "area_Pred_cm2") - fnum(r, "area_GT_cm2"))
             / fnum(r, "area_GT_cm2") for r in pp]) * 100)
        m["mean_IoU"] = float(np.mean([fnum(r, "iou") for r in pp]))
    else:
        for c in ("ED_MAE_cm", "ED_RMSE_cm", "ED_bias_cm", "ED_R2_1to1",
                  "area_bias_pct", "mean_IoU"):
            m[c] = NAN

    # 3) 크기 분포 (ED) — 방법 (A) per-seed
    G = np.array([fnum(r, "ed_cm") for r in gg if r["source"] == "GT"])
    P = np.array([fnum(r, "ed_cm") for r in gg if r["source"] == "Pred"])
    m["n_gt_inst"], m["n_pred_inst"] = len(G), len(P)
    if len(G) >= MIN_INST and len(P) >= MIN_INST:
        ks = stats.ks_2samp(G, P)
        m["ED_KS_D"] = float(ks.statistic)
        m["ED_KS_p"] = float(ks.pvalue)
        m["ED_wasserstein"] = float(stats.wasserstein_distance(G, P))
        m["ED_GT_mean"] = float(G.mean())
        m["ED_Pred_mean"] = float(P.mean())
    else:
        for c in ("ED_KS_D", "ED_KS_p", "ED_wasserstein", "ED_GT_mean",
                  "ED_Pred_mean"):
            m[c] = NAN

    # 4) 조대 clod 면적비 (프레임 기준 / 전체 clod 기준), % 단위
    for tag, gk, pk in [("ratioFrame", "ratio_frame_GT", "ratio_frame_Pred"),
                        ("ratioClod", "ratio_ofclod_GT", "ratio_ofclod_Pred")]:
        sel = [r for r in rr if r[gk] != "" and r[pk] != ""]
        if sel:
            ar = A.agree([fnum(r, gk) * 100 for r in sel],
                         [fnum(r, pk) * 100 for r in sel])
            m[f"{tag}_MAE"] = ar["MAE"]
            m[f"{tag}_bias"] = ar["bias"]
            m[f"{tag}_RMSE"] = ar["RMSE"]
            m[f"{tag}_R2_1to1"] = ar["R2_1to1"]
            m[f"{tag}_n"] = len(sel)
        else:
            for s in ("MAE", "bias", "RMSE", "R2_1to1"):
                m[f"{tag}_{s}"] = NAN
            m[f"{tag}_n"] = 0

    # 검출 붕괴 판정 — 요약 지표와 독립적인 검출 기준으로만 판단
    rec, pre = m["Recall"], m["Precision"]
    m["collapsed"] = int((np.isfinite(pre) and pre < COLLAPSE_PREC)
                         or (np.isfinite(rec) and rec < COLLAPSE_REC))
    return m


def group_raw(keyfn, paths=None):
    """원자료 4종을 keyfn 으로 묶어 (이미지행, 매칭쌍행, 개체행, 검출행) 반환."""
    pim, ins, par, det = (load_rows(p) for p in (paths or (P_IMG, P_INS, P_PAR,
                                                           P_DET)))
    IM, IN, PA = {}, {}, {}
    for r in pim:
        IM.setdefault(keyfn(r), []).append(r)
    for r in ins:
        IN.setdefault(keyfn(r), []).append(r)
    for r in par:
        PA.setdefault(keyfn(r), []).append(r)
    return IM, IN, PA, {keyfn(r): r for r in det}


def agg_metric(sel, k):
    """seed 목록에서 지표 k 의 (유효수, 평균, SD, 최소, 최대)."""
    v = np.array([float(m.get(k, NAN)) for m in sel], float)
    v = v[np.isfinite(v)]
    n = len(v)
    return (n, float(v.mean()) if n else NAN,
            float(v.std(ddof=1)) if n > 1 else NAN,
            float(v.min()) if n else NAN, float(v.max()) if n else NAN)


def seed_metrics(use, models, seeds):
    IM, IN, PA, DE = group_raw(lambda r: (r["dataset"], r["model"],
                                          int(r["seed"])))
    rows = []
    for ds in use:
        for lbl, tr in models:
            for sd in seeds:
                k = (ds, lbl, sd)
                if k not in IM:
                    continue
                rows.append(metrics_one(IM[k], PA.get(k, []), IN.get(k, []),
                                        DE.get(k, {}), dataset=ds, model=lbl,
                                        train_res=tr, seed=sd))

    cols = ["dataset", "model", "train_res", "seed", "collapsed", "n_images",
            "n_GT_clod", "n_Pred_clod", "TP", "n_gt_inst", "ED_GT_mean",
            "ED_Pred_mean", "ratioFrame_n", "ratioClod_n"] + MKEYS
    with open(OUT / f"frag_seeds8_metrics_{STAMP}.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for m in rows:
            w.writerow([m.get(c, "") if c in ("dataset", "model")
                        else rd(m.get(c, ""), 6) for c in cols])
    return rows


# ────────────────────── 8-seed 집계 ──────────────────────
def summarize(rows, use, models, drop_collapsed=False, quiet=False):
    """8-seed 집계. drop_collapsed=True 면 검출 붕괴 seed 를 제외한 robust 판."""
    tag = "robust" if drop_collapsed else "all"
    out_long, out_wide = [], []
    for ds in use:
        for lbl, tr in models:
            sel = [m for m in rows if m["dataset"] == ds and m["model"] == lbl]
            n_all = len(sel)
            n_col = sum(m["collapsed"] for m in sel)
            if drop_collapsed:
                sel = [m for m in sel if not m["collapsed"]]
            if not sel:
                continue
            wide = [ds, lbl, tr, len(sel), n_all, n_col]
            for k, name in METRICS:
                n, mu, sd, mn, mx = agg_metric(sel, k)
                out_long.append([ds, lbl, tr, k, name, n, rd(mu, 6), rd(sd, 6),
                                 rd(mn, 6), rd(mx, 6), n_all, n_col])
                wide += [rd(mu, 6), rd(sd, 6), n]
            # KS p 보조 통계 — p 의 산술평균은 의미가 약하므로 중앙값/기각 seed 수 병기
            pv = np.array([float(m.get("ED_KS_p", NAN)) for m in sel], float)
            pv = pv[np.isfinite(pv)]
            wide += [rd(float(np.median(pv)) if len(pv) else NAN, 6),
                     int((pv < 0.05).sum()) if len(pv) else "", len(pv)]
            out_wide.append(wide)

    hdr = ["dataset", "model", "train_res", "n_seeds_used", "n_seeds_total",
           "n_seeds_collapsed"]
    for k, _ in METRICS:
        hdr += [f"{k}_mean", f"{k}_sd", f"{k}_n"]
    hdr += ["ED_KS_p_median", "ED_KS_p_n_reject_005", "ED_KS_p_n_valid"]
    sfx = "" if tag == "all" else "_robust"
    with open(OUT / f"frag_seeds8_summary{sfx}_{STAMP}.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(out_wide)
    with open(OUT / f"frag_seeds8_summary_long{sfx}_{STAMP}.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "train_res", "metric", "metric_label",
                    "n_valid_seeds", "mean", "sd", "min", "max",
                    "n_seeds_total", "n_seeds_collapsed"])
        w.writerows(out_long)
    if quiet:
        return out_long, out_wide

    # ── 콘솔 표 — 지표군별 해상도 x (평균±SD) ──
    GROUPS = [("1) clod 개수",
               [("Recall", 3), ("count_MAE", 2), ("count_RMSE", 2),
                ("count_R2_1to1", 2), ("count_bias", 2), ("FN_per_img", 2),
                ("FP_per_img", 2)]),
              ("2) 매칭 clod 크기",
               [("n_pairs", 1), ("ED_MAE_cm", 3), ("ED_R2_1to1", 3),
                ("area_bias_pct", 2), ("mean_IoU", 3)]),
              ("3) 크기 분포 (ED, per-seed KS)",
               [("n_pred_inst", 1), ("ED_KS_D", 3), ("ED_KS_p", 3),
                ("ED_wasserstein", 3)]),
              ("4) 조대 clod 면적비 (프레임 기준, %p)",
               [("ratioFrame_MAE", 2), ("ratioFrame_bias", 2),
                ("ratioFrame_RMSE", 2), ("ratioFrame_R2_1to1", 2)])]
    L = dict(METRICS)
    LG = {(r[0], r[1], r[3]): r for r in out_long}
    for ds in use:
        for gname, keys in GROUPS:
            note = ("붕괴 seed 제외" if drop_collapsed else "전체 8 seed")
            print("\n" + "=" * 110)
            print(f"[{ds}] {gname} — 8-seed 평균 ± SD ({note}, n = 유효 seed 수)")
            print("=" * 110)
            head = f"{'model':>9} {'res':>6}"
            for k, _ in keys:
                head += f" | {L[k]:>26}"
            print(head)
            for lbl, tr in models:
                line = f"{lbl:>9} {tr:>6}"
                for k, prec in keys:
                    r = LG.get((ds, lbl, k))
                    if r is None:
                        line += f" | {'-':>26}"
                        continue
                    mu = float(r[6]) if r[6] != "" else NAN
                    sd = float(r[7]) if r[7] != "" else NAN
                    s = ("nan" if not np.isfinite(mu)
                         else f"{mu:.{prec}f} ± "
                              f"{'nan' if not np.isfinite(sd) else f'{sd:.{prec}f}'}")
                    line += f" | {s + f' (n={r[5]})':>26}"
                print(line)
    return out_long, out_wide


# ────────────────── 검출 붕괴 seed 점검 ──────────────────
def collapse_table(rows, use, models):
    print("\n" + "=" * 110)
    print(f"=== 검출 붕괴 seed 점검 (Precision < {COLLAPSE_PREC} 또는 "
          f"Recall < {COLLAPSE_REC}) ===")
    print("    파쇄도 지표와 독립적인 검출 기준으로만 판정 -> 순환논리 회피")
    print("=" * 110)
    print(f"{'set':>6} {'model':>9} {'붕괴 seed':>22} {'n_붕괴':>7} {'Precision 범위':>22}")
    for ds in use:
        for lbl, tr in models:
            sel = [m for m in rows if m["dataset"] == ds and m["model"] == lbl]
            if not sel:
                continue
            bad = [m["seed"] for m in sel if m["collapsed"]]
            pr = [m["Precision"] for m in sel if np.isfinite(m["Precision"])]
            rng = f"{min(pr):.3f} – {max(pr):.3f}" if pr else "nan"
            print(f"{ds:>6} {lbl:>9} {(str(bad) if bad else '-'):>22} "
                  f"{len(bad):>7} {rng:>22}")


# ────────────────── KS 방법 (B) — 8seed 통합 분포 ──────────────────
def ks_pooled(use, models, seeds, metric_rows):
    from scipy import stats
    ins = load_rows(P_INS)
    bad = {(m["dataset"], m["model"], m["seed"]) for m in metric_rows
           if m["collapsed"]}
    rows = []
    print("\n" + "=" * 110)
    print("=== KS 방법 (B) — 8 seed 예측을 통합한 분포 vs GT 분포 (1회 검정) ===")
    print("    GT 는 seed 와 무관하므로 중복 없이 1벌만 사용, Pred 는 seed 합본")
    print("    표본이 약 8배로 커져 D 가 같아도 p 가 작아진다 -> 요약표의 기본값은 (A) per-seed")
    print("    'robust' 열은 검출 붕괴 seed 를 제외하고 합본한 결과")
    print("=" * 110)
    print(f"{'set':>6} {'model':>9} {'nGT':>5} {'nPred':>7} {'KS D':>8} "
          f"{'KS p':>11} {'Wasser':>8} | {'nPred_r':>8} {'KS D_r':>8} "
          f"{'KS p_r':>11} {'Wass_r':>8}")

    def ks_of(G, P):
        if len(G) < MIN_INST or len(P) < MIN_INST:
            return None
        ks = stats.ks_2samp(G, P)
        return (float(ks.statistic), float(ks.pvalue),
                float(stats.wasserstein_distance(G, P)), float(P.mean()))

    for ds in use:
        for lbl, tr in models:
            sel = [r for r in ins if r["dataset"] == ds and r["model"] == lbl
                   and int(r["seed"]) in seeds]
            s0 = [r for r in sel if int(r["seed"]) == seeds[0]]
            G = np.array([fnum(r, "ed_cm") for r in s0 if r["source"] == "GT"])
            P = np.array([fnum(r, "ed_cm") for r in sel if r["source"] == "Pred"])
            Pr = np.array([fnum(r, "ed_cm") for r in sel if r["source"] == "Pred"
                           and (ds, lbl, int(r["seed"])) not in bad])
            nb = len([s for s in seeds if (ds, lbl, s) in bad])
            a, b = ks_of(G, P), ks_of(G, Pr)
            rows.append([ds, lbl, tr, len(G), len(seeds), nb, len(P),
                         rd(float(G.mean()), 4),
                         *( [rd(a[3], 4), rd(a[0], 4), rd(a[1], 8), rd(a[2], 4)]
                            if a else [""] * 4),
                         len(Pr),
                         *( [rd(b[3], 4), rd(b[0], 4), rd(b[1], 8), rd(b[2], 4)]
                            if b else [""] * 4)])
            def cell(t, i, w=8, p=4):
                return fm(t[i] if t else NAN, w, p)
            print(f"{ds:>6} {lbl:>9} {len(G):>5} {len(P):>7} {cell(a, 0)} "
                  f"{cell(a, 1, 11, 6)} {cell(a, 2)} | {len(Pr):>8} "
                  f"{cell(b, 0)} {cell(b, 1, 11, 6)} {cell(b, 2)}")
    with open(OUT / f"frag_seeds8_ks_pooled_{STAMP}.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "train_res", "n_GT", "n_seeds_total",
                    "n_seeds_collapsed", "n_Pred_pooled_all", "GT_mean_ED_cm",
                    "Pred_mean_ED_cm_all", "KS_D_all", "KS_p_all",
                    "wasserstein_all", "n_Pred_pooled_robust",
                    "Pred_mean_ED_cm_robust", "KS_D_robust", "KS_p_robust",
                    "wasserstein_robust"])
        w.writerows(rows)
    return rows


# ────────────────── seed 0 재현성 대조 ──────────────────
def verify_seed0(rows, use):
    """seed 0 결과가 기존 frag_by_resolution.csv(단일 모델) 과 일치하는지 확인."""
    p = OUT / "frag_by_resolution.csv"
    if not p.exists():
        return
    old = {(r["dataset"], r["model"]): r for r in load_rows(p)}
    pairs = [("Recall", "Recall"), ("count_MAE", "count_MAE"),
             ("count_RMSE", "count_RMSE"), ("count_R2_1to1", "count_R2_1to1"),
             ("count_bias", "count_bias"), ("ED_MAE_cm", "ED_MAE_cm"),
             ("ED_R2_1to1", "ED_R2_1to1"), ("area_bias_pct", "area_bias_pct"),
             ("ED_KS_D", "ED_KS_D"), ("ED_wasserstein", "ED_wasserstein"),
             ("ratioFrame_MAE", "ratioFrame_MAE"),
             ("ratioFrame_bias", "ratioFrame_bias")]
    n = bad = 0
    print("\n" + "=" * 110)
    print("=== seed 0 재현성 대조 — 기존 clod_frag_all_260909 결과와 일치해야 함 ===")
    print("=" * 110)
    for m in rows:
        if m["seed"] != 0 or m["dataset"] not in use:
            continue
        o = old.get((m["dataset"], m["model"]))
        if not o:
            continue
        for nk, ok in pairs:
            ov = o.get(ok, "")
            nv = m.get(nk, NAN)
            if ov in ("", None) or not np.isfinite(float(nv)):
                continue
            n += 1
            if abs(float(ov) - float(nv)) > 5e-4:
                bad += 1
                print(f"  [불일치] {m['dataset']} {m['model']} {nk}: "
                      f"기존 {ov} vs 신규 {float(nv):.5f}")
    print(f"  대조 {n}개 항목 | 불일치 {bad}개 -> "
          f"{'완전 재현' if bad == 0 else '확인 필요'}")


# ─────────────────────────── 그림 ───────────────────────────
def plot(out_long, use, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"test": "#4C78A8", "val": "#E45756"}
    xs = [tr for _, tr in models]
    labels = [l for l, _ in models]

    def ms(ds, k):
        d = {r[1]: (r[6], r[7]) for r in out_long if r[0] == ds and r[3] == k}
        mu = np.array([float(d[l][0]) if d.get(l, ("", ""))[0] != "" else np.nan
                       for l in labels], float)
        sd = np.array([float(d[l][1]) if d.get(l, ("", ""))[1] != "" else np.nan
                       for l in labels], float)
        return mu, sd

    PANELS = [("Recall", "Recall (clod, major axis >= 4 cm)",
               "(a) Detection recall", None),
              ("count_MAE", "clod count MAE (clods / image)",
               "(b) Clod-count error", None),
              ("count_R2_1to1", "count R$^2$ (1:1 line)",
               "(c) Clod-count agreement", None),
              ("ED_MAE_cm", "ED MAE (cm)", "(d) Per-clod size error", None),
              ("area_bias_pct", "area bias (%)", "(e) Per-clod area bias", None),
              ("ED_KS_D", "KS D (ED distribution)",
               "(f) Size-distribution distance", None),
              ("ED_wasserstein", "Wasserstein distance (cm)",
               "(g) Size-distribution shift", None),
              ("ratioFrame_MAE", "coarse-clod area ratio MAE (%p)",
               "(h) Fragmentation-index error", None),
              ("ratioFrame_bias", "coarse-clod area ratio bias (%p)",
               "(i) Fragmentation-index bias", None)]
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 13.2))
    for ax, (k, ylab, ttl, ylim) in zip(axes.ravel(), PANELS):
        for ds in use:
            mu, sd = ms(ds, k)
            ax.errorbar(xs, mu, yerr=sd, fmt="o-", color=COL.get(ds, "#54A24B"),
                        lw=1.9, ms=6, capsize=3.5, elinewidth=1.2, alpha=.9,
                        label=f"{ds} (mean ± SD, 8 seeds)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs], rotation=45, fontsize=8.5)
        ax.minorticks_off()
        ax.set_xlabel("training / inference resolution (px)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=11)
        if ylim:
            ax.set_ylim(*ylim)
        if k in ("count_R2_1to1", "count_bias", "area_bias_pct",
                 "ratioFrame_bias"):
            ax.axhline(0, color="#888", ls=":", lw=1)
        ax.grid(alpha=.25, lw=.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
    fig.suptitle("Fragmentation-assessment fidelity across single-resolution models "
                 "— 8 seeds per resolution, collapsed seeds excluded\n"
                 "(matched condition, conf>=0.5, IoU>=0.5; error bars = SD across "
                 "seeds)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0.01, 1, 0.955])
    p = FIG / f"frag_seeds8_{STAMP}.png"
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print(f"\n저장: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="test", help="test / val / test,val")
    ap.add_argument("--models", default="", help="예: SR-512,SR-736 또는 512,736")
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--report", action="store_true",
                    help="추론 없이 누적 원자료로 집계만 다시 수행")
    ap.add_argument("--force", action="store_true", help="원자료 삭제 후 전체 재추론")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    use = [s.strip() for s in args.sets.split(",") if s.strip()]
    models = MODELS
    if args.models:
        keep = {s.strip() for s in args.models.split(",")}
        models = [m for m in MODELS if m[0] in keep or str(m[1]) in keep]
    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    if not args.report:
        run_all(use, models, seeds, force=args.force)
    rows = seed_metrics(use, models, seeds)
    if not rows:
        print("집계할 결과가 없습니다.")
        return
    collapse_table(rows, use, models)
    summarize(rows, use, models, drop_collapsed=False, quiet=True)   # 전체 8 seed
    out_long, _ = summarize(rows, use, models, drop_collapsed=True)  # 붕괴 seed 제외
    ks_pooled(use, models, seeds, rows)
    verify_seed0(rows, use)
    plot(out_long, use, models)
    print(f"\nCSV: {OUT}")


if __name__ == "__main__":
    main()
