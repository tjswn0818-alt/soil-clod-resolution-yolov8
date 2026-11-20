"""증강 제거 실험 평가 (seed 0-7). 논문 본 평가와 같은 프로토콜.

추론: review_infer_261026.infer_cell (conf 0.001, max_det 1000, 검출 레벨 P3/P4/P5 기록)
      -> <AUG_DIR>/cache/preds/{ds}_{cond}_R{res}_s{sd}.pkl
기준선(base): 본 실험 YOLOv8m-seg SR-512 / SR-1600 seed 0-7 예측 캐시 (review_common.load_cell),
      P1 용 전체 GT IoU 는 r1_common.cell_all.
지표 (seed 별):
  P0: t 0.5 (score > t) Recall / Precision / FN·FP per image, AP50·Recall@0.01 (score >= t),
      검출 레벨 비율: TP 검출, conf > 0.5 & >=4cm 전체 검출.
  P1: r1_common.evaluate 와 같은 정의 + TP(>=4cm 예측) 레벨 비율.
부트스트랩 (데이터셋·프로토콜별): rng(2026) -> II(B, nI) 이미지 재표집은 조건·해상도 공통(같은 이미지
  집합 위의 짝지은 비교), seed 재표집 SS 는 조건 x 해상도마다 독립 (서로 다른 모델). B = 10,000, 2단계.
  대비: 조건별 512-1600, 조건-기준선 (해상도별), (조건 512-1600) - (기준선 512-1600).
수렴: 각 run results.csv - best epoch(best.pt train_metrics 대조), mask mAP50-95 최댓값의 95% 첫 도달
  epoch, epoch 90 -> 100 변화(mask mAP50-95, val seg loss).
그 밖: seed 별 recall 표, recall < 0.5 run 수, args.yaml 의 기준선(같은 해상도·seed) 대비 차이.
출력: <config.AUG_DIR>/aug8_*.csv
실행 순서: aug_a1_train.py -> aug_a2_eval.py (r1_s0_inventory_repro.py 의 s0_inventory_runs.csv 필요)
"""
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from pycocotools import mask as coco_mask

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402  code/config.py (경로 설정)
import r1_common as Q  # noqa: E402

R, U, C, F = Q.R, Q.U, Q.C, Q.F
ROOT = config.AUG_DIR
RES = [512, 1600]
SEEDS = list(range(8))
CONDS = ["base", "mosaic0", "scale0", "mosaic0_scale0"]
EXPECT = {"mosaic0": {"mosaic"}, "scale0": {"scale"}, "mosaic0_scale0": {"mosaic", "scale"}}
PATHKEYS = {"project", "name", "save_dir", "model", "data"}   # 경로 계열 (끝 3단계만 기록)
PC = ROOT / "cache" / "preds"
IC = ROOT / "cache" / "iouall"
INV = config.R1_DIR / "s0_inventory_runs.csv"
HDR = ["Recall_t05", "Precision_t05", "FN_per_img", "FP_per_img", "AP50", "Recall_t001",
       "TP_P3_pct", "TP_P4_pct", "TP_P5_pct", "Det4cm_P3_pct", "Det4cm_P4_pct", "Det4cm_P5_pct"]


def tag(c, r, s):
    return f"{c}_R{r}_s{s}"


def wcsv(name, hdr, rows):
    with open(ROOT / name, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)


# ---------------------------------------------------------------- 추론
def infer_all():
    import review_infer_261026 as RI
    import torch
    from ultralytics import YOLO
    PC.mkdir(parents=True, exist_ok=True)
    D = U.load_data()
    for c in CONDS[1:]:
        for res in RES:
            for sd in SEEDS:
                w = ROOT / "weights" / f"{tag(c, res, sd)}.pt"
                todo = [ds for ds in U.DATASETS
                        if not (PC / f"{ds}_{tag(c, res, sd)}.pkl").exists()]
                if not w.exists() or not todo:
                    continue
                model = YOLO(str(w))
                for ds in todo:
                    cell = RI.infer_cell(model, D[ds], C.GRID[res])
                    p = PC / f"{ds}_{tag(c, res, sd)}.pkl"
                    with open(str(p) + ".tmp", "wb") as fh:
                        pickle.dump(dict(dataset=ds, train_res=res, r32=C.GRID[res], seed=sd,
                                         cond=c, images=cell), fh)
                    Path(str(p) + ".tmp").replace(p)
                del model
                torch.cuda.empty_cache()
                print(f"  추론 {tag(c, res, sd)}", flush=True)


# ---------------------------------------------------------------- 셀 로드
def iou_all(ds, c, res, sd, ims):
    """r1_common.cell_all 과 같은 계산 (예측 셀만 증강 조건 모델)."""
    IC.mkdir(parents=True, exist_ok=True)
    p = IC / f"{ds}_{tag(c, res, sd)}.pkl"
    if p.exists():
        with open(p, "rb") as fh:
            return pickle.load(fh)
    r32 = C.GRID[res]
    G = Q.gt_all(ds)
    out = {}
    for f in sorted(ims):
        im, g = ims[f], G[f]
        gts_r = [cv2.resize(m, (r32, r32), interpolation=U.INTERP_MASK) for m in g["masks"]]
        grle = [coco_mask.encode(np.asfortranarray(m)) for m in gts_r]
        preds = im["preds"]
        n, k = len(grle), len(preds)
        iou = (np.asarray(coco_mask.iou([q["rle"] for q in preds], grle, [0] * n),
                          np.float64).reshape(k, n) if n and k else np.zeros((k, n)))
        big_idx = np.where(g["big"])[0]
        assert len(big_idx) == im["n_gt"]
        if len(im["kept_idx"]) and im["n_gt"]:
            assert np.allclose(iou[np.ix_(im["kept_idx"], big_idx)], im["iou"])
        out[f] = dict(score=np.array([q["score"] for q in preds], np.float64),
                      area=np.array([q["area"] for q in preds], np.float64),
                      major_px=np.array([q["major_px"] for q in preds], np.float64),
                      level=np.array([q["level"] for q in preds], np.int8),
                      iou=iou, big=g["big"].copy(), gt_major_cm=g["major_cm"].copy(),
                      gt_area_r=np.array([int(m.sum()) for m in gts_r]), r32=r32)
    with open(p, "wb") as fh:
        pickle.dump(out, fh)
    return out


def load_cell(c, ds, res, sd):
    if c == "base":
        return C.load_cell(ds, res, sd)["images"], Q.cell_all(ds, res, sd)
    p = PC / f"{ds}_{tag(c, res, sd)}.pkl"
    if not p.exists():
        return None, None
    with open(p, "rb") as fh:
        ims = pickle.load(fh)["images"]
    return ims, iou_all(ds, c, res, sd, ims)


# ---------------------------------------------------------------- 지표
def p1_levels(P, t=0.5):
    sel = P["score"] > t
    p2g = Q.match_sel(P["iou"], P["area"], sel, np.arange(len(P["big"])))
    m = p2g >= 0
    pk = sel & (P["major_px"] >= Q.FRAC4 * P["r32"]) & m
    return P["level"][pk]


def metrics_cell(ims, A1):
    """ims: 예측 셀(images), A1: 전체 GT IoU 셀. 반환 dict (+ 이미지별 tp/gt)."""
    files = sorted(ims)
    o = {}
    # P0
    Ps = [R.prep(ims[f]) for f in files]
    ev = [R.match_fast(P, 0.5, strict=True) for P in Ps]
    TP, FP, FN = (sum(e[i] for e in ev) for i in range(3))
    lv_tp = np.array([L for e in ev for _, L in e[3]], int)
    lv_all = np.concatenate([P["lv"][P["sc"] > 0.5] for P in Ps]) if Ps else np.array([])
    Rr, Pp, r01 = [], [], None
    for t in R.TS:
        a = np.array([R.match_fast(P, t)[:3] for P in Ps]).sum(0)
        if abs(t - 0.01) < 1e-9:
            r01 = a[0] / (a[0] + a[2])
        if a[0] + a[1]:
            Rr.append(a[0] / (a[0] + a[2]))
            Pp.append(a[0] / (a[0] + a[1]))
    nI = len(files)
    o["P0"] = dict(R=TP / (TP + FN), P=TP / (TP + FP) if TP + FP else 0.0, FN=FN / nI,
                   FP=FP / nI, AP50=R.ap101(Rr, Pp), R001=r01, lv_tp=lv_tp, lv_all=lv_all,
                   tp_i=np.array([e[0] for e in ev]), gt_i=np.array([e[0] + e[2] for e in ev]))
    # P1
    e5 = [Q.evaluate(A1[f], 0.5, "P1", True) for f in files]
    TP = sum(e["tp"] for e in e5); FN = sum(e["fn"] for e in e5)
    NP = sum(e["npred"] for e in e5); TPp = sum(e["tp_pred"] for e in e5)
    FP = sum(e["fp"] for e in e5)
    Rr, Pp, r01 = [], [], None
    for t in R.TS:
        ev2 = [Q.evaluate(A1[f], t, "P1", False) for f in files]
        tp = sum(e["tp"] for e in ev2); fn = sum(e["fn"] for e in ev2)
        npd = sum(e["npred"] for e in ev2); tpp = sum(e["tp_pred"] for e in ev2)
        if abs(t - 0.01) < 1e-9:
            r01 = tp / (tp + fn)
        if npd:
            Rr.append(tp / (tp + fn))
            Pp.append(tpp / npd)
    lv1 = np.concatenate([p1_levels(A1[f]) for f in files])
    o["P1"] = dict(R=TP / (TP + FN), P=TPp / NP if NP else 0.0, FN=FN / nI, FP=FP / nI,
                   AP50=R.ap101(Rr, Pp), R001=r01, lv_tp=lv1,
                   lv_all=o["P0"]["lv_all"],
                   tp_i=np.array([e["tp"] for e in e5]), gt_i=np.array([e["n_gt"] for e in e5]))
    return o


def pct(lv):
    lv = np.asarray(lv)
    return [round(float((lv == L).mean() * 100), 2) if len(lv) else np.nan for L in (3, 4, 5)]


# ---------------------------------------------------------------- run 폴더
def run_dir(c, res, sd):
    if c == "base":
        return config.RS_ROOT / [r["run_dir"] for r in csv.DictReader(open(INV, encoding="utf-8-sig"))
                                 if r["family"] == "SR" and int(r["model"]) == res
                                 and int(r["seed"]) == sd][0]
    return ROOT / "runs" / tag(c, res, sd) / "train"


def convergence():
    import torch
    rows = []
    for c in CONDS:
        for res in RES:
            for sd in SEEDS:
                d = run_dir(c, res, sd)
                if not (d / "results.csv").exists():
                    continue
                rr = list(csv.DictReader(open(d / "results.csv", encoding="utf-8")))
                g = lambda k: np.array([float(r[k]) for r in rr])
                ep = g("epoch").astype(int)
                if 100 not in ep or not (d / "weights" / "best.pt").exists():
                    continue                 # 학습 중이거나 미완료 run
                m95, b95, m50, seg = (g("metrics/mAP50-95(M)"), g("metrics/mAP50-95(B)"),
                                      g("metrics/mAP50(M)"), g("val/seg_loss"))
                ck = torch.load(d / "weights" / "best.pt", map_location="cpu", weights_only=False)
                tm = ck["train_metrics"]
                err = (np.abs(b95 - tm["metrics/mAP50-95(B)"]) + np.abs(m95 - tm["metrics/mAP50-95(M)"])
                       + np.abs(m50 - tm["metrics/mAP50(M)"]))
                be = int(ep[int(np.argmin(err))])
                mx = m95.max()
                e95 = int(ep[np.where(m95 >= 0.95 * mx)[0][0]])
                i90, i100 = list(ep).index(90), list(ep).index(100)
                rows.append([c, res, sd, len(ep), be, round(float(err.min()), 6), round(mx, 5),
                             int(ep[int(np.argmax(m95))]), e95, round(m95[i100], 5),
                             round(m95[i100] - m95[i90], 5), round(seg[i100] - seg[i90], 5),
                             round(seg[i100], 5), round(float(rr[-1]["time"]) / 60, 2)])
                del ck
    hdr = ["condition", "train_res", "seed", "n_epochs", "best_epoch", "best_match_err",
           "mAP50-95M_max", "epoch_of_max", "first_epoch_ge95pct_max", "mAP50-95M_at_100",
           "delta_mAP50-95M_90to100", "delta_val_seg_loss_90to100", "val_seg_loss_at_100",
           "train_min"]
    wcsv("aug8_convergence_seed.csv", hdr, rows)
    srows = []
    for c in CONDS:
        for res in RES:
            s = np.array([r[4:] for r in rows if r[0] == c and r[1] == res], float)
            if not len(s):
                continue
            o = [c, res, len(s)]
            for j in range(s.shape[1]):
                o += [round(float(np.nanmean(s[:, j])), 4),
                      round(float(np.nanstd(s[:, j], ddof=1)), 4) if len(s) > 1 else np.nan]
            srows.append(o)
    wcsv("aug8_convergence_summary.csv", ["condition", "train_res", "n_seeds"]
         + [f"{h}_{m}" for h in hdr[4:] for m in ("mean", "sd")], srows)


def seed_tables():
    rows = list(csv.DictReader(open(ROOT / "aug8_seed_metrics.csv", encoding="utf-8-sig")))
    P = defaultdict(dict)
    for r in rows:
        P[(r["condition"], r["protocol"], r["dataset"], int(r["train_res"]))][int(r["seed"])] = \
            float(r["Recall_t05"])
    out, col = [], []
    for k in sorted(P, key=lambda k: (k[1], k[2], CONDS.index(k[0]), k[3])):
        v = [P[k].get(s, np.nan) for s in SEEDS]
        a = np.array(v)
        out.append([*k, *[round(x, 4) for x in v], round(float(np.nanmean(a)), 4),
                    round(float(np.nanstd(a, ddof=1)), 4), round(float(np.nanmin(a)), 4)])
        bad = [s for s, x in zip(SEEDS, v) if x < 0.5]
        col.append([*k, len(bad), " ".join(map(str, bad))])
    wcsv("aug8_seed_recall_table.csv", ["condition", "protocol", "dataset", "train_res"]
         + [f"seed{s}" for s in SEEDS] + ["mean", "sd", "min"], out)
    wcsv("aug8_collapse_count.csv", ["condition", "protocol", "dataset", "train_res",
                                     "n_recall_lt0.5", "seeds_lt0.5"], col)


def _short(v):
    """경로 값은 끝 3단계만 남김 (절대경로를 결과에 남기지 않음)."""
    return "/".join(Path(str(v).replace("\\", "/")).parts[-3:]) if v else v


def args_diff():
    rows, dd = [], []
    for c in CONDS[1:]:
        for res in RES:
            for sd in SEEDS:
                fa = run_dir(c, res, sd) / "args.yaml"
                fb = run_dir("base", res, sd) / "args.yaml"
                if not fa.exists():
                    rows.append([c, res, sd, "missing", "", "", ""])
                    continue
                a = yaml.safe_load(open(fa, encoding="utf-8"))
                b = yaml.safe_load(open(fb, encoding="utf-8"))
                keys = sorted(set(a) | set(b))
                diff = [k for k in keys if a.get(k) != b.get(k)]
                other = [k for k in diff if k not in EXPECT[c] and k not in PATHKEYS]
                ok_exp = all(k in diff for k in EXPECT[c])
                for k in diff:
                    pk = k in PATHKEYS
                    dd.append([c, res, sd, k, _short(b.get(k)) if pk else b.get(k),
                               _short(a.get(k)) if pk else a.get(k),
                               "intended" if k in EXPECT[c] else ("path" if pk else "OTHER")])
                rows.append([c, res, sd, int(ok_exp), " ".join(diff), " ".join(other),
                             f"mosaic={a['mosaic']} scale={a['scale']} close_mosaic={a['close_mosaic']} "
                             f"mixup={a['mixup']} copy_paste={a['copy_paste']} degrees={a['degrees']} "
                             f"translate={a['translate']} fliplr={a['fliplr']} flipud={a['flipud']} "
                             f"hsv={a['hsv_h']}/{a['hsv_s']}/{a['hsv_v']} epochs={a['epochs']} seed={a['seed']}"])
    wcsv("aug8_args_check.csv", ["condition", "train_res", "seed", "intended_changed",
                                 "diff_keys_all", "diff_keys_unexpected", "aug_values"], rows)
    wcsv("aug8_args_diff_detail.csv", ["condition", "train_res", "seed", "key", "baseline",
                                       "condition_value", "kind"], dd)


# ---------------------------------------------------------------- main
def main():
    infer_all()
    seed_rows, M = [], {}
    for c in CONDS:
        for ds in U.DATASETS:
            for res in RES:
                for sd in SEEDS:
                    ims, A1 = load_cell(c, ds, res, sd)
                    if ims is None:
                        print(f"  [누락] {tag(c, res, sd)} {ds}")
                        continue
                    o = metrics_cell(ims, A1)
                    for pr in ("P0", "P1"):
                        x = o[pr]
                        M[(c, ds, pr, res, sd)] = x
                        seed_rows.append([c, pr, ds, res, C.GRID[res], sd,
                                          round(x["R"], 5), round(x["P"], 5), round(x["FN"], 4),
                                          round(x["FP"], 4), round(x["AP50"], 5),
                                          round(x["R001"], 5), *pct(x["lv_tp"]),
                                          *pct(x["lv_all"])])
            print(f"  지표 {c}", flush=True)
    wcsv("aug8_seed_metrics.csv", ["condition", "protocol", "dataset", "train_res", "eval_res",
                                   "seed"] + HDR, seed_rows)
    srows, side = [], []
    for pr in ("P0", "P1"):
        for ds in U.DATASETS:
            for res in RES:
                mu_b = None
                for c in CONDS:
                    s = np.array([r[6:] for r in seed_rows if r[0] == c and r[1] == pr
                                  and r[2] == ds and r[3] == res], float)
                    if not len(s):
                        continue
                    mu = np.nanmean(s, 0)
                    sdv = np.nanstd(s, 0, ddof=1) if len(s) > 1 else np.full(s.shape[1], np.nan)
                    srows.append([c, pr, ds, res, C.GRID[res], len(s)]
                                 + [v for a, b in zip(mu, sdv) for v in (round(a, 4), round(b, 4))])
                    if c == "base":
                        mu_b = mu
                    elif mu_b is not None:
                        side.append([c, pr, ds, res, C.GRID[res]]
                                    + [v for x, y in zip(mu_b, mu)
                                       for v in (round(x, 4), round(y, 4), round(y - x, 4))])
    wcsv("aug8_summary.csv", ["condition", "protocol", "dataset", "train_res", "eval_res",
                              "n_seeds"] + [f"{h}_{m}" for h in HDR for m in ("mean", "sd")], srows)
    wcsv("aug8_vs_baseline.csv", ["condition", "protocol", "dataset", "train_res", "eval_res"]
         + [f"{h}_{a}" for h in HDR for a in ("base", "cond", "cond_minus_base")], side)

    # 부트스트랩
    brows = []
    for ds in U.DATASETS:
        for pr in ("P0", "P1"):
            ok = [c for c in CONDS if all((c, ds, pr, r, s) in M for r in RES for s in SEEDS)]
            if "base" not in ok:
                continue
            nI = len(M[("base", ds, pr, RES[0], 0)]["tp_i"])
            gt = M[("base", ds, pr, RES[0], 0)]["gt_i"]
            rng = np.random.default_rng(R.RNG_SEED)
            II = rng.integers(0, nI, (R.B, nI))
            SS = rng.integers(0, len(SEEDS), (len(CONDS), len(RES), R.B, len(SEEDS)))
            bt, pt = {}, {}
            for ci, c in enumerate(CONDS):
                if c not in ok:
                    continue
                for a, res in enumerate(RES):
                    assert np.array_equal(M[(c, ds, pr, res, 0)]["gt_i"], gt)
                    tp = np.stack([M[(c, ds, pr, res, s)]["tp_i"] for s in SEEDS])
                    rs = tp[:, II].sum(-1) / gt[II].sum(1)
                    bt[(c, res)] = rs[SS[ci, a].T, np.arange(R.B)].mean(0)
                    pt[(c, res)] = float((tp.sum(1) / gt.sum()).mean())

            def add(c, contrast, d, p_a, p_b):
                lo, hi = np.percentile(d, [2.5, 97.5])
                brows.append([pr, ds, c, contrast, round(p_a, 4), round(p_b, 4),
                              round(p_a - p_b, 4), round(lo, 4), round(hi, 4),
                              round(float((d > 0).mean()), 4), int(lo <= 0 <= hi)])

            for c in ok:
                add(c, "R512-R1600", bt[(c, 512)] - bt[(c, 1600)], pt[(c, 512)], pt[(c, 1600)])
            for c in ok:
                if c == "base":
                    continue
                for res in RES:
                    add(c, f"cond-base@R{res}", bt[(c, res)] - bt[("base", res)],
                        pt[(c, res)], pt[("base", res)])
                dd = (bt[(c, 512)] - bt[(c, 1600)]) - (bt[("base", 512)] - bt[("base", 1600)])
                add(c, "(R512-R1600)cond-(R512-R1600)base", dd,
                    pt[(c, 512)] - pt[(c, 1600)], pt[("base", 512)] - pt[("base", 1600)])
    wcsv("aug8_bootstrap.csv", ["protocol", "dataset", "condition", "contrast", "value_a",
                                "value_b", "delta_point", "CI_2stage_lo", "CI_2stage_hi",
                                "P_gt0_2stage", "CI_contains0"], brows)
    convergence()
    seed_tables()
    args_diff()
    print("평가 완료")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["infer"]:
        infer_all()
    elif a == ["conv"]:
        convergence()
    elif a == ["args"]:
        args_diff()
    else:
        main()
