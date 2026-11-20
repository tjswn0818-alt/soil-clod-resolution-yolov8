"""3단계 — 체크포인트·수렴.

(a) results.csv (SR 72 run): best epoch, epoch 100 값, val mask mAP50-95 가 최댓값의
    95% 에 처음 도달한 epoch, epoch 90 -> 100 변화량(mAP50-95(M), val/seg_loss).
    best epoch 은 (i) best.pt 의 ckpt['epoch'] + 1 (저장값), (ii) Ultralytics fitness
    재계산 = sum_{B,M} (0.1 * mAP50 + 0.9 * mAP50-95) 의 argmax(첫 최대) 두 가지로 산출.
(b) last.pt 를 best.pt 와 같은 조건으로 평가: review_infer_261026.infer_cell
    (conf 0.001, 같은 리사이즈·필터·IoU) -> P0, t 0.5 strict Recall/Precision,
    AP50, t 0.01 recall (score >= t). 예측은 r1 cache/preds_last 에 저장.
(c) epoch 별 가중치: SR seed 1-7 없음, seed 0 은 epoch0/epoch50 만 존재 -> 보고만.
"""
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

import r1_common as Q
import config  # noqa: E402,F401  (r1_common 이 경로 등록)

R, U, C, F = Q.R, Q.U, Q.C, Q.F
S = U.S
INV = Q.OUT / "s0_inventory_runs.csv"
PL = Q.CACHE / "preds_last"
PL.mkdir(parents=True, exist_ok=True)


def run_dirs():
    rows = list(csv.DictReader(open(INV, encoding="utf-8-sig")))
    return {(int(r["model"]), int(r["seed"])): config.RS_ROOT / r["run_dir"]
            for r in rows if r["family"] == "SR"}


def read_res(p):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    g = lambda k: np.array([float(r[k]) for r in rows])
    return dict(ep=g("epoch").astype(int), m95=g("metrics/mAP50-95(M)"),
                m50=g("metrics/mAP50(M)"), b95=g("metrics/mAP50-95(B)"),
                b50=g("metrics/mAP50(B)"), seg=g("val/seg_loss"))


def part_a(rd):
    rows, curves = [], {}
    bf_err = []
    for (tr, sd), d in sorted(rd.items()):
        r = read_res(d / "results.csv")
        # 설치된 Ultralytics(8.4.x) 의 seg fitness = mAP50-95(B) + mAP50-95(M)
        # (best.pt train_metrics['fitness'] 로 확인: 0.60878 + 0.51563 = 1.12442)
        fit = r["b95"] + r["m95"]
        be_fit = int(r["ep"][int(np.argmax(fit))])
        ck = torch.load(d / "weights" / "best.pt", map_location="cpu", weights_only=False)
        # strip_optimizer 가 epoch=-1, best_fitness=None 으로 바꾸므로 best.pt 의
        # train_metrics(저장 시점 val 지표)와 results.csv 행을 대조해 best epoch 을 찾는다.
        tm = ck["train_metrics"]
        err = (np.abs(r["b95"] - tm["metrics/mAP50-95(B)"])
               + np.abs(r["m95"] - tm["metrics/mAP50-95(M)"])
               + np.abs(r["m50"] - tm["metrics/mAP50(M)"]))
        be_ck = int(r["ep"][int(np.argmin(err))])
        bf_err.append(float(err.min()))
        mx = r["m95"].max()
        e95 = int(r["ep"][np.where(r["m95"] >= 0.95 * mx)[0][0]])
        i90, i100 = list(r["ep"]).index(90), list(r["ep"]).index(100)
        rows.append([tr, Q.LBL[tr], sd, be_ck, be_fit, int(be_ck == be_fit),
                     round(r["m95"][int(np.argmax(fit))], 5), round(r["m95"][i100], 5),
                     round(mx, 5), int(r["ep"][int(np.argmax(r["m95"]))]), e95,
                     round(r["m95"][i100] - r["m95"][i90], 5),
                     round(r["seg"][i100] - r["seg"][i90], 5),
                     round(r["seg"][i100], 5)])
        curves[(tr, sd)] = r
        del ck
    Q.wcsv("s3_convergence_seed.csv",
           ["train_res", "eval_res", "seed", "best_epoch_ckpt", "best_epoch_fitness",
            "agree", "mAP50-95M_at_best", "mAP50-95M_at_100", "mAP50-95M_max",
            "epoch_of_max_mAP50-95M", "first_epoch_ge95pct_max",
            "delta_mAP50-95M_90to100", "delta_val_seg_loss_90to100",
            "val_seg_loss_at_100"], rows)
    print(f"best_fitness 대조 최대 |차| = {np.nanmax(bf_err):.2e}, "
          f"fitness-argmax 와 일치 {sum(r[5] for r in rows)}/{len(rows)}")
    srows = []
    for tr in Q.RESL:
        s = np.array([r[3:] for r in rows if r[0] == tr], float)
        s[~np.isfinite(s)] = np.nan
        o = [tr, Q.LBL[tr]]
        for j in range(s.shape[1]):
            o += [round(float(np.nanmean(s[:, j])), 4),
                  round(float(np.nanstd(s[:, j], ddof=1)), 4)]
        srows.append(o)
    n_inf = sum(1 for r in rows if not np.isfinite(r[12]))
    print(f"val_seg_loss 비유한 값(Δ90→100) seed 수: {n_inf}")
    hdr = ["best_epoch_ckpt", "best_epoch_fitness", "agree", "mAP50-95M_at_best",
           "mAP50-95M_at_100", "mAP50-95M_max", "epoch_of_max", "first_epoch_ge95pct",
           "delta_mAP50-95M_90to100", "delta_val_seg_loss_90to100",
           "val_seg_loss_at_100"]
    Q.wcsv("s3_convergence_summary.csv",
           ["train_res", "eval_res"] + [f"{h}_{x}" for h in hdr for x in ("mean", "sd")],
           srows)
    return curves


def figure(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 10,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True,
                         "axes.linewidth": 1.0})
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    crow = []
    for k, tr in enumerate(Q.RESL):
        ep = curves[(tr, 0)]["ep"]
        for ax, key in zip(axs, ("m95", "seg")):
            yy = np.array([curves[(tr, s)][key] for s in Q.SEEDS], float)
            yy[~np.isfinite(yy)] = np.nan
            y = np.nanmean(yy, 0)
            ax.plot(ep, y, lw=1.2, color=cmap(k / (len(Q.RESL) - 1)),
                    label=f"{Q.LBL[tr]}")
        for i, e in enumerate(ep):
            crow.append([tr, Q.LBL[tr], int(e)] + [
                round(float(np.nanmean([v if np.isfinite(v) else np.nan for v in
                                        (curves[(tr, s)][key][i] for s in Q.SEEDS)])), 5)
                for key in ("m95", "seg")])
    for ax, yl in zip(axs, ("Validation mask mAP50-95", "Validation segmentation loss")):
        ax.axvspan(90.5, 100.5, color="0.85", lw=0, zorder=0)
        ax.set_xlim(0, 101)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(yl)
    axs[0].text(95.5, axs[0].get_ylim()[0] + 0.02 * np.ptp(axs[0].get_ylim()),
                "close\nmosaic", ha="center", va="bottom", fontsize=8)
    axs[0].legend(title="Input size (px)", frameon=False, fontsize=8, ncol=2)
    for ax, t in zip(axs, ("(a)", "(b)")):
        ax.text(0.02, 0.97, t, transform=ax.transAxes, va="top", fontweight="bold")
    fig.tight_layout()
    fig.savefig(Q.OUT / "s3_fig_convergence.png")
    fig.savefig(Q.OUT / "s3_fig_convergence.pdf")
    Q.wcsv("s3_convergence_curves_mean8seed.csv",
           ["train_res", "eval_res", "epoch", "val_mask_mAP50-95_mean",
            "val_seg_loss_mean"], crow)


def part_b(rd):
    import review_infer_261026 as RI      # 패치(NMS idx, 청크 마스크) 적용
    from ultralytics import YOLO
    D = U.load_data()
    for (tr, sd), d in sorted(rd.items()):
        todo = [ds for ds in U.DATASETS if not (PL / f"{ds}_{tr}_s{sd}.pkl").exists()]
        if not todo:
            continue
        ck = d / "weights" / "last.pt"
        model = YOLO(str(ck))
        t0 = time.time()
        for ds in todo:
            cell = RI.infer_cell(model, D[ds], Q.C.GRID[tr])
            tmp = PL / f"{ds}_{tr}_s{sd}.pkl.tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(dict(dataset=ds, train_res=tr, r32=Q.C.GRID[tr], seed=sd,
                                 ckpt=str(ck), images=cell), fh)
            tmp.replace(PL / f"{ds}_{tr}_s{sd}.pkl")
        del model
        torch.cuda.empty_cache()
        print(f"  last R{tr} s{sd} {time.time() - t0:.1f}s", flush=True)


def metrics(ims):
    A = {f: R.prep(ims[f]) for f in sorted(ims)}
    TP = FP = FN = 0
    for P in A.values():
        tp, fp, fn, _ = R.match_fast(P, 0.5, strict=True)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
    Rr, Pp, r01 = [], [], None
    for t in R.TS:
        a = np.array([R.match_fast(P, t)[:3] for P in A.values()]).sum(0)
        if abs(t - 0.01) < 1e-9:
            r01 = a[0] / (a[0] + a[2])
        if a[0] + a[1]:
            Rr.append(a[0] / (a[0] + a[2]))
            Pp.append(a[0] / (a[0] + a[1]))
    return [TP / (TP + FN), TP / (TP + FP) if TP + FP else 0.0, R.ap101(Rr, Pp), r01]


def part_b_eval():
    srow, rows = [], []
    for ds in U.DATASETS:
        for tr in Q.RESL:
            for sd in Q.SEEDS:
                b = metrics(C.load_cell(ds, tr, sd)["images"])
                with open(PL / f"{ds}_{tr}_s{sd}.pkl", "rb") as fh:
                    l = metrics(pickle.load(fh)["images"])
                rows.append([ds, tr, Q.LBL[tr], sd] + [round(x, 5) for x in b + l])
            s = np.array([r[4:] for r in rows if r[0] == ds and r[1] == tr], float)
            o = [ds, tr, Q.LBL[tr]]
            for j in range(8):
                o += [round(s[:, j].mean(), 4), round(s[:, j].std(ddof=1), 4)]
            srow.append(o)
    k = ["Recall_t05", "Precision_t05", "AP50", "Recall_t001"]
    Q.wcsv("s3_best_vs_last_seed.csv", ["dataset", "train_res", "eval_res", "seed"]
           + [f"best_{x}" for x in k] + [f"last_{x}" for x in k], rows)
    Q.wcsv("s3_best_vs_last_summary.csv", ["dataset", "train_res", "eval_res"]
           + [f"{p}_{x}_{m}" for p in ("best", "last") for x in k
              for m in ("mean", "sd")], srow)
    for r in srow:
        print(r)


if __name__ == "__main__":
    rd = run_dirs()
    curves = part_a(rd)
    figure(curves)
    print("3(a) 완료", flush=True)
    part_b(rd)
    part_b_eval()
    print("3단계 완료")
