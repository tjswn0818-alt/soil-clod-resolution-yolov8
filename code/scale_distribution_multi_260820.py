"""
P3/P4/P5 탐지 스케일 분포 — 다중 모델 x 다중 데이터셋 확장판.

측정 방식은 scale_distribution_260820.py 와 동일:
  (A) 고신뢰 셀 : head.cv3[i](클래스 분기)에 forward hook -> sigmoid -> conf 초과 셀 수 (NMS 전)
  (B) 최종 검출 : 같은 forward 의 raw 출력에서 conf 필터 + NMS(IoU 0.7) 후 살아남은
                  anchor index 를 레벨 경계로 되돌려 카운트 (객체 1개 = 1건)
  * 둘 다 '추론 시 활성 분포'이며 학습 시 TaskAlignedAssigner 배정 결과가 아니다.
  * YOLOv8 Detect 는 cat(cv2=4*reg_max, cv3=nc) 순이므로 클래스 로짓은 뒤쪽 nc채널.
    본 스크립트는 cv3 분기를 직접 hook 하므로 채널 배치와 무관하다.

출력
  results/scale_distribution_multi_<tag>.csv        통합 long CSV
  results/scale_distribution_summary_<tag>.csv      조합별 요약
  figures/scale_dist_<model>_<dataset>_<tag>.png    조합별 누적막대 (10장)
  figures/scale_distribution_overview_<tag>.png     5x2 개요 그리드

사용:
  python scale_distribution_multi_260820.py --verify      # 1조합 검증만
  python scale_distribution_multi_260820.py               # 전체 실행
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_pixel_final as E  # noqa: E402

LEVELS = ["P3", "P4", "P5"]
STRIDES = [8, 16, 32]

MODELS = [
    ("SR-736",       E.WEIGHTS / "sr" / "R736.pt"),
    ("SR-1600",      E.WEIGHTS / "sr" / "R1600.pt"),
    ("MR[128-1600]", E.WEIGHTS / "mr" / "C.pt"),
    ("MR[512-1600]", E.WEIGHTS / "mr" / "C2.pt"),
    ("MR[512-1024]", E.WEIGHTS / "mr" / "C3.pt"),
]
DATASETS = [
    ("test", E.PIXEL_ROOT / "18.v1i.coco-segmentation" / "train"),
    ("val",  E.PIXEL_ROOT / "pixel_split" / "val"),
]
IMGSZS = [128, 256, 512, 736, 1024, 1600]

# recall 대조표 (모델키 -> CSV 소스)
SR_KEY = {"SR-736": "736", "SR-1600": "1600"}
MR_KEY = {"MR[128-1600]": "C", "MR[512-1600]": "C2", "MR[512-1024]": "C3"}


# ────────────────────────────────────────────── 측정 코어
def flatten_tensors(o, acc=None):
    acc = [] if acc is None else acc
    if torch.is_tensor(o):
        acc.append(o)
    elif isinstance(o, (tuple, list)):
        for it in o:
            flatten_tensors(it, acc)
    return acc


def pick_raw(preds, nc, nm):
    want = 4 + nc + nm
    best = None
    for t in flatten_tensors(preds):
        if t.ndim == 3 and t.shape[1] == want:
            if best is None or t.shape[-1] > best.shape[-1]:
                best = t
    if best is None:
        raise RuntimeError(f"raw 출력(채널 {want}) 없음")
    return best


class LevelHook:
    def __init__(self, head):
        if not (hasattr(head, "cv3") and len(head.cv3) == head.nl):
            raise RuntimeError("Detect 헤드에 cv3 분기가 없습니다.")
        self.out, self.handles = {}, []
        for i, m in enumerate(head.cv3):
            self.handles.append(m.register_forward_hook(self._mk(i)))

    def _mk(self, i):
        def fn(mod, inp, out):
            self.out[i] = out.detach()
        return fn

    def clear(self):
        self.out = {}

    def close(self):
        for h in self.handles:
            h.remove()


def nms_levels(y, nc, conf, iou_thres, level_sizes):
    from torchvision.ops import nms
    box, cls = y[0, :4, :], y[0, 4:4 + nc, :]
    score, _ = cls.max(0)
    keep0 = (score > conf).nonzero(as_tuple=True)[0]
    counts = [0, 0, 0]
    if keep0.numel() == 0:
        return counts
    b = box[:, keep0]
    xyxy = torch.stack([b[0] - b[2] / 2, b[1] - b[3] / 2,
                        b[0] + b[2] / 2, b[1] + b[3] / 2], 1)
    idx = keep0[nms(xyxy.float(), score[keep0].float(), iou_thres)]
    bounds = np.cumsum([0] + level_sizes)
    for a in idx.tolist():
        for li in range(3):
            if bounds[li] <= a < bounds[li + 1]:
                counts[li] += 1
                break
    return counts


def run_one(net, hook, img, r32, nc, nm, conf, iou_thres, device):
    hook.clear()
    im = cv2.resize(img, (r32, r32), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255)
    t = t.unsqueeze(0).to(device)
    with torch.inference_mode():
        preds = net(t)
    y = pick_raw(preds, nc, nm)
    cells, grid = [], []
    for i in range(3):
        mx = hook.out[i].sigmoid().amax(1)
        cells.append(int((mx > conf).sum().item()))
        grid.append(int(mx.shape[-1] * mx.shape[-2]))
    return cells, grid, nms_levels(y, nc, conf, iou_thres, grid)


def load_model(path, device):
    ym = YOLO(str(path))
    net = ym.model.to(device).eval()
    head = ym.model.model[-1]
    return net, head, LevelHook(head), int(head.nc), int(getattr(head, "nm", 32))


def gt_count(ddir):
    coco = json.load(open(ddir / "_annotations.coco.json", encoding="utf-8"))
    ann_by = {}
    for a in coco["annotations"]:
        ann_by.setdefault(a["image_id"], []).append(a)
    n = 0
    for im in coco["images"]:
        H, W = im["height"], im["width"]
        thr = E.MIN_MAJOR_FRAC * max(H, W)
        for a in ann_by.get(im["id"], []):
            m = E.seg_to_mask(a["segmentation"], H, W).astype(np.uint8)
            if m.sum() and E.major_axis_px(m) >= thr:
                n += 1
    return n


# ────────────────────────────────────────────── recall 대조
def load_recall():
    """(model, dataset) -> {eval_res: recall}"""
    R = {}
    for ds, fn in [("test", "recall_matrix_test.csv"), ("val", "recall_matrix_val.csv")]:
        p = E.RESULTS / fn
        if not p.exists():
            continue
        for row in csv.DictReader(open(p, encoding="utf-8-sig")):
            for name, key in SR_KEY.items():
                if row["train_res"] == key:
                    R[(name, ds)] = {e: float(row[f"eval_{e}"]) for e in IMGSZS
                                     if f"eval_{e}" in row}
    p = E.RESULTS / "res_sweep_pixel_final_long.csv"
    if p.exists():
        for row in csv.DictReader(open(p, encoding="utf-8-sig")):
            for name, key in MR_KEY.items():
                if row["train_res"] == key:
                    R.setdefault((name, "test"), {})[int(row["eval_res"])] = float(row["Recall"])
    p = E.RESULTS / "val_crossres_MR_260814.csv"
    if p.exists():
        for row in csv.DictReader(open(p, encoding="utf-8-sig")):
            for name, key in MR_KEY.items():
                if row["model"] == key:
                    R.setdefault((name, "val"), {})[int(row["eval_res"])] = float(row["Recall"])
    # eval736 보정본 반영
    p = E.RESULTS / "eval736_corrected_260814.csv"
    if p.exists():
        inv = {**{v: k for k, v in SR_KEY.items()}, **{v: k for k, v in MR_KEY.items()}}
        for row in csv.DictReader(open(p, encoding="utf-8-sig")):
            name = inv.get(row["model"])
            if name and (name, row["dataset"]) in R:
                R[(name, row["dataset"])][736] = float(row["Recall"])
    return R


# ────────────────────────────────────────────── 그림
def plot_combo(rows, model, ds, out_png, conf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    x = np.arange(len(IMGSZS))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, kind, title in zip(axes, ["cells", "det"],
                               ["High-confidence cells (pre-NMS)",
                                "Final detections (post-NMS)"]):
        bottom = np.zeros(len(IMGSZS))
        tot = [sum(rows[(r, L)][kind] for L in LEVELS) for r in IMGSZS]
        for k, L in enumerate(LEVELS):
            v = np.array([(rows[(r, L)][kind] / t * 100 if t else 0)
                          for r, t in zip(IMGSZS, tot)])
            ax.bar(x, v, bottom=bottom, color=colors[k], label=L,
                   edgecolor="white", linewidth=0.6)
            for xi, (vi, bi) in enumerate(zip(v, bottom)):
                if vi >= 6:
                    ax.text(xi, bi + vi / 2, f"{vi:.0f}", ha="center", va="center",
                            fontsize=8.5, color="white", fontweight="bold")
            bottom += v
        for xi, t in enumerate(tot):
            ax.text(xi, 101.5, f"n={t}", ha="center", va="bottom", fontsize=8, color="#444")
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in IMGSZS])
        ax.set_xlabel("input resolution (px)")
        ax.set_ylabel("share (%)")
        ax.set_ylim(0, 108)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(title="scale", loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.suptitle(f"{model}  |  {ds} set   (conf>{conf}, n = absolute count)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_overview(data, out_png, conf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    x = np.arange(len(IMGSZS))
    fig, axes = plt.subplots(len(MODELS), len(DATASETS),
                             figsize=(11, 3.0 * len(MODELS)), sharex=True)
    for mi, (mname, _) in enumerate(MODELS):
        for di, (ds, _) in enumerate(DATASETS):
            ax = axes[mi][di]
            rows = data.get((mname, ds))
            if rows is None:
                ax.axis("off")
                continue
            tot = [sum(rows[(r, L)]["det"] for L in LEVELS) for r in IMGSZS]
            bottom = np.zeros(len(IMGSZS))
            for k, L in enumerate(LEVELS):
                v = np.array([(rows[(r, L)]["det"] / t * 100 if t else 0)
                              for r, t in zip(IMGSZS, tot)])
                ax.bar(x, v, bottom=bottom, color=colors[k], label=L if mi == 0 and di == 0 else None,
                       edgecolor="white", linewidth=0.5)
                bottom += v
            for xi, t in enumerate(tot):
                ax.text(xi, 102, f"{t}", ha="center", va="bottom", fontsize=7, color="#444")
            ax.set_ylim(0, 112)
            ax.set_xticks(x)
            ax.set_xticklabels([str(r) for r in IMGSZS], fontsize=8)
            ax.set_title(f"{mname} · {ds}", fontsize=9.5)
            ax.grid(axis="y", alpha=0.2, linewidth=0.5)
            ax.set_axisbelow(True)
            if di == 0:
                ax.set_ylabel("share (%)", fontsize=8.5)
    fig.legend(loc="upper right", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(f"Final-detection scale distribution (post-NMS, conf>{conf})   "
                 f"numbers above bars = absolute detection count", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ────────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--tag", default="260820")
    ap.add_argument("--verify", action="store_true",
                    help="SR-1600 x val x imgsz736 한 조합만 검증 실행")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 검증 실행 ──
    if args.verify:
        mname, mpath = MODELS[1]                      # SR-1600
        ds, ddir = DATASETS[1]                        # val
        print("=" * 80)
        print(f"[VERIFY] {mname} x {ds} x imgsz 736   conf={args.conf} iou={args.iou}")
        print("=" * 80)
        net, head, hook, nc, nm = load_model(mpath, device)
        print(f"  head={type(head).__name__} nc={nc} nm={nm} reg_max={head.reg_max} "
              f"stride={[int(s) for s in head.stride]}")
        files = sorted(ddir.glob("*.jpg"))
        img = cv2.imread(str(files[0]))
        r32 = int(np.ceil(736 / 32) * 32)
        cells, grid, det = run_one(net, hook, img, r32, nc, nm, args.conf, args.iou, device)
        exp = sum((r32 // s) ** 2 for s in STRIDES)
        print(f"  이미지 {files[0].name}")
        print(f"  imgsz 736 -> 격자 {r32} | 레벨별 셀 {grid} 합 {sum(grid)} "
              f"| 기대 {exp} {'OK' if sum(grid) == exp else 'MISMATCH'}")
        print(f"  고신뢰 셀 : P3 {cells[0]} / P4 {cells[1]} / P5 {cells[2]}")
        print(f"  최종 검출 : P3 {det[0]} / P4 {det[1]} / P5 {det[2]}")
        hook.close()
        ok = sum(grid) == exp and all(c >= 0 for c in cells)
        print(f"\n[VERIFY] {'정상 — 전체 실행 가능' if ok else '비정상'}")
        print("=" * 80)
        return

    # ── 전체 실행 ──
    recall = load_recall()
    gts = {}
    imgs_cache = {}
    for ds, ddir in DATASETS:
        files = sorted(ddir.glob("*.jpg"))
        imgs_cache[ds] = [cv2.imread(str(f)) for f in files]
        gts[ds] = gt_count(ddir)
        print(f"[{ds}] {ddir}  이미지 {len(files)}장 | 4cm 이상 GT {gts[ds]}개")

    long_rows, summary, data = [], [], {}

    for mname, mpath in MODELS:
        if not Path(mpath).exists():
            print(f"[skip] {mname}: {mpath}")
            continue
        net, head, hook, nc, nm = load_model(mpath, device)
        for ds, _ in DATASETS:
            rows = {}
            print(f"\n=== {mname} x {ds} ===")
            print(f"{'imgsz':>6} {'grid':>6} | {'cells P3/P4/P5':>22} | "
                  f"{'det P3/P4/P5':>18} | {'det합':>6} {'P3%':>6} {'P4%':>6} {'P5%':>6}")
            for r in IMGSZS:
                r32 = int(np.ceil(r / 32) * 32)
                ct = np.zeros(3, np.int64)
                dt = np.zeros(3, np.int64)
                for im in imgs_cache[ds]:
                    c, g, d = run_one(net, hook, im, r32, nc, nm, args.conf, args.iou, device)
                    ct += np.array(c, np.int64)
                    dt += np.array(d, np.int64)
                torch.cuda.empty_cache()
                cs, dsum = int(ct.sum()), int(dt.sum())
                for k, L in enumerate(LEVELS):
                    rows[(r, L)] = dict(cells=int(ct[k]), det=int(dt[k]))
                    long_rows.append([
                        mname, ds, r, L, int(ct[k]), int(dt[k]),
                        round(ct[k] / cs, 4) if cs else 0.0,
                        round(dt[k] / dsum, 4) if dsum else 0.0])
                dp = dt / dsum * 100 if dsum else np.zeros(3)
                print(f"{r:>6} {r32:>6} | {f'{ct[0]}/{ct[1]}/{ct[2]}':>22} | "
                      f"{f'{dt[0]}/{dt[1]}/{dt[2]}':>18} | {dsum:>6} "
                      f"{dp[0]:>5.1f}% {dp[1]:>5.1f}% {dp[2]:>5.1f}%")
            data[(mname, ds)] = rows
            plot_combo(rows, mname, ds,
                       E.FIGURES / f"scale_dist_{mname.replace('[','_').replace(']','').replace('-','_')}_{ds}_{args.tag}.png",
                       args.conf)

            # 요약
            det_tot = {r: sum(rows[(r, L)]["det"] for L in LEVELS) for r in IMGSZS}
            p5 = sum(rows[(r, "P5")]["det"] for r in IMGSZS)
            p5c = sum(rows[(r, "P5")]["cells"] for r in IMGSZS)
            valid = [r for r in IMGSZS if det_tot[r] >= 20]        # 표본 빈약 구간 제외
            share = {r: (rows[(r, "P3")]["det"] / det_tot[r] if det_tot[r] else 0) for r in IMGSZS}
            best_p3 = max(valid, key=lambda r: share[r]) if valid else None
            rc = recall.get((mname, ds), {})
            best_rec = max(rc, key=rc.get) if rc else None
            base = min(valid, key=lambda r: det_tot[r]) if valid else None
            summary.append([
                mname, ds, gts[ds], p5c, p5,
                best_p3, round(share[best_p3], 4) if best_p3 else None,
                best_rec, round(rc.get(best_rec, 0), 4) if best_rec else None,
                (best_p3 == best_rec) if (best_p3 and best_rec) else None,
                det_tot[512], det_tot[736], det_tot[1600],
                round(det_tot[1600] / det_tot[512], 2) if det_tot[512] else None,
                round(det_tot[1600] / gts[ds], 2),
                round(rows[(1600, "P4")]["det"] / det_tot[1600], 4) if det_tot[1600] else None,
            ])
        hook.close()
        del net
        torch.cuda.empty_cache()

    # ── 저장 ──
    p_long = E.RESULTS / f"scale_distribution_multi_{args.tag}.csv"
    with open(p_long, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "dataset", "imgsz", "level",
                    "confident_cells", "final_detections",
                    "fraction_cells", "fraction_det"])
        w.writerows(long_rows)

    p_sum = E.RESULTS / f"scale_distribution_summary_{args.tag}.csv"
    with open(p_sum, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "dataset", "n_GT", "P5_cells_total", "P5_det_total",
                    "res_max_P3share", "max_P3share", "res_best_recall", "best_recall",
                    "match", "det_512", "det_736", "det_1600",
                    "ratio_1600_over_512", "det1600_over_GT", "P4_share_at_1600"])
        w.writerows(summary)

    plot_overview(data, E.FIGURES / f"scale_distribution_overview_{args.tag}.png", args.conf)

    # ── 요약표 출력 ──
    print("\n" + "=" * 126)
    print("=== 조합별 요약 ===")
    print("=" * 126)
    print(f"{'model':>13} {'ds':>5} {'GT':>4} | {'P5셀':>6} {'P5검출':>7} | "
          f"{'P3최대해상도':>12} {'P3비율':>7} | {'최적recall해상도':>16} {'recall':>7} {'일치':>5} | "
          f"{'det512':>7} {'det1600':>8} {'배율':>6} {'/GT':>7} {'P4@1600':>8}")
    print("-" * 126)
    for s in summary:
        print(f"{s[0]:>13} {s[1]:>5} {s[2]:>4} | {s[3]:>6} {s[4]:>7} | "
              f"{str(s[5]):>12} {s[6]:>7.3f} | {str(s[7]):>16} {s[8]:>7.3f} "
              f"{('O' if s[9] else 'X'):>5} | {s[10]:>7} {s[12]:>8} {s[13]:>6} "
              f"{s[14]:>7} {s[15]:>8.3f}")

    print(f"\n저장: {p_long}")
    print(f"저장: {p_sum}")
    print(f"저장: {E.FIGURES / f'scale_distribution_overview_{args.tag}.png'}")


if __name__ == "__main__":
    main()
