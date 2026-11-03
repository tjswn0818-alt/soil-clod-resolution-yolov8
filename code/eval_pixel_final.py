"""
실험 A+B (6x6 교차) + 실험 C 평가.

- 테스트셋: 18.v1i.coco-segmentation/train (고정)
- 각 모델(학습 R)을 eval_R = [1600,1024,736,512,256,128] 로 리사이즈한 이미지에서 평가
- GT 필터: 장축 >= 4cm (px_per_cm=44.0 -> 이미지 한 변의 8%). 원본 격자에서 1회 확정하여
  모든 셀에서 동일한 GT 집합(=동일 분모)을 쓴다.
- 매칭: IoU>=0.5, 예측 면적 내림차순 greedy (common.py::_match 와 동일)
- anchor: R1600 모델을 eval 1600 에서 평가했을 때 TP 로 잡은 GT 집합으로 고정

출력: res_sweep_pixel_final.csv      (행=학습R(+C), 열=평가R, 지표별 블록)
      res_sweep_pixel_final_long.csv (셀 단위 long format)
"""
import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent

# ── 경로 (스크립트 위치 기준 상대경로) ──────────────────────────
import config  # noqa: E402  code/config.py (경로 설정)
RS_ROOT = config.RS_ROOT        # 작업 루트 (weights/, data/, results/ ...)
PIXEL_ROOT = config.PIXEL_ROOT  # 데이터셋 루트 (18.v1i.coco-segmentation/, pixel_split/)
WEIGHTS = RS_ROOT / "weights"
RESULTS = RS_ROOT / "results"
FIGURES = RS_ROOT / "figures"
LOGS_DIR = RS_ROOT / "logs"
RUNS = RS_ROOT / "runs"
for _d in (RESULTS, FIGURES, LOGS_DIR, RUNS):
    _d.mkdir(parents=True, exist_ok=True)


def sr_ckpt(r):
    """단일 해상도(SR) 모델 가중치."""
    return WEIGHTS / "sr" / f"R{r}.pt"


def mr_ckpt(tag):
    """다해상도 혼합(MR) 모델 가중치. tag in {C, C2, C3}."""
    return WEIGHTS / "mr" / f"{tag}.pt"
TEST_DIR = PIXEL_ROOT / "18.v1i.coco-segmentation" / "train"
AB_ROOT = WEIGHTS / "sr"
C_CKPT  = mr_ckpt("C")

RES_LADDER = [1600, 1024, 736, 512, 256, 128]
EVAL_LADDER = list(RES_LADDER)
ANCHOR_TRAIN, ANCHOR_EVAL = 1600, 1600

PX_PER_CM      = 44.0
MIN_MAJOR_PX   = 176.0
REF_SIDE       = PX_PER_CM * 50.0        # 2200
MIN_MAJOR_FRAC = MIN_MAJOR_PX / REF_SIDE  # 0.08
IOU_THR        = 0.5
SIZE_BINS      = ("Small", "Medium", "Large")

METRICS = ["Recall", "Precision", "F1", "mIoU",
           "R_Small", "R_Medium", "R_Large", "BIoU_anchor", "Retain", "FPS"]


def size_bin(a):
    return "Small" if a <= 16 else ("Medium" if a <= 25 else "Large")


def seg_to_mask(seg, h, w):
    if isinstance(seg, list):
        rle = coco_mask.merge(coco_mask.frPyObjects(seg, h, w))
    elif isinstance(seg.get("counts"), list):
        rle = coco_mask.frPyObjects(seg, h, w)
    else:
        rle = seg
    return coco_mask.decode(rle).astype(np.uint8)


def major_axis_px(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    if len(c) >= 5:
        try:
            return float(max(cv2.fitEllipse(c)[1]))
        except Exception:
            pass
    r = cv2.minAreaRect(c)
    return float(max(r[1])) if r[1][0] > 0 and r[1][1] > 0 else 0.0


def boundary_iou(gt, pred, ratio):
    a, b = gt.astype(np.uint8), pred.astype(np.uint8)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    d = max(1, int(round(ratio * np.sqrt(float(a.sum())))))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * d + 1, 2 * d + 1))
    ab = a - cv2.erode(a, k)
    bb = b - cv2.erode(b, k)
    u = np.logical_or(ab, bb).sum()
    return float(np.logical_and(ab, bb).sum() / u) if u else 0.0


def match(preds, gts):
    n = len(gts)
    taken = [False] * n
    pairs = {}
    if not preds or n == 0:
        return taken, pairs, 0
    ious = coco_mask.iou([coco_mask.encode(np.asfortranarray(m)) for m in preds],
                         [coco_mask.encode(np.asfortranarray(m)) for m in gts], [0] * n)
    areas = [int(m.sum()) for m in preds]
    tp = 0
    for di in sorted(range(len(preds)), key=lambda i: areas[i], reverse=True):
        gi, best = -1, IOU_THR
        for g in range(n):
            if not taken[g] and ious[di, g] >= best:
                best, gi = ious[di, g], g
        if gi >= 0:
            taken[gi] = True
            pairs[gi] = (di, float(ious[di, gi]))
            tp += 1
    return taken, pairs, tp


def load_gt():
    """원본 격자에서 4cm 필터 + 크기구간 확정 (모든 eval_R 에서 동일 GT 집합)."""
    coco = json.load(open(TEST_DIR / "_annotations.coco.json", encoding="utf-8"))
    ann_by = {}
    for a in coco["annotations"]:
        ann_by.setdefault(a["image_id"], []).append(a)
    out, n_raw, n_kept = {}, 0, 0
    for im in coco["images"]:
        H, W = im["height"], im["width"]
        side = max(H, W)
        thr = MIN_MAJOR_FRAC * side
        cm_per_px = 50.0 / side
        masks, bins_ = [], []
        for a in ann_by.get(im["id"], []):
            m = seg_to_mask(a["segmentation"], H, W).astype(np.uint8)
            n_raw += 1
            if m.sum() == 0 or major_axis_px(m) < thr:
                continue
            masks.append(m)
            bins_.append(size_bin(m.sum() * cm_per_px ** 2))
            n_kept += 1
        out[im["file_name"]] = dict(masks=masks, bins=bins_)
    return out, n_raw, n_kept


def pure_forward_fps(model, imgs_resized, res, warmup=2, reps=3):
    net = model.model.to("cuda").eval()
    ts = [torch.from_numpy(i[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to("cuda")
          for i in imgs_resized]
    with torch.inference_mode():
        for _ in range(warmup):
            net(ts[0])
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            for t in ts:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                net(t)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
    del ts
    torch.cuda.empty_cache()
    ms = float(np.mean(times)) * 1000
    return 1000.0 / ms, ms


def eval_cell(model, files, images, gt_by_file, eval_r, conf, biou_ratio, anchor):
    """한 셀(모델 x eval_r) 평가. anchor=None 이면 anchor 자체를 만드는 셀."""
    r32 = int(np.ceil(eval_r / 32) * 32)
    # 임계는 마스크가 실제로 놓이는 격자(r32) 기준이어야 정확히 4.000cm 가 된다.
    # eval_r 기준으로 계산하면 32 배수가 아닌 해상도에서 임계가 완화된다.
    thr = MIN_MAJOR_FRAC * r32
    TP = FP = FN = 0
    ious_all, biou_vals = [], []
    size_tot = {b: 0 for b in SIZE_BINS}
    size_det = {b: 0 for b in SIZE_BINS}
    hits = {}
    retain_hit = retain_tot = 0
    resized_imgs = []

    for f in files:
        g = gt_by_file[f]
        img_r = cv2.resize(images[f], (r32, r32), interpolation=cv2.INTER_AREA)
        resized_imgs.append(img_r)
        gts = [cv2.resize(m, (r32, r32), interpolation=cv2.INTER_NEAREST) for m in g["masks"]]

        res = model.predict(img_r, imgsz=r32, conf=conf, retina_masks=True,
                            verbose=False, device=0)[0]
        preds = []
        if res.masks is not None:
            for pm in res.masks.data.cpu().numpy():
                m = (pm > 0.5).astype(np.uint8)
                if m.shape != (r32, r32):
                    m = cv2.resize(m, (r32, r32), interpolation=cv2.INTER_NEAREST)
                if m.sum() and major_axis_px(m) >= thr:
                    preds.append(m)

        flags, pairs, tp = match(preds, gts)
        TP += tp
        FP += len(preds) - tp
        FN += len(gts) - tp
        ious_all += [i for (_, i) in pairs.values()]
        for b, fl in zip(g["bins"], flags):
            size_tot[b] += 1
            if fl:
                size_det[b] += 1
        hits[f] = {gi for gi in pairs}

        ref = hits[f] if anchor is None else anchor.get(f, set())
        for gi in ref:
            retain_tot += 1
            if gi in pairs:
                retain_hit += 1
                di, _ = pairs[gi]
                biou_vals.append(boundary_iou(gts[gi], preds[di], biou_ratio))

    fps, ms = pure_forward_fps(model, resized_imgs, r32)
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    return dict(
        TP=TP, FP=FP, FN=FN,
        Recall=R, Precision=P,
        F1=(2 * P * R / (P + R) if P + R else 0.0),
        mIoU=float(np.mean(ious_all)) if ious_all else 0.0,
        **{f"R_{b}": (size_det[b] / size_tot[b] if size_tot[b] else float("nan")) for b in SIZE_BINS},
        BIoU_anchor=float(np.mean(biou_vals)) if biou_vals else float("nan"),
        Retain=(retain_hit / retain_tot if retain_tot else float("nan")),
        FPS=fps, ms=ms, N_anchor=retain_tot,
    ), hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--biou-ratio", type=float, default=0.10)
    ap.add_argument("--out", default=str(RESULTS / "res_sweep_pixel_final.csv"))
    ap.add_argument("--skip-c", action="store_true")
    args = ap.parse_args()

    gt_by_file, n_raw, n_kept = load_gt()
    files = sorted(gt_by_file)
    images = {f: cv2.imread(str(TEST_DIR / f)) for f in files}
    if any(v is None for v in images.values()):
        raise SystemExit("이미지 로드 실패")

    print(f"테스트셋: {TEST_DIR}")
    print(f"  이미지 {len(files)}장 | GT {n_raw} -> 장축>=4cm {n_kept}개 (모든 셀 공통)")
    print(f"  conf={args.conf} IoU={IOU_THR} | anchor = R{ANCHOR_TRAIN} @ eval{ANCHOR_EVAL}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}\n")

    # 평가 대상: (라벨, 체크포인트)
    targets = []
    for r in RES_LADDER:
        ck = sr_ckpt(r)
        if ck.exists():
            targets.append((str(r), ck))
        else:
            print(f"[skip] 없음: {ck}")
    if not args.skip_c:
        for lbl, ck in [("C", C_CKPT),
                        ("C2", mr_ckpt("C2")),
                        ("C3", mr_ckpt("C3"))]:
            if ck.exists():
                targets.append((lbl, ck))
            else:
                print(f"[skip] 실험 {lbl} 가중치 없음: {ck}")

    anchor = None
    cells = {}          # (train_label, eval_r) -> metrics dict

    for label, ck in targets:
        model = YOLO(str(ck))
        for er in EVAL_LADDER:
            is_anchor_cell = (label == str(ANCHOR_TRAIN) and er == ANCHOR_EVAL and anchor is None)
            m, hits = eval_cell(model, files, images, gt_by_file, er,
                                args.conf, args.biou_ratio,
                                None if is_anchor_cell else anchor)
            if is_anchor_cell:
                anchor = hits
                n_anchor = sum(len(v) for v in anchor.values())
                print(f"  [anchor 확정] R{ANCHOR_TRAIN} @ eval{ANCHOR_EVAL} -> GT {n_anchor}개")
            cells[(label, er)] = m
            print(f"  train {label:>4} x eval {er:>4} | TP{m['TP']:4d} FP{m['FP']:4d} FN{m['FN']:4d} "
                  f"| R {m['Recall']:.3f} P {m['Precision']:.3f} F1 {m['F1']:.3f} "
                  f"mIoU {m['mIoU']:.3f} | FPS {m['FPS']:6.1f} "
                  f"| BIoU {m['BIoU_anchor']:.3f} Retain {m['Retain']:.3f}")
        del model
        torch.cuda.empty_cache()
        print()

    row_labels = [lbl for lbl, _ in targets]

    # ── 행렬 CSV ──
    with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "train_res"] + [f"eval_{e}" for e in EVAL_LADDER])
        for met in METRICS:
            for lbl in row_labels:
                w.writerow([met, lbl] + [round(cells[(lbl, e)][met], 4) for e in EVAL_LADDER])

    # ── long CSV ──
    long_path = str(Path(args.out).with_name(Path(args.out).stem + "_long.csv"))
    with open(long_path, "w", newline="", encoding="utf-8-sig") as fh:
        keys = ["TP", "FP", "FN"] + METRICS + ["ms", "N_anchor"]
        w = csv.writer(fh)
        w.writerow(["train_res", "eval_res"] + keys)
        for lbl in row_labels:
            for e in EVAL_LADDER:
                c = cells[(lbl, e)]
                w.writerow([lbl, e] + [round(c[k], 4) if isinstance(c[k], float) else c[k] for k in keys])

    # ── 콘솔 행렬 출력 ──
    for met in METRICS:
        print(f"\n=== {met} (행=학습R, 열=평가R) ===")
        print(f"{'train\\eval':>11}" + "".join(f"{e:>9}" for e in EVAL_LADDER))
        for lbl in row_labels:
            print(f"{lbl:>11}" + "".join(f"{cells[(lbl, e)][met]:>9.3f}" for e in EVAL_LADDER))

    print(f"\nCSV: {args.out}")
    print(f"CSV(long): {long_path}")


if __name__ == "__main__":
    main()
