"""
리뷰 R1 (261116) 0단계 — 위치 확인 + 기준값 재현. 기존 파일은 읽기만 한다.

(A) 인벤토리
  - SR 72 / MR 48 가중치 사본(weights/...)을 sha256 으로 run 폴더의 best.pt 와 대조
  - run 폴더별 best.pt / last.pt / results.csv / args.yaml / epoch*.pt 존재
  - 저-conf 예측 캐시(results_review_261026/preds) 셀 수
(B) 기준값 재현 — 기존 코드 import
  - review_analysis_261026: load_all(F.match 원본 대조 포함), match_fast, ap101, TS
  - t = 0.5 (strict, 논문 규약) Recall / Precision / FP per image, 8 seed 평균
  - AP50, t = 0.01 recall
  - coarse-clod 면적비 MAE(%p): 저장 예측 RLE(score > 0.5, 4cm 통과) -> 원본 격자 NEAREST 복원
    -> 면적 합 / 프레임, GT 는 F.load_gt 마스크. seed 별 MAE 후 평균.
"""
import csv
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402  code/config.py (경로 설정)
import review_analysis_261026 as R          # noqa: E402
import review_common_261026 as C            # noqa: E402

U, F, S, M = R.U, R.F, R.U.S, R.U.M
import cv2                                   # noqa: E402
from pycocotools import mask as coco_mask    # noqa: E402

OUT = config.R1_DIR
OUT.mkdir(parents=True, exist_ok=True)
RUN_ROOTS = config.RUN_ROOTS   # 학습 run 폴더 (best.pt / last.pt / results.csv)


def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(n), b""):
            h.update(b)
    return h.hexdigest()


def wcsv(name, hdr, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)


def inventory():
    best = {}
    for root in RUN_ROOTS:
        for b in root.rglob("best.pt"):
            if b.parent.name == "weights":
                best.setdefault(sha(b), []).append(b.parent.parent)
    rows = []
    specs = [("SR", tr, sd, S.ckpt_of(tr, sd)) for tr in U.TRAIN_RES for sd in U.SEEDS]
    specs += [("MR", lbl, sd, M.ckpt_of(tag, sd)) for lbl, tag in U.MR_SPECS
              for sd in U.SEEDS]
    for fam, key, sd, ck in specs:
        h = sha(ck) if ck.exists() else ""
        runs = best.get(h, [])
        rd = runs[0] if runs else None
        ep = sorted(p.name for p in rd.glob("weights/epoch*.pt")) if rd else []
        rows.append([fam, key, sd, config.rel(ck), int(ck.exists()), h[:12],
                     config.rel(rd) if rd else "", len(runs),
                     int(bool(rd) and (rd / "weights/last.pt").exists()),
                     int(bool(rd) and (rd / "results.csv").exists()),
                     int(bool(rd) and (rd / "args.yaml").exists()),
                     ";".join(ep)])
    wcsv("s0_inventory_runs.csv",
         ["family", "model", "seed", "ckpt_copy", "ckpt_exists", "sha256_head",
          "run_dir", "n_run_matches", "last_pt", "results_csv", "args_yaml",
          "epoch_pts"], rows)
    preds = sorted((C.PRED).glob("*.pkl"))
    n_ok = sum(r[7] >= 1 for r in rows)
    print(f"[A] 가중치 {len(rows)}개 | run 폴더 매칭 {n_ok} | "
          f"last.pt {sum(r[8] for r in rows)} | results.csv {sum(r[9] for r in rows)} | "
          f"epoch*.pt 보유 run {sum(bool(r[11]) for r in rows)} | preds pkl {len(preds)}")
    for r in rows:
        if r[7] != 1:
            print("   매칭 이상:", r[:3], r[7])
    return rows


def area_mae(ds, tr, sd, gt):
    ims = C.load_cell(ds, tr, sd)["images"]
    d = []
    for f in sorted(ims):
        g, im = gt[f], ims[f]
        cpp, H, W = g["cpp"], g["H"], g["W"]
        ga = sum(float(m.sum()) for m in g["masks"]) * cpp ** 2
        pa = 0.0
        for i in im["kept_idx"]:
            p = im["preds"][i]
            if p["score"] > 0.5:
                m = coco_mask.decode(p["rle"])
                pa += float(cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).sum())
        pa *= cpp ** 2
        d.append(100 * (pa - ga) / F.FRAME_CM ** 2)
    d = np.array(d)
    return float(np.abs(d).mean()), float(d.mean())


def repro():
    A, _ = R.load_all()
    print("  F.match 원본 대조 통과")
    gts = {ds: F.load_gt(F.DATASETS[ds]) for ds in U.DATASETS}
    rows = []
    for ds in U.DATASETS:
        nimg = len(gts[ds])
        for tr in R.RESL:
            v = []
            for sd in R.SEEDS:
                imgs = A[(ds, tr, sd)]
                TP = FP = FN = 0
                for P in imgs.values():
                    tp, fp, fn, _ = R.match_fast(P, 0.5, strict=True)
                    TP, FP, FN = TP + tp, FP + fp, FN + fn
                Rr, Pp = [], []
                r01 = None
                for t in R.TS:
                    a = np.array([R.match_fast(P, t)[:3] for P in imgs.values()]).sum(0)
                    if abs(t - 0.01) < 1e-9:
                        r01 = a[0] / (a[0] + a[2])
                    if a[0] + a[1]:
                        Rr.append(a[0] / (a[0] + a[2]))
                        Pp.append(a[0] / (a[0] + a[1]))
                mae, bias = area_mae(ds, tr, sd, gts[ds])
                v.append([TP / (TP + FN), TP / (TP + FP) if TP + FP else 0.0,
                          FP / nimg, FN / nimg, R.ap101(Rr, Pp), r01, mae, bias,
                          TP + FN])
            v = np.array(v)
            rows.append([ds, tr, R.LBL[tr]] + [round(x, 4) for x in v.mean(0)])
            print(f"  {ds} {R.LBL[tr]:>5} R {v[:,0].mean():.4f} P {v[:,1].mean():.4f} "
                  f"FP/img {v[:,2].mean():.3f} AP50 {v[:,4].mean():.4f} "
                  f"R@0.01 {v[:,5].mean():.4f} MAE {v[:,6].mean():.3f} nGT {v[0,8]:.0f}",
                  flush=True)
    wcsv("s0_repro_baseline.csv",
         ["dataset", "train_res", "eval_res", "Recall_t05", "Precision_t05",
          "FP_per_img", "FN_per_img", "AP50", "Recall_t001", "areaRatio_MAE_pp",
          "areaRatio_bias_pp", "n_GT"], rows)
    ref = {("Recall_t05", 448): .895, ("Recall_t05", 512): .910,
           ("Recall_t05", 736): .891, ("Recall_t05", 1024): .888,
           ("Recall_t05", 1280): .843, ("Recall_t05", 1600): .809,
           ("Precision_t05", 512): .630, ("Precision_t05", 1600): .781,
           ("FP_per_img", 512): 3.06, ("FP_per_img", 1600): 1.31,
           ("AP50", 512): .771, ("AP50", 1600): .815,
           ("Recall_t001", 512): .964, ("Recall_t001", 1600): .943,
           ("areaRatio_MAE_pp", 512): 1.46, ("areaRatio_MAE_pp", 1600): 0.66}
    cols = ["Recall_t05", "Precision_t05", "FP_per_img", "FN_per_img", "AP50",
            "Recall_t001", "areaRatio_MAE_pp"]
    chk, bad = [], 0
    for (k, tr), want in ref.items():
        got = [r for r in rows if r[0] == "test" and r[1] == tr][0][3 + cols.index(k)]
        dec = 2 if k in ("FP_per_img", "areaRatio_MAE_pp") else 3
        ok = round(got, dec) == want
        bad += not ok
        chk.append([k, tr, want, got, int(ok)])
    wcsv("s0_repro_check.csv", ["metric", "res", "paper", "recomputed", "match"], chk)
    print(f"[B] 기준값 {len(chk)}개 중 불일치 {bad}")
    for c in chk:
        print("   ", c)


if __name__ == "__main__":
    inventory()
    repro()
