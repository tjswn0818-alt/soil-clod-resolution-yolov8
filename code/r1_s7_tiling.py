"""7단계 — 분리·슬라이싱 추론 (3x3 타일, 50% 중첩), P0·P1.

타일: 원본 영상 W x H 에서 타일 크기 (round(W/2), round(H/2)) = 프레임 한 변의 절반(25 cm),
      좌상단 x = 0, (W - tw)/2, W - tw (y 동일) -> 3x3, 인접 타일 50% 중첩.
      타일을 s x s 로 INTER_LINEAR 리사이즈해 추론 (평가 파이프라인의 정사각 리사이즈와 같은 방식).
GT 배정: 마스크 bbox 전체가 들어가는 타일 중 타일 중심과 GT 마스크 무게중심이 가장 가까운 타일 1개.
         어느 타일에도 완전히 들어가지 않는 GT 는 '미배정'으로 수를 기록하고 Recall 분모에서 제외.
크기 기준: GT >= 4cm 는 원본 격자 판정(F.load_gt 와 동일). 예측 4cm = 타일 격자 장축
           >= 0.16 * s (25 cm 타일에서 4 cm), 프레임 기준선은 0.08 * r32 (기존과 동일).
추론: review_infer_261026 와 같은 predict 설정 (conf 0.001, iou 0.7, max_det 1000, retina,
      NMS 인덱스 -> P3/P4/P5).
매칭(타일 단위, 기존 greedy):
  P0: score > t, 4cm 통과 예측  vs  그 타일에 배정된 GT(>=4cm)
  P1: score > t, 모든 예측      vs  그 타일과 겹치는 모든 GT(크기 무관, 타일로 잘라 리사이즈);
      Recall = 매칭된 배정 GT(>=4cm) / 배정 GT(>=4cm)
기준선 'full': 같은 코드로 타일 1개 = 전체 프레임, s = r32 -> 논문 Recall 재현 확인.
"""
import pickle
import sys
import time

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask

import r1_common as Q

R, U, C, F = Q.R, Q.U, Q.C, Q.F
S = U.S
import review_infer_261026 as RI          # noqa: E402  패치 적용
from ultralytics import YOLO               # noqa: E402

TC = Q.CACHE / "tiles"
TC.mkdir(parents=True, exist_ok=True)
# (조건명, 학습 해상도, 추론 크기, mode)
CONDS = ([("full", tr, C.GRID[tr], "full") for tr in (1600, 1280, 1024, 512, 448, 256, 128)]
         + [("A_halfinput", 1600, 800, "tile"), ("A_halfinput", 1280, 640, "tile"),
            ("A_halfinput", 1024, 512, "tile"), ("A_halfinput", 512, 256, "tile"),
            ("A_halfinput", 448, 224, "tile"),
            ("B_sameinput", 512, 512, "tile"), ("B_sameinput", 256, 256, "tile"),
            ("B_sameinput", 128, 128, "tile")])
CROSS_REF = {512: 1024, 256: 512, 128: 256}   # B 조건과 대응하는 교차평가 (학습, 추론)


def tiles_of(H, W, mode):
    if mode == "full":
        return [(0, 0, W, H)]
    tw, th = int(round(W / 2)), int(round(H / 2))
    xs = [0, int(round((W - tw) / 2)), W - tw]
    ys = [0, int(round((H - th) / 2)), H - th]
    return [(x, y, tw, th) for y in ys for x in xs]


def assign(g, T):
    """GT 마다 배정 타일 인덱스 (-1 = 미배정)."""
    out = []
    for m in g["masks"]:
        ys, xs = np.nonzero(m)
        cy, cx = ys.mean(), xs.mean()
        best, bi = 1e18, -1
        for k, (x, y, w, h) in enumerate(T):
            if xs.min() >= x and xs.max() < x + w and ys.min() >= y and ys.max() < y + h:
                d = (cx - (x + w / 2)) ** 2 + (cy - (y + h / 2)) ** 2
                if d < best:
                    best, bi = d, k
        out.append(bi)
    return np.array(out, int)


def infer_img(model, img_s, s):
    RI._STASH.clear()
    r = model.predict(img_s, imgsz=s, conf=RI.CONF_LOW, iou=U.IOU_NMS,
                      max_det=RI.MAX_DET, retina_masks=True, verbose=False, device=0)[0]
    recs = []
    if r.masks is not None:
        idx = RI._STASH["idx"][0].numpy()
        if "keep" in RI._STASH and not bool(RI._STASH["keep"].all()):
            idx = idx[RI._STASH["keep"].numpy()]
        lv = RI.level_of(idx, s)
        sc = r.boxes.conf.cpu().numpy().astype(np.float32)
        md = r.masks.data
        for c0 in range(0, md.shape[0], RI.CHUNK):
            blk = md[c0:c0 + RI.CHUNK].cpu().numpy()
            for j, pm in enumerate(blk):
                m = (pm > 0.5).astype(np.uint8)
                if m.shape != (s, s):
                    m = cv2.resize(m, (s, s), interpolation=U.INTERP_MASK)
                a = int(m.sum())
                if a == 0:
                    continue
                k = c0 + j
                recs.append(dict(score=float(sc[k]), level=int(lv[k]),
                                 major_px=F.major_axis_px(m), area=a,
                                 rle=coco_mask.encode(np.asfortranarray(m))))
    return recs


def run_cell(model, ds, img, G, s, mode):
    """이미지별 -> 타일별 예측 + 매칭 재료."""
    out = {}
    for f in sorted(G):
        g = G[f]
        H, W = g["H"], g["W"]
        T = tiles_of(H, W, mode)
        asg = assign(g, T)
        tl = []
        for k, (x, y, w, h) in enumerate(T):
            crop = img[f][y:y + h, x:x + w]
            recs = infer_img(model, cv2.resize(crop, (s, s), interpolation=U.INTERP_IMG), s)
            gidx, grle = [], []
            for gi, m in enumerate(g["masks"]):
                c = m[y:y + h, x:x + w]
                if c.any():
                    gidx.append(gi)
                    grle.append(coco_mask.encode(np.asfortranarray(
                        cv2.resize(c, (s, s), interpolation=U.INTERP_MASK))))
            n = len(grle)
            if recs and n:
                iou = np.asarray(coco_mask.iou([q["rle"] for q in recs], grle, [0] * n),
                                 np.float64).reshape(len(recs), n)
            else:
                iou = np.zeros((len(recs), n))
            for q in recs:
                del q["rle"]
            tl.append(dict(gidx=np.array(gidx, int), iou=iou,
                           score=np.array([q["score"] for q in recs]),
                           area=np.array([q["area"] for q in recs], float),
                           major=np.array([q["major_px"] for q in recs]),
                           level=np.array([q["level"] for q in recs], int)))
        out[f] = dict(assign=asg, big=g["big"].copy(), tiles=tl, s=s, mode=mode)
    return out


def eval_img(D, t, proto, strict=True):
    """이미지 1장 -> (tp, n_assigned_big, matched (score, level) list, n_unassigned_big)."""
    s, big, asg = D["s"], D["big"], D["assign"]
    thr = (0.08 if D["mode"] == "full" else 0.16) * s
    tp, ms = 0, []
    for k, T in enumerate(D["tiles"]):
        above = T["score"] > t if strict else T["score"] >= t
        if proto == "P0":
            cols_g = [j for j, gi in enumerate(T["gidx"]) if big[gi] and asg[gi] == k]
            sel = above & (T["major"] >= thr)
        else:
            cols_g = list(range(len(T["gidx"])))
            sel = above
        if not cols_g:
            continue
        p2g = Q.match_sel(T["iou"], T["area"], sel, np.array(cols_g, int))
        for p in np.where(p2g >= 0)[0]:
            gi = T["gidx"][p2g[p]]
            if big[gi] and asg[gi] == k:
                tp += 1
                ms.append((T["score"][p], T["level"][p]))
    nb = int((big & (asg >= 0)).sum())
    return tp, nb, ms, int((big & (asg < 0)).sum())


def main():
    D = U.load_data()
    G = {ds: Q.gt_all(ds) for ds in U.DATASETS}
    for cond, tr, s, mode in CONDS:
        for sd in Q.SEEDS:
            todo = [ds for ds in U.DATASETS
                    if not (TC / f"{cond}_{ds}_{tr}_{s}_s{sd}.pkl").exists()]
            if not todo:
                continue
            model = YOLO(str(S.ckpt_of(tr, sd)))
            t0 = time.time()
            for ds in todo:
                try:
                    cell = run_cell(model, ds, D[ds]["img"], G[ds], s, mode)
                except Exception as e:                       # 실패 조건 기록 후 계속
                    print(f"  [FAIL] {cond} {ds} R{tr}@{s} s{sd}: {e!r}", flush=True)
                    continue
                p = TC / f"{cond}_{ds}_{tr}_{s}_s{sd}.pkl"
                with open(str(p) + ".tmp", "wb") as fh:
                    pickle.dump(cell, fh)
                (TC / (p.name + ".tmp")).replace(p)
            del model
            torch.cuda.empty_cache()
            print(f"  {cond} R{tr}@{s} s{sd} {time.time() - t0:.1f}s", flush=True)
    report()


def report():
    cells = {}
    for cond, tr, s, mode in CONDS:
        for ds in U.DATASETS:
            for sd in Q.SEEDS:
                p = TC / f"{cond}_{ds}_{tr}_{s}_s{sd}.pkl"
                if p.exists():
                    with open(p, "rb") as fh:
                        cells[(cond, ds, tr, s, sd)] = pickle.load(fh)
    seed_rows, sum_rows, conf_rows, lvl_rows = [], [], [], []
    TPd = {}
    for (cond, ds, tr, s, sd), cell in sorted(cells.items()):
        files = sorted(cell)
        for proto in ("P0", "P1"):
            ev = [eval_img(cell[f], 0.5, proto) for f in files]
            ev01 = [eval_img(cell[f], 0.01, proto, strict=False) for f in files]
            tp, nb = sum(e[0] for e in ev), sum(e[1] for e in ev)
            tp01 = sum(e[0] for e in ev01)
            TPd[(cond, ds, tr, s, proto, sd)] = (np.array([e[0] for e in ev]),
                                                np.array([e[1] for e in ev]))
            ms = [m for e in ev01 for m in e[2]]
            sc = np.array([m[0] for m in ms])
            lv = np.array([m[1] for m in ms])
            lv5 = np.array([m[1] for e in ev for m in e[2]])
            seed_rows.append([proto, cond, ds, tr, s, sd, nb, sum(e[3] for e in ev), tp,
                              round(tp / nb, 5), round(tp01 / nb, 5),
                              round(float(np.median(sc)), 4) if len(sc) else "",
                              round(float((sc > 0.5).mean()), 4) if len(sc) else ""]
                             + [round(float((lv5 == L).mean() * 100), 2) if len(lv5) else ""
                                for L in (3, 4, 5)])
            for x, L in zip(sc, lv):
                conf_rows.append([proto, cond, ds, tr, s, sd, round(float(x), 5), f"P{L}"])
    Q.wcsv("s7_tiling_seed.csv",
           ["protocol", "condition", "dataset", "train_res", "infer_size", "seed",
            "n_GT4_assigned", "n_GT4_unassigned", "TP", "Recall_t05", "Recall_t001",
            "matched_conf_median_t001", "matched_frac_conf_gt05_t001",
            "P3_pct_matched_t05", "P4_pct_matched_t05", "P5_pct_matched_t05"], seed_rows)
    Q.wcsv("s7_tiling_matched_conf_t001.csv",
           ["protocol", "condition", "dataset", "train_res", "infer_size", "seed",
            "score", "level"], conf_rows)
    keys = sorted({(r[0], r[1], r[2], r[3], r[4]) for r in seed_rows},
                  key=lambda k: (k[0], k[2], k[1], -k[3], -k[4]))
    for k in keys:
        sel = [r for r in seed_rows if tuple(r[:5]) == k]
        o = list(k) + [len(sel), sel[0][6], sel[0][7]]
        for j in (9, 10, 11, 12, 13, 14, 15):
            v = np.array([r[j] for r in sel if r[j] != ""], float)
            o += [round(v.mean(), 4) if len(v) else "",
                  round(v.std(ddof=1), 4) if len(v) > 1 else ""]
        sum_rows.append(o)

    # 교차 부트스트랩: 조건별 Recall CI, 타일 - 기준(full 같은 모델) 차이 CI
    ci_rows = []
    for proto in ("P0", "P1"):
        for ds in U.DATASETS:
            ks = sorted({(c, tr, s) for (c, d, tr, s, p, sd) in TPd if d == ds and p == proto})
            if not ks:
                continue
            nI = len(TPd[(ks[0][0], ds, ks[0][1], ks[0][2], proto, 0)][0])
            rng = np.random.default_rng(R.RNG_SEED)
            II = rng.integers(0, nI, (R.B, nI))
            SS = rng.integers(0, len(Q.SEEDS), (R.B, len(Q.SEEDS)))
            boot = {}
            for kk in ks:
                tp = np.stack([TPd[(kk[0], ds, kk[1], kk[2], proto, sd)][0] for sd in Q.SEEDS])
                nb = TPd[(kk[0], ds, kk[1], kk[2], proto, 0)][1]
                rs = tp[:, II].sum(-1) / nb[II].sum(1)            # (seed, B)
                boot[kk] = rs[SS.T, np.arange(R.B)].mean(0)
                lo, hi = np.percentile(boot[kk], [2.5, 97.5])
                ci_rows.append([proto, ds, kk[0], kk[1], kk[2], "",
                                round(float((tp.sum(1) / nb.sum()).mean()), 4),
                                round(lo, 4), round(hi, 4)])
            for kk in ks:
                if kk[0] == "full":
                    continue
                base = ("full", kk[1], C.GRID[kk[1]])
                if base in boot:
                    d = boot[kk] - boot[base]
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    pt = ([r for r in ci_rows if r[:5] == [proto, ds, *kk]][0][6]
                          - [r for r in ci_rows if r[:5] == [proto, ds, *base]][0][6])
                    ci_rows.append([proto, ds, kk[0], kk[1], kk[2], f"minus_full@{base[2]}",
                                    round(pt, 4), round(lo, 4), round(hi, 4)])
    Q.wcsv("s7_tiling_recall_ci.csv",
           ["protocol", "dataset", "condition", "train_res", "infer_size", "contrast",
            "Recall_8seed_mean_or_delta", "CI_2stage_lo", "CI_2stage_hi"], ci_rows)

    # 교차평가 대응 셀 (기존 unified cross 행렬, P0 기준)
    cross = []
    for ds in U.DATASETS:
        rows = {r["model"]: r for r in U.load_rows(
            U.OUT / f"{U.PFX}_cross_recall_{ds}_{U.STAMP}.csv")}
        for tr, ev in CROSS_REF.items():
            cross.append([ds, tr, ev, rows[U.SR_LABEL[tr]][f"eval_{ev}"]])
    Q.wcsv("s7_cross_eval_reference_P0.csv",
           ["dataset", "train_res", "eval_res", "Recall_8seed_mean"], cross)
    Q.wcsv("s7_tiling_summary.csv",
           ["protocol", "condition", "dataset", "train_res", "infer_size", "n_seeds",
            "n_GT4_assigned", "n_GT4_unassigned"]
           + [f"{k}_{m}" for k in ("Recall_t05", "Recall_t001", "matched_conf_median_t001",
                                   "matched_frac_conf_gt05_t001", "P3_pct", "P4_pct",
                                   "P5_pct") for m in ("mean", "sd")], sum_rows)
    for r in sum_rows:
        print(r[:8], "R", r[8], "R01", r[10], "conf_med", r[12], "P3/4/5", r[16], r[18], r[20])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report()
    else:
        main()
