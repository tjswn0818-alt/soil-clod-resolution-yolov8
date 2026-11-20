"""8단계 (추가 요청) — 재학습·재추론 없음. 캐시 예측만 사용.

1) 교차 부트스트랩 (seed x image, B = 10,000, 2단계 percentile 95% CI)
   - review_analysis_261026.step3 와 같은 순서: rng(2026) -> II(B, nI) -> SS(nR, B, nS),
     데이터셋(test, val) 별로 rng 를 새로 시작.
   (a) P1 AP50, P1 recall@0.01 (score >= t): 512-1600, 736-1600, 1024-1600
   (b) last.pt P0 recall t 0.5 (score > t): 같은 쌍. best.pt 도 함께 계산(검산).
   (c) 7단계 조건 A/B - 전체 프레임 (P0, P1): s7_tiling.report 와 같은 방식
       (같은 모델이므로 seed 재표집을 조건 간 공유) + P(Δ>0).
2) P1: >=4cm 예측이 <4cm GT 와 매칭된 수 해상도별 정리.
3) train GT 장축 > 70.7 cm 인스턴스 보고 + 오버레이 (라벨 수정 없음).
"""
import json
import pickle

import cv2
import numpy as np

import r1_common as Q

R, U, C, F = Q.R, Q.U, Q.C, Q.F
PAIRS = [(512, 1600), (736, 1600), (1024, 1600)]
NR, NS, B = len(Q.RESL), len(Q.SEEDS), R.B
ci = lambda x: np.percentile(x, [2.5, 97.5])


def boot_idx(nI):
    rng = np.random.default_rng(R.RNG_SEED)
    II = rng.integers(0, nI, (B, nI))
    SS = rng.integers(0, NS, (NR, B, NS))
    cnt = np.zeros((B, nI))
    np.add.at(cnt, (np.arange(B)[:, None], II), 1)
    return cnt, SS


def two_stage(per_seed, a, SS):
    """per_seed: (nS, B) -> (B,) seed 재표집 평균 (해상도 a 의 SS 사용)."""
    return per_seed[SS[a].T, np.arange(B)].mean(0)


def ap_vec(Rm, Pm, valid):
    """R.ap101 의 벡터판. Rm, Pm, valid: (n, nT)."""
    Rm = np.where(valid, Rm, np.inf)
    Pm = np.where(valid, Pm, 0.0)
    o = np.argsort(Rm, 1)
    r = np.take_along_axis(Rm, o, 1)
    p = np.take_along_axis(Pm, o, 1)
    env = np.maximum.accumulate(p[:, ::-1], 1)[:, ::-1]
    env = np.concatenate([env, np.zeros((len(env), 1))], 1)
    q = np.zeros((len(r), 101))
    for k, rt in enumerate(np.linspace(0, 1, 101)):
        j = (r < rt).sum(1)
        q[:, k] = env[np.arange(len(r)), j]
    return q.mean(1)


def pair_rows(tag, ds, pt, bt, extra=()):
    out = []
    for x, y in PAIRS:
        a, b = Q.RESL.index(x), Q.RESL.index(y)
        d = bt[a] - bt[b]
        lo, hi = ci(d)
        out.append([tag, ds, f"{Q.LBL[x]}-{Q.LBL[y]}", round(pt[a], 4), round(pt[b], 4),
                    round(pt[a] - pt[b], 4), round(lo, 4), round(hi, 4),
                    round(float((d > 0).mean()), 4), int(lo <= 0 <= hi), *extra])
    return out


def part1ab():
    rows, chk = [], []
    nT = len(R.TS)
    for ds in U.DATASETS:
        cnt = None
        need = sorted({x for p in PAIRS for x in p})
        apb = {}; r01b = {}; ap_pt = {}; r01_pt = {}
        lb, lp = {}, {}; bb, bp = {}, {}
        for tr in need:
            a = Q.RESL.index(tr)
            aps, r01s, lrs, brs = [], [], [], []
            for sd in Q.SEEDS:
                imgs = Q.cell_all(ds, tr, sd)
                files = sorted(imgs)
                if cnt is None:
                    cnt, SS = boot_idx(len(files))
                    nI = len(files)
                tp = np.zeros((nI, nT)); tpp = np.zeros((nI, nT)); npd = np.zeros((nI, nT))
                gt = np.zeros(nI)
                for i, f in enumerate(files):
                    for k, t in enumerate(R.TS):
                        e = Q.evaluate(imgs[f], t, "P1", False)
                        tp[i, k], npd[i, k] = e["tp"], e["npred"]
                        tpp[i, k] = e.get("tp_pred", e["tp"])
                        gt[i] = e["tp"] + e["fn"]
                # 점 추정 (전체 이미지) + 부트스트랩
                W = np.vstack([np.ones((1, nI)), cnt])          # 0행 = 원표본
                TPb, TPPb, NPb, GTb = W @ tp, W @ tpp, W @ npd, W @ gt
                Rm = TPb / GTb[:, None]
                Pm = np.divide(TPPb, NPb, out=np.zeros_like(NPb), where=NPb > 0)
                ap = ap_vec(Rm, Pm, NPb > 0)
                aps.append(ap); r01s.append(Rm[:, 0])
                # last / best t 0.5 P0
                for src, store in (("last", lrs), ("best", brs)):
                    if src == "best":
                        ims = C.load_cell(ds, tr, sd)["images"]
                    else:
                        with open(Q.CACHE / "preds_last" / f"{ds}_{tr}_s{sd}.pkl", "rb") as fh:
                            ims = pickle.load(fh)["images"]
                    tpi = np.zeros(nI); gti = np.zeros(nI)
                    for i, f in enumerate(files):
                        P = R.prep(ims[f])
                        t_, fp_, fn_, _ = R.match_fast(P, 0.5, strict=True)
                        tpi[i], gti[i] = t_, t_ + fn_
                    store.append((W @ tpi) / (W @ gti))
            for name, lst, d_b, d_p in (("AP50", aps, apb, ap_pt), ("R001", r01s, r01b, r01_pt),
                                        ("last", lrs, lb, lp), ("best", brs, bb, bp)):
                arr = np.stack(lst)                     # (nS, 1+B)
                d_p[a] = arr[:, 0].mean()
                d_b[a] = two_stage(arr[:, 1:], a, SS)
            print(f"  [{ds}] R{tr} 완료", flush=True)
        for tag, pt, bt in (("P1_AP50", ap_pt, apb), ("P1_Recall_t001", r01_pt, r01b),
                            ("last_P0_Recall_t05", lp, lb), ("best_P0_Recall_t05", bp, bb)):
            rows += pair_rows(tag, ds, pt, bt)
    Q.wcsv("s8_bootstrap_pairdiff_P1ap_r001_last.csv",
           ["metric", "dataset", "pair", "value_a", "value_b", "delta_point",
            "CI_2stage_lo", "CI_2stage_hi", "P_gt0_2stage", "CI_contains0"], rows)
    # 검산: 점 추정 == s1 / s3 요약, best test 쌍 CI == s1 P0
    s1 = {(r["protocol"], r["dataset"], r["eval_res"]): r
          for r in U.load_rows(Q.OUT / "s1_ap50_recall001_summary.csv")}
    s3 = {(r["dataset"], r["eval_res"]): r
          for r in U.load_rows(Q.OUT / "s3_best_vs_last_summary.csv")}
    bad = 0
    for r in rows:
        x, y = r[2].split("-")
        for lab, v in ((x, r[3]), (y, r[4])):
            if r[0] == "P1_AP50":
                ref = float(s1[("P1", r[1], lab)]["AP50_mean"])
            elif r[0] == "P1_Recall_t001":
                ref = float(s1[("P1", r[1], lab)]["Recall_t001_mean"])
            elif r[0] == "last_P0_Recall_t05":
                ref = float(s3[(r[1], lab)]["last_Recall_t05_mean"])
            else:
                ref = float(s3[(r[1], lab)]["best_Recall_t05_mean"])
            bad += abs(ref - v) > 1e-3
    print(f"검산 점 추정 vs s1/s3 요약: 불일치 {bad}")
    old = {r["pair"]: r for r in U.load_rows(Q.OUT / "s1_bootstrap_pairdiff_t05.csv")
           if r["protocol"] == "P0"}
    for r in rows:
        if r[0] == "best_P0_Recall_t05" and r[1] == "test" and r[2] in old:
            print(f"  검산 best P0 test {r[2]}: CI {r[6]},{r[7]} vs s1 "
                  f"{old[r[2]]['CI_2stage_lo']},{old[r[2]]['CI_2stage_hi']}")
    return rows


def part1c():
    import r1_s7_tiling as T7
    cells = {}
    for cond, tr, s, mode in T7.CONDS:
        for ds in U.DATASETS:
            for sd in Q.SEEDS:
                p = T7.TC / f"{cond}_{ds}_{tr}_{s}_s{sd}.pkl"
                with open(p, "rb") as fh:
                    cells[(cond, ds, tr, s, sd)] = pickle.load(fh)
    rows = []
    for proto in ("P0", "P1"):
        for ds in U.DATASETS:
            TPd = {}
            for (cond, d, tr, s, sd), cell in cells.items():
                if d != ds:
                    continue
                ev = [T7.eval_img(cell[f], 0.5, proto) for f in sorted(cell)]
                TPd[(cond, tr, s, sd)] = (np.array([e[0] for e in ev]),
                                          np.array([e[1] for e in ev]))
            ks = sorted({k[:3] for k in TPd})
            nI = len(TPd[ks[0] + (0,)][0])
            rng = np.random.default_rng(R.RNG_SEED)          # s7_tiling.report 와 같은 순서
            II = rng.integers(0, nI, (B, nI))
            SS = rng.integers(0, NS, (B, NS))
            boot, pt = {}, {}
            for kk in ks:
                tp = np.stack([TPd[kk + (sd,)][0] for sd in Q.SEEDS])
                nb = TPd[kk + (0,)][1]
                rs = tp[:, II].sum(-1) / nb[II].sum(1)
                boot[kk] = rs[SS.T, np.arange(B)].mean(0)
                pt[kk] = float((tp.sum(1) / nb.sum()).mean())
            for kk in ks:
                if kk[0] == "full":
                    continue
                base = ("full", kk[1], C.GRID[kk[1]])
                d = boot[kk] - boot[base]
                lo, hi = ci(d)
                rows.append([proto, ds, kk[0], kk[1], Q.LBL[kk[1]], kk[2], round(pt[base], 4),
                             round(pt[kk], 4), round(pt[kk] - pt[base], 4), round(lo, 4),
                             round(hi, 4), round(float((d > 0).mean()), 4),
                             int(lo <= 0 <= hi)])
    Q.wcsv("s8_tiling_minus_full_ci.csv",
           ["protocol", "dataset", "condition", "train_res", "full_res", "tile_infer_size",
            "Recall_full", "Recall_tile", "delta_point", "CI_2stage_lo", "CI_2stage_hi",
            "P_gt0_2stage", "CI_contains0"], rows)
    old = {(r["protocol"], r["dataset"], r["condition"], r["train_res"]): r
           for r in U.load_rows(Q.OUT / "s7_tiling_recall_ci.csv") if r["contrast"]}
    bad = sum(abs(float(old[(r[0], r[1], r[2], str(r[3]))]["CI_2stage_lo"]) - r[9]) > 1e-4
              for r in rows)
    print(f"검산 7단계 Δ CI vs s7_tiling_recall_ci: 불일치 {bad}/{len(rows)}")
    return rows


def part2():
    rows = U.load_rows(Q.OUT / "s1_P1_pred4_matched_to_smallGT.csv")
    out = []
    for ds in U.DATASETS:
        for tr in Q.RESL:
            for t in ("0.25", "0.5", "0.75"):
                s = [r for r in rows if r["dataset"] == ds and int(r["train_res"]) == tr
                     and r["t"] == t]
                sm = np.array([int(r["TP_pred4_on_GT_lt4cm"]) for r in s], float)
                tpp = np.array([int(r["TP_pred4"]) for r in s], float)
                npd = np.array([int(r["n_pred4"]) for r in s], float)
                out.append([ds, tr, Q.LBL[tr], t, len(s), int(sm.sum()), round(sm.mean(), 3),
                            round(sm.std(ddof=1), 3), int(sm.min()), int(sm.max()),
                            round(tpp.mean(), 3), round(npd.mean(), 3),
                            round(float((sm / tpp).mean()), 4),
                            round(float((sm / npd).mean()), 4)])
    Q.wcsv("s8_P1_pred4_on_smallGT_by_res.csv",
           ["dataset", "train_res", "eval_res", "t", "n_seeds", "n_sum_8seed",
            "n_mean_per_seed", "n_sd", "n_min_seed", "n_max_seed", "TP_pred4_mean",
            "n_pred4_mean", "frac_of_TP_pred4_mean", "frac_of_pred4_mean"], out)
    return out


def part3():
    d = F.PIXEL_ROOT / "pixel_split" / "train"
    od = Q.OUT / "s8_train_outlier"
    od.mkdir(exist_ok=True)
    coco = json.load(open(d / "_annotations.coco.json", encoding="utf-8"))
    ann_by = {}
    for a in coco["annotations"]:
        ann_by.setdefault(a["image_id"], []).append(a)
    rows = []
    for im in coco["images"]:
        H, W = im["height"], im["width"]
        side = max(H, W)
        cpp = F.FRAME_CM / side
        for k, a in enumerate(ann_by.get(im["id"], [])):
            m = F.seg_to_mask(a["segmentation"], H, W).astype(np.uint8)
            if int(m.sum()) == 0:
                continue
            mj = F.major_axis_px(m)
            if mj * cpp <= 70.7:
                continue
            seg = a["segmentation"]
            npoly = len(seg) if isinstance(seg, list) else 0
            nvert = sum(len(p) // 2 for p in seg) if isinstance(seg, list) else 0
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnts, key=cv2.contourArea)
            (ex, ey), (ea, eb), eang = cv2.fitEllipse(c)
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            hull = cv2.convexHull(c).reshape(-1, 2).astype(float)
            feret = float(np.max(np.linalg.norm(hull[:, None] - hull[None], axis=-1)))
            ys, xs = np.where(m)
            area = int(m.sum())
            rows.append([im["file_name"], im.get("extra", {}).get("name", ""), im["id"],
                         a["id"], k, W, H, npoly, nvert, len(cnts), len(c), area,
                         round(area * cpp ** 2, 2), a.get("area", ""),
                         int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()),
                         int(xs.min() == 0 or ys.min() == 0 or xs.max() == W - 1
                             or ys.max() == H - 1),
                         round(mj, 1), round(mj * cpp, 2),
                         round(max(ea, eb), 1), round(min(ea, eb), 1),
                         round(ex, 1), round(ey, 1), round(eang, 1),
                         round(max(rw, rh) * cpp, 2), round(feret * cpp, 2),
                         "cv2.fitEllipse(최대 외곽 contour, CHAIN_APPROX_SIMPLE) 장축; "
                         "점 5개 미만이면 minAreaRect 장변 (clod_frag_260907.major_axis_px)"])
            img = cv2.imread(str(d / im["file_name"]))
            ov = img.copy()
            ov[m > 0] = (0.5 * ov[m > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.drawContours(ov, [c], -1, (0, 255, 255), 3)
            cv2.ellipse(ov, ((ex, ey), (ea, eb), eang), (255, 0, 0), 3)
            cv2.rectangle(ov, (0, 0), (W - 1, H - 1), (0, 255, 0), 3)
            txt = (f"ann {a['id']} (#{k}) major {mj * cpp:.1f} cm (fitEllipse) / "
                   f"Feret {feret * cpp:.1f} cm / area {area} px")
            cv2.putText(ov, txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 6)
            cv2.putText(ov, txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2)
            stem = f"{im['extra'].get('name', im['file_name']).rsplit('.', 1)[0]}_ann{a['id']}"
            cv2.imwrite(str(od / f"{stem}_overlay.png"), ov)
            x0, y0 = max(int(xs.min()) - 150, 0), max(int(ys.min()) - 150, 0)
            x1, y1 = min(int(xs.max()) + 150, W), min(int(ys.max()) + 150, H)
            cv2.imwrite(str(od / f"{stem}_crop.png"), ov[y0:y1, x0:x1])
    Q.wcsv("s8_train_gt_major_gt70p7cm.csv",
           ["file_name", "original_name", "image_id", "ann_id", "ann_index_in_image",
            "W", "H", "n_polygons", "n_vertices", "n_external_contours",
            "n_points_largest_contour", "mask_area_px", "mask_area_cm2",
            "coco_area_field(미사용)", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
            "touches_edge", "major_px", "major_cm", "ellipse_major_px",
            "ellipse_minor_px", "ellipse_cx", "ellipse_cy", "ellipse_angle",
            "minAreaRect_long_cm", "max_Feret_cm", "major_axis_method"], rows)
    for r in rows:
        print(r[:13], r[19:28])
    return rows


if __name__ == "__main__":
    import sys
    parts = sys.argv[1:] or ["3", "2", "1c", "1ab"]
    for p in parts:
        {"1ab": part1ab, "1c": part1c, "2": part2, "3": part3}[p]()
        print(f"[8-{p}] 완료", flush=True)
