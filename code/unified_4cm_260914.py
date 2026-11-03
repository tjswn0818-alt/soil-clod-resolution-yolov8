"""
토양 clod 분할 평가 — 단일 통일 파이프라인 (unified_4cm_*).

배경 — 왜 통일하는가
  같은 조건(512 matched, test)의 Recall 이 파일마다 달랐다.
      matched9_260823 / cross9_long_260823        0.8772   seed 0,  INTER_AREA
      matched9_linear_260825 / cross9_long_linear 0.8860   seed 0,  INTER_LINEAR
      cross_seeds_long_260824 = frag_seeds8       0.8904   8 seed,  INTER_AREA
      seed8_matched_linear_260825                 0.9101   8 seed,  INTER_LINEAR
  원인은 매칭·필터 차이가 아니라 (a) seed 집합 (b) 리사이즈 보간 두 가지다.
  (매칭 규약과 4cm 필터는 모든 기존 스크립트가 이미 동일하게 처리하고 있었음을
   clod_frag_symmetry_260914.py 가 2778/2778 항목 일치로 확인했다.)
  이 스크립트는 그 둘을 고정해 모든 분석을 하나의 코드 경로에서 산출한다.

═══ 공통 평가 조건 (모든 분석에 동일 적용) ═══
  데이터셋      test, val 둘 다
  seed          0-7, 지표는 seed 별 계산 후 평균±SD
  리사이즈 보간  이미지 cv2.INTER_LINEAR / 마스크 cv2.INTER_NEAREST
                (ultralytics 학습 경로 LetterBox 와 동일 -> train/eval 전처리 일치)
  크기 기준      장축(major axis) 길이. 면적 기준 아님.
  coarse clod   장축 >= 4 cm  (MIN_MAJOR_FRAC = 0.08, 프레임 50 cm)
                GT  : 원본 격자에서 장축 >= 4cm 인 clod 만
                예측: 평가 격자에서 장축 >= 4cm 인 예측만
                둘 다 '매칭 전'에 적용 -> GT·예측 대칭
  px->cm        이미지별 cm/px = 50 / max(H, W)
  confidence    0.5
  매칭          예측 면적 내림차순 greedy, IoU >= 0.5 로 TP, GT 1개 : 예측 1개
  ED            2*sqrt(A/pi)

═══ 산출 ═══
  (1) SR matched   9 해상도 (학습=추론)
  (2) Cross 9x9    9 학습 x 9 추론 해상도, Recall 행렬. 대각선 = (1)
  (3) 검출 수준     P3/P4/P5 점유율 (matched, >=4cm 예측 기준)
  (4) MR           5 범위 x 추론 448/512/736/1024
  (5) 파쇄도 특성   SR matched 9 + MR 20 조건
  (6) FPS          기존 측정 승계 (타이밍은 필터·매칭과 무관, 이미 bilinear 규약)

  raw 는 (2)(4) 전 셀에 대해 셀 단위로 남기고, 파쇄도용 이미지/개체/매칭쌍 raw 는
  파쇄도 대상 셀(SR matched 대각선 + MR 전 조건)에만 남긴다.

═══ 저해상도 주의 ═══
  128/256 은 검출 붕괴로 매칭쌍이 적어 ED R2 / KS 가 불안정하다.
  표본 부족(매칭쌍 < 3, 예측 개체 < 3)이면 NaN 으로 남기고 유효 seed/표본 수를 기록한다.
  파쇄도 표에서 128/256 은 참고용으로만 볼 것.

실행
  python unified_4cm_260914.py                 # 전체 (약 2시간, 재개 지원)
  python unified_4cm_260914.py --stage cross   # 단계별
  python unified_4cm_260914.py --report        # 추론 없이 집계만
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clod_frag_260907 as F              # noqa: E402  마스크/매칭/도형 커널
import clod_frag_seeds8_260910 as S       # noqa: E402  지표/집계 공유 함수
import clod_frag_mr_seeds8_260910 as M    # noqa: E402  MR 가중치 규약
import scale_distribution_multi_260820 as SD  # noqa: E402  레벨 hook

STAMP = "260914"
PFX = "unified_4cm"
RES = F.RS_ROOT / "results"
FIG = F.FIG
OUT = RES / "unified"
FRAME_CM = F.FRAME_CM
NAN = float("nan")

INTERP_IMG = cv2.INTER_LINEAR
INTERP_MASK = cv2.INTER_NEAREST
CONF = 0.5
IOU_NMS = 0.7

DATASETS = ["test", "val"]
SEEDS = list(range(8))
TRAIN_RES = [128, 256, 384, 448, 512, 736, 1024, 1280, 1600]
EVAL_GRID = [int(np.ceil(r / 32) * 32) for r in TRAIN_RES]   # 32 배수 격자
SR_LABEL = {r: f"SR-{int(np.ceil(r / 32) * 32)}" for r in TRAIN_RES}
# MR 조건. MR[448-736] 은 MR[448-1024] 의 상한만 1024->736 으로 좁힌 추가 조건
# (학습 ladder 736/512/448).
MR_SPECS = list(M.MR_SPECS) + [("MR[448-736]", "_448_736")]
MR_EVALS = M.EVAL_RES
# 각 MR 의 학습 해상도 범위 (격자 기준) — 추론 해상도가 범위 안인지 표시용
MR_RANGE = {"MR[448-1024]": (448, 1024), "MR[448-1280]": (448, 1280),
            "MR[448-1600]": (448, 1600), "MR[384-1600]": (384, 1600),
            "MR[128-1600]": (128, 1600), "MR[448-736]": (448, 736)}


def in_range(lbl, ev):
    lo, hi = MR_RANGE.get(lbl, (0, 10 ** 6))
    return int(lo <= ev <= hi)
LEVEL_CONFS = [0.50, 0.25]     # 0.50 = 공통 조건(주), 0.25 = 기존 표와의 연속성

P_CELL = OUT / f"{PFX}_cells_{STAMP}.csv"
P_IMG = OUT / f"{PFX}_perimage_{STAMP}.csv"
P_INS = OUT / f"{PFX}_instances_{STAMP}.csv"
P_PAR = OUT / f"{PFX}_pairs_{STAMP}.csv"
P_LVL = OUT / f"{PFX}_detlevel_raw_{STAMP}.csv"

H_CELL = ["dataset", "family", "model", "train_res", "eval_res", "seed",
          "matched", "n_images", "n_GT", "n_pred_raw", "n_pred_4cm",
          "TP", "FP", "FN", "Recall", "Precision", "F1", "mIoU",
          "FN_per_img", "FP_per_img"]
H_IMG = ["dataset", "family", "model", "eval_res", "seed", "file",
         "n_clod_GT", "n_clod_Pred", "area4cm_GT_cm2", "area4cm_Pred_cm2",
         "ratio_frame_GT", "ratio_frame_Pred", "area_allclod_GT_cm2",
         "area_allclod_Pred_cm2", "ratio_ofclod_GT", "ratio_ofclod_Pred",
         "TP", "FN", "FP"]
H_INS = ["dataset", "family", "model", "eval_res", "seed", "file", "source",
         "idx", "area_cm2", "ed_cm", "major_cm", "matched"]
H_PAR = ["dataset", "family", "model", "eval_res", "seed", "file", "gt_idx",
         "pred_idx", "iou", "area_GT_cm2", "area_Pred_cm2", "ed_GT_cm",
         "ed_Pred_cm", "major_GT_cm", "major_Pred_cm"]
H_LVL = ["dataset", "model", "train_res", "eval_res", "seed", "conf",
         "filter4cm", "det_P3", "det_P4", "det_P5", "det_total",
         "P3_pct", "P4_pct", "P5_pct", "dominant"]

rd, fm, fnum = S.rd, S.fm, S.fnum
append_rows, load_rows = S.append_rows, S.load_rows

# 검출 지표 (SR/MR 요약 공통)
DET_METRICS = [("Recall", "Recall", 4), ("Precision", "Precision", 4),
               ("F1", "F1", 4), ("mIoU", "mIoU (매칭 clod)", 4),
               ("TP", "TP", 2), ("FN_per_img", "FN/img", 3),
               ("FP_per_img", "FP/img", 3),
               ("n_pred_4cm", "예측 clod 수(>=4cm)", 2),
               ("n_pred_raw", "예측 총수(필터 전)", 2)]
# 파쇄도 지표 (요청 ⑤)
FRAG_METRICS = [("count_MAE", 3), ("count_RMSE", 3), ("count_R2_1to1", 3),
                ("count_bias", 3), ("n_pairs", 2), ("ED_MAE_cm", 4),
                ("ED_R2_1to1", 4), ("area_bias_pct", 3), ("mean_IoU", 4),
                ("ED_KS_D", 4), ("ED_KS_p", 4), ("ED_wasserstein", 4),
                ("ratioFrame_MAE", 3), ("ratioFrame_bias", 3)]


# ══════════════════════ 단일 평가 커널 ══════════════════════
def eval_cell(model, gt, img_cache, eval_r, keep_raw):
    """공통 조건 1셀 평가. 예측은 매칭 전에 장축>=4cm 로 필터(대칭)."""
    r32 = int(np.ceil(eval_r / 32) * 32)
    thr = F.MIN_MAJOR_FRAC * r32
    TP = FP = FN = 0
    n_raw = n_4cm = 0
    ious, per_img, inst, pair = [], [], [], []
    for f in sorted(gt):
        g = gt[f]
        cpp = g["cpp"]
        img_r = cv2.resize(img_cache[f], (r32, r32), interpolation=INTERP_IMG)
        gts_r = [cv2.resize(m, (r32, r32), interpolation=INTERP_MASK)
                 for m in g["masks"]]
        r = model.predict(img_r, imgsz=r32, conf=CONF, retina_masks=True,
                          verbose=False, device=0)[0]
        preds_r, all_r = [], []
        if r.masks is not None:
            for pm in r.masks.data.cpu().numpy():
                m = (pm > 0.5).astype(np.uint8)
                if m.shape != (r32, r32):
                    m = cv2.resize(m, (r32, r32), interpolation=INTERP_MASK)
                if m.sum() == 0:
                    continue
                all_r.append(m)
                if F.major_axis_px(m) >= thr:          # ← 매칭 전 4cm 필터
                    preds_r.append(m)
        flags, pairs, tp = F.match(preds_r, gts_r)     # ← 그 다음 매칭
        TP += tp
        FP += len(preds_r) - tp
        FN += len(gts_r) - tp
        n_raw += len(all_r)
        n_4cm += len(preds_r)
        ious += [i for (_, i) in pairs.values()]
        if not keep_raw:
            continue
        # 파쇄도용 raw — 면적/직경은 원본 격자로 되돌려 cm 환산
        H, W = g["H"], g["W"]
        up = lambda m: cv2.resize(m, (W, H), interpolation=INTERP_MASK)
        gs = [F.shape_stats(m, cpp) for m in g["masks"]]
        ps = [F.shape_stats(up(m), cpp) for m in preds_r]
        pred_all_cm2 = sum(float(up(m).sum()) * cpp ** 2 for m in all_r)
        frame = FRAME_CM ** 2
        ga, pa = sum(x[0] for x in gs), sum(x[0] for x in ps)
        per_img.append([f, len(gs), len(ps), round(ga, 3), round(pa, 3),
                        round(ga / frame, 5), round(pa / frame, 5),
                        round(g["total_clod_cm2"], 3), round(pred_all_cm2, 3),
                        rd(ga / g["total_clod_cm2"]) if g["total_clod_cm2"] > 0
                        else "",
                        rd(pa / pred_all_cm2) if pred_all_cm2 > 0 else "",
                        tp, len(gs) - tp, len(ps) - tp])
        inv = {di: gi for gi, (di, _) in pairs.items()}
        for i, (a, e, mj) in enumerate(gs):
            inst.append([f, "GT", i, round(a, 4), round(e, 4), round(mj, 4),
                         int(flags[i])])
        for i, (a, e, mj) in enumerate(ps):
            inst.append([f, "Pred", i, round(a, 4), round(e, 4), round(mj, 4),
                         int(i in inv)])
        for gi, (di, iou) in pairs.items():
            pair.append([f, gi, di, round(iou, 4), round(gs[gi][0], 4),
                         round(ps[di][0], 4), round(gs[gi][1], 4),
                         round(ps[di][1], 4), round(gs[gi][2], 4),
                         round(ps[di][2], 4)])
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    return dict(r32=r32, TP=TP, FP=FP, FN=FN, Recall=R, Precision=P,
                F1=(2 * P * R / (P + R) if P + R else 0.0),
                mIoU=float(np.mean(ious)) if ious else 0.0,
                n_raw=n_raw, n_4cm=n_4cm,
                per_img=per_img, inst=inst, pair=pair)


def load_data():
    D = {}
    for ds in DATASETS:
        d = F.DATASETS[ds]
        gt = F.load_gt(d)
        files = sorted(gt)
        D[ds] = dict(dir=d, gt=gt, files=files,
                     ngt=sum(len(gt[f]["masks"]) for f in files),
                     img={f: cv2.imread(str(d / f)) for f in files})
        print(f"[{ds}] {d.name} | 이미지 {len(files)}장 | "
              f"GT coarse clod(>=4cm) {D[ds]['ngt']}개")
    return D


def write_cell(ds, fam, lbl, tr, sd, m, matched, n_img):
    append_rows(P_CELL, H_CELL, [[
        ds, fam, lbl, tr, m["r32"], sd, int(matched), n_img,
        m["TP"] + m["FN"], m["n_raw"], m["n_4cm"], m["TP"], m["FP"], m["FN"],
        rd(m["Recall"]), rd(m["Precision"]), rd(m["F1"]), rd(m["mIoU"]),
        rd(m["FN"] / n_img, 4), rd(m["FP"] / n_img, 4)]])
    if m["per_img"]:
        pre = [ds, fam, lbl, m["r32"], sd]
        append_rows(P_IMG, H_IMG, [pre + r for r in m["per_img"]])
        append_rows(P_INS, H_INS, [pre + r for r in m["inst"]])
        append_rows(P_PAR, H_PAR, [pre + r for r in m["pair"]])


# ══════════════════════ 단계 1-2: SR cross 9x9 (대각선 = matched) ══════════════════════
def stage_cross(D, seeds):
    done = {(r["dataset"], r["model"], int(r["eval_res"]), int(r["seed"]))
            for r in load_rows(P_CELL) if r["family"] == "SR"}
    todo = [(ds, tr, s, ev) for ds in DATASETS for tr in TRAIN_RES
            for s in seeds for ev in EVAL_GRID
            if (ds, SR_LABEL[tr], ev, s) not in done]
    print("\n" + "=" * 112)
    print(f"[1-2] SR 교차 9x9 — {len(TRAIN_RES)}학습 x {len(EVAL_GRID)}추론 x "
          f"{len(seeds)}seed x {len(DATASETS)}set = "
          f"{len(TRAIN_RES) * len(EVAL_GRID) * len(seeds) * len(DATASETS)}셀, "
          f"남은 {len(todo)}셀   (대각선 = matched, 파쇄도 raw 저장)")
    print("=" * 112)
    if not todo:
        print("완료됨 — 건너뜀")
        return
    for tr in TRAIN_RES:
        lbl = SR_LABEL[tr]
        for sd in seeds:
            need = [(ds, ev) for ds in DATASETS for ev in EVAL_GRID
                    if (ds, lbl, ev, sd) not in done]
            if not need:
                continue
            ck = S.ckpt_of(tr, sd)
            if not ck.exists():
                print(f"  [skip] 가중치 없음: {ck}")
                continue
            model = YOLO(str(ck))          # (학습해상도, seed) 당 1회 로드
            t0 = time.time()
            for ds, ev in need:
                dd = D[ds]
                matched = (ev == int(np.ceil(tr / 32) * 32))
                m = eval_cell(model, dd["gt"], dd["img"], ev, keep_raw=matched)
                write_cell(ds, "SR", lbl, tr, sd, m, matched, len(dd["files"]))
            del model
            torch.cuda.empty_cache()
            print(f"  {lbl:>8} s{sd} | {len(need):>2}셀 {time.time() - t0:>6.1f}s",
                  flush=True)


# ══════════════════════ 단계 4: MR ══════════════════════
def stage_mr(D, seeds):
    done = {(r["dataset"], r["model"], int(r["eval_res"]), int(r["seed"]))
            for r in load_rows(P_CELL) if r["family"] == "MR"}
    todo = [(ds, l, ev, s) for ds in DATASETS for l, _ in MR_SPECS
            for ev in MR_EVALS for s in seeds if (ds, l, ev, s) not in done]
    print("\n" + "=" * 112)
    print(f"[4] MR — {len(MR_SPECS)}범위 x {len(MR_EVALS)}추론 x {len(seeds)}seed "
          f"x {len(DATASETS)}set = "
          f"{len(MR_SPECS) * len(MR_EVALS) * len(seeds) * len(DATASETS)}셀, "
          f"남은 {len(todo)}셀")
    print("=" * 112)
    if not todo:
        print("완료됨 — 건너뜀")
        return
    for lbl, tag in MR_SPECS:
        for sd in seeds:
            need = [(ds, ev) for ds in DATASETS for ev in MR_EVALS
                    if (ds, lbl, ev, sd) not in done]
            if not need:
                continue
            ck = M.ckpt_of(tag, sd)
            if not ck.exists():
                print(f"  [skip] 가중치 없음: {ck}")
                continue
            model = YOLO(str(ck))
            t0 = time.time()
            for ds, ev in need:
                dd = D[ds]
                m = eval_cell(model, dd["gt"], dd["img"], ev, keep_raw=True)
                write_cell(ds, "MR", lbl, "", sd, m, False, len(dd["files"]))
            del model
            torch.cuda.empty_cache()
            print(f"  {lbl:>13} s{sd} | {len(need):>2}셀 "
                  f"{time.time() - t0:>6.1f}s", flush=True)


# ══════════════════════ 단계 3: 검출 수준 P3/P4/P5 ══════════════════════
def nms_levels_4cm(y, nc, conf, level_sizes, thr4):
    from torchvision.ops import nms
    box, cls = y[0, :4, :], y[0, 4:4 + nc, :]
    score, _ = cls.max(0)
    keep0 = (score > conf).nonzero(as_tuple=True)[0]
    out = {0: [0, 0, 0], 1: [0, 0, 0]}
    if keep0.numel() == 0:
        return out
    b = box[:, keep0]
    xyxy = torch.stack([b[0] - b[2] / 2, b[1] - b[3] / 2,
                        b[0] + b[2] / 2, b[1] + b[3] / 2], 1)
    sel = nms(xyxy.float(), score[keep0].float(), IOU_NMS)
    idx, ls = keep0[sel], torch.maximum(b[2], b[3])[sel]
    bounds = np.cumsum([0] + level_sizes)
    for a, L in zip(idx.tolist(), ls.tolist()):
        for li in range(3):
            if bounds[li] <= a < bounds[li + 1]:
                out[0][li] += 1
                if L >= thr4:
                    out[1][li] += 1
                break
    return out


def stage_level(D, seeds):
    """matched 조건, >=4cm 예측 기준 P3/P4/P5 점유율.

    레벨 귀속에는 raw head 앵커 인덱스가 필요해 마스크를 못 쓰므로
    박스 장변 max(w,h) >= 0.08*r32 를 4cm 대용으로 쓴다
    (마스크 장축 기준과의 판정 일치율 약 95%, 128/256 은 격자가 거칠어 신뢰도 낮음).
    """
    done = {(r["dataset"], int(r["train_res"]), int(r["seed"]))
            for r in load_rows(P_LVL)}
    todo = [(ds, tr, s) for ds in DATASETS for tr in TRAIN_RES for s in seeds
            if (ds, tr, s) not in done]
    print("\n" + "=" * 112)
    print(f"[3] 검출 수준 P3/P4/P5 (matched, >=4cm) — 남은 {len(todo)}셀")
    print("=" * 112)
    if not todo:
        print("완료됨 — 건너뜀")
        return
    for tr in TRAIN_RES:
        for sd in seeds:
            need = [ds for ds in DATASETS if (ds, tr, sd) not in done]
            if not need:
                continue
            ck = S.ckpt_of(tr, sd)
            if not ck.exists():
                continue
            net, head, hook, nc, nm = SD.load_model(str(ck), "cuda")
            r32 = int(np.ceil(tr / 32) * 32)
            thr4 = F.MIN_MAJOR_FRAC * r32
            for ds in need:
                acc = {(c, fl): np.zeros(3, np.int64) for c in LEVEL_CONFS
                       for fl in (0, 1)}
                for f in D[ds]["files"]:
                    hook.clear()
                    im = cv2.resize(D[ds]["img"][f], (r32, r32),
                                    interpolation=INTERP_IMG)
                    t = torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1)
                    t = t.float().div(255).unsqueeze(0).to("cuda")
                    with torch.inference_mode():
                        pr = net(t)
                    y = SD.pick_raw(pr, nc, nm)
                    grid = [int(hook.out[i].shape[-1] * hook.out[i].shape[-2])
                            for i in range(3)]
                    for c in LEVEL_CONFS:
                        g = nms_levels_4cm(y, nc, c, grid, thr4)
                        for fl in (0, 1):
                            acc[(c, fl)] += np.array(g[fl], np.int64)
                rows = []
                for c in LEVEL_CONFS:
                    for fl in (0, 1):
                        dt = acc[(c, fl)]
                        tot = int(dt.sum())
                        sh = dt / tot * 100 if tot else np.zeros(3)
                        rows.append([ds, SR_LABEL[tr], tr, r32, sd, c, fl,
                                     int(dt[0]), int(dt[1]), int(dt[2]), tot,
                                     round(sh[0], 2), round(sh[1], 2),
                                     round(sh[2], 2),
                                     ["P3", "P4", "P5"][int(np.argmax(sh))]
                                     if tot else ""])
                append_rows(P_LVL, H_LVL, rows)
            hook.close()
            del net
            torch.cuda.empty_cache()
            print(f"  {SR_LABEL[tr]:>8} s{sd} 완료", flush=True)


# ══════════════════════ 집계 ══════════════════════
def agg_rows(sel, keys):
    o = []
    for k, prec in keys:
        v = np.array([float(x[k]) for x in sel
                      if x.get(k) not in ("", None)], float)
        v = v[np.isfinite(v)]
        n = len(v)
        o += [n, rd(float(v.mean()) if n else NAN, 6),
              rd(float(v.std(ddof=1)) if n > 1 else NAN, 6)]
    return o


def summarize_det(cells, seeds):
    """(1) SR matched 요약 + (4) MR 요약."""
    keys = [(k, p) for k, _, p in DET_METRICS]
    hdr_tail = []
    for k, lab, _ in DET_METRICS:
        hdr_tail += [f"{k}_n", f"{k}_mean", f"{k}_sd"]

    sr, mr = [], []
    for ds in DATASETS:
        for tr in TRAIN_RES:
            lbl, ev = SR_LABEL[tr], int(np.ceil(tr / 32) * 32)
            sel = [c for c in cells if c["dataset"] == ds and c["model"] == lbl
                   and int(c["eval_res"]) == ev and c["family"] == "SR"]
            if sel:
                sr.append([ds, lbl, tr, ev, len(sel)] + agg_rows(sel, keys))
        for lbl, _ in MR_SPECS:
            for ev in MR_EVALS:
                sel = [c for c in cells if c["dataset"] == ds
                       and c["model"] == lbl and int(c["eval_res"]) == ev]
                if sel:
                    # 학습 범위 밖 추론(예: MR[448-736] @1024)은 참고용 표시
                    mr.append([ds, lbl, "", ev, len(sel)] + agg_rows(sel, keys)
                              + [in_range(lbl, ev)])
    for name, rows in [("sr", sr), ("mr", mr)]:
        with open(OUT / f"{PFX}_{name}_summary_{STAMP}.csv", "w", newline="",
                  encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "model", "train_res", "eval_res", "n_seeds"]
                       + hdr_tail + (["eval_in_train_range"] if name == "mr"
                                     else []))
            w.writerows(rows)
    return sr, mr


def cross_matrix(cells):
    """(2) 9x9 Recall 행렬 (mean / sd) — 데이터셋별."""
    out = {}
    for ds in DATASETS:
        mean, sdv = [], []
        for tr in TRAIN_RES:
            lbl = SR_LABEL[tr]
            rm, rs = [lbl, tr], [lbl, tr]
            for ev in EVAL_GRID:
                v = np.array([float(c["Recall"]) for c in cells
                              if c["dataset"] == ds and c["family"] == "SR"
                              and c["model"] == lbl
                              and int(c["eval_res"]) == ev], float)
                rm.append(rd(float(v.mean()), 5) if len(v) else "")
                rs.append(rd(float(v.std(ddof=1)), 5) if len(v) > 1 else "")
            mean.append(rm)
            sdv.append(rs)
        for tag, rows in [("", mean), ("_sd", sdv)]:
            with open(OUT / f"{PFX}_cross_recall_{ds}{tag}_{STAMP}.csv", "w",
                      newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(["model", "train_res"] + [f"eval_{e}"
                                                     for e in EVAL_GRID])
                w.writerows(rows)
        out[ds] = mean
    return out


def summarize_level():
    rows = load_rows(P_LVL)
    out = []
    for ds in DATASETS:
        for tr in TRAIN_RES:
            for c in LEVEL_CONFS:
                for fl in (1, 0):
                    sel = [r for r in rows if r["dataset"] == ds
                           and int(r["train_res"]) == tr
                           and abs(float(r["conf"]) - c) < 1e-9
                           and int(r["filter4cm"]) == fl]
                    if not sel:
                        continue
                    st = {}
                    for col in ("P3_pct", "P4_pct", "P5_pct", "det_total"):
                        v = np.array([float(x[col]) for x in sel])
                        st[col] = (float(v.mean()),
                                   float(v.std(ddof=1)) if len(v) > 1 else NAN)
                    dom = ["P3", "P4", "P5"][int(np.argmax(
                        [st["P3_pct"][0], st["P4_pct"][0], st["P5_pct"][0]]))]
                    out.append([ds, SR_LABEL[tr], tr, c, fl, len(sel),
                                rd(st["det_total"][0], 2),
                                rd(st["det_total"][1], 2)]
                               + [rd(st[f"P{i}_pct"][j], 3)
                                  for i in (3, 4, 5) for j in (0, 1)] + [dom])
    with open(OUT / f"{PFX}_detlevel_{STAMP}.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "train_res", "conf", "filter4cm",
                    "n_seeds", "det_total_mean", "det_total_sd",
                    "P3_pct_mean", "P3_pct_sd", "P4_pct_mean", "P4_pct_sd",
                    "P5_pct_mean", "P5_pct_sd", "dominant"])
        w.writerows(out)
    return out


def summarize_frag(seeds):
    """(5) 파쇄도 특성 — SR matched 9 + MR 20, seed 별 계산 후 평균±SD."""
    IM, IN, PA, _ = S.group_raw(
        lambda r: (r["dataset"], r["family"], r["model"], int(r["eval_res"]),
                   int(r["seed"])), (P_IMG, P_INS, P_PAR, P_CELL))
    cells = {(r["dataset"], r["family"], r["model"], int(r["eval_res"]),
              int(r["seed"])): r for r in load_rows(P_CELL)}
    conds = ([("SR", SR_LABEL[t], int(np.ceil(t / 32) * 32)) for t in TRAIN_RES]
             + [("MR", l, e) for l, _ in MR_SPECS for e in MR_EVALS])
    seed_rows = []
    for ds in DATASETS:
        for fam, lbl, ev in conds:
            for sd in seeds:
                k = (ds, fam, lbl, ev, sd)
                if k not in IM:
                    continue
                c = cells.get(k, {})
                d0 = {"n_GT_clod": c.get("n_GT", ""),
                      "n_Pred_clod": c.get("n_pred_4cm", ""),
                      "TP": c.get("TP", ""), "Recall": c.get("Recall", ""),
                      "Precision": c.get("Precision", "")}
                seed_rows.append(S.metrics_one(
                    IM[k], PA.get(k, []), IN.get(k, []), d0,
                    dataset=ds, family=fam, model=lbl, eval_res=ev, seed=sd))
    cols = (["dataset", "family", "model", "eval_res", "seed", "collapsed",
             "n_GT_clod", "n_Pred_clod", "n_gt_inst", "n_pred_inst"]
            + [k for k, _ in FRAG_METRICS])
    with open(OUT / f"{PFX}_frag_seed_metrics_{STAMP}.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for m in seed_rows:
            w.writerow([m.get(c2, "") if c2 in ("dataset", "family", "model")
                        else rd(m.get(c2, ""), 6) for c2 in cols])

    hdr_tail = []
    for k, _ in FRAG_METRICS:
        hdr_tail += [f"{k}_n", f"{k}_mean", f"{k}_sd"]
    sr, mr = [], []
    for ds in DATASETS:
        for fam, lbl, ev in conds:
            sel = [m for m in seed_rows if m["dataset"] == ds
                   and m["model"] == lbl and m["eval_res"] == ev]
            if not sel:
                continue
            n_col = sum(m["collapsed"] for m in sel)
            ok = [m for m in sel if not m["collapsed"]]
            row = [ds, lbl, ev, len(ok), len(sel), n_col]
            for k, _ in FRAG_METRICS:
                n, mu, s, _mn, _mx = S.agg_metric(ok, k)
                row += [n, rd(mu, 6), rd(s, 6)]
            if fam == "MR":
                row += [in_range(lbl, ev)]
            (sr if fam == "SR" else mr).append(row)
    for name, rows in [("sr", sr), ("mr", mr)]:
        with open(OUT / f"{PFX}_frag_{name}_{STAMP}.csv", "w", newline="",
                  encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "model", "eval_res", "n_seeds_used",
                        "n_seeds_total", "n_seeds_collapsed"] + hdr_tail
                       + (["eval_in_train_range"] if name == "mr" else []))
            w.writerows(rows)
    return seed_rows, sr, mr


def carry_fps():
    """(6) FPS — 기존 측정 승계. 타이밍은 4cm 필터·매칭과 무관하고
    이미 INTER_LINEAR 규약이므로 재측정 불필요."""
    src = RES / "fps_sr_matched_260825.csv"
    if not src.exists():
        print("  [건너뜀] fps_sr_matched_260825.csv 없음")
        return []
    rows = load_rows(src)
    sr = {}
    for r in load_rows(OUT / f"{PFX}_sr_summary_{STAMP}.csv"):
        sr[(r["dataset"], int(r["train_res"]))] = r
    out = []
    for r in rows:
        tr = int(r["res"])
        o = [tr, r["grid_r32"], r["pure_mean_ms"], r["pure_sd_ms"],
             r["pure_FPS"], r["full_mean_ms"], r["full_sd_ms"], r["full_FPS"]]
        for ds in DATASETS:
            s = sr.get((ds, tr))
            o += [s["Recall_mean"] if s else "", s["Recall_sd"] if s else ""]
        out.append(o)
    with open(OUT / f"{PFX}_fps_{STAMP}.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["train_res", "grid_r32", "pure_mean_ms", "pure_sd_ms",
                    "pure_FPS", "full_mean_ms", "full_sd_ms", "full_FPS",
                    "Recall_test_mean", "Recall_test_sd",
                    "Recall_val_mean", "Recall_val_sd"])
        w.writerows(out)
    return out


# ══════════════════════ 검증 ══════════════════════
def verify(cells, sr_sum, cross):
    v = []
    print("\n" + "=" * 112)
    print("=== 검증 ===")
    print("=" * 112)

    # (a) matched Recall == cross 대각선
    #     두 출력은 같은 cells 에서 나오지만 저장 자릿수가 달라(6 vs 5) 반올림 차이가
    #     난다. 따라서 원자료(cells)에서 양쪽 경로를 다시 계산해 비트 단위로 비교한다.
    bad = n = 0
    for ds in DATASETS:
        for tr in TRAIN_RES:
            lbl, ev = SR_LABEL[tr], int(np.ceil(tr / 32) * 32)
            sel = [c for c in cells if c["dataset"] == ds
                   and c["family"] == "SR" and c["model"] == lbl
                   and int(c["eval_res"]) == ev]
            mt = [c for c in sel if int(c["matched"]) == 1]
            if not sel or not mt:
                continue
            n += 1
            x = float(np.mean([float(c["Recall"]) for c in mt]))     # (1) matched
            y = float(np.mean([float(c["Recall"]) for c in sel]))    # (2) 대각선
            same_cells = len(sel) == len(mt)
            if x != y or not same_cells:
                bad += 1
                print(f"  [불일치] {ds} {lbl}: matched {x!r} ({len(mt)}셀) vs "
                      f"대각선 {y!r} ({len(sel)}셀)")
    print(f"(a) matched Recall == cross 대각선 : {n}쌍 대조, 불일치 {bad}개 "
          f"-> {'완전 일치 (동일 셀·동일 값)' if bad == 0 else '확인 필요'}")
    v.append(["matched_eq_cross_diagonal", n, bad,
              "PASS" if bad == 0 else "FAIL"])

    # (b) 512 matched Recall 을 기존 3개 값과 비교
    print("\n(b) 512 matched Recall (test) — 기존 값과의 비교")
    ref = [("matched9_260823 / cross9_long_260823", 0.8772, "seed 0, AREA"),
           ("matched9_linear_260825", 0.8860, "seed 0, LINEAR"),
           ("cross_seeds_long_260824 = frag_seeds8", 0.8904, "8 seed, AREA"),
           ("seed8_matched_linear_260825", 0.9101, "8 seed, LINEAR")]
    cur = [r for r in sr_sum if r[0] == "test" and r[1] == "SR-512"]
    if cur:
        c = float(cur[0][6])
        csd = cur[0][7]
        sds = f"{float(csd):.4f}" if csd not in ("", None) else "nan"
        print(f"    {'통일 파이프라인 (8 seed, LINEAR)':<42} {c:.4f} ± {sds}")
        for nm, val, note in ref:
            print(f"    {nm:<42} {val:.4f}   ({note})  차이 {c - val:+.4f}")
            v.append([f"512_matched_vs_{nm}", val, rd(c, 5), rd(c - val, 5)])

    # (c) 예측 4cm 필터의 FP 감소량
    print("\n(c) 예측 >=4cm 필터가 제거하는 예측 수 (test, matched, 8 seed 합)")
    print(f"    {'model':>9} {'예측 총수':>10} {'>=4cm':>8} {'제거':>8} {'제거%':>7}")
    tot_raw = tot_4 = 0
    for tr in TRAIN_RES:
        lbl, ev = SR_LABEL[tr], int(np.ceil(tr / 32) * 32)
        sel = [c for c in cells if c["dataset"] == "test" and c["model"] == lbl
               and int(c["eval_res"]) == ev and c["family"] == "SR"]
        if not sel:
            continue
        a = sum(int(float(c["n_pred_raw"])) for c in sel)
        b = sum(int(float(c["n_pred_4cm"])) for c in sel)
        tot_raw += a
        tot_4 += b
        print(f"    {lbl:>9} {a:>10} {b:>8} {a - b:>8} "
              f"{100 * (a - b) / a if a else 0:>7.1f}")
        v.append([f"filter_drop_{lbl}", a, b, a - b])
    print(f"    {'합계':>9} {tot_raw:>10} {tot_4:>8} {tot_raw - tot_4:>8} "
          f"{100 * (tot_raw - tot_4) / tot_raw if tot_raw else 0:>7.1f}")
    with open(OUT / f"{PFX}_verify_{STAMP}.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["check", "a", "b", "c"])
        w.writerows(v)


def print_tables(sr_sum, mr_sum, frag_sr, frag_mr, lvl):
    def show(title, rows, cols, keys, idx0):
        print("\n" + "=" * 112)
        print(title)
        print("=" * 112)
        head = f"{'model':>13} {'eval':>5}"
        for c in cols:
            head += f" | {c:>17}"
        print(head)
        for r in rows:
            line = f"{r[1]:>13} {r[3]:>5}"
            for j, (k, p) in enumerate(keys):
                mu, sd = r[idx0 + 3 * j + 1], r[idx0 + 3 * j + 2]
                s = ("nan" if mu == "" else
                     f"{float(mu):.{p}f} ± "
                     f"{'nan' if sd == '' else f'{float(sd):.{p}f}'}")
                line += f" | {s:>17}"
            print(line)

    for ds in DATASETS:
        show(f"[{ds}] (1) SR matched — 8-seed 평균 ± SD",
             [r for r in sr_sum if r[0] == ds],
             [l for _, l, _ in DET_METRICS[:7]],
             [(k, p) for k, _, p in DET_METRICS[:7]], 5)
    for ds in DATASETS:
        # 학습 범위 밖 추론은 라벨에 * 를 붙여 참고용임을 표시
        rows = []
        for r in mr_sum:
            if r[0] != ds:
                continue
            r = list(r)
            if r[-1] == 0:
                r[1] = r[1] + " *"
            rows.append(r)
        show(f"[{ds}] (4) MR — 8-seed 평균 ± SD   (* = 학습 범위 밖 추론, 참고용)",
             rows, [l for _, l, _ in DET_METRICS[:7]],
             [(k, p) for k, _, p in DET_METRICS[:7]], 5)
    FK = [("count_MAE", 2), ("count_R2_1to1", 2), ("ED_MAE_cm", 3),
          ("ED_R2_1to1", 3), ("area_bias_pct", 2), ("ED_KS_D", 3),
          ("ED_wasserstein", 3), ("ratioFrame_MAE", 2)]
    allk = [k for k, _ in FRAG_METRICS]
    for nm, rows in [("SR", frag_sr), ("MR", frag_mr)]:
        for ds in DATASETS:
            sub = [r for r in rows if r[0] == ds]
            if not sub:
                continue
            print("\n" + "=" * 112)
            print(f"[{ds}] (5) 파쇄도 특성 — {nm} (8-seed 평균 ± SD, "
                  f"붕괴 seed 제외)")
            print("=" * 112)
            print(f"{'model':>13} {'eval':>5} {'n':>3}" +
                  "".join(f" | {k:>16}" for k, _ in FK))
            for r in sub:
                line = f"{r[1]:>13} {r[2]:>5} {r[3]:>3}"
                for k, p in FK:
                    j = allk.index(k)
                    mu, sd = r[6 + 3 * j + 1], r[6 + 3 * j + 2]
                    s = ("nan" if mu == "" else
                         f"{float(mu):.{p}f}±"
                         f"{'nan' if sd == '' else f'{float(sd):.{p}f}'}")
                    line += f" | {s:>16}"
                print(line)
    for ds in DATASETS:
        print("\n" + "=" * 112)
        print(f"[{ds}] (3) 검출 수준 — conf 0.5, >=4cm 예측 기준 "
              f"(괄호는 기존 규약 conf 0.25 · 필터 없음)")
        print("=" * 112)
        print(f"{'model':>9} {'det':>8} {'P3%':>14} {'P4%':>14} {'dom':>4} "
              f"| {'기존 P3%':>10} {'기존 dom':>8}")
        for tr in TRAIN_RES:
            a = [r for r in lvl if r[0] == ds and r[2] == tr
                 and abs(r[3] - 0.50) < 1e-9 and r[4] == 1]
            b = [r for r in lvl if r[0] == ds and r[2] == tr
                 and abs(r[3] - 0.25) < 1e-9 and r[4] == 0]
            if not a:
                continue
            a = a[0]
            g = lambda r, i: (f"{float(r[i]):.1f}±{float(r[i + 1]):.1f}"
                              if r[i] != "" and r[i + 1] != "" else "nan")
            print(f"{SR_LABEL[tr]:>9} {float(a[6]):>8.1f} {g(a, 8):>14} "
                  f"{g(a, 10):>14} {a[14]:>4} | "
                  f"{float(b[0][8]) if b else float('nan'):>10.1f} "
                  f"{b[0][14] if b else '-':>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "cross", "mr", "level"])
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--sets", default="test,val")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    global DATASETS
    DATASETS = [s.strip() for s in args.sets.split(",") if s.strip()]
    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 112)
    print("통일 파이프라인 — 보간 INTER_LINEAR, conf 0.5, IoU 0.5 greedy,")
    print("GT·예측 모두 매칭 전 장축 >= 4cm 필터 (대칭), 프레임 50cm")
    print("=" * 112)
    if not args.report:
        D = load_data()
        if args.stage in ("all", "cross"):
            stage_cross(D, seeds)
        if args.stage in ("all", "mr"):
            stage_mr(D, seeds)
        if args.stage in ("all", "level"):
            stage_level(D, seeds)

    cells = load_rows(P_CELL)
    if not cells:
        print("집계할 셀이 없습니다.")
        return
    sr_sum, mr_sum = summarize_det(cells, seeds)
    cross = cross_matrix(cells)
    lvl = summarize_level()
    _, frag_sr, frag_mr = summarize_frag(seeds)
    carry_fps()
    print_tables(sr_sum, mr_sum, frag_sr, frag_mr, lvl)
    verify(cells, sr_sum, cross)
    print(f"\nCSV: {OUT}")


if __name__ == "__main__":
    main()
