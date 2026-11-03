"""
리뷰 대응 추가 — NMS 전 후보에서 GT 별 P3 / P4 최고 confidence.

저장된 preds (review_infer_261026.py) 는 NMS 이후 예측이라 NMS 전 후보가 없다.
그래서 같은 전처리·같은 predict 호출로 forward 를 다시 돌리고, NMS 직전의 raw 출력
(앵커별 xywh, score, mask 계수)과 proto 를 가로챈다.
  - 검증: raw 에 기존과 같은 NMS(conf 0.001, iou U.IOU_NMS, max_det 1000)를 다시 걸면
          저장된 preds 의 score 가 그대로 나와야 한다 (이미지마다 대조).

GT 별 계산 (평가 격자 r32, GT 는 U 규약: 원본 장축 >= 4cm -> NEAREST 리사이즈)
  후보      = NMS 전 모든 앵커 (score 하한 없음)
  후보 마스크 = 기존 retina 경로와 동일: scale_boxes(clip) -> process_mask_native
  IoU       = 마스크 IoU (정수 픽셀 수, GPU float32 matmul -> 정확)
  P{3,4,5}_max      : IoU >= 0.5 인 해당 레벨 후보의 최고 score (없으면 0, n_cand 로 구분)
  P{3,4,5}_max_4cm  : 같은 조건 + 후보 장축 >= 4cm (F.major_axis_px, 기존 예측 필터)
  사전 필터 (손실 없음): 마스크 ⊆ 박스 열·행 범위이므로 |박스 ∩ GT| < 0.5|GT| 인 후보는
                       IoU >= 0.5 가 될 수 없다 -> 마스크 계산 생략.
부가: conf 0.5 기존 매칭에서 이 GT 가 매칭됐는지, 매칭 예측의 레벨 (저장 preds 사용).
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import ultralytics.utils.nms as unms
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import ops

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import review_analysis_261026 as R      # noqa: E402
import review_common_261026 as C        # noqa: E402

U, F, S = C.U, C.F, C.U.S
OUT, SFX = C.OUT, C.SFX
RAW_CSV = OUT / f"gt_level_maxconf_{SFX}.csv"
HDR = ["dataset", "train_res", "eval_res", "seed", "file", "gt_idx", "gt_area_px",
       "gt_major_px", "P3_max", "P4_max", "P5_max", "n_cand_P3", "n_cand_P4",
       "n_cand_P5", "P3_max_4cm", "P4_max_4cm", "P5_max_4cm",
       "matched_conf05", "matched_level_conf05"]
CHUNK = 64

# ── raw 가로채기 ──
_STASH = {}
_orig_nms = unms.non_max_suppression


def _nms_stash(prediction, *a, **k):
    p = prediction[0] if isinstance(prediction, (list, tuple)) else prediction
    _STASH["raw"] = p.detach().clone()          # nms 가 in-place 로 xyxy 변환하므로 복제
    return _orig_nms(prediction, *a, **k)


unms.non_max_suppression = _nms_stash
_orig_cr = SegmentationPredictor.construct_results


def _cr_stash(self, preds, img, orig_imgs, protos):
    _STASH["protos"] = protos[0].detach().clone()
    return _orig_cr(self, preds, img, orig_imgs, protos)


SegmentationPredictor.construct_results = _cr_stash


def per_image(model, img, gts, r32, saved_scores):
    _STASH.clear()
    model.predict(img, imgsz=r32, conf=0.999, iou=U.IOU_NMS, retina_masks=True,
                  verbose=False, device=0)
    raw, protos = _STASH["raw"][0], _STASH["protos"]      # (4+1+nm, N), (nm, h, w)
    N = raw.shape[1]
    n3, n4 = (r32 // 8) ** 2, (r32 // 16) ** 2
    assert N == n3 + n4 + (r32 // 32) ** 2, (N, r32)
    lv = torch.full((N,), 5, device=raw.device)
    lv[:n3] = 3
    lv[n3:n3 + n4] = 4

    # 검증: 같은 NMS 를 다시 걸어 저장 score 재현
    out = _orig_nms(raw[None].clone(), 0.001, U.IOU_NMS, None, False, max_det=1000,
                    nc=1)[0]
    re_sc = np.sort(out[:, 4].cpu().numpy())[::-1]
    sv = np.sort(np.asarray(saved_scores, np.float32))[::-1]
    ok = all(np.isclose(re_sc, s, atol=0, rtol=0).any() for s in sv)

    score = raw[4]
    xyxy = ops.xywh2xyxy(raw[:4].T)
    xyxy = ops.scale_boxes((r32, r32), xyxy.clone(), (r32, r32))   # 기존 경로의 clip
    coef = raw[5:].T
    G = len(gts)
    res = np.zeros((G, 3))
    res4 = np.zeros((G, 3))
    ncand = np.zeros((G, 3), int)
    if G == 0:
        return res, ncand, res4, ok
    gt_t = torch.from_numpy(np.stack(gts).astype(np.float32)).to(raw.device)  # (G,H,W)
    gt_area = gt_t.sum((1, 2))
    # 사전 필터: 박스 열·행 범위(보수적으로 넓힘) 안의 GT 픽셀 수
    ii = torch.nn.functional.pad(gt_t.cumsum(1).cumsum(2), (1, 0, 1, 0))  # (G,H+1,W+1)
    x1 = xyxy[:, 0].floor().clamp(0, r32).long()
    y1 = xyxy[:, 1].floor().clamp(0, r32).long()
    x2 = xyxy[:, 2].ceil().clamp(0, r32).long()
    y2 = xyxy[:, 3].ceil().clamp(0, r32).long()
    inbox = (ii[:, y2, x2] - ii[:, y1, x2] - ii[:, y2, x1] + ii[:, y1, x1])  # (G,N)
    cand = (inbox >= 0.5 * gt_area[:, None]).any(0).nonzero().view(-1)
    gflat = gt_t.view(G, -1).T                                    # (P,G)
    best = {}                                     # (g, level) -> 정렬용 (score, idx) 목록
    for c0 in range(0, len(cand), CHUNK):
        idx = cand[c0:c0 + CHUNK]
        m = ops.process_mask_native(protos, coef[idx], xyxy[idx], (r32, r32))
        mf = m.view(len(idx), -1).float()
        inter = mf @ gflat                                        # (n,G) 정확한 정수
        area = mf.sum(1, keepdim=True)
        iou = inter / (area + gt_area[None] - inter)
        ok_m = (iou >= F.IOU_THR) & (area > 0)
        for j, g in zip(*ok_m.nonzero(as_tuple=True)):
            a = int(idx[j])
            best.setdefault((int(g), int(lv[a]) - 3), []).append(
                (float(score[a]), a))
    thr4 = F.MIN_MAJOR_FRAC * r32
    for (g, L), lst in best.items():
        lst.sort(reverse=True)
        ncand[g, L] = len(lst)
        res[g, L] = lst[0][0]
        for s, a in lst:                          # 장축 >= 4cm 인 최고 score 후보
            m = ops.process_mask_native(protos, coef[a:a + 1], xyxy[a:a + 1],
                                        (r32, r32))[0].cpu().numpy()
            if F.major_axis_px(m) >= thr4:
                res4[g, L] = s
                break
    return res, ncand, res4, ok


def run():
    done = {(r["dataset"], int(r["train_res"]), int(r["seed"]))
            for r in U.load_rows(RAW_CSV)}
    D = U.load_data()
    nbad = 0
    for tr in U.TRAIN_RES:
        r32 = C.GRID[tr]
        for sd in U.SEEDS:
            need = [ds for ds in U.DATASETS if (ds, tr, sd) not in done]
            if not need:
                continue
            model = YOLO(str(S.ckpt_of(tr, sd)))
            t0 = time.time()
            for ds in need:
                cell = C.load_cell(ds, tr, sd)["images"]
                rows = []
                for f in sorted(D[ds]["gt"]):
                    g = D[ds]["gt"][f]
                    img_r = cv2.resize(D[ds]["img"][f], (r32, r32),
                                       interpolation=U.INTERP_IMG)
                    gts_r = [cv2.resize(m, (r32, r32), interpolation=U.INTERP_MASK)
                             for m in g["masks"]]
                    im = cell[f]
                    assert im["n_gt"] == len(gts_r)
                    res, nc, res4, ok = per_image(
                        model, img_r, gts_r, r32, [p["score"] for p in im["preds"]])
                    nbad += not ok
                    # 기존 conf 0.5 매칭 결과 (저장 preds)
                    P = R.prep(im)
                    sel = np.where(P["sc"] > 0.5)[0] if len(P["cand"]) else []
                    gt_to = [-1] * len(gts_r)
                    if len(sel):
                        cs = np.isin(P["cand"], sel)
                        idx = np.where(cs)[0]
                        if len(idx):
                            gt_to, _ = C.greedy(P["iou"][idx],
                                                list(P["ar"][P["cand"][idx]]),
                                                P["n"])
                            gt_to = [P["cand"][idx[d]] if d >= 0 else -1
                                     for d in gt_to]
                    for gi, m in enumerate(gts_r):
                        d = gt_to[gi]
                        rows.append([ds, tr, r32, sd, f, gi, int(m.sum()),
                                     round(F.major_axis_px(m), 2)]
                                    + [float(f"{x:.6g}") for x in res[gi]]
                                    + list(nc[gi])
                                    + [float(f"{x:.6g}") for x in res4[gi]]
                                    + [int(d >= 0),
                                       f"P{P['lv'][d]}" if d >= 0 else ""])
                U.append_rows(RAW_CSV, HDR, rows)
            del model
            torch.cuda.empty_cache()
            print(f"  R{tr} s{sd} | {need} {time.time() - t0:.1f}s | "
                  f"NMS 재현 불일치 이미지 누적 {nbad}", flush=True)


# ══════════════ 요약·그림 ══════════════
def summarize():
    rows = U.load_rows(RAW_CSV)
    out = []
    for ds in U.DATASETS:
        for tr in U.TRAIN_RES:
            sub = [r for r in rows if r["dataset"] == ds and int(r["train_res"]) == tr]
            per_seed = {}
            for sd in U.SEEDS:
                ss = [r for r in sub if int(r["seed"]) == sd]
                per_seed[sd] = {k: float(np.median([float(r[k]) for r in ss]))
                                for k in ("P3_max", "P4_max", "P3_max_4cm",
                                          "P4_max_4cm")}
            row = [ds, tr, C.GRID[tr], len(sub)]
            for k in ("P3_max", "P4_max", "P3_max_4cm", "P4_max_4cm"):
                v = np.array([per_seed[s][k] for s in U.SEEDS])
                allv = np.array([float(r[k]) for r in sub])
                q1, q3 = np.percentile(allv, [25, 75])
                row += [round(v.mean(), 4), round(v.std(ddof=1), 4),
                        round(float(np.median(allv)), 4), round(q1, 4), round(q3, 4)]
            row += [round(float(np.mean([float(r["P4_max"]) > float(r["P3_max"])
                                         for r in sub])), 4),
                    round(float(np.mean([int(r["n_cand_P3"]) == 0 for r in sub])), 4),
                    round(float(np.mean([int(r["n_cand_P4"]) == 0 for r in sub])), 4)]
            out.append(row)
    h = ["dataset", "train_res", "eval_res", "n_gt_x_seed"]
    for k in ("P3_max", "P4_max", "P3_max_4cm", "P4_max_4cm"):
        h += [f"{k}_seedmedian_mean", f"{k}_seedmedian_sd", f"{k}_pooled_median",
              f"{k}_pooled_q25", f"{k}_pooled_q75"]
    h += ["frac_P4_gt_P3", "frac_no_P3_cand", "frac_no_P4_cand"]
    R.wcsv(f"gt_level_maxconf_summary_{SFX}.csv", h, out)
    return out


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=200, sharey=True)
    for ax, ds in zip(axes, U.DATASETS):
        rr = [r for r in out if r[0] == ds]
        x = np.arange(len(rr))
        for k, (nm, col) in enumerate([("P3 최고 점수", 4), ("P4 최고 점수", 9)]):
            m = np.array([r[col] for r in rr])
            s = np.array([r[col + 1] for r in rr])
            ax.fill_between(x, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                            color=R.PAL[k], alpha=0.15, lw=0)
            ax.plot(x, m, "-o", color=R.PAL[k], lw=2, ms=6, mec="white", mew=1.2,
                    label=nm)
        ax.axhline(0.5, color=R.INK2, lw=1, ls="--")
        ax.text(-0.3, 0.51, "conf 0.5", color=R.INK2, fontsize=8, va="bottom")
        ax.set_xticks(x, [str(r[2]) for r in rr])
        ax.set_xlabel("학습 = 추론 해상도 (px)", color=R.INK)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{ds}: GT 별 레벨 최고 confidence 중앙값", color=R.INK,
                     fontsize=11, loc="left")
        R.style(ax)
    axes[0].set_ylabel("중앙값 (seed 별 중앙값의 평균 ± SD)", color=R.INK)
    axes[0].legend(frameon=False, fontsize=9, loc="center left",
                   bbox_to_anchor=(0.0, 0.32), labelcolor=R.INK)
    fig.text(0.01, 0.005, "NMS 전 후보 중 GT 와 mask IoU ≥ 0.5 인 후보의 레벨별 최고 "
             "score (해당 후보가 없으면 0)", color=R.INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / f"gt_level_maxconf_{SFX}.png")
    plt.close(fig)


if __name__ == "__main__":
    if "--report" not in sys.argv:
        only = [a for a in sys.argv[1:] if ":" in a]
        if only:                                   # 시험용: 512:0 등 -> 별도 파일
            RAW_CSV = OUT / f"_trial_gt_level_{SFX}.csv"
            tr0 = {int(a.split(":")[0]) for a in only}
            U.TRAIN_RES = [t for t in U.TRAIN_RES if t in tr0]
            U.SEEDS = sorted({int(a.split(":")[1]) for a in only})
        run()
    if not any(":" in a for a in sys.argv[1:]):
        figure(summarize())
