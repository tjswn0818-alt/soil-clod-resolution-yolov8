"""2단계 — coarse-clod 면적비: 합산(논문) vs 합집합(픽셀 OR).

논문 코드와 같은 격자: 예측(P0, score > 0.5, 평가격자 장축 >= 4cm)을 원본 격자로
NEAREST 복원, GT(>=4cm)는 원본 격자 마스크. 면적 = 픽셀수 * cpp^2, 프레임 = 50^2 cm^2.
"""
import cv2
import numpy as np
from pycocotools import mask as coco_mask

import r1_common as Q

R, U, C, F = Q.R, Q.U, Q.C, Q.F
FR = F.FRAME_CM ** 2


def ratios(masks, cpp, shape):
    if not masks:
        return 0.0, 0.0
    s = sum(float(m.sum()) for m in masks)
    u = np.zeros(shape, bool)
    for m in masks:
        u |= m.astype(bool)
    return 100 * s * cpp ** 2 / FR, 100 * float(u.sum()) * cpp ** 2 / FR


def main():
    img_rows, sum_rows = [], []
    gt_cache = {}
    for ds in U.DATASETS:
        gt = F.load_gt(F.DATASETS[ds])
        for f, g in gt.items():
            gt_cache[(ds, f)] = ratios(g["masks"], g["cpp"], (g["H"], g["W"]))
        for tr in Q.RESL:
            per_seed = []
            for sd in Q.SEEDS:
                ims = C.load_cell(ds, tr, sd)["images"]
                e = []
                for f in sorted(ims):
                    g, im = gt[f], ims[f]
                    H, W = g["H"], g["W"]
                    pm = [cv2.resize(coco_mask.decode(im["preds"][i]["rle"]), (W, H),
                                     interpolation=cv2.INTER_NEAREST)
                          for i in im["kept_idx"] if im["preds"][i]["score"] > 0.5]
                    ps, pu = ratios(pm, g["cpp"], (H, W))
                    gs, gu = gt_cache[(ds, f)]
                    img_rows.append([ds, tr, Q.LBL[tr], sd, f, len(g["masks"]), len(pm),
                                     round(gs, 5), round(gu, 5), round(ps, 5),
                                     round(pu, 5)])
                    e.append([gs, gu, ps, pu])
                e = np.array(e)
                d_sum, d_uni = e[:, 2] - e[:, 0], e[:, 3] - e[:, 1]
                with np.errstate(invalid="ignore", divide="ignore"):
                    ov_p = np.where(e[:, 2] > 0, (e[:, 2] - e[:, 3]) / e[:, 2], np.nan)
                per_seed.append([np.abs(d_sum).mean(), d_sum.mean(),
                                 np.abs(d_uni).mean(), d_uni.mean(),
                                 np.nanmean(ov_p), np.nanmax(ov_p)])
            v = np.array(per_seed)
            sub = np.array([r[7:] for r in img_rows if r[0] == ds and r[1] == tr],
                           float)
            with np.errstate(invalid="ignore", divide="ignore"):
                ovg = np.where(sub[:, 0] > 0, (sub[:, 0] - sub[:, 1]) / sub[:, 0], np.nan)
                ovp = np.where(sub[:, 2] > 0, (sub[:, 2] - sub[:, 3]) / sub[:, 2], np.nan)
            nimg = len(gt)
            row = [ds, tr, Q.LBL[tr],
                   round(np.nanmean(ovg[:nimg]) * 100, 4),
                   round(np.nanmax(ovg[:nimg]) * 100, 4),
                   round(np.nanmean(ovp) * 100, 4), round(np.nanmax(ovp) * 100, 4)]
            for j in range(4):
                row += [round(v[:, j].mean(), 4), round(v[:, j].std(ddof=1), 4)]
            sum_rows.append(row)
            print(row, flush=True)
    Q.wcsv("s2_area_ratio_per_image.csv",
           ["dataset", "train_res", "eval_res", "seed", "file", "n_GT4", "n_pred4",
            "GT_sum_pct", "GT_union_pct", "Pred_sum_pct", "Pred_union_pct"], img_rows)
    Q.wcsv("s2_area_ratio_summary.csv",
           ["dataset", "train_res", "eval_res", "GT_overlap_mean_pct",
            "GT_overlap_max_pct", "Pred_overlap_mean_pct_imgxseed",
            "Pred_overlap_max_pct_imgxseed",
            "MAE_sum_pp_mean", "MAE_sum_pp_sd", "bias_sum_pp_mean", "bias_sum_pp_sd",
            "MAE_union_pp_mean", "MAE_union_pp_sd", "bias_union_pp_mean",
            "bias_union_pp_sd"], sum_rows)


if __name__ == "__main__":
    main()
