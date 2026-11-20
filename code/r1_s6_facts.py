"""6단계 — 사실 확인 (읽기 전용).
(1) 영상 원본 크기 (json + 실제 파일), 장변 < 1600 인 영상 수
(2) 경계에 닿은 GT (마스크가 0행/마지막행/0열/마지막열 픽셀 포함), 전체와 >= 4cm
(3) GT(>=4cm) 장축 분포, 4.0–4.4 cm 개수
"""
import json

import cv2
import numpy as np

import r1_common as Q

F, U = Q.F, Q.U
SPLITS = {"train": F.PIXEL_ROOT / "pixel_split" / "train",
          "val": F.DATASETS["val"], "test": F.DATASETS["test"]}


def main():
    size_rows, size_sum, edge_rows, maj_rows, maj_sum = [], [], [], [], []
    for sp, d in SPLITS.items():
        coco = json.load(open(d / "_annotations.coco.json", encoding="utf-8"))
        Ws, Hs = [], []
        for im in coco["images"]:
            a = cv2.imread(str(d / im["file_name"]))
            h, w = a.shape[:2]
            size_rows.append([sp, im["file_name"], im["width"], im["height"], w, h,
                              int((w, h) == (im["width"], im["height"]))])
            Ws.append(w)
            Hs.append(h)
        Ws, Hs = np.array(Ws), np.array(Hs)
        L = np.maximum(Ws, Hs)
        size_sum.append([sp, len(Ws), Ws.min(), int(np.median(Ws)), Ws.max(), Hs.min(),
                         int(np.median(Hs)), Hs.max(), L.min(), int(np.median(L)), L.max(),
                         int((L < 1600).sum()), int((Ws != Hs).sum())])
        G = Q.load_gt_all(d)
        n_all = n_edge = n_big = n_big_edge = 0
        majs = []
        for f, g in G.items():
            for m, big, mj in zip(g["masks"], g["big"], g["major_cm"]):
                e = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
                n_all += 1
                n_edge += e
                if big:
                    n_big += 1
                    n_big_edge += e
                    majs.append(mj)
                    maj_rows.append([sp, f, round(mj, 4), int(e)])
        edge_rows.append([sp, n_all, n_edge, round(n_edge / n_all, 4), n_big, n_big_edge,
                          round(n_big_edge / n_big, 4) if n_big else ""])
        mj = np.array(majs)
        maj_sum.append([sp, len(mj), *np.round(np.percentile(mj, [0, 10, 25, 50, 75, 90, 100]), 3),
                        int(((mj >= 4.0) & (mj < 4.4)).sum()),
                        round(float(((mj >= 4.0) & (mj < 4.4)).mean()), 4),
                        int(((mj >= 4.0) & (mj < 5.0)).sum())])
    Q.wcsv("s6_image_sizes.csv", ["split", "file", "json_w", "json_h", "file_w", "file_h",
                                  "json_eq_file"], size_rows)
    Q.wcsv("s6_image_size_summary.csv",
           ["split", "n", "W_min", "W_median", "W_max", "H_min", "H_median", "H_max",
            "long_min", "long_median", "long_max", "n_long_lt1600", "n_nonsquare"], size_sum)
    Q.wcsv("s6_gt_edge_touch.csv",
           ["split", "n_GT_all", "n_edge_all", "ratio_all", "n_GT_ge4cm", "n_edge_ge4cm",
            "ratio_ge4cm"], edge_rows)
    Q.wcsv("s6_gt4cm_major_detail.csv", ["split", "file", "major_cm", "touches_edge"],
           maj_rows)
    Q.wcsv("s6_gt4cm_major_summary.csv",
           ["split", "n", "min", "p10", "p25", "median", "p75", "p90", "max",
            "n_4.0_4.4", "frac_4.0_4.4", "n_4.0_5.0"], maj_sum)

    for r in size_sum + edge_rows + maj_sum:
        print(r)


if __name__ == "__main__":
    main()
