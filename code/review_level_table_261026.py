"""
리뷰 대응 추가 — 검출 수준(P3/P4/P5) 분포 표를 앵커 인덱스 기반 level 로 재작성.

입력: results_review_261026/preds (review_infer_261026.py, NMS 앵커 인덱스 -> level)
대상: conf > 0.5 검출 (STEP 1 과 같은 strict 규약), SR matched 9 해상도 x 8 seed, test/val
  all_4cm : 전체 검출 중 장축 >= 4cm (마스크 장축, 기존 예측 필터 F.major_axis_px)  <- (a)
  all_raw : 전체 검출, 4cm 필터 없음 (기존 filter4cm=0 과 대응)                     <- (a) 참고
  TP      : 기존 매칭(면적 내림차순 greedy, IoU >= 0.5)에서 GT 와 짝지어진 예측       <- (b)
비교 기준: results/unified/unified_4cm_detlevel_260914 (박스 장변 >= 0.08*r32 대용,
           conf 0.5) — filter4cm=1 은 all_4cm, filter4cm=0 은 all_raw 와 대응.
           (비교 기준 파일은 not included in this repository -> 비교 부분은 주석 처리)
요약: seed 별 비율 -> 8 seed 평균 ± SD(n-1). 검출 0 개 seed 는 비율 NaN 으로 제외하고 n 기록.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import review_analysis_261026 as R      # noqa: E402
import review_common_261026 as C        # noqa: E402

U = C.U
CONF = 0.5
SETS = ["all_4cm", "TP", "all_raw"]


def counts_cell(ims):
    cnt = {k: np.zeros(3, int) for k in SETS}
    for f in sorted(ims):
        im = ims[f]
        for p in im["preds"]:
            if p["score"] > CONF:
                cnt["all_raw"][p["level"] - 3] += 1
        P = R.prep(im)
        sel = P["sc"] > CONF
        for L in P["lv"][sel]:
            cnt["all_4cm"][L - 3] += 1
        # TP: match_fast 와 같은 경로로 매칭된 예측의 레벨
        _, _, _, m = R.match_fast(P, CONF, strict=True)
        for _, L in m:
            cnt["TP"][L - 3] += 1
    return cnt


def main():
    seed_rows, per = [], {}
    for ds in U.DATASETS:
        for tr in U.TRAIN_RES:
            for sd in U.SEEDS:
                cnt = counts_cell(C.load_cell(ds, tr, sd)["images"])
                for k in SETS:
                    c = cnt[k]
                    tot = int(c.sum())
                    pct = c / tot * 100 if tot else np.full(3, np.nan)
                    per[(ds, tr, sd, k)] = (c, tot, pct)
                    seed_rows.append([ds, tr, C.GRID[tr], sd, CONF, k, *c, tot,
                                      *[round(x, 3) for x in pct]])
    R.wcsv(f"level_table_seed_{C.SFX}.csv",
           ["dataset", "train_res", "eval_res", "seed", "conf", "set", "det_P3",
            "det_P4", "det_P5", "det_total", "P3_pct", "P4_pct", "P5_pct"],
           seed_rows)

    def agg(vals):
        v = np.array([x for x in vals if np.isfinite(x)], float)
        return (len(v), v.mean() if len(v) else np.nan,
                v.std(ddof=1) if len(v) > 1 else np.nan)

    summ = {}
    rows = []
    for ds in U.DATASETS:
        for tr in U.TRAIN_RES:
            for k in SETS:
                tots = [per[(ds, tr, s, k)][1] for s in U.SEEDS]
                row = [ds, tr, C.GRID[tr], k, round(np.mean(tots), 2),
                       round(np.std(tots, ddof=1), 2)]
                st = []
                for L in range(3):
                    n, m, s = agg([per[(ds, tr, sd, k)][2][L] for sd in U.SEEDS])
                    st.append((m, s))
                    row += [round(m, 2), round(s, 2)]
                dom = ["P3", "P4", "P5"][int(np.nanargmax([x[0] for x in st]))]
                ndom = [int(np.nanargmax(per[(ds, tr, sd, k)][2])) if
                        per[(ds, tr, sd, k)][1] else -1 for sd in U.SEEDS]
                row += [n, dom, sum(d == 0 for d in ndom), sum(d == 1 for d in ndom),
                        sum(d == 2 for d in ndom)]
                summ[(ds, tr, k)] = row
                rows.append(row)
    R.wcsv(f"level_table_summary_{C.SFX}.csv",
           ["dataset", "train_res", "eval_res", "set", "det_total_mean",
            "det_total_sd", "P3_pct_mean", "P3_pct_sd", "P4_pct_mean", "P4_pct_sd",
            "P5_pct_mean", "P5_pct_sd", "n_seeds_valid", "dominant",
            "n_seed_P3_dom", "n_seed_P4_dom", "n_seed_P5_dom"], rows)

    # ── 기존 박스 장변 대용과 비교 (level_table_compare) ──
    # 비교 기준 unified_4cm_detlevel_260914.csv / unified_4cm_detlevel_raw_260914.csv 는
    # not included in this repository -> 생략.


if __name__ == "__main__":
    main()
