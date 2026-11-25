"""처리시간용 GPU 병합(conf 0.5 NMS, max_det 300, process_mask) 최종 개수 vs 평가용 RLE 병합(merge_base)
최종 예측 수, seed 0 test 20장 이미지별 대조. 출력 x1_latency_equiv.csv"""
import csv
from collections import defaultdict

import numpy as np

import r1_x1_common as X

res = X.load_pkl(X.OUT / "cache" / "x1_eval_raw.pkl")
ev = {}
for key, out, files in res:
    kind, cond, tr, s, ds, sd = key
    if kind == "merge" and ds == "test" and sd == 0:
        ev[(cond, tr)] = dict(zip(files, out["base"][:, 1]))
lat = defaultdict(dict)
for r in csv.DictReader(open(X.OUT / "x1_latency_raw.csv", encoding="utf-8-sig")):
    if r["mode"] != "fullframe" and int(r["iter"]) < 20:
        lat[(r["condition"], int(r["train_res"]), r["mode"])][r["file"]] = int(r["n_final"])
rows = []
for (cond, tr, mode), d in sorted(lat.items()):
    e = ev[(cond, tr)]
    diff = np.array([d[f] - e[f] for f in sorted(d)])
    rows.append([cond, tr, mode, len(diff), int((diff == 0).sum()), int(np.abs(diff).max()),
                 sum(d.values()), sum(e[f] for f in d)])
X.wcsv("x1_latency_equiv.csv", ["condition", "train_res", "mode", "n_img", "n_img_equal",
                                "max_abs_diff", "n_final_latency_sum", "n_final_eval_sum"], rows)
for r in rows:
    print(r)
