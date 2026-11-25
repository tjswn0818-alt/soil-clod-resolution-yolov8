"""GPU 병합 로직 검증: 평가 캐시(seed 0 test)의 score > 0.5 타일 마스크(RLE 디코드)를 r1_x1_latency.merge 에
그대로 넣어 최종 개수가 평가용 RLE 병합(merge_base)과 같은지 이미지별 대조. 출력 x1_latency_logic_check.csv"""
import numpy as np
import torch
from pycocotools import mask as coco_mask

import r1_x1_common as X
import r1_x1_latency as L

res = X.load_pkl(X.OUT / "cache" / "x1_eval_raw.pkl")
ev = {}
for key, out, files in res:
    kind, cond, tr, s, ds, sd = key
    if kind == "merge" and ds == "test" and sd == 0:
        ev[(cond, tr)] = dict(zip(files, out["base"][:, 1]))
rows = []
for cond, tr, s, sf in X.TILE_CONDS:
    cell = X.load_pkl(X.cache_path("tile" + cond, "test", tr, s, 0))
    eq, tot_g, tot_e = 0, 0, 0
    for f in sorted(cell):
        tiles = []
        for recs in cell[f]["tiles"]:
            q = [r for r in recs if r["score"] > X.T_SCORE]
            if q:
                m = torch.from_numpy(np.stack([coco_mask.decode(r["rle"]) for r in q])).cuda()
                d = torch.zeros((len(q), 6), device="cuda")
                d[:, 4] = torch.tensor([r["score"] for r in q], device="cuda")
            else:
                m = torch.zeros((0, s, s), dtype=torch.uint8, device="cuda")
                d = torch.zeros((0, 6), device="cuda")
            tiles.append((d, m))
        g = L.merge(tiles, s)
        eq += int(g == ev[(cond, tr)][f])
        tot_g += g
        tot_e += int(ev[(cond, tr)][f])
    rows.append([cond, tr, s, len(cell), eq, tot_g, tot_e])
    print(rows[-1], flush=True)
X.wcsv("x1_latency_logic_check.csv", ["condition", "train_res", "tile_size", "n_img", "n_img_equal",
                                      "n_final_gpu_merge", "n_final_eval_merge"], rows)
