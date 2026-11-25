"""실험 1 추론: 타일 8조건 + B 전체 프레임(학습 tr, 추론 2tr) + A 전체 프레임 seed 0 재추론 확인.
seed 0 을 모든 조건에서 먼저 돌리고 seed 1-7 을 이어서 돌린다. 캐시가 있으면 건너뜀.
"""
import time

import torch
from ultralytics import YOLO

import r1_x1_common as X

JOBS = []
for sd in X.SEEDS:
    for cond, tr, s, sfull in X.TILE_CONDS:
        JOBS.append(("tile" + cond, tr, s, "tile", sd))
    for cond, tr, s, sfull in X.TILE_CONDS:
        if cond == "B":
            JOBS.append(("fullB", tr, sfull, "full", sd))
    if sd == 0:
        for cond, tr, s, sfull in X.TILE_CONDS:
            if cond == "A":
                JOBS.append(("fullAchk", tr, sfull, "full", sd))


def main():
    D = X.U.load_data()
    for kind, tr, s, mode, sd in JOBS:
        todo = [ds for ds in X.U.DATASETS if not X.cache_path(kind, ds, tr, s, sd).exists()]
        if not todo:
            continue
        model = YOLO(str(X.S.ckpt_of(tr, sd)))
        t0 = time.time()
        for ds in todo:
            cell = {f: X.infer_image(model, D[ds]["img"][f], s, mode) for f in D[ds]["files"]}
            X.save_pkl(cell, X.cache_path(kind, ds, tr, s, sd))
        del model
        torch.cuda.empty_cache()
        print(f"  {kind} R{tr}@{s} s{sd} {time.time() - t0:.1f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
