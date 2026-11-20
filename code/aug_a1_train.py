"""증강 제거 실험: YOLOv8m-seg matched SR 512/1600 x seed 0-7 x 조건 3개 (48 run).

학습 설정은 YOLOv8m-seg seed 1-7 SR 학습과 동일 (results/training_config.yaml):
aug_train_common.build_train_args + seed, save_period -1, plots False, 같은 가중 샘플러.
바뀌는 것은 조건별 증강 하나뿐:
  mosaic0        : mosaic = 0.0 (close_mosaic 은 기본값 10 그대로; mosaic 이 0 이라 무관)
  scale0         : scale = 0.0
  mosaic0_scale0 : mosaic = 0.0, scale = 0.0
기존 데이터셋 YAML / 라벨은 읽기만 한다 (convert_dataset 호출 안 함).
재개: best.pt 사본(weights/{cond}_R{res}_s{seed}.pt) 이 있으면 건너뜀.
실행 순서: seed 0-2 (seed -> 조건 -> 해상도) 를 먼저, 이어서 seed 3-7 (조건 -> seed -> 해상도).
출력: <config.AUG_DIR>/runs/, weights/, train_manifest.csv
"""
import csv
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402  code/config.py (경로 설정)
import aug_train_common as PC  # noqa: E402

ROOT = config.AUG_DIR

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import WeightedRandomSampler  # noqa: E402
from ultralytics.models.yolo.segment import SegmentationTrainer  # noqa: E402

CONDS = {"mosaic0": dict(mosaic=0.0),
         "scale0": dict(scale=0.0),
         "mosaic0_scale0": dict(mosaic=0.0, scale=0.0)}
ORDER = ([(c, r, s) for s in (0, 1, 2) for c in CONDS for r in (512, 1600)]
         + [(c, r, s) for c in CONDS for s in (3, 4, 5, 6, 7) for r in (512, 1600)])
RUNS = ROOT / "runs"
WDIR = ROOT / "weights"
MAN = ROOT / "train_manifest.csv"
INV = config.R1_DIR / "s0_inventory_runs.csv"   # r1_s0_inventory_repro.py 출력


class WeightedSegTrainer(SegmentationTrainer):
    """YOLOv8m-seg 학습에 쓴 WeightedSegTrainer 와 동일."""

    def __init__(self, *args, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_weights = sample_weights

    def get_dataloader(self, dataset_path, batch_size, rank=0, mode="train"):
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        if mode != "train" or self._sample_weights is None:
            return loader
        ds = loader.dataset
        w_list = [self._sample_weights.get(Path(p).stem, 1.0) for p in ds.im_files]
        w = np.array(w_list, dtype=np.float64)
        w = w / w.sum() * len(w)
        sampler = WeightedRandomSampler(w.tolist(), num_samples=len(w), replacement=True)
        nl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, sampler=sampler,
            num_workers=loader.num_workers, collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory)
        nl.reset = lambda: None
        return nl


def v8_time(res):
    """기준선 v8 seed 1 run 의 100 epoch 누적 시간(초) (예상 시간 환산용)."""
    rd = [r["run_dir"] for r in csv.DictReader(open(INV, encoding="utf-8-sig"))
          if r["family"] == "SR" and int(r["model"]) == res and int(r["seed"]) == 1][0]
    rows = list(csv.DictReader(open(config.RS_ROOT / rd / "results.csv", encoding="utf-8")))
    return float(rows[-1]["time"])


def tag(c, r, s):
    return f"{c}_R{r}_s{s}"


def main():
    WDIR.mkdir(parents=True, exist_ok=True)
    names, weights = PC.compute_sample_weights()
    wd = dict(zip(names, weights))
    first_ratio = None
    for i, (c, res, sd) in enumerate(ORDER, 1):
        dst = WDIR / f"{tag(c, res, sd)}.pt"
        if dst.exists():
            continue
        args = PC.build_train_args(res, RUNS / tag(c, res, sd))
        args.update(seed=sd, save_period=-1, plots=False, **CONDS[c])
        print(f"### [{i}/{len(ORDER)}] {tag(c, res, sd)} 시작 ({CONDS[c]})", flush=True)
        t0 = time.time()
        try:
            tr = WeightedSegTrainer(overrides=args, sample_weights=wd)
            tr.train()
            best = RUNS / tag(c, res, sd) / "train" / "weights" / "best.pt"
            shutil.copy2(best, dst)
            dt = time.time() - t0
            print(f"### {tag(c, res, sd)} 완료 {dt / 60:.1f}분", flush=True)
            if first_ratio is None:
                try:
                    first_ratio = dt / v8_time(res)
                    rest = [(cc, r, s) for cc, r, s in ORDER
                            if not (WDIR / f"{tag(cc, r, s)}.pt").exists()]
                    est = sum(v8_time(r) for _, r, _ in rest) * first_ratio
                    print(f"### 예상: 남은 {len(rest)}개 run 약 {est / 3600:.1f}시간 "
                          f"(첫 run / 기준선 v8 같은 해상도 시간 비 {first_ratio:.2f})",
                          flush=True)
                except Exception as e:  # 예상 시간 실패는 학습과 무관
                    print(f"### 예상 시간 계산 실패: {e!r}", flush=True)
        except Exception as e:
            print(f"### {tag(c, res, sd)} 실패: {e!r}", flush=True)
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()
    write_manifest()
    print("### 학습 루프 종료", flush=True)


def write_manifest():
    rows = []
    for c, res, sd in sorted(ORDER):
        d = RUNS / tag(c, res, sd) / "train"
        rc = d / "results.csv"
        if not rc.exists():
            rows.append([c, res, sd, "", "", "", "missing"])
            continue
        rr = list(csv.DictReader(open(rc, encoding="utf-8")))
        import yaml
        a = yaml.safe_load(open(d / "args.yaml", encoding="utf-8"))
        rows.append([c, res, sd, int(float(rr[-1]["epoch"])), round(float(rr[-1]["time"]) / 60, 2),
                     f"mosaic={a['mosaic']} scale={a['scale']} close_mosaic={a['close_mosaic']}",
                     int((WDIR / f"{tag(c, res, sd)}.pt").exists())])
    with open(MAN, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "res", "seed", "epochs", "train_min", "aug_args", "weights_saved"])
        w.writerows(rows)


if __name__ == "__main__":
    if sys.argv[1:] == ["manifest"]:
        write_manifest()
    else:
        main()
