"""증강 제거 실험(aug_a1_train.py) 공용: YOLOv8m-seg 학습과 같은 인자와 가중 샘플링.

YOLOv8m-seg 학습 스크립트의 build_train_args / compute_sample_weights 를 옮긴 것
(값은 results/training_config.yaml 과 동일). 데이터 경로는 config.RS_ROOT 기준:
  <RES_SWEEP_ROOT>/data/pixel_yolo_split/dataset.yaml, .../train/labels/*.txt (YOLO-seg)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402  code/config.py (경로 설정)

YOLO_BASE = config.RS_ROOT / "data" / "pixel_yolo_split"
YOLO_YAML = YOLO_BASE / "dataset.yaml"
LABEL_DIR = YOLO_BASE / "train" / "labels"     # 가중치는 학습셋만

PX_PER_CM = 44.0
MIN_MAJOR_PX = 176.0                 # 4cm * 44px/cm
REF_SIDE = PX_PER_CM * 50.0          # 2200px 기준 격자
LARGE_W = 3.0
SMALL_W = 0.3
EPOCHS = 100
BATCH = 4
MODEL = "yolov8m-seg.pt"


def major_axis_from_yolo_poly(coords_norm, img_w, img_h):
    pts = np.array(coords_norm, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= img_w
    pts[:, 1] *= img_h
    p = pts.astype(np.int32)
    if len(p) < 5:
        r = cv2.minAreaRect(p)
        return max(r[1]) if r[1][0] > 0 else 0.0
    try:
        return max(cv2.fitEllipse(p)[1])
    except Exception:
        r = cv2.minAreaRect(p)
        return max(r[1]) if r[1][0] > 0 else 0.0


def compute_sample_weights():
    """장축을 REF_SIDE(2200px) 기준 절대 px 로 환산 -> MIN_MAJOR_PX 와 비교."""
    names, weights = [], []
    for lf in sorted(LABEL_DIR.glob("*.txt")):
        lines = lf.read_text(encoding="utf-8").strip().splitlines()
        names.append(lf.stem)
        if not lines:
            weights.append(SMALL_W)
            continue
        large = total = 0
        for line in lines:
            p = line.strip().split()
            if len(p) < 7:
                continue
            if major_axis_from_yolo_poly(list(map(float, p[1:])), REF_SIDE, REF_SIDE) >= MIN_MAJOR_PX:
                large += 1
            total += 1
        weights.append(SMALL_W if total == 0 else SMALL_W + (LARGE_W - SMALL_W) * (large / total))
    return names, weights


def build_train_args(imgsz, project, name="train"):
    return dict(
        model=MODEL,
        data=str(YOLO_YAML),
        epochs=EPOCHS,
        imgsz=imgsz,
        batch=BATCH,
        project=str(project),
        name=name,
        exist_ok=True,
        device=0,
        workers=0,
        val=True,
        save=True,
        save_period=50,
        plots=True,
        verbose=True,
        # 증강
        fliplr=0.5,
        flipud=0.5,
        degrees=90,
        hsv_h=0.03,
        hsv_s=0.3,
        hsv_v=0.2,
        scale=0.4,
    )
