"""경로 설정. 데이터·가중치 위치는 환경변수로 지정한다.

RES_SWEEP_ROOT : 작업 루트. weights/ (sr, sr_seed, mr), data/pixel_yolo_split/, logs/ 를 두고
                 results/ 아래에 결과 CSV 를 쓴다. 기본값: 저장소 루트
PIXEL_DATA_ROOT: 데이터셋 루트. 18.v1i.coco-segmentation/train (test), pixel_split/val (val).
                 기본값: <RES_SWEEP_ROOT>/datasets
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RS_ROOT = Path(os.environ.get("RES_SWEEP_ROOT", REPO_ROOT)).resolve()
PIXEL_ROOT = Path(os.environ.get("PIXEL_DATA_ROOT", RS_ROOT / "datasets")).resolve()
REVIEW_DIR = RS_ROOT / "results" / "results_review_261026"
