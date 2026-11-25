"""경로 설정. 데이터·가중치 위치는 환경변수로 지정한다.

RES_SWEEP_ROOT : 작업 루트. weights/ (sr, sr_seed, mr), data/pixel_yolo_split/, logs/ 를 두고
                 results/ 아래에 결과 CSV 를 쓴다. 기본값: 저장소 루트
PIXEL_DATA_ROOT: 데이터셋 루트. 18.v1i.coco-segmentation/train (test), pixel_split/val (val).
                 기본값: <RES_SWEEP_ROOT>/datasets
EXTRA_TEST_DATA_ROOT: 추가 시험셋(20장, COCO segmentation) 폴더 (_annotations.coco.json 과 이미지).
                 사진은 공개하지 않는다. 기본값: <PIXEL_DATA_ROOT>/extra_test20
학습 run 폴더(results.csv, args.yaml, last.pt)는 <RES_SWEEP_ROOT>/runs, <RES_SWEEP_ROOT>/outputs_pixel_AB
아래에 둔다. s0_inventory_runs.csv 의 run_dir, ckpt_copy 는 RES_SWEEP_ROOT 기준 상대경로.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RS_ROOT = Path(os.environ.get("RES_SWEEP_ROOT", REPO_ROOT)).resolve()
PIXEL_ROOT = Path(os.environ.get("PIXEL_DATA_ROOT", RS_ROOT / "datasets")).resolve()
REVIEW_DIR = RS_ROOT / "results" / "results_review_261026"
R1_DIR = RS_ROOT / "results" / "results_review_r1_261116"         # r1_*.py 출력
AUG_DIR = RS_ROOT / "results" / "results_review_r1_aug_261119"     # aug_*.py 출력 (runs/, weights/ 포함)
R1_EXTRA_DIR = RS_ROOT / "results" / "results_review_r1_extra_261123"   # r1_x1_*.py, r1_x2_*.py 출력
EXTRA_TEST_ROOT = Path(os.environ.get("EXTRA_TEST_DATA_ROOT", PIXEL_ROOT / "extra_test20")).resolve()
RUN_ROOTS = [RS_ROOT / "runs", RS_ROOT / "outputs_pixel_AB"]


def rel(p):
    """RES_SWEEP_ROOT 기준 상대경로 (결과 CSV 에 절대경로를 남기지 않음)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(RS_ROOT).as_posix()
    except ValueError:
        return p.name
