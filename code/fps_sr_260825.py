"""
SR 모델 추론 FPS 측정 (bilinear 전처리, test 세트).

측정 A) matched: SR 9개(seed 0)를 각각 자기 학습 해상도로 추론
측정 B) speed–recall plot (legacy): SR-736 을 eval 1600/1280/1024/736/512 로 추론

프로토콜
  - 입력: test 20장을 cv2.INTER_LINEAR 로 r32 x r32 리사이즈 (타이밍 구간 밖 = 데이터 준비)
  - warmup 10회 -> torch.cuda.synchronize() -> 본 측정 100회 (20장을 5바퀴)
  - 매 반복 전후 cuda.synchronize() + time.perf_counter()
  - batch = 1, FP32

두 가지 지표를 함께 낸다 (해석이 다르므로 반드시 구분해 인용할 것)
  pure_forward : backbone+head forward 만. 전/후처리·NMS 제외. (기존 eval_cell FPS 와 동일 규약)
  full_predict : ultralytics model.predict 전체 (정규화·forward·NMS·마스크 후처리 포함,
                 conf 0.5, retina_masks=True — 평가와 동일 설정). 실제 파이프라인에 가깝다.

출력
  results/fps_sr_matched_260825.csv
  results/fps_sr736_evalsweep_260825.csv
  figures/speed_recall_legacy_260825.png   (speed–recall plot, legacy)
"""
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_pixel_final as E  # noqa: E402

STAMP = "260825"
LADDER = [1600, 1280, 1024, 736, 512, 448, 384, 256, 128]
SWEEP_EVALS = [1600, 1280, 1024, 736, 512]
SWEEP_MODEL = 736
TEST_DIR = E.PIXEL_ROOT / "18.v1i.coco-segmentation" / "train"
WARMUP = 10
N_ITER = 100
CONF = 0.5
INTERP = cv2.INTER_LINEAR


def r32_of(r):
    return int(np.ceil(r / 32) * 32)


def prep(images, r32):
    """bilinear 로 리사이즈된 BGR 리스트 (타이밍 구간 밖)."""
    return [cv2.resize(im, (r32, r32), interpolation=INTERP) for im in images]


def bench_pure(model, imgs_r, device="cuda"):
    net = model.model.to(device).eval()
    ts = [torch.from_numpy(i[:, :, ::-1].copy()).permute(2, 0, 1).float()
          .div(255).unsqueeze(0).to(device) for i in imgs_r]
    with torch.inference_mode():
        for k in range(WARMUP):
            net(ts[k % len(ts)])
        torch.cuda.synchronize()
        times = []
        for k in range(N_ITER):
            t = ts[k % len(ts)]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            net(t)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    del ts
    torch.cuda.empty_cache()
    a = np.array(times) * 1000.0
    return a


def bench_predict(model, imgs_r, r32):
    for k in range(WARMUP):
        model.predict(imgs_r[k % len(imgs_r)], imgsz=r32, conf=CONF,
                      retina_masks=True, verbose=False, device=0)
    torch.cuda.synchronize()
    times = []
    for k in range(N_ITER):
        im = imgs_r[k % len(imgs_r)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(im, imgsz=r32, conf=CONF, retina_masks=True,
                      verbose=False, device=0)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    torch.cuda.empty_cache()
    a = np.array(times) * 1000.0
    return a


def stats(a):
    return dict(mean_ms=a.mean(), sd_ms=a.std(ddof=1), med_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a, 95)), fps=1000.0 / a.mean())


def load_recall(ds="test"):
    """(train,eval)->Recall. 원래 중간 파일 cross9_long_linear_260825.csv 에서 읽었으나
    not included in this repository -> 빈 dict (Recall_test/Recall_val 열은 빈칸으로 남음)."""
    return {}


def main():
    import ultralytics
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 96)
    print(f"GPU: {dev} | torch {torch.__version__} | ultralytics {ultralytics.__version__}")
    print(f"프로토콜: batch=1, FP32, warmup {WARMUP}회 + 본측정 {N_ITER}회, "
          f"전처리 bilinear, conf {CONF}, retina_masks=True")
    print("=" * 96)

    files = sorted(TEST_DIR.glob("*.jpg"))
    images = [cv2.imread(str(f)) for f in files]
    print(f"test 세트 {len(images)}장 로드\n")

    rec = load_recall("test")
    rec_val = load_recall("val")

    # ── A) matched ──
    rowsA = []
    print("=" * 96)
    print("=== A) matched — 각 SR 모델을 자기 해상도로 추론 ===")
    print("=" * 96)
    print(f"{'res':>6} {'r32':>6} | {'pure ms':>9} {'±sd':>7} {'pure FPS':>9} | "
          f"{'full ms':>9} {'±sd':>7} {'full FPS':>9} | {'Recall':>7}")
    for r in LADDER:
        ck = E.sr_ckpt(r)
        if not ck.exists():
            print(f"{r:>6}  가중치 없음")
            continue
        rr = r32_of(r)
        imgs_r = prep(images, rr)
        model = YOLO(str(ck))
        sp = stats(bench_pure(model, imgs_r))
        sf = stats(bench_predict(model, imgs_r, rr))
        del model
        torch.cuda.empty_cache()
        R = rec.get((r, r), float("nan"))
        print(f"{r:>6} {rr:>6} | {sp['mean_ms']:>9.2f} {sp['sd_ms']:>7.2f} "
              f"{sp['fps']:>9.1f} | {sf['mean_ms']:>9.2f} {sf['sd_ms']:>7.2f} "
              f"{sf['fps']:>9.1f} | {R:>7.4f}")
        rowsA.append([r, rr, round(E.MIN_MAJOR_FRAC * rr, 2),
                      round(sp["mean_ms"], 3), round(sp["sd_ms"], 3),
                      round(sp["med_ms"], 3), round(sp["p95_ms"], 3), round(sp["fps"], 2),
                      round(sf["mean_ms"], 3), round(sf["sd_ms"], 3),
                      round(sf["med_ms"], 3), round(sf["p95_ms"], 3), round(sf["fps"], 2),
                      round(R, 4) if R == R else "",
                      round(rec_val.get((r, r), float("nan")), 4)
                      if rec_val.get((r, r), float("nan")) == rec_val.get((r, r), float("nan")) else ""])

    pA = E.RESULTS / f"fps_sr_matched_{STAMP}.csv"
    with open(pA, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["res", "grid_r32", "thr_4cm_px",
                    "pure_mean_ms", "pure_sd_ms", "pure_med_ms", "pure_p95_ms", "pure_FPS",
                    "full_mean_ms", "full_sd_ms", "full_med_ms", "full_p95_ms", "full_FPS",
                    "Recall_test", "Recall_val"])
        w.writerows(rowsA)

    # ── B) speed–recall plot (legacy): SR-736 x eval 스윕 ──
    rowsB = []
    print("\n" + "=" * 96)
    print(f"=== B) speed–recall plot (legacy) — SR-{SWEEP_MODEL} 모델을 여러 eval 해상도로 추론 ===")
    print("=" * 96)
    print(f"{'eval':>6} {'r32':>6} | {'pure ms':>9} {'pure FPS':>9} | "
          f"{'full ms':>9} {'full FPS':>9} | {'R_test':>7} {'R_val':>7}")
    ck = E.sr_ckpt(SWEEP_MODEL)
    for ev in SWEEP_EVALS:
        rr = r32_of(ev)
        imgs_r = prep(images, rr)
        model = YOLO(str(ck))
        sp = stats(bench_pure(model, imgs_r))
        sf = stats(bench_predict(model, imgs_r, rr))
        del model
        torch.cuda.empty_cache()
        Rt = rec.get((SWEEP_MODEL, ev), float("nan"))
        Rv = rec_val.get((SWEEP_MODEL, ev), float("nan"))
        print(f"{ev:>6} {rr:>6} | {sp['mean_ms']:>9.2f} {sp['fps']:>9.1f} | "
              f"{sf['mean_ms']:>9.2f} {sf['fps']:>9.1f} | {Rt:>7.4f} {Rv:>7.4f}")
        rowsB.append([SWEEP_MODEL, ev, rr,
                      round(sp["mean_ms"], 3), round(sp["sd_ms"], 3), round(sp["fps"], 2),
                      round(sf["mean_ms"], 3), round(sf["sd_ms"], 3), round(sf["fps"], 2),
                      round(Rt, 4), round(Rv, 4)])

    pB = E.RESULTS / f"fps_sr736_evalsweep_{STAMP}.csv"
    with open(pB, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["train_res", "eval_res", "grid_r32",
                    "pure_mean_ms", "pure_sd_ms", "pure_FPS",
                    "full_mean_ms", "full_sd_ms", "full_FPS",
                    "Recall_test", "Recall_val"])
        w.writerows(rowsB)

    print(f"\n저장: {pA}")
    print(f"저장: {pB}")
    plot(rowsA, rowsB)


def plot(rowsA, rowsB):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # 좌: matched 속도-정확도
    ax = axes[0]
    x = [r[12] for r in rowsA]                 # full_FPS
    y = [r[13] for r in rowsA if r[13] != ""]  # Recall_test
    lab = [r[0] for r in rowsA]
    ax.plot(x, y, "o-", color="#4C78A8", lw=1.6, ms=7)
    for xi, yi, li in zip(x, y, lab):
        ax.annotate(f"{li}", (xi, yi), textcoords="offset points",
                    xytext=(6, 5), fontsize=9)
    ax.set_xlabel("FPS (full predict pipeline)")
    ax.set_ylabel("Recall (test)")
    ax.set_title("A) matched: each SR model at its own resolution", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    # 우: SR-736 eval 스윕
    ax = axes[1]
    xb = [r[8] for r in rowsB]      # full_FPS
    yb = [r[9] for r in rowsB]      # Recall_test
    yv = [r[10] for r in rowsB]     # Recall_val
    ax.plot(xb, yb, "o-", color="#F58518", lw=1.8, ms=8, label="test")
    ax.plot(xb, yv, "s--", color="#54A24B", lw=1.5, ms=7, label="val")
    for xi, yi, li in zip(xb, yb, [r[1] for r in rowsB]):
        ax.annotate(f"{li}px", (xi, yi), textcoords="offset points",
                    xytext=(6, 6), fontsize=9)
    ax.set_xlabel("FPS (full predict pipeline)")
    ax.set_ylabel("Recall")
    ax.set_title(f"B) SR-{SWEEP_MODEL} across inference resolutions", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)

    fig.suptitle("Speed-accuracy trade-off (RTX 4090, batch=1, FP32, bilinear)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = E.FIGURES / f"speed_recall_legacy_{STAMP}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
