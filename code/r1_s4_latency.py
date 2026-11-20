"""4단계 — 종단 지연시간 (seed 0 SR 9개, test 20장 메모리 상주, batch 1).

구간 (매 구간 경계에서 torch.cuda.synchronize + perf_counter)
  pre   : 원본 BGR -> cv2.resize(r32, INTER_LINEAR; 평가와 동일) -> RGB -> tensor
          -> GPU 전송 -> float/half, /255
  fwd   : net(x)  (Ultralytics predictor 와 같이 Conv+BN fuse 한 모델)
  post  : NMS (conf 0.5, iou 0.7, max_det 300; ultralytics.utils.nms) + 마스크 처리
          (SegmentationPredictor.construct_result 와 같은 함수)
          + 빈 마스크 제거 + 마스크·박스 CPU 전송
          mask_mode = input  : ops.process_mask(..., (r32, r32), upsample=True)
          mask_mode = native : 박스를 원본 크기로 선형 환산 후 ops.process_mask_native(..., (H, W))
                               (평가 파이프라인은 정사각 리사이즈라 letterbox 용 scale_boxes 대신 선형 환산)
  total : pre + fwd + post
warm-up 10회 후 100회, 20장 순환. FP32 / FP16 각각.
"""
import csv
import platform
import subprocess
import time

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import nms, ops

import r1_common as Q

U, F, S = Q.U, Q.F, Q.U.S
WARMUP, N_ITER, CONF, IOU, MAXDET = 10, 100, 0.5, 0.7, 300


def env_info():
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True)
    cpu = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor).Name"],
                         capture_output=True, text=True).stdout.strip()
    return [["cpu", cpu or platform.processor()],
            ["gpu_driver", smi.stdout.strip()],
            ["torch", torch.__version__], ["cuda", torch.version.cuda],
            ["cudnn", torch.backends.cudnn.version()],
            ["ultralytics", ultralytics.__version__],
            ["python", platform.python_version()], ["os", platform.platform()]]


def run_once(net, im, r32, half, mode):
    dev = "cuda"
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x = cv2.resize(im, (r32, r32), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(dev)
    x = (x.half() if half else x.float()) / 255.0
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    out = net(x)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    p0 = out[0]
    protos = p0[1] if isinstance(p0, tuple) else out[1]
    det = nms.non_max_suppression(p0, CONF, IOU, None, False, max_det=MAXDET, nc=1)[0]
    n = 0
    if det.shape[0]:
        if mode == "input":
            masks = ops.process_mask(protos[0], det[:, 6:], det[:, :4], (r32, r32),
                                     upsample=True)
        else:
            H, W = im.shape[:2]
            b = det[:, :4].clone()
            b[:, [0, 2]] *= W / r32
            b[:, [1, 3]] *= H / r32
            masks = ops.process_mask_native(protos[0], det[:, 6:], b, (H, W))
        keep = masks.amax((-2, -1)) > 0
        if not bool(keep.all()):
            det, masks = det[keep], masks[keep]
        masks = masks.cpu()
        det = det.cpu()
        n = int(det.shape[0])
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    return [(t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3, n]


def main():
    Q.wcsv("s4_environment.csv", ["item", "value"], env_info())
    files = sorted((F.DATASETS["test"]).glob("*.jpg"))
    imgs = [cv2.imread(str(f)) for f in files]
    assert len(imgs) == 20
    fps_old = {int(r["res"]): r for r in U.load_rows(U.RES / "fps_sr_matched_260825.csv")}
    raw, rows = [], []
    for tr in Q.RESL:
        r32 = Q.C.GRID[tr]
        for half in (False, True):
            m = YOLO(str(S.ckpt_of(tr, 0)))
            net = m.model.fuse(verbose=False).to("cuda").eval()
            if half:
                net = net.half()
            for mode in ("input", "native"):
                with torch.inference_mode():
                    for k in range(WARMUP):
                        run_once(net, imgs[k % 20], r32, half, mode)
                    res = [run_once(net, imgs[k % 20], r32, half, mode)
                           for k in range(N_ITER)]
                a = np.array(res)
                prec = "FP16" if half else "FP32"
                for k, r in enumerate(res):
                    raw.append([tr, r32, prec, mode, k, files[k % 20].name]
                               + [round(v, 4) for v in r[:4]] + [r[4]])
                o = [tr, r32, prec, mode]
                for j in range(4):
                    o += [round(a[:, j].mean(), 3), round(a[:, j].std(ddof=1), 3),
                          round(float(np.median(a[:, j])), 3)]
                o += [round(1000 / a[:, 3].mean(), 2), round(a[:, 4].mean(), 2)]
                fo = fps_old.get(tr, {})
                o += [fo.get("pure_mean_ms", ""), fo.get("pure_FPS", ""),
                      fo.get("full_mean_ms", ""), fo.get("full_FPS", "")]
                rows.append(o)
                print(o[:4], "pre", o[4], "fwd", o[7], "post", o[10], "tot", o[13],
                      flush=True)
            del m, net
            torch.cuda.empty_cache()
    Q.wcsv("s4_latency_raw.csv", ["train_res", "r32", "precision", "mask_mode", "iter",
                                  "file", "pre_ms", "fwd_ms", "post_ms", "total_ms",
                                  "n_det"], raw)
    Q.wcsv("s4_latency_summary.csv",
           ["train_res", "r32", "precision", "mask_mode",
            "pre_mean", "pre_sd", "pre_median", "fwd_mean", "fwd_sd", "fwd_median",
            "post_mean", "post_sd", "post_median", "total_mean", "total_sd",
            "total_median", "total_FPS", "n_det_mean",
            "paper_pure_forward_ms", "paper_pure_FPS", "paper_full_predict_ms",
            "paper_full_FPS"], rows)


if __name__ == "__main__":
    main()
