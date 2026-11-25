"""실험 1 처리시간 (r1_s4_latency.run_once 방식).

공통: seed 0, FP32, mask_mode input, warm-up 10 + 100회, test 20장 메모리 상주·순환,
      구간 경계마다 torch.cuda.synchronize + perf_counter. NMS conf 0.5, iou 0.7, max_det 300.
전체 프레임: r1_s4_latency.run_once(net, im, 2s, FP32, "input") 그대로 (pre / fwd / post / total).
타일 (3x3, T7.tiles_of):
  pre   : 9개 타일 crop -> cv2.resize(s, INTER_LINEAR) -> RGB -> tensor -> GPU -> float /255
  fwd   : net(x)
  post  : NMS + ops.process_mask(..., (s, s), upsample=True) + 빈 마스크 제거 (GPU 상)
  merge : 5) 내부 경계 2 px 규칙 (GPU) -> 4) 캔버스 오프셋 기준 타일 쌍 겹침 영역 교집합 (GPU, float32 matmul)
          -> 6) score 내림차순 greedy mask-NMS (IoU >= 0.5, CPU, n x n 행렬)
          -> 남은 마스크·박스 CPU 전송 -> 7) 장축 >= 0.08 * 2s (F.major_axis_px, CPU)
  total : pre + fwd + post + merge
  seq   : batch 1 x 9회 순차 (pre/fwd/post 를 타일마다 재고 합산)
  batch : batch 9 x 1회
  전체 프레임 run_once 는 마스크 CPU 전송을 post 에 포함, 장축 필터 없음 (s4 와 동일).
  타일 방식은 CPU 전송과 장축 필터가 merge 에 들어감.
"""
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import nms, ops

import r1_x1_common as X
import r1_s4_latency as L4

F, S, T7 = X.F, X.S, X.T7
WARMUP, N_ITER = L4.WARMUP, L4.N_ITER
CONF, IOU, MAXDET = L4.CONF, L4.IOU, L4.MAXDET
MARGIN = 2


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def to_tensor(imgs):
    x = np.stack([im[:, :, ::-1] for im in imgs])
    x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).to("cuda")
    return x.float() / 255.0


def post_one(p0b, protos, s):
    det = nms.non_max_suppression(p0b, CONF, IOU, None, False, max_det=MAXDET, nc=1)
    out = []
    for i, d in enumerate(det):
        if d.shape[0]:
            m = ops.process_mask(protos[i], d[:, 6:], d[:, :4], (s, s), upsample=True)
            keep = m.amax((-2, -1)) > 0
            if not bool(keep.all()):
                d, m = d[keep], m[keep]
        else:
            m = torch.zeros((0, s, s), dtype=torch.uint8, device="cuda")
        out.append((d, m))
    return out


def merge(tiles, s):
    """tiles: [(det, masks)] x 9 (GPU). 반환 최종 개수."""
    half = s // 2
    keep_t, sc, offs = [], [], []
    for k, (d, m) in enumerate(tiles):
        row, col = k // 3, k % 3
        if m.shape[0]:
            cols = m.amax(1) > 0                      # (n, s) x 방향 존재
            rows = m.amax(2) > 0
            ar = torch.arange(s, device="cuda")
            big = s + 10
            xmin = torch.where(cols, ar, big).amin(1)
            xmax = torch.where(cols, ar, -1).amax(1)
            ymin = torch.where(rows, ar, big).amin(1)
            ymax = torch.where(rows, ar, -1).amax(1)
            ok = torch.ones(m.shape[0], dtype=torch.bool, device="cuda")
            if col > 0:
                ok &= xmin > MARGIN
            if col < 2:
                ok &= (s - 1 - xmax) > MARGIN
            if row > 0:
                ok &= ymin > MARGIN
            if row < 2:
                ok &= (s - 1 - ymax) > MARGIN
            d, m = d[ok], m[ok]
        keep_t.append((d, m))
        offs.append((col * half, row * half))
    sizes = [m.shape[0] for _, m in keep_t]
    n = sum(sizes)
    if n == 0:
        return 0
    base = np.cumsum([0] + sizes)
    inter = torch.zeros((n, n), dtype=torch.float32, device="cuda")
    areas = torch.cat([m.flatten(1).sum(1).float() for _, m in keep_t])
    for k in range(9):
        for l in range(k, 9):
            if not sizes[k] or not sizes[l]:
                continue
            (xk, yk), (xl, yl) = offs[k], offs[l]
            x0, x1 = max(xk, xl), min(xk, xl) + s
            y0, y1 = max(yk, yl), min(yk, yl) + s
            if x0 >= x1 or y0 >= y1:
                continue
            a = keep_t[k][1][:, y0 - yk:y1 - yk, x0 - xk:x1 - xk].flatten(1).float()
            b = keep_t[l][1][:, y0 - yl:y1 - yl, x0 - xl:x1 - xl].flatten(1).float()
            blk = a @ b.T
            inter[base[k]:base[k + 1], base[l]:base[l + 1]] = blk
            inter[base[l]:base[l + 1], base[k]:base[k + 1]] = blk.T
    iou = inter / (areas[:, None] + areas[None, :] - inter)
    scores = torch.cat([d[:, 4] for d, _ in keep_t])
    iou_c, sc_c = iou.cpu().numpy(), scores.cpu().numpy()
    order = np.argsort(-sc_c, kind="stable")
    kept = []
    for i in order:
        if all(iou_c[i, j] < 0.5 for j in kept):
            kept.append(i)
    kept = sorted(kept)
    # 남은 마스크·박스 CPU 전송 + 장축
    allm = torch.cat([m for _, m in keep_t])[kept].cpu().numpy()
    alld = torch.cat([d for d, _ in keep_t])[kept].cpu()          # noqa: F841
    thr = F.MIN_MAJOR_FRAC * 2 * s
    return int(sum(F.major_axis_px(mm) >= thr for mm in allm))


def run_tiles(net, im, s, batch):
    H, W = im.shape[:2]
    T = T7.tiles_of(H, W, "tile")
    t_pre = t_fwd = t_post = 0.0
    t0 = sync()
    if batch:
        x = to_tensor([cv2.resize(im[y:y + h, xx:xx + w], (s, s), interpolation=cv2.INTER_LINEAR)
                       for (xx, y, w, h) in T])
        t1 = sync()
        out = net(x)
        t2 = sync()
        p0 = out[0]
        protos = p0[1] if isinstance(p0, tuple) else out[1]
        tiles = post_one(p0, protos, s)
        t3 = sync()
        t_pre, t_fwd, t_post = t1 - t0, t2 - t1, t3 - t2
    else:
        tiles = []
        ta = t0
        for (xx, y, w, h) in T:
            x = to_tensor([cv2.resize(im[y:y + h, xx:xx + w], (s, s), interpolation=cv2.INTER_LINEAR)])
            tb = sync()
            out = net(x)
            tc = sync()
            p0 = out[0]
            protos = p0[1] if isinstance(p0, tuple) else out[1]
            tiles += post_one(p0, protos, s)
            td = sync()
            t_pre += tb - ta
            t_fwd += tc - tb
            t_post += td - tc
            ta = td
        t3 = ta
    nfin = merge(tiles, s)
    t4 = sync()
    t_merge = t4 - t3
    return [t_pre * 1e3, t_fwd * 1e3, t_post * 1e3, t_merge * 1e3, (t4 - t0) * 1e3, nfin]


def main():
    files = sorted(F.DATASETS["test"].glob("*.jpg"))
    imgs = [cv2.imread(str(f)) for f in files]
    assert len(imgs) == 20
    raw, rows = [], []
    for cond, tr, s, sf in X.TILE_CONDS:
        m = YOLO(str(S.ckpt_of(tr, 0)))
        net = m.model.fuse(verbose=False).to("cuda").eval()
        for mode in ("fullframe", "tile_seq_b1x9", "tile_batch_b9x1"):
            with torch.inference_mode():
                fn = ((lambda im: L4.run_once(net, im, sf, False, "input")) if mode == "fullframe"
                      else (lambda im, b=(mode == "tile_batch_b9x1"): run_tiles(net, im, s, b)))
                for k in range(WARMUP):
                    fn(imgs[k % 20])
                res = [fn(imgs[k % 20]) for k in range(N_ITER)]
            for k, r in enumerate(res):
                if mode == "fullframe":
                    r = r[:3] + [0.0, r[3], r[4]]
                raw.append([cond, tr, s, sf, mode, k, files[k % 20].name]
                           + [round(v, 4) for v in r[:5]] + [r[5]])
            a = np.array([q[7:13] for q in raw[-N_ITER:]], float)     # pre..total, n_final
            o = [cond, tr, s, sf, mode, sf if mode == "fullframe" else s]
            for j in range(5):
                o += [round(a[:, j].mean(), 3), round(a[:, j].std(ddof=1), 3),
                      round(float(np.median(a[:, j])), 3)]
            o += [round(1000 / a[:, 4].mean(), 2), round(a[:, 5].mean(), 2)]
            rows.append(o)
            print(o[:6], "pre", o[6], "fwd", o[9], "post", o[12], "merge", o[15], "tot", o[18], flush=True)
        del m, net
        torch.cuda.empty_cache()
    X.wcsv("x1_latency_raw.csv", ["condition", "train_res", "tile_size", "fullframe_size", "mode", "iter",
                                  "file", "pre_ms", "fwd_ms", "post_ms", "merge_ms", "total_ms",
                                  "n_final"], raw)
    hdr = ["condition", "train_res", "tile_size", "fullframe_size", "mode", "net_input"]
    for k in ("pre", "fwd", "post", "merge", "total"):
        hdr += [f"{k}_mean", f"{k}_sd", f"{k}_median"]
    X.wcsv("x1_latency_summary.csv", hdr + ["total_FPS", "n_final_mean"], rows)
    X.wcsv("x1_latency_environment.csv", ["item", "value"], L4.env_info())


if __name__ == "__main__":
    main()
