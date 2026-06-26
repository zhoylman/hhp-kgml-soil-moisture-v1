#!/usr/bin/env python
"""
Cluster array worker: load all fold models ONCE, process a RANGE of windows.

Each SLURM array task runs one of these on one GPU, handling windows
[win_start : win_start+win_count) of the stride-30 schedule. For each window it
reads the 180 daily forcing grids once, freezes `year`, runs every fold model on
the shared forcing, and writes each fold's center-60 (atomic, resume-safe).

VRAM is the constraint on the 11 GB 2080 Ti nodes -> use a small --chunk_px
(~1024-1536). Larger cards (A40/A4500) can use more but the small value is safe
everywhere, so one uniform array spans all GPU types.

Schedule MUST match R/5_2: centers = sorted_dates[89 : N-90 : 30] (0-based),
window end = center + 89 days, center-60 = window days 61..120 (0-based 60:120).
"""
import os, sys, glob, argparse, datetime as dt
import numpy as np, rasterio, torch
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raster_inference import load_model, seq_model_pixelwise_infer, tile_grid

def build_schedule(daily_dir, timesteps=180, stride=30):
    fs = glob.glob(f"{daily_dir}/normalized-data-*.tif")
    dates = sorted(dt.date.fromisoformat(os.path.basename(f)[len("normalized-data-"):-4]) for f in fs)
    n = len(dates)
    centers = dates[(timesteps//2) - 1 : n - (timesteps//2) : stride]   # match 5_2 seq(90,n-90,30)
    ends = [c + dt.timedelta(days=(timesteps//2) - 1) for c in centers]  # center + 89
    return ends

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily_dir", required=True)
    ap.add_argument("--models", required=True)      # "fold_1=/path,..."
    ap.add_argument("--out_root", required=True)     # predictions-center60-<depth>
    ap.add_argument("--win_start", type=int, required=True)
    ap.add_argument("--win_count", type=int, required=True)
    ap.add_argument("--timesteps", type=int, default=180)
    ap.add_argument("--channels", type=int, default=61)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--chunk_px", type=int, default=1536)   # safe for 11 GB; bump on A40/A4500
    ap.add_argument("--freeze_channel", type=int, default=-1)
    ap.add_argument("--freeze_value", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--read_workers", type=int, default=16)
    a = ap.parse_args()

    ends = build_schedule(a.daily_dir, a.timesteps)
    sel = ends[a.win_start : a.win_start + a.win_count]
    print(f"[worker] schedule={len(ends)} windows; this task does idx {a.win_start}..{a.win_start+a.win_count-1} "
          f"({len(sel)} windows: {sel[0] if sel else '-'} .. {sel[-1] if sel else '-'})", flush=True)

    folds = [kv.split("=", 1) for kv in a.models.split(",")]
    dev = torch.device(a.device)
    models = [(fid, load_model(p, dev, use_state_dict=False,
                               model_py_dir=os.path.dirname(os.path.abspath(__file__)), model_class="uNet"))
              for fid, p in folds]
    print(f"[worker] loaded {len(models)} models once", flush=True)

    def outp(fid, d_end): return os.path.join(a.out_root, fid, f"pred-center60-{d_end.isoformat()}.tif")

    for d_end in sel:
        dates = [d_end - dt.timedelta(days=k) for k in range(a.timesteps - 1, -1, -1)]
        paths = [f"{a.daily_dir}/normalized-data-{d.isoformat()}.tif" for d in dates]
        if not all(os.path.exists(p) for p in paths):
            print(f"[worker] skip {d_end} (missing inputs)", flush=True); continue
        if all(os.path.exists(outp(fid, d_end)) for fid, _ in folds):
            print(f"[worker] {d_end} already complete; skip", flush=True); continue
        center_dates = dates[60:120]

        with rasterio.open(paths[0]) as s0:
            H, W = s0.height, s0.width; prof = s0.profile.copy()
        full = np.empty((a.timesteps, a.channels, H, W), dtype=np.float32)
        def rd(i):
            with rasterio.open(paths[i]) as s: return i, s.read(out_dtype="float32")
        with ThreadPoolExecutor(a.read_workers) as ex:
            for i, arr in ex.map(rd, range(a.timesteps)): full[i] = arr
        if a.freeze_channel >= 0:
            full[:, a.freeze_channel, :, :] = np.float32(a.freeze_value)

        acc = {fid: np.full((60, H, W), np.nan, np.float32) for fid, _ in folds}
        for (x, y, w, h) in tile_grid(W, H, a.tile, 0):
            tile = np.ascontiguousarray(full[:, :, y:y+h, x:x+w])
            x5d = torch.from_numpy(tile).unsqueeze(0)
            nanmask = ~np.isfinite(tile[0, 0])
            for fid, model in models:
                yo = seq_model_pixelwise_infer(model, x5d, device=dev, chunk_px=a.chunk_px)
                c60 = yo[0, 60:120].numpy().copy(); c60[:, nanmask] = np.nan
                acc[fid][:, y:y+h, x:x+w] = c60
            del x5d, tile
        del full

        prof.update(count=60, dtype="float32", compress="lzw", tiled=True,
                    blockxsize=256, blockysize=256, bigtiff="IF_SAFER")
        for fid, _ in folds:
            op = outp(fid, d_end); os.makedirs(os.path.dirname(op), exist_ok=True)
            tmp = op + ".tmp"
            with rasterio.open(tmp, "w", **prof) as dst:
                dst.write(acc[fid])
                for bi, d in enumerate(center_dates, start=1): dst.set_band_description(bi, f"vwc_{d.isoformat()}")
            os.replace(tmp, op)
        print(f"[worker] wrote {d_end} ({len(folds)} folds)", flush=True)
    print("[worker] DONE", flush=True)

if __name__ == "__main__":
    main()
