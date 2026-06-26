#!/usr/bin/env python
"""
Per-window, all-folds inference WITHOUT materializing a stack.

Reads the 180 existing daily normalized forcing rasters for one window ONCE into
memory, applies the (optional) year-feature freeze once, then runs ALL fold models
on that shared in-memory forcing and writes each fold's center-60 prediction.

Replaces (build 13 GB stack)+(10x re-read stack, 10x process spinup). The forcing
already exists on disk; we just read it once and reuse it across folds.

Output per fold: <out_root>/<fold_id>/pred-center60-<d_end>.tif  (60 bands,
named vwc_<date>), bit-compatible with the existing 5_2 stitch/ensemble stages.
"""
import os, sys, argparse, datetime as dt
import numpy as np, rasterio, torch
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raster_inference import load_model, seq_model_pixelwise_infer, tile_grid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily_dir", required=True)
    ap.add_argument("--d_end", required=True)                  # YYYY-MM-DD (window end)
    ap.add_argument("--models", required=True)                 # "fold_1=/path/model.pt,fold_2=/path,..."
    ap.add_argument("--out_root", required=True)               # predictions-center60-{depth}{suffix}
    ap.add_argument("--timesteps", type=int, default=180)
    ap.add_argument("--channels", type=int, default=61)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--chunk_px", type=int, default=8192)
    ap.add_argument("--freeze_channel", type=int, default=-1)  # -1 = no freeze
    ap.add_argument("--freeze_value", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--read_workers", type=int, default=24)
    ap.add_argument("--model_py_dir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()

    d_end = dt.date.fromisoformat(a.d_end)
    dates = [d_end - dt.timedelta(days=k) for k in range(a.timesteps - 1, -1, -1)]  # ascending, ends at d_end
    paths = [f"{a.daily_dir}/normalized-data-{d.isoformat()}.tif" for d in dates]
    if not all(os.path.exists(p) for p in paths):
        print("MISSING_INPUTS", flush=True); sys.exit(3)
    center_dates = dates[60:120]                               # bands 61..120 (1-based)

    folds = [kv.split("=", 1) for kv in a.models.split(",")]   # [(fold_id, path)]
    def outp(fid): return os.path.join(a.out_root, fid, f"pred-center60-{a.d_end}.tif")
    todo = [(fid, p) for fid, p in folds if not os.path.exists(outp(fid))]
    if not todo:
        print("ALL_FOLDS_DONE", flush=True); return

    # ---- 1) read the 180 dailies ONCE -> [T, C, H, W] ----
    with rasterio.open(paths[0]) as s0:
        H, W = s0.height, s0.width
        prof = s0.profile.copy(); crs = s0.crs; transform = s0.transform
    full = np.empty((a.timesteps, a.channels, H, W), dtype=np.float32)
    def rd(i):
        with rasterio.open(paths[i]) as s:
            return i, s.read(out_dtype="float32")
    with ThreadPoolExecutor(a.read_workers) as ex:
        for i, arr in ex.map(rd, range(a.timesteps)):
            full[i] = arr
    # ---- 2) freeze the year channel once (preserve NaN/land mask: only set finite cells) ----
    if a.freeze_channel >= 0:
        full[:, a.freeze_channel, :, :] = np.float32(a.freeze_value)

    dev = torch.device(a.device)
    models = [(fid, load_model(p, dev, use_state_dict=False, model_py_dir=a.model_py_dir, model_class="uNet"))
              for fid, p in todo]

    # ---- 3) tiled inference; one read of the forcing serves all folds ----
    acc = {fid: np.full((60, H, W), np.nan, dtype=np.float32) for fid, _ in todo}
    for (x, y, w, h) in tile_grid(W, H, a.tile, 0):
        tile = np.ascontiguousarray(full[:, :, y:y+h, x:x+w])          # [T,C,h,w]
        x5d = torch.from_numpy(tile).unsqueeze(0)                       # [1,T,C,h,w] (CPU; chunks move to GPU inside)
        nanmask = ~np.isfinite(tile[0, 0])                             # land/sea mask from band0 of day0
        for fid, model in models:
            y_out = seq_model_pixelwise_infer(model, x5d, device=dev, chunk_px=a.chunk_px)  # [1,Tout,h,w] cpu
            c60 = y_out[0, 60:120].numpy().copy()                       # [60,h,w]
            c60[:, nanmask] = np.nan
            acc[fid][:, y:y+h, x:x+w] = c60
        del x5d, tile

    # ---- 4) write per-fold center-60 (matches 5_2 trim output) ----
    prof.update(count=60, dtype="float32", compress="lzw", tiled=True,
                blockxsize=256, blockysize=256, bigtiff="IF_SAFER")
    for fid, _ in todo:
        op = outp(fid); os.makedirs(os.path.dirname(op), exist_ok=True)
        tmp = op + ".tmp"
        with rasterio.open(tmp, "w", **prof) as dst:
            dst.write(acc[fid])
            for bi, d in enumerate(center_dates, start=1):
                dst.set_band_description(bi, f"vwc_{d.isoformat()}")
        os.replace(tmp, op)                                            # atomic -> resume-safe
    print("OK", flush=True)

if __name__ == "__main__":
    main()
