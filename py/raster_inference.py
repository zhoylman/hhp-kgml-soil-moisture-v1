#!/usr/bin/env python3
"""
Tile-based inference over multi-band rasters (GeoTIFF) with a saved PyTorch model.

Supports:
- Spatial UNet-like models (expects [B,C,H,W] or [B,T,C,H,W] inputs).
- Sequence models trained on points (expects [N,T,C]); applied per-pixel via an adapter.

Examples
--------
# UNet/spatial model (bands already ordered & normalized)
python raster_inference.py \
  --model /path/to/model.pt \
  --inp /path/to/in.tif \
  --out /path/to/out.tif \
  --timesteps 180 --channels_per_step 61 --model_layout TC \
  --write_mode multiband \
  --model_kind unet --device cuda --tile 1024 --overlap 32 --batch_size 8

# Sequence model saved as state_dict; use your training modules to build the net
python raster_inference.py \
  --model /path/to/model.pt \
  --inp /path/to/in.tif \
  --out /path/to/out.tif \
  --timesteps 180 --channels_per_step 61 --model_layout TC \
  --write_mode multiband \
  --model_kind seq --use_state_dict \
  --model_py_dir /home/zhoylman/hhp-kgml-soil-moisture-v1/py \
  --model_class uNet \
  --device cuda --seq_chunk_px 524288

# Whole directory
python raster_inference.py \
  --model /path/to/model.pt \
  --in_dir /path/to/stacks \
  --out_dir /path/to/preds \
  --timesteps 180 --channels_per_step 61 --model_layout TC \
  --write_mode multiband --model_kind seq --use_state_dict \
  --model_py_dir /home/zhoylman/hhp-kgml-soil-moisture-v1/py --model_class uNet \
  --device cuda
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


# ------------------------
# Device helpers
# ------------------------

def device_from_arg(arg: str) -> torch.device:
    if arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if arg == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# ------------------------
# Model loading
# ------------------------

def build_model_from_state_dict(model_path: str,
                                device: torch.device,
                                model_py_dir: Optional[str],
                                model_class: str):
    """
    Load model from a state_dict using user's model_config + define_hyperparams.
    Assumes the class constructor signature from the user's repo:
        uNet(num_classes, input_size, max_channels, dropout)
    """
    if model_py_dir:
        sys.path.append(model_py_dir)

    # Import user's modules
    import model_config  # noqa: E401
    import define_hyperparams as HP  # noqa: E401

    # Instantiate model class from model_config
    ModelClass = getattr(model_config, model_class)
    # Pull common hyperparams from user's define_hyperparams.py
    num_classes = getattr(HP, "num_classes", 1)
    input_size = getattr(HP, "input_size", 61)        # fallback sensible default
    max_channels = getattr(HP, "max_channels", 64)
    dropout = getattr(HP, "dropout", 0.0)

    model = ModelClass(num_classes, input_size, max_channels, dropout).to(device)

    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_model(model_path: str,
               device: torch.device,
               use_state_dict: bool,
               model_py_dir: Optional[str],
               model_class: str):
    """
    Load either:
      - a state_dict (if --use_state_dict), via build_model_from_state_dict, or
      - a pickled full model (torch.save(model, ...)).
    If not explicitly use_state_dict, auto-detect OrderedDict and fallback.
    """
    if use_state_dict:
        return build_model_from_state_dict(model_path, device, model_py_dir, model_class)

    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        # Looks like a state_dict (OrderedDict)
        return build_model_from_state_dict(model_path, device, model_py_dir, model_class)

    model = obj
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ------------------------
# Raster tiling utils
# ------------------------

def tile_grid(width: int, height: int, tile: int, overlap: int):
    """Yield (x, y, w, h) windows covering (width,height) with tiles+overlap."""
    stride = max(1, tile - overlap)
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            w = min(tile, width - x)
            h = min(tile, height - y)
            yield (x, y, w, h)


def maybe_apply_band_xform(arr: np.ndarray, mode=None, params=None) -> np.ndarray:
    """
    Optionally apply per-band transforms.
    - arr: [C, H, W]
    - mode: None | 'standardize' | 'minmax'
    - params: dict with arrays 'mean','std' or 'min','max' of length C
    """
    if mode is None:
        return arr
    if mode == "standardize":
        mean = np.asarray(params["mean"]).reshape(-1, 1, 1)
        std  = np.asarray(params["std"]).reshape(-1, 1, 1)
        return (arr - mean) / (std + 1e-8)
    if mode == "minmax":
        mn = np.asarray(params["min"]).reshape(-1, 1, 1)
        mx = np.asarray(params["max"]).reshape(-1, 1, 1)
        return (arr - mn) / (mx - mn + 1e-8)
    return arr


def prepare_input_5d(data_chw: np.ndarray,
                     timesteps: Optional[int],
                     channels_per_step: Optional[int],
                     model_layout: str):
    """
    data_chw: [C_total, H, W]
    Returns:
      - if model_layout == 'TC' and timesteps provided: [T, C, H, W]
      - if model_layout == 'flat' and timesteps provided: [T*C, H, W]
      - else returns original [C_total, H, W]
    """
    C_total, H, W = data_chw.shape

    if timesteps is None:
        return data_chw

    # infer channels_per_step if not provided
    if channels_per_step is None:
        if C_total % timesteps != 0:
            raise ValueError(f"Cannot infer channels_per_step: C_total={C_total} not divisible by T={timesteps}")
        channels_per_step = C_total // timesteps

    expected = timesteps * channels_per_step
    if expected != C_total:
        raise ValueError(f"Band count mismatch: expected T*C={expected}, got {C_total}")

    # Assume band order: t0_c0..t0_c(C-1), t1_c0.., ..., t(T-1)_c(C-1)
    x = data_chw.reshape(timesteps, channels_per_step, H, W)  # [T,C,H,W]
    # Optional input-channel freeze (e.g. the `year` feature): override one channel
    # across all timesteps with a constant. Set via env so no signature churn.
    _fc = os.environ.get("FREEZE_CHANNEL"); _fv = os.environ.get("FREEZE_VALUE")
    if _fc is not None and _fv is not None:
        x[:, int(_fc), :, :] = np.float32(float(_fv))   # year held at validation-period center
    if model_layout == "TC":
        return x
    else:  # 'flat'
        return x.reshape(timesteps * channels_per_step, H, W)


# ------------------------
# Sequence model pixelwise adapter
# ------------------------

def seq_model_pixelwise_infer(model, x5d, *, device, chunk_px=262144):
    """
    Apply a sequence model per pixel.
    x5d: torch.Tensor [B,T,C,H,W] on device
    Returns: torch.Tensor [B,T_out,H,W] on CPU
    """
    assert x5d.dim() == 5, f"expected 5D [B,T,C,H,W], got {tuple(x5d.shape)}"
    B, T, C, H, W = x5d.shape
    S = H * W

    # [B,T,C,H,W] -> [B,H,W,T,C] -> [B*S, T, C]
    x = x5d.permute(0, 3, 4, 1, 2).contiguous().view(B * S, T, C)

    outs = []
    for start in range(0, B * S, chunk_px):
        end = min(start + chunk_px, B * S)
        xb = x[start:end].to(device, non_blocking=True)  # [chunk, T, C]
        with torch.no_grad():
            yb = model(xb)  # [chunk, T_out] or [chunk, 1] or [chunk, T_out, K]
        yb = yb.detach().float().cpu()

        # normalize shapes to [chunk, T_out]
        if yb.dim() == 1:
            yb = yb.unsqueeze(1)
        elif yb.dim() == 3:
            # if multi-target last dim, take first target channel
            yb = yb[:, :, 0]

        outs.append(yb)

    y = torch.cat(outs, dim=0)  # [B*S, T_out]
    T_out = y.shape[1]
    # [B*S, T_out] -> [B,H,W,T_out] -> [B,T_out,H,W]
    y = y.view(B, H, W, T_out).permute(0, 3, 1, 2).contiguous()  # [B,T_out,H,W]
    return y


# ------------------------
# Core inference
# ------------------------

from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def _read_tile(src, x, y, w, h, *,
               timesteps, channels_per_step, model_layout,
               transform_mode, transform_params, pin_memory):
    """Read + prepare one tile -> returns dict with CPU tensor and metadata."""
    win = Window(col_off=x, row_off=y, width=w, height=h)
    data = src.read(window=win, out_dtype="float32")  # [C,H,W] (rasterio is fast under threads)
    nan_mask = np.any(~np.isfinite(data), axis=0)
    data = maybe_apply_band_xform(data, transform_mode, transform_params)

    prep = prepare_input_5d(
        data_chw=data,
        timesteps=timesteps,
        channels_per_step=channels_per_step,
        model_layout=model_layout
    )

    if prep.ndim == 4:     # [T,C,H,W] -> [1,T,C,H,W]
        t = torch.from_numpy(prep).unsqueeze(0)
    elif prep.ndim == 3:   # [C',H,W] -> [1,C',H,W]
        t = torch.from_numpy(prep).unsqueeze(0)
    else:
        raise RuntimeError(f"Unexpected prepared input shape: {prep.shape}")

    if pin_memory:
        t = t.pin_memory()

    return dict(tensor=t, shape=(h, w), offs=(x, y), nan_mask=nan_mask)

def _load_batch(src, batch_tiles, **kwargs):
    """Read & prepare a batch of tiles in parallel (CPU side)."""
    items = []
    # Parallelize tile reads — I/O bound → threads are perfect.
    with ThreadPoolExecutor(max_workers=kwargs["read_workers"]) as ex:
        futs = []
        for (x, y, w, h) in batch_tiles:
            futs.append(ex.submit(_read_tile, src, x, y, w, h,
                                  timesteps=kwargs["timesteps"],
                                  channels_per_step=kwargs["channels_per_step"],
                                  model_layout=kwargs["model_layout"],
                                  transform_mode=kwargs["transform_mode"],
                                  transform_params=kwargs["transform_params"],
                                  pin_memory=kwargs["pin_memory"]))
        for f in futs:
            items.append(f.result())

    # Pad into a single CPU tensor (still on CPU), keep metadata arrays
    max_h = max(it["shape"][0] for it in items)
    max_w = max(it["shape"][1] for it in items)
    t0 = items[0]["tensor"]

    if t0.ndim == 5:  # [B,T,C,H,W]
        _, T_in, C_in, _, _ = t0.shape
        batch_cpu = torch.zeros((len(items), T_in, C_in, max_h, max_w), dtype=torch.float32)
        for i, it in enumerate(items):
            h, w = it["shape"]
            batch_cpu[i, :, :, :h, :w] = it["tensor"]
    else:              # [B,C',H,W]
        _, C_in, _, _ = t0.shape
        batch_cpu = torch.zeros((len(items), C_in, max_h, max_w), dtype=torch.float32)
        for i, it in enumerate(items):
            h, w = it["shape"]
            batch_cpu[i, :, :h, :w] = it["tensor"]

    # Convert nan_masks to same padded shapes for quick broadcast later
    nan_masks = []
    for it in items:
        h, w = it["shape"]
        m = it["nan_mask"]
        if m.shape != (h, w):
            m = m[:h, :w]
        mm = np.zeros((max_h, max_w), dtype=bool)
        mm[:h, :w] = m
        nan_masks.append(mm)

    offsets = [it["offs"] for it in items]
    shapes  = [it["shape"] for it in items]
    return batch_cpu, offsets, shapes, nan_masks

def run_inference_on_raster(
    model: torch.nn.Module,
    inp_path: str,
    out_path: str,
    *,
    tile: int = 512,
    overlap: int = 64,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
    transform_mode=None,
    transform_params=None,
    dtype_out="float32",
    timesteps: Optional[int] = None,
    channels_per_step: Optional[int] = None,
    model_layout: str = "TC",
    write_mode: str = "multiband",
    model_kind: str = "unet",
    seq_chunk_px: int = 262144,
    # NEW:
    read_workers: int = 8,
    prefetch_batches: int = 2,
    pin_memory: bool = False,
) -> str:

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # let GDAL fan out too
    os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

    with rasterio.Env(NUM_THREADS="ALL_CPUS"), rasterio.open(inp_path) as src:
        width, height, count = src.width, src.height, src.count
        crs, transform = src.crs, src.transform

        out_profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": dtype_out,
            "crs": crs,
            "transform": transform,
            "compress": "LZW",
            "tiled": True,
            "blockxsize": min(256, width),
            "blockysize": min(256, height),
            "BIGTIFF": "IF_SAFER"
        }

        # build tile list
        tiles = list(tile_grid(width, height, tile, overlap))
        # group into batches
        batches = [tiles[i:i+batch_size] for i in range(0, len(tiles), batch_size)]

        T_out = None
        sum_buf = None
        cnt_buf = None

        # --- Prefetch pipeline (one or two batches ahead) ---
        from collections import deque
        q = deque()

        def enqueue(idx):
            if idx < len(batches):
                batch_cpu, offs, shapes, nan_masks = _load_batch(
                    src, batches[idx],
                    timesteps=timesteps,
                    channels_per_step=channels_per_step,
                    model_layout=model_layout,
                    transform_mode=transform_mode,
                    transform_params=transform_params,
                    pin_memory=pin_memory,
                    read_workers=read_workers
                )
                q.append((idx, batch_cpu, offs, shapes, nan_masks))

        # prime the queue
        for k in range(min(prefetch_batches, len(batches))):
            enqueue(k)

        from tqdm import trange
        bi = 0
        for _ in trange(len(batches), desc=f"Infer {Path(inp_path).name}"):
            # ensure a filled queue
            if len(q) == 0:
                enqueue(bi)

            idx, batch_cpu, offsets, shapes, nan_masks = q.popleft()
            # schedule next read while GPU works
            enqueue(idx + prefetch_batches)

            # H2D copy (non_blocking if pinned)
            batch_tensor = batch_cpu.to(device, non_blocking=pin_memory)

            # Forward
            with torch.no_grad():
                if model_kind == "seq":
                    if batch_tensor.dim() == 4:
                        raise RuntimeError("Sequence model requires [B,T,C,H,W] input; got 4D.")
                    pred_t = seq_model_pixelwise_infer(
                        model, batch_tensor, device=device, chunk_px=seq_chunk_px
                    )  # CPU tensor [B,T_out,H,W]
                else:
                    pred = model(batch_tensor)
                    if pred.dim() == 5:
                        pred = pred[:, :, 0, :, :] if pred.shape[2] >= 1 else pred
                    elif pred.dim() == 4:
                        if timesteps is not None and pred.shape[1] == timesteps:
                            pass
                        else:
                            pred = pred.unsqueeze(1)
                    else:
                        raise RuntimeError(f"Unexpected model output shape: {tuple(pred.shape)}")
                    pred_t = pred.detach().float().cpu()

            pred_np = pred_t.numpy()  # [B,T_out,H,W]
            if T_out is None:
                T_out = pred_np.shape[1]
                if write_mode == "multiband":
                    sum_buf = np.zeros((T_out, height, width), dtype=np.float64)
                    cnt_buf = np.zeros((T_out, height, width), dtype=np.uint16)
                else:
                    sum_buf = np.zeros((1, height, width), dtype=np.float64)
                    cnt_buf = np.zeros((1, height, width), dtype=np.uint16)

            # accumulate
            for bi2, ((x, y), (h, w), m) in enumerate(zip(offsets, shapes, nan_masks)):
                tile_pred_all = pred_np[bi2, :, :h, :w]  # [T_out,h,w]
                m3 = np.broadcast_to(m[:h, :w], (tile_pred_all.shape[0], h, w))
                tile_pred_all = np.where(m3, np.nan, tile_pred_all)

                if write_mode == "multiband":
                    valid = np.isfinite(tile_pred_all)
                    sum_buf[:, y:y+h, x:x+w][valid] += tile_pred_all[valid]
                    cnt_buf[:, y:y+h, x:x+w][valid] += 1
                else:
                    tile_pred_0 = tile_pred_all[0]
                    valid = np.isfinite(tile_pred_0)
                    sum_buf[0, y:y+h, x:x+w][valid] += tile_pred_0[valid]
                    cnt_buf[0, y:y+h, x:x+w][valid] += 1

            bi += 1  # advance loop idx

        # finalize mean
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = sum_buf / cnt_buf
        avg[~np.isfinite(avg)] = np.nan

        # write
        out_profile["count"] = T_out if write_mode == "multiband" else 1
        with rasterio.open(out_path, "w", **out_profile) as dst:
            if write_mode == "multiband":
                for b in range(T_out):
                    dst.write(avg[b].astype(dtype_out), b + 1)
            else:
                dst.write(avg[0].astype(dtype_out), 1)

    return str(out_path)

def infer_directory(model, in_dir, out_dir, **kwargs):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tifs = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in [".tif", ".tiff"]])
    if not tifs:
        raise FileNotFoundError(f"No GeoTIFFs found in {in_dir}")

    for tif in tifs:
        out_name = tif.stem + "_pred.tif"
        out_path = out_dir / out_name
        run_inference_on_raster(model, str(tif), str(out_path), **kwargs)


# ------------------------
# CLI
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model.pt")

    # single file or directory
    ap.add_argument("--inp", help="Input multiband GeoTIFF")
    ap.add_argument("--out", help="Output prediction GeoTIFF")
    ap.add_argument("--in_dir", help="Folder of input GeoTIFFs")
    ap.add_argument("--out_dir", help="Folder for predictions")

    # IO/compute controls
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--dtype_out", default="float32",
                    choices=["float32", "float64", "uint16", "int16"])

    # Time/channel layout
    ap.add_argument("--timesteps", type=int, default=None,
                    help="Number of time steps T in the input (e.g., 180).")
    ap.add_argument("--channels_per_step", type=int, default=None,
                    help="Number of feature channels per time step C (if omitted, inferred as bands/T).")
    ap.add_argument("--model_layout", choices=["TC", "flat"], default="TC",
                    help="Model input layout: 'TC' -> [B,T,C,H,W]; 'flat' -> [B,T*C,H,W].")
    ap.add_argument("--write_mode", choices=["single", "multiband"], default="multiband",
                    help="Output single-band or multi-band GeoTIFF.")

    # Model kind
    ap.add_argument("--model_kind", choices=["unet", "seq"], default="unet",
                    help="unet = spatial conv model; seq = per-pixel sequence model.")
    ap.add_argument("--seq_chunk_px", type=int, default=262144,
                    help="Pixels per chunk for sequence model adapter (controls VRAM).")

    # Loading mode (state_dict vs full model)
    ap.add_argument("--use_state_dict", action="store_true",
                    help="Treat --model as a state_dict and build the net with user's modules.")
    ap.add_argument("--model_py_dir", default=None,
                    help="Directory containing model_config.py and define_hyperparams.py.")
    ap.add_argument("--model_class", default="uNet",
                    help="Model class name in model_config.py for state_dict loading.")

    # Optional transforms (if you *didn’t* pre-normalize)
    ap.add_argument("--xform_mode", default=None, choices=[None, "standardize", "minmax"])
    ap.add_argument("--xform_params", default=None, help="Path to npz/json with per-band stats")

    # new ones for the parallel to increase CPU usage
    ap.add_argument("--read_workers", type=int, default=8,
                    help="Parallel tile-read worker threads (I/O is GIL-friendly).")
    ap.add_argument("--prefetch_batches", type=int, default=2,
                    help="Number of batches to prefetch (1–2 is usually plenty).")
    ap.add_argument("--pin_memory", action="store_true",
                    help="Pin CPU tensors before H2D (.to(non_blocking=True)).")

    args = ap.parse_args()

    dev = device_from_arg(args.device)
    print(f"Using device: {dev}")

    model = load_model(args.model, dev, args.use_state_dict, args.model_py_dir, args.model_class)

    xform_params = None
    if args.xform_mode and args.xform_params:
        if args.xform_params.endswith(".npz"):
            z = np.load(args.xform_params)
            xform_params = {k: z[k] for k in z.files}
        else:
            import json
            with open(args.xform_params, "r") as f:
                xform_params = json.load(f)

    common_kwargs = dict(
        tile=args.tile,
        overlap=args.overlap,
        batch_size=args.batch_size,
        device=dev,
        transform_mode=args.xform_mode,
        transform_params=xform_params,
        dtype_out=args.dtype_out,
        timesteps=args.timesteps,
        channels_per_step=args.channels_per_step,
        model_layout=args.model_layout,
        write_mode=args.write_mode,
        model_kind=args.model_kind,
        seq_chunk_px=args.seq_chunk_px,
    )

    if args.in_dir and args.out_dir:
        infer_directory(model, args.in_dir, args.out_dir, **common_kwargs)
    elif args.inp and args.out:
        run_inference_on_raster(model, args.inp, args.out, **common_kwargs)
    else:
        raise SystemExit("Provide either --inp/--out or --in_dir/--out_dir")


if __name__ == "__main__":
    # Recommended env tweaks (set in shell, shown here for convenience):
    #   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:true
    #   export GDAL_CACHEMAX=1024
    main()