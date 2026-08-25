#!/usr/bin/env python
"""
Point-based center-keep inference + evaluation for k-fold models (production
OR the no-pretrain ablation), WITHOUT a full gridded regen. Reproduces the
EXACT center-keep + Gaussian cross-fade stitching used in
R/5_2-run-kfold-inference-stitch-ensemble.R (keep_len=60, stride=30, sigma=12),
applied to per-site 1-D time series instead of 2-D raster grids.

Feature construction mirrors R/3_3-build-dataset-finetune-middle.R exactly:
  - raw daily driver series from seq-data/observational-sites-seq-data.csv
  - rolling precip/PET sums (7/15/30/60/365/730d) + climatic water deficits
  - circular_yday = (sin(2*pi*yday/365)+1)/2  (added AFTER seq normalization,
    i.e. never itself min-max normalized)
  - static covariates (terrain/soil) + lat/lon, normalized with the STATIC
    pretrain min-max scaler
  - all other seq features normalized with the SEQ pretrain min-max scaler
  - year channel FROZEN at 0.772727 (2013 under 1979-2023) at inference time,
    matching the canonical year-frozen convention used everywhere else

Run:
  DEPTH_FLAG=middle MODEL_SET=production  FOLDS=1,2,3 python py/exp_point_centerkeep_eval.py
  DEPTH_FLAG=middle MODEL_SET=ablation     FOLDS=1,2,3 python py/exp_point_centerkeep_eval.py
"""
import os, sys, glob, re
import numpy as np, pandas as pd, torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_config import *
from define_hyperparams import *

BASE = "/data/ssd2/soil-moisture-ml"
REPO = "/home/zhoylman/hhp-kgml-soil-moisture-v1"

DEPTH     = os.environ.get("DEPTH_FLAG", "middle")
MODEL_SET = os.environ.get("MODEL_SET", "production")   # production | ablation
FOLDS     = [int(x) for x in os.environ.get("FOLDS", ",".join(str(i) for i in range(1, 11))).split(",")]
assert DEPTH in ("shallow", "middle") and MODEL_SET in ("production", "ablation")

DEPTH_LABEL = {"shallow": "Shallow", "middle": "Middle"}[DEPTH]
suffix = "-shallow" if DEPTH == "shallow" else ""   # matches file naming convention seen on disk
OUT_DIR = f"{REPO}/cache/ablation-pointeval/{DEPTH}_{MODEL_SET}"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- center-keep / Gaussian cross-fade params (EXACT match to R/5_2) -------
KEEP_LEN, STRIDE_DAYS, SIGMA_DAYS = 60, 30, 12
OVERLAP = KEEP_LEN - STRIDE_DAYS   # 30
YEAR_CH, YEAR_FROZEN = 9, (2013 - 1979) / (2023 - 1979)

def gaussian_weights(n=KEEP_LEN, sigma=SIGMA_DAYS):
    mid = (n + 1) / 2
    i = np.arange(1, n + 1)
    return np.exp(-0.5 * ((i - mid) / sigma) ** 2)

W = gaussian_weights()   # length-60, peaks at center

# ---- feature order: EXACT match to band-order.json / training window CSVs -
FEATURE_ORDER = [
    "bi","fm100","fm1000","pet","pr","srad","tmmn","tmmx","vpd","year",
    "precip_roll_sum_short_short","pet_roll_sum_short_short",
    "precip_roll_sum_short","pet_roll_sum_short",
    "precip_roll_sum","pet_roll_sum",
    "precip_roll_sum_mid","pet_roll_sum_mid",
    "precip_roll_sum_long","pet_roll_sum_long",
    "precip_roll_sum_long_long","pet_roll_sum_long_long",
    "def_short","def","def_mid","def_long","def_long_long",
    "ppt","solclear","solslope","soltotal","soltrans","tdmean","tmax","tmean","tmin","vpdmax","vpdmin",
    "X0_b1","X1_constant","X2_elevation",
    "X0_ssurgo_awc","X1_ssurgo_clay","X2_ssurgo_ksat","X3_ssurgo_sand",
    "X0_bd_mean","X1_clay_mean","X10_lambda_mean","X11_hb_mean","X12_alpha_mean",
    "X2_ksat_mean","X3_n_mean","X4_om_mean","X5_ph_mean","X6_sand_mean","X7_silt_mean",
    "X8_theta_r_mean","X9_theta_s_mean",
    "latitude","longitude","circular_yday",
]
PRISM_COLS = ["ppt","solclear","solslope","soltotal","soltrans","tdmean","tmax","tmean","tmin","vpdmax","vpdmin"]
TERRAIN_SOIL_COLS = [
    "X0_b1","X1_constant","X2_elevation",
    "X0_ssurgo_awc","X1_ssurgo_clay","X2_ssurgo_ksat","X3_ssurgo_sand",
    "X0_bd_mean","X1_clay_mean","X10_lambda_mean","X11_hb_mean","X12_alpha_mean",
    "X2_ksat_mean","X3_n_mean","X4_om_mean","X5_ph_mean","X6_sand_mean","X7_silt_mean",
    "X8_theta_r_mean","X9_theta_s_mean",
]
# STATIC scaler covers everything from all-sites-static-data.csv (PRISM normals +
# terrain/soil) PLUS lat/lon (joined in before static normalization, per R/3_3).
# circular_yday is added AFTER seq normalization in the R pipeline and is never
# itself min-max normalized (already scaled to [0,1] by construction).
STATIC_COLS = PRISM_COLS + TERRAIN_SOIL_COLS + ["latitude", "longitude"]

def load_minmax(path):
    d = pd.read_csv(path)
    mn = d[d.min_max_id == "min"].set_index("name")["value"].to_dict()
    mx = d[d.min_max_id == "max"].set_index("name")["value"].to_dict()
    return mn, mx

seq_mn, seq_mx = load_minmax(f"{BASE}/min-max-definitions/seq-min-max-definitions-pretrain{suffix}.csv")
stat_mn, stat_mx = load_minmax(f"{BASE}/min-max-definitions/static-min-max-definitions-pretrain{suffix}.csv")

def normalize(df, cols, mn, mx):
    out = df.copy()
    for c in cols:
        if c not in mn:
            raise KeyError(f"{c} missing from min-max scaler")
        out[c] = (out[c] - mn[c]) / (mx[c] - mn[c])
    return out

# ---- forcing source selection --------------------------------------------
# "raster" (default, FIXED): per-site daily 61-band feature series extracted
#   directly from historical-forcing-data-normalized/*.tif at each site's
#   pixel (nearest-neighbor, matching sm_eval_utils.R's extract_at_sites()).
#   Gap-free by construction (every day has a raster) and ALREADY normalized
#   -- no rolling-sum computation or min-max scaling needed downstream.
# "csv" (legacy): observational-sites-seq-data.csv, which has real per-site
#   gaps -> make_windows()'s contiguity check silently drops windows, causing
#   Table 1 (canonical gridded, n=727 shallow) vs the old point-based ablation
#   (n=685) to diverge. Kept only for side-by-side comparison.
FORCING_SOURCE = os.environ.get("FORCING_SOURCE", "raster")
RASTER_FORCING_DIR = f"{BASE}/seq-data/site-forcing-from-raster"

# ---- load raw daily seq-data (wide: var, time, site1, site2, ...) once -----
# (only needed for the legacy "csv" path; skip the load entirely otherwise --
#  it's a large file and the raster path never touches it)
if FORCING_SOURCE == "csv":
    print("[load] observational-sites-seq-data.csv ...", flush=True)
    SEQ_RAW = pd.read_csv(f"{BASE}/seq-data/observational-sites-seq-data.csv", low_memory=False)
    SEQ_RAW["time"] = pd.to_datetime(SEQ_RAW["time"])

STATIC_RAW = pd.read_csv(f"{BASE}/static-data/all-sites-static-data.csv").drop(columns=["network"])
META = pd.read_csv(f"{BASE}/observations/final-soil-moisture-data-generalized-meta.csv",
                    dtype={"site_id": str})[["site_id", "latitude", "longitude"]].drop_duplicates("site_id")

ROLL_SPECS = [  # (window_days, precip_col, pet_col, def_col_or_None)
    (7,   "precip_roll_sum_short_short", "pet_roll_sum_short_short", None),
    (15,  "precip_roll_sum_short",       "pet_roll_sum_short",       "def_short"),
    (30,  "precip_roll_sum",             "pet_roll_sum",             "def"),
    (60,  "precip_roll_sum_mid",         "pet_roll_sum_mid",         "def_mid"),
    (365, "precip_roll_sum_long",        "pet_roll_sum_long",        "def_long"),
    (730, "precip_roll_sum_long_long",   "pet_roll_sum_long_long",   "def_long_long"),
]

def build_site_daily_from_raster(site_id):
    """Gap-free, ALREADY-NORMALIZED daily feature frame straight from the
    canonical raster archive (see R/6_17-extract-site-forcing-from-rasters.R).
    No rolling sums / normalization needed -- both are already baked in."""
    path = f"{RASTER_FORCING_DIR}/{site_id}.csv"
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d["time"] = pd.to_datetime(d["time"])
    return d.sort_values("time").dropna().reset_index(drop=True)

def build_site_daily(site_id):
    if FORCING_SOURCE == "raster":
        return build_site_daily_from_raster(site_id)
    return build_site_daily_csv(site_id)

def build_site_daily_csv(site_id):
    """Raw daily seq-data -> full feature frame (unnormalized) for one site."""
    if site_id not in SEQ_RAW.columns:
        return None
    d = SEQ_RAW[["var", "time", site_id]].rename(columns={site_id: "val"})
    d = d.pivot_table(index="time", columns="var", values="val").reset_index().sort_values("time")
    d = d.set_index("time")
    for win, pcol, ecol, dcol in ROLL_SPECS:
        d[pcol] = d["pr"].rolling(win, min_periods=win).sum()
        d[ecol] = d["pet"].rolling(win, min_periods=win).sum()
        if dcol is not None:
            d[dcol] = d[pcol] - d[ecol]
    d["year"] = d.index.year.astype(float)
    d["circular_yday"] = (np.sin(2 * np.pi * d.index.dayofyear / 365) + 1) / 2
    d = d.reset_index()

    srow = STATIC_RAW[STATIC_RAW.site_id.astype(str) == str(site_id)]
    mrow = META[META.site_id == str(site_id)]
    if srow.empty or mrow.empty:
        return None
    for c in [c for c in STATIC_COLS if c not in ("latitude", "longitude")]:
        d[c] = float(srow.iloc[0][c])
    d["latitude"] = float(mrow.iloc[0]["latitude"])
    d["longitude"] = float(mrow.iloc[0]["longitude"])
    return d.dropna().reset_index(drop=True)

def normalize_site(d):
    if FORCING_SOURCE == "raster":
        return d   # already normalized in the raster archive -- no-op
    seq_cols = [c for c in FEATURE_ORDER if c not in STATIC_COLS and c != "circular_yday"]
    d = normalize(d, seq_cols, seq_mn, seq_mx)
    d = normalize(d, STATIC_COLS, stat_mn, stat_mx)
    return d

def make_windows(d, stride=STRIDE_DAYS):
    """d: daily feature frame (normalized), sorted by time, complete (no gaps
    assumed within any 180-day span used). Returns list of (end_date, X[180,61])."""
    dates = d["time"].values
    n = len(d)
    if n <= 200:
        return []
    # R's `seq(90, N-90, by=30)` indexes a 1-based vector; the 0-based equivalent
    # is 89, not 90 (off-by-one previously shifted every window's end date by a
    # full 30-day stitching block relative to the canonical gridded pipeline).
    centers_idx = np.arange(89, n - 90, stride)
    out = []
    feat = d[FEATURE_ORDER].to_numpy(dtype=np.float32)
    for c in centers_idx:
        end = c + 89
        if end >= n:
            break
        start = end - 179
        if start < 0:
            continue
        # require contiguous daily dates (no gaps) over the window
        seg_dates = d["time"].iloc[start:end + 1]
        if (seg_dates.diff().dropna().dt.days != 1).any():
            continue
        out.append((d["time"].iloc[end], feat[start:end + 1]))
    return out

def stitch_site(windows, model, device):
    """windows: list of (end_date, X[180,61]) in time order. Returns DataFrame
    date, ml (center-keep + Gaussian cross-faded daily point series)."""
    if not windows:
        return pd.DataFrame(columns=["date", "ml"])
    Xb = torch.tensor(np.stack([w[1] for w in windows])).float().to(device)
    Xb[:, :, YEAR_CH] = YEAR_FROZEN
    with torch.no_grad():
        yh = model(Xb).cpu().numpy()   # [n_windows, 180] (or [n_windows,180,1] -> squeeze)
    yh = yh.reshape(yh.shape[0], -1)
    center_all = yh[:, 60:120]   # center-keep 60 days, window-relative idx 61-120 (0-idx 60:120)

    day_val, day_wt = {}, {}
    prev_center, prev_end = None, None
    for i, (end_date, _) in enumerate(windows):
        end_date = pd.Timestamp(end_date)
        center_start = end_date - pd.Timedelta(days=119)   # matches R: center <- d_end-89; seq(center-30,center+29)
        cur_dates = pd.date_range(center_start, periods=KEEP_LEN)
        cur = center_all[i]
        if prev_center is None:
            lead_n = KEEP_LEN - OVERLAP
            for j in range(lead_n):
                day_val[cur_dates[j]] = cur[j]; day_wt[cur_dates[j]] = 1.0
        else:
            prev_idx = np.arange(KEEP_LEN - OVERLAP, KEEP_LEN)   # tail 30 of previous
            cur_idx = np.arange(0, OVERLAP)                       # head 30 of current
            for j in range(OVERLAP):
                dt = cur_dates[cur_idx[j]]
                a, b = prev_center[prev_idx[j]], cur[cur_idx[j]]
                w1, w2 = W[prev_idx[j]], W[cur_idx[j]]
                day_val[dt] = (a * w1 + b * w2) / (w1 + w2)
                day_wt[dt] = 1.0
        prev_center, prev_end = cur, end_date
    # tail of the LAST window (its own tail-30, no next window to blend with)
    if prev_center is not None:
        last_end = pd.Timestamp(windows[-1][0])
        last_center_start = last_end - pd.Timedelta(days=119)
        last_dates = pd.date_range(last_center_start, periods=KEEP_LEN)
        for j in range(KEEP_LEN - OVERLAP, KEEP_LEN):
            day_val.setdefault(last_dates[j], prev_center[j])

    out = pd.DataFrame({"date": list(day_val.keys()), "ml": list(day_val.values())}).sort_values("date")
    return out.reset_index(drop=True)

def kge(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    if m.sum() < 10:
        return np.nan, np.nan, np.nan
    s, o = sim[m], obs[m]
    if s.std() == 0 or o.std() == 0 or o.mean() == 0:
        return np.nan, np.nan, np.nan
    r = np.corrcoef(s, o)[0, 1]
    k = 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)
    pbias = 100 * (s.sum() - o.sum()) / o.sum()
    return k, r, pbias

def model_dirs(fold):
    if MODEL_SET == "production":
        pat = f"{BASE}/results-kfold-{DEPTH}/fold_{fold}_*"
    else:
        pat = f"{BASE}/results-kfold-{DEPTH}-nopretrain-ablation/fold_{fold}_*"
    hits = glob.glob(pat)
    return hits[0] if hits else None

def main():
    obs = pd.read_csv(f"{BASE}/observations/final-soil-moisture-data-generalized-no-frozen.csv",
                       dtype={"site_id": str}, low_memory=False)
    obs = obs[obs.generalized_depth == DEPTH_LABEL][["site_id", "date", "soil_moisture"]]
    obs["date"] = pd.to_datetime(obs["date"])

    tmpl = StreamflowTrainDatasetTemplate(csv_file=glob.glob(f"{BASE}/full-dataloader-{DEPTH}/*.csv")[0])
    input_size = tmpl.x_tensor.shape[2]
    device = "cuda"

    all_rows = []
    for fold in FOLDS:
        mdir = model_dirs(fold)
        if mdir is None:
            print(f"[skip] fold {fold}: no model dir found ({MODEL_SET})"); continue
        model = uNet(num_classes, input_size, max_channels, dropout).to(device)
        model.load_state_dict(torch.load(f"{mdir}/model.pt", map_location=device))
        model.eval()

        val = pd.read_csv(f"{BASE}/split-definitions-kfold-{DEPTH}/validation_split_fold_{fold}.csv",
                           dtype={"site_id": str})
        sites = sorted(val.site_id.unique())
        print(f"[fold {fold}] {MODEL_SET} {DEPTH}: {len(sites)} validation sites, model={mdir}", flush=True)

        for si, site in enumerate(sites):
            d = build_site_daily(site)
            if d is None or len(d) < 250:
                continue
            d = normalize_site(d)
            windows = make_windows(d)
            if not windows:
                continue
            ml = stitch_site(windows, model, device)
            ml["date"] = pd.to_datetime(ml["date"])
            o = obs[obs.site_id == site][["date", "soil_moisture"]].rename(columns={"soil_moisture": "obs"})
            mt = ml.merge(o, on="date", how="inner").dropna()
            k, r, pb = kge(mt["ml"].to_numpy(), mt["obs"].to_numpy())
            all_rows.append((fold, site, len(mt), k, r, pb))
            if si % 20 == 0:
                print(f"  [{site}] n={len(mt)} KGE={k:.3f}" if np.isfinite(k) else f"  [{site}] n={len(mt)} KGE=NA", flush=True)

        del model; torch.cuda.empty_cache()

    R = pd.DataFrame(all_rows, columns=["fold", "site_id", "n_obs", "KGE", "r", "pbias"])
    out_csv = f"{OUT_DIR}/point_eval_{'_'.join(map(str, FOLDS))}.csv"
    R.to_csv(out_csv, index=False)
    v = R[R.n_obs >= 365]
    print(f"\n[{MODEL_SET}] {DEPTH}: {len(v)} robust sites | median KGE={v.KGE.median():.3f} "
          f"r={v.r.median():.3f} |pbias|={v.pbias.abs().median():.1f}")
    print(f"saved -> {out_csv}")

if __name__ == "__main__":
    main()
