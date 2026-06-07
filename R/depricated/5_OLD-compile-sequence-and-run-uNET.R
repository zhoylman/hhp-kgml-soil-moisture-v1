suppressPackageStartupMessages({
  library(terra)
  library(tidyverse)
  library(glue)
  library(stringr)
  library(lubridate)
})

# ---------------------------------------------------------------------
# Paths (edit only these if needed)
# ---------------------------------------------------------------------
daily_dir   = "/data/ssd4/soil-moisture-ml-inference/inference-rasters/historical-forcing-data-normalized"
stack_dir   = "/data/ssd4/soil-moisture-ml-inference/inference-rasters/inference-stack-temp"
pred_dir    = "/data/ssd4/soil-moisture-ml-inference/predictions-180-middle"        # full 180-band predictions (temp)
center_dir  = "/data/ssd4/soil-moisture-ml-inference/predictions-center60-middle"   # trimmed 60-band outputs
time_dir    = "/data/ssd4/soil-moisture-ml-inference/time-index-middle"             # CSVs for band->date mapping
smoothed_dir = "/data/ssd4/soil-moisture-ml-inference/predictions-smoothed-daily-middle"  # NEW: final blended daily TIFFs

dir.create(smoothed_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(stack_dir,  recursive = TRUE, showWarnings = FALSE)
dir.create(pred_dir,   recursive = TRUE, showWarnings = FALSE)
dir.create(center_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(time_dir,   recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------
# Scheduling knobs
# ---------------------------------------------------------------------
keep_len   <- 60L         # we keep the center 60 bands from each 180-day inference
stride_days <- 30L        # < 60 creates overlap; overlap = keep_len - stride_days
sigma_days <- 12          # Gaussian width (in days) for cross-fade; 8–15 works well

overlap <- keep_len - stride_days
stopifnot(overlap > 0)

# PyTorch inference script + model (use your env’s python binary)
python_bin   = "/home/zhoylman/miniconda3/envs/gpu-ml/bin/python"
infer_script = "/home/zhoylman/soil-moisture-ml/py/raster_inference.py"
model_path   = "/data/ssd2/soil-moisture-ml/results-kfold-middle/fold_1_uNET_fold_1_08_12_2025-16_34_27_163406/model.pt"

# Inference args (keep consistent with training)
timesteps          = 180L
channels_per_step  = 61L
tile_px            = 256L
overlap_px         = 32L
batch_size         = 1L
seq_chunk_px       = 4096L
device             = "cuda"  # or "auto"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

# Build a single 180-day *forcing* stack ending at d_end (Date) by reading daily TIFFs
build_180_stack = function(d_end, dir_daily = daily_dir) {
  dates = seq(d_end - 179, d_end, by = "1 day")
  paths = file.path(dir_daily, paste0("normalized-data-", as.character(dates), ".tif"))
  if (!all(file.exists(paths))) return(NULL)
  # Concatenate days (each file contains the 61 per-day features)
  rs = lapply(paths, rast)
  stk = rast(rs)
  names(stk) = paste0("day", sprintf("%03d_", seq_along(dates)), names(stk))
  stk
}

# Write band->date index CSV for either full 180 or center 60
write_time_csv = function(d_end, which = c("full180","center60")) {
  which = match.arg(which)
  full_dates = seq(d_end - 179, d_end, by = "1 day")
  if (which == "full180") {
    tibble(band = seq_along(full_dates), date = full_dates) |>
      write_csv(file.path(time_dir, glue("time-180-{as.character(d_end)}.csv")))
  } else {
    keep_idx   = 61:120
    tibble(band = seq_along(keep_idx), date = full_dates[keep_idx]) |>
      write_csv(file.path(time_dir, glue("time-center60-{as.character(d_end)}.csv")))
  }
}

# Run the Python inference once per stack, return 0 on success
run_inference = function(inp_tif, out_tif) {
  env = c(
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
    "GDAL_NUM_THREADS=ALL_CPUS",
    "GDAL_CACHEMAX=4096"
  )
  args = c(
    infer_script,
    "--model", model_path,
    "--inp",   inp_tif,
    "--out",   out_tif,
    "--timesteps",         as.character(timesteps),
    "--channels_per_step", as.character(channels_per_step),
    "--model_layout", "TC",
    "--write_mode",   "multiband",
    "--model_kind",   "seq",
    "--seq_chunk_px", as.character(seq_chunk_px),
    "--tile",         as.character(tile_px),
    "--overlap",      as.character(overlap_px),
    "--batch_size",   as.character(batch_size),
    "--device",       device
  )
  system2(command = python_bin, args = args, env = env, stdout = "", stderr = "")
}

# Trim the predicted 180-band raster to center 60 and write out
trim_center_60 = function(pred_180_path, d_end, out_dir = center_dir) {
  r180 = rast(pred_180_path)
  keep_idx = 61:120
  r60 = r180[[keep_idx]]
  # Name layers with ISO dates for clarity
  dates = seq(d_end - 179, d_end, by = "1 day")[keep_idx]
  names(r60) = paste0("vwc_", dates)
  out60 = file.path(out_dir, glue("pred-center60-{as.character(d_end)}.tif"))
  writeRaster(
    r60, out60, overwrite = TRUE,
    wopt = list(datatype = "FLT4S", gdal = c("COMPRESS=LZW","BIGTIFF=IF_SAFER"))
  )
  out60
}

# ---------------------------------------------------------------------
# Build schedule (stride=60, center-keep)
# ---------------------------------------------------------------------
daily_df = tibble(
  path = list.files(daily_dir, pattern = "^normalized-data-.*\\.tif$", full.names = TRUE)
) |>
  mutate(date = as.Date(str_extract(basename(path), "(?<=normalized-data-).*(?=\\.tif)"))) |>
  arrange(date)

stopifnot(nrow(daily_df) > 200)

all_dates = daily_df$date
# centers every 60 days, at least 90 days from edges so each 180d window has full context
# centers every stride_days; each 180-d window keeps center 60
centers <- all_dates[seq(90, length(all_dates) - 90, by = stride_days)]
ends    <- centers + 89

message("Total windows to process (stride=60): ", length(ends))

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
tictoc::tic()
for (d_end in ends) {
  d_end = as.Date(d_end, origin = '1970-01-01')
  message("\n=== Window ending ", d_end, " ===")
  
  # 1) Build forcing stack (180 days)
  stack_path = file.path(stack_dir, glue("inference-{as.character(d_end)}.tif"))
  if (!file.exists(stack_path)) {
    stk = build_180_stack(d_end)
    if (is.null(stk)) {
      message("  Skipping ", d_end, " (missing daily inputs).")
      next
    }
    writeRaster(
      stk, stack_path, overwrite = TRUE
      )
    rm(stk); gc()
  } else {
    message("  Forcing stack already exists.")
  }
  
  # 2) Write time index for the full 180 (handy for debugging/trace)
  write_time_csv(d_end, "full180")
  
  # 3) Run inference on this stack
  pred180_path = file.path(pred_dir, glue("pred-180-{as.character(d_end)}.tif"))
  if (!file.exists(pred180_path)) {
    status = run_inference(inp_tif = stack_path, out_tif = pred180_path)
    if (!identical(status, 0L)) {
      message("  Inference FAILED for ", d_end, " (status=", status, "). Keeping stack for debugging.")
      next
    }
  } else {
    message("  180-band prediction already exists.")
  }
  
  # 4) Trim to center 60 and write time index for the kept slice
  pred60_path = file.path(center_dir, glue("pred-center60-{as.character(d_end)}.tif"))
  if (!file.exists(pred60_path)) {
    pred60_path = trim_center_60(pred180_path, d_end)
    write_time_csv(d_end, "center60")
  }
  
  # 5) Clean up big intermediates to save space
  if (file.exists(stack_path)) file.remove(stack_path)
  #if (file.exists(pred180_path)) file.remove(pred180_path)
  
  # 6) Terra temp hygiene per iteration
  tf = try(terra::tmpFiles(), silent = TRUE)
  if (!inherits(tf, "try-error") && length(tf)) suppressWarnings(file.remove(tf))
  gc()
}
tictoc::toc()
message("\nAll windows processed. Center-60 predictions live in: ", center_dir)
message("Time-index CSVs live in: ", time_dir)


# ============================
# Stitch (Gaussian cross-fade)
# ============================

# 1) List and sort the 60-band center slices by their window end date (YYYY-MM-DD in filename)
center_df <- tibble(
  path = list.files(center_dir, pattern = "^pred-center60-\\d{4}-\\d{2}-\\d{2}\\.tif$", full.names = TRUE)
) |>
  mutate(d_end = as.Date(str_extract(basename(path), "\\d{4}-\\d{2}-\\d{2}"))) |>
  arrange(d_end)

stopifnot(nrow(center_df) >= 1)

# 2) Gaussian weights across the 60-day slice (center between 30 & 31 -> 30.5)
gaussian_weights <- function(n = keep_len, sigma = sigma_days, mid = (keep_len + 1) / 2) {
  i <- seq_len(n)
  exp(-0.5 * ((i - mid) / sigma)^2)
}
w <- gaussian_weights()

# 3) Helper: write a single-band day raster with safe GDAL options
write_day <- function(r, date) {
  out <- file.path(smoothed_dir, paste0("vwc_", as.character(date), ".tif"))
  writeRaster(
    r, out, overwrite = TRUE,
    wopt = list(datatype = "FLT4S", gdal = c("COMPRESS=LZW", "BIGTIFF=IF_SAFER"))
  )
}

# 4) Helper: dates covered by a center-60 file given its window end-date
#    180d window is centered at (d_end - 89). Kept 60d are [center-30 .. center+29]
dates_for_center60 <- function(d_end) {
  center <- d_end - 89
  seq(center - 30, center + 29, by = "1 day")
}

# 5) Blend function for two rasters (handles NAs)
blend_two <- function(a, b, w1, w2) {
  lapp(c(a, b), fun = function(aa, bb) {
    num <- ifelse(!is.na(aa), aa * w1, 0) + ifelse(!is.na(bb), bb * w2, 0)
    den <- ifelse(!is.na(aa), w1, 0)      + ifelse(!is.na(bb), w2, 0)
    ifelse(den > 0, num / den, NA_real_)
  })
}

# 6) Pairwise stitch across the time axis
#    Logic:
#      - For the first chunk: write its leading (keep_len - overlap) days directly.
#      - For each adjacent pair (k, k+1): cross-fade the last `overlap` days of k
#        with the first `overlap` days of k+1 (Gaussian), then write the interior
#        (non-overlap) days of k+1 directly.
#      - After the last chunk: write its trailing `overlap` days directly.

paths <- center_df$path
n_chunks <- length(paths)

# Open first chunk
r_prev   <- rast(paths[1])
dates_prev <- dates_for_center60(center_df$d_end[1])

# Leading non-overlap days of the first chunk: 1 .. (keep_len - overlap)
lead_n <- keep_len - overlap
if (lead_n > 0) {
  for (i in 1:lead_n) write_day(r_prev[[i]], dates_prev[i])
}

# Iterate over pairs
if (n_chunks >= 2) {
  for (k in 1:(n_chunks - 1)) {
    r_cur     <- rast(paths[k + 1])
    dates_cur <- dates_for_center60(center_df$d_end[k + 1])
    
    # --- Cross-fade overlap region ---
    # prev tail indices & cur head indices
    prev_idx <- (keep_len - overlap + 1):keep_len
    cur_idx  <- 1:overlap
    
    # weights for those bands
    w_prev <- w[prev_idx]
    w_cur  <- w[cur_idx]
    
    for (j in seq_len(overlap)) {
      # same calendar date in both chunks (by construction: stride < keep_len)
      dt <- dates_cur[j]  # equals dates_prev[length(dates_prev) - overlap + j]
      
      # blend and write
      out <- blend_two(r_prev[[prev_idx[j]]], r_cur[[cur_idx[j]]], w_prev[j], w_cur[j])
      write_day(out, dt)
    }
    
    # --- Write current chunk's interior (non-overlap on both sides) ---
    mid_start <- overlap + 1
    mid_end   <- keep_len - overlap
    if (mid_end >= mid_start) {
      for (i in mid_start:mid_end) write_day(r_cur[[i]], dates_cur[i])
    }
    
    # move forward: drop previous, keep current for next pair
    rm(r_prev); gc()
    r_prev     <- r_cur
    dates_prev <- dates_cur
    
    # tidy terra temp
    tf <- try(terra::tmpFiles(), silent = TRUE)
    if (!inherits(tf, "try-error") && length(tf)) suppressWarnings(file.remove(tf))
  }
}

# Tail of the last chunk (no next chunk to blend with): write directly
if (n_chunks >= 1) {
  tail_idx <- (keep_len - overlap + 1):keep_len
  if (tail_idx[1] <= keep_len) {
    for (i in tail_idx) write_day(r_prev[[i]], dates_prev[i])
  }
  rm(r_prev); gc()
}

message("Smoothed daily series written to: ", smoothed_dir)
