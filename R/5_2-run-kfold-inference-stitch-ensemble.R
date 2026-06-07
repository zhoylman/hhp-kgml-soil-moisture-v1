##############################################################
# Title: HHP Soil Moisture Inference – Multi-Fold (10-fold)
# Description:
#   - Toggle "middle" or "shallow" at the top.
#   - Builds shared 180-day forcing stacks per window.
#   - Runs each fold model on the same stack.
#   - Trims to center-60, stitches (Gaussian cross-fade) to daily.
#   - Ensembles across folds to median + IQR per day.
# Author: Zachary H. Hoylman
##############################################################

suppressPackageStartupMessages({
  library(terra)
  library(tidyverse)
  library(glue)
  library(stringr)
  library(lubridate)
  library(tictoc)
})

terraOptions(memfrac = 0.7)

# ---------------------------------------------------------------------
# MODEL SELECTION FLAG: choose "middle" or "shallow"
# ---------------------------------------------------------------------
depth_flag <- "shallow"   # <<< CHANGE HERE >>> 
stopifnot(depth_flag %in% c("middle", "shallow"))

# ---------------------------------------------------------------------
# Paths (auto-configured from depth_flag)
# ---------------------------------------------------------------------
daily_dir   = glue("/data/ssd4/soil-moisture-ml-inference/inference-rasters/historical-forcing-data-normalized")
stack_dir   = glue("/data/ssd4/soil-moisture-ml-inference/inference-rasters/inference-stack-temp-{depth_flag}")
pred_dir    = glue("/data/ssd3/soil-moisture-ml-inference/predictions-180-{depth_flag}")         # 180-band per fold
center_dir  = glue("/data/ssd3/soil-moisture-ml-inference/predictions-center60-{depth_flag}")    # 60-band per fold
smoothed_dir= glue("/data/ssd3/soil-moisture-ml-inference/predictions-smoothed-daily-{depth_flag}") # daily stitched per fold
time_dir    = glue("/data/ssd3/soil-moisture-ml-inference/time-index-{depth_flag}")               # band->date CSVs

# Per-depth models root
models_root = glue("/data/ssd2/soil-moisture-ml/results-kfold-{depth_flag}")

# Ensemble outputs (across folds)
ensemble_root      = glue("/data/ssd3/soil-moisture-ml-inference/ensemble-smoothed-daily-{depth_flag}")
ensemble_median_dir= file.path(ensemble_root, "median")
ensemble_iqr_dir   = file.path(ensemble_root, "iqr")

# Create directories
dirs_to_make = c(stack_dir, pred_dir, center_dir, smoothed_dir, time_dir,
                 ensemble_median_dir, ensemble_iqr_dir)
invisible(lapply(dirs_to_make, dir.create, recursive = TRUE, showWarnings = FALSE))

# ---------------------------------------------------------------------
# Model / inference configuration
# ---------------------------------------------------------------------
python_bin   = "/home/zhoylman/miniconda3/envs/gpu-ml/bin/python"
infer_script = "/home/zhoylman/hhp-kgml-soil-moisture-v1/py/raster_inference.py"
timesteps          = 180L
channels_per_step  = 61L
tile_px            = 512L
overlap_px         = 0L
batch_size         = 1L
seq_chunk_px       = 4096L
device             = "cuda"         # or "auto"
# I/O overlap + pinned memory (for raster_inference.py prefetcher)
read_workers       = 16L    # try 8–16
prefetch_batches   = 2L    # 1–2 is plenty
pin_memory_flag    = TRUE  # adds --pin_memory when TRUE

# ---------------------------------------------------------------------
# Scheduling knobs
# ---------------------------------------------------------------------
keep_len   <- 60L         # keep center 60 days from each 180-day inference
stride_days <- 30L        # step between window centers; overlap = keep_len - stride_days
sigma_days <- 12          # Gaussian width for cross-fade (8–15 works well)

overlap <- keep_len - stride_days
stopifnot(overlap > 0, sigma_days < overlap)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
find_fold_models <- function(root) {
  dirs <- list.dirs(root, recursive = FALSE, full.names = TRUE)
  if (length(dirs) == 0) return(character(0))
  candidates <- file.path(dirs, "model.pt")
  keep <- file.exists(candidates)
  if (!any(keep)) return(character(0))
  folds <- basename(dirs[keep])
  fold_ids <- sub("^(fold_\\d+).*", "\\1", folds)
  setNames(candidates[keep], fold_ids)
}
fold_models <- find_fold_models(models_root)
if (length(fold_models) == 0) stop("No fold model.pt files found in: ", models_root)
message("Discovered fold models: ", paste(names(fold_models), collapse = ", "))

# Per-fold output dirs
pred_dir_f      <- setNames(file.path(pred_dir,      names(fold_models)), names(fold_models))
center_dir_f    <- setNames(file.path(center_dir,    names(fold_models)), names(fold_models))
smoothed_dir_f  <- setNames(file.path(smoothed_dir,  names(fold_models)), names(fold_models))
invisible(lapply(c(pred_dir_f, center_dir_f, smoothed_dir_f),
                 dir.create, recursive = TRUE, showWarnings = FALSE))

# Build a single 180-day forcing stack ending at d_end
build_180_stack = function(d_end, dir_daily = daily_dir) {
  dates = seq(d_end - 179, d_end, by = "1 day")
  paths = file.path(dir_daily, paste0("normalized-data-", as.character(dates), ".tif"))
  if (!all(file.exists(paths))) return(NULL)
  rs = lapply(paths, rast)
  stk = rast(rs)
  names(stk) = paste0("day", sprintf("%03d_", seq_along(dates)), names(stk))
  stk
}

write_time_csv = function(d_end, which = c("full180","center60")) {
  which = match.arg(which)
  full_dates = seq(d_end - 179, d_end, by = "1 day")
  if (which == "full180") {
    tibble(band = seq_along(full_dates), date = full_dates) |>
      write_csv(file.path(time_dir, glue("time-180-{as.character(d_end)}.csv")))
  } else {
    keep_idx = 61:120
    tibble(band = seq_along(keep_idx), date = full_dates[keep_idx]) |>
      write_csv(file.path(time_dir, glue("time-center60-{as.character(d_end)}.csv")))
  }
}
run_inference = function(inp_tif, out_tif, model_path) {
  env = c(
    "CUDA_DEVICE_ORDER=PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES=1",                           # use the A6000
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256",
    "GDAL_NUM_THREADS=16",                              # moderate; won’t starve cron
    "GDAL_CACHEMAX=8192",                               # MB (8 GB)
    "OMP_NUM_THREADS=16",
    "OPENBLAS_NUM_THREADS=16",
    "MKL_NUM_THREADS=16",
    "NUMEXPR_MAX_THREADS=16",
    "OMP_WAIT_POLICY=PASSIVE",                          # reduce CPU spin while GPU runs
    "MALLOC_ARENA_MAX=4"                                # avoid glibc arena bloat
  )
  logf <- sub("\\.tif$", ".log", out_tif)
  
  args = c(
    infer_script, "--model", model_path,
    "--inp", inp_tif, "--out", out_tif,
    "--timesteps", as.character(timesteps),
    "--channels_per_step", as.character(channels_per_step),
    "--model_layout","TC",
    "--write_mode","multiband",
    "--model_kind","seq",
    "--seq_chunk_px", as.character(seq_chunk_px),
    "--tile", as.character(tile_px),
    "--overlap", as.character(overlap_px),
    "--batch_size", as.character(batch_size),
    "--device", device,
    # NEW: overlap I/O / H2D copies, pinned memory
    "--read_workers", as.character(read_workers),
    "--prefetch_batches", as.character(prefetch_batches)
  )
  
  if (isTRUE(pin_memory_flag)) {
    args <- c(args, "--pin_memory")
  }
  
  # Returns 0 on success; tee logs to .log file
  system2(command = python_bin, args = args, env = env, stdout = logf, stderr = logf)
}


gaussian_weights <- function(n = keep_len, sigma = sigma_days, mid = (keep_len + 1) / 2) {
  i <- seq_len(n)
  exp(-0.5 * ((i - mid) / sigma)^2)
}
dates_for_center60 <- function(d_end) {
  center <- d_end - 89
  seq(center - 30, center + 29, by = "1 day")
}
blend_two <- function(a, b, w1, w2) {
  lapp(c(a, b), fun = function(aa, bb) {
    num <- ifelse(!is.na(aa), aa * w1, 0) + ifelse(!is.na(bb), bb * w2, 0)
    den <- ifelse(!is.na(aa), w1, 0)      + ifelse(!is.na(bb), w2, 0)
    ifelse(den > 0, num / den, NA_real_)
  })
}
write_day <- function(r, date, out_dir) {
  out <- file.path(out_dir, paste0("vwc_", as.character(date), ".tif"))
  writeRaster(r, out, overwrite = TRUE,
              wopt = list(datatype = "FLT4S", gdal = c("COMPRESS=LZW","BIGTIFF=IF_SAFER","TILED=YES")))
}

# ---------------------------------------------------------------------
# Build schedule
# ---------------------------------------------------------------------
daily_df = tibble(
  path = list.files(daily_dir, pattern = "^normalized-data-.*\\.tif$", full.names = TRUE)
) |>
  mutate(date = as.Date(str_extract(basename(path), "(?<=normalized-data-).*(?=\\.tif)"))) |>
  arrange(date)

if (nrow(daily_df) <= 200) stop("Not enough daily files to cover 180-day windows.")
all_dates = daily_df$date
centers <- all_dates[seq(90, length(all_dates) - 90, by = stride_days)]
ends    <- centers + 89

# Find the latest end-date that ALL folds have as pred-180-*.tif
latest_complete_end = function(pred_dir_f) {
  last_by_fold = vapply(
    names(pred_dir_f),
    function(fid) {
      f = list.files(
        pred_dir_f[[fid]],
        pattern = "^pred-180-\\d{4}-\\d{2}-\\d{2}\\.tif$",
        full.names = FALSE
      )
      if (!length(f)) return(as.Date(NA))
      max(as.Date(stringr::str_extract(f, "\\d{4}-\\d{2}-\\d{2}")))
    },
    FUN.VALUE = as.Date(NA)
  )
  if (all(is.na(last_by_fold))) return(as.Date(NA))
  # Resume after the minimum "last done" across folds (guarantees all folds complete up to that date)
  min(last_by_fold, na.rm = TRUE)
}

# ...after you compute `ends` originally
res_done = latest_complete_end(pred_dir_f)
if (!is.na(res_done)) {
  ends = ends[ends > res_done]
  message(
    "Auto-resume: all folds complete through ", res_done,
    " | remaining windows: ", length(ends)
  )
} else {
  message("Auto-resume: no completed windows detected; starting from the beginning.")
}

message("Total windows to process (stride=", stride_days, "): ", length(ends))

# ---------------------------------------------------------------------
# Main loop – build stack once, run all folds on it
# ---------------------------------------------------------------------
tictoc::tic(glue("Running {depth_flag} models"))
for (d_end in ends) {
  d_end = as.Date(d_end)
  message("\n=== Window ending ", d_end, " ===")
  
  # 1) Build forcing stack (shared by all folds for this window)
  stack_path = file.path(stack_dir, glue("inference-{as.character(d_end)}.tif"))
  if (!file.exists(stack_path)) {
    stk = build_180_stack(d_end)
    if (is.null(stk)) {
      message("  Skipping ", d_end, " (missing daily inputs).")
      next
    }
    writeRaster(stk, stack_path, overwrite = TRUE,
                wopt = list(datatype = "FLT4S", gdal = c("COMPRESS=LZW","BIGTIFF=IF_SAFER","TILED=YES")))
    rm(stk); gc()
  } else {
    message("  Forcing stack already exists.")
  }
  
  # 2) Write time index CSVs
  write_time_csv(d_end, "full180")
  write_time_csv(d_end, "center60")
  
  # 3) Run inference for each fold on this stack; trim to center-60
  for (fold_id in names(fold_models)) {
    model_path   <- fold_models[[fold_id]]
    pred180_path <- file.path(pred_dir_f[[fold_id]], glue("pred-180-{as.character(d_end)}.tif"))
    pred60_path  <- file.path(center_dir_f[[fold_id]], glue("pred-center60-{as.character(d_end)}.tif"))
    
    if (!file.exists(pred180_path)) {
      message("  [", fold_id, "] inference …")
      status = run_inference(stack_path, pred180_path, model_path)
      if (!identical(status, 0L)) {
        message("  [", fold_id, "] FAILED for ", d_end, " (status=", status, ").")
        next
      }
    } else {
      message("  [", fold_id, "] 180-band prediction already exists.")
    }
    
    if (!file.exists(pred60_path)) {
      r180  = rast(pred180_path)
      r60   = r180[[61:120]]
      dates = seq(d_end - 179, d_end, by = "1 day")[61:120]
      names(r60) = paste0("vwc_", dates)
      writeRaster(r60, pred60_path, overwrite = TRUE,
                  wopt = list(datatype = "FLT4S", gdal = c("COMPRESS=LZW","BIGTIFF=IF_SAFER","TILED=YES")))
      rm(r180, r60); gc()
    } else {
      message("  [", fold_id, "] center-60 already exists.")
    }
  }
  
  # 4) Clean up big intermediate forcing stack
  if (file.exists(stack_path)) file.remove(stack_path)
  
  # 5) Terra temp hygiene
  tf = try(terra::tmpFiles(), silent = TRUE)
  if (!inherits(tf, "try-error") && length(tf)) suppressWarnings(file.remove(tf))
  gc()
}
tictoc::toc()
message("\nAll windows processed. Center-60 per-fold outputs in: ", center_dir)

# ============================
# Stitch (Gaussian cross-fade) PER FOLD -> daily TIFFs
# ============================
w <- gaussian_weights()

for (fold_id in names(fold_models)) {
  message("\n--- Stitching fold: ", fold_id, " ---")
  fold_center_dir   <- center_dir_f[[fold_id]]
  fold_smoothed_dir <- smoothed_dir_f[[fold_id]]
  dir.create(fold_smoothed_dir, recursive = TRUE, showWarnings = FALSE)
  
  center_df <- tibble(
    path = list.files(fold_center_dir, pattern = "^pred-center60-\\d{4}-\\d{2}-\\d{2}\\.tif$", full.names = TRUE)
  ) |>
    mutate(d_end = as.Date(str_extract(basename(path), "\\d{4}-\\d{2}-\\d{2}"))) |>
    arrange(d_end)
  
  if (nrow(center_df) == 0) {
    message("  No center-60 files found for ", fold_id, ". Skipping.")
    next
  }
  
  paths <- center_df$path
  n_chunks <- length(paths)
  
  # Open first chunk
  r_prev     <- rast(paths[1])
  dates_prev <- dates_for_center60(center_df$d_end[1])
  
  # Write leading non-overlap days directly
  lead_n <- keep_len - overlap
  if (lead_n > 0) {
    for (i in 1:lead_n) write_day(r_prev[[i]], dates_prev[i], fold_smoothed_dir)
  }
  
  # Iterate over adjacent pairs
  if (n_chunks >= 2) {
    for (k in 1:(n_chunks - 1)) {
      r_cur     <- rast(paths[k + 1])
      dates_cur <- dates_for_center60(center_df$d_end[k + 1])
      
      # Cross-fade overlap
      prev_idx <- (keep_len - overlap + 1):keep_len
      cur_idx  <- 1:overlap
      w_prev <- w[prev_idx]
      w_cur  <- w[cur_idx]
      
      for (j in seq_len(overlap)) {
        dt <- dates_cur[j]
        out <- blend_two(r_prev[[prev_idx[j]]], r_cur[[cur_idx[j]]], w_prev[j], w_cur[j])
        write_day(out, dt, fold_smoothed_dir)
      }
      
      # Write interior of current chunk directly
      mid_start <- overlap + 1
      mid_end   <- keep_len - overlap
      if (mid_end >= mid_start) {
        for (i in mid_start:mid_end) write_day(r_cur[[i]], dates_cur[i], fold_smoothed_dir)
      }
      
      rm(r_prev); gc()
      r_prev     <- r_cur
      dates_prev <- dates_cur
      
      tf <- try(terra::tmpFiles(), silent = TRUE)
      if (!inherits(tf, "try-error") && length(tf)) suppressWarnings(file.remove(tf))
    }
  }
  
  # Tail of the last chunk
  if (n_chunks >= 1) {
    tail_idx <- (keep_len - overlap + 1):keep_len
    if (tail_idx[1] <= keep_len) {
      for (i in tail_idx) write_day(r_prev[[i]], dates_prev[i], fold_smoothed_dir)
    }
    rm(r_prev); gc()
  }
  
  message("  Smoothed daily series for ", fold_id, " -> ", fold_smoothed_dir)
}

message("\nAll folds stitched to daily rasters in: ", smoothed_dir)

# ============================
# ENSEMBLE: Median and IQR across folds (future_map version)
# ============================
library(furrr)
plan(multisession, workers = 96)  # or multicore on Linux

# 1. Collect all per-fold daily rasters
fold_files = purrr::map_dfr(names(fold_models), function(fid) {
  fdir = smoothed_dir_f[[fid]]
  files = list.files(fdir, pattern = "^vwc_\\d{4}-\\d{2}-\\d{2}\\.tif$", full.names = TRUE)
  if (!length(files)) return(NULL)
  tibble(fold = fid,
         path = files,
         date = as.Date(str_extract(basename(files), "\\d{4}-\\d{2}-\\d{2}")))
})

# 2. Skip if nothing found
if (nrow(fold_files) == 0) stop("No daily stitched files found in any fold.")

# 3. Identify all dates and those already processed
dates_all = sort(unique(fold_files$date))
done_median = list.files(ensemble_median_dir, pattern = "^vwc_\\d{4}-\\d{2}-\\d{2}\\.tif$", full.names = FALSE)
done_iqr    = list.files(ensemble_iqr_dir,    pattern = "^vwc_\\d{4}-\\d{2}-\\d{2}\\.tif$", full.names = FALSE)
dates_done  = sort(intersect(
  as.Date(str_extract(done_median, "\\d{4}-\\d{2}-\\d{2}")),
  as.Date(str_extract(done_iqr,    "\\d{4}-\\d{2}-\\d{2}"))
))

# 4. Build a list-column mapping each date → file paths
files_by_date = fold_files |>
  filter(!(date %in% dates_done)) |>   # skip already done
  group_by(date) |>
  summarise(paths = list(path), .groups = "drop")

message("Total dates: ", length(dates_all),
        " | already done: ", length(dates_done),
        " | remaining: ", nrow(files_by_date))

# 5. Function to compute median + IQR for one date
ensemble_one_day = function(date, paths) {
  out_median = file.path(ensemble_median_dir, paste0("vwc_", date, ".tif"))
  out_iqr    = file.path(ensemble_iqr_dir,    paste0("vwc_", date, ".tif"))
  if (file.exists(out_median) && file.exists(out_iqr)) return(invisible(TRUE))
  
  # Read and compute quantiles
  X = terra::rast(paths)
  QQ = terra::app(X, fun = function(v) {
    qs = quantile(v, probs = c(0.25, 0.5, 0.75), na.rm = TRUE, names = FALSE)
    c(qs[2], qs[3] - qs[1])  # median, iqr
  })
  names(QQ) = c("median","iqr")
  
  # Write outputs
  terra::writeRaster(QQ[[1]], out_median, overwrite = TRUE,
                     wopt = list(datatype = "FLT4S", gdal = c("COMPRESS=LZW","BIGTIFF=IF_SAFER","TILED=YES")))
  terra::writeRaster(QQ[[2]], out_iqr, overwrite = TRUE,
                     wopt = list(datatype = "FLT4S", gdal = c("COMPRESS=LZW","BIGTIFF=IF_SAFER","TILED=YES")))
  
  rm(X, QQ); gc()
  invisible(TRUE)
}

# 6. Parallel ensemble using future_map2
tictoc::tic("Parallel ensemble (future_map, 4 workers)")
results = furrr::future_map2(
  files_by_date$date, files_by_date$paths,
  ~ ensemble_one_day(.x, .y),
  .progress = TRUE
)
tictoc::toc()

# 7. Final report
n_done = sum(unlist(results))
message("Ensemble complete: ", n_done, " new daily rasters written.")
message("Median in: ", ensemble_median_dir, " | IQR in: ", ensemble_iqr_dir)

# Cleanup
tf = try(terra::tmpFiles(), silent = TRUE)
if (!inherits(tf, "try-error") && length(tf)) suppressWarnings(file.remove(tf))
gc()
plan(sequential)
message("\nDONE (depth_flag = ", depth_flag, ").")