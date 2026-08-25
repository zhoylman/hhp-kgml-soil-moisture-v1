##############################################################
# Title: Extract per-site daily 61-band forcing series directly from the
#   canonical gridded forcing archive (historical-forcing-data-normalized).
# Description:
#   The point-based center-keep eval pipeline (py/exp_point_centerkeep_eval.py)
#   previously built per-site driver series from
#   observations/../seq-data/observational-sites-seq-data.csv, which has real
#   gaps for many sites -> make_windows()'s strict 180-day contiguity check
#   drops large numbers of candidate windows for gappy sites, roughly halving
#   effective n_obs for ~40+ sites and causing Table 1 (canonical gridded,
#   n=727 shallow) vs Table 2 (point-based ablation, n=685 shallow) to diverge.
#   The canonical gridded pipeline never has this problem: every day in the
#   forcing archive has a complete, gap-free raster covering all of CONUS by
#   construction. This script extracts, at each station's pixel (NEAREST
#   neighbor, matching terra::extract's default -- the SAME convention used
#   by sm_eval_utils.R's extract_at_sites() for Table 1's canonical
#   extraction), the full 61-band ALREADY-NORMALIZED feature vector for every
#   day in the archive. Output is a drop-in per-site daily feature table
#   (time + 61 named columns), gap-free by construction, requiring NO further
#   rolling-sum computation or normalization downstream -- both are already
#   baked into the archive.
##############################################################

suppressPackageStartupMessages({ library(terra); library(tidyverse); library(glue); library(furrr) })

repo       = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
data_root  = "/data/ssd2/soil-moisture-ml"
daily_dir  = "/data/ssd4/soil-moisture-ml-inference/inference-rasters/historical-forcing-data-normalized"
out_dir    = glue("{data_root}/seq-data/site-forcing-from-raster")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

meta = read_csv(glue("{data_root}/observations/final-soil-moisture-data-generalized-meta.csv"), show_col_types = FALSE) |>
  distinct(site_id, .keep_all = TRUE) |>
  arrange(site_id)
cat("Sites to extract:", nrow(meta), "\n")

files_tbl = tibble(path = list.files(daily_dir, pattern = "^normalized-data-.*\\.tif$", full.names = TRUE)) |>
  mutate(date = as.Date(str_extract(basename(path), "(?<=normalized-data-).*(?=\\.tif)"))) |>
  filter(!is.na(date)) |>
  arrange(date)
cat("Daily raster files:", nrow(files_tbl), " (", min(files_tbl$date), "to", max(files_tbl$date), ")\n")

band_names = names(rast(files_tbl$path[1]))
stopifnot(length(band_names) == 61)
cat("Bands (n=", length(band_names), "):", paste(band_names, collapse=","), "\n")

xy = as.matrix(meta[, c("longitude", "latitude")])

# ---- resume support: skip a batch if its output already exists ----
batch_size = 200L
n_files = nrow(files_tbl)
batch_starts = seq(1, n_files, by = batch_size)
batch_dir = glue("{out_dir}/.batches")
dir.create(batch_dir, showWarnings = FALSE, recursive = TRUE)

extract_batch = function(bstart) {
  bend = min(bstart + batch_size - 1, n_files)
  out_rds = glue("{batch_dir}/batch_{sprintf('%06d', bstart)}.rds")
  if (file.exists(out_rds)) return(invisible(TRUE))
  sub = files_tbl[bstart:bend, ]
  res = vector("list", nrow(sub))
  for (i in seq_len(nrow(sub))) {
    r = rast(sub$path[i])
    ext = terra::extract(r, xy)   # default method="simple" (nearest) -- matches extract_at_sites()
    res[[i]] = as.matrix(ext)     # [n_sites x 61]
  }
  arr = array(NA_real_, dim = c(nrow(sub), nrow(meta), length(band_names)))
  for (i in seq_len(nrow(sub))) arr[i, , ] = res[[i]]
  saveRDS(list(dates = sub$date, arr = arr), out_rds)
  invisible(TRUE)
}

cat("\nExtracting", length(batch_starts), "batches of", batch_size, "files each (", n_files, "files total)...\n")
plan(multisession, workers = 32)
invisible(future_walk(batch_starts, extract_batch,
                       .options = furrr_options(seed = NULL,
                                                 globals = c("extract_batch","files_tbl","xy","meta","band_names","batch_size","n_files","batch_dir"),
                                                 packages = c("terra","glue"))))
plan(sequential)
cat("All batches extracted.\n")

# ---- assemble: one CSV per site (time + 61 named feature columns) ----
cat("\nAssembling per-site CSVs...\n")
batch_files = sort(list.files(batch_dir, pattern = "^batch_.*\\.rds$", full.names = TRUE))
all_dates = vector("list", length(batch_files))
site_mats = vector("list", nrow(meta))
for (s in seq_len(nrow(meta))) site_mats[[s]] = vector("list", length(batch_files))

for (bi in seq_along(batch_files)) {
  b = readRDS(batch_files[bi])
  all_dates[[bi]] = b$dates
  for (s in seq_len(nrow(meta))) {
    site_mats[[s]][[bi]] = b$arr[, s, ]
  }
}
dates_full = do.call(c, all_dates)

write_site = function(s) {
  mat = do.call(rbind, site_mats[[s]])
  colnames(mat) = band_names
  df = as_tibble(mat) |> mutate(time = dates_full, .before = 1)
  write_csv(df, glue("{out_dir}/{meta$site_id[s]}.csv"))
  invisible(TRUE)
}
plan(multisession, workers = 32)
invisible(future_walk(seq_len(nrow(meta)), write_site,
                       .options = furrr_options(seed = NULL,
                                                 globals = c("write_site","site_mats","dates_full","band_names","meta","out_dir"),
                                                 packages = c("tidyverse","glue"))))
plan(sequential)

cat("\nDONE. Wrote", nrow(meta), "per-site CSVs to", out_dir, "\n")
