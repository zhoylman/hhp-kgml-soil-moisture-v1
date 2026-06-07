# =============================================================================
#  KGML Soil Moisture – GRIDMET Feature Builder & Inference Stack Generator
#  Project: UMRB Knowledge-Guided Machine Learning (KGML) Soil Moisture
#  Author:  Dr. Zachary Hoylman (Montana Climate Office)
#  Contact: <your email / phone>
#  Date:    <YYYY-MM-DD>
# -----------------------------------------------------------------------------
#  PURPOSE
#    Build day-by-day, normalized multi-band feature rasters from the full
#    GRIDMET archive and aligned statics, then assemble rolling 180×61 stacks
#    for ML inference (with optional 60-day center-keep scheduling).
#
#  INPUTS (read-only)
#    - GRIDMET archive (daily NetCDF + *_time.csv):  /data/ssd2/gridmet
#    - Static rasters (aligned to template):         /data/ssd2/soil-moisture-ml/static-data-rasters
#    - Min–max tables (CSV):                         /data/ssd2/soil-moisture-ml/min-max-definitions/
#
#  OUTPUTS (write)
#    - Daily normalized feature TIFFs:               /data/ssd4/soil-moisture-ml-inference/inference-rasters/historical-forcing-data-normalized/
#      (filename: normalized-data-YYYY-MM-DD.tif)
#    - 180×61 inference stacks:                      /data/ssd4/soil-moisture-ml-inference/inference-rasters/inference-stack-temp/
#      (e.g., current-inference.tif, inference-YYYY-MM-DD.tif)
#
#  KEY FEATURES (per day)
#    GRIDMET snapshots (bi, fm100, fm1000, pet, pr, srad, tmmn, tmmx, vpd),
#    rolling sums (7/15/30/60/365/730-day precip & PET), deficits (15/30/60/365/730),
#    year, circular_yday, plus aligned static predictors (PRISM, terrain, soil).
#
#  NORMALIZATION
#    Applies pre-established min–max by band name; circular_yday is passed through.
#
#  PERFORMANCE NOTES
#    - Parallelized with {foreach} + {doSNOW}; opens GRIDMET once per worker.
#    - Intersects dates across all needed variables; enforces ≥730-day history.
#    - Produces BIGTIFF with LZW compression; cleans terra temp files.
#
#  USAGE
#    - Set paths below; run end-to-end to generate:
#        1) Daily normalized features for all viable dates (≥730 days history)
#        2) Rolling 180-day stacks (stride=1 and/or stride=60 center-keep)
#
#  DEPENDENCIES
#    R >= 4.2; terra, sf, lubridate, glue, foreach, doSNOW, parallel, purrr, dplyr, readr, stringr
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(sf)
  library(terra)
  library(glue)
  library(lubridate)
  library(foreach)
  library(doSNOW)
  library(parallel)
  library(purrr)
  library(readr)
  library(stringr)
})

# ------------------------
# Paths & config
# ------------------------
gridmet_dir   = "/data/ssd2/gridmet"  # long archive ("/data/ssd2/gridmet") or operational ("/data/ssd2/gridmet-operational")
static_dir    = "/data/ssd2/soil-moisture-ml/static-data-rasters"
minmax_seq    = "/data/ssd2/soil-moisture-ml/min-max-definitions/seq-min-max-definitions-pretrain.csv"
minmax_static = "/data/ssd2/soil-moisture-ml/min-max-definitions/static-min-max-definitions-pretrain.csv"

out_daily_dir = "/data/ssd4/soil-moisture-ml-inference/inference-rasters/historical-forcing-data-normalized" # daily feature stacks
out_stack_dir = "/data/ssd4/soil-moisture-ml-inference/inference-rasters/inference-stack-temp"               # 180x61 stacks

dir.create(out_daily_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(out_stack_dir, recursive = TRUE, showWarnings = FALSE)

terraOptions(progress = 1)

# ------------------------
# Helpers (driver scope)
# ------------------------

read_gridmet_with_names = function(dir, time_col = "datetime") {
  ncs   = list.files(dir, pattern = "^gridmet_.*\\.nc$", full.names = TRUE)
  times = list.files(dir, pattern = "^gridmet_.*_time\\.csv$", full.names = TRUE)
  
  stem = function(p) str_remove(basename(p), "(\\.nc|_time\\.csv)$")
  time_lookup = set_names(times, stem(times))
  
  rasters = map(ncs, function(nc_path) {
    s = stem(nc_path)  # e.g., gridmet_pr
    time_csv = time_lookup[[s]]
    if (is.null(time_csv)) {
      warning("No matching *_time.csv for ", basename(nc_path))
      return(NULL)
    }
    tdf = suppressMessages(readr::read_csv(time_csv, show_col_types = FALSE))
    stopifnot(time_col %in% names(tdf))
    nm_chr = as.character(tdf[[time_col]])          # ISO strings; may include " 00:00:00"
    r = terra::rast(nc_path)
    if (terra::nlyr(r) == length(nm_chr)) names(r) = nm_chr
    r
  })
  
  names(rasters) = basename(ncs)
  rasters = rasters[!vapply(rasters, is.null, logical(1))]
  rasters
}

# robust date parsing from layer names
names_to_dates = function(r) as.Date(sub(" .*", "", names(r)))

# get single layer at date d (Date)
layer_at_date = function(r, d) {
  d  = as.Date(d)
  ds = names_to_dates(r)
  idx = which(ds == d)
  if (!length(idx)) {
    stop(sprintf("Layer not found for date %s (avail %s .. %s)",
                 as.character(d),
                 as.character(min(ds, na.rm=TRUE)),
                 as.character(max(ds, na.rm=TRUE))))
  }
  r[[idx]]
}

# sum previous n days INCLUDING d (window: d-(n-1) .. d)
sum_prev_n_by_date = function(r, d, n) {
  d  = as.Date(d)
  ds = names_to_dates(r)
  ok = which(ds <= d & ds > d - n)
  if (!length(ok)) {
    stop(sprintf("No layers in window [%s .. %s]",
                 as.character(d - n + 1), as.character(d)))
  }
  terra::app(r[[ok]], 'sum', na.rm = TRUE)
}

# lon/lat rasters aligned to template
make_lonlat_rasters = function(template, mask = TRUE) {
  base = template[[1]]
  lon  = terra::init(base, fun = "x"); names(lon) = "longitude"
  lat  = terra::init(base, fun = "y"); names(lat) = "latitude"
  if (mask) {
    lon = terra::mask(lon, base)
    lat = terra::mask(lat, base)
  }
  list(lon = lon, lat = lat)
}

# date → circular year-day feature
circular_yday_raster = function(template, d) {
  angle = 2 * pi * lubridate::yday(d) / 365
  val   = (sin(angle) + 1) / 2
  out   = (template[[1]] * 0) + val
  names(out) = "circular_yday"
  out
}

# normalize with pre-established stats
normalize_stack = function(r, mm_df, skip = c("circular_yday"), clamp01 = FALSE) {
  stopifnot(inherits(r, "SpatRaster"))
  lyr_out = lapply(names(r), function(nm) {
    lyr = r[[nm]]
    if (nm %in% skip) return(lyr)
    mn = mm_df$value[mm_df$name == nm & mm_df$min_max_id == "min"]
    mx = mm_df$value[mm_df$name == nm & mm_df$min_max_id == "max"]
    if (length(mn) == 0 || length(mx) == 0 || any(is.na(c(mn, mx))) || isTRUE(all.equal(mn, mx))) return(lyr)
    out = (lyr - mn) / (mx - mn)
    if (clamp01) {
      out = terra::ifel(out < 0, 0, out)
      out = terra::ifel(out > 1, 1, out)
    }
    out
  })
  out = terra::rast(lyr_out)
  names(out) = names(r)
  out
}

align_rasters_to_template = function(dir, template,
                                      pattern = "\\.(tif|tiff|img|nc)$",
                                      method = "bilinear",
                                      crop_to_template = TRUE,
                                      mask_with_template = TRUE,
                                      verbose = TRUE) {
  files = list.files(dir, pattern = pattern, full.names = TRUE, recursive = FALSE)
  if (!length(files)) stop("No rasters found in '", dir, "'")
  out = vector("list", length(files))
  names(out) = basename(files)
  for (i in seq_along(files)) {
    f = files[i]
    if (verbose) message(sprintf("[%d/%d] %s", i, length(files), basename(f)))
    r0 = terra::rast(f)
    r1 = terra::project(r0, template, method = method)
    if (crop_to_template) r1 = terra::crop(r1, template)
    r2 = terra::resample(r1, template, method = method)
    if (mask_with_template) r2 = terra::mask(r2, template)
    out[[i]] = r2
  }
  out
}

# ------------------------
# Discover GRIDMET + dates
# ------------------------

gm_tmp = read_gridmet_with_names(gridmet_dir)

# pick a template: prefer VPD if present, else fall back to the first product
template0 = if ("gridmet_vpd.nc" %in% names(gm_tmp)) {
  gm_tmp[["gridmet_vpd.nc"]][[terra::nlyr(gm_tmp[["gridmet_vpd.nc"]])]]
} else {
  gm_tmp[[1]][[terra::nlyr(gm_tmp[[1]])]]
}

# variables actually used in the daily build
vars_needed = c("gridmet_bi.nc","gridmet_fm100.nc","gridmet_fm1000.nc",
                 "gridmet_pet.nc","gridmet_pr.nc","gridmet_srad.nc",
                 "gridmet_tmmn.nc","gridmet_tmmx.nc","gridmet_vpd.nc")

missing_vars = setdiff(vars_needed, names(gm_tmp))
if (length(missing_vars)) stop("Missing GRIDMET products: ", paste(missing_vars, collapse = ", "))

date_lists_needed = lapply(gm_tmp[vars_needed], names_to_dates)
# --- after: date_lists_needed <- lapply(gm_tmp[vars_needed], names_to_dates)
common_dates = Reduce(intersect, date_lists_needed)

# Intersect can drop the Date class; re-assert it
if (!inherits(common_dates, "Date")) {
  common_dates = as.Date(common_dates, origin = "1970-01-01")
}
common_dates = sort(common_dates)

min_needed   = 730L - 1L
target_dates = common_dates[common_dates >= (min(common_dates) + min_needed)]

#optional to infil
#target_dates = target_dates[target_dates >= as.Date('2025-01-01')]

message(sprintf("Target date span: %s .. %s (%s days)",
                format(min(target_dates), "%Y-%m-%d"),
                format(max(target_dates), "%Y-%m-%d"),
                length(target_dates)))

# ------------------------
# Align/calc static rasters ONCE (driver)
# ------------------------

template = template0

static_data = align_rasters_to_template(
  dir = static_dir,
  template = template,
  pattern = "\\.(tif|nc)$",
  method = "bilinear",
  crop_to_template = TRUE,
  mask_with_template = TRUE,
  verbose = FALSE
)

lonlat = make_lonlat_rasters(template, mask = TRUE)

# ordered static predictors (update names to match your files)
static_rast_ordered = list(
  ppt      = static_data$`prism_ppt_4km.tif`,
  solclear = static_data$`prism_solclear_4km.tif`,
  solslope = static_data$`prism_solslope_4km.tif`,
  soltotal = static_data$`prism_soltotal_4km.tif`,
  soltrans = static_data$`prism_soltrans_4km.tif`,
  tdmean   = static_data$`prism_tdmean_4km.tif`,
  tmax     = static_data$`prism_tmax_4km.tif`,
  tmean    = static_data$`prism_tmean_4km.tif`,
  tmin     = static_data$`prism_tmin_4km.tif`,
  vpdmax   = static_data$`prism_vpdmax_4km.tif`,
  vpdmin   = static_data$`prism_vpdmin_4km.tif`,
  X0_b1           = static_data$`terrain_twi_4km.tif`,
  X1_constant     = static_data$`terrain_topo_div_4km.tif`,
  X2_elevation    = static_data$`terrain_elev_4km.tif`,
  X0_ssurgo_awc   = static_data$`ssurgo_awc_4km.tif`,
  X1_ssurgo_clay  = static_data$`ssurgo_clay_4km.tif`,
  X2_ssurgo_ksat  = static_data$`ssurgo_ksat_4km.tif`,
  X3_ssurgo_sand  = static_data$`ssurgo_sand_4km.tif`,
  X0_bd_mean      = static_data$`polaris_bd_mean_4km.tif`,
  X1_clay_mean    = static_data$`polaris_clay_mean_4km.tif`,
  X10_lambda_mean = static_data$`polaris_lambda_mean_4km.tif`,
  X11_hb_mean     = static_data$`polaris_hb_mean_4km.tif`,
  X12_alpha_mean  = static_data$`polaris_alpha_mean_4km.tif`,
  X2_ksat_mean    = static_data$`polaris_ksat_mean_4km.tif`,
  X3_n_mean       = static_data$`polaris_n_mean_4km.tif`,
  X4_om_mean      = static_data$`polaris_om_mean_4km.tif`,
  X5_ph_mean      = static_data$`polaris_ph_mean_4km.tif`,
  X6_sand_mean    = static_data$`polaris_sand_mean_4km.tif`,
  X7_silt_mean    = static_data$`polaris_silt_mean_4km.tif`,
  X8_theta_r_mean = static_data$`polaris_theta_r_mean_4km.tif`,
  X9_theta_s_mean = static_data$`polaris_theta_s_mean_4km.tif`,
  latitude  = lonlat$lat,
  longitude = lonlat$lon
)

static_stack_path <- file.path("/data/ssd4/soil-moisture-ml-inference/inference-rasters/static-data-stack/static_aligned_stack.tif")
terra::writeRaster(
  rast(static_rast_ordered), static_stack_path,
  overwrite = TRUE,
  wopt = list(datatype = "FLT4S", gdal = c("COMPRESS=LZW","BIGTIFF=IF_SAFER"))
)

# min-max definitions (one load)
mm_seq       = readr::read_csv(minmax_seq,    show_col_types = FALSE)
mm_static    = readr::read_csv(minmax_static, show_col_types = FALSE)
min_max_data = bind_rows(mm_seq, mm_static)

# ------------------------
# Parallel setup
# ------------------------
ncores = 16L  # tune for your storage; 8–16 is usually safe
cl = makeCluster(ncores)
registerDoSNOW(cl)
clusterEvalQ(cl, { library(terra); library(lubridate); library(glue); terraOptions(progress = 0); TRUE })

# open GRIDMET once per worker, cache as .GRID (no external pointer returned to master)
clusterExport(cl, c("gridmet_dir"), envir = environment())
clusterEvalQ(cl, {
  library(terra); library(readr); library(purrr); library(stringr)
  read_gridmet_with_names = function(dir, time_col = "datetime") {
    ncs   = list.files(dir, pattern = "^gridmet_.*\\.nc$", full.names = TRUE)
    times = list.files(dir, pattern = "^gridmet_.*_time\\.csv$", full.names = TRUE)
    stem  = function(p) stringr::str_remove(basename(p), "(\\.nc|_time\\.csv)$")
    time_lookup = purrr::set_names(times, stem(times))
    rasters = purrr::map(ncs, function(nc_path) {
      s = stem(nc_path)
      time_csv = time_lookup[[s]]
      if (is.null(time_csv)) return(NULL)
      tdf = suppressMessages(readr::read_csv(time_csv, show_col_types = FALSE))
      nm  = as.character(tdf[["datetime"]])
      r   = terra::rast(nc_path)
      if (terra::nlyr(r) == length(nm)) names(r) = nm
      r
    })
    names(rasters) = basename(ncs)
    rasters[!vapply(rasters, is.null, logical(1))]
  }
  .GRID <<- read_gridmet_with_names(gridmet_dir)
  # local helpers (robust date matching)
  names_to_dates = function(r) as.Date(sub(" .*", "", names(r)))
  layer_at_date = function(r, d) {
    d  = as.Date(d); ds = names_to_dates(r); idx = which(ds == d)
    if (!length(idx)) stop(sprintf("Layer not found for date %s (avail %s..%s)", as.character(d), as.character(min(ds, na.rm=TRUE)), as.character(max(ds, na.rm=TRUE))))
    r[[idx]]
  }
  sum_prev_n_by_date = function(r, d, n) {
    d  = as.Date(d); ds = names_to_dates(r)
    ok = which(ds <= d & ds > d - n)
    if (!length(ok)) stop(sprintf("No layers in window [%s..%s]", as.character(d - n + 1), as.character(d)))
    terra::app(r[[ok]], 'sum',na.rm = TRUE)
  }
  TRUE
})

# Export driver-side objects used inside the loop
# DO NOT export static_rast_ordered anymore
clusterExport(
  cl,
  c("static_stack_path", "min_max_data", "circular_yday_raster",
    "out_daily_dir", "normalize_stack"),
  envir = environment()
)
# progress bar
pb = utils::txtProgressBar(min = 0, max = length(target_dates), style = 3)
progress = function(n) utils::setTxtProgressBar(pb, n)
opts = list(progress = progress)

# ------------------------
# Daily feature generation
# ------------------------
message("Starting daily feature writes to: ", out_daily_dir)
t0 = Sys.time()


 res  =  foreach(d_i = seq_along(target_dates), .options.snow = opts,
          .packages = c("terra", "lubridate", "glue")) %dopar% {
            d  = target_dates[[d_i]]
            gm = .GRID  # per-worker cache
            
            # snapshots
            bi     = layer_at_date(gm[["gridmet_bi.nc"]],     d); names(bi)    = "bi"
            fm100  = layer_at_date(gm[["gridmet_fm100.nc"]],  d); names(fm100) = "fm100"
            fm1000 = layer_at_date(gm[["gridmet_fm1000.nc"]], d); names(fm1000)= "fm1000"
            pet    = layer_at_date(gm[["gridmet_pet.nc"]],    d); names(pet)   = "pet"
            pr     = layer_at_date(gm[["gridmet_pr.nc"]],     d); names(pr)    = "pr"
            srad   = layer_at_date(gm[["gridmet_srad.nc"]],   d); names(srad)  = "srad"
            tmmn   = layer_at_date(gm[["gridmet_tmmn.nc"]],   d); names(tmmn)  = "tmmn"
            tmmx   = layer_at_date(gm[["gridmet_tmmx.nc"]],   d); names(tmmx)  = "tmmx"
            vpd    = layer_at_date(gm[["gridmet_vpd.nc"]],    d); names(vpd)   = "vpd"
            
            year = (vpd * 0) + lubridate::year(d); names(year) = "year"
            
            # rolling windows (include d)
            pr_7    = sum_prev_n_by_date(gm[["gridmet_pr.nc"]],   d, 7);    names(pr_7)   = "precip_roll_sum_short_short"
            pet_7   = sum_prev_n_by_date(gm[["gridmet_pet.nc"]],  d, 7);    names(pet_7)  = "pet_roll_sum_short_short"
            pr_15   = sum_prev_n_by_date(gm[["gridmet_pr.nc"]],   d, 15);   names(pr_15)  = "precip_roll_sum_short"
            pet_15  = sum_prev_n_by_date(gm[["gridmet_pet.nc"]],  d, 15);   names(pet_15) = "pet_roll_sum_short"
            pr_30   = sum_prev_n_by_date(gm[["gridmet_pr.nc"]],   d, 30);   names(pr_30)  = "precip_roll_sum"
            pet_30  = sum_prev_n_by_date(gm[["gridmet_pet.nc"]],  d, 30);   names(pet_30) = "pet_roll_sum"
            pr_60   = sum_prev_n_by_date(gm[["gridmet_pr.nc"]],   d, 60);   names(pr_60)  = "precip_roll_sum_mid"
            pet_60  = sum_prev_n_by_date(gm[["gridmet_pet.nc"]],  d, 60);   names(pet_60) = "pet_roll_sum_mid"
            pr_365  = sum_prev_n_by_date(gm[["gridmet_pr.nc"]],   d, 365);  names(pr_365) = "precip_roll_sum_long"
            pet_365 = sum_prev_n_by_date(gm[["gridmet_pet.nc"]],  d, 365);  names(pet_365)= "pet_roll_sum_long"
            pr_730  = sum_prev_n_by_date(gm[["gridmet_pr.nc"]],   d, 730);  names(pr_730) = "precip_roll_sum_long_long"
            pet_730 = sum_prev_n_by_date(gm[["gridmet_pet.nc"]],  d, 730);  names(pet_730)= "pet_roll_sum_long_long"
            
            def_short     = pr_15  - pet_15 ; names(def_short)     = "def_short"
            def_30        = pr_30  - pet_30 ; names(def_30)        = "def"
            def_mid       = pr_60  - pet_60 ; names(def_mid)       = "def_mid"
            def_long      = pr_365 - pet_365; names(def_long)      = "def_long"
            def_long_long = pr_730 - pet_730; names(def_long_long) = "def_long_long"
            
            circ = circular_yday_raster(vpd, d)
            
            dynamic_list = list(
              bi = bi, fm100 = fm100, fm1000 = fm1000, pet = pet, pr = pr, srad = srad,
              tmmn = tmmn, tmmx = tmmx, vpd = vpd, year = year,
              precip_roll_sum_short_short = pr_7,
              pet_roll_sum_short_short    = pet_7,
              precip_roll_sum_short       = pr_15,
              pet_roll_sum_short          = pet_15,
              precip_roll_sum             = pr_30,
              pet_roll_sum                = pet_30,
              precip_roll_sum_mid         = pr_60,
              pet_roll_sum_mid            = pet_60,
              precip_roll_sum_long        = pr_365,
              pet_roll_sum_long           = pet_365,
              precip_roll_sum_long_long   = pr_730,
              pet_roll_sum_long_long      = pet_730,
              def_short = def_short,
              def       = def_30,
              def_mid   = def_mid,
              def_long  = def_long,
              def_long_long = def_long_long
            )
            
            dynamic_stack = terra::rast(dynamic_list)
            names(dynamic_stack) = names(dynamic_list)
            
            static_stack = terra::rast(static_stack_path)
            
            full_stack = c(dynamic_stack, static_stack, circ)
            
            # normalize
            full_norm = normalize_stack(full_stack, mm_df = min_max_data)
            
            out_path = glue("{out_daily_dir}/normalized-data-{as.character(d)}.tif")
            terra::writeRaster(full_norm, out_path,
                               overwrite = TRUE,
                               wopt = list(datatype = "FLT4S",
                                           gdal = c("COMPRESS=LZW","BIGTIFF=IF_SAFER")))
            
            # clean terra temp
            tf = try(terra::tmpFiles(), silent = TRUE)
            if (!inherits(tf, "try-error") && length(tf)) suppressWarnings(file.remove(tf))
            
            NULL
          }


close(pb)
stopCluster(cl)

message("Daily feature generation finished in ",
        round(difftime(Sys.time(), t0, units = "mins"), 1), " minutes.")
# 
# # ------------------------
# # Build 180×61 inference stacks (rolling), stride=1 AND stride=60
# # ------------------------
# 
# # helper: build a single 180-day stack ending at date 'd_end'
# build_180_stack = function(d_end, dir_daily = out_daily_dir) {
#   dates = seq(d_end - 179, d_end, by = "1 day")
#   files = file.path(dir_daily, paste0("normalized-data-", as.character(dates), ".tif"))
#   if (!all(file.exists(files))) return(NULL)
#   rs = lapply(files, terra::rast)
#   terra::rast(rs)  # concatenates all bands in order
# }
# 
# # Latest (stride=1): pick last 180 by DATE-SORT, not filename order
# daily_df = tibble(path = list.files(out_daily_dir, pattern = "^normalized-data-.*\\.tif$",
#                                      full.names = TRUE)) |>
#   mutate(date = as.Date(str_extract(basename(path), "(?<=normalized-data-).*(?=\\.tif)"))) |>
#   arrange(date)
# 
# if (nrow(daily_df) >= 180) {
#   latest_180_paths = tail(daily_df$path, 180)
#   latest_180 = terra::rast(lapply(latest_180_paths, terra::rast))
#   terra::writeRaster(latest_180, file.path(out_stack_dir, "current-inference.tif"),
#                      overwrite = TRUE,
#                      wopt = list(datatype = "FLT4S",
#                                  gdal = c("COMPRESS=LZW", "BIGTIFF=IF_SAFER")))
# }
# 
# # Stride-60 “center-keep” stacks:
# # choose centers every 60 days, staying 90 days away from edges
# all_dates = daily_df$date
# if (length(all_dates) >= 180) {
#   centers = all_dates[seq(90, length(all_dates) - 90, by = 60)]
#   ends    = centers + 89
#   for (d_end in ends) {
#     stk = build_180_stack(d_end)
#     if (is.null(stk)) next
#     outf = file.path(out_stack_dir, glue("inference-{as.character(d_end)}.tif"))
#     terra::writeRaster(stk, outf, overwrite = TRUE,
#                        wopt = list(datatype = "FLT4S",
#                                    gdal = c("COMPRESS=LZW","BIGTIFF=IF_SAFER")))
#   }
# }
# 
# # ------------------------
# # Optional: clean daily temp rasters (uncomment if you want to purge)
# # ------------------------
# # unlink(out_daily_dir, recursive = TRUE, force = TRUE)