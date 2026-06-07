library(tidyverse)
library(sf)
library(terra)
library(purrr)
library(glue)
library(lubridate)
library(tools)
library(foreach)
library(parallel)
library(magrittr)

final_out = list()

# Set up parallel backend
cl = makeCluster(20)
doSNOW::registerDoSNOW(cl)
pb = utils::txtProgressBar(min = 0, max = 179, style = 3)
progress = function(n) utils::setTxtProgressBar(pb, n)
opts = list(progress = progress)
tictoc::tic()
final_out = foreach(step_back_int = 0:179, .packages = c('tidyverse', 'sf', 'terra', 'glue',
                                         'lubridate', 'tools')) %dopar% {
                                           library(tidyverse)
                                           library(sf)
                                           library(terra)
                                           library(purrr)
                                           library(glue)
                                           library(lubridate)
                                           library(tools)
                                           library(foreach)
                                           library(parallel)
                                           
                                           # Load min-max definitions for normalization
                                           min_max_seq_pretrain = readr::read_csv("/data/ssd2/soil-moisture-ml/min-max-definitions/seq-min-max-definitions-pretrain.csv")
                                           min_max_static_pretrain = readr::read_csv("/data/ssd2/soil-moisture-ml/min-max-definitions/static-min-max-definitions-pretrain.csv")
                                           min_max_data = bind_rows(min_max_seq_pretrain, min_max_static_pretrain)
                                           
                                           read_gridmet_with_names = function(dir, time_col = "datetime") {
                                             ncs   = list.files(dir, pattern = "^gridmet_.*\\.nc$", full.names = TRUE)
                                             times = list.files(dir, pattern = "^gridmet_.*_time\\.csv$", full.names = TRUE)
                                             
                                             # Match *_time.csv to .nc by stem
                                             stem = function(p) str_remove(basename(p), "(\\.nc|_time\\.csv)$")
                                             time_lookup = set_names(times, stem(times))
                                             
                                             rasters = map(ncs, function(nc_path) {
                                               s = stem(nc_path)  # gridmet_pr, gridmet_tmmx, etc.
                                               time_csv = time_lookup[[s]]
                                               
                                               if (is.null(time_csv)) {
                                                 warning("No matching *_time.csv found for ", basename(nc_path))
                                                 return(NULL)
                                               }
                                               
                                               # Read time column
                                               tdf = suppressMessages(read_csv(time_csv, show_col_types = FALSE))
                                               if (!time_col %in% names(tdf)) {
                                                 stop(paste("Column", time_col, "not found in", basename(time_csv)))
                                               }
                                               nm = as.character(tdf[[time_col]])
                                               
                                               # Load raster and assign names
                                               r = rast(nc_path)
                                               if (nlyr(r) != length(nm)) {
                                                 warning(basename(nc_path), ": layer count does not match timestamp count")
                                               } else {
                                                 names(r) = nm
                                               }
                                               
                                               r
                                             })
                                             
                                             names(rasters) = basename(ncs)
                                             rasters
                                           }
                                           
                                           
                                           sum_last_n_days = function(r, n_days, step_back = 0) {
                                             stopifnot(inherits(r, "SpatRaster"))
                                             nl = nlyr(r)
                                             
                                             start_idx = nl - step_back - (n_days - 1)
                                             end_idx   = nl - step_back
                                             
                                             if (start_idx < 1 || end_idx > nl) {
                                               stop(glue("Invalid range: start_idx={start_idx}, end_idx={end_idx}, total_layers={nl}"))
                                             }
                                             
                                             sum(r[[start_idx:end_idx]], na.rm = TRUE)
                                           }
                                           
                                           last_layer = function(r, step_back = 0) {
                                             stopifnot(inherits(r, "SpatRaster"))
                                             nl = nlyr(r)
                                             lyr_idx = nl - step_back
                                             if (lyr_idx < 1) stop("step_back is too large for the number of layers.")
                                             
                                             r[[lyr_idx]]
                                           }
                                           
                                           make_constant_raster = function(template_rast, value, keep_na_mask = TRUE) {
                                             stopifnot(inherits(template_rast, "SpatRaster"))
                                             
                                             if (keep_na_mask) {
                                               # Preserve NA structure from template
                                               r_const = (template_rast[[1]] * 0) + value
                                             } else {
                                               # Fill every cell
                                               r_const = setValues(template_rast[[1]], rep(value, ncell(template_rast)))
                                             }
                                             
                                             return(r_const)
                                           }
                                           
                                           # Align all rasters in `dir` to the grid of `template`
                                           align_rasters_to_template = function(
                                              dir,
                                              template,                      # SpatRaster already in memory
                                              pattern = "\\.(tif|tiff|img|nc)$",
                                              method  = c("near", "bilinear", "cubic"),
                                              crop_to_template = TRUE,
                                              mask_with_template = TRUE,
                                              verbose = TRUE
                                           ) {
                                             stopifnot(inherits(template, "SpatRaster"))
                                             method = match.arg(method)
                                             
                                             files = list.files(dir, pattern = pattern, full.names = TRUE, recursive = FALSE)
                                             if (!length(files)) stop("No rasters found in '", dir, "' matching: ", pattern)
                                             
                                             out = vector("list", length(files))
                                             names(out) = basename(files)
                                             
                                             for (i in seq_along(files)) {
                                               f = files[i]
                                               if (verbose) message(sprintf("[%d/%d] %s", i, length(files), basename(f)))
                                               
                                               aligned = try({
                                                 r0 = rast(f)
                                                 
                                                 # 1) Project to template CRS
                                                 r1 = project(r0, template, method = method)
                                                 
                                                 # 2) Crop to template extent (after projection)
                                                 if (crop_to_template) {
                                                   if (!relate(ext(r1), ext(template), "intersects")) {
                                                     if (verbose) message("  -> No overlap with template after projection; skipping.")
                                                     next
                                                   }
                                                   r1 = crop(r1, template)
                                                 }
                                                 
                                                 # 3) Resample to match template grid
                                                 r2 = resample(r1, template, method = method)
                                                 
                                                 # 4) Mask to template if requested
                                                 if (mask_with_template) r2 = mask(r2, template)
                                                 
                                                 r2
                                               }, silent = TRUE)
                                               
                                               if (inherits(aligned, "try-error") || is.null(aligned)) {
                                                 if (verbose) message("  -> Failed to align: ", basename(f))
                                                 next
                                               }
                                               
                                               out[[i]] = aligned
                                             }
                                             
                                             out = out[!vapply(out, is.null, logical(1))]
                                             return(out)
                                             if (!length(out)) warning("No aligned rasters produced.")
                                           }
                                           
                                           # Example usage
                                           gridmet_rasters = read_gridmet_with_names("/data/ssd2/gridmet-operational")
                                           
                                           template = last_layer(gridmet_rasters$gridmet_bi.nc)
                                           
                                           static_data = align_rasters_to_template(
                                             dir = "/data/ssd2/soil-moisture-ml/static-data-rasters",
                                             template = template,
                                             pattern = "\\.(tif|nc)$",
                                             method = "near",
                                             crop_to_template = TRUE,
                                             mask_with_template = TRUE
                                           )
                                           
                                           
                                           make_lonlat_rasters = function(template,
                                                                          mask = TRUE,
                                                                          mask_layer = 1,
                                                                          return = c("stack", "list")) {
                                             stopifnot(inherits(template, "SpatRaster"))
                                             return = match.arg(return)
                                             
                                             # use a single layer as the grid/mask source
                                             base = template[[mask_layer]]
                                             
                                             # generate lon/lat rasters aligned to 'base'
                                             lon = init(base, fun = "x"); names(lon) = "lon"
                                             lat = init(base, fun = "y"); names(lat) = "lat"
                                             
                                             # crop is implicitly identical because we used the template grid;
                                             # mask (set lon/lat to NA where template has NA) if requested
                                             if (mask) {
                                               lon = mask(lon, base)
                                               lat = mask(lat, base)
                                             }
                                             
                                             if (return == "stack") {
                                               c(lon, lat)           # 2-layer SpatRaster (lon, lat)
                                             } else {
                                               list(lon = lon, lat = lat)
                                             }
                                           }
                                           
                                           convert_julian_to_circular = function(yday) {
                                             angle = 2 * pi * yday / 365
                                             sin_day_norm = (sin(angle) + 1) / 2  # Normalize to [0,1]
                                             return(sin_day_norm)
                                           }
                                           
                                           
                                           normalize_raster_stack_pre_established = function(
                                              r, 
                                              min_max_data, 
                                              skip = c("circular_yday"), 
                                              clamp01 = FALSE, 
                                              quiet = FALSE
                                           ) {
                                             stopifnot(inherits(r, "SpatRaster"))
                                             mm = as.data.frame(min_max_data)
                                             
                                             norm_layers = lapply(names(r), function(nm) {
                                               lyr = r[[nm]]
                                               
                                               # skip list (e.g., circular_yday)
                                               if (nm %in% skip) {
                                                 if (!quiet) message("[skip] ", nm)
                                                 return(lyr)
                                               }
                                               
                                               # look up min/max
                                               mn = mm$value[mm$name == nm & mm$min_max_id == "min"]
                                               mx = mm$value[mm$name == nm & mm$min_max_id == "max"]
                                               
                                               # pass through if missing/invalid definition
                                               if (length(mn) == 0 || length(mx) == 0 || any(is.na(c(mn, mx))) || isTRUE(all.equal(mn, mx))) {
                                                 if (!quiet) message("[pass] no valid min/max for ", nm, " (leaving unchanged)")
                                                 return(lyr)
                                               }
                                               
                                               out = (lyr - mn) / (mx - mn)
                                               
                                               if (clamp01) {
                                                 # clamp to [0,1] without changing NA mask
                                                 out = terra::ifel(out < 0, 0, out)
                                                 out = terra::ifel(out > 1, 1, out)
                                               }
                                               
                                               out
                                             })
                                             
                                             out = terra::rast(norm_layers)
                                             names(out) = names(r)
                                             out
                                           }
                                           
                                           static_rast_ordered = list(
                                             ppt              = static_data$prism_ppt_4km.tif,
                                             solclear         = static_data$prism_solclear_4km.tif,
                                             solslope         = static_data$prism_solslope_4km.tif,
                                             soltotal         = static_data$prism_soltotal_4km.tif,
                                             soltrans         = static_data$prism_soltrans_4km.tif,
                                             tdmean           = static_data$prism_tdmean_4km.tif,   # corrected from tmean
                                             tmax             = static_data$prism_tmax_4km.tif,
                                             tmean            = static_data$prism_tmean_4km.tif,
                                             tmin             = static_data$prism_tmin_4km.tif,
                                             vpdmax           = static_data$prism_vpdmax_4km.tif,
                                             vpdmin           = static_data$prism_vpdmin_4km.tif,
                                             X0_b1            = static_data$terrain_twi_4km.tif,
                                             X1_constant      = static_data$terrain_topo_div_4km.tif,
                                             X2_elevation     = static_data$terrain_elev_4km.tif,
                                             X0_ssurgo_awc    = static_data$ssurgo_awc_4km.tif,
                                             X1_ssurgo_clay   = static_data$ssurgo_clay_4km.tif,
                                             X2_ssurgo_ksat   = static_data$ssurgo_ksat_4km.tif,
                                             X3_ssurgo_sand   = static_data$ssurgo_sand_4km.tif,
                                             X0_bd_mean       = static_data$polaris_bd_mean_4km.tif,
                                             X1_clay_mean     = static_data$polaris_clay_mean_4km.tif,
                                             X10_lambda_mean  = static_data$polaris_lambda_mean_4km.tif,
                                             X11_hb_mean      = static_data$polaris_hb_mean_4km.tif,
                                             X12_alpha_mean   = static_data$polaris_alpha_mean_4km.tif,
                                             X2_ksat_mean     = static_data$polaris_ksat_mean_4km.tif,
                                             X3_n_mean        = static_data$polaris_n_mean_4km.tif,
                                             X4_om_mean       = static_data$polaris_om_mean_4km.tif,
                                             X5_ph_mean       = static_data$polaris_ph_mean_4km.tif,
                                             X6_sand_mean     = static_data$polaris_sand_mean_4km.tif,
                                             X7_silt_mean     = static_data$polaris_silt_mean_4km.tif,
                                             X8_theta_r_mean  = static_data$polaris_theta_r_mean_4km.tif,
                                             X9_theta_s_mean  = static_data$polaris_theta_s_mean_4km.tif,
                                             latitude         = make_lonlat_rasters(template, return = "list")$lat,
                                             longitude        = make_lonlat_rasters(template, return = "list")$lon
                                           )                                     
                                           # ---- Dynamic rasters ----
                                           # --- single-layer snapshots (step_back_int already defined) ---
                                           bi    = last_layer(gridmet_rasters$gridmet_bi.nc,     step_back = step_back_int)
                                           fm100 = last_layer(gridmet_rasters$gridmet_fm100.nc,  step_back = step_back_int)
                                           fm1000= last_layer(gridmet_rasters$gridmet_fm1000.nc, step_back = step_back_int)
                                           pet   = last_layer(gridmet_rasters$gridmet_pet.nc,    step_back = step_back_int)
                                           pr    = last_layer(gridmet_rasters$gridmet_pr.nc,     step_back = step_back_int)
                                           srad  = last_layer(gridmet_rasters$gridmet_srad.nc,   step_back = step_back_int)
                                           tmmn  = last_layer(gridmet_rasters$gridmet_tmmn.nc,   step_back = step_back_int)
                                           tmmx  = last_layer(gridmet_rasters$gridmet_tmmx.nc,   step_back = step_back_int)
                                           vpd   = last_layer(gridmet_rasters$gridmet_vpd.nc,    step_back = step_back_int)
                                           
                                           # year from the vpd layer's timestamp
                                           year  = make_constant_raster(template, lubridate::year(as.Date(names(vpd))), keep_na_mask = TRUE)
                                           
                                           # --- rolling sums (precip & pet), then deficits ---
                                           pr_7   = sum_last_n_days(gridmet_rasters$gridmet_pr.nc,   7,   step_back = step_back_int)
                                           pet_7  = sum_last_n_days(gridmet_rasters$gridmet_pet.nc,  7,   step_back = step_back_int)
                                           
                                           pr_15  = sum_last_n_days(gridmet_rasters$gridmet_pr.nc,   15,  step_back = step_back_int)
                                           pet_15 = sum_last_n_days(gridmet_rasters$gridmet_pet.nc,  15,  step_back = step_back_int)
                                           
                                           pr_30  = sum_last_n_days(gridmet_rasters$gridmet_pr.nc,   30,  step_back = step_back_int)
                                           pet_30 = sum_last_n_days(gridmet_rasters$gridmet_pet.nc,  30,  step_back = step_back_int)
                                           
                                           pr_60  = sum_last_n_days(gridmet_rasters$gridmet_pr.nc,   60,  step_back = step_back_int)
                                           pet_60 = sum_last_n_days(gridmet_rasters$gridmet_pet.nc,  60,  step_back = step_back_int)
                                           
                                           pr_365 = sum_last_n_days(gridmet_rasters$gridmet_pr.nc,   365, step_back = step_back_int)
                                           pet_365= sum_last_n_days(gridmet_rasters$gridmet_pet.nc,  365, step_back = step_back_int)
                                           
                                           pr_730 = sum_last_n_days(gridmet_rasters$gridmet_pr.nc,   730, step_back = step_back_int)
                                           pet_730= sum_last_n_days(gridmet_rasters$gridmet_pet.nc,  730, step_back = step_back_int)
                                           
                                           def_short      = pr_15  - pet_15
                                           def            = pr_30  - pet_30
                                           def_mid        = pr_60  - pet_60
                                           def_long       = pr_365 - pet_365
                                           def_long_long  = pr_730 - pet_730
                                           
                                           circular_yday = make_constant_raster(template, convert_julian_to_circular(names(vpd) |> as.Date() |> yday()), keep_na_mask = T)
                                           names(circular_yday) = 'circular_yday'
                                           # --- now assemble in the exact order you wanted ---
                                           dynamic_rast_ordered = list(
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
                                             def       = def,
                                             def_mid   = def_mid,
                                             def_long  = def_long,
                                             def_long_long = def_long_long
                                           )
                                           
                                           # If you want a single SpatRaster:
                                           dynamic_stack = terra::rast(dynamic_rast_ordered)
                                           names(dynamic_stack) = names(dynamic_rast_ordered)
                                           
                                           # ---- Combine into one SpatRaster ----
                                           full_raster_stack = c(rast(dynamic_rast_ordered),rast(static_rast_ordered), circular_yday)
                                           full_raster_stack_normalized = normalize_raster_stack_pre_established(full_raster_stack, min_max_data)
                                           writeRaster(full_raster_stack_normalized, glue::glue('/data/ssd2/soil-moisture-ml/inference-rasters/temp/normalized-data-{names(vpd) |> as.Date()}.tif'))
                                           
                                           on.exit({
                                             # Drop big objects created in this iteration
                                             suppressWarnings(rm(
                                               bi, fm100, fm1000, pet, pr, srad, tmmn, tmmx, vpd,
                                               year, pr_7, pet_7, pr_15, pet_15, pr_30, pet_30,
                                               pr_60, pet_60, pr_365, pet_365, pr_730, pet_730,
                                               def_short, def, def_mid, def_long, def_long_long,
                                               circular_yday, dynamic_rast_ordered, dynamic_stack,
                                               full_raster_stack, full_raster_stack_normalized
                                             ))
                                             # Clear terra temp files produced by this worker
                                             tf <- try(terra::tmpFiles(), silent = TRUE)
                                             if (!inherits(tf, "try-error") && length(tf)) {
                                               suppressWarnings(file.remove(tf))
                                             }
                                             # Final GC
                                             invisible(gc())
                                           }, add = TRUE)
                                         }

stopCluster(cl)
tictoc::toc()

files = list.files('/data/ssd2/soil-moisture-ml/inference-rasters/temp', full.names = T) |>
  as_tibble() |>
  mutate(time = str_extract(value, "(?<=-data-).*?(?=\\.tif)") |> as.Date()) |>
  slice_tail(n = 180) %$%
  value |>
  map(rast) |>
  rast()

writeRaster(files, '/data/ssd2/soil-moisture-ml/inference-rasters/inference-stack/current-inference.tif')

#erase temp data?
            