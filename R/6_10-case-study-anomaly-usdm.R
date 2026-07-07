library(tidyverse)
library(terra)
library(sf)
library(glue)
library(patchwork)
library(magick)

# ============================================================================
#  Case-study figure: KGML beta-fit soil-moisture anomaly (SMI) vs. the U.S.
#  Drought Monitor, 2017 Northern Plains flash drought.
#  Rows: (a) KGML SMI 0-10 cm, (b) KGML SMI 10-50 cm, (c) official USDM.
#  Cols: three dates. KGML SMI = per-pixel BETA fit (beta_fit_smi) of the
#  ensemble-median soil moisture vs. its 30-yr same-day climatology -> qnorm,
#  binned to USDM-style classes. (Beta-fit SMI, NOT precipitation SPI.)
# ============================================================================

repo = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
figs_dir = glue("{repo}/figs")
cache_dir = glue("{repo}/cache/anomaly"); dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
usdm_dir  = glue("{repo}/cache/usdm");    dir.create(usdm_dir,  showWarnings = FALSE, recursive = TRUE)

# ---- event config (set CASE_EVENT=2017|2021; KGML map dates + matching USDM Tuesdays) ----
event = Sys.getenv("CASE_EVENT", "2017")
events = list(
  "2017" = list(kgml = c("2017-05-09","2017-06-06","2017-07-11"),   # N. Plains flash drought: onset -> peak -> 2nd peak (USDM Tuesdays, exact match)
                usdm = c("2017-05-09","2017-06-06","2017-07-11")),
  "2021" = list(kgml = c("2021-05-01","2021-07-01","2021-09-01"),   # N. Plains drought: spring onset -> summer peak -> late season
                usdm = c("2021-04-27","2021-06-29","2021-08-31")))
stopifnot(event %in% names(events))
kgml_dates = as.Date(events[[event]]$kgml)
usdm_dates = as.Date(events[[event]]$usdm)

# ---- SMI method (SMI_METHOD=original|varaug) --------------------------------
# varaug = the ops variance-augmented SMI (ANOMALY-METHOD-variance-augmented.md):
# MoM Beta on the trailing-30 same-day climatology with v_eff = var(clim) +
# sigma2_obs (current day's CROSS-FOLD variance), cap +/-3.09. Middle uses the
# YEAR-FROZEN archive (promoted product); shallow is the as-is PLACEHOLDER until
# its frozen regen lands. Cached SMI rasters are tagged by method.
smi_method = Sys.getenv("SMI_METHOD", "original")
stopifnot(smi_method %in% c("original", "varaug"))
if (smi_method == "varaug") {
  ens = c(shallow = "/data/ssd3/soil-moisture-ml-inference/ensemble-smoothed-daily-shallow/median",
          middle  = "/data/ssd3/soil-moisture-ml-inference/ensemble-smoothed-daily-middle-yearfrozen/median")
  folds_root = c(shallow = "/data/ssd3/soil-moisture-ml-inference/predictions-smoothed-daily-shallow",
                 middle  = "/data/ssd3/soil-moisture-ml-inference/predictions-smoothed-daily-middle-yearfrozen")
  out_png = if (event == "2017") "case_study_anomaly_vs_usdm_varaug.png" else glue("case_study_anomaly_vs_usdm_{event}_varaug.png")
} else {
  ens = c(shallow = "/data/ssd3/soil-moisture-ml-inference/ensemble-smoothed-daily-shallow/median",
          middle  = "/data/ssd4/soil-moisture-ml-inference/ensemble-smoothed-daily-middle/median")
  folds_root = NULL
  out_png = if (event == "2017") "case_study_anomaly_vs_usdm.png" else glue("case_study_anomaly_vs_usdm_{event}.png")
}
depth_lab = c(shallow = "Soil Moisture Anomaly (0-10 cm)", middle = "Soil Moisture Anomaly (10-50 cm)")

proj_out = "EPSG:5070"   # original maps were rendered in Albers (EPSG:5070); 4326 flattens the N. Plains
missouri_basin = sf::read_sf("~/temp/WBD_10_HU2_Shape/Shape/WBDHU2.shp") |> st_transform(proj_out) |> select(-name)

# ---- beta-fit SMI (per pixel): beta CDF of the value vs its climatology -> qnorm ----
beta_fit_smi = function(x, climatology_length = 30L, return_latest = TRUE) {
  x = as.numeric(x); x = x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  x = utils::tail(x, climatology_length)
  x = pmin(pmax(x, 1e-6), 1 - 1e-6)
  if (length(unique(x)) < 3L) return(NA_real_)
  m = mean(x); v = stats::var(x); if (!is.finite(v) || v <= 0) return(NA_real_)
  t = max(m * (1 - m) / v - 1, 2); a0 = m * t; b0 = (1 - m) * t
  fit = try(MASS::fitdistr(x, function(x, a, b) dbeta(x, a, b, log = TRUE), start = list(a = a0, b = b0)), silent = TRUE)
  Fvals = if (!inherits(fit, "try-error")) pbeta(x, fit$estimate[["a"]], fit$estimate[["b"]]) else stats::ecdf(x)(x)
  out = if (return_latest) utils::tail(Fvals, 1L) else Fvals
  stats::qnorm(pmin(pmax(out, 1e-12), 1 - 1e-12))
}

# ---- ops variance-augmented SMI (VERBATIM math from v1-ops R/3_3-finalize.R):
# x = [same-day climatology values (current LAST), sigma_obs^2]. MoM Beta on
# (mean, var + sigma_obs^2), z capped +/-3.09.
beta_fit_smi_varaug = function(x, climatology_length = 30L) {
  x = as.numeric(x); n = length(x)
  if (n < 2L) return(NA_real_)
  sigma_obs2 = x[n]
  clim = x[-n]; clim = clim[is.finite(clim)]
  if (length(clim) < 3L) return(NA_real_)
  clim = utils::tail(clim, climatology_length)
  clim = pmin(pmax(clim, 1e-6), 1 - 1e-6)
  if (length(unique(clim)) < 3L) return(NA_real_)
  cur = clim[length(clim)]
  m = mean(clim); v = stats::var(clim)
  if (!is.finite(v) || v <= 0) return(NA_real_)
  if (is.finite(sigma_obs2) && sigma_obs2 > 0) v = v + sigma_obs2
  t = max(m * (1 - m) / v - 1, 2)
  cdf = pbeta(min(max(cur, 1e-6), 1 - 1e-6), m * t, (1 - m) * t)
  z = stats::qnorm(min(max(cdf, 1e-12), 1 - 1e-12))
  pmin(pmax(z, -3.09), 3.09)
}

# ---- gridded SMI for one date/depth (30-yr same-day climatology), cached ----
smi_for_day = function(date0, dir_in, fold_root = NULL, clim_years = 30L,
                       cores = min(16L, max(1L, parallel::detectCores() - 2L))) {
  yrs = seq(lubridate::year(date0) - (clim_years - 1L), lubridate::year(date0))
  files = file.path(dir_in, sprintf("vwc_%04d-%s.tif", yrs, format(date0, "%m-%d")))
  files = files[file.exists(files)]
  r = rast(files)
  cl = parallel::makeCluster(cores); on.exit(parallel::stopCluster(cl), add = TRUE)
  parallel::clusterEvalQ(cl, library(MASS))
  if (smi_method == "varaug") {
    # append the current day's CROSS-FOLD variance as the LAST layer
    ffiles = file.path(fold_root, paste0("fold_", 1:10), sprintf("vwc_%s.tif", date0))
    ffiles = ffiles[file.exists(ffiles)]
    stopifnot(length(ffiles) >= 8)
    var_r = app(rast(ffiles), fun = function(v) stats::var(v, na.rm = TRUE))
    r = c(r, var_r)
    parallel::clusterExport(cl, "beta_fit_smi_varaug", envir = globalenv())
    app(r, beta_fit_smi_varaug, cores = cl)
  } else {
    parallel::clusterExport(cl, "beta_fit_smi", envir = globalenv())
    app(r, beta_fit_smi, cores = cl)
  }
}

# ---- binned CONUS map ----
brks = c(-Inf, -2, -1.6, -1.3, -0.8, -0.5, 0.5, 0.8, 1.3, 1.6, 2, Inf)
lbls = c("< -2 (D4)","-2 – -1.6 (D3)","-1.6 – -1.3 (D2)","-1.3 – -0.8 (D1)","-0.8 – -0.5 (D0)",
         "-0.5 – 0.5","0.5 – 0.8","0.8 – 1.3","1.3 – 1.6","1.6 – 2","> 2")
pal = setNames(c("#730000","#E60000","#FFAA00","#FCD37F","#FFFF00","#FFFFFF",
                 "#82FCF9","#32E1FA","#325CFE","#4030E3","#303B83"), lbls)
states = st_read("https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_20m.json", quiet = TRUE) |>
  st_make_valid() |> filter(!NAME %in% c("Alaska","Hawaii","Puerto Rico")) |> st_transform(proj_out)
conus = st_union(states) |> st_as_sf()

map_smi = function(r_smi, title, subtitle) {
  # match the original exactly: render in Albers (EPSG:5070), bilinear reprojection.
  r = terra::project(r_smi, proj_out, method = "bilinear")
  r = mask(crop(r, vect(conus)), vect(conus))
  df = as.data.frame(r, xy = TRUE, na.rm = TRUE); v = names(r)[1]
  df$cat = factor(cut(df[[v]], breaks = brks, labels = lbls, right = TRUE, include.lowest = TRUE), levels = lbls)
  # dummy layer: one zero-size (invisible) tile per class so EVERY class appears in the legend
  dummy = data.frame(x = mean(df$x), y = mean(df$y), cat = factor(lbls, levels = lbls))
  ggplot() +
    geom_tile(data = dummy, aes(x, y, fill = cat), width = 0, height = 0) +
    geom_raster(data = df, aes(x, y, fill = cat)) +
    geom_sf(data = states, fill = NA, color = "grey25", linewidth = 0.2) +
    geom_sf(data = missouri_basin, fill = NA, color = "black", linewidth = 0.4) +
    coord_sf(crs = 5070, expand = FALSE) +
    scale_fill_manual(values = pal, drop = FALSE, limits = lbls, name = "Standardized\nAnomaly", na.translate = FALSE) +
    labs(title = title, subtitle = subtitle) +
    theme_minimal(base_size = 11) +
    theme(panel.grid = element_blank(), axis.title = element_blank(), axis.text = element_blank(),
          plot.title = element_text(face = "bold", hjust = 0.5, margin = margin(b = 1)),
          plot.subtitle = element_text(face = "bold", hjust = 0.5, margin = margin(b = 1)),
          legend.key.height = unit(0.35, "cm"), legend.key.width = unit(0.45, "cm"),
          legend.title = element_text(size = 8), legend.text = element_text(size = 7),
          plot.margin = margin(t = 1, r = 2, b = 1, l = 2))
}

# ---- build the 6 KGML maps (cache the SMI rasters) ----
kgml_maps = list()
for (dep in c("shallow", "middle")) for (i in seq_along(kgml_dates)) {
  dt = kgml_dates[i]                                  # [i] keeps Date class ([[ would strip it)
  tag = if (smi_method == "varaug") "_varaug" else ""
  tif = glue("{cache_dir}/{dep}_{dt}{tag}.tif")
  r = if (file.exists(tif)) rast(tif) else {
    x = smi_for_day(dt, ens[[dep]], fold_root = if (is.null(folds_root)) NULL else folds_root[[dep]])
    writeRaster(x, tif, overwrite = TRUE); x }
  kgml_maps[[glue("{dep}_{dt}")]] = map_smi(r, depth_lab[[dep]], format(dt, "%m-%d-%Y"))
  message(glue("map: {dep} {dt}"))
}

# ---- download USDM official maps ----
usdm_panels = lapply(seq_along(usdm_dates), function(i) {
  dt = usdm_dates[i]
  ds = format(dt, "%Y%m%d"); png = glue("{usdm_dir}/usdm_{ds}.png")
  if (!file.exists(png)) download.file(glue("https://droughtmonitor.unl.edu/data/png/{ds}/{ds}_usdm.png"), png, quiet = TRUE)
  patchwork::wrap_elements(grid::rasterGrob(magick::image_read(png), interpolate = TRUE))
})

# ---- assemble 3 x 3 ----
sh = kgml_maps[paste0("shallow_", kgml_dates)]
mi = kgml_maps[paste0("middle_",  kgml_dates)]
combined = (sh[[1]] | sh[[2]] | sh[[3]]) /
           (mi[[1]] | mi[[2]] | mi[[3]]) /
           (usdm_panels[[1]] | usdm_panels[[2]] | usdm_panels[[3]]) +
           patchwork::plot_layout(heights = c(1, 1, 1.35))   # USDM PNGs are taller; give them more room so KGML rows aren't stretched
ggsave(glue("{figs_dir}/{out_png}"), combined, width = 18, height = 11.5, dpi = 200, bg = "white")
message(glue("Wrote figs/{out_png}"))
