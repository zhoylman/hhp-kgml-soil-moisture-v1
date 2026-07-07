##############################################################
# Title: 1991-2020 VWC Climatology (year-frozen archive) + paper maps
# Description:
#   Computes the 1991-2020 climatological mean VWC (annual + seasonal)
#   from the year-frozen ensemble-median daily archive, per depth, and
#   renders EPSG:5070 maps in the house style (cf. 6_10).
#
#   Method: cache per year-month mean rasters (360 per depth; resume-safe,
#   parallel), then combine with day-count weighting so the result is the
#   EXACT mean of all dailies:  annual = sum(month_mean*n_days)/sum(n_days).
#   Seasonal = same over DJF/MAM/JJA/SON months (calendar-month seasons).
#
#   Run:  DEPTH_FLAG=middle Rscript R/6_11-vwc-climatology-maps.R
#         (shallow: rerun with DEPTH_FLAG=shallow once its archive exists)
# Author: Zachary H. Hoylman
##############################################################

suppressPackageStartupMessages({
  library(terra); library(tidyverse); library(glue); library(sf); library(parallel)
})

depth_flag = Sys.getenv("DEPTH_FLAG", "middle")
stopifnot(depth_flag %in% c("middle", "shallow"))

# ARCH_SUFFIX: "-yearfrozen" (default, corrected archive) or "" (old as-is archive,
# used only as a PLACEHOLDER while a depth's frozen regen is still running).
# As-is outputs are tagged "-asis" so they can never be mistaken for the frozen product.
arch_suffix = Sys.getenv("ARCH_SUFFIX", "-yearfrozen")
clim_tag    = if (arch_suffix == "-yearfrozen") "-yearfrozen" else "-asis"

Y0 = 1991L; Y1 = 2020L
arch_dir  = glue("/data/ssd3/soil-moisture-ml-inference/ensemble-smoothed-daily-{depth_flag}{arch_suffix}/median")
clim_root = glue("/data/ssd3/soil-moisture-ml-inference/climatology-{Y0}-{Y1}-{depth_flag}{clim_tag}")
cache_dir = file.path(clim_root, "monthly-cache")
repo      = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
figs_dir  = file.path(repo, "figs")
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

depth_label = c(middle = "Mid-depth Soil Moisture (10-50 cm)",
                shallow = "Shallow Soil Moisture (0-10 cm)")[[depth_flag]]

# ---------------------------------------------------------------------
# 1. Monthly mean cache (exact block means; resume-safe; parallel)
# ---------------------------------------------------------------------
files = tibble(path = list.files(arch_dir, pattern = "^vwc_\\d{4}-\\d{2}-\\d{2}\\.tif$", full.names = TRUE)) |>
  mutate(date = as.Date(str_extract(basename(path), "\\d{4}-\\d{2}-\\d{2}")),
         yr = year(date), mo = month(date)) |>
  filter(yr >= Y0, yr <= Y1)
stopifnot(nrow(files) > 0)
message(glue("[{depth_flag}] {nrow(files)} dailies in {Y0}-{Y1} (expect ~10957)"))

ym = files |> group_by(yr, mo) |> summarise(n = n(), .groups = "drop")
message(glue("[{depth_flag}] {nrow(ym)} year-months (expect 360)"))

monthly_path = function(yr, mo) file.path(cache_dir, sprintf("mean_%04d-%02d.tif", yr, mo))

build_month = function(i) {
  yr = ym$yr[i]; mo = ym$mo[i]; out = monthly_path(yr, mo)
  if (file.exists(out)) return(ym$n[i])
  fs = files |> filter(yr == !!yr, mo == !!mo) |> pull(path)
  tmp = paste0(out, ".tmp.tif")
  terra::app(terra::rast(fs), mean, na.rm = FALSE,
             filename = tmp, overwrite = TRUE,
             wopt = list(gdal = c("COMPRESS=LZW"), datatype = "FLT4S"))
  file.rename(tmp, out)
  ym$n[i]
}
message("Building monthly means (cached, parallel)...")
invisible(mclapply(seq_len(nrow(ym)), build_month, mc.cores = 16, mc.preschedule = FALSE))
stopifnot(all(file.exists(monthly_path(ym$yr, ym$mo))))

# ---------------------------------------------------------------------
# 2. Day-weighted combine -> annual + seasonal climatologies
# ---------------------------------------------------------------------
weighted_clim = function(sel, out_name) {
  r = terra::rast(monthly_path(sel$yr, sel$mo))
  wmean = terra::weighted.mean(r, w = sel$n)
  out = file.path(clim_root, out_name)
  terra::writeRaster(wmean, out, overwrite = TRUE,
                     gdal = c("COMPRESS=LZW"), datatype = "FLT4S")
  message("  wrote ", out)
  out
}
message("Combining to climatologies (day-weighted)...")
ann_tif = weighted_clim(ym, glue("vwc-clim-annual-{Y0}-{Y1}-{depth_flag}.tif"))
seasons = list(DJF = c(12, 1, 2), MAM = 3:5, JJA = 6:8, SON = 9:11)
sea_tif = imap_chr(seasons, \(mos, nm)
  weighted_clim(filter(ym, mo %in% mos), glue("vwc-clim-{nm}-{Y0}-{Y1}-{depth_flag}.tif")))

# ---------------------------------------------------------------------
# 3. Maps — EPSG:5070 bilinear, house style (cf. 6_10)
# ---------------------------------------------------------------------
proj_out = "EPSG:5070"
states = st_read("https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_20m.json", quiet = TRUE) |>
  st_make_valid() |>
  filter(!NAME %in% c("Virgin Islands", "Hawaii", "Alaska", "Puerto Rico")) |>
  st_transform(proj_out)

rast_df = function(tif) {
  r = terra::project(terra::rast(tif), proj_out, method = "bilinear")
  as.data.frame(r, xy = TRUE, na.rm = TRUE) |> rlang::set_names(c("x", "y", "vwc"))
}

vwc_lims = c(0.05, 0.45)   # shared scale across panels/depths for comparability
map_theme = theme_minimal(base_size = 12) +
  theme(panel.grid = element_blank(), axis.title = element_blank(), axis.text = element_blank(),
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "bottom", legend.title.position = "top",
        legend.title = element_text(hjust = 0.5))
vwc_scale = scale_fill_viridis_c(
  name = expression("Mean VWC (m"^3*" m"^-3*")"),
  option = "viridis", direction = 1, limits = vwc_lims, oob = scales::squish,
  guide = guide_colorbar(barwidth = 14, barheight = 0.5))

# 3a. Annual climatology — combined two-panel (shallow LEFT, middle RIGHT).
# Renders whenever both depths' annual rasters exist; otherwise waits (the
# second depth's run triggers it automatically).
# Prefer each depth's year-frozen climatology; fall back to the -asis placeholder.
ann_tif_for = function(d, tag) glue("/data/ssd3/soil-moisture-ml-inference/climatology-{Y0}-{Y1}-{d}{tag}/vwc-clim-annual-{Y0}-{Y1}-{d}.tif")
pick_ann = function(d) {
  fz = ann_tif_for(d, "-yearfrozen"); as = ann_tif_for(d, "-asis")
  if (file.exists(fz)) list(tif = fz, placeholder = FALSE)
  else if (file.exists(as)) list(tif = as, placeholder = TRUE)
  else NULL
}
panel_labels = c(shallow = "Shallow (0–10 cm)", middle = "Mid-depth (10–50 cm)")
picks = list(shallow = pick_ann("shallow"), middle = pick_ann("middle"))
if (!any(map_lgl(picks, is.null))) {
  ann_df = imap(picks, \(p, d) rast_df(p$tif) |> mutate(depth = panel_labels[[d]])) |>
    bind_rows() |>
    mutate(depth = factor(depth, unname(panel_labels)))   # shallow left, middle right
  ph = names(picks)[map_lgl(picks, "placeholder")]
  sub = if (length(ph)) glue("NOTE: {paste(ph, collapse = ', ')} panel is a PLACEHOLDER (as-is archive; frozen regen pending)") else NULL
  p_ann = ggplot() +
    geom_raster(data = ann_df, aes(x, y, fill = vwc)) +
    geom_sf(data = states, fill = NA, color = "grey25", linewidth = 0.15) +
    coord_sf(crs = 5070, expand = FALSE) +
    facet_wrap(~depth, ncol = 2) +
    vwc_scale +
    labs(title = glue("Soil Moisture Climatology ({Y0}–{Y1})"), subtitle = sub) +
    map_theme + theme(strip.text = element_text(face = "bold", size = 13))
  ggsave(glue("{figs_dir}/vwc_climatology_annual.png"), p_ann,
         width = 14, height = 6, dpi = 300, bg = "white")
  message("  wrote combined two-panel: figs/vwc_climatology_annual.png",
          if (length(ph)) glue("  [{paste(ph, collapse=',')} = as-is placeholder]") else "")
} else {
  message("  combined annual figure: waiting for the other depth's climatology raster")
}

# 3b. Seasonal 4-panel
sea_df = imap(sea_tif, \(tif, nm) rast_df(tif) |> mutate(season = nm)) |>
  bind_rows() |>
  mutate(season = factor(season, c("DJF", "MAM", "JJA", "SON")))
p_sea = ggplot() +
  geom_raster(data = sea_df, aes(x, y, fill = vwc)) +
  geom_sf(data = states, fill = NA, color = "grey25", linewidth = 0.15) +
  coord_sf(crs = 5070, expand = FALSE) +
  facet_wrap(~season, ncol = 2) +
  vwc_scale +
  labs(title = glue("Seasonal Soil Moisture Climatology ({Y0}–{Y1})"), subtitle = depth_label) +
  map_theme + theme(strip.text = element_text(face = "bold", size = 12))
ggsave(glue("{figs_dir}/vwc_climatology_seasonal_{depth_flag}.png"), p_sea,
       width = 11, height = 8, dpi = 300, bg = "white")

message(glue("DONE: rasters in {clim_root}; maps in figs/vwc_climatology_*_{depth_flag}.png"))
