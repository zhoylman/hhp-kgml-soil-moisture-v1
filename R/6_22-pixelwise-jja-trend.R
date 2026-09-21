##############################################################
# Title: Pixel-wise growing-season (JJA) soil moisture trend, 1981-2025
# Description:
#   For each depth and each of the 45 full JJA years (1981-2025), builds a
#   per-pixel JJA-mean raster (cell-wise average of ~92 daily rasters, cached
#   -- resume-safe). Stacks the 45 annual rasters and fits an INDEPENDENT
#   OLS regression (VWC ~ year) at every pixel via closed-form analytic
#   formulas (much faster than calling lm() per-pixel at ~474k valid cells),
#   extracting slope (per decade) and a two-sided p-value. Non-significant
#   pixels (p >= 0.05) are masked out of the trend map -- this is a true
#   per-pixel test, NOT the single CONUS-average trend from R/6_21.
##############################################################

suppressPackageStartupMessages({
  library(terra); library(tidyverse); library(glue); library(sf); library(gt)
})
try(chromote::set_chrome_args(c("--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--headless=new")), silent = TRUE)

repo = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
figs_dir = glue("{repo}/figs"); tabs_dir = glue("{repo}/tables")
cache_dir = glue("{repo}/cache/pixelwise-jja-trend"); dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)
annual_dir = glue("{cache_dir}/annual-jja-means"); dir.create(annual_dir, recursive = TRUE, showWarnings = FALSE)

dirs = c(shallow = "/data/ssd3/soil-moisture-ml-inference/ensemble-smoothed-daily-shallow-yearfrozen/median",
         middle  = "/data/ssd3/soil-moisture-ml-inference/ensemble-smoothed-daily-middle-yearfrozen/median")
depth_lab = c(shallow = "Shallow (0-10 cm)", middle = "Mid-depth (10-50 cm)")
ALPHA = 0.05   # significance threshold for masking

# ---- 1. per-year, per-depth JJA-mean RASTER (cell-wise mean of ~92 daily files, cached) ----
build_annual_jja_raster = function(dep, yr, dir_in) {
  out_tif = glue("{annual_dir}/{dep}_{yr}_jja_mean.tif")
  if (file.exists(out_tif)) return(out_tif)
  files = list.files(dir_in, pattern = glue("^vwc_{yr}-0[678]-\\d{{2}}\\.tif$"), full.names = TRUE)
  if (length(files) < 90) { message(glue("[{dep} {yr}] only {length(files)} JJA files, skipping")); return(NA_character_) }
  r = mean(rast(files), na.rm = TRUE)
  writeRaster(r, out_tif, overwrite = TRUE)
  out_tif
}

years = 1981:2025
annual_paths = map(names(dirs), function(dep) {
  message(glue("[{dep}] building/checking {length(years)} annual JJA-mean rasters..."))
  paths = map_chr(years, ~ build_annual_jja_raster(dep, .x, dirs[[dep]]))
  names(paths) = years
  paths[!is.na(paths)]
}) |> set_names(names(dirs))   # map() over an unnamed vector drops names -- must reattach explicitly

# ---- 2. per-pixel OLS regression (VWC ~ year), analytic closed-form (fast) ----
# Returns c(slope_per_decade, p_value, n_years_used) per pixel. Pixels with
# any NA across the stack, or fewer than 10 valid years, return NA (masked).
make_trend_fun = function(x_years) {
  xbar = mean(x_years); Sxx = sum((x_years - xbar)^2); n = length(x_years)
  function(v) {
    ok = is.finite(v)
    if (sum(ok) < 10) return(c(NA_real_, NA_real_))
    y = v[ok]; x = x_years[ok]
    if (length(unique(x)) < 2) return(c(NA_real_, NA_real_))
    xb = mean(x); yb = mean(y)
    Sxx_i = sum((x - xb)^2); if (Sxx_i <= 0) return(c(NA_real_, NA_real_))
    Sxy_i = sum((x - xb) * (y - yb))
    slope = Sxy_i / Sxx_i
    intercept = yb - slope * xb
    resid = y - (intercept + slope * x)
    df_resid = length(y) - 2
    if (df_resid < 1) return(c(NA_real_, NA_real_))
    sigma2 = sum(resid^2) / df_resid
    se_slope = sqrt(sigma2 / Sxx_i)
    if (!is.finite(se_slope) || se_slope == 0) return(c(slope * 10, NA_real_))
    tstat = slope / se_slope
    pval = 2 * stats::pt(-abs(tstat), df = df_resid)
    c(slope * 10, pval)   # slope per DECADE
  }
}

trend_results = list()
for (dep in names(dirs)) {
  paths = annual_paths[[dep]]
  yrs_used = as.integer(names(paths))
  message(glue("[{dep}] stacking {length(paths)} annual rasters ({min(yrs_used)}-{max(yrs_used)}) for pixel-wise regression..."))
  stk = rast(unname(paths))
  fun = make_trend_fun(yrs_used)
  cores = min(16, max(1, parallel::detectCores() - 4))
  out = app(stk, fun, cores = cores)
  names(out) = c("slope_per_decade", "p_value")
  trend_results[[dep]] = out
  writeRaster(out, glue("{cache_dir}/{dep}_trend_raw.tif"), overwrite = TRUE)
  message(glue("[{dep}] pixel-wise trend computed."))
}

# ---- 3. summary stats: % of CONUS area significant, split by sign ----
summary_stats = map_dfr(names(dirs), function(dep) {
  r = trend_results[[dep]]
  slope = r[["slope_per_decade"]]; pval = r[["p_value"]]
  v_slope = values(slope)[, 1]; v_p = values(pval)[, 1]
  ok = is.finite(v_slope) & is.finite(v_p)
  n_total = sum(ok)
  sig = ok & v_p < ALPHA
  tibble(depth = dep,
         pct_area_significant = round(100 * sum(sig) / n_total, 2),
         pct_area_sig_drying  = round(100 * sum(sig & v_slope < 0) / n_total, 2),
         pct_area_sig_wetting = round(100 * sum(sig & v_slope > 0) / n_total, 2),
         median_slope_where_sig = round(median(v_slope[sig], na.rm = TRUE), 5))
})
write_csv(summary_stats, glue("{tabs_dir}/pixelwise_jja_trend_summary.csv"))
cat("\n=== Pixel-wise JJA trend summary (alpha = 0.05) ===\n")
print(as.data.frame(summary_stats))

gt_tab = summary_stats |> gt() |>
  tab_header(title = md("**Pixel-wise growing-season (JJA) soil moisture trend, 1981-2025**"),
             subtitle = glue("% of CONUS area with a statistically significant (p < {ALPHA}) OLS trend")) |>
  cols_label(depth = "Depth", pct_area_significant = "% significant (either direction)",
             pct_area_sig_drying = "% significant drying", pct_area_sig_wetting = "% significant wetting",
             median_slope_where_sig = "Median slope where significant (VWC/decade)") |>
  opt_table_outline() |> tab_options(table.font.size = px(13), data_row.padding = px(5), column_labels.font.weight = "bold")
gtsave(gt_tab, glue("{figs_dir}/pixelwise_jja_trend_summary_table.png"), expand = 30, zoom = 2.5, vwidth = 1100)

# ---- 4. map: significant trend only (masked), EPSG:5070, diverging RdBu (blue=wetting, red=drying) ----
proj_out = "EPSG:5070"
`%notin%` = Negate(`%in%`)
us_sf = rnaturalearth::ne_states(country = "United States of America", returnclass = "sf") |>
  filter(name %notin% c("Alaska", "Hawaii", "Puerto Rico", "Virgin Islands", "Guam",
                        "American Samoa", "Northern Mariana Islands")) |>
  st_transform(proj_out)

map_theme = theme_minimal(base_size = 12) +
  theme(panel.grid = element_blank(), axis.title = element_blank(), axis.text = element_blank(),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
        plot.subtitle = element_text(hjust = 0.5, size = 10),
        strip.text = element_text(face = "bold", size = 15),
        legend.position = "bottom", legend.title.position = "top", legend.title = element_text(hjust = 0.5))

make_panel_df = function(dep) {
  r = trend_results[[dep]]
  slope_masked = r[["slope_per_decade"]]
  slope_masked[r[["p_value"]] >= ALPHA] = NA
  r5070 = terra::project(slope_masked, proj_out, method = "bilinear")
  as.data.frame(r5070, xy = TRUE, na.rm = TRUE) |> rlang::set_names(c("x", "y", "slope")) |> mutate(depth = depth_lab[dep])
}
map_df = map_dfr(names(dirs), make_panel_df) |> mutate(depth = factor(depth, levels = depth_lab))
lim = max(abs(quantile(map_df$slope, c(0.01, 0.99), na.rm = TRUE)))

p = ggplot() +
  geom_raster(data = map_df, aes(x, y, fill = slope)) +
  geom_sf(data = us_sf, fill = NA, color = "grey30", linewidth = 0.15) +
  coord_sf(crs = proj_out, expand = FALSE) +
  facet_wrap(~depth) +
  scale_fill_distiller(palette = "RdBu", direction = 1, limits = c(-lim, lim), oob = scales::squish,
                        name = expression("VWC trend (m"^3*" m"^-3*" decade"^-1*")"),
                        guide = guide_colorbar(barwidth = 16, barheight = 0.6)) +
  labs(title = "Pixel-Wise Growing-Season (JJA) Soil Moisture Trend, 1981-2025",
       subtitle = glue("Independent OLS trend per pixel; non-significant pixels (p >= {ALPHA}) masked (white)")) +
  map_theme
ggsave(glue("{figs_dir}/pixelwise_jja_trend_map.png"), p, width = 13, height = 6.5, dpi = 300, bg = "white")
cat(glue("\nWrote figs/pixelwise_jja_trend_map.png, figs/pixelwise_jja_trend_summary_table.png\n"))
cat(glue("Wrote tables/pixelwise_jja_trend_summary.csv\n"))
