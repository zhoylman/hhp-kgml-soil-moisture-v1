##############################################################
# Title: Covariate-space coverage — simple nested min/max mask, mapped
# Description:
#   Maps the binary "unsampled" mask from R/6_16c-covariate-coverage-simple-mask.R
#   (a CONUS cell is unsampled if it falls outside the station -- or
#   pretraining -- min/max range on ANY of the 31 raw static covariates)
#   across CONUS, with station/pretraining points overlaid. Same map styling
#   as R/6_16b's main figure (geom_raster + rnaturalearth state outlines,
#   EPSG:5070) for consistency.
##############################################################

suppressPackageStartupMessages({
  library(tidyverse); library(sf); library(glue); library(patchwork); library(rnaturalearth)
})
try(chromote::set_chrome_args(c("--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--headless=new")), silent = TRUE)

repo = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
figs_dir = glue("{repo}/figs"); cache_dir = glue("{repo}/cache/covariate-coverage")
r = readRDS(file.path(cache_dir, "coverage_results.rds"))
conus_cc = r$conus_cc; station_cc = r$station_cc; pretrain_cc = r$pretrain_cc
cov_cols = setdiff(names(conus_cc), c("x","y","latitude","longitude"))

# ---- rebuild the final (all-31-feature) nested in-range mask for each reference set ----
in_range_mask = function(ref_df) {
  m = rep(TRUE, nrow(conus_cc))
  for (v in cov_cols) {
    lo = min(ref_df[[v]], na.rm = TRUE); hi = max(ref_df[[v]], na.rm = TRUE)
    m = m & (conus_cc[[v]] >= lo & conus_cc[[v]] <= hi)
  }
  m
}
station_in_range  = in_range_mask(station_cc)
pretrain_in_range = in_range_mask(pretrain_cc)
cat("Stations: % CONUS unsampled =", round(100 * mean(!station_in_range), 2), "\n")
cat("Pretraining: % CONUS unsampled =", round(100 * mean(!pretrain_in_range), 2), "\n")

`%notin%` = Negate(`%in%`)
us_sf = rnaturalearth::ne_states(country = "United States of America", returnclass = "sf") |>
  filter(name %notin% c("Alaska", "Hawaii", "Puerto Rico", "Virgin Islands", "Guam",
                        "American Samoa", "Northern Mariana Islands")) |>
  st_transform(5070)

map_theme = theme_minimal(base_size = 12) +
  theme(panel.grid = element_blank(), axis.title = element_blank(), axis.text = element_blank(),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
        plot.subtitle = element_text(hjust = 0.5, size = 9.5),
        legend.position = "bottom", legend.title.position = "top", legend.title = element_text(hjust = 0.5))

make_panel = function(in_range, pts_xy, n_pts, label, pct_unsampled) {
  map_df = conus_cc |> transmute(x, y, status = if_else(in_range, "Within station range", "Outside range (≥ 1 of 31 covariates)"))
  ggplot() +
    geom_raster(data = map_df, aes(x, y, fill = status)) +
    geom_sf(data = us_sf, fill = NA, color = "grey20", linewidth = 0.15) +
    geom_point(data = pts_xy, aes(x, y), color = "red", size = 0.35, alpha = 0.55, shape = 16) +
    coord_sf(crs = 5070, expand = FALSE) +
    scale_fill_manual(values = c("Within station range" = "#4B0092", "Outside range (≥ 1 of 31 covariates)" = "grey85"), name = NULL) +
    labs(title = label,
         subtitle = glue("{n_pts} points overlaid (red) · {pct_unsampled}% of CONUS area falls outside the range on ≥ 1 covariate")) +
    map_theme
}

p_station = make_panel(station_in_range, r$station_xy, nrow(station_cc), "(a) Station covariate-range mask",
                        round(100 * mean(!station_in_range), 2))
p_pretrain = make_panel(pretrain_in_range, r$pretrain_xy, nrow(pretrain_cc), "(b) Pretraining covariate-range mask",
                         round(100 * mean(!pretrain_in_range), 2))

combined = (p_station | p_pretrain) +
  patchwork::plot_annotation(
    title = "Simple nested min/max covariate-range coverage",
    subtitle = "Purple = within the min/max range spanned by the reference set on ALL 31 static covariates; grey = outside on ≥ 1",
    theme = theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 15),
                  plot.subtitle = element_text(hjust = 0.5, size = 10)))
ggsave(glue("{figs_dir}/covariate_coverage_simple_map.png"), combined, width = 13, height = 6.5, dpi = 300, bg = "white")
cat("\nWrote figs/covariate_coverage_simple_map.png\n")
