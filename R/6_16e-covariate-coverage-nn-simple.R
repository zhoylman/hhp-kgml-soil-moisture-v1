##############################################################
# Title: Covariate-space coverage — nearest-neighbor distance (no PCA)
# Description:
#   Simpler, joint-aware alternative to both the PCA-based NN-distance
#   metric (R/6_16) and the marginal min/max box mask (R/6_16c/d). Fixes the
#   box mask's core flaw: a cell can pass a per-feature min/max test while
#   representing a combination no ACTUAL station is close to (e.g. AWC=50 is
#   "in range" if one station has AWC=5 and another has AWC=95, even if no
#   station is anywhere near 50). Here, distance is computed directly to the
#   nearest REAL station (its actual 31-covariate value vector), in
#   standardized (z-scored on the CONUS distribution) raw covariate space --
#   no PCA, no effective-dimensionality argument needed.
#   Reuses the cached raw covariate matrices from R/6_16.
##############################################################

suppressPackageStartupMessages({
  library(tidyverse); library(sf); library(glue); library(patchwork)
  library(rnaturalearth); library(FNN); library(gt)
})
try(chromote::set_chrome_args(c("--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--headless=new")), silent = TRUE)

repo = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
figs_dir = glue("{repo}/figs"); tabs_dir = glue("{repo}/tables"); cache_dir = glue("{repo}/cache/covariate-coverage")
r = readRDS(file.path(cache_dir, "coverage_results.rds"))
conus_cc = r$conus_cc; station_cc = r$station_cc; pretrain_cc = r$pretrain_cc
cov_cols = setdiff(names(conus_cc), c("x","y","latitude","longitude"))
cat("n covariates:", length(cov_cols), " | CONUS cells:", nrow(conus_cc), " | stations:", nrow(station_cc), " | pretrain:", nrow(pretrain_cc), "\n")

# ---- z-score standardize using the CONUS distribution (not station/pretrain) ----
mu = sapply(conus_cc[cov_cols], mean, na.rm = TRUE)
sdv = sapply(conus_cc[cov_cols], sd, na.rm = TRUE)
z = function(df) scale(as.matrix(df[cov_cols]), center = mu, scale = sdv)
Z_conus = z(conus_cc); Z_station = z(station_cc); Z_pretrain = z(pretrain_cc)

# ---- nearest-neighbor distance: CONUS cell -> nearest station / pretrain point ----
cat("\nComputing CONUS -> nearest-station distances (FNN, 31-D standardized space)...\n")
nn_station  = FNN::get.knnx(data = Z_station,  query = Z_conus, k = 1)
cat("Computing CONUS -> nearest-pretrain distances...\n")
nn_pretrain = FNN::get.knnx(data = Z_pretrain, query = Z_conus, k = 1)
dist_station  = as.numeric(nn_station$nn.dist)
dist_pretrain = as.numeric(nn_pretrain$nn.dist)

# ---- threshold: station-to-station self-NN distance (leave-one-out), 95th pct ----
self_nn_station = FNN::get.knn(data = Z_station, k = 1)$nn.dist[, 1]
thr_by_pct = quantile(self_nn_station, c(0.50, 0.75, 0.90, 0.95))
cat("\nStation self-NN distance percentiles (defines 'far' at each threshold):\n")
print(round(thr_by_pct, 3))

curve = map_dfr(names(thr_by_pct), function(pl) {
  thr = thr_by_pct[[pl]]
  tibble(threshold_pctile = pl, threshold_value = thr,
         pct_conus_unsampled_stations = round(100 * mean(dist_station > thr), 2),
         pct_conus_unsampled_pretrain = round(100 * mean(dist_pretrain > thr), 2))
})
write_csv(curve, glue("{tabs_dir}/covariate_coverage_nn_simple_thresholds.csv"))
cat("\n=== % CONUS area unsampled at each station self-NN distance threshold ===\n")
print(as.data.frame(curve))

headline95 = curve |> filter(threshold_pctile == "95%")
cat(glue("\nHEADLINE (95th-pct threshold): stations leave {headline95$pct_conus_unsampled_stations}% of CONUS area unsampled; ",
         "pretraining leaves {headline95$pct_conus_unsampled_pretrain}%.\n"))

# ---- figure 1: threshold-sensitivity curve ----
curve_long = curve |> pivot_longer(starts_with("pct_conus"), names_to = "ref", values_to = "pct") |>
  mutate(ref = if_else(ref == "pct_conus_unsampled_stations", "Stations", "Pretraining"),
         threshold_pctile = factor(threshold_pctile, levels = c("50%","75%","90%","95%")))
p_curve = ggplot(curve_long, aes(x = threshold_pctile, y = pct, color = ref, group = ref)) +
  geom_line(linewidth = 1.1) + geom_point(size = 2.5) +
  scale_color_manual(values = c("Stations" = "#D55E00", "Pretraining" = "#0072B2")) +
  labs(title = "Sensitivity of the 'unsampled' area fraction\nto the distance threshold",
       subtitle = "Threshold = Nth percentile of station-to-station nearest-neighbor distance",
       x = "Threshold percentile (of station self-NN distance)", y = "% of CONUS area unsampled", color = NULL) +
  theme_bw(base_size = 13) + theme(legend.position = "bottom")

# ---- figure 2: map, distance to nearest station (95th-pct threshold marked) + points ----
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

make_map = function(dist, pts_xy, n_pts, label, pct_unsampled) {
  map_df = conus_cc |> transmute(x, y, d = pmin(dist, quantile(dist, 0.99)))
  ggplot() +
    geom_raster(data = map_df, aes(x, y, fill = d)) +
    geom_sf(data = us_sf, fill = NA, color = "grey20", linewidth = 0.15) +
    geom_point(data = pts_xy, aes(x, y), color = "red", size = 0.35, alpha = 0.55, shape = 16) +
    coord_sf(crs = 5070, expand = FALSE) +
    scale_fill_viridis_c(name = "Standardized Euclidean\ndistance to nearest point", option = "magma",
                          guide = guide_colorbar(barwidth = 12, barheight = 0.5)) +
    labs(title = label,
         subtitle = glue("{n_pts} points overlaid (red) · {pct_unsampled}% of CONUS area exceeds the 95th-pct. station self-NN distance")) +
    map_theme
}
p_map_station  = make_map(dist_station, r$station_xy, nrow(station_cc), "(a) Distance to nearest station",
                           headline95$pct_conus_unsampled_stations)
p_map_pretrain = make_map(dist_pretrain, r$pretrain_xy, nrow(pretrain_cc), "(b) Distance to nearest pretraining point",
                           headline95$pct_conus_unsampled_pretrain)

combined_map = (p_map_station | p_map_pretrain) +
  patchwork::plot_annotation(title = "Covariate-space nearest-neighbor distance (31 raw standardized covariates, no PCA)",
                              theme = theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 15)))
ggsave(glue("{figs_dir}/covariate_coverage_nn_simple_map.png"), combined_map, width = 13, height = 6.5, dpi = 300, bg = "white")
ggsave(glue("{figs_dir}/covariate_coverage_nn_simple_curve.png"), p_curve, width = 7, height = 5, dpi = 300, bg = "white")

# ---- summary table ----
gt_tab = curve |> gt() |>
  tab_header(title = md("**Nearest-neighbor covariate-space coverage (no PCA)**"),
             subtitle = "Distance in z-scored (CONUS-standardized) 31-covariate space") |>
  cols_label(threshold_pctile = "Threshold (pctile of station self-NN dist.)", threshold_value = "Threshold value",
             pct_conus_unsampled_stations = "% CONUS unsampled (stations)", pct_conus_unsampled_pretrain = "% CONUS unsampled (pretraining)") |>
  fmt_number(threshold_value, decimals = 3) |>
  fmt_number(c(pct_conus_unsampled_stations, pct_conus_unsampled_pretrain), decimals = 2) |>
  opt_table_outline() |> tab_options(table.font.size = px(13), data_row.padding = px(5), column_labels.font.weight = "bold")
gtsave(gt_tab, glue("{figs_dir}/covariate_coverage_nn_simple_table.png"), expand = 30, zoom = 2.5, vwidth = 1000)

cat("\nWrote figs/covariate_coverage_nn_simple_{map,curve,table}.png, tables/covariate_coverage_nn_simple_thresholds.csv\n")
