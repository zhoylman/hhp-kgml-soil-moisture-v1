##############################################################
# Title: Covariate-space coverage — figures + summary table (stage 2)
# Description:
#   Consumes cache/covariate-coverage/coverage_results.rds (built by
#   R/6_16-covariate-coverage-analysis.R) and produces:
#     figs/covariate_coverage_main.png   (2-panel main figure)
#     figs/covariate_coverage_si.png     (3-panel SI figure)
#     tables/covariate_coverage_summary.csv (+ gt png)
#   See R/6_16 header for full method/data-source documentation.
# Author: Claude (agent), for Z. Hoylman
##############################################################

suppressPackageStartupMessages({
  library(tidyverse); library(sf); library(glue); library(patchwork)
  library(rnaturalearth); library(gt); library(scales)
})
try(chromote::set_chrome_args(c("--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--headless=new")), silent = TRUE)

repo = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
figs_dir = glue("{repo}/figs"); tabs_dir = glue("{repo}/tables")
cache_dir = glue("{repo}/cache/covariate-coverage")
res = readRDS(file.path(cache_dir, "coverage_results.rds"))

`%notin%` = Negate(`%in%`)
us_sf = rnaturalearth::ne_states(country = "United States of America", returnclass = "sf") |>
  filter(name %notin% c("Alaska", "Hawaii", "Puerto Rico", "Virgin Islands", "Guam",
                        "American Samoa", "Northern Mariana Islands")) |>
  st_transform(5070)

map_theme = theme_minimal(base_size = 12) +
  theme(panel.grid = element_blank(), axis.title = element_blank(), axis.text = element_blank(),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
        plot.subtitle = element_text(hjust = 0.5, size = 9),
        legend.position = "bottom", legend.title.position = "top", legend.title = element_text(hjust = 0.5))

# ============================================================================
# MAIN FIGURE — (i) distance-to-nearest-station map, (ii) PC1-PC2 coverage
# ============================================================================

# ---- (i) CONUS map: Euclidean distance (N95 PCs) to nearest station ----
map_df = res$conus_cc |> transmute(x, y, dist = res$conus_dist_euclid_station_N95)
station_thr = res$nn_results |> filter(pc_choice == "N95", reference == "station", distance == "Euclidean") |> pull(threshold_95pct_selfNN)

p_map_station = ggplot() +
  geom_raster(data = map_df, aes(x, y, fill = pmin(dist, quantile(dist, 0.99)))) +
  geom_sf(data = us_sf, fill = NA, color = "grey20", linewidth = 0.15) +
  geom_point(data = res$station_xy, aes(x, y), color = "red", size = 0.35, alpha = 0.55, shape = 16) +
  coord_sf(crs = 5070, expand = FALSE) +
  scale_fill_viridis_c(name = glue("Euclidean distance to\nnearest station (N={res$n95} PC space)"),
                        option = "magma", guide = guide_colorbar(barwidth = 12, barheight = 0.5)) +
  labs(title = "(i) Covariate-space distance to nearest station",
       subtitle = glue("Stations (red, n=734) overlaid; {round(100*mean(map_df$dist>station_thr),1)}% of CONUS area ",
                        "exceeds the 95th-pct. station self-NN distance ({round(station_thr,2)})")) +
  map_theme

# ---- (ii) PC1-PC2: CONUS density background + stations + pretraining ----
pc12_conus = as_tibble(res$PC_conus[, 1:2]) |> rlang::set_names(c("PC1", "PC2"))
pc12_station = as_tibble(res$PC_station[, 1:2]) |> rlang::set_names(c("PC1", "PC2"))
pc12_pretrain = as_tibble(res$PC_pretrain[, 1:2]) |> rlang::set_names(c("PC1", "PC2"))

set.seed(1)
conus_sub = pc12_conus[sample(nrow(pc12_conus), min(80000, nrow(pc12_conus))), ]  # subsample for hex/point rendering only; density itself uses full grid via stat_density_2d

p_pc_scatter = ggplot() +
  stat_density_2d(data = pc12_conus, aes(PC1, PC2, fill = after_stat(density)),
                   geom = "raster", contour = FALSE, n = 200) +
  scale_fill_viridis_c(option = "viridis", name = "CONUS density\n(PC1-PC2)",
                        guide = guide_colorbar(barwidth = 10, barheight = 0.5)) +
  ggnewscale::new_scale_fill() +
  geom_point(data = pc12_pretrain, aes(PC1, PC2, color = glue("Pretraining (n={format(nrow(pc12_pretrain), big.mark=',')})")), shape = 1, size = 0.5, stroke = 0.25, alpha = 0.4) +
  geom_point(data = pc12_station, aes(PC1, PC2, color = glue("Stations (n={nrow(pc12_station)})")), size = 0.7, alpha = 0.75) +
  scale_color_manual(name = NULL, values = rlang::set_names(
    c("white", "red"),
    c(glue("Pretraining (n={format(nrow(pc12_pretrain), big.mark=',')})"), glue("Stations (n={nrow(pc12_station)})")))) +
  guides(color = guide_legend(override.aes = list(size = 2.5, alpha = 1, stroke = 1))) +
  coord_cartesian(xlim = quantile(pc12_conus$PC1, c(0.001, 0.999)), ylim = quantile(pc12_conus$PC2, c(0.001, 0.999))) +
  labs(title = "(ii) PC1-PC2 coverage: stations vs. pretraining sample",
       subtitle = "Background = CONUS density; pretraining fills the space stations miss",
       x = glue("PC1 ({round(100*res$scree_df$var_explained[1],1)}% var)"),
       y = glue("PC2 ({round(100*res$scree_df$var_explained[2],1)}% var)")) +
  theme_bw(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
        plot.subtitle = element_text(hjust = 0.5, size = 9),
        legend.position = "bottom",
        legend.key = element_rect(fill = "grey40"))

main_fig = p_map_station | p_pc_scatter
ggsave(glue("{figs_dir}/covariate_coverage_main.png"), main_fig, width = 14, height = 6.5, dpi = 300, bg = "white")
message("wrote figs/covariate_coverage_main.png")

p_pc_scatter_standalone = p_pc_scatter +
  labs(title = "CONUS covariate density (PC1-PC2): stations vs. pretraining sample", subtitle = NULL)
ggsave(glue("{figs_dir}/covariate_coverage_pca_standalone.png"), p_pc_scatter_standalone, width = 8, height = 6.5, dpi = 300, bg = "white")
message("wrote figs/covariate_coverage_pca_standalone.png")

# ============================================================================
# SI FIGURE — (a) scree, (b) Mahalanobis map, (c) raw-axis binned occupancy
# ============================================================================

# ---- (a) scree plot ----
scree = res$scree_df
p_scree = ggplot(scree, aes(pc)) +
  geom_col(aes(y = var_explained), fill = "grey70", width = 0.7) +
  geom_line(aes(y = cum_var), color = "#4B0092", linewidth = 0.8) +
  geom_point(aes(y = cum_var), color = "#4B0092", size = 1.2) +
  geom_hline(yintercept = c(0.90, 0.95), linetype = "dashed", color = c("grey40","black"), linewidth = 0.4) +
  annotate("text", x = 25, y = 0.90, label = glue("90% -> {res$n90} PCs"), vjust = -0.6, size = 3.2) +
  annotate("text", x = 25, y = 0.95, label = glue("95% -> {res$n95} PCs"), vjust = -0.6, size = 3.2) +
  annotate("text", x = length(scree$pc)*0.6, y = 0.35,
           label = glue("Participation ratio\n(effective dimensionality)\n= {round(res$participation_ratio,1)} of {nrow(scree)} raw vars"),
           size = 3.0, hjust = 0) +
  scale_y_continuous(labels = percent, breaks = seq(0,1,0.25)) +
  labs(title = "(a) CONUS PCA scree", x = "Principal component", y = "Variance explained (bars) / cumulative (line)") +
  theme_bw(base_size = 11) + theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# ---- (b) Mahalanobis map ----
map_df_m = res$conus_cc |> transmute(x, y, dist = res$conus_dist_maha_station_N95)
station_thr_m = res$nn_results |> filter(pc_choice == "N95", reference == "station", distance == "Mahalanobis") |> pull(threshold_95pct_selfNN)
p_map_maha = ggplot() +
  geom_raster(data = map_df_m, aes(x, y, fill = pmin(dist, quantile(dist, 0.99)))) +
  geom_sf(data = us_sf, fill = NA, color = "grey20", linewidth = 0.15) +
  geom_point(data = res$station_xy, aes(x, y), color = "red", size = 0.3, alpha = 0.5, shape = 16) +
  coord_sf(crs = 5070, expand = FALSE) +
  scale_fill_viridis_c(name = glue("Mahalanobis distance to\nnearest station (N={res$n95} PCs)"),
                        option = "magma", guide = guide_colorbar(barwidth = 12, barheight = 0.5)) +
  labs(title = "(b) Mahalanobis-distance version",
       subtitle = glue("{round(100*mean(map_df_m$dist>station_thr_m),1)}% of area exceeds threshold ",
                        "(cf. {round(100*mean(map_df$dist>station_thr),1)}% Euclidean) -- conclusion unchanged")) +
  map_theme

# ---- (c) binned-occupancy heatmaps on interpretable raw axes ----
raw_pairs = list(c("ppt", "tmean"), c("X0_ssurgo_awc", "vpdmax"))
raw_label = c(ppt = "PRISM ppt (mm)", tmean = "PRISM tmean (C)", vpdmax = "PRISM max VPD (hPa)",
              X0_ssurgo_awc = "SSURGO AWC (cm/cm)", X3_ssurgo_sand = "SSURGO sand (%)")

make_occ_panel = function(vx, vy, ref_name, ref_df) {
  bx = quantile(res$conus_cc[[vx]], seq(0, 1, length.out = 11), na.rm = TRUE) |> unique()
  by = quantile(res$conus_cc[[vy]], seq(0, 1, length.out = 11), na.rm = TRUE) |> unique()
  cdf = res$conus_cc |> transmute(bx = cut(.data[[vx]], bx, include.lowest = TRUE), by = cut(.data[[vy]], by, include.lowest = TRUE)) |>
    drop_na() |> count(bx, by, name = "n_conus")
  rdf = ref_df |> transmute(bx = cut(.data[[vx]], bx, include.lowest = TRUE), by = cut(.data[[vy]], by, include.lowest = TRUE)) |>
    drop_na() |> distinct(bx, by) |> mutate(occupied_by_ref = TRUE)
  plot_df = cdf |> left_join(rdf, by = c("bx","by")) |> mutate(occupied_by_ref = replace_na(occupied_by_ref, FALSE),
                                                                status = if_else(occupied_by_ref, "CONUS + reference", "CONUS only (gap)"))
  ggplot(plot_df, aes(bx, by, fill = status)) +
    geom_tile(color = "white", linewidth = 0.2) +
    scale_fill_manual(values = c("CONUS + reference" = "#4B0092", "CONUS only (gap)" = "grey85"), name = NULL) +
    labs(title = glue("{ref_name}"), x = raw_label[[vx]], y = raw_label[[vy]]) +
    theme_bw(base_size = 9) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6), axis.text.y = element_text(size = 6),
          plot.title = element_text(hjust = 0.5, size = 9), legend.position = "bottom", legend.text = element_text(size = 7))
}

occ_panels = list(
  make_occ_panel("ppt", "tmean", "Stations (ppt x tmean)", res$station_cc),
  make_occ_panel("ppt", "tmean", "Pretraining (ppt x tmean)", res$pretrain_cc),
  make_occ_panel("X0_ssurgo_awc", "vpdmax", "Stations (AWC x max VPD)", res$station_cc),
  make_occ_panel("X0_ssurgo_awc", "vpdmax", "Pretraining (AWC x max VPD)", res$pretrain_cc)
)
p_occ_grid = wrap_plots(occ_panels, ncol = 2) + plot_annotation(title = "(c) Decile-binned occupancy on interpretable raw axes") &
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 12))

si_fig = (p_scree | p_map_maha) / p_occ_grid + plot_layout(heights = c(1, 1.3))
ggsave(glue("{figs_dir}/covariate_coverage_si.png"), si_fig, width = 13, height = 13, dpi = 300, bg = "white")
message("wrote figs/covariate_coverage_si.png")

# ============================================================================
# TABLE — one row per metric x choice, station vs. pretraining coverage
# ============================================================================

nn_tab = res$nn_results |>
  transmute(metric = "Nearest-neighbor distance (frac. area unsampled)",
            choice = glue("{distance}, N_PC={n_pc} ({pc_choice})"),
            reference, value = area_frac_unsampled) |>
  pivot_wider(names_from = reference, values_from = value)

box_tab = res$box_results |>
  transmute(metric = "Range-box containment (frac. area outside station/pretrain box)",
            choice = glue("N_PC={n_pc} ({pc_choice})"), reference, value = area_frac_outside_box) |>
  pivot_wider(names_from = reference, values_from = value)

occ_tab = res$occ_pc_results |>
  transmute(metric = "Binned occupancy, PC space (frac. occupied-CONUS-area in zero-reference bins)",
            choice = glue("N_PC={n_pc}, bins/axis={n_bins_per_axis} ({role})"), reference,
            value = area_frac_occupied_bins_with_zero_ref) |>
  pivot_wider(names_from = reference, values_from = value)

raw_tab = res$raw_occ_summary |>
  transmute(metric = "Binned occupancy, raw axis (deciles; frac. area in zero-reference bins)",
            choice = variable, reference, value = area_frac_occupied_bins_with_zero_ref) |>
  pivot_wider(names_from = reference, values_from = value)

dens_tab = res$dens_results_df |>
  transmute(metric = "Density ratio, PC1-PC2 (frac. CONUS mass where reference density ~ 0)",
            choice = "PC1-PC2, threshold=1% of ref max", reference, value = area_frac_zero_density) |>
  pivot_wider(names_from = reference, values_from = value)

station_thresh95 = res$nn_results |> filter(pc_choice=="N95", reference=="station", distance=="Euclidean") |> pull(threshold_95pct_selfNN)
pretrain_thresh95 = res$nn_results |> filter(pc_choice=="N95", reference=="pretrain", distance=="Euclidean") |> pull(threshold_95pct_selfNN)
station_unsampled = res$conus_dist_euclid_station_N95 > station_thresh95
pretrain_unsampled = res$conus_dist_euclid_pretrain_N95 > pretrain_thresh95
overlap_frac_closed = mean(!pretrain_unsampled[station_unsampled])
overlap_tab = tibble(metric = "Punchline: of the area unsampled by stations (Euclidean NN, N95), frac. that pretraining DOES cover",
                      choice = glue("Euclidean, N_PC={res$n95} (N95)"), station = NA_real_, pretrain = overlap_frac_closed)

summary_tab = bind_rows(nn_tab, box_tab, occ_tab, raw_tab, dens_tab, overlap_tab) |>
  rename(station_coverage_gap = station, pretraining_coverage_gap = pretrain)
write_csv(summary_tab, glue("{tabs_dir}/covariate_coverage_summary.csv"))
message("wrote tables/covariate_coverage_summary.csv")

# ---- gt render (house style, cf. 6_6) ----
g = summary_tab |>
  mutate(across(c(station_coverage_gap, pretraining_coverage_gap), ~round(.x, 4))) |>
  gt(groupname_col = "metric") |>
  tab_header(title = md("**Static covariate-space coverage: 734 stations vs. ~20,000-site pretraining sample**"),
             subtitle = "Lower = better covariate-space coverage. See figs/covariate_coverage_{main,si}.png") |>
  cols_label(choice = "Choice / arbitrary setting", station_coverage_gap = "Stations",
             pretraining_coverage_gap = "Pretraining") |>
  fmt_number(c(station_coverage_gap, pretraining_coverage_gap), decimals = 4) |>
  sub_missing(missing_text = "--") |>
  tab_style(cell_text(weight = "bold"), cells_row_groups()) |>
  opt_table_outline() |>
  tab_options(table.font.size = px(11), data_row.padding = px(4), column_labels.font.weight = "bold",
              row_group.font.weight = "bold")
gtsave(g, glue("{figs_dir}/covariate_coverage_summary_table.png"), expand = 30, zoom = 2.5, vwidth = 1200)
message("wrote figs/covariate_coverage_summary_table.png")

message("== Stage 2 (figures + table) complete. ==")
