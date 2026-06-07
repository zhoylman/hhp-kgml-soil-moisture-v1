library(tidyverse)
library(sf)
library(glue)
library(rnaturalearth)

# ============================================================================
#  Paper figure: maps of (a) the pre-training sites and (b) the soil-moisture
#  observation sites (colored by network). CONUS, EPSG:5070. Standalone.
# ============================================================================

figs_dir = "/home/zhoylman/hhp-kgml-soil-moisture-v1/figs"
data_root = "/data/ssd2/soil-moisture-ml"
exclude_networks = c("OK Mesonet")   # not used in the study

`%notin%` = Negate(`%in%`)

# ---- CONUS state boundaries ----
us_sf = rnaturalearth::ne_states(country = "United States of America", returnclass = "sf") |>
  filter(name %notin% c("Alaska", "Hawaii", "Puerto Rico", "Virgin Islands", "Guam",
                        "American Samoa", "Northern Mariana Islands")) |>
  st_transform(5070)

# ---- observation sites (by network) ----
# Main analysis meta has the FULL UMRB Mesonet (267) + SNTL/SCAN/USCRN; NEON
# (OOS) comes from the additional_OOS_data meta. (The additional file mislabels
# UMRB as "MT Mesonet" and is missing ~91 of them, so don't use it for UMRB.)
main_meta = read_csv(file.path(data_root, "observations/final-soil-moisture-data-generalized-meta.csv"),
                     show_col_types = FALSE) |>
  transmute(network, site_id = as.character(site_id), longitude, latitude)
neon_meta = read_csv(file.path(data_root, "additional_OOS_data/station-meta-conus-w-data-final.csv"),
                     show_col_types = FALSE) |>
  filter(network == "NEON") |>
  transmute(network, site_id = as.character(site_id), longitude, latitude)
obs_meta = bind_rows(main_meta, neon_meta) |>
  filter(network %notin% exclude_networks) |>
  add_count(network) |>
  mutate(network_label = glue("{network}\n(n = {n})")) |>        # 2 lines so labels don't clip
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326) |>
  st_transform(5070)

# ---- pre-training sites ----
pretrain = st_read(file.path(data_root, "random-pretraining-roi/pretraining-roi.geojson"), quiet = TRUE) |>
  st_transform(5070)

base_theme = theme_bw(base_size = 16) +
  theme(plot.title = element_text(hjust = 0.5, size = 18),
        plot.subtitle = element_text(hjust = 0.5, size = 12),
        legend.position = "bottom", legend.key = element_blank(),
        legend.background = element_blank(), legend.text = element_text(size = 12),
        axis.text = element_blank(), axis.ticks = element_blank(), panel.grid = element_blank(),
        strip.background = element_rect(colour = "transparent", fill = "transparent"))

# ---- (a) pre-training sites ----
p_pretrain = ggplot() +
  geom_sf(data = us_sf, fill = "transparent", color = "black", size = 1.5) +
  geom_sf(data = pretrain, color = "#4B0092", size = 0.3, alpha = 0.3) +
  ggtitle("Pre-training Sites (SPoRT-LIS)",
          subtitle = glue("(n sites = {format(nrow(pretrain), big.mark = ',')})")) +
  base_theme

# ---- (b) observation sites by network ----
p_obs = ggplot() +
  geom_sf(data = us_sf, fill = "transparent", color = "black", size = 1.5) +
  geom_sf(data = obs_meta, aes(fill = network_label),
          shape = 21, color = "black", size = 1.5, stroke = 0.3, alpha = 0.6) +
  scale_fill_brewer(palette = "Set1") +
  ggtitle("Soil Moisture Observations Across the U.S.",
          subtitle = glue("(n sites = {nrow(obs_meta)})")) +
  labs(fill = NULL) +
  guides(fill = guide_legend(override.aes = list(size = 3, alpha = 1))) +
  base_theme

ggsave(file.path(figs_dir, "site_map_pretraining.png"),  p_pretrain, width = 8, height = 6, dpi = 300, bg = "white")
ggsave(file.path(figs_dir, "site_map_observations.png"), p_obs,      width = 8, height = 6.5, dpi = 300, bg = "white")

# ---- combined two-panel (if patchwork available) ----
if (requireNamespace("patchwork", quietly = TRUE)) {
  library(patchwork)
  combined = p_pretrain / p_obs + patchwork::plot_annotation(tag_levels = "a")
  ggsave(file.path(figs_dir, "site_map_combined.png"), combined, width = 8, height = 12, dpi = 300, bg = "white")
}

message(glue("Wrote site maps to {figs_dir}/ (obs networks: {paste(sort(unique(obs_meta$network)), collapse=', ')})"))
