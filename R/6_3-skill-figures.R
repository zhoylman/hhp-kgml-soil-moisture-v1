library(tidyverse)
library(sf)
library(glue)

# ============================================================================
#  CONSUMER — skill figures from the validation CSVs (robust sites only).
#  k-fold: KGML vs SPoRT-LIS boxplots (KGE/r/|%bias|) + spatial diff maps.
#  OOS:    KGML standalone skill boxplots by network + KGE map.
#  Reuses sm_eval_utils plotting functions; no extraction.
# ============================================================================

repo = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
source(glue("{repo}/R/sm_eval_utils.R"))
tables_dir = glue("{repo}/tables"); figs_dir = glue("{repo}/figs")

missouri_basin = sf::read_sf("~/temp/WBD_10_HU2_Shape/Shape/WBDHU2.shp") |>
  sf::st_transform(5070) |> dplyr::select(-name)

depth_name = c(Shallow = "Shallow Soil Moisture (0-10cm)", Middle = "Mid-depth Soil Moisture (10-50cm)")

# ---- k-fold: KGML vs SPoRT ----
kf = read_csv(glue("{tables_dir}/kfold_validation.csv"), show_col_types = FALSE) |> filter(robust)
site_meta = kf |> distinct(site_id, network, longitude, latitude) |>
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326) |> st_set_crs("EPSG:4326")

for (dep in c("Shallow", "Middle")) {
  k = filter(kf, depth == dep)
  long_k = k |> transmute(site_id, KGE, r, pbias) |> pivot_longer(c(KGE, r, pbias), names_to = "Metric", values_to = "KGML")
  long_s = k |> transmute(site_id, KGE = KGE_sport, r = r_sport, pbias = pbias_sport) |>
    pivot_longer(c(KGE, r, pbias), names_to = "Metric", values_to = "SPoRT-LIS")
  difference = left_join(long_k, long_s, by = c("site_id", "Metric")) |> mutate(diff = KGML - `SPoRT-LIS`)

  plot_skill_boxes_2x2(difference, site_meta, depth_label = depth_name[[dep]],
                       save_path = glue("{figs_dir}/kge_boxes_4panel_{tolower(dep)}.png"))

  diff_sp = difference |> left_join(distinct(k, site_id, longitude, latitude), by = "site_id") |>
    st_as_sf(coords = c("longitude", "latitude"), crs = 4326)
  plot_spatial_differences_by_metric(filter(diff_sp, Metric %in% c("KGE", "r")), missouri_basin,
                                     depth = depth_name[[dep]],
                                     save_path = glue("{figs_dir}/kgml_vs_sport_diff_map_{tolower(dep)}.png"))
  pbias_sp = diff_sp |> filter(Metric == "pbias") |> mutate(diff = abs(`SPoRT-LIS`) - abs(KGML))
  plot_spatial_differences_by_metric(pbias_sp, missouri_basin, depth = depth_name[[dep]], limit = 25,
                                     legend_title = "|% Bias| advantage\n(|SPoRT-LIS| - |KGML|)",
                                     plot_title = "Absolute % Bias Advantage (KGML vs SPoRT-LIS)",
                                     save_path = glue("{figs_dir}/kgml_vs_sport_pbias_map_{tolower(dep)}.png"))
}

# ---- OOS: KGML standalone skill ----
oos = read_csv(glue("{tables_dir}/oos_validation.csv"), show_col_types = FALSE) |> filter(robust) |>
  rename(KGE_KGML = KGE, r_KGML = r, pbias_KGML = pbias)
for (dep in c("Shallow", "Middle")) {
  o = filter(oos, depth == dep)
  plot_kgml_skill_boxes(o, depth_label = glue("{depth_name[[dep]]} — out-of-sample"),
                        save_path = glue("{figs_dir}/kgml_oos_skill_boxes_{tolower(dep)}.png"))
  o_sf = o |> filter(!is.na(longitude), !is.na(latitude)) |> st_as_sf(coords = c("longitude", "latitude"), crs = 4326)
  plot_kgml_skill_map(o_sf, missouri_basin, value_col = "KGE_KGML", limits = c(-1, 1),
                      legend_title = "KGML KGE", plot_title = "KGML out-of-sample skill (KGE)",
                      depth = glue("{depth_name[[dep]]} — out-of-sample"),
                      save_path = glue("{figs_dir}/kgml_oos_kge_map_{tolower(dep)}.png"))
}
message("Wrote skill figures (kfold boxplots/maps + OOS boxplots/maps) to figs/")
