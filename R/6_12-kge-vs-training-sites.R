library(tidyverse)
library(glue)
library(furrr)
library(hydroGOF)

# ============================================================================
#  Learning curve: out-of-sample median KGE vs. number of sites in training.
#  Recreates the old kge_by_fold-{depth}.png (old repo R/X_check-results-updated.R).
#  Reads the training-size ablation runs (uNET_<N>_sites_*) in results-{depth},
#  each a model trained on N sites and evaluated on the held-out sites (per-site
#  hydrograph_data/data_*.csv with yhat/yobs), plus a full-model point from the
#  10-fold k-fold results. Dashed line = SPoRT-LIS median KGE.
# ============================================================================

data_root = "/data/ssd2/soil-moisture-ml"
obs_dir   = glue("{data_root}/observations")
figs_dir  = "/home/zhoylman/hhp-kgml-soil-moisture-v1/figs"

kge_from_file = function(path) {
  df = suppressMessages(readr::read_csv(path, show_col_types = FALSE)) |>
    dplyr::filter(!is.na(yobs), !is.na(yhat))
  if (nrow(df) < 5) return(NA_real_)
  hydroGOF::KGE(df$yhat, df$yobs)
}

make_figure = function(d) {
  depth       = c("shallow", "middle")[d]
  depth_2     = c("Shallow", "Middle")[d]
  sport_depth = c("SPoRT_raw_0-10cm", "SPoRT_raw_10-40cm")[d]
  depth_name  = c("Shallow Soil Moisture (0-10cm)", "Mid-depth Soil Moisture (10-40cm)")[d]
  ablation    = glue("{data_root}/results-{depth}")
  kfold_dir   = glue("{data_root}/results-kfold-{depth}")

  # ---- ablation learning curve: median KGE per training-site count ----
  abl_files = list.files(ablation, pattern = "data_.*\\.csv$", recursive = TRUE, full.names = TRUE)
  abl = tibble(path = abl_files,
               n_sites = as.integer(str_extract(str_extract(path, "uNET_\\d+_sites"), "\\d+"))) |>
    filter(!is.na(n_sites)) |>
    mutate(KGE = future_map_dbl(path, kge_from_file)) |>
    group_by(n_sites) |>
    summarise(median = median(KGE, na.rm = TRUE), .groups = "drop")

  # ---- full-model point: 10-fold median KGE at the full training-set size ----
  kf_files = list.files(kfold_dir, pattern = "data_.*\\.csv$", recursive = TRUE, full.names = TRUE)
  kf_kge   = future_map_dbl(kf_files, kge_from_file)
  n_train_full = readr::read_csv(glue("{data_root}/split-definitions-kfold-{depth}/train_split_fold_1.csv"),
                                 show_col_types = FALSE)$site_id |> unique() |> length()
  curve = bind_rows(abl, tibble(n_sites = n_train_full, median = median(kf_kge, na.rm = TRUE))) |>
    arrange(n_sites)

  # ---- SPoRT-LIS reference (median raw KGE vs obs, this depth) ----
  obs_sport = readr::read_csv(glue("{obs_dir}/final-soil-moisture-data-generalized-sport-specific-no-frozen.csv"),
                              show_col_types = FALSE) |>
    filter(generalized_depth == depth_2) |> transmute(site_id = as.character(site_id), date, obs = soil_moisture)
  sport_sims = readr::read_csv(glue("{obs_dir}/observational-sites-raw-sport.csv"), show_col_types = FALSE) |>
    mutate(date = as.Date(time)) |> filter(var == sport_depth) |>
    pivot_longer(cols = -c(var, date, time), names_to = "site_id", values_to = "sport")
  sport_kge = obs_sport |>
    inner_join(sport_sims |> select(site_id, date, sport), by = c("site_id", "date")) |>
    drop_na(obs, sport) |> group_by(site_id) |>
    summarise(KGE = hydroGOF::KGE(sport, obs), .groups = "drop") |>
    pull(KGE) |> median(na.rm = TRUE)

  # ---- plot ----
  plot = ggplot(curve, aes(x = n_sites, y = median)) +
    geom_smooth(method = "lm", formula = y ~ x) +
    geom_point(size = 3, color = "#341539") +
    geom_hline(yintercept = sport_kge, linetype = "dashed", color = "darkgreen", linewidth = 1) +
    annotate("text", x = max(curve$n_sites), y = sport_kge - 0.01,
             label = "SPoRT-LIS", color = "darkgreen", hjust = 1, size = 4) +
    labs(title = "KGE vs. Number of Sites in Training", subtitle = depth_name,
         x = "Number of Sites", y = "Median KGE",
         caption = "KGE = Out-of-Sample\nKling-Gupta Efficiency") +
    theme_bw(base_size = 15) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5),
          axis.title = element_text(face = "bold"),
          axis.text = element_text(color = "black"))

  out = glue("{figs_dir}/kge_vs_training_sites_{depth}.png")
  ggsave(plot, file = out, height = 6, width = 6, dpi = 300, bg = "white")
  message(glue("[{depth_2}] wrote {out} | SPoRT-LIS median KGE = {round(sport_kge,3)} | {nrow(curve)} points"))
  print(as.data.frame(curve |> mutate(median = round(median, 3))))
}

plan(multisession, workers = 20)
for (d in 1:2) make_figure(d)
plan(sequential)
