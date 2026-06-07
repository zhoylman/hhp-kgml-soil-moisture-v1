library(tidyverse)
library(hydroGOF)
library(sf)
library(magrittr)
library(furrr)
library(glue)
library(progressr)
library(terra)

# ============================================================================
#  Post-inference skill comparison: KGML (k-fold ensemble) vs SPoRT-LIS
#  - Per-fold KGE/r against in-situ observations on the smoothed-daily rasters.
#  - Boxplot panels + spatial difference maps, generated for BOTH depths.
#
#  Depth is single-sourced: everything (obs, SPoRT baseline, ML folds, figure
#  names) is driven from `d` inside analyze_depth(). The loop at the bottom runs
#  both depths. Do NOT hardcode a depth in run_fold_eval() — that previously
#  desynced the ML side (always "Middle") from the SPoRT/obs baseline.
# ============================================================================

# ---- Config ---------------------------------------------------------------
# Shared helpers (metrics, cached extraction, figures) — single source of truth.
source("/home/zhoylman/hhp-kgml-soil-moisture-v1/R/sm_eval_utils.R")

# Figures and tables belong to THIS repo (~/soil-moisture-ml is the old repo).
figs_dir   = "/home/zhoylman/hhp-kgml-soil-moisture-v1/figs"
tables_dir = "/home/zhoylman/hhp-kgml-soil-moisture-v1/tables"
dir.create(figs_dir,   showWarnings = FALSE, recursive = TRUE)
dir.create(tables_dir, showWarnings = FALSE, recursive = TRUE)

# Per-fold smoothed-daily rasters live on ssd3 (shallow) or ssd4 (middle).
# Resolve across both roots rather than hardcoding one.
smoothed_roots = c("/data/ssd3/soil-moisture-ml-inference",
                   "/data/ssd4/soil-moisture-ml-inference")

resolve_fold_dir = function(lower_depth_flag, fold) {
  cand = file.path(smoothed_roots,
                   glue::glue("predictions-smoothed-daily-{lower_depth_flag}"),
                   glue::glue("fold_{fold}"))
  hit = cand[dir.exists(cand)]
  if (length(hit) == 0) cand[1] else hit[1]   # let downstream warn if truly absent
}

# compute_metrics() now lives in sm_eval_utils.R

# ---- Extraction cache (in-repo) -------------------------------------------
# Extracting the site time series from the 16k+ daily rasters per fold is the
# only slow part. Cache each fold+depth's extracted (date, site_id, ml) table,
# keyed on the dates/sites it covers, and reuse it when a rerun needs only a
# subset. Delete this dir to force a full re-extract (e.g. after rasters are
# reprocessed). NOTE: consider adding `cache/` to .gitignore.
use_cache = TRUE
cache_dir = "/home/zhoylman/hhp-kgml-soil-moisture-v1/cache/extractions"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)

# Per-fold extracted long table (date, site_id, ml): thin wrapper over the
# shared extract_at_sites(), pointing at this fold's smoothed-daily raster dir
# and a per-fold cache file.
get_fold_smoothed = function(lower_depth_flag, fold, obs_dates, site_ids, obs_meta_of_interest) {
  extract_at_sites(
    raster_dir = resolve_fold_dir(lower_depth_flag, fold),
    obs_dates  = obs_dates,
    site_ids   = site_ids,
    meta_xy    = obs_meta_of_interest,
    cache_file = if (use_cache) file.path(cache_dir, glue::glue("{lower_depth_flag}_fold_{fold}.rds")) else NULL,
    label      = glue::glue("Fold {fold} [{lower_depth_flag}]")
  )
}

run_fold_eval = function(fold, depth_flag) {
  tryCatch({
    lower_depth_flag = tolower(depth_flag)

    # ---- Load observations (data frames only) ----
    # KGML is scored against the GENERALIZED obs (Middle = 10-50 cm), matching
    # the fine-tune target. SPoRT uses the sport-specific 10-40 cm obs upstream.
    obs = readr::read_csv(
      "/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-no-frozen.csv",
      show_col_types = FALSE,
      col_types = readr::cols(
        site_id = readr::col_character(),
        date = readr::col_date(),
        soil_moisture = readr::col_double(),
        generalized_depth = readr::col_character()
      )
    ) |>
      dplyr::filter(generalized_depth == depth_flag) |>
      dplyr::select(site_id, date, obs = soil_moisture)

    obs_meta = readr::read_csv(
      "/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-meta.csv",
      show_col_types = FALSE,
      col_types = readr::cols(
        site_id   = readr::col_character(),
        latitude  = readr::col_double(),
        longitude = readr::col_double()
      )
    )

    # ---- Validation sites for this fold ----
    val_csv = glue::glue("/data/ssd2/soil-moisture-ml/split-definitions-kfold-{lower_depth_flag}/validation_split_fold_{fold}.csv")
    validation_sites_temp = readr::read_csv(
      val_csv, show_col_types = FALSE,
      col_types = readr::cols(site_id = readr::col_character())
    )
    site_ids = unique(validation_sites_temp$site_id)

    obs_of_interest       = dplyr::filter(obs, site_id %in% site_ids)
    obs_meta_of_interest  = dplyr::filter(obs_meta, site_id %in% site_ids)

    if (nrow(obs_of_interest) == 0) {
      cli::cli_warn("Fold {fold}: no observations after filtering; returning empty.")
      return(list(results = tibble::tibble(), smoothed_data = tibble::tibble()))
    }

    # ---- Smoothed values at obs points (cache or on-the-fly extract) ----
    obs_dates     = sort(unique(obs_of_interest$date))
    smoothed_data = get_fold_smoothed(lower_depth_flag, fold, obs_dates,
                                      site_ids, obs_meta_of_interest)

    if (nrow(smoothed_data) == 0) {
      cli::cli_warn("Fold {fold} [{lower_depth_flag}]: no prediction files for obs dates; returning empty.")
      return(list(results = tibble::tibble(), smoothed_data = tibble::tibble()))
    }

    # ---- Join & metrics ----
    results =
      obs_of_interest |>
      dplyr::left_join(smoothed_data, by = c("date", "site_id")) |>
      dplyr::group_by(site_id) |>
      dplyr::group_modify(~ compute_metrics(.x, site_id = unique(.x$site_id), fold = glue::glue("{fold}"))) |>
      dplyr::ungroup() |>
      dplyr::mutate(fold = fold)

    list(results = results, smoothed_data = smoothed_data)
  },
  error = function(e) {
    cli::cli_warn("Fold {fold} failed: {e$message}")
    list(results = tibble::tibble(), smoothed_data = tibble::tibble())
  })
}

# ---- Depth-independent inputs (load once) ---------------------------------
site_meta = read_csv("/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-meta.csv") |>
  sf::st_as_sf(coords = c('longitude', 'latitude')) %>%
  sf::st_set_crs('EPSG:4326')

missouri_basin <- sf::read_sf("~/temp/WBD_10_HU2_Shape/Shape/WBDHU2.shp") |>
  sf::st_transform(5070) |>
  select(-name)

# Plotting functions (plot_skill_boxes_2x2, plot_spatial_differences_by_metric) now in sm_eval_utils.R

# ---------------------------------------------------------------------------
#  Per-depth analysis (everything below is driven by `d`)
# ---------------------------------------------------------------------------
analyze_depth = function(d) {
  depth       = c('shallow', 'middle')[d]
  depth_2     = c('Shallow', 'Middle')[d]
  sport_depth = c('SPoRT_raw_0-10cm', 'SPoRT_raw_10-40cm')[d]
  depth_name  = c('Shallow Soil Moisture (0-10cm)',
                  'Mid-depth Soil Moisture (10-50cm)')[d]

  message(glue::glue("\n==== Analyzing depth: {depth_2} ===="))

  # ---- SPoRT-LIS baseline for this depth ----
  # FAIR COMPARISON BY DESIGN: each model is scored against the obs depth band
  # it was actually built for (see 1_1-preprocess-soil-moisture-obs.R):
  #   * SPoRT-LIS -> "sport-specific" obs: Middle = 10-40 cm (depth >10 & <=40)
  #   * KGML      -> "generalized"    obs: Middle = 10-50 cm (depth >10 & <=50, run_fold_eval)
  # Shallow (depth <=10, 0-10 cm) is identical in both files, so this only
  # affects Middle. Do NOT "unify" these to one obs file.
  obs_sport = readr::read_csv("/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-sport-specific-no-frozen.csv") |>
    dplyr::filter(generalized_depth == depth_2) |>
    dplyr::select(site_id, date, obs = soil_moisture)

  sport_sims = read_csv("/data/ssd2/soil-moisture-ml/observations/observational-sites-raw-sport.csv") |>
    mutate(time = as.Date(time)) |>
    rename(date = time) |>
    filter(var == sport_depth) |>
    pivot_longer(cols = -c('var', 'date'))

  sport_results = obs_sport |>
    left_join(sport_sims |>
                select(site_id = name, date, sport = value),
              by = c("site_id", "date")) |>
    drop_na(obs, sport) |>
    group_by(site_id) |>
    summarise(n     = n(),
              KGE   = hydroGOF::KGE(sport, obs),
              r     = cor(sport, obs),
              pbias = hydroGOF::pbias(sport, obs),
              .groups = "drop")

  # ---- KGML k-fold ensemble skill for this depth ----
  future::plan(future::multisession, workers = 10)
  outs = furrr::future_map(1:10, ~ run_fold_eval(.x, depth_flag = depth_2),
                           .progress = interactive(),
                           .options = furrr::furrr_options(seed = NULL))
  future::plan(future::sequential)

  all_results  = outs |> purrr::map("results") |> dplyr::bind_rows()

  ml_median    = median(all_results$KGE, na.rm = TRUE)
  sport_median = median(sport_results$KGE, na.rm = TRUE)
  message(glue::glue('[{depth_2}] ML median KGE = {round(ml_median,3)}, SPoRT-LIS median KGE = {round(sport_median,3)}'))
  message(glue::glue('[{depth_2}] ML is {round(((ml_median - sport_median)/sport_median)*100, 2)}% better across CONUS'))

  umrb_ids = site_meta |> filter(network == 'UMRB Mesonet') %$% site_id
  ml_median_UMRB    = all_results  |> filter(site_id %in% umrb_ids) |> summarise(median = median(KGE, na.rm = TRUE))
  sport_median_UMRB = sport_results |> filter(site_id %in% umrb_ids) |> summarise(median = median(KGE, na.rm = TRUE))
  message(glue::glue('[{depth_2}] ML median KGE = {round(ml_median_UMRB$median,3)}, SPoRT-LIS median KGE = {round(sport_median_UMRB$median,3)} (UMRB)'))
  message(glue::glue('[{depth_2}] ML is {round(((ml_median_UMRB$median - sport_median_UMRB$median)/sport_median_UMRB$median)*100, 2)}% better across UMRB'))

  # ---- Tidy per-metric scores, side-by-side ----
  ml_scores = all_results |>
    dplyr::select(site_id, KGE, r, pbias) |>
    tidyr::pivot_longer(cols = c(KGE, r, pbias), names_to = "Metric", values_to = "KGML")

  sport_scores = sport_results |>
    dplyr::select(site_id, KGE, r, pbias) |>
    tidyr::pivot_longer(cols = c(KGE, r, pbias), names_to = "Metric", values_to = "SPoRT-LIS")

  panel = dplyr::left_join(ml_scores, sport_scores, by = c("site_id", "Metric"))

  difference = panel |>
    dplyr::mutate(diff = KGML - `SPoRT-LIS`)

  difference_spatial = difference |>
    dplyr::left_join(site_meta, by = "site_id") |>
    sf::st_as_sf()

  # ---- Per-site results table (one row per site, both models side-by-side) ----
  site_info = site_meta |>
    dplyr::mutate(lon = sf::st_coordinates(geometry)[, 1],
                  lat = sf::st_coordinates(geometry)[, 2]) |>
    sf::st_drop_geometry() |>
    dplyr::select(site_id, network, lon, lat)

  per_site = panel |>
    dplyr::rename(SPoRT = `SPoRT-LIS`) |>
    dplyr::mutate(diff = KGML - SPoRT) |>
    tidyr::pivot_wider(names_from = Metric, values_from = c(KGML, SPoRT, diff),
                       names_glue = "{Metric}_{.value}") |>
    dplyr::left_join(all_results   |> dplyr::select(site_id, fold, n_ml = n), by = "site_id") |>
    dplyr::left_join(sport_results |> dplyr::select(site_id, n_sport = n),    by = "site_id") |>
    dplyr::left_join(site_info, by = "site_id") |>
    dplyr::transmute(depth = depth_2, network, site_id, fold,
                     n_ml, n_sport, lon, lat,
                     KGE_KGML,   KGE_SPoRT,   KGE_diff,
                     r_KGML,     r_SPoRT,     r_diff,
                     pbias_KGML, pbias_SPoRT, pbias_diff,
                     # absolute % bias advantage: > 0 means KGML is closer to 0
                     abs_pbias_adv = abs(pbias_SPoRT) - abs(pbias_KGML)) |>
    dplyr::arrange(network, site_id)

  readr::write_csv(per_site, glue::glue("{tables_dir}/per_site_skill_{depth}.csv"))
  message(glue::glue("[{depth_2}] wrote per-site table ({nrow(per_site)} sites)."))

  # ---- Figures ----
  plot_skill_boxes_2x2(
    difference  = difference,
    site_meta   = site_meta,
    depth_label = depth_name,
    save_path   = glue::glue("{figs_dir}/kge_boxes_4panel_{depth}.png")
  )

  # KGE & r: higher = better, diff = KGML - SPoRT-LIS, clamp +/-0.4
  plot_spatial_differences_by_metric(
    data           = difference_spatial |> dplyr::filter(Metric %in% c("KGE", "r")),
    missouri_basin = missouri_basin,
    depth          = depth_name,
    save_path      = glue::glue("{figs_dir}/kgml_vs_sport_diff_map_{depth}.png")
  )

  # |% Bias|: closer to 0 = better. Plot the advantage |SPoRT| - |KGML| so that
  # positive (purple) = KGML has the smaller-magnitude bias. Percent scale.
  pbias_spatial = difference_spatial |>
    dplyr::filter(Metric == "pbias") |>
    dplyr::mutate(diff = abs(`SPoRT-LIS`) - abs(KGML))

  plot_spatial_differences_by_metric(
    data           = pbias_spatial,
    missouri_basin = missouri_basin,
    depth          = depth_name,
    limit          = 25,
    legend_title   = "|% Bias| advantage\n(|SPoRT-LIS| - |KGML|)",
    plot_title     = "Absolute % Bias Advantage (KGML vs SPoRT-LIS)",
    save_path      = glue::glue("{figs_dir}/kgml_vs_sport_pbias_map_{depth}.png")
  )

  invisible(list(all_results = all_results, sport_results = sport_results, per_site = per_site))
}

# Run both depths (1 = shallow, 2 = middle)
results_by_depth = lapply(1:2, analyze_depth)
names(results_by_depth) = c("shallow", "middle")

# ===========================================================================
#  Tables (per-site + aggregations) -> tables/
# ===========================================================================
per_site_all = dplyr::bind_rows(lapply(results_by_depth, `[[`, "per_site"))
readr::write_csv(per_site_all, glue::glue("{tables_dir}/per_site_skill_all.csv"))

umrb_ids = site_meta |> dplyr::filter(network == 'UMRB Mesonet') %$% site_id

# Median/mean skill by depth, for All sites and the UMRB network, with the
# % improvement in median KGE (KGML vs SPoRT-LIS).
agg_region = function(df, region_label) {
  df |>
    dplyr::group_by(depth) |>
    dplyr::summarise(
      n_sites        = dplyr::n(),
      n_obs_med      = median(n_ml, na.rm = TRUE),   # typical obs-days/site (KGML)
      n_obs_min      = min(n_ml, na.rm = TRUE),
      n_sport_valid  = sum(!is.na(KGE_SPoRT)),        # sites with a computable SPoRT metric
      KGE_KGML_med   = median(KGE_KGML, na.rm = TRUE),
      KGE_SPoRT_med  = median(KGE_SPoRT, na.rm = TRUE),
      r_KGML_med     = median(r_KGML,   na.rm = TRUE),
      r_SPoRT_med    = median(r_SPoRT,  na.rm = TRUE),
      KGE_KGML_mean  = mean(KGE_KGML,   na.rm = TRUE),
      KGE_SPoRT_mean = mean(KGE_SPoRT,  na.rm = TRUE),
      KGE_diff_med   = median(KGE_diff, na.rm = TRUE),
      # |% bias| (closer to 0 = better)
      abs_pbias_KGML_med  = median(abs(pbias_KGML),  na.rm = TRUE),
      abs_pbias_SPoRT_med = median(abs(pbias_SPoRT), na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      region = region_label,
      KGE_pct_better = round((KGE_KGML_med - KGE_SPoRT_med) / abs(KGE_SPoRT_med) * 100, 1)
    )
}

skill_summary = dplyr::bind_rows(
  agg_region(per_site_all, "All sites"),
  agg_region(dplyr::filter(per_site_all, site_id %in% umrb_ids), "UMRB Network")
) |>
  dplyr::relocate(region, depth, n_sites)
readr::write_csv(skill_summary, glue::glue("{tables_dir}/skill_summary.csv"))

# Per-network breakdown by depth.
skill_by_network = per_site_all |>
  dplyr::group_by(depth, network) |>
  dplyr::summarise(
    n_sites       = dplyr::n(),
    n_obs_med     = median(n_ml, na.rm = TRUE),
    n_obs_min     = min(n_ml, na.rm = TRUE),
    n_sport_valid = sum(!is.na(KGE_SPoRT)),
    KGE_KGML_med  = median(KGE_KGML, na.rm = TRUE),
    KGE_SPoRT_med = median(KGE_SPoRT, na.rm = TRUE),
    r_KGML_med    = median(r_KGML,   na.rm = TRUE),
    r_SPoRT_med   = median(r_SPoRT,  na.rm = TRUE),
    KGE_diff_med  = median(KGE_diff, na.rm = TRUE),
    abs_pbias_KGML_med  = median(abs(pbias_KGML),  na.rm = TRUE),
    abs_pbias_SPoRT_med = median(abs(pbias_SPoRT), na.rm = TRUE),
    .groups = "drop"
  ) |>
  dplyr::arrange(depth, dplyr::desc(n_sites))
readr::write_csv(skill_by_network, glue::glue("{tables_dir}/skill_by_network.csv"))

message(glue::glue("\nWrote tables to {tables_dir}/:"))
message("  per_site_skill_shallow.csv, per_site_skill_middle.csv, per_site_skill_all.csv")
message("  skill_summary.csv, skill_by_network.csv")
