library(tidyverse)
library(hydroGOF)
library(sf)
library(magrittr)
library(glue)
library(terra)

# ============================================================================
#  TRUE out-of-sample validation: KGML 10-fold ENSEMBLE vs SPoRT-LIS at sites
#  that were NEVER in any fold's train OR validation set.
#
#  The k-fold splits were built on long-record sites; ~258 (shallow) / ~273
#  (middle) obs sites — mostly the newer/shorter UMRB Mesonet stations — were
#  excluded entirely. Those are genuinely independent: no fold model ever saw
#  them, so we evaluate the full ensemble-median product against them.
#
#  Same fair-comparison obs as 6_1: KGML vs generalized obs (Middle 10-50 cm),
#  SPoRT vs sport-specific obs (Middle 10-40 cm). Shares helpers with 6_1.
# ============================================================================

source("/home/zhoylman/hhp-kgml-soil-moisture-v1/R/sm_eval_utils.R")

# ---- Config ---------------------------------------------------------------
figs_dir   = "/home/zhoylman/hhp-kgml-soil-moisture-v1/figs"
tables_dir = "/home/zhoylman/hhp-kgml-soil-moisture-v1/tables"
cache_dir  = "/home/zhoylman/hhp-kgml-soil-moisture-v1/cache/extractions-oos"
invisible(lapply(c(figs_dir, tables_dir, cache_dir),
                 dir.create, showWarnings = FALSE, recursive = TRUE))

obs_dir   = "/data/ssd2/soil-moisture-ml/observations"
split_dir = "/data/ssd2/soil-moisture-ml/split-definitions-kfold"   # + -{depth}

# Ensemble-median rasters live on ssd3 (shallow) / ssd4 (middle); resolve both.
ens_roots = c("/data/ssd3/soil-moisture-ml-inference",
              "/data/ssd4/soil-moisture-ml-inference")
resolve_ensemble_dir = function(lower_depth) {
  cand = file.path(ens_roots, glue::glue("ensemble-smoothed-daily-{lower_depth}"), "median")
  hit = cand[dir.exists(cand)]
  if (length(hit) == 0) cand[1] else hit[1]
}

# A site needs at least this many obs days for its KGE/r/pbias to be "robust".
# We report ALL never-seen sites and, separately, this robust subset.
robust_min_days = 365L

# ---- Depth-independent inputs ---------------------------------------------
meta_df = read_csv(file.path(obs_dir, "final-soil-moisture-data-generalized-meta.csv"),
                   show_col_types = FALSE) |>
  dplyr::transmute(site_id = as.character(site_id), network, longitude, latitude) |>
  tidyr::drop_na(longitude, latitude)

site_meta = meta_df |>
  sf::st_as_sf(coords = c("longitude", "latitude")) |>
  sf::st_set_crs("EPSG:4326")

missouri_basin = sf::read_sf("~/temp/WBD_10_HU2_Shape/Shape/WBDHU2.shp") |>
  sf::st_transform(5070) |>
  dplyr::select(-name)

# Sites that appear in ANY fold's train OR validation split (for this depth).
kfold_site_ids = function(lower_depth) {
  spl = list.files(glue::glue("{split_dir}-{lower_depth}"),
                   pattern = "(train|validation)_split_fold_.*csv$", full.names = TRUE)
  spl |>
    purrr::map(~ readr::read_csv(.x, show_col_types = FALSE)$site_id) |>
    unlist() |> unique() |> as.character()
}

# ---------------------------------------------------------------------------
#  Per-depth OOS analysis
# ---------------------------------------------------------------------------
analyze_oos = function(d) {
  depth       = c('shallow', 'middle')[d]
  depth_2     = c('Shallow', 'Middle')[d]
  sport_depth = c('SPoRT_raw_0-10cm', 'SPoRT_raw_10-40cm')[d]
  depth_name  = c('Shallow Soil Moisture (0-10cm) — out-of-sample',
                  'Mid-depth Soil Moisture (10-50cm) — out-of-sample')[d]

  message(glue::glue("\n==== OOS validation, depth: {depth_2} ===="))

  # ---- KGML obs (generalized; Middle = 10-50 cm) ----
  obs_all = readr::read_csv(
    file.path(obs_dir, "final-soil-moisture-data-generalized-no-frozen.csv"),
    show_col_types = FALSE,
    col_types = readr::cols(site_id = readr::col_character(), date = readr::col_date(),
                            soil_moisture = readr::col_double(), generalized_depth = readr::col_character())
  ) |>
    dplyr::filter(generalized_depth == depth_2) |>
    dplyr::select(site_id, date, obs = soil_moisture)

  # never-seen = has obs at this depth, has coords, NOT in any fold
  oos_ids = setdiff(unique(obs_all$site_id), kfold_site_ids(depth))
  oos_ids = intersect(oos_ids, meta_df$site_id)
  message(glue::glue("[{depth_2}] never-seen sites with coords: {length(oos_ids)}"))

  obs       = dplyr::filter(obs_all, site_id %in% oos_ids)
  meta_oos  = dplyr::filter(meta_df, site_id %in% oos_ids) |> dplyr::arrange(site_id)
  obs_dates = sort(unique(obs$date))
  site_ids  = sort(unique(obs$site_id))
  meta_oos  = dplyr::filter(meta_oos, site_id %in% site_ids)

  # ---- KGML ensemble extraction (cached) + metrics ----
  smoothed = extract_at_sites(
    raster_dir = resolve_ensemble_dir(depth),
    obs_dates  = obs_dates, site_ids = site_ids, meta_xy = meta_oos,
    cache_file = file.path(cache_dir, glue::glue("{depth}_ensemble.rds")),
    label      = glue::glue("OOS ensemble [{depth}]")
  )

  ml_results = obs |>
    dplyr::left_join(smoothed, by = c("date", "site_id")) |>
    dplyr::group_by(site_id) |>
    dplyr::group_modify(~ compute_metrics(.x, site_id = unique(.x$site_id), fold = "ensemble")) |>
    dplyr::ungroup()

  # ---- SPoRT-LIS baseline at the same never-seen sites ----
  obs_sport = readr::read_csv(
    file.path(obs_dir, "final-soil-moisture-data-generalized-sport-specific-no-frozen.csv"),
    show_col_types = FALSE) |>
    dplyr::filter(generalized_depth == depth_2) |>
    dplyr::transmute(site_id = as.character(site_id), date, obs = soil_moisture) |>
    dplyr::filter(site_id %in% oos_ids)

  sport_sims = readr::read_csv(file.path(obs_dir, "observational-sites-raw-sport.csv"),
                               show_col_types = FALSE) |>
    dplyr::mutate(time = as.Date(time)) |>
    dplyr::rename(date = time) |>
    dplyr::filter(var == sport_depth) |>
    tidyr::pivot_longer(cols = -c('var', 'date'))

  sport_results = obs_sport |>
    dplyr::left_join(sport_sims |> dplyr::select(site_id = name, date, sport = value),
                     by = c("site_id", "date")) |>
    tidyr::drop_na(obs, sport) |>
    dplyr::group_by(site_id) |>
    dplyr::summarise(n     = dplyr::n(),
                     KGE   = hydroGOF::KGE(sport, obs),
                     r     = cor(sport, obs),
                     pbias = hydroGOF::pbias(sport, obs),
                     .groups = "drop")

  # ---- Side-by-side panel ----
  ml_scores = ml_results |>
    dplyr::select(site_id, KGE, r, pbias) |>
    tidyr::pivot_longer(cols = c(KGE, r, pbias), names_to = "Metric", values_to = "KGML")
  sport_scores = sport_results |>
    dplyr::select(site_id, KGE, r, pbias) |>
    tidyr::pivot_longer(cols = c(KGE, r, pbias), names_to = "Metric", values_to = "SPoRT-LIS")
  panel = dplyr::left_join(ml_scores, sport_scores, by = c("site_id", "Metric"))

  # ---- Per-site table ----
  per_site = panel |>
    dplyr::rename(SPoRT = `SPoRT-LIS`) |>
    dplyr::mutate(diff = KGML - SPoRT) |>
    tidyr::pivot_wider(names_from = Metric, values_from = c(KGML, SPoRT, diff),
                       names_glue = "{Metric}_{.value}") |>
    dplyr::left_join(ml_results    |> dplyr::select(site_id, n_ml = n),    by = "site_id") |>
    dplyr::left_join(sport_results |> dplyr::select(site_id, n_sport = n), by = "site_id") |>
    dplyr::left_join(meta_df, by = "site_id") |>
    dplyr::transmute(depth = depth_2, network, site_id, n_ml, n_sport, longitude, latitude,
                     robust = !is.na(n_ml) & n_ml >= robust_min_days,
                     KGE_KGML,   KGE_SPoRT,   KGE_diff,
                     r_KGML,     r_SPoRT,     r_diff,
                     pbias_KGML, pbias_SPoRT, pbias_diff,
                     abs_pbias_adv = abs(pbias_SPoRT) - abs(pbias_KGML)) |>
    dplyr::arrange(network, site_id)

  readr::write_csv(per_site, glue::glue("{tables_dir}/per_site_oos_{depth}.csv"))
  message(glue::glue("[{depth_2}] OOS sites scored: {nrow(per_site)} (robust >= {robust_min_days}d: {sum(per_site$robust)})"))
  message(glue::glue("[{depth_2}] median KGE  KGML={round(median(per_site$KGE_KGML,na.rm=TRUE),3)} vs SPoRT={round(median(per_site$KGE_SPoRT,na.rm=TRUE),3)}"))

  # ---- Figures: KGML standalone OOS skill (NOT vs SPoRT — SPoRT sims don't
  #      cover these post-2022 sites; see n_sport_valid). SPoRT stays in the
  #      tables where computable.) ----
  plot_kgml_skill_boxes(
    per_site = per_site, depth_label = depth_name,
    save_path = glue::glue("{figs_dir}/kgml_oos_skill_boxes_{depth}.png")
  )

  per_site_sf = per_site |>
    dplyr::filter(!is.na(longitude), !is.na(latitude)) |>
    sf::st_as_sf(coords = c("longitude", "latitude"), crs = 4326)

  plot_kgml_skill_map(
    data = per_site_sf, missouri_basin = missouri_basin,
    value_col = "KGE_KGML", limits = c(-1, 1), legend_title = "KGML KGE",
    plot_title = "KGML out-of-sample skill (KGE)", depth = depth_name,
    save_path = glue::glue("{figs_dir}/kgml_oos_kge_map_{depth}.png")
  )

  invisible(per_site)
}

# Run both depths
per_site_all = dplyr::bind_rows(lapply(1:2, analyze_oos))
readr::write_csv(per_site_all, glue::glue("{tables_dir}/per_site_oos_all.csv"))

# ---- Aggregations (all sites, robust subset, and UMRB) --------------------
umrb_ids = meta_df |> dplyr::filter(network == "UMRB Mesonet") |> dplyr::pull(site_id)

agg = function(df, region_label) {
  df |>
    dplyr::group_by(depth) |>
    dplyr::summarise(
      n_sites             = dplyr::n(),
      n_obs_med           = median(n_ml, na.rm = TRUE),   # typical obs-days/site (KGML)
      n_obs_min           = min(n_ml, na.rm = TRUE),
      n_sport_valid       = sum(!is.na(KGE_SPoRT)),        # sites with a computable SPoRT metric
      KGE_KGML_med        = median(KGE_KGML, na.rm = TRUE),
      KGE_SPoRT_med       = median(KGE_SPoRT, na.rm = TRUE),
      r_KGML_med          = median(r_KGML, na.rm = TRUE),
      r_SPoRT_med         = median(r_SPoRT, na.rm = TRUE),
      abs_pbias_KGML_med  = median(abs(pbias_KGML), na.rm = TRUE),
      abs_pbias_SPoRT_med = median(abs(pbias_SPoRT), na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::mutate(region = region_label,
                  KGE_pct_better = round((KGE_KGML_med - KGE_SPoRT_med) / abs(KGE_SPoRT_med) * 100, 1))
}

skill_summary_oos = dplyr::bind_rows(
  agg(per_site_all, "All OOS sites"),
  agg(dplyr::filter(per_site_all, robust), "All OOS (robust >=365d)"),
  agg(dplyr::filter(per_site_all, site_id %in% umrb_ids), "UMRB OOS"),
  agg(dplyr::filter(per_site_all, site_id %in% umrb_ids, robust), "UMRB OOS (robust)")
) |>
  dplyr::relocate(region, depth, n_sites)
readr::write_csv(skill_summary_oos, glue::glue("{tables_dir}/skill_summary_oos.csv"))

skill_by_network_oos = per_site_all |>
  dplyr::group_by(depth, network) |>
  dplyr::summarise(
    n_sites       = dplyr::n(),
    n_obs_med     = median(n_ml, na.rm = TRUE),
    n_obs_min     = min(n_ml, na.rm = TRUE),
    n_sport_valid = sum(!is.na(KGE_SPoRT)),
    KGE_KGML_med  = median(KGE_KGML, na.rm = TRUE),
    KGE_SPoRT_med = median(KGE_SPoRT, na.rm = TRUE),
    r_KGML_med    = median(r_KGML, na.rm = TRUE),
    abs_pbias_KGML_med  = median(abs(pbias_KGML), na.rm = TRUE),
    abs_pbias_SPoRT_med = median(abs(pbias_SPoRT), na.rm = TRUE),
    .groups = "drop"
  ) |>
  dplyr::arrange(depth, dplyr::desc(n_sites))
readr::write_csv(skill_by_network_oos, glue::glue("{tables_dir}/skill_by_network_oos.csv"))

message(glue::glue("\nWrote OOS tables to {tables_dir}/ and figures to {figs_dir}/"))
