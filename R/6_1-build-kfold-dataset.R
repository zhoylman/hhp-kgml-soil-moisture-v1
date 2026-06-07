library(tidyverse)
library(glue)
library(terra)
library(furrr)

# ============================================================================
#  BUILDER 1/2 — k-fold validation dataset (the official in-fold dataset).
#  Each site scored against ITS HELD-OUT FOLD's prediction (no leakage).
#  Emits:
#    tables/kfold_validation.csv  — per site x depth (KGE/r/pbias + SMI MAE)
#    cache/datasets/kfold_matched.rds — per site x day (obs, ml, obs_smi,
#      kgml_smi, sport_smi, truth_class) -> consumed by the drought-class &
#      detection scripts (no re-extraction needed).
#  Heavy step (raster extraction) is served from cache/climatology/ (built once
#  by the standardization pass). See memory: validation-dataset-decisions.
# ============================================================================

source("/home/zhoylman/hhp-kgml-soil-moisture-v1/R/sm_eval_utils.R")

repo       = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
data_root  = "/data/ssd2/soil-moisture-ml"
obs_dir    = glue("{data_root}/observations")
split_dir  = glue("{data_root}/split-definitions-kfold")
clim_dir   = glue("{repo}/cache/climatology")
out_ds     = glue("{repo}/cache/datasets"); dir.create(out_ds, showWarnings = FALSE, recursive = TRUE)
clim_start = as.Date("1996-01-01")
roots = c("/data/ssd3/soil-moisture-ml-inference", "/data/ssd4/soil-moisture-ml-inference")
sport_var = c(shallow = "SPoRT_raw_0-10cm", middle = "SPoRT_raw_10-40cm")
resolve_fold_dir = function(dep, fold) {
  cand = file.path(roots, glue("predictions-smoothed-daily-{dep}"), glue("fold_{fold}"))
  hit = cand[dir.exists(cand)]; if (length(hit)) hit[1] else cand[1]
}

site_meta = read_csv(glue("{obs_dir}/final-soil-moisture-data-generalized-meta.csv"), show_col_types = FALSE) |>
  transmute(site_id = as.character(site_id), network, longitude, latitude) |>
  distinct(site_id, .keep_all = TRUE)

# ---- per fold: matched (date, obs, ml, obs_smi, kgml_smi) for held-out sites ----
run_fold = function(fold, depth_flag) {
  tryCatch({
    dep = tolower(depth_flag)
    obs = read_csv(glue("{obs_dir}/final-soil-moisture-data-generalized-no-frozen.csv"),
                   show_col_types = FALSE,
                   col_types = cols(site_id = col_character(), date = col_date(),
                                    soil_moisture = col_double(), generalized_depth = col_character())) |>
      dplyr::filter(generalized_depth == depth_flag) |>
      dplyr::transmute(site_id, date, obs = soil_moisture)
    val = read_csv(glue("{split_dir}-{dep}/validation_split_fold_{fold}.csv"),
                   show_col_types = FALSE, col_types = cols(site_id = col_character()))
    sites = intersect(intersect(unique(val$site_id), unique(obs$site_id)), site_meta$site_id)
    if (!length(sites)) return(NULL)
    meta_xy = site_meta |> dplyr::filter(site_id %in% sites) |> dplyr::arrange(site_id)

    model = extract_at_sites(                                   # cache HIT (cache/climatology)
      raster_dir = resolve_fold_dir(dep, fold),
      obs_dates  = seq(clim_start, as.Date("2027-01-01"), by = "day"),
      site_ids = meta_xy$site_id, meta_xy = meta_xy,
      cache_file = file.path(clim_dir, glue("{dep}_fold_{fold}_30yr.rds")),
      label = glue("kfold {fold} [{dep}]"))
    if (!nrow(model)) return(NULL)

    dplyr::bind_rows(lapply(meta_xy$site_id, function(s) {
      o  = dplyr::filter(obs, site_id == s) |> dplyr::select(date, obs)
      es = dplyr::filter(model, site_id == s) |> dplyr::select(date, ml)
      mt = dplyr::inner_join(o, es, by = "date") |> tidyr::drop_na(obs, ml)
      if (nrow(mt) < 5) return(NULL)
      osm = standardize_doy_beta(dplyr::transmute(o, date, value = obs)) |>
        dplyr::transmute(date, obs_smi = pmin(pmax(z, -2), 2))
      msm = standardize_doy_beta(dplyr::transmute(es, date, value = ml)) |>
        dplyr::transmute(date, kgml_smi = pmin(pmax(z, -2), 2))
      mt |>
        dplyr::left_join(osm, by = "date") |>
        dplyr::left_join(msm, by = "date") |>
        dplyr::mutate(network = meta_xy$network[meta_xy$site_id == s][1], site_id = s,
                      depth = depth_flag, .before = 1)
    }))
  }, error = function(e) { cli::cli_warn("fold {fold} [{depth_flag}] failed: {e$message}"); NULL })
}

run_depth = function(depth_flag) {
  plan(multisession, workers = 10)
  outs = future_map(1:10, ~ run_fold(.x, depth_flag),
                    .options = furrr_options(
                      seed = NULL,
                      globals = c("run_fold", "extract_at_sites", "standardize_doy_beta",
                                  "resolve_fold_dir", "site_meta", "clim_start", "clim_dir",
                                  "obs_dir", "split_dir", "roots", "depth_flag"),
                      packages = c("dplyr", "tidyr", "readr", "tibble", "stringr",
                                   "terra", "glue", "MASS", "cli", "rlang"))) |> purrr::compact()
  plan(sequential)
  bind_rows(outs)
}

matched = bind_rows(lapply(c("Shallow", "Middle"), run_depth))

# ---- SPoRT SMI per site (standardize the SPoRT sims) + truth class ----
sport_smi = bind_rows(lapply(c("Shallow", "Middle"), function(depth_flag) {
  dep = tolower(depth_flag)
  sims = read_csv(glue("{obs_dir}/observational-sites-raw-sport.csv"), show_col_types = FALSE) |>
    mutate(date = as.Date(time)) |> filter(var == sport_var[[dep]]) |>
    pivot_longer(cols = -c(var, date, time), names_to = "site_id", values_to = "sport") |>
    filter(site_id %in% unique(matched$site_id[matched$depth == depth_flag])) |> drop_na(sport)
  bind_rows(lapply(unique(sims$site_id), function(s) {
    standardize_doy_beta(filter(sims, site_id == s) |> transmute(date, value = sport)) |>
      transmute(site_id = s, depth = depth_flag, date, sport_smi = pmin(pmax(z, -2), 2))
  }))
}))

matched = matched |>
  left_join(sport_smi, by = c("site_id", "depth", "date")) |>
  mutate(truth_class = smi_to_class(obs_smi))

saveRDS(matched, file.path(out_ds, "kfold_matched.rds"))

# ---- SPoRT-LIS raw per-site metrics (fair depth: sport-specific 10-40 obs) ----
sport_raw = bind_rows(lapply(c("Shallow", "Middle"), function(depth_flag) {
  dep = tolower(depth_flag)
  obs_sp = read_csv(glue("{obs_dir}/final-soil-moisture-data-generalized-sport-specific-no-frozen.csv"),
                    show_col_types = FALSE) |>
    filter(generalized_depth == depth_flag) |> transmute(site_id = as.character(site_id), date, obs = soil_moisture)
  sims = read_csv(glue("{obs_dir}/observational-sites-raw-sport.csv"), show_col_types = FALSE) |>
    mutate(date = as.Date(time)) |> filter(var == sport_var[[dep]]) |>
    pivot_longer(cols = -c(var, date, time), names_to = "site_id", values_to = "sport")
  obs_sp |> inner_join(sims |> select(site_id, date, sport), by = c("site_id", "date")) |> drop_na(obs, sport) |>
    group_by(site_id) |>
    summarise(KGE_sport = hydroGOF::KGE(sport, obs), r_sport = cor(sport, obs),
              pbias_sport = hydroGOF::pbias(sport, obs), .groups = "drop") |>
    mutate(depth = depth_flag)
}))

# ---- per-site CSV (the publishable dataset) ----
kfold_validation = matched |>
  group_by(network, site_id, depth) |>
  summarise(n_obs = sum(!is.na(obs) & !is.na(ml)),
            KGE = hydroGOF::KGE(ml, obs), r = cor(ml, obs, use = "complete.obs"),
            pbias = hydroGOF::pbias(ml, obs),
            n_smi = sum(!is.na(obs_smi) & !is.na(kgml_smi)),
            smi_mae = if (sum(!is.na(obs_smi) & !is.na(kgml_smi)) >= 30)
                        mean(abs(kgml_smi - obs_smi), na.rm = TRUE) else NA_real_,
            .groups = "drop") |>
  left_join(sport_raw, by = c("site_id", "depth")) |>
  left_join(site_meta |> select(site_id, longitude, latitude), by = "site_id") |>
  mutate(robust = n_obs >= 365) |>
  select(network, site_id, depth, longitude, latitude, n_obs, robust,
         KGE, r, pbias, KGE_sport, r_sport, pbias_sport, smi_mae, n_smi) |>
  arrange(depth, network, site_id)

readr::write_csv(kfold_validation, glue("{repo}/tables/kfold_validation.csv"))

cat(glue("\nWrote tables/kfold_validation.csv ({nrow(kfold_validation)} rows) + cache/datasets/kfold_matched.rds ({nrow(matched)} rows)\n"))
cat("\n=== k-fold skill (robust >=365d) by depth ===\n")
print(as.data.frame(kfold_validation |> filter(robust) |> group_by(depth) |>
  summarise(n = n(), KGE = round(median(KGE, na.rm = TRUE), 3), r = round(median(r, na.rm = TRUE), 3),
            abs_pbias = round(median(abs(pbias), na.rm = TRUE), 1),
            smi_mae = round(median(smi_mae, na.rm = TRUE), 3), .groups = "drop")))
