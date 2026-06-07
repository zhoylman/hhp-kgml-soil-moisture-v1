library(tidyverse)
library(glue)
library(terra)
library(furrr)

# ============================================================================
#  UNIFIED k-fold validation -> standalone kfold_validation.csv (the official
#  in-fold validation dataset). Each site is scored against ITS HELD-OUT FOLD's
#  prediction (no leakage; mirrors 6_1). Same per-site schema as the OOS CSV:
#  network, site_id, depth, lon/lat, n_obs, robust, KGE, r, pbias, smi_mae.
#  Reuses the cached 30-yr held-out-fold series in cache/climatology/ (instant),
#  so this just re-derives obs-day skill + the beta-SMI drought-class MAE.
# ============================================================================

source("/home/zhoylman/hhp-kgml-soil-moisture-v1/R/sm_eval_utils.R")

repo       = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
data_root  = "/data/ssd2/soil-moisture-ml"
obs_dir    = glue("{data_root}/observations")
split_dir  = glue("{data_root}/split-definitions-kfold")
cache_dir  = glue("{repo}/cache/climatology")
clim_start = as.Date("1996-01-01")
roots = c("/data/ssd3/soil-moisture-ml-inference", "/data/ssd4/soil-moisture-ml-inference")
resolve_fold_dir = function(dep, fold) {
  cand = file.path(roots, glue("predictions-smoothed-daily-{dep}"), glue("fold_{fold}"))
  hit = cand[dir.exists(cand)]; if (length(hit)) hit[1] else cand[1]
}

site_meta = read_csv(glue("{obs_dir}/final-soil-moisture-data-generalized-meta.csv"), show_col_types = FALSE) |>
  transmute(site_id = as.character(site_id), network, longitude, latitude) |>
  distinct(site_id, .keep_all = TRUE)

run_fold = function(fold, depth_flag) {
  tryCatch({
    dep = tolower(depth_flag)
    obs = read_csv(glue("{obs_dir}/final-soil-moisture-data-generalized-no-frozen.csv"),
                   show_col_types = FALSE,
                   col_types = cols(site_id = col_character(), date = col_date(),
                                    soil_moisture = col_double(), generalized_depth = col_character())) |>
      dplyr::filter(generalized_depth == depth_flag) |>
      dplyr::transmute(site_id, date, value = soil_moisture)
    val = read_csv(glue("{split_dir}-{dep}/validation_split_fold_{fold}.csv"),
                   show_col_types = FALSE, col_types = cols(site_id = col_character()))
    sites = intersect(intersect(unique(val$site_id), unique(obs$site_id)), site_meta$site_id)
    if (!length(sites)) return(NULL)
    meta_xy = site_meta |> dplyr::filter(site_id %in% sites) |> dplyr::arrange(site_id)

    model = extract_at_sites(                                   # cache HIT (from 6_6)
      raster_dir = resolve_fold_dir(dep, fold),
      obs_dates  = seq(clim_start, as.Date("2027-01-01"), by = "day"),
      site_ids = meta_xy$site_id, meta_xy = meta_xy,
      cache_file = file.path(cache_dir, glue("{dep}_fold_{fold}_30yr.rds")),
      label = glue("kfold {fold} [{dep}]"))
    if (!nrow(model)) return(NULL)

    dplyr::bind_rows(lapply(meta_xy$site_id, function(s) {
      o = dplyr::filter(obs, site_id == s) |> dplyr::select(date, obs = value)
      es = dplyr::filter(model, site_id == s) |> dplyr::select(date, ml)
      j = dplyr::inner_join(o, es, by = "date") |> tidyr::drop_na(obs, ml)
      if (nrow(j) < 5) return(NULL)
      m = compute_metrics(j, site_id = s)
      so = standardize_doy_beta(dplyr::transmute(o, date, value = obs)) |>
        dplyr::transmute(date, smi_obs = pmin(pmax(z, -2), 2))
      sm = standardize_doy_beta(dplyr::transmute(es, date, value = ml)) |>
        dplyr::transmute(date, smi_mod = pmin(pmax(z, -2), 2))
      sj = dplyr::inner_join(so, sm, by = "date") |> tidyr::drop_na()
      net = meta_xy$network[meta_xy$site_id == s][1]
      tibble(network = net, site_id = s, depth = depth_flag, n_obs = m$n,
             KGE = m$KGE, r = m$r, pbias = m$pbias,
             smi_mae = if (nrow(sj) >= 30) mean(abs(sj$smi_mod - sj$smi_obs)) else NA_real_,
             n_smi = nrow(sj))
    }))
  }, error = function(e) { cli::cli_warn("fold {fold} [{depth_flag}] failed: {e$message}"); NULL })
}

run_depth = function(depth_flag) {
  plan(multisession, workers = 10)
  outs = future_map(1:10, ~ run_fold(.x, depth_flag),
                    .options = furrr_options(
                      seed = NULL,
                      globals = c("run_fold", "extract_at_sites", "standardize_doy_beta", "compute_metrics",
                                  "resolve_fold_dir", "site_meta", "clim_start", "cache_dir",
                                  "obs_dir", "split_dir", "roots", "depth_flag"),
                      packages = c("dplyr", "tidyr", "readr", "tibble", "stringr",
                                   "terra", "glue", "MASS", "cli", "rlang", "hydroGOF"))) |> purrr::compact()
  plan(sequential)
  bind_rows(outs)
}

kfold_validation = bind_rows(lapply(c("Shallow", "Middle"), run_depth)) |>
  left_join(site_meta |> select(site_id, longitude, latitude), by = "site_id") |>
  mutate(robust = n_obs >= 365) |>
  select(network, site_id, depth, longitude, latitude, n_obs, robust, KGE, r, pbias, smi_mae, n_smi) |>
  arrange(depth, network, site_id)

readr::write_csv(kfold_validation, glue("{repo}/tables/kfold_validation.csv"))

cat(glue("\nWrote tables/kfold_validation.csv : {nrow(kfold_validation)} site x depth rows\n"))
cat("\n=== k-fold skill by network x depth (median) ===\n")
print(as.data.frame(kfold_validation |> group_by(depth, network) |>
  summarise(n = n(), robust = sum(robust),
            KGE = round(median(KGE, na.rm = TRUE), 3), r = round(median(r, na.rm = TRUE), 3),
            abs_pbias = round(median(abs(pbias), na.rm = TRUE), 1),
            smi_mae = round(median(smi_mae, na.rm = TRUE), 3), .groups = "drop")))
