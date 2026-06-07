library(tidyverse)
library(sf)
library(glue)
library(terra)
library(furrr)

# ============================================================================
#  Drought DETECTION skill: KGML (held-out fold) vs SPoRT-LIS.
#  Both standardized to SMI against their own <=30-yr moving-DOY beta
#  climatology (same as 6_6). An observed drought EVENT is SMI_obs below a
#  threshold (D0 <= -0.5, D1 <= -0.8, D2 <= -1.3). For each model we build the
#  2x2 contingency vs observed events on the MATCHED sample (sites/days where
#  obs, KGML and SPoRT all exist -> 2005-2022, the SPoRT sim period) and report
#  POD, FAR, CSI, HSS, and frequency bias. Per depth, both seasons.
# ============================================================================

source("/home/zhoylman/hhp-kgml-soil-moisture-v1/R/sm_eval_utils.R")

tables_dir = "/home/zhoylman/hhp-kgml-soil-moisture-v1/tables"
cache_dir  = "/home/zhoylman/hhp-kgml-soil-moisture-v1/cache/climatology"
obs_dir    = "/data/ssd2/soil-moisture-ml/observations"
split_dir  = "/data/ssd2/soil-moisture-ml/split-definitions-kfold"
clim_start = as.Date("1996-01-01")

roots = c("/data/ssd3/soil-moisture-ml-inference", "/data/ssd4/soil-moisture-ml-inference")
resolve_fold_dir = function(dep, fold) {
  cand = file.path(roots, glue("predictions-smoothed-daily-{dep}"), glue("fold_{fold}"))
  hit = cand[dir.exists(cand)]; if (length(hit)) hit[1] else cand[1]
}
sport_var = c(shallow = "SPoRT_raw_0-10cm", middle = "SPoRT_raw_10-40cm")
THRESH = c(D0 = -0.5, D1 = -0.8, D2 = -1.3)   # observed-SMI drought-event thresholds

site_meta = read_csv(file.path(obs_dir, "final-soil-moisture-data-generalized-meta.csv"),
                     show_col_types = FALSE) |>
  transmute(site_id = as.character(site_id), network)

# 2x2 detection metrics from binary truth/pred event vectors
det_metrics = function(truth_ev, pred_ev) {
  # as.numeric to avoid integer overflow in H*C (counts reach ~1e6)
  H = as.numeric(sum(truth_ev & pred_ev)); M = as.numeric(sum(truth_ev & !pred_ev))
  Fa = as.numeric(sum(!truth_ev & pred_ev)); C = as.numeric(sum(!truth_ev & !pred_ev))
  pod = H / (H + M); far = if ((H + Fa) > 0) Fa / (H + Fa) else NA_real_
  csi = if ((H + M + Fa) > 0) H / (H + M + Fa) else NA_real_
  den = (H + M) * (M + C) + (H + Fa) * (Fa + C)
  hss = if (den > 0) 2 * (H * C - M * Fa) / den else NA_real_
  tibble(n = H + M + Fa + C, n_event = H + M,
         POD = pod, FAR = far, CSI = csi, HSS = hss, freq_bias = (H + Fa) / (H + M))
}

# ---- KGML(held-out fold) + obs SMI per fold's sites -> (site_id,date,obs_smi,kgml_smi)
run_fold = function(fold, depth_flag) {
  tryCatch({
    dep = tolower(depth_flag)
    obs = read_csv(file.path(obs_dir, "final-soil-moisture-data-generalized-no-frozen.csv"),
                   show_col_types = FALSE,
                   col_types = cols(site_id = col_character(), date = col_date(),
                                    soil_moisture = col_double(), generalized_depth = col_character())) |>
      dplyr::filter(generalized_depth == depth_flag) |>
      dplyr::transmute(site_id, date, value = soil_moisture)
    val = read_csv(glue("{split_dir}-{dep}/validation_split_fold_{fold}.csv"),
                   show_col_types = FALSE, col_types = cols(site_id = col_character()))
    site_ids = intersect(intersect(unique(val$site_id), unique(obs$site_id)), site_meta$site_id)
    if (!length(site_ids)) return(NULL)
    meta_xy = read_csv(file.path(obs_dir, "final-soil-moisture-data-generalized-meta.csv"), show_col_types = FALSE) |>
      transmute(site_id = as.character(site_id), longitude, latitude) |>
      dplyr::filter(site_id %in% site_ids) |> dplyr::arrange(site_id)

    model = extract_at_sites(
      raster_dir = resolve_fold_dir(dep, fold),
      obs_dates = seq(clim_start, as.Date("2027-01-01"), by = "day"),
      site_ids = meta_xy$site_id, meta_xy = meta_xy,
      cache_file = file.path(cache_dir, glue("{dep}_fold_{fold}_30yr.rds")),
      label = glue("detect fold {fold} [{dep}]"))
    if (!nrow(model)) return(NULL)

    dplyr::bind_rows(lapply(meta_xy$site_id, function(s) {
      o = standardize_doy_beta(dplyr::select(dplyr::filter(obs, site_id == s), date, value))
      m = standardize_doy_beta(dplyr::transmute(dplyr::filter(model, site_id == s), date, value = ml))
      dplyr::inner_join(dplyr::transmute(o, date, obs_smi = pmin(pmax(z, -2), 2)),
                        dplyr::transmute(m, date, kgml_smi = pmin(pmax(z, -2), 2)), by = "date") |>
        tidyr::drop_na() |> dplyr::mutate(site_id = s, .before = 1)
    }))
  }, error = function(e) { cli::cli_warn("fold {fold} [{depth_flag}] detect failed: {e$message}"); NULL })
}

run_depth_kgml = function(depth_flag) {
  plan(multisession, workers = 10)
  outs = future_map(1:10, ~ run_fold(.x, depth_flag),
                    .options = furrr_options(
                      seed = NULL,
                      globals = c("run_fold", "extract_at_sites", "standardize_doy_beta",
                                  "resolve_fold_dir", "site_meta", "clim_start", "cache_dir",
                                  "obs_dir", "split_dir", "roots", "depth_flag"),
                      packages = c("dplyr", "tidyr", "readr", "tibble", "stringr",
                                   "terra", "glue", "MASS", "cli", "rlang"))) |> purrr::compact()
  plan(sequential)
  dplyr::bind_rows(outs)
}

# ---- SPoRT SMI per site for a depth (standardize the sims) ----------------
sport_smi_depth = function(depth_flag, keep_sites) {
  dep = tolower(depth_flag)
  sims = read_csv(file.path(obs_dir, "observational-sites-raw-sport.csv"), show_col_types = FALSE) |>
    dplyr::mutate(date = as.Date(time)) |>
    dplyr::filter(var == sport_var[[dep]]) |>
    tidyr::pivot_longer(cols = -c(var, date, time), names_to = "site_id", values_to = "value") |>
    dplyr::filter(site_id %in% keep_sites) |>
    tidyr::drop_na(value)
  dplyr::bind_rows(lapply(unique(sims$site_id), function(s) {
    standardize_doy_beta(dplyr::select(dplyr::filter(sims, site_id == s), date, value)) |>
      dplyr::transmute(site_id = s, date, sport_smi = pmin(pmax(z, -2), 2))
  })) |> tidyr::drop_na(sport_smi)
}

# ---- Run both depths ------------------------------------------------------
detect_depth = function(depth_flag) {
  # Cache the matched obs/KGML/SPoRT SMI so metric tweaks don't redo the ~10-min
  # standardization. Delete cache/detect_matched_*.rds to force recompute.
  mcache = file.path(cache_dir, glue("detect_matched_{tolower(depth_flag)}.rds"))
  if (file.exists(mcache)) {
    m = readRDS(mcache)
  } else {
    km = run_depth_kgml(depth_flag)                                # site_id,date,obs_smi,kgml_smi
    sp = sport_smi_depth(depth_flag, unique(km$site_id))           # site_id,date,sport_smi
    m  = dplyr::inner_join(km, sp, by = c("site_id", "date")) |> tidyr::drop_na()
    saveRDS(m, mcache)
  }
  m$month = as.integer(format(m$date, "%m"))

  one = function(df, season_label) {
    dplyr::bind_rows(lapply(names(THRESH), function(tn) {
      thr = THRESH[[tn]]; ev = df$obs_smi < thr
      dplyr::bind_rows(
        det_metrics(ev, df$kgml_smi  < thr) |> dplyr::mutate(model = "KGML"),
        det_metrics(ev, df$sport_smi < thr) |> dplyr::mutate(model = "SPoRT-LIS")
      ) |> dplyr::mutate(threshold = tn, smi_thr = thr)
    })) |> dplyr::mutate(season = season_label, depth = depth_flag)
  }
  dplyr::bind_rows(one(m, "all non-frozen"),
                   one(dplyr::filter(m, month %in% 5:10), "May-Oct"))
}

detection = dplyr::bind_rows(lapply(c("Shallow", "Middle"), detect_depth)) |>
  dplyr::select(season, depth, threshold, smi_thr, model, n, n_event, POD, FAR, CSI, HSS, freq_bias)
readr::write_csv(detection, file.path(tables_dir, "drought_detection_kgml_vs_sport.csv"))

# ---- Per-class MAE (Table-3 definition: |SMI_mod - SMI_obs| by truth class) -
# KGML vs SPoRT on the SAME matched sample, all 11 classes, both seasons.
labs11 = c("D4","D3","D2","D1","D0","Neutral","W0","W1","W2","W3","W4")
class_mae = function(m, season_label, depth_flag) {
  m |>
    dplyr::mutate(truth = smi_to_class(obs_smi)) |>
    dplyr::group_by(truth) |>
    dplyr::summarise(MAE_KGML  = mean(abs(kgml_smi  - obs_smi)),
                     MAE_SPoRT = mean(abs(sport_smi - obs_smi)),
                     n = dplyr::n(), .groups = "drop") |>
    dplyr::transmute(season = season_label, depth = depth_flag,
                     class = factor(labs11[truth], levels = labs11),
                     MAE_KGML, MAE_SPoRT, n) |>
    dplyr::arrange(class)
}
class_mae_tbl = dplyr::bind_rows(lapply(c("Shallow", "Middle"), function(dep) {
  m = readRDS(file.path(cache_dir, glue::glue("detect_matched_{tolower(dep)}.rds")))
  m$month = as.integer(format(m$date, "%m"))
  dplyr::bind_rows(class_mae(dplyr::filter(m, month %in% 5:10), "May-Oct", dep),
                   class_mae(m, "all non-frozen", dep))
}))
readr::write_csv(class_mae_tbl, file.path(tables_dir, "drought_class_mae_kgml_vs_sport.csv"))

cat("\n=== Drought detection skill: KGML vs SPoRT-LIS (May-Oct, matched sample) ===\n")
print(as.data.frame(detection |> dplyr::filter(season == "May-Oct") |>
                      dplyr::mutate(dplyr::across(c(POD, FAR, CSI, HSS, freq_bias), \(x) round(x, 3)))))
