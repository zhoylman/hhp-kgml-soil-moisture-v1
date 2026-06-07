library(tidyverse)
library(sf)
library(glue)
library(terra)
library(furrr)
library(irr)

# ============================================================================
#  Drought-class error assessment (beta-standardized soil moisture).
#  Each series is standardized against ITS OWN moving day-of-year beta
#  climatology (<=30 yr, +/-15 day window, >=6 yr min) -> 11 USDM-style classes
#  (D4..W4). "Truth" = observed class; model class from the held-out-fold sims
#  (in-fold) or the 10-fold ensemble (OOS). We then score class agreement:
#  exact-class accuracy, mean |class error|, and squared-weighted Cohen's kappa.
#  Reuses standardize_doy_beta()/smi_to_class() from sm_eval_utils.R.
# ============================================================================

source("/home/zhoylman/hhp-kgml-soil-moisture-v1/R/sm_eval_utils.R")

# ---- Config ---------------------------------------------------------------
figs_dir   = "/home/zhoylman/hhp-kgml-soil-moisture-v1/figs"
tables_dir = "/home/zhoylman/hhp-kgml-soil-moisture-v1/tables"
cache_dir  = "/home/zhoylman/hhp-kgml-soil-moisture-v1/cache/climatology"
invisible(lapply(c(figs_dir, tables_dir, cache_dir), dir.create, showWarnings = FALSE, recursive = TRUE))

obs_dir   = "/data/ssd2/soil-moisture-ml/observations"
split_dir = "/data/ssd2/soil-moisture-ml/split-definitions-kfold"
clim_start = as.Date("1996-01-01")          # <= 30 yr of model rasters (data ends 2026)

roots = c("/data/ssd3/soil-moisture-ml-inference", "/data/ssd4/soil-moisture-ml-inference")
resolve_fold_dir = function(dep, fold) {
  cand = file.path(roots, glue("predictions-smoothed-daily-{dep}"), glue("fold_{fold}"))
  hit = cand[dir.exists(cand)]; if (length(hit)) hit[1] else cand[1]
}
resolve_ensemble_dir = function(dep) {
  cand = file.path(roots, glue("ensemble-smoothed-daily-{dep}"), "median")
  hit = cand[dir.exists(cand)]; if (length(hit)) hit[1] else cand[1]
}

site_meta = read_csv(file.path(obs_dir, "final-soil-moisture-data-generalized-meta.csv"),
                     show_col_types = FALSE) |>
  transmute(site_id = as.character(site_id), network, longitude, latitude) |>
  drop_na(longitude, latitude)

# ---- Per-site drought-class agreement -------------------------------------
# obs_s / mod_s: data.frames(date, value). Returns one row of metrics plus the
# matched (truth,pred) class pairs (for the pooled confusion matrix).
site_drought = function(obs_s, mod_s, site_id, depth, network) {
  to = standardize_doy_beta(obs_s); to$cls = smi_to_class(to$z); to$z2 = pmin(pmax(to$z, -2), 2)
  mo = standardize_doy_beta(mod_s); mo$cls = smi_to_class(mo$z); mo$z2 = pmin(pmax(mo$z, -2), 2)
  j = dplyr::inner_join(dplyr::transmute(to, date, truth = cls, truth_z = z2),
                        dplyr::transmute(mo, date, pred = cls, pred_z = z2), by = "date") |>
    tidyr::drop_na(truth, pred, truth_z, pred_z)
  if (nrow(j) < 30) {
    return(list(metrics = tibble(depth, network, site_id, n = nrow(j),
                                 exact_acc = NA_real_, class_err = NA_real_, kappa = NA_real_),
                pairs = tibble(depth = character(), date = as.Date(character()),
                               truth = integer(), pred = integer(),
                               truth_z = numeric(), pred_z = numeric())))
  }
  kap = tryCatch(irr::kappa2(data.frame(j$truth, j$pred), weight = "squared")$value,
                 error = function(e) NA_real_)
  list(
    metrics = tibble(depth, network, site_id, n = nrow(j),
                     exact_acc = mean(j$truth == j$pred),
                     class_err = mean(abs(j$truth - j$pred)),
                     kappa = kap),
    pairs = tibble(depth = depth, date = j$date, truth = j$truth, pred = j$pred,
                   truth_z = j$truth_z, pred_z = j$pred_z)
  )
}

# ---- One fold's worth of sites (in-fold; held-out fold sims) --------------
run_fold_drought = function(fold, depth_flag) {
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
    site_ids = intersect(unique(val$site_id), unique(obs$site_id))
    site_ids = intersect(site_ids, site_meta$site_id)
    if (!length(site_ids)) return(NULL)
    meta_xy = site_meta |> dplyr::filter(site_id %in% site_ids) |> dplyr::arrange(site_id)

    # full <=30yr model series at these sites from the HELD-OUT fold
    req_dates = seq(clim_start, as.Date("2027-01-01"), by = "day")
    model = extract_at_sites(
      raster_dir = resolve_fold_dir(dep, fold),
      obs_dates = req_dates, site_ids = meta_xy$site_id, meta_xy = meta_xy,
      cache_file = file.path(cache_dir, glue("{dep}_fold_{fold}_30yr.rds")),
      label = glue("clim fold {fold} [{dep}]")
    )
    if (!nrow(model)) return(NULL)

    res = lapply(meta_xy$site_id, function(s) {
      net = meta_xy$network[meta_xy$site_id == s][1]
      site_drought(dplyr::select(dplyr::filter(obs, site_id == s), date, value),
                   dplyr::transmute(dplyr::filter(model, site_id == s), date, value = ml),
                   s, depth_flag, net)
    })
    list(metrics = dplyr::bind_rows(lapply(res, `[[`, "metrics")),
         pairs   = dplyr::bind_rows(lapply(res, `[[`, "pairs")))
  }, error = function(e) { cli::cli_warn("fold {fold} [{depth_flag}] drought failed: {e$message}"); NULL })
}

# ---- Run in-fold, both depths (per fold parallel) -------------------------
# Quick-test hook: DROUGHT_TEST=1 runs only fold 1, Shallow.
.test   = nzchar(Sys.getenv("DROUGHT_TEST"))
.folds  = if (.test) 1L else 1:10
.depths = if (.test) "Shallow" else c("Shallow", "Middle")

run_depth = function(depth_flag) {
  plan(multisession, workers = 10)
  outs = future_map(.folds, ~ run_fold_drought(.x, depth_flag),
                    .options = furrr_options(
                      seed = NULL,
                      globals = c("run_fold_drought", "site_drought", "extract_at_sites",
                                  "standardize_doy_beta", "smi_to_class", "resolve_fold_dir",
                                  "site_meta", "clim_start", "cache_dir", "obs_dir",
                                  "split_dir", "roots", "depth_flag"),
                      packages = c("dplyr", "tidyr", "readr", "tibble", "stringr",
                                   "terra", "irr", "glue", "MASS", "cli", "rlang", "sf"))) |>
    purrr::compact()
  plan(sequential)
  list(metrics = bind_rows(lapply(outs, `[[`, "metrics")),
       pairs   = bind_rows(lapply(outs, `[[`, "pairs")))
}

infold = lapply(.depths, run_depth)
metrics_all = bind_rows(lapply(infold, `[[`, "metrics"))
pairs_all   = bind_rows(lapply(infold, `[[`, "pairs"))

if (nrow(metrics_all) == 0)
  stop("No drought metrics produced — all folds returned empty (check worker globals/extraction).")
cat(sprintf("Computed metrics for %d site-rows (%d with valid kappa).\n",
            nrow(metrics_all), sum(!is.na(metrics_all$kappa))))

readr::write_csv(metrics_all, file.path(tables_dir, "drought_class_per_site.csv"))

# ---- Aggregations ---------------------------------------------------------
umrb = site_meta |> filter(network == "UMRB Mesonet") |> pull(site_id)
agg = function(df, lbl) df |> group_by(depth) |>
  summarise(region = lbl, n_sites = sum(!is.na(exact_acc)),
            exact_acc_med = median(exact_acc, na.rm = TRUE),
            class_err_med = median(class_err, na.rm = TRUE),
            kappa_med     = median(kappa, na.rm = TRUE), .groups = "drop")
drought_summary = bind_rows(
  agg(metrics_all, "All sites"),
  agg(filter(metrics_all, site_id %in% umrb), "UMRB Mesonet")
) |> relocate(region, depth, n_sites)
readr::write_csv(drought_summary, file.path(tables_dir, "drought_class_summary.csv"))

drought_by_net = metrics_all |> group_by(depth, network) |>
  summarise(n_sites = sum(!is.na(exact_acc)),
            exact_acc_med = median(exact_acc, na.rm = TRUE),
            class_err_med = median(class_err, na.rm = TRUE),
            kappa_med     = median(kappa, na.rm = TRUE), .groups = "drop") |>
  arrange(depth, desc(n_sites))
readr::write_csv(drought_by_net, file.path(tables_dir, "drought_class_by_network.csv"))

labs11 = c("D4","D3","D2","D1","D0","Neutral","W0","W1","W2","W3","W4")

# ---- Table-3-style ordinal error per theoretical (truth) class ------------
# Matches Hoylman et al. (2024) Table 3 / Table S2: pooled over sites, grouped
# by the OBSERVED (truth) class. The error is on the standardized anomaly
# (SMI_mod - SMI_obs, both truncated +/-2) — NOT integer class indices. (The
# paper reports MSE < MAE for several classes, which is impossible for integer
# class differences; the magnitudes match SMI-space errors.) MSE = mean(err^2),
# MAE = mean(|err|). Reported for both all-non-frozen days (our choice) and
# May-Oct (the paper's warm season). NOTE: paper Table 3 is DEPTH-AVERAGED; we
# report per depth (shallow / middle) -> compare to per-depth Table S2.
ordinal_error = function(pairs, season_label) {
  pairs |>
    dplyr::mutate(truth_class = factor(labs11[truth], levels = labs11)) |>
    dplyr::group_by(depth, truth, truth_class) |>
    dplyr::summarise(MSE = mean((pred_z - truth_z)^2),
                     MAE = mean(abs(pred_z - truth_z)),
                     n   = dplyr::n(), .groups = "drop") |>
    dplyr::arrange(depth, truth) |>
    dplyr::mutate(season = season_label) |>
    dplyr::select(season, depth, truth_class, MSE, MAE, n)
}
ordinal_tbl = dplyr::bind_rows(
  ordinal_error(dplyr::filter(pairs_all, as.integer(format(date, "%m")) %in% 5:10), "May-Oct"),
  ordinal_error(pairs_all, "all non-frozen")
)
readr::write_csv(ordinal_tbl, file.path(tables_dir, "drought_ordinal_error_byclass.csv"))

# ---- Table-2-style anomaly-space metrics (SMI_mod vs SMI_obs) --------------
# Pearson r and RMSE on the standardized anomalies (SMI, truncated +/-2),
# pooled over sites, per depth. Matches Hoylman et al. (2024) Table 2, which
# DOES report per-depth rows (so KGML shallow/middle compare directly). Both
# seasons.
anomaly_metrics = function(pairs, season_label) {
  pairs |>
    dplyr::group_by(depth) |>
    dplyr::summarise(r = cor(pred_z, truth_z),
                     RMSE = sqrt(mean((pred_z - truth_z)^2)),
                     n = dplyr::n(), .groups = "drop") |>
    dplyr::mutate(season = season_label) |>
    dplyr::select(season, depth, r, RMSE, n)
}
anomaly_tbl = dplyr::bind_rows(
  anomaly_metrics(dplyr::filter(pairs_all, as.integer(format(date, "%m")) %in% 5:10), "May-Oct"),
  anomaly_metrics(pairs_all, "all non-frozen")
)
readr::write_csv(anomaly_tbl, file.path(tables_dir, "drought_anomaly_metrics.csv"))
cat("\n=== KGML anomaly-space r / RMSE (SMI), for comparison with paper Table 2 ===\n")
print(as.data.frame(dplyr::mutate(anomaly_tbl, r = round(r, 3), RMSE = round(RMSE, 3))))
cat("\n=== KGML ordinal class error (May-Oct), for comparison with paper Table 3 ===\n")
print(as.data.frame(dplyr::filter(ordinal_tbl, season == "May-Oct") |>
                      dplyr::mutate(MSE = round(MSE, 3), MAE = round(MAE, 3))))

# ---- Figure: pooled confusion matrix (truth vs predicted class) per depth --
conf = pairs_all |>
  count(depth, truth, pred) |>
  group_by(depth, truth) |> mutate(frac = n / sum(n)) |> ungroup() |>
  mutate(depth = factor(depth, levels = c("Shallow","Middle")),
         truth = factor(labs11[truth], levels = labs11),
         pred  = factor(labs11[pred],  levels = labs11))

p_conf = ggplot(conf, aes(pred, truth, fill = frac)) +
  geom_tile(color = "grey85") +
  geom_abline(slope = 1, intercept = 0, color = "black", linewidth = 0.4) +
  facet_wrap(~ depth) +
  scale_fill_viridis_c(name = "Row\nfraction", option = "D", limits = c(0, 1)) +
  labs(title = "Drought-class agreement: observed (truth) vs KGML (held-out fold)",
       subtitle = "Row-normalized; diagonal = exact-class match. 11 USDM-style classes.",
       x = "KGML class", y = "Observed class") +
  coord_equal() + theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold"),
        strip.text = element_text(face = "bold"))
ggsave(file.path(figs_dir, "drought_class_confusion.png"), p_conf, width = 12, height = 6, dpi = 300, bg = "white")

message(glue("\nWrote drought-class tables to {tables_dir}/ and confusion figure to {figs_dir}/"))
print(as.data.frame(drought_summary))
