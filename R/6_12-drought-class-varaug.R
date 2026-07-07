##############################################################
# Title: Drought-class skill of the VARIANCE-AUGMENTED SMI (ops handoff check)
# Description:
#   Re-scores the drought classification table (cf. 6_4 / drought_class_mae
#   figure) with the ops variance-augmented SMI (ANOMALY-METHOD-variance-
#   augmented.md), at the k-fold validation stations, MIDDLE depth.
#
#   Three KGML variants + two SPoRT variants vs the same obs truth classes:
#     kgml_smi      — existing research SMI (±15d pooled, Beta MLE)   [from 6_1 rds]
#     kgml_smi_mom  — ops formulation WITHOUT augmentation (isolates MoM effect)
#     kgml_smi_aug  — ops formulation WITH v_eff = sigma2_clim + sigma2_obs
#     sport_smi     — existing pooled-MLE SPoRT SMI                    [from 6_1 rds]
#     sport_smi_mom — SPoRT under the SAME ops MoM formulation (apples-to-apples
#                     fit method). NOTE: no augmented SPoRT exists — sigma2_obs is
#                     cross-fold ensemble variance and SPoRT is a single run; the
#                     fair pairs are raw<->raw and MoM<->MoM (aug vs MoM shows the
#                     ensemble-only capability).
#   sigma2_obs(site, day) = sample variance across the 10 fold predictions,
#   extracted from the year-frozen per-fold smoothed dailies.
#
#   beta_fit_smi_ops mirrors ops R/3_3-finalize.R VERBATIM in math:
#   same-calendar-day trailing-30 climatology (current value LAST/included),
#   MoM with t = max(m(1-m)/v - 1, 2), v += sigma_obs2, cap +/-3.09.
#
#   Run: Rscript R/6_12-drought-class-varaug.R   (middle only; ~1h extraction)
##############################################################

suppressPackageStartupMessages({ library(tidyverse); library(glue); library(furrr) })
source("/home/zhoylman/hhp-kgml-soil-moisture-v1/R/sm_eval_utils.R")

repo      = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
data_root = "/data/ssd2/soil-moisture-ml"
clim_dir  = glue("{repo}/cache/climatology")
roots     = c("/data/ssd3/soil-moisture-ml-inference", "/data/ssd4/soil-moisture-ml-inference")
# DEPTH_FLAG middle|shallow; ARCH_SUFFIX "-yearfrozen" (frozen archive) or "" (as-is;
# used as the shallow PLACEHOLDER until its frozen regen completes)
dep       = tolower(Sys.getenv("DEPTH_FLAG", "middle"))
depth_flag = c(middle = "Middle", shallow = "Shallow")[[dep]]
arch_suffix = Sys.getenv("ARCH_SUFFIX", "-yearfrozen")

matched = readRDS(glue("{repo}/cache/datasets/kfold_matched{arch_suffix}.rds")) |>
  filter(depth == depth_flag)
site_meta = read_csv(glue("{data_root}/observations/final-soil-moisture-data-generalized-meta.csv"),
                     show_col_types = FALSE) |>
  transmute(site_id = as.character(site_id), longitude, latitude) |>
  distinct(site_id, .keep_all = TRUE) |>
  filter(site_id %in% unique(matched$site_id)) |>
  arrange(site_id)
message(glue("matched rows: {nrow(matched)} | sites: {nrow(site_meta)}"))

# ---- 1. ALL-FOLD extraction at ALL validation sites (for sigma2_obs) --------
fold_dir = function(fold) {
  cand = file.path(roots, glue("predictions-smoothed-daily-{dep}{arch_suffix}"), glue("fold_{fold}"))
  hit = cand[dir.exists(cand)]; if (length(hit)) hit[1] else cand[1]
}
plan(multisession, workers = 10)
fold_series = future_map(1:10, function(f) {
  extract_at_sites(
    raster_dir = fold_dir(f),
    obs_dates  = seq(as.Date("1996-01-01"), as.Date("2027-01-01"), by = "day"),
    site_ids   = site_meta$site_id, meta_xy = site_meta,
    cache_file = file.path(clim_dir, glue("{dep}{arch_suffix}_fold_{f}_ALLSITES_30yr.rds")),
    label = glue("allsites fold {f}")) |>
    mutate(fold = f)
}, .options = furrr_options(seed = NULL,
     globals = c("extract_at_sites", "fold_dir", "site_meta", "clim_dir", "dep", "arch_suffix", "roots"),
     packages = c("dplyr", "tidyr", "readr", "tibble", "stringr", "terra", "glue", "cli", "rlang")))
plan(sequential)

sigma2 = bind_rows(fold_series) |>
  group_by(site_id, date) |>
  summarise(sigma2_obs = stats::var(ml, na.rm = TRUE),
            n_folds = sum(!is.na(ml)), .groups = "drop") |>
  filter(n_folds >= 8)
rm(fold_series); gc()
message(glue("sigma2_obs rows: {nrow(sigma2)} | median sigma_obs: ",
             "{round(sqrt(median(sigma2$sigma2_obs, na.rm=TRUE)), 4)} m3/m3"))

# ---- 2. ops beta_fit_smi (verbatim math), vectorized per site ----------------
# For each site: daily held-out series ml(d). For date d: clim = same-mm-dd
# values for years <= year(d) (trailing 30, current LAST). MoM fit on
# (mean, var [+ sigma2_obs]); z = qnorm(pbeta(cur)); cap +/-3.09.
smi_ops_site = function(df) {   # df: date, ml, sigma2_obs (full daily series, one site)
  df = df |> arrange(date) |>
    mutate(mmdd = format(date, "%m-%d"), yr = as.integer(format(date, "%Y")))
  out = df |> group_by(mmdd) |> group_modify(function(g, key) {
    g = g |> arrange(yr)
    v_ml = pmin(pmax(g$ml, 1e-6), 1 - 1e-6)
    n = nrow(g)
    z_mom = z_aug = rep(NA_real_, n)
    for (i in seq_len(n)) {
      lo = max(1L, i - 29L)                     # trailing 30 incl current
      clim = v_ml[lo:i]; clim = clim[is.finite(clim)]
      if (length(clim) < 3L || length(unique(clim)) < 3L) next
      cur = v_ml[i]; m = mean(clim); v = stats::var(clim)
      if (!is.finite(v) || v <= 0) next
      mom_z = function(vv) {
        t = max(m * (1 - m) / vv - 1, 2)
        cdf = pbeta(min(max(cur, 1e-6), 1 - 1e-6), m * t, (1 - m) * t)
        min(max(qnorm(min(max(cdf, 1e-12), 1 - 1e-12)), -3.09), 3.09)
      }
      z_mom[i] = mom_z(v)
      s2 = g$sigma2_obs[i]
      z_aug[i] = if (is.finite(s2) && s2 > 0) mom_z(v + s2) else z_mom[i]
    }
    g |> mutate(kgml_smi_mom = z_mom, kgml_smi_aug = z_aug)
  }) |> ungroup()
  out |> select(date, kgml_smi_mom, kgml_smi_aug)
}

# full daily held-out series per site = union of the 6_1 per-fold caches
heldout = map_dfr(1:10, function(f) {
  ca = file.path(clim_dir, glue("{dep}{arch_suffix}_fold_{f}_30yr.rds"))
  if (!file.exists(ca)) return(tibble())
  readRDS(ca)$smoothed_data
}) |> distinct(site_id, date, .keep_all = TRUE) |>
  left_join(sigma2 |> select(site_id, date, sigma2_obs), by = c("site_id", "date"))

plan(multisession, workers = 24)
smi_new = heldout |>
  group_by(site_id) |> group_split() |>
  future_map(\(df) smi_ops_site(df) |> mutate(site_id = df$site_id[1]),
             .options = furrr_options(seed = NULL, globals = c("smi_ops_site"),
                                      packages = c("dplyr", "tidyr", "tibble"))) |>
  bind_rows()
plan(sequential)

# ---- 3. SPoRT under the same ops MoM formulation (apples-to-apples fit) -----
sport_var_dep = c(middle = "SPoRT_raw_10-40cm", shallow = "SPoRT_raw_0-10cm")[[dep]]
sport_sims = read_csv(glue("{data_root}/observations/observational-sites-raw-sport.csv"),
                      show_col_types = FALSE) |>
  mutate(date = as.Date(time)) |> filter(var == sport_var_dep) |>
  pivot_longer(cols = -c(var, date, time), names_to = "site_id", values_to = "ml") |>
  filter(site_id %in% site_meta$site_id) |> drop_na(ml) |>
  mutate(sigma2_obs = NA_real_)          # single run: no ensemble variance -> aug == mom

plan(multisession, workers = 24)
sport_mom = sport_sims |>
  group_by(site_id) |> group_split() |>
  future_map(\(df) smi_ops_site(df) |> mutate(site_id = df$site_id[1]),
             .options = furrr_options(seed = NULL, globals = c("smi_ops_site"),
                                      packages = c("dplyr", "tidyr", "tibble"))) |>
  bind_rows() |>
  select(site_id, date, sport_smi_mom = kgml_smi_mom)
plan(sequential)

# ---- 4. classes + per-class MAE table ----------------------------------------
scored = matched |>
  left_join(smi_new, by = c("site_id", "date")) |>
  left_join(sport_mom, by = c("site_id", "date")) |>
  mutate(cls_kgml = smi_to_class(kgml_smi),
         cls_mom  = smi_to_class(kgml_smi_mom),
         cls_aug  = smi_to_class(kgml_smi_aug),
         cls_sport = smi_to_class(sport_smi),
         cls_sport_mom = smi_to_class(sport_smi_mom)) |>
  filter(!is.na(truth_class))

saveRDS(scored, glue("{repo}/cache/datasets/kfold_matched_varaug_{dep}.rds"))

# 6_5-style manuscript metric: per-truth-class MAE on the STANDARDIZED ANOMALY
# (mean |SMI_model - SMI_obs|), robust sites, May-Oct, identical rows across all
# five variants (apples-to-apples).
class_lbls = c("D4","D3","D2","D1","D0","Neutral","W0","W1","W2","W3","W4")
robust = read_csv(glue("{repo}/tables/kfold_validation{arch_suffix}.csv"), show_col_types = FALSE) |>
  filter(robust, depth == depth_flag) |> distinct(site_id, depth)
tab = scored |>
  semi_join(robust, by = c("site_id", "depth")) |>
  mutate(month = as.integer(format(date, "%m"))) |> filter(month %in% 5:10) |>
  filter(!is.na(obs_smi), !is.na(kgml_smi), !is.na(kgml_smi_mom), !is.na(kgml_smi_aug),
         !is.na(sport_smi), !is.na(sport_smi_mom)) |>
  group_by(truth_class) |>
  summarise(n = n(),
            KGML_raw  = mean(abs(kgml_smi     - obs_smi)),
            KGML_mom  = mean(abs(kgml_smi_mom - obs_smi)),
            KGML_aug  = mean(abs(kgml_smi_aug - obs_smi)),
            SPoRT_raw = mean(abs(sport_smi    - obs_smi)),
            SPoRT_mom = mean(abs(sport_smi_mom - obs_smi)), .groups = "drop") |>
  mutate(class = factor(class_lbls[truth_class], levels = class_lbls), depth = depth_flag) |>
  arrange(class)
write_csv(tab, glue("{repo}/tables/drought_class_mae_varaug_{dep}.csv"))

cat(glue("\n=== {toupper(dep)}{arch_suffix} per-class SMI MAE (May-Oct, robust, identical rows) ===\n"))
print(as.data.frame(tab |> mutate(across(where(is.numeric) & !n, ~round(.x, 3))) |>
  select(class, n, KGML_raw, KGML_mom, KGML_aug, SPoRT_raw, SPoRT_mom)))
cat(glue("\nOverall: KGML raw {round(weighted.mean(tab$KGML_raw, tab$n),3)}",
         " | mom {round(weighted.mean(tab$KGML_mom, tab$n),3)}",
         " | aug {round(weighted.mean(tab$KGML_aug, tab$n),3)}",
         " || SPoRT raw {round(weighted.mean(tab$SPoRT_raw, tab$n),3)}",
         " | mom {round(weighted.mean(tab$SPoRT_mom, tab$n),3)}\n"))
