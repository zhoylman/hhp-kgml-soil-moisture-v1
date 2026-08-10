library(tidyverse)
library(data.table)
library(glue)
library(terra)

# ============================================================================
#  BUILDER 2/2 — out-of-sample validation dataset (sites in NO k-fold split).
#  Scored against the KGML 10-fold ENSEMBLE median. Sources (all frozen/QC'd):
#    - UMRB/MT Mesonet (re-pulled longer records, data/oos-mt-mesonet-obs.csv)
#    - SNTL, USCRN not in any fold        [additional_OOS_data raw-depths]
#  EXCLUDED: NEON (bad obs quality), OK Mesonet, SCAN (all in-fold).
#  Emits:
#    tables/oos_validation.csv          — per site x depth (KGE/r/pbias + SMI MAE)
#    cache/datasets/oos_matched.rds     — per site x day (obs, ml, obs_smi, kgml_smi,
#                                         sport_smi[=NA], truth_class)
#  Depths generalized Shallow (<=10) / Middle (10-50). See memory: validation-dataset-decisions.
# ============================================================================

source("/home/zhoylman/hhp-kgml-soil-moisture-v1/R/sm_eval_utils.R")

repo      = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
data_root = "/data/ssd2/soil-moisture-ml"
add_dir   = glue("{data_root}/additional_OOS_data")
cache_dir = glue("{repo}/cache/extractions-oos-final")
out_ds    = glue("{repo}/cache/datasets"); dir.create(out_ds, showWarnings = FALSE, recursive = TRUE)
EXCLUDE_NETWORKS = c("NEON", "OK Mesonet")

ens_roots = c("/data/ssd3/soil-moisture-ml-inference", "/data/ssd4/soil-moisture-ml-inference")
# ARCH_SUFFIX "-yearfrozen" (default, promoted/served archive) or "" (as-is). The
# year-freeze materially changes 2023+ predictions (year 2023->2013), so OOS must
# use the same archive that is served operationally.
arch_suffix = Sys.getenv("ARCH_SUFFIX", "-yearfrozen")
resolve_ens = function(dep) {
  cand = file.path(ens_roots, glue("ensemble-smoothed-daily-{dep}{arch_suffix}"), "median")
  hit = cand[dir.exists(cand)]; if (length(hit)) hit[1] else cand[1]
}
gen_depth = function(d) dplyr::case_when(d <= 10 ~ "Shallow", d > 10 & d <= 50 ~ "Middle", TRUE ~ NA_character_)

kf = c(list.files(glue("{data_root}/split-definitions-kfold-shallow"), "csv$", full.names = TRUE),
       list.files(glue("{data_root}/split-definitions-kfold-middle"),  "csv$", full.names = TRUE)) |>
  map(~ readr::read_csv(.x, show_col_types = FALSE, col_types = cols(site_id = col_character()))$site_id) |>
  unlist() |> unique()

meta = readr::read_csv(glue("{add_dir}/station-meta-conus-w-data-final.csv"), show_col_types = FALSE) |>
  transmute(site_id = as.character(site_id), longitude, latitude) |> distinct(site_id, .keep_all = TRUE)

# ---- assemble OOS obs ----
raw = fread(glue("{add_dir}/final-soil-moisture-data-raw-depths.csv"))
raw = raw[network %in% c("SNTL", "USCRN")]                       # NEON & OK Mesonet excluded
raw_oos = as_tibble(raw) |>
  mutate(site_id = as.character(site_id)) |>
  filter(!(site_id %in% kf)) |>
  mutate(generalized_depth = gen_depth(depth)) |> filter(!is.na(generalized_depth)) |>
  group_by(network, site_id, base = site_id, date = as.Date(date), generalized_depth) |>
  summarise(soil_moisture = mean(moisture_corrected, na.rm = TRUE), .groups = "drop") |>
  left_join(meta, by = "site_id")

mt = readr::read_csv(glue("{repo}/data/oos-mt-mesonet-obs.csv"), show_col_types = FALSE) |>
  mutate(site_id = as.character(site_id)) |>
  left_join(readr::read_csv(glue("{repo}/tables/oos_sites_to_download.csv"), show_col_types = FALSE) |>
              select(site_id, longitude, latitude) |> distinct(site_id, .keep_all = TRUE), by = "site_id") |>
  transmute(network, site_id, base = site_id, date, generalized_depth, soil_moisture, longitude, latitude)

oos_obs = bind_rows(raw_oos, mt) |> filter(!is.na(longitude), !is.na(latitude),
                                           !(network %in% EXCLUDE_NETWORKS))
message(glue("OOS obs: {nrow(oos_obs)} rows | {n_distinct(oos_obs$base)} locations | {paste(sort(unique(oos_obs$network)), collapse=', ')}"))

# ---- per depth: extract ensemble, build matched (per site x day) ----
match_depth = function(depth_flag) {
  dep = tolower(depth_flag)
  od  = oos_obs |> filter(generalized_depth == depth_flag)
  if (!nrow(od)) return(NULL)
  locs = od |> distinct(base, longitude, latitude) |> distinct(base, .keep_all = TRUE) |> arrange(base)
  ens = extract_at_sites(
    raster_dir = resolve_ens(dep), obs_dates = sort(unique(od$date)),
    site_ids = locs$base, meta_xy = locs |> transmute(site_id = base, longitude, latitude),
    cache_file = file.path(cache_dir, glue("{dep}{arch_suffix}_ensemble.rds")), label = glue("OOS ensemble [{dep}{arch_suffix}]"))

  smi_mod = ens |> group_by(site_id) |>
    group_modify(~ standardize_doy_beta(transmute(.x, date, value = ml)) |>
                   transmute(date, kgml_smi = pmin(pmax(z, -3.09), 3.09))) |> ungroup() |> rename(base = site_id)

  bind_rows(lapply(unique(od$base), function(b) {
    o  = od |> filter(base == !!b) |> select(network, site_id, date, obs = soil_moisture)
    es = ens |> filter(site_id == !!b) |> select(date, ml)
    mt = inner_join(o, es, by = "date") |> drop_na(obs, ml)
    if (nrow(mt) < 5) return(NULL)
    osm = standardize_doy_beta(transmute(o, date, value = obs)) |> transmute(date, obs_smi = pmin(pmax(z, -3.09), 3.09))
    mt |>
      left_join(osm, by = "date") |>
      left_join(filter(smi_mod, base == !!b) |> select(date, kgml_smi), by = "date") |>
      mutate(depth = depth_flag, sport_smi = NA_real_)
  }))
}

matched = bind_rows(lapply(c("Shallow", "Middle"), match_depth)) |>
  mutate(truth_class = smi_to_class(obs_smi)) |>
  select(network, site_id, depth, date, obs, ml, obs_smi, kgml_smi, sport_smi, truth_class)
saveRDS(matched, file.path(out_ds, "oos_matched.rds"))

# ---- per-site CSV ----
coords = oos_obs |> distinct(site_id, longitude, latitude) |> distinct(site_id, .keep_all = TRUE)
oos_validation = matched |>
  group_by(network, site_id, depth) |>
  summarise(n_obs = sum(!is.na(obs) & !is.na(ml)),
            KGE = hydroGOF::KGE(ml, obs), r = cor(ml, obs, use = "complete.obs"),
            pbias = hydroGOF::pbias(ml, obs),
            n_smi = sum(!is.na(obs_smi) & !is.na(kgml_smi)),
            smi_mae = if (sum(!is.na(obs_smi) & !is.na(kgml_smi)) >= 30)
                        mean(abs(kgml_smi - obs_smi), na.rm = TRUE) else NA_real_,
            .groups = "drop") |>
  left_join(coords, by = "site_id") |>
  mutate(robust = n_obs >= 365) |>
  select(network, site_id, depth, longitude, latitude, n_obs, robust, KGE, r, pbias, smi_mae, n_smi) |>
  arrange(depth, network, site_id)

readr::write_csv(oos_validation, glue("{repo}/tables/oos_validation.csv"))

cat(glue("\nWrote tables/oos_validation.csv ({nrow(oos_validation)} rows) + cache/datasets/oos_matched.rds ({nrow(matched)} rows)\n"))
cat("\n=== OOS skill (robust >=365d) by depth, lumped + by network ===\n")
print(as.data.frame(oos_validation |> filter(robust) |> group_by(depth) |>
  summarise(n = n(), KGE = round(median(KGE, na.rm = TRUE), 3), r = round(median(r, na.rm = TRUE), 3),
            abs_pbias = round(median(abs(pbias), na.rm = TRUE), 1), .groups = "drop")))
print(as.data.frame(oos_validation |> filter(robust) |> group_by(depth, network) |>
  summarise(n = n(), KGE = round(median(KGE, na.rm = TRUE), 3), .groups = "drop")))
