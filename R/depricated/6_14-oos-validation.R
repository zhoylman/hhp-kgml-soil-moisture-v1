library(tidyverse)
library(data.table)
library(glue)
library(terra)

# ============================================================================
#  UNIFIED out-of-sample validation -> standalone oos_validation.csv.
#  Merges all OOS sources (sites in NO k-fold split), scores each against the
#  KGML 10-fold ENSEMBLE, and writes one per-site table.
#
#  Sources (all already frozen/QC-filtered):
#   - NEON         (39 sites x ~5 soil-plot replicates SITE_h1..h5; metrics per
#                   replicate -> median per site)           [raw-depths file]
#   - SNTL (29), USCRN (4) not in any fold                  [raw-depths file]
#   - UMRB/MT Mesonet (re-pulled longer records, 69)        [data/oos-mt-mesonet-obs.csv]
#  Excludes OK Mesonet and SCAN (all in-fold). Depths generalized: Shallow
#  (<=10 cm) / Middle (10-50 cm). Metrics: KGE, r, pbias (raw) + drought-class
#  MAE = mean|SMI_mod - SMI_obs| (beta SMI, needs >=6 yr obs; NA otherwise).
# ============================================================================

source("/home/zhoylman/hhp-kgml-soil-moisture-v1/R/sm_eval_utils.R")

repo      = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
data_root = "/data/ssd2/soil-moisture-ml"
add_dir   = glue("{data_root}/additional_OOS_data")
cache_dir = glue("{repo}/cache/extractions-oos-final")
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)

ens_roots = c("/data/ssd3/soil-moisture-ml-inference", "/data/ssd4/soil-moisture-ml-inference")
resolve_ens = function(dep) {
  cand = file.path(ens_roots, glue("ensemble-smoothed-daily-{dep}"), "median")
  hit = cand[dir.exists(cand)]; if (length(hit)) hit[1] else cand[1]
}
gen_depth = function(d) dplyr::case_when(d <= 10 ~ "Shallow", d > 10 & d <= 50 ~ "Middle", TRUE ~ NA_character_)

# ---- k-fold union (to confirm OOS) ----
kf = c(list.files(glue("{data_root}/split-definitions-kfold-shallow"), "csv$", full.names = TRUE),
       list.files(glue("{data_root}/split-definitions-kfold-middle"),  "csv$", full.names = TRUE)) |>
  map(~ readr::read_csv(.x, show_col_types = FALSE, col_types = cols(site_id = col_character()))$site_id) |>
  unlist() |> unique()

meta = readr::read_csv(glue("{add_dir}/station-meta-conus-w-data-final.csv"), show_col_types = FALSE) |>
  transmute(site_id = as.character(site_id), longitude, latitude) |>
  distinct(site_id, .keep_all = TRUE)

# ---- assemble OOS obs: network, unit_id, base, date, depth, soil_moisture, coords ----
raw = fread(glue("{add_dir}/final-soil-moisture-data-raw-depths.csv"))
raw = raw[network %in% c("NEON", "SNTL", "USCRN")]
raw_oos = as_tibble(raw) |>
  mutate(site_id = as.character(site_id)) |>
  filter(network == "NEON" | !(site_id %in% kf)) |>          # NEON all OOS; SNTL/USCRN only if not in fold
  mutate(generalized_depth = gen_depth(depth)) |>
  filter(!is.na(generalized_depth)) |>
  mutate(base = if_else(network == "NEON", str_remove(site_id, "_h\\d+$"), site_id)) |>
  group_by(network, unit_id = site_id, base, date = as.Date(date), generalized_depth) |>
  summarise(soil_moisture = mean(moisture_corrected, na.rm = TRUE), .groups = "drop") |>
  left_join(meta, by = c("base" = "site_id"))

mt = readr::read_csv(glue("{repo}/data/oos-mt-mesonet-obs.csv"), show_col_types = FALSE) |>
  mutate(site_id = as.character(site_id)) |>
  left_join(readr::read_csv(glue("{repo}/tables/oos_sites_to_download.csv"), show_col_types = FALSE) |>
              select(site_id, longitude, latitude) |> distinct(site_id, .keep_all = TRUE), by = "site_id") |>
  transmute(network, unit_id = site_id, base = site_id, date, generalized_depth,
            soil_moisture, longitude, latitude)

oos_obs = bind_rows(raw_oos, mt) |> filter(!is.na(longitude), !is.na(latitude))
message(glue("OOS obs: {nrow(oos_obs)} rows | {n_distinct(oos_obs$base)} locations | networks: {paste(sort(unique(oos_obs$network)), collapse=', ')}"))

# ---- per-depth: extract ensemble at base locations, standardize, score units ----
score_depth = function(depth_flag) {
  dep = tolower(depth_flag)
  od  = oos_obs |> filter(generalized_depth == depth_flag)
  if (!nrow(od)) return(NULL)
  locs = od |> distinct(base, longitude, latitude) |> distinct(base, .keep_all = TRUE) |> arrange(base)

  ens = extract_at_sites(
    raster_dir = resolve_ens(dep),
    obs_dates  = sort(unique(od$date)),
    site_ids   = locs$base,
    meta_xy    = locs |> transmute(site_id = base, longitude, latitude),
    cache_file = file.path(cache_dir, glue("{dep}_ensemble.rds")),
    label      = glue("OOS-final ensemble [{dep}]"))

  # ensemble SMI per location (its own <=30 yr climatology)
  smi_mod = ens |> group_by(site_id) |>
    group_modify(~ standardize_doy_beta(transmute(.x, date, value = ml)) |> transmute(date, smi_mod = pmin(pmax(z, -2), 2))) |>
    ungroup() |> rename(base = site_id)

  # per-unit metrics
  units = od |> distinct(network, unit_id, base)
  res = purrr::pmap_dfr(units, function(network, unit_id, base) {
    o  = od |> filter(unit_id == !!unit_id) |> select(date, obs = soil_moisture)
    es = ens |> filter(site_id == !!base) |> select(date, ml)
    j  = inner_join(o, es, by = "date") |> drop_na(obs, ml)
    if (nrow(j) < 5) return(NULL)
    m  = compute_metrics(j, site_id = unit_id)            # KGE/r/pbias
    # drought-class (SMI) MAE
    so = standardize_doy_beta(transmute(o, date, value = obs)) |> transmute(date, smi_obs = pmin(pmax(z, -2), 2))
    sj = inner_join(so, filter(smi_mod, base == !!base) |> select(date, smi_mod), by = "date") |> drop_na()
    tibble(network, unit_id, base, n_obs = m$n, KGE = m$KGE, r = m$r, pbias = m$pbias,
           smi_mae = if (nrow(sj) >= 30) mean(abs(sj$smi_mod - sj$smi_obs)) else NA_real_,
           n_smi = nrow(sj))
  })
  if (!nrow(res)) return(NULL)

  # NEON: median across replicates -> one row per base site; others pass through
  res |>
    group_by(network, site_id = base) |>
    summarise(n_replicates = n_distinct(unit_id),
              n_obs = round(median(n_obs)), KGE = median(KGE, na.rm = TRUE),
              r = median(r, na.rm = TRUE), pbias = median(pbias, na.rm = TRUE),
              smi_mae = median(smi_mae, na.rm = TRUE), n_smi = round(median(n_smi)),
              .groups = "drop") |>
    mutate(depth = depth_flag, robust = n_obs >= 365) |>
    left_join(distinct(locs, site_id = base, longitude, latitude), by = "site_id")
}

oos_validation = bind_rows(lapply(c("Shallow", "Middle"), score_depth)) |>
  select(network, site_id, depth, longitude, latitude, n_replicates, n_obs, robust,
         KGE, r, pbias, smi_mae, n_smi) |>
  arrange(depth, network, site_id)

readr::write_csv(oos_validation, glue("{repo}/tables/oos_validation.csv"))

cat(glue("\nWrote tables/oos_validation.csv : {nrow(oos_validation)} site x depth rows\n"))
cat("\n=== OOS skill by network x depth (median) ===\n")
print(as.data.frame(oos_validation |> group_by(depth, network) |>
  summarise(n = n(), robust = sum(robust),
            KGE = round(median(KGE, na.rm = TRUE), 3),
            r = round(median(r, na.rm = TRUE), 3),
            abs_pbias = round(median(abs(pbias), na.rm = TRUE), 1),
            smi_mae = round(median(smi_mae, na.rm = TRUE), 3), .groups = "drop")))
