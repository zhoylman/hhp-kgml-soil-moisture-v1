library(tidyverse)
library(glue)
library(terra)

# ============================================================================
#  Does the OOS skill improve / stabilize with LONGER records?
#  For the 69 MT Mesonet OOS stations, compare KGML 10-fold-ensemble skill
#  (KGE / r / pbias) computed against (a) the OLD obs (final obs file, ends
#  2025-04) vs (b) the re-pulled longer obs (data/oos-mt-mesonet-obs.csv, ends
#  2026-06). Same ensemble-median predictions; only the obs record differs.
# ============================================================================

source("/home/zhoylman/hhp-kgml-soil-moisture-v1/R/sm_eval_utils.R")

tables_dir = "/home/zhoylman/hhp-kgml-soil-moisture-v1/tables"
cache_dir  = "/home/zhoylman/hhp-kgml-soil-moisture-v1/cache/extractions-oos-longer"
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
obs_dir = "/data/ssd2/soil-moisture-ml/observations"

ens_roots = c("/data/ssd3/soil-moisture-ml-inference", "/data/ssd4/soil-moisture-ml-inference")
resolve_ensemble_dir = function(dep) {
  cand = file.path(ens_roots, glue("ensemble-smoothed-daily-{dep}"), "median")
  hit = cand[dir.exists(cand)]; if (length(hit)) hit[1] else cand[1]
}

new_obs = read_csv("/home/zhoylman/hhp-kgml-soil-moisture-v1/data/oos-mt-mesonet-obs.csv",
                   show_col_types = FALSE) |> mutate(site_id = as.character(site_id))
coords  = read_csv(file.path("/home/zhoylman/hhp-kgml-soil-moisture-v1/tables/oos_sites_to_download.csv"),
                   show_col_types = FALSE) |> transmute(site_id, longitude, latitude)
old_obs = read_csv(file.path(obs_dir, "final-soil-moisture-data-generalized-no-frozen.csv"),
                   show_col_types = FALSE,
                   col_types = cols(site_id = col_character(), date = col_date(),
                                    soil_moisture = col_double(), generalized_depth = col_character())) |>
  filter(as.character(site_id) %in% unique(new_obs$site_id)) |>
  transmute(site_id, date, generalized_depth, soil_moisture)

per_site_metrics = function(obs_df, ens, tag) {
  obs_df |>
    rename(obs = soil_moisture) |>
    inner_join(ens, by = c("site_id", "date")) |>
    group_by(site_id) |>
    group_modify(~ compute_metrics(.x |> rename(ml = ml), site_id = unique(.x$site_id))) |>
    ungroup() |>
    mutate(record = tag)
}

compare_depth = function(depth_flag) {
  dep = tolower(depth_flag)
  no = new_obs |> filter(generalized_depth == depth_flag) |> transmute(site_id, date, soil_moisture)
  oo = old_obs |> filter(generalized_depth == depth_flag) |> transmute(site_id, date, soil_moisture)
  sites = intersect(unique(no$site_id), coords$site_id)
  meta_xy = coords |> filter(site_id %in% sites) |> arrange(site_id)
  all_dates = sort(unique(c(no$date, oo$date)))

  ens = extract_at_sites(
    raster_dir = resolve_ensemble_dir(dep),
    obs_dates = all_dates, site_ids = meta_xy$site_id, meta_xy = meta_xy,
    cache_file = file.path(cache_dir, glue("{dep}_ensemble.rds")),
    label = glue("OOS-longer ensemble [{dep}]"))

  bind_rows(
    per_site_metrics(filter(oo, site_id %in% sites), ens, "OLD (to 2025-04)"),
    per_site_metrics(filter(no, site_id %in% sites), ens, "NEW (to 2026-06)")
  ) |> mutate(depth = depth_flag)
}

ps = bind_rows(lapply(c("Shallow", "Middle"), compare_depth))
readr::write_csv(ps, file.path(tables_dir, "oos_longer_record_per_site.csv"))

summary_tbl = ps |>
  group_by(depth, record) |>
  summarise(n_sites = n(),
            n_robust = sum(n >= 365, na.rm = TRUE),
            med_n_obs = median(n, na.rm = TRUE),
            KGE_med = median(KGE, na.rm = TRUE),
            r_med   = median(r, na.rm = TRUE),
            abs_pbias_med = median(abs(pbias), na.rm = TRUE), .groups = "drop") |>
  arrange(depth, desc(record))
readr::write_csv(summary_tbl, file.path(tables_dir, "oos_longer_record_summary.csv"))

cat("\n=== OOS skill at 69 MT sites: OLD vs LONGER record (ensemble) ===\n")
print(as.data.frame(summary_tbl |> mutate(across(c(KGE_med, r_med, abs_pbias_med), \(x) round(x, 3)))))

# paired: sites robust in NEW but not OLD, and KGE shift on common sites
paired = ps |> select(depth, site_id, record, KGE, n) |>
  pivot_wider(names_from = record, values_from = c(KGE, n))
cat("\n=== paired (same sites): median KGE OLD vs NEW, and robust-gain ===\n")
print(as.data.frame(paired |> group_by(depth) |>
  summarise(med_KGE_old = round(median(`KGE_OLD (to 2025-04)`, na.rm = TRUE), 3),
            med_KGE_new = round(median(`KGE_NEW (to 2026-06)`, na.rm = TRUE), 3),
            robust_old = sum(`n_OLD (to 2025-04)` >= 365, na.rm = TRUE),
            robust_new = sum(`n_NEW (to 2026-06)` >= 365, na.rm = TRUE), .groups = "drop")))
