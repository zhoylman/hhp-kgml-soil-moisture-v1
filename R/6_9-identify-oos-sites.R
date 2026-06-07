library(tidyverse)
library(glue)

# ============================================================================
#  Identify out-of-sample (never-seen) sites for re-download.
#  Walks every k-fold split (train + validation, both depths) and flags which
#  stations were NEVER used in any fold -> candidates for an independent OOS
#  validation on longer records. Writes site_id + network (+ coords + per-depth
#  flags) so the obs can be re-pulled from the API.
#
#  Same logic as the validation (R/6_2): OOS at a depth = a station not in any
#  of that depth's 10 train OR 10 validation splits.
# ============================================================================

obs_dir   = "/data/ssd2/soil-moisture-ml/observations"
split_dir = "/data/ssd2/soil-moisture-ml/split-definitions-kfold"
out_dir   = "/home/zhoylman/hhp-kgml-soil-moisture-v1/tables"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Station registry (site_id + network + coords)
meta = read_csv(file.path(obs_dir, "final-soil-moisture-data-generalized-meta.csv"),
                show_col_types = FALSE) |>
  transmute(site_id = as.character(site_id), network, longitude, latitude) |>
  distinct(site_id, .keep_all = TRUE)

# Union of all site_ids appearing in ANY fold's train or validation split.
kfold_ids = function(depth) {
  files = list.files(glue("{split_dir}-{depth}"),
                     pattern = "(train|validation)_split_fold_.*\\.csv$", full.names = TRUE)
  files |>
    map(~ readr::read_csv(.x, show_col_types = FALSE,
                          col_types = cols(site_id = col_character()))$site_id) |>
    unlist() |> unique() |> as.character()
}
kf_shallow = kfold_ids("shallow")
kf_middle  = kfold_ids("middle")
kf_any     = union(kf_shallow, kf_middle)

# Classify every station.
classified = meta |>
  mutate(
    in_kfold_shallow = site_id %in% kf_shallow,
    in_kfold_middle  = site_id %in% kf_middle,
    oos_shallow      = !in_kfold_shallow,   # never trained/validated at shallow
    oos_middle       = !in_kfold_middle,    # never trained/validated at middle
    fully_oos        = !(site_id %in% kf_any)  # never used at any depth
  )
readr::write_csv(classified, file.path(out_dir, "oos_site_classification.csv"))

# Download list: any station that is OOS for at least one depth (these enable an
# OOS validation at that depth). per-depth flags let you filter per run.
to_download = classified |>
  filter(oos_shallow | oos_middle) |>
  select(site_id, network, longitude, latitude, oos_shallow, oos_middle, fully_oos) |>
  arrange(network, site_id)
readr::write_csv(to_download, file.path(out_dir, "oos_sites_to_download.csv"))

# ---- Report ----
cat(sprintf("Stations in registry: %d\n", nrow(meta)))
cat(sprintf("In k-fold (any depth): %d | OOS shallow: %d | OOS middle: %d | fully OOS (no fold, any depth): %d\n",
            length(kf_any), sum(classified$oos_shallow), sum(classified$oos_middle), sum(classified$fully_oos)))
cat(sprintf("\nTo-download list (OOS in >=1 depth): %d stations\n", nrow(to_download)))
cat("By network:\n")
print(as.data.frame(count(to_download, network, oos_shallow, oos_middle, name = "n")))
cat(sprintf("\nWrote:\n  %s  (full classification, all stations)\n  %s  (download list)\n",
            file.path(out_dir, "oos_site_classification.csv"),
            file.path(out_dir, "oos_sites_to_download.csv")))
