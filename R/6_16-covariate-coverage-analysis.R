##############################################################
# Title: Static covariate-space coverage — stations vs. pretraining sample
# Description:
#   Descriptive/statistical analysis (NO model training) supporting the
#   manuscript claim that SPoRT-LIS pretraining (~20,000 CONUS locations)
#   covers climate/soil/physiographic combinations that the 734 fine-tuning
#   in-situ stations never sample. Addresses reviewer risk that the
#   pretraining ablation benefit reflects optimization advantage rather
#   than genuine covariate-space information/coverage advantage.
#
#   Three matrices on identical static-covariate columns (31 vars: PRISM
#   1991-2020 normals, SSURGO 0-152cm, POLARIS, terrain), all in EPSG:5070
#   (area-proportional cell counts):
#     - CONUS   : /data/ssd4/.../static-data-stack/static_aligned_stack.tif
#     - stations: 734-site full fine-tuning network (shallow depth
#                 split-definitions-shallow/train_734.csv), static covars
#                 from static-data/all-sites-static-data.csv, coords from
#                 observations/final-soil-moisture-data-generalized-meta.csv
#     - pretrain: 20,000-site sample, static covars from
#                 static-data/pretrain-sites-static-data.csv, coords from
#                 random-pretraining-roi/pretraining-roi.geojson
#
#   Method: z-score (CONUS distribution) -> PCA (CONUS) -> project stations
#   & pretrain into CONUS PC space -> 4 coverage metrics (NN-distance incl.
#   Mahalanobis, range-box containment, binned occupancy on PCs + raw
#   interpretable axes, 2-D density ratio) computed for BOTH reference sets,
#   at multiple arbitrary-choice settings to test sensitivity.
#
#   IMPORTANT — this quantifies covariate-SPACE coverage only. It is
#   necessary but NOT sufficient evidence of predictive extrapolation
#   failure/success; no model training or prediction is performed here.
#
# Author: Claude (agent), for Z. Hoylman
##############################################################

suppressPackageStartupMessages({
  library(tidyverse); library(terra); library(sf); library(glue)
  library(FNN); library(ks); library(patchwork); library(gt)
})

set.seed(42)
repo      = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
data_root = "/data/ssd2/soil-moisture-ml"
figs_dir  = glue("{repo}/figs")
tabs_dir  = glue("{repo}/tables")
cache_dir = glue("{repo}/cache/covariate-coverage"); dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

# ---- static covariate columns (identical across all three matrices; order
# matches py/exp_point_centerkeep_eval.py PRISM_COLS + TERRAIN_SOIL_COLS) ----
prism_cols = c("ppt","solclear","solslope","soltotal","soltrans","tdmean",
               "tmax","tmean","tmin","vpdmax","vpdmin")
terrain_soil_cols = c(
  "X0_b1","X1_constant","X2_elevation",                       # terrain: TWI, topo diversity/ruggedness, elevation
  "X0_ssurgo_awc","X1_ssurgo_clay","X2_ssurgo_ksat","X3_ssurgo_sand",  # SSURGO 0-152cm
  "X0_bd_mean","X1_clay_mean","X10_lambda_mean","X11_hb_mean","X12_alpha_mean",
  "X2_ksat_mean","X3_n_mean","X4_om_mean","X5_ph_mean","X6_sand_mean",
  "X7_silt_mean","X8_theta_r_mean","X9_theta_s_mean")           # POLARIS
static_cols = c(prism_cols, terrain_soil_cols)   # 31 covariates
stopifnot(length(static_cols) == 31)

# ---- data-quality scrub: POLARIS van Genuchten alpha (X12_alpha_mean) carries
# a known corrupted overflow/sentinel value (~6.646e36) in a small number of
# cells. This exact sentinel also appears as the reported max in the model
# pipeline's own min-max-definitions/static-min-max-definitions*.csv, so it is
# a pre-existing upstream POLARIS export artifact, not something introduced
# here. Any |value| > 50 (true physical range is roughly [-1, 1]) is treated
# as missing before z-scoring/PCA so it cannot distort the covariate space.
scrub_sentinel = function(df, thresh = 50) {
  bad = abs(df$X12_alpha_mean) > thresh
  n_bad = sum(bad, na.rm = TRUE)
  if (n_bad > 0) df$X12_alpha_mean[which(bad)] = NA_real_
  attr(df, "n_sentinel_scrubbed") = n_bad
  df
}

var_label = c(
  ppt = "PRISM precipitation", solclear = "PRISM solar (clear-sky)",
  solslope = "PRISM solar (slope)", soltotal = "PRISM solar (total)",
  soltrans = "PRISM solar (transmittance)", tdmean = "PRISM dewpoint",
  tmax = "PRISM tmax", tmean = "PRISM tmean", tmin = "PRISM tmin",
  vpdmax = "PRISM VPD max", vpdmin = "PRISM VPD min",
  X0_b1 = "Terrain: TWI", X1_constant = "Terrain: topo diversity/ruggedness",
  X2_elevation = "Terrain: elevation (NED)",
  X0_ssurgo_awc = "SSURGO: AWC", X1_ssurgo_clay = "SSURGO: clay",
  X2_ssurgo_ksat = "SSURGO: Ksat", X3_ssurgo_sand = "SSURGO: sand",
  X0_bd_mean = "POLARIS: bulk density", X1_clay_mean = "POLARIS: clay",
  X10_lambda_mean = "POLARIS: Brooks-Corey pore-size idx",
  X11_hb_mean = "POLARIS: Brooks-Corey air-entry", X12_alpha_mean = "POLARIS: van Genuchten alpha",
  X2_ksat_mean = "POLARIS: Ksat", X3_n_mean = "POLARIS: van Genuchten n",
  X4_om_mean = "POLARIS: organic matter", X5_ph_mean = "POLARIS: pH",
  X6_sand_mean = "POLARIS: sand", X7_silt_mean = "POLARIS: silt",
  X8_theta_r_mean = "POLARIS: van Genuchten theta_r", X9_theta_s_mean = "POLARIS: van Genuchten theta_s")

message("== 1. Load three matrices ==")

# ---- (A) CONUS grid: static raster, reprojected to EPSG:5070 (bilinear, house
# convention, cf. R/6_10, R/6_11) so cell counts are area-proportional. ----
static_tif = "/data/ssd4/soil-moisture-ml-inference/inference-rasters/static-data-stack/static_aligned_stack.tif"
r_raw = terra::rast(static_tif)
r_5070 = terra::project(r_raw, "EPSG:5070", method = "bilinear")
conus_all = terra::as.data.frame(r_5070, xy = TRUE, na.rm = FALSE)
conus_all = scrub_sentinel(conus_all)
message(glue("CONUS grid: scrubbed {attr(conus_all,'n_sentinel_scrubbed')} X12_alpha_mean sentinel/overflow cells to NA."))

# NA report BEFORE dropping anything (per-variable, on the reprojected grid)
na_report_conus = tibble(variable = static_cols,
                          n_total = nrow(conus_all),
                          n_na = map_dbl(static_cols, ~ sum(is.na(conus_all[[.x]]))),
                          pct_na = 100 * n_na / n_total, source = "CONUS grid")

conus_cc = conus_all[complete.cases(conus_all[, static_cols]), ]
message(glue("CONUS grid: {nrow(conus_all)} total cells (4326->5070, bilinear); ",
             "{nrow(conus_cc)} complete-case land cells retained ({round(100*nrow(conus_cc)/nrow(conus_all),1)}%) ",
             "after masking water/ice/outside-CONUS (NA in >=1 of {length(static_cols)} covariates)."))

# ---- (B) 734 fine-tuning stations (shallow depth full-network split; static
# covariates present regardless of depth since these are location covariates) ----
train734 = read_csv(glue("{data_root}/split-definitions-shallow/train_734.csv"),
                     show_col_types = FALSE, col_types = cols(site_id = col_character()))
station_ids = sort(unique(train734$site_id))
stopifnot(length(station_ids) == 734)

station_static = read_csv(glue("{data_root}/static-data/all-sites-static-data.csv"),
                           show_col_types = FALSE, col_types = cols(site_id = col_character())) |>
  filter(site_id %in% station_ids)
site_meta = read_csv(glue("{data_root}/observations/final-soil-moisture-data-generalized-meta.csv"),
                      show_col_types = FALSE, col_types = cols(site_id = col_character())) |>
  transmute(site_id, longitude, latitude) |> distinct(site_id, .keep_all = TRUE)
station_df = station_static |> left_join(site_meta, by = "site_id")
stopifnot(nrow(station_df) == 734, sum(is.na(station_df$longitude)) == 0)
station_df = scrub_sentinel(station_df)
message(glue("Stations: scrubbed {attr(station_df,'n_sentinel_scrubbed')} X12_alpha_mean sentinel cells to NA."))

na_report_station = tibble(variable = static_cols, n_total = nrow(station_df),
                            n_na = map_dbl(static_cols, ~ sum(is.na(station_df[[.x]]))),
                            pct_na = 100 * n_na / n_total, source = "734 stations")

# ---- (C) ~20,000 pretraining locations ----
pretrain_static = read_csv(glue("{data_root}/static-data/pretrain-sites-static-data.csv"), show_col_types = FALSE)
pretrain_geo = st_read(glue("{data_root}/random-pretraining-roi/pretraining-roi.geojson"), quiet = TRUE) |>
  st_coordinates() |> as_tibble() |> rename(longitude = X, latitude = Y) |>
  mutate(site_id = st_read(glue("{data_root}/random-pretraining-roi/pretraining-roi.geojson"), quiet = TRUE)$site_id)
pretrain_df = pretrain_static |> left_join(pretrain_geo, by = "site_id")
stopifnot(nrow(pretrain_df) == 20000)
pretrain_df = scrub_sentinel(pretrain_df)
message(glue("Pretraining: scrubbed {attr(pretrain_df,'n_sentinel_scrubbed')} X12_alpha_mean sentinel cells to NA."))

na_report_pretrain = tibble(variable = static_cols, n_total = nrow(pretrain_df),
                            n_na = map_dbl(static_cols, ~ sum(is.na(pretrain_df[[.x]]))),
                            pct_na = 100 * n_na / n_total, source = "~20,000 pretraining sample")

na_report = bind_rows(na_report_conus, na_report_station, na_report_pretrain) |>
  mutate(flag_high_missing = pct_na > 1) |>
  arrange(desc(pct_na))
write_csv(na_report, glue("{tabs_dir}/covariate_coverage_na_report.csv"))
n_flag = sum(na_report$flag_high_missing)
message(glue("NA report written ({tabs_dir}/covariate_coverage_na_report.csv). ",
             "{n_flag} (source x variable) combinations exceed 1% missing ",
             "(max = {round(max(na_report$pct_na),2)}% ). All treated by complete-case removal."))

station_cc = station_df[complete.cases(station_df[, static_cols]), ]
pretrain_cc = pretrain_df[complete.cases(pretrain_df[, static_cols]), ]
message(glue("Complete-case retained: {nrow(station_cc)}/734 stations, ",
             "{nrow(pretrain_cc)}/20000 pretraining locations, ",
             "{nrow(conus_cc)}/{nrow(conus_all)} CONUS cells."))

message("== 2. Z-score standardize using the CONUS distribution ==")
conus_mean = colMeans(conus_cc[, static_cols])
conus_sd   = apply(conus_cc[, static_cols], 2, sd)
zscore = function(df) sweep(sweep(as.matrix(df[, static_cols]), 2, conus_mean, "-"), 2, conus_sd, "/")
Z_conus    = zscore(conus_cc)
Z_station  = zscore(station_cc)
Z_pretrain = zscore(pretrain_cc)

message("== 3. PCA on the CONUS matrix ==")
pca = prcomp(Z_conus, center = FALSE, scale. = FALSE)
eig = pca$sdev^2
var_explained = eig / sum(eig)
cum_var = cumsum(var_explained)
n90 = which(cum_var >= 0.90)[1]
n95 = which(cum_var >= 0.95)[1]
# effective dimensionality (participation ratio) on the FULL eigenvalue spectrum
participation_ratio = sum(eig)^2 / sum(eig^2)
message(glue("PCA: {n90} PCs reach 90% variance; {n95} PCs reach 95% variance (of {length(eig)} total). ",
             "Participation ratio (effective dimensionality) = {round(participation_ratio,2)} ",
             "out of {length(static_cols)} raw variables -- PRISM intercorrelation inflates the raw count."))

scree_df = tibble(pc = seq_along(eig), eigenvalue = eig, var_explained = var_explained, cum_var = cum_var)
write_csv(scree_df, glue("{tabs_dir}/covariate_coverage_scree.csv"))

project_pc = function(Z) Z %*% pca$rotation
PC_conus    = project_pc(Z_conus)
PC_station  = project_pc(Z_station)
PC_pretrain = project_pc(Z_pretrain)

# ---- two "retained PCs" choices used throughout (explicit sensitivity axis) ----
pc_choices = list(N90 = n90, N95 = n95)
message(glue("Retained-PC choices for sensitivity: N90={n90}, N95={n95}."))

message("== 4-5. Coverage metrics for BOTH reference sets, at both PC choices ==")

# ---- helper: Mahalanobis whitening transform from a reference point cloud's
# own covariance (so Euclidean NN in whitened space == Mahalanobis NN in the
# original space, w.r.t. that reference set's covariance/shape) ----
whiten = function(X, ref) {
  S = cov(ref)
  Lc = chol(S)                       # S = t(Lc) %*% Lc (upper-tri chol)
  X %*% solve(Lc)                    # right-multiply: rows are points
}

nn_metric = function(pc_conus, pc_ref, k_self = 1) {
  # station/pretrain self NN (leave-one-out) -> 95th pct threshold
  self = FNN::get.knn(pc_ref, k = k_self)$nn.dist[, k_self]
  thresh = quantile(self, 0.95)
  d_to_ref = FNN::get.knnx(data = pc_ref, query = pc_conus, k = 1)$nn.dist[, 1]
  frac_unsampled = mean(d_to_ref > thresh)
  list(threshold = thresh, frac_unsampled = frac_unsampled, d_to_ref = d_to_ref, self = self)
}

rangebox_metric = function(pc_conus, pc_ref) {
  lo = apply(pc_ref, 2, min); hi = apply(pc_ref, 2, max)
  outside = rowSums(sweep(pc_conus, 2, lo, "<") | sweep(pc_conus, 2, hi, ">")) > 0
  mean(outside)
}

binned_occupancy = function(pc_conus, pc_ref, n_pc, n_bins) {
  # quantile bins per PC, defined on the CONUS distribution (breaks), applied to both
  breaks = lapply(seq_len(n_pc), function(j) {
    b = unique(quantile(pc_conus[, j], probs = seq(0, 1, length.out = n_bins + 1)))
    if (length(b) < 2) b = range(pc_conus[, j]) + c(-1e-9, 1e-9)
    b
  })
  bin_id = function(M) {
    codes = sapply(seq_len(n_pc), function(j) as.integer(cut(M[, j], breaks[[j]], include.lowest = TRUE)))
    if (n_pc == 1) codes = matrix(codes, ncol = 1)
    apply(codes, 1, paste, collapse = "_")
  }
  conus_bin = bin_id(pc_conus[, seq_len(n_pc), drop = FALSE])
  ref_bin   = bin_id(pc_ref[, seq_len(n_pc), drop = FALSE])
  occ_tab = table(conus_bin)                     # CONUS-occupied bins with area (cell count)
  ref_bins = unique(ref_bin)
  occupied_bins = names(occ_tab)
  empty_of_occupied = setdiff(occupied_bins, ref_bins)
  area_frac_empty = sum(occ_tab[empty_of_occupied]) / sum(occ_tab)
  list(n_bins_total = n_bins^n_pc, n_occupied_conus = length(occupied_bins),
       n_occupied_by_ref = length(intersect(occupied_bins, ref_bins)),
       area_frac_empty_of_occupied = area_frac_empty)
}

density_ratio_metric = function(pc_conus12, pc_ref12, zero_thresh_pct = 0.01) {
  H = ks::Hns(pc_conus12)
  xmin = apply(pc_conus12, 2, min); xmax = apply(pc_conus12, 2, max)
  k_conus = ks::kde(x = pc_conus12, H = H, gridsize = c(150, 150), xmin = xmin, xmax = xmax)
  k_ref   = ks::kde(x = pc_ref12,   H = H, gridsize = c(150, 150), xmin = xmin, xmax = xmax)
  conus_mass = k_conus$estimate / sum(k_conus$estimate)     # normalized to sum=1 -> "area mass"
  ref_dens = k_ref$estimate
  eff_zero = ref_dens < (zero_thresh_pct / 100 * max(ref_dens))
  area_frac_zero_density = sum(conus_mass[eff_zero])
  list(area_frac_zero_density = area_frac_zero_density, k_conus = k_conus, k_ref = k_ref, H = H)
}

# ---- run (a) NN-distance (Euclidean + Mahalanobis) x (b) range-box across both
# PC choices and both reference sets ----
results_nn = list(); results_box = list()
for (pcname in names(pc_choices)) {
  npc = pc_choices[[pcname]]
  pcc = PC_conus[, seq_len(npc), drop = FALSE]
  for (ref_name in c("station", "pretrain")) {
    pcr = (if (ref_name == "station") PC_station else PC_pretrain)[, seq_len(npc), drop = FALSE]

    eu = nn_metric(pcc, pcr)
    results_nn[[length(results_nn) + 1]] = tibble(
      pc_choice = pcname, n_pc = npc, reference = ref_name, distance = "Euclidean",
      threshold_95pct_selfNN = eu$threshold, area_frac_unsampled = eu$frac_unsampled)

    W_conus = whiten(pcc, pcr); W_ref = whiten(pcr, pcr)
    ma = nn_metric(W_conus, W_ref)
    results_nn[[length(results_nn) + 1]] = tibble(
      pc_choice = pcname, n_pc = npc, reference = ref_name, distance = "Mahalanobis",
      threshold_95pct_selfNN = ma$threshold, area_frac_unsampled = ma$frac_unsampled)

    if (pcname == "N95" && ref_name == "station") assign("conus_dist_euclid_station_N95", eu$d_to_ref, envir = .GlobalEnv)
    if (pcname == "N95" && ref_name == "station") assign("conus_dist_maha_station_N95", ma$d_to_ref, envir = .GlobalEnv)
    if (pcname == "N95" && ref_name == "pretrain") assign("conus_dist_euclid_pretrain_N95", eu$d_to_ref, envir = .GlobalEnv)

    box_frac = rangebox_metric(pcc, pcr)
    results_box[[length(results_box) + 1]] = tibble(
      pc_choice = pcname, n_pc = npc, reference = ref_name, area_frac_outside_box = box_frac)
  }
}
nn_results = bind_rows(results_nn)
box_results = bind_rows(results_box)
write_csv(nn_results, glue("{tabs_dir}/covariate_coverage_nn_distance.csv"))
write_csv(box_results, glue("{tabs_dir}/covariate_coverage_rangebox.csv"))

# ---- (c) binned occupancy on PCs: N=3 and N=5, both reference sets, bins=4
# (primary) plus bins=3,5 sensitivity at N=5 ----
bin_settings = tribble(~n_pc, ~n_bins, ~role,
                        3, 4, "primary",
                        5, 4, "primary",
                        5, 3, "sensitivity",
                        5, 5, "sensitivity")
occ_results = list()
for (i in seq_len(nrow(bin_settings))) {
  npc = bin_settings$n_pc[i]; nb = bin_settings$n_bins[i]
  for (ref_name in c("station", "pretrain")) {
    pcr = (if (ref_name == "station") PC_station else PC_pretrain)
    m = binned_occupancy(PC_conus, pcr, npc, nb)
    occ_results[[length(occ_results) + 1]] = tibble(
      n_pc = npc, n_bins_per_axis = nb, role = bin_settings$role[i], reference = ref_name,
      n_bins_total_possible = m$n_bins_total, n_bins_occupied_by_CONUS = m$n_occupied_conus,
      n_bins_occupied_by_ref = m$n_occupied_by_ref,
      area_frac_occupied_bins_with_zero_ref = m$area_frac_empty_of_occupied)
  }
}
occ_pc_results = bind_rows(occ_results)
write_csv(occ_pc_results, glue("{tabs_dir}/covariate_coverage_binned_occupancy_pc.csv"))

# ---- (c) binned occupancy on interpretable RAW axes (1-D, deciles) ----
raw_axes = c("ppt", "tmean", "vpdmax", "X3_ssurgo_sand", "X0_ssurgo_awc")
raw_axis_label = c(ppt = "PRISM precipitation (mm)", tmean = "PRISM mean temp (C)",
                    vpdmax = "PRISM max VPD (hPa)", X3_ssurgo_sand = "SSURGO sand (%)",
                    X0_ssurgo_awc = "SSURGO AWC (cm/cm)")
raw_occ = list()
for (v in raw_axes) {
  for (ref_name in c("station", "pretrain")) {
    ref_df = if (ref_name == "station") station_cc else pretrain_cc
    breaks = unique(quantile(conus_cc[[v]], probs = seq(0, 1, length.out = 11)))
    conus_bin = cut(conus_cc[[v]], breaks, include.lowest = TRUE)
    ref_bin = cut(ref_df[[v]], breaks, include.lowest = TRUE)
    occ_tab = table(conus_bin)
    ref_present = table(ref_bin) > 0
    empty = names(occ_tab)[!(names(occ_tab) %in% names(ref_present)[ref_present])]
    raw_occ[[length(raw_occ) + 1]] = tibble(
      variable = v, reference = ref_name, bin = names(occ_tab), n_conus_cells = as.numeric(occ_tab),
      area_frac = as.numeric(occ_tab) / sum(occ_tab),
      ref_present = names(occ_tab) %in% names(ref_present)[ref_present])
  }
}
raw_occ_df = bind_rows(raw_occ)
write_csv(raw_occ_df, glue("{tabs_dir}/covariate_coverage_binned_occupancy_raw.csv"))
raw_occ_summary = raw_occ_df |> group_by(variable, reference) |>
  summarise(area_frac_occupied_bins_with_zero_ref = sum(area_frac[!ref_present]), .groups = "drop")
write_csv(raw_occ_summary, glue("{tabs_dir}/covariate_coverage_binned_occupancy_raw_summary.csv"))

# ---- (d) density ratio on PC1-PC2 ----
dens_results = list()
dens_cache = list()
for (ref_name in c("station", "pretrain")) {
  pcr = if (ref_name == "station") PC_station else PC_pretrain
  dm = density_ratio_metric(PC_conus[, 1:2], pcr[, 1:2])
  dens_results[[length(dens_results) + 1]] = tibble(reference = ref_name,
                                                      area_frac_zero_density = dm$area_frac_zero_density)
  dens_cache[[ref_name]] = dm
}
dens_results_df = bind_rows(dens_results)
write_csv(dens_results_df, glue("{tabs_dir}/covariate_coverage_density_ratio.csv"))

message("== 6. Characterize what's missing (flagged by Euclidean NN, station ref, N95) ==")
unsampled_mask = conus_dist_euclid_station_N95 > (nn_results |> filter(pc_choice == "N95", reference == "station", distance == "Euclidean") |> pull(threshold_95pct_selfNN))
message(glue("{sum(unsampled_mask)} / {length(unsampled_mask)} CONUS cells flagged unsampled ",
             "({round(100*mean(unsampled_mask),1)}% of area) under Euclidean/N95/station."))

smd = map_dfr(static_cols, function(v) {
  tibble(variable = v, label = var_label[[v]],
         mean_unsampled = mean(conus_cc[[v]][unsampled_mask]),
         mean_station = mean(station_cc[[v]]),
         sd_conus = sd(conus_cc[[v]]),
         smd = (mean(conus_cc[[v]][unsampled_mask]) - mean(station_cc[[v]])) / sd(conus_cc[[v]]))
}) |> arrange(desc(abs(smd)))
write_csv(smd, glue("{tabs_dir}/covariate_coverage_smd_unsampled.csv"))
message("Top drivers of unsampled-cell divergence (|SMD|, CONUS-sd units):")
print(smd |> select(label, smd) |> head(8))

# ---- ecoregion cross-tab: search locally only; skip + state plainly if absent ----
eco_hits = tryCatch(system("find /data /home/zhoylman /usr/share -iname '*ecoregion*' 2>/dev/null", intern = TRUE), error = function(e) character(0))
eco_hits = eco_hits[nzchar(eco_hits)]
if (length(eco_hits)) {
  message(glue("Found possible ecoregion file(s) locally: {paste(eco_hits, collapse=', ')} -- NOT used automatically; inspect before use."))
} else {
  message("No EPA Level II ecoregion shapefile found locally (searched /data, /home/zhoylman, /usr/share). Ecoregion cross-tab SKIPPED per instructions (no internet fetch).")
}

message("== Save station/pretrain/CONUS coords (5070) for mapping ==")
to_5070 = function(df) st_as_sf(df, coords = c("longitude", "latitude"), crs = 4326) |> st_transform(5070) |>
  mutate(x = st_coordinates(geometry)[,1], y = st_coordinates(geometry)[,2]) |> st_drop_geometry()
station_xy = to_5070(station_cc)
pretrain_xy = to_5070(pretrain_cc)

saveRDS(list(conus_cc = conus_cc, station_cc = station_cc, pretrain_cc = pretrain_cc,
             station_xy = station_xy, pretrain_xy = pretrain_xy,
             PC_conus = PC_conus, PC_station = PC_station, PC_pretrain = PC_pretrain,
             conus_dist_euclid_station_N95 = conus_dist_euclid_station_N95,
             conus_dist_maha_station_N95 = conus_dist_maha_station_N95,
             conus_dist_euclid_pretrain_N95 = conus_dist_euclid_pretrain_N95,
             nn_results = nn_results, box_results = box_results, occ_pc_results = occ_pc_results,
             raw_occ_summary = raw_occ_summary, dens_results_df = dens_results_df,
             dens_cache = dens_cache, scree_df = scree_df, n90 = n90, n95 = n95,
             participation_ratio = participation_ratio, smd = smd, na_report = na_report,
             unsampled_mask = unsampled_mask),
        file.path(cache_dir, "coverage_results.rds"))

message("== Stage 1 (compute) complete. Run 6_16b for figures/table. ==")
