library(gt)
library(dplyr)
library(readr)

# Headless Chrome needs --no-sandbox in this environment for gtsave() PNG output.
try(chromote::set_chrome_args(c("--no-sandbox", "--disable-gpu",
                                "--disable-dev-shm-usage", "--headless=new")), silent = TRUE)

# ============================================================================
#  Publication summary table: KGML vs SPoRT-LIS soil-moisture accuracy.
#  One gt table combining KGE, Pearson r, and |% bias| — each shown as
#  k-fold KGML, k-fold SPoRT-LIS, the % improvement, and the true
#  out-of-sample (10-fold ensemble) KGML skill at never-seen sites.
#  Reads the tables written by 6_1 (in-fold) and 6_2 (OOS).
# ============================================================================

tables_dir = "/home/zhoylman/hhp-kgml-soil-moisture-v1/tables"
figs_dir   = "/home/zhoylman/hhp-kgml-soil-moisture-v1/figs"

# ---- k-fold medians (KGML vs SPoRT), All sites + UMRB, both depths ----
kf = read_csv(file.path(tables_dir, "skill_summary.csv"), show_col_types = FALSE) |>
  dplyr::filter(region %in% c("All sites", "UMRB Network")) |>
  dplyr::transmute(
    depth, region = ifelse(region == "All sites", "All sites", "UMRB Mesonet"),
    KGE_kgml = KGE_KGML_med,  KGE_sport = KGE_SPoRT_med,
    r_kgml   = r_KGML_med,    r_sport   = r_SPoRT_med,
    bias_kgml = abs_pbias_KGML_med, bias_sport = abs_pbias_SPoRT_med
  )

# ---- OOS medians (KGML ensemble at never-seen sites) ----
oos_raw = read_csv(file.path(tables_dir, "per_site_oos_all.csv"), show_col_types = FALSE)
oos_med = function(df, lbl) df |>
  dplyr::group_by(depth) |>
  dplyr::summarise(region = lbl,
                   KGE_oos  = median(KGE_KGML, na.rm = TRUE),
                   r_oos    = median(r_KGML, na.rm = TRUE),
                   bias_oos = median(abs(pbias_KGML), na.rm = TRUE),
                   .groups = "drop")
oos = dplyr::bind_rows(
  oos_med(oos_raw, "All sites"),
  oos_med(dplyr::filter(oos_raw, network == "UMRB Mesonet"), "UMRB Mesonet")
)

# ---- assemble, compute signed % change (higher better for KGE/r; for |bias|
#      a negative Δ% = bias reduced) ----
tab = dplyr::left_join(kf, oos, by = c("depth", "region")) |>
  dplyr::mutate(
    KGE_delta  = (KGE_kgml  - KGE_sport)  / abs(KGE_sport)  * 100,
    r_delta    = (r_kgml    - r_sport)    / abs(r_sport)    * 100,
    bias_delta = (bias_kgml - bias_sport) / abs(bias_sport) * 100,
    depth  = factor(depth,  levels = c("Shallow", "Middle")),
    region = factor(region, levels = c("All sites", "UMRB Mesonet"))
  ) |>
  dplyr::arrange(depth, region) |>
  dplyr::select(depth, region,
                KGE_kgml, KGE_sport, KGE_delta, KGE_oos,
                r_kgml,   r_sport,   r_delta,   r_oos,
                bias_kgml, bias_sport, bias_delta, bias_oos)

readr::write_csv(tab, file.path(tables_dir, "summary_kfold_vs_oos.csv"))

delta_cols = c("KGE_delta", "r_delta", "bias_delta")
oos_cols   = c("KGE_oos", "r_oos", "bias_oos")

gt_tbl = tab |>
  gt::gt(rowname_col = "region", groupname_col = "depth") |>
  gt::tab_header(
    title    = gt::md("**KGML vs. SPoRT-LIS soil-moisture accuracy**"),
    subtitle = gt::md("Median skill from 10-fold cross-validation (per-fold, same sites) with the % change of KGML over SPoRT-LIS, alongside true out-of-sample skill (10-fold ensemble at never-seen sites).")
  ) |>
  # one spanner per metric
  gt::tab_spanner(label = "KGE",        columns = c(KGE_kgml, KGE_sport, KGE_delta, KGE_oos)) |>
  gt::tab_spanner(label = "Pearson r",  columns = c(r_kgml, r_sport, r_delta, r_oos)) |>
  gt::tab_spanner(label = gt::md("|% Bias|"), columns = c(bias_kgml, bias_sport, bias_delta, bias_oos)) |>
  gt::cols_label(
    KGE_kgml = "KGML", KGE_sport = "SPoRT", KGE_delta = gt::md("&Delta;%"), KGE_oos = "OOS",
    r_kgml   = "KGML", r_sport   = "SPoRT", r_delta   = gt::md("&Delta;%"), r_oos   = "OOS",
    bias_kgml= "KGML", bias_sport= "SPoRT", bias_delta= gt::md("&Delta;%"), bias_oos= "OOS"
  ) |>
  gt::fmt_number(columns = c(KGE_kgml, KGE_sport, KGE_oos, r_kgml, r_sport, r_oos), decimals = 2) |>
  gt::fmt_number(columns = c(bias_kgml, bias_sport, bias_oos), decimals = 1) |>
  gt::fmt_number(columns = tidyselect::all_of(delta_cols), decimals = 0,
                 force_sign = TRUE, pattern = "{x}%") |>
  # accent the OOS columns (different, independent site set) and bold the Δ%
  gt::tab_style(style = gt::cell_fill(color = "#F2EEF8"),
                locations = gt::cells_body(columns = tidyselect::all_of(oos_cols))) |>
  gt::tab_style(style = gt::cell_text(weight = "bold", color = "#4B0092"),
                locations = gt::cells_body(columns = tidyselect::all_of(delta_cols))) |>
  gt::tab_style(style = gt::cell_text(weight = "bold"),
                locations = gt::cells_row_groups()) |>
  # footnotes
  gt::tab_footnote(gt::md("**OOS:** 10-fold *ensemble* at sites in no fold's train or validation set (never-seen). Shallow *n* = 225 (UMRB 183); Middle *n* = 240 (UMRB 181). No SPoRT-LIS baseline at most of these (post-2022 stations; SPoRT sims end 2022)."),
                   locations = gt::cells_column_labels(columns = KGE_oos)) |>
  gt::tab_footnote(gt::md("Higher is better for KGE and *r*; **lower** is better for |% Bias| (so a negative &Delta;% = bias reduced)."),
                   locations = gt::cells_column_spanners(spanners = "|% Bias|")) |>
  gt::tab_source_note(gt::md(
    "Evaluated against in-situ obs with frozen-soil periods excluded. Depth-matched per model: shallow = 0–10 cm; middle KGML vs 10–50 cm obs, SPoRT-LIS vs 10–40 cm obs (each model's native band). k-fold = each site scored against its held-out fold; OOS = full 10-fold ensemble. &Delta;% is the paired k-fold change of KGML over SPoRT-LIS."
  )) |>
  gt::opt_all_caps() |>
  gt::opt_table_outline() |>
  gt::tab_options(
    table.font.size       = gt::px(13),
    data_row.padding      = gt::px(5),
    column_labels.padding = gt::px(4),
    heading.title.font.size    = gt::px(17),
    heading.subtitle.font.size = gt::px(12)
  )

out_png = file.path(figs_dir, "summary_table_kfold_vs_oos.png")
gt::gtsave(gt_tbl, out_png, expand = 30, zoom = 2.5, vwidth = 1200)
message("Wrote ", out_png, " and ", file.path(tables_dir, "summary_kfold_vs_oos.csv"))
