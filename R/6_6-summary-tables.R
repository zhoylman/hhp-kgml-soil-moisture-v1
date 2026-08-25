library(tidyverse)
library(glue)
library(gt)

try(chromote::set_chrome_args(c("--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--headless=new")), silent = TRUE)

# ============================================================================
#  CONSUMER — two summary gt tables from the validation CSVs (robust sites):
#    A) k-fold: KGML vs SPoRT-LIS (KGE / r / |% bias|) + % improvement
#    B) out-of-sample: KGML skill by network (independent never-seen sites)
# ============================================================================

repo = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
tables_dir = glue("{repo}/tables"); figs_dir = glue("{repo}/figs")
# Absolute differences, not % improvement -- KGE and r are not ratio-scale
# (arbitrary zero point, unbounded below), so relative change has no stable
# interpretation. |% Bias| already IS a percentage, so its difference is
# reported in percentage points rather than a relative "% of %" change.
diff = function(k, s, digits) round(k - s, digits)

kf  = read_csv(glue("{tables_dir}/kfold_validation.csv"), show_col_types = FALSE) |> filter(robust)
oos = read_csv(glue("{tables_dir}/oos_validation.csv"),   show_col_types = FALSE) |> filter(robust)

# ---- A) k-fold: KGML vs SPoRT-LIS ----
kf_med = function(df, lbl, umrb = FALSE) {
  if (umrb) df = filter(df, network == "UMRB Mesonet")
  df |> group_by(depth) |> summarise(region = lbl, n = n(),
    KGE_kgml = median(KGE, na.rm = TRUE), KGE_sport = median(KGE_sport, na.rm = TRUE),
    r_kgml = median(r, na.rm = TRUE), r_sport = median(r_sport, na.rm = TRUE),
    bias_kgml = median(abs(pbias), na.rm = TRUE), bias_sport = median(abs(pbias_sport), na.rm = TRUE),
    .groups = "drop")
}
A = bind_rows(kf_med(kf, "All sites"), kf_med(kf, "UMRB Mesonet", TRUE)) |>
  mutate(depth = factor(depth, c("Shallow","Middle")), region = factor(region, c("All sites","UMRB Mesonet")),
         KGE_diff = diff(KGE_kgml, KGE_sport, 2), r_diff = diff(r_kgml, r_sport, 2), bias_diff = diff(bias_kgml, bias_sport, 1)) |>
  arrange(depth, region) |>
  transmute(depth, region, n, KGE_kgml, KGE_sport, KGE_diff, r_kgml, r_sport, r_diff, bias_kgml, bias_sport, bias_diff)
write_csv(A, glue("{tables_dir}/summary_kfold.csv"))

gA = A |> gt(rowname_col = "region", groupname_col = "depth") |>
  tab_header(title = md("**K-fold cross-validation: KGML vs. SPoRT-LIS**")) |>
  tab_spanner("KGE", c(KGE_kgml, KGE_sport, KGE_diff)) |>
  tab_spanner("Pearson r", c(r_kgml, r_sport, r_diff)) |>
  tab_spanner(md("|% Bias|"), c(bias_kgml, bias_sport, bias_diff)) |>
  cols_label(n = "Sites", KGE_kgml = "KGML", KGE_sport = "SPoRT-LIS", KGE_diff = md("Δ"),
             r_kgml = "KGML", r_sport = "SPoRT-LIS", r_diff = md("Δ"),
             bias_kgml = "KGML", bias_sport = "SPoRT-LIS", bias_diff = md("Δ (pp)")) |>
  fmt_number(c(KGE_kgml, KGE_sport, r_kgml, r_sport), decimals = 2) |>
  fmt_number(c(bias_kgml, bias_sport), decimals = 1) |>
  fmt_number(c(KGE_diff, r_diff), decimals = 2, force_sign = TRUE) |>
  fmt_number(bias_diff, decimals = 1, force_sign = TRUE) |>
  tab_style(cell_text(weight = "bold", color = "#4B0092"), cells_body(columns = c(KGE_diff, r_diff, bias_diff))) |>
  tab_style(cell_text(weight = "bold"), cells_row_groups()) |>
  opt_table_outline() |>
  tab_options(table.font.size = px(13), data_row.padding = px(5), column_labels.font.weight = "bold",
              row_group.font.weight = "bold")
gtsave(gA, glue("{figs_dir}/summary_table_kfold.png"), expand = 30, zoom = 2.5, vwidth = 1050)

# ---- B) out-of-sample: KGML skill, all OOS sites lumped (per depth) ----
B = oos |> group_by(depth) |>
  summarise(n = n(), KGE = median(KGE, na.rm = TRUE), r = median(r, na.rm = TRUE),
            bias = median(abs(pbias), na.rm = TRUE), .groups = "drop") |>
  mutate(depth = factor(depth, c("Shallow","Middle"))) |> arrange(depth)
write_csv(B, glue("{tables_dir}/summary_oos.csv"))

gB = B |> gt(rowname_col = "depth") |>
  tab_header(title = md("**Out-of-sample validation: KGML at independent sites**")) |>
  cols_label(n = "Sites", KGE = "KGE", r = "Pearson r", bias = md("|% Bias|")) |>
  fmt_number(c(KGE, r), decimals = 2) |> fmt_number(bias, decimals = 1) |>
  opt_table_outline() |>
  tab_options(table.font.size = px(14), data_row.padding = px(6), column_labels.font.weight = "bold")
gtsave(gB, glue("{figs_dir}/summary_table_oos.png"), expand = 30, zoom = 2.5, vwidth = 650)

# ---- C) retrospective transfer (network value): KGML vs SPoRT-LIS at random held-out sites ----
#   Reads the per-site eval CSVs from the fine-tune experiment (py/exp_eval_vs_sport.py):
#   fine-tune on >=2015 obs, evaluate randomly held-out stations during 2005-2010.
exp_root = "/data/ssd2/soil-moisture-ml"
retro_one = function(depth_lbl, depth_dir) {
  read_csv(glue("{exp_root}/results-exp-finetune-{depth_dir}_rand_ev0510/eval_scan_vs_sport.csv"),
           show_col_types = FALSE) |>
    filter(!is.na(KGE_KGML), !is.na(KGE_SPoRT)) |>
    summarise(depth = depth_lbl, n = n(),
      KGE_kgml = median(KGE_KGML, na.rm = TRUE),  KGE_sport = median(KGE_SPoRT, na.rm = TRUE),
      r_kgml   = median(r_KGML,   na.rm = TRUE),  r_sport   = median(r_SPoRT,   na.rm = TRUE),
      bias_kgml = median(abs(pbias_KGML), na.rm = TRUE), bias_sport = median(abs(pbias_SPoRT), na.rm = TRUE))
}
C = bind_rows(retro_one("Shallow", "shallow"), retro_one("Middle", "middle")) |>
  mutate(depth = factor(depth, c("Shallow","Middle")),
         KGE_diff = diff(KGE_kgml, KGE_sport, 2), r_diff = diff(r_kgml, r_sport, 2), bias_diff = diff(bias_kgml, bias_sport, 1)) |>
  arrange(depth) |>
  transmute(depth, n, KGE_kgml, KGE_sport, KGE_diff, r_kgml, r_sport, r_diff, bias_kgml, bias_sport, bias_diff)
write_csv(C, glue("{tables_dir}/summary_retrospective.csv"))

gC = C |> gt(rowname_col = "depth") |>
  tab_header(title = md("**Retrospective Value of KGML**"),
             subtitle = md("Random 30% site holdout · fine-tune ≥ 2015, evaluate 2005–2010")) |>
  tab_spanner("KGE", c(KGE_kgml, KGE_sport, KGE_diff)) |>
  tab_spanner("Pearson r", c(r_kgml, r_sport, r_diff)) |>
  tab_spanner(md("|% Bias|"), c(bias_kgml, bias_sport, bias_diff)) |>
  cols_label(n = "Sites", KGE_kgml = "KGML", KGE_sport = "SPoRT-LIS", KGE_diff = md("Δ"),
             r_kgml = "KGML", r_sport = "SPoRT-LIS", r_diff = md("Δ"),
             bias_kgml = "KGML", bias_sport = "SPoRT-LIS", bias_diff = md("Δ (pp)")) |>
  fmt_number(c(KGE_kgml, KGE_sport, r_kgml, r_sport), decimals = 2) |>
  fmt_number(c(bias_kgml, bias_sport), decimals = 1) |>
  fmt_number(c(KGE_diff, r_diff), decimals = 2, force_sign = TRUE) |>
  fmt_number(bias_diff, decimals = 1, force_sign = TRUE) |>
  tab_style(cell_text(weight = "bold", color = "#4B0092"), cells_body(columns = c(KGE_diff, r_diff, bias_diff))) |>
  opt_table_outline() |>
  tab_options(table.font.size = px(13), data_row.padding = px(5), column_labels.font.weight = "bold")
gtsave(gC, glue("{figs_dir}/summary_table_retrospective.png"), expand = 30, zoom = 2.5, vwidth = 1050)

message("Wrote summary_table_kfold + summary_table_oos + summary_table_retrospective (csv + png)")
print(as.data.frame(A)); cat("\n"); print(as.data.frame(B)); cat("\n"); print(as.data.frame(C))
