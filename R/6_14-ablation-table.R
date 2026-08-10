##############################################################
# Title: SI Table -- Effect of SPoRT-LIS Pretraining (no-pretrain ablation)
# Description:
#   Compares the production (pretrained-then-fine-tuned) KGML k-fold models
#   against an ablation trained with the IDENTICAL recipe (30 epochs, same
#   two-phase LR schedule, same loss) but WITHOUT loading SPoRT-LIS pretrained
#   weights (random init instead). Both are evaluated with the same point-
#   based center-keep inference pipeline (py/exp_point_centerkeep_eval.py,
#   keep_len=60/stride=30/sigma=12, matching R/5_2's gridded stitching) so the
#   comparison is apples-to-apples even though absolute values differ slightly
#   from the canonical gridded kfold_validation.csv (see sanity check: r=0.91
#   correlation, ~0.03 KGE median offset from a window-phase anchoring
#   difference -- irrelevant here since it affects both arms equally).
#   Style matches R/6_6's Table 2 (K-fold cross-validation: KGML vs SPoRT-LIS).
##############################################################

suppressPackageStartupMessages({ library(tidyverse); library(glue); library(gt) })
try(chromote::set_chrome_args(c("--no-sandbox","--disable-gpu","--disable-dev-shm-usage","--headless=new")), silent = TRUE)

repo = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
tables_dir = glue("{repo}/tables"); figs_dir = glue("{repo}/figs")
pt_root = glue("{repo}/cache/ablation-pointeval")

load_set = function(depth, model_set) {
  f = glue("{pt_root}/{depth}_{model_set}/point_eval_1_2_3_4_5_6_7_8_9_10.csv")
  if (!file.exists(f)) f = Sys.glob(glue("{pt_root}/{depth}_{model_set}/point_eval_*.csv"))[1]
  read_csv(f, show_col_types = FALSE) |> mutate(depth = str_to_title(depth), site_id = as.character(site_id))
}

both = bind_rows(
  lapply(c("shallow","middle"), function(dep) {
    pre = load_set(dep, "production")  |> filter(n_obs >= 365) |> select(site_id, depth, KGE, r, pbias)
    nop = load_set(dep, "ablation")    |> filter(n_obs >= 365) |> select(site_id, depth, KGE, r, pbias)
    inner_join(pre, nop, by = c("site_id","depth"), suffix = c("_pre","_nop"))
  })
)

# Same sign convention as R/6_6 (Tables 1 & 2): pct(k,s) = (k-s)/abs(s)*100,
# k = model of interest (pretrained), s = baseline (no-pretrain). Negative
# bias_imp means pretrained has LOWER |%bias| (i.e. improved), matching the
# established convention (e.g. Table 2's "-40%" for KGML vs SPoRT-LIS bias).
pct = function(k, s) round((k - s) / abs(s) * 100)

A = both |>
  group_by(depth) |>
  summarise(n = n(),
            KGE_pre = median(KGE_pre, na.rm=TRUE), KGE_nop = median(KGE_nop, na.rm=TRUE),
            r_pre   = median(r_pre,   na.rm=TRUE), r_nop   = median(r_nop,   na.rm=TRUE),
            bias_pre = median(abs(pbias_pre), na.rm=TRUE), bias_nop = median(abs(pbias_nop), na.rm=TRUE),
            .groups = "drop") |>
  mutate(depth = factor(depth, levels = c("Shallow","Middle")),
         KGE_imp  = pct(KGE_pre, KGE_nop),
         r_imp    = pct(r_pre, r_nop),
         bias_imp = pct(bias_pre, bias_nop)) |>   # negative = pretrained has LOWER bias (improved)
  arrange(depth)
write_csv(A, glue("{tables_dir}/ablation_nopretrain_pointeval.csv"))

g = A |> gt(rowname_col = "depth") |>
  tab_header(title = md("**Effect of SPoRT-LIS Pretraining on KGML Skill**"),
             subtitle = "Point-based center-keep evaluation · identical recipe, pretrained vs. random-init") |>
  tab_spanner("KGE", c(KGE_pre, KGE_nop, KGE_imp)) |>
  tab_spanner("Pearson r", c(r_pre, r_nop, r_imp)) |>
  tab_spanner(md("|% Bias|"), c(bias_pre, bias_nop, bias_imp)) |>
  cols_label(n = "Sites", KGE_pre = "Pretrained", KGE_nop = "No-Pretrain", KGE_imp = "% Improvement",
             r_pre = "Pretrained", r_nop = "No-Pretrain", r_imp = "% Improvement",
             bias_pre = "Pretrained", bias_nop = "No-Pretrain", bias_imp = "% Improvement") |>
  fmt_number(c(KGE_pre, KGE_nop, r_pre, r_nop), decimals = 2) |>
  fmt_number(c(bias_pre, bias_nop), decimals = 1) |>
  fmt_number(c(KGE_imp, r_imp, bias_imp), decimals = 0, force_sign = TRUE, pattern = "{x}%") |>
  tab_style(cell_text(weight = "bold", color = "#4B0092"), cells_body(columns = c(KGE_imp, r_imp, bias_imp))) |>
  tab_style(cell_text(weight = "bold"), cells_stub()) |>
  opt_table_outline() |>
  tab_options(table.font.size = px(13), data_row.padding = px(5), column_labels.font.weight = "bold")
gtsave(g, glue("{figs_dir}/summary_table_ablation_nopretrain.png"), expand = 30, zoom = 2.5, vwidth = 1050)

cat(glue("Wrote figs/summary_table_ablation_nopretrain.png + tables/ablation_nopretrain_pointeval.csv\n"))
print(as.data.frame(A |> mutate(across(where(is.numeric), \(x) round(x,3)))))
