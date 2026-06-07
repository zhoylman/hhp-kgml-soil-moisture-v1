library(tidyverse)
library(glue)
library(gt)

try(chromote::set_chrome_args(c("--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--headless=new")), silent = TRUE)

# ============================================================================
#  CONSUMER — KGML vs SPoRT-LIS drought detection + per-class error.
#  Reads kfold_matched.rds (has obs_smi, kgml_smi, sport_smi). On the matched
#  site-days where all three exist (robust sites): (1) per-class MAE table
#  (mean|SMI_mod - SMI_obs| by truth class) and (2) detection contingency
#  (POD/CSI/HSS/FAR) at D0/D1/D2 thresholds. Both as CSV + gt figures.
# ============================================================================

repo = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
tables_dir = glue("{repo}/tables"); figs_dir = glue("{repo}/figs")
labs11 = c("D4","D3","D2","D1","D0","Neutral","W0","W1","W2","W3","W4")
THRESH = c(D0 = -0.5, D1 = -0.8, D2 = -1.3)

robust = read_csv(glue("{tables_dir}/kfold_validation.csv"), show_col_types = FALSE) |>
  filter(robust) |> distinct(site_id, depth)
m = readRDS(glue("{repo}/cache/datasets/kfold_matched.rds")) |>
  semi_join(robust, by = c("site_id", "depth")) |>
  filter(!is.na(obs_smi), !is.na(kgml_smi), !is.na(sport_smi)) |>
  mutate(month = as.integer(format(date, "%m"))) |>
  filter(month %in% 5:10)

# ---- per-class MAE (KGML vs SPoRT), May-Oct ----
class_mae = m |> mutate(truth = truth_class) |>
  group_by(depth, truth) |>
  summarise(MAE_KGML = mean(abs(kgml_smi - obs_smi)),
            MAE_SPoRT = mean(abs(sport_smi - obs_smi)), n = n(), .groups = "drop") |>
  mutate(class = factor(labs11[truth], levels = labs11)) |> arrange(depth, class)
write_csv(class_mae, glue("{tables_dir}/drought_class_mae_kgml_vs_sport.csv"))

# ---- detection metrics ----
det = function(t, p) {
  H = as.numeric(sum(t & p)); M = as.numeric(sum(t & !p)); Fa = as.numeric(sum(!t & p)); C = as.numeric(sum(!t & !p))
  den = (H+M)*(M+C) + (H+Fa)*(Fa+C)
  tibble(n_event = H+M, POD = H/(H+M), FAR = if ((H+Fa)>0) Fa/(H+Fa) else NA,
         CSI = if ((H+M+Fa)>0) H/(H+M+Fa) else NA, HSS = if (den>0) 2*(H*C-M*Fa)/den else NA)
}
detection = bind_rows(lapply(c("Shallow","Middle"), function(dep) {
  d = filter(m, depth == dep)
  bind_rows(lapply(names(THRESH), function(tn) {
    thr = THRESH[[tn]]; ev = d$obs_smi < thr
    bind_rows(det(ev, d$kgml_smi < thr) |> mutate(model = "KGML"),
              det(ev, d$sport_smi < thr) |> mutate(model = "SPoRT-LIS")) |> mutate(threshold = tn)
  })) |> mutate(depth = dep)
}))
write_csv(detection, glue("{tables_dir}/drought_detection_kgml_vs_sport.csv"))

# ---- gt: per-class MAE (KGML vs SPoRT) ----
tab = class_mae |> mutate(depth = factor(depth, levels = c("Shallow","Middle"))) |> arrange(depth, class) |>
  select(depth, class, n, MAE_KGML, MAE_SPoRT)
g = tab |> gt(rowname_col = "class", groupname_col = "depth") |>
  tab_header(title = md("**Drought Classification Error (MAE)**")) |>
  tab_spanner(label = "MAE", columns = c(MAE_KGML, MAE_SPoRT)) |>
  cols_label(n = "Obs (n)", MAE_KGML = "KGML", MAE_SPoRT = "SPoRT-LIS") |>
  fmt_number(c(MAE_KGML, MAE_SPoRT), decimals = 3) |> fmt_number(n, decimals = 0, sep_mark = ",") |>
  cols_width(class ~ px(115), n ~ px(78), MAE_KGML ~ px(72), MAE_SPoRT ~ px(92)) |>
  opt_table_outline() |>
  tab_options(table.font.size = px(12), data_row.padding = px(3), table.width = px(380),
              column_labels.font.weight = "bold", row_group.font.weight = "bold")
kw = which(tab$MAE_KGML < tab$MAE_SPoRT)
if (length(kw)) g = g |> tab_style(cell_text(weight = "bold", color = "#4B0092"), cells_body(columns = MAE_KGML, rows = kw))
gtsave(g, glue("{figs_dir}/drought_class_mae_kgml_vs_sport.png"), expand = 20, zoom = 2.5, vwidth = 450)

message("Wrote drought_class_mae + drought_detection (csv + png)")
print(as.data.frame(detection |> mutate(across(c(POD,FAR,CSI,HSS), \(x) round(x,3)))))
