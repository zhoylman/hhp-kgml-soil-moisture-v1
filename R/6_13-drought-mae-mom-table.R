##############################################################
# Title: Manuscript drought-class MAE table (method-of-moments, non-augmented)
# Description:
#   Renders the KGML-vs-SPoRT-LIS per-class MAE gt table for the manuscript
#   using the METHOD-OF-MOMENTS, NON-AUGMENTED SSMI (see Section 2.13: each
#   site's KGML anomaly comes from a single held-out fold, so the ensemble
#   variance augmentation of Section 2.12 does not apply; all sources use the
#   climatological variance alone). Reads the MoM columns produced by 6_12
#   (drought_class_mae_varaug_{shallow,middle}.csv), both depths.
#   Style matches 6_5's drought_class_mae figure.
##############################################################

suppressPackageStartupMessages({ library(tidyverse); library(glue); library(gt) })
try(chromote::set_chrome_args(c("--no-sandbox","--disable-gpu","--disable-dev-shm-usage","--headless=new")), silent = TRUE)

repo = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
tables_dir = glue("{repo}/tables"); figs_dir = glue("{repo}/figs")
labs11 = c("D4","D3","D2","D1","D0","Neutral","W0","W1","W2","W3","W4")

tab = bind_rows(
  read_csv(glue("{tables_dir}/drought_class_mae_varaug_shallow.csv"), show_col_types = FALSE),
  read_csv(glue("{tables_dir}/drought_class_mae_varaug_middle.csv"),  show_col_types = FALSE)) |>
  transmute(depth = factor(depth, levels = c("Shallow","Middle")),
            class = factor(class, levels = labs11),
            n, MAE_KGML = KGML_mom, MAE_SPoRT = SPoRT_mom) |>
  arrange(depth, class)
write_csv(tab, glue("{tables_dir}/drought_class_mae_mom.csv"))

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
if (length(kw)) g = g |> tab_style(cell_text(weight = "bold", color = "#4B0092"),
                                    cells_body(columns = MAE_KGML, rows = kw))
gtsave(g, glue("{figs_dir}/drought_class_mae_kgml_vs_sport_mom.png"), expand = 20, zoom = 2.5, vwidth = 450)

cat(glue("Wrote figs/drought_class_mae_kgml_vs_sport_mom.png + tables/drought_class_mae_mom.csv\n"))
cat(glue("KGML beats SPoRT in {length(kw)}/{nrow(tab)} class-depth cells\n"))
print(as.data.frame(tab |> mutate(across(c(MAE_KGML,MAE_SPoRT), \(x) round(x,3)))))
