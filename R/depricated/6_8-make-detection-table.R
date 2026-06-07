library(gt)
library(dplyr)
library(readr)

# Headless Chrome needs --no-sandbox for gtsave() PNG output here.
try(chromote::set_chrome_args(c("--no-sandbox", "--disable-gpu",
                                "--disable-dev-shm-usage", "--headless=new")), silent = TRUE)

# ============================================================================
#  gt figure: drought-class error (MAE) by class, KGML vs SPoRT-LIS (from 6_7).
#  MAE = mean |SMI_mod - SMI_obs| within each observed (truth) class (the paper
#  Table-3 definition). All 11 USDM classes, per depth, matched 2005-2022 sample.
# ============================================================================
tables_dir = "/home/zhoylman/hhp-kgml-soil-moisture-v1/tables"
figs_dir   = "/home/zhoylman/hhp-kgml-soil-moisture-v1/figs"
labs11 = c("D4","D3","D2","D1","D0","Neutral","W0","W1","W2","W3","W4")

tab = read_csv(file.path(tables_dir, "drought_class_mae_kgml_vs_sport.csv"), show_col_types = FALSE) |>
  filter(season == "May-Oct") |>
  mutate(depth = factor(depth, levels = c("Shallow", "Middle")),
         class = factor(class, levels = labs11)) |>
  arrange(depth, class) |>
  select(depth, class, n, MAE_KGML, MAE_SPoRT)

kgml_win  = which(tab$MAE_KGML  < tab$MAE_SPoRT)
sport_win = which(tab$MAE_SPoRT < tab$MAE_KGML)

g = tab |>
  gt(rowname_col = "class", groupname_col = "depth") |>
  tab_header(
    title    = md("**Drought-class error (MAE): KGML vs. SPoRT-LIS**"),
    subtitle = md("Mean absolute SMI error within each observed drought/wetness class (lower = better). Matched 2005–2022 sample, May–Oct; KGML uses held-out-fold predictions.")
  ) |>
  tab_spanner(label = "MAE  (|SMIₘₒ𝒹 − SMIₒᵦₛ|)", columns = c(MAE_KGML, MAE_SPoRT)) |>
  cols_label(n = "Obs (n)", MAE_KGML = "KGML", MAE_SPoRT = "SPoRT-LIS") |>
  fmt_number(columns = c(MAE_KGML, MAE_SPoRT), decimals = 3) |>
  fmt_number(columns = n, decimals = 0, sep_mark = ",") |>
  tab_style(style = cell_text(weight = "bold", color = "#4B0092"),
            locations = cells_body(columns = MAE_KGML,  rows = kgml_win)) |>
  tab_style(style = cell_text(weight = "bold"),
            locations = cells_body(columns = MAE_SPoRT, rows = sport_win)) |>
  tab_source_note(md("SMI = beta-standardized Soil Moisture Index (±15-day climatology, ≤30 yr, ≥6 yr min, truncated ±2), each product vs. its own climatology. Classes follow the U.S. Drought Monitor (D4 = Exceptional Drought … W4 = Exceptionally Wet) defined by SMIₒᵦₛ. Both models scored on the matched site-days where obs, KGML and SPoRT all exist (2005–2022). Shallow = 0–10 cm; Middle = KGML 10–50 cm vs SPoRT 10–40 cm. Bold = lower (better).")) |>
  cols_width(class ~ px(115), n ~ px(78), MAE_KGML ~ px(72), MAE_SPoRT ~ px(92)) |>
  opt_table_outline() |>
  tab_options(table.font.size = px(12), data_row.padding = px(3),
              table.width = px(380), column_labels.font.weight = "bold",
              row_group.font.weight = "bold",
              heading.title.font.size = px(15), heading.subtitle.font.size = px(10),
              source_notes.font.size = px(9), footnotes.font.size = px(9))

out = file.path(figs_dir, "drought_class_mae_kgml_vs_sport.png")
gtsave(g, out, expand = 20, zoom = 2.5, vwidth = 450)
message("Wrote ", out)
