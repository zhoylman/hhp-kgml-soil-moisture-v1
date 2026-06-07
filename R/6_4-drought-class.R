library(tidyverse)
library(glue)

# ============================================================================
#  CONSUMER — drought-class error (Hoylman et al. 2024 Table-3 analog).
#  Reads kfold_matched.rds (per site x day SMI). Per OBSERVED (truth) class:
#  MSE & MAE on the standardized anomaly (mean (SMI_mod - SMI_obs)^2 / |.|),
#  pooled over sites. Plus a row-normalized confusion matrix. Robust (>=365 d)
#  sites only.  No extraction — pure consumer of the k-fold dataset.
# ============================================================================

repo = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
source(glue("{repo}/R/sm_eval_utils.R"))
tables_dir = glue("{repo}/tables"); figs_dir = glue("{repo}/figs")
min_days = 365L
labs11 = c("D4","D3","D2","D1","D0","Neutral","W0","W1","W2","W3","W4")

robust = read_csv(glue("{tables_dir}/kfold_validation.csv"), show_col_types = FALSE) |>
  filter(robust) |> distinct(site_id, depth)
m = readRDS(glue("{repo}/cache/datasets/kfold_matched.rds")) |>
  semi_join(robust, by = c("site_id", "depth")) |>
  filter(!is.na(obs_smi), !is.na(kgml_smi)) |>
  mutate(month = as.integer(format(date, "%m")), truth = truth_class)

ordinal_error = function(d, season_label) {
  d |> mutate(truth_class = factor(labs11[truth], levels = labs11)) |>
    group_by(depth, truth, truth_class) |>
    summarise(MSE = mean((kgml_smi - obs_smi)^2), MAE = mean(abs(kgml_smi - obs_smi)),
              n = n(), .groups = "drop") |>
    arrange(depth, truth) |> transmute(season = season_label, depth, truth_class, MSE, MAE, n)
}
ordinal = bind_rows(ordinal_error(filter(m, month %in% 5:10), "May-Oct"),
                    ordinal_error(m, "all non-frozen"))
write_csv(ordinal, glue("{tables_dir}/drought_class_ordinal_error.csv"))

# confusion (row-normalized truth x KGML class), May-Oct
conf = m |> filter(month %in% 5:10) |>
  mutate(pred = smi_to_class(kgml_smi)) |>
  count(depth, truth, pred) |> group_by(depth, truth) |> mutate(frac = n / sum(n)) |> ungroup() |>
  mutate(depth = factor(depth, levels = c("Shallow","Middle")),
         truth = factor(labs11[truth], levels = labs11), pred = factor(labs11[pred], levels = labs11))
p = ggplot(conf, aes(pred, truth, fill = frac)) +
  geom_tile(color = "grey85") + geom_abline(slope = 1, intercept = 0, linewidth = 0.4) +
  facet_wrap(~ depth) + scale_fill_viridis_c(name = "Row\nfraction", limits = c(0, 1)) +
  labs(title = "Drought-class agreement: observed vs KGML (held-out fold)",
       subtitle = "Row-normalized; diagonal = exact match. 11 USDM-style classes.",
       x = "KGML class", y = "Observed class") +
  coord_equal() + theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), plot.title = element_text(face = "bold"),
        strip.text = element_text(face = "bold"))
ggsave(glue("{figs_dir}/drought_class_confusion.png"), p, width = 12, height = 6, dpi = 300, bg = "white")

message(glue("Wrote tables/drought_class_ordinal_error.csv + figs/drought_class_confusion.png"))
print(as.data.frame(ordinal |> filter(season == "May-Oct") |> mutate(MSE = round(MSE,3), MAE = round(MAE,3))))
