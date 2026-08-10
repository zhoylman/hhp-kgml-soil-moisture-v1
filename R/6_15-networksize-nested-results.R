##############################################################
# Title: Network-size experiment (REDESIGNED per reviewer feedback)
# Description:
#   Fixed validation per fold (existing k-fold holdout, unchanged across
#   sizes), nested training prefixes (100/200/300/400/500/600, plus the full
#   pool size using the existing production models), 10 replicate folds.
#   Aggregates per-fold median KGE at each size, fits LINEAR and SATURATING
#   (asymptotic) models to the fold-level medians, and compares fit -- the
#   direct answer to whether a plateau is statistically detectable.
##############################################################

suppressPackageStartupMessages({ library(tidyverse); library(glue); library(hydroGOF); library(patchwork) })

repo = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
data_root = "/data/ssd2/soil-moisture-ml"
obs_dir   = glue("{data_root}/observations")
figs_dir = glue("{repo}/figs"); tables_dir = glue("{repo}/tables")

nested = read_csv(glue("{repo}/cache/networksize-pointeval/networksize_point_eval.csv"), show_col_types = FALSE) |>
  mutate(site_id = as.character(site_id))

# ---- add the "full pool" point per fold from the EXISTING production models (point-eval already run) ----
full_pool = map_dfr(c("shallow","middle"), function(dep) {
  p = read_csv(glue("{repo}/cache/ablation-pointeval/{dep}_production/point_eval_1_2_3_4_5_6_7_8_9_10.csv"), show_col_types = FALSE) |>
    mutate(depth = dep, site_id = as.character(site_id))
  pool_sizes = map_dfr(1:10, function(f) {
    n = read_csv(glue("{data_root}/split-definitions-kfold-{dep}/train_split_fold_{f}.csv"), show_col_types = FALSE)$site_id |> n_distinct()
    tibble(fold = f, size = n)
  }) |> mutate(size = round(size, -1))   # bucket e.g. 647/648 -> 650 so all 10 folds' "full" point aligns
  p |> left_join(pool_sizes, by = "fold")
}) |> select(depth, fold, size, site_id, n_obs, KGE, r, pbias)

all_pts = bind_rows(nested, full_pool)

# ---- per-fold median KGE at each size (the replicate unit for model fitting) ----
per_fold = all_pts |>
  group_by(depth, fold, size) |>
  summarise(median_kge = median(KGE, na.rm = TRUE), n_sites = n(), .groups = "drop")

write_csv(per_fold, glue("{tables_dir}/networksize_per_fold_medians.csv"))

# ---- aggregate across the 10 fold replicates at each size (for plotting + table) ----
agg = per_fold |>
  group_by(depth, size) |>
  summarise(n_folds = n(), median = median(median_kge, na.rm = TRUE),
            p25 = quantile(median_kge, 0.25, na.rm = TRUE),
            p75 = quantile(median_kge, 0.75, na.rm = TRUE),
            sd = sd(median_kge, na.rm = TRUE), .groups = "drop") |>
  arrange(depth, size)
write_csv(agg, glue("{tables_dir}/networksize_summary.csv"))
print(as.data.frame(agg |> mutate(across(where(is.numeric), \(x) round(x,3)))))

# ---- fit LINEAR vs SATURATING models per depth, on fold-level medians ----
# Primary saturating comparison: log-linear (KGE ~ log(size)) -- the standard,
# always-convergent way to test for a concave/diminishing-returns relationship
# (no iterative fitting required, unlike nls). Also attempt a 3-parameter
# asymptotic nls with MANUAL starting values (more flexible curve shape) as a
# secondary check; report if it fails to converge rather than silently omit it.
fit_compare = map_dfr(c("Shallow","Middle"), function(dep_label) {
  dep = tolower(dep_label)
  d = per_fold |> filter(depth == dep)
  lin     = lm(median_kge ~ size, data = d)
  loglin  = lm(median_kge ~ log(size), data = d)
  asym = tryCatch(
    nls(median_kge ~ Asym + (R0 - Asym) * exp(-exp(lrc) * size), data = d,
        start = list(Asym = max(d$median_kge), R0 = min(d$median_kge), lrc = log(1/300)),
        control = nls.control(maxiter = 200, warnOnly = TRUE)),
    error = function(e) NULL)
  out = bind_rows(
    tibble(depth = dep_label, model = "linear", AIC = AIC(lin), BIC = BIC(lin), R2 = summary(lin)$r.squared),
    tibble(depth = dep_label, model = "log-linear (saturating)", AIC = AIC(loglin), BIC = BIC(loglin), R2 = summary(loglin)$r.squared)
  )
  if (!is.null(asym)) {
    out = bind_rows(out, tibble(depth = dep_label, model = "asymptotic nls (saturating)", AIC = AIC(asym), BIC = BIC(asym),
                                 R2 = 1 - sum(residuals(asym)^2)/sum((d$median_kge - mean(d$median_kge))^2)))
  } else {
    out = bind_rows(out, tibble(depth = dep_label, model = "asymptotic nls (saturating)", AIC = NA, BIC = NA, R2 = NA))
  }
  out
})
write_csv(fit_compare, glue("{tables_dir}/networksize_model_comparison.csv"))
cat("\n=== Linear vs. saturating model comparison (lower AIC/BIC = better fit) ===\n")
print(as.data.frame(fit_compare |> mutate(across(where(is.numeric), \(x) round(x,2)))))

# ---- SPoRT-LIS reference (median raw KGE vs obs, per depth) -- same approach as 6_7 ----
sport_kge_for = function(dep) {
  depth_2     = c(shallow = "Shallow", middle = "Middle")[[dep]]
  sport_depth = c(shallow = "SPoRT_raw_0-10cm", middle = "SPoRT_raw_10-40cm")[[dep]]
  obs_sport = readr::read_csv(glue("{obs_dir}/final-soil-moisture-data-generalized-sport-specific-no-frozen.csv"),
                               show_col_types = FALSE) |>
    filter(generalized_depth == depth_2) |> transmute(site_id = as.character(site_id), date, obs = soil_moisture)
  sport_sims = readr::read_csv(glue("{obs_dir}/observational-sites-raw-sport.csv"), show_col_types = FALSE) |>
    mutate(date = as.Date(time)) |> filter(var == sport_depth) |>
    pivot_longer(cols = -c(var, date, time), names_to = "site_id", values_to = "sport")
  obs_sport |>
    inner_join(sport_sims |> select(site_id, date, sport), by = c("site_id", "date")) |>
    drop_na(obs, sport) |> group_by(site_id) |>
    summarise(KGE = hydroGOF::KGE(sport, obs), .groups = "drop") |>
    pull(KGE) |> median(na.rm = TRUE)
}

# ---- figure: match the aesthetic of the original kge_vs_training_sites_combined.png ----
# (purple points, blue lm-fit line + CI ribbon, dashed green SPoRT-LIS reference,
#  patchwork side-by-side panels, tag_levels "a"/"b")
DEPTH_LEVELS = c("shallow", "middle")
DEPTH_NAMES  = c(shallow = "Shallow Soil Moisture (0-10 cm)", middle = "Mid-depth Soil Moisture (10-50 cm)")

make_panel = function(dep, show_sport = TRUE) {
  d       = per_fold |> filter(depth == dep)
  d_agg   = agg |> filter(depth == dep)

  base = ggplot(d, aes(x = size, y = median_kge)) +
    geom_smooth(method = "lm", formula = y ~ x) +
    geom_point(data = d_agg, aes(x = size, y = median), size = 3, color = "#341539")

  if (show_sport) {
    sport_kge = sport_kge_for(dep)
    message(glue("[{dep}] SPoRT-LIS median KGE = {round(sport_kge,3)}"))
    base = base +
      geom_hline(yintercept = sport_kge, linetype = "dashed", color = "darkgreen", linewidth = 1) +
      annotate("text", x = max(d$size), y = sport_kge - 0.01,
               label = "SPoRT-LIS", color = "darkgreen", hjust = 1, size = 4)
  }

  base +
    labs(title = DEPTH_NAMES[[dep]], x = "Number of Training Sites", y = "Median KGE") +
    theme_bw(base_size = 15) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5),
          axis.title = element_text(face = "bold"), axis.text = element_text(color = "black"))
}

plots = list(make_panel("shallow", show_sport = FALSE), make_panel("middle", show_sport = FALSE))
combined = (plots[[1]] | plots[[2]]) +
  patchwork::plot_annotation(title = "Out-of-Sample KGE vs. Number of Training Sites",
                              tag_levels = "a",
                              theme = theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 17)))
ggsave(glue("{figs_dir}/networksize_nested_linear_vs_saturating.png"), combined, width = 13, height = 5.5, dpi = 300, bg = "white")
cat(glue("\nWrote figs/networksize_nested_linear_vs_saturating.png\n"))
cat(glue("Wrote tables/networksize_summary.csv, networksize_per_fold_medians.csv, networksize_model_comparison.csv\n"))
