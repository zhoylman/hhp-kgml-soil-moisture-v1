##############################################################
# Title: Covariate-space coverage -- simple nested min/max range mask
# Description:
#   Simplified replacement for the PCA-based coverage metrics in
#   R/6_16-covariate-coverage-analysis.R. No PCA, no distance metrics, no
#   effective-dimensionality framing -- just: for each of the 31 raw static
#   covariates, take the station (or pretraining) min/max, and flag any
#   CONUS cell that falls outside that range on ANY feature. Features are
#   added one at a time (nested AND-mask), so the excluded/"unsampled" area
#   fraction is monotonically non-decreasing as more features are added --
#   directly showing which features are most restrictive and how coverage
#   degrades as the covariate space is built up dimension by dimension.
#   Reuses the cached raw (untransformed) covariate matrices from
#   R/6_16-covariate-coverage-analysis.R (cache/covariate-coverage/coverage_results.rds)
#   -- no re-extraction needed.
##############################################################

suppressPackageStartupMessages({ library(tidyverse); library(glue); library(gt); library(patchwork) })

repo      = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
figs_dir  = glue("{repo}/figs")
tabs_dir  = glue("{repo}/tables")

r = readRDS(glue("{repo}/cache/covariate-coverage/coverage_results.rds"))
conus_cc = r$conus_cc; station_cc = r$station_cc; pretrain_cc = r$pretrain_cc

cov_cols = setdiff(names(conus_cc), c("x","y","latitude","longitude"))
cat("n covariates:", length(cov_cols), "\n")
cat("CONUS cells:", nrow(conus_cc), " | stations:", nrow(station_cc), " | pretrain:", nrow(pretrain_cc), "\n")

# ---- per-feature marginal exclusion (each feature's OWN range, independent of others) ----
marginal_excl = function(ref_df, label) {
  map_dfr(cov_cols, function(v) {
    lo = min(ref_df[[v]], na.rm = TRUE); hi = max(ref_df[[v]], na.rm = TRUE)
    out = conus_cc[[v]] < lo | conus_cc[[v]] > hi
    tibble(variable = v, ref = label, lo = lo, hi = hi,
           pct_conus_outside = round(100 * mean(out, na.rm = TRUE), 3))
  })
}
marg = bind_rows(marginal_excl(station_cc, "Stations"), marginal_excl(pretrain_cc, "Pretraining")) |>
  arrange(desc(pct_conus_outside))
write_csv(marg, glue("{tabs_dir}/covariate_coverage_simple_marginal.csv"))
cat("\nTop 8 single-feature exclusions (marginal, not nested):\n")
print(marg |> filter(ref == "Stations") |> head(8) |> select(variable, pct_conus_outside))

# ---- nested AND-mask: add features one at a time, most-restrictive first ----
# Order = descending marginal exclusion (station-based) so the curve shows the
# worst offenders first; this is an arbitrary but transparent, stated choice --
# also report a second run with reverse order to show the order-sensitivity
# (final all-31 number is IDENTICAL regardless of order; only the interior
# shape of the curve depends on ordering).
nested_curve = function(ref_df, label, order_vars) {
  in_range = rep(TRUE, nrow(conus_cc))
  map_dfr(seq_along(order_vars), function(k) {
    v = order_vars[k]
    lo = min(ref_df[[v]], na.rm = TRUE); hi = max(ref_df[[v]], na.rm = TRUE)
    in_range <<- in_range & (conus_cc[[v]] >= lo & conus_cc[[v]] <= hi)
    tibble(ref = label, k = k, variable_added = v,
           pct_conus_outside_cumulative = round(100 * mean(!in_range, na.rm = TRUE), 3))
  })
}
order_station_worst_first = marg |> filter(ref == "Stations") |> pull(variable)
order_pretrain_worst_first = marg |> filter(ref == "Pretraining") |> arrange(desc(pct_conus_outside)) |> pull(variable)

nested = bind_rows(
  nested_curve(station_cc, "Stations", order_station_worst_first),
  nested_curve(pretrain_cc, "Pretraining", order_pretrain_worst_first)
)
write_csv(nested, glue("{tabs_dir}/covariate_coverage_simple_nested.csv"))

# order-sensitivity check: same features, reverse order -> final (k=31) value must match
nested_rev = bind_rows(
  nested_curve(station_cc, "Stations (reverse order)", rev(order_station_worst_first)),
  nested_curve(pretrain_cc, "Pretraining (reverse order)", rev(order_pretrain_worst_first))
)
final_check = bind_rows(nested, nested_rev) |> group_by(ref) |> filter(k == max(k)) |>
  select(ref, pct_conus_outside_cumulative)
cat("\nFinal (all-31-feature) excluded-area fraction, forward vs. reverse ordering (must match within reference set):\n")
print(final_check)

headline = nested |> group_by(ref) |> filter(k == max(k)) |>
  transmute(reference = ref, n_features = k, pct_conus_area_unsampled = pct_conus_outside_cumulative)
write_csv(headline, glue("{tabs_dir}/covariate_coverage_simple_headline.csv"))
cat("\n=== HEADLINE: % of CONUS area outside the min/max range spanned by ALL 31 station covariates simultaneously ===\n")
print(as.data.frame(headline))

# ---- figure: nested exclusion curve (station vs. pretraining), features added worst-first ----
p1 = ggplot(nested, aes(x = k, y = pct_conus_outside_cumulative, color = ref)) +
  geom_line(linewidth = 1.1) + geom_point(size = 1.8) +
  scale_color_manual(values = c("Stations" = "#D55E00", "Pretraining" = "#0072B2")) +
  labs(title = "Cumulative CONUS area excluded by\nnested min/max covariate range mask",
       subtitle = "Features added one at a time, most-restrictive-first (per reference set)",
       x = "Number of covariates included in the mask (nested, cumulative)",
       y = "% of CONUS area outside the range box", color = NULL) +
  theme_bw(base_size = 14) + theme(legend.position = "bottom")

p2 = ggplot(marg |> filter(ref == "Stations") |> mutate(variable = fct_reorder(variable, pct_conus_outside)),
            aes(x = pct_conus_outside, y = variable)) +
  geom_col(fill = "#D55E00") +
  labs(title = "Single-feature (marginal) CONUS area outside station range",
       subtitle = "Each bar independent of the others -- NOT the nested/cumulative mask",
       x = "% of CONUS area outside this feature's station min/max", y = NULL) +
  theme_bw(base_size = 11)

combined = p1 / p2 + patchwork::plot_layout(heights = c(1, 1.3))
ggsave(glue("{figs_dir}/covariate_coverage_simple_mask.png"), combined, width = 9, height = 11, dpi = 300, bg = "white")

# ---- small summary table ----
tab = headline |>
  left_join(marg |> group_by(ref) |> summarise(worst_single_feature = variable[which.max(pct_conus_outside)],
                                                 worst_single_feature_pct = max(pct_conus_outside)),
            by = c("reference" = "ref"))
gt_tab = tab |> gt() |>
  tab_header(title = md("**Simple nested min/max coverage mask**"),
             subtitle = "All 31 static covariates; area-outside = fails the range test on >=1 feature") |>
  cols_label(reference = "Reference set", n_features = "N features", pct_conus_area_unsampled = "% CONUS area unsampled (all 31, nested)",
             worst_single_feature = "Most restrictive single feature", worst_single_feature_pct = "% CONUS excluded by that feature alone") |>
  fmt_number(c(pct_conus_area_unsampled, worst_single_feature_pct), decimals = 2) |>
  opt_table_outline() |> tab_options(table.font.size = px(13), data_row.padding = px(5), column_labels.font.weight = "bold")
gtsave(gt_tab, glue("{figs_dir}/covariate_coverage_simple_mask_table.png"), expand = 30, zoom = 2.5, vwidth = 1100)

cat(glue("\nWrote figs/covariate_coverage_simple_mask.png, figs/covariate_coverage_simple_mask_table.png\n"))
cat(glue("Wrote tables/covariate_coverage_simple_{{marginal,nested,headline}}.csv\n"))
