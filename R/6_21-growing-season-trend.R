##############################################################
# Title: Growing-season (JJA) mean soil moisture trend, 1981-2025
# Description:
#   For each depth, computes the area-weighted CONUS-mean VWC for every
#   June-July-August day in the year-frozen ensemble-median archive, then
#   averages to one JJA value per year. Plots the resulting annual JJA-mean
#   time series with a linear trend line (+ Mann-Kendall/Sen's slope as a
#   robust, non-parametric cross-check). Daily spatial means are cached to
#   CSV (resume-safe) since scanning ~8,000+ daily rasters is the expensive
#   step; the trend itself is trivial once those are in hand.
##############################################################

suppressPackageStartupMessages({
  library(terra); library(tidyverse); library(glue); library(furrr); library(patchwork)
})

repo = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
figs_dir = glue("{repo}/figs"); tabs_dir = glue("{repo}/tables")
cache_dir = glue("{repo}/cache/growing-season-trend"); dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

dirs = c(shallow = "/data/ssd3/soil-moisture-ml-inference/ensemble-smoothed-daily-shallow-yearfrozen/median",
         middle  = "/data/ssd3/soil-moisture-ml-inference/ensemble-smoothed-daily-middle-yearfrozen/median")
depth_lab = c(shallow = "Shallow (0-10 cm)", middle = "Mid-depth (10-50 cm)")

# ---- area weights (WGS84 grid -> cell area varies with latitude); computed
# once from a template raster and reused for every day (identical grid).
# IMPORTANT: extract to a plain numeric vector, NOT a SpatRaster -- terra
# SpatRaster objects hold a non-exportable C++ pointer ('externalptr') and
# cannot be serialized to parallel worker processes (this crashed the first
# attempt: "non-exportable reference... externalptr" from furrr/future).
template = rast(list.files(dirs[["shallow"]], pattern = "^vwc_\\d{4}-\\d{2}-\\d{2}\\.tif$", full.names = TRUE)[1])
w_vals = as.numeric(values(cellSize(template, unit = "km"))[, 1])

daily_mean_cached = function(path, w_vals) {
  r = rast(path)
  v = as.numeric(values(r)[, 1])
  stats::weighted.mean(v, w_vals, na.rm = TRUE)
}

compute_depth = function(dep) {
  dir_in = dirs[[dep]]
  files_tbl = tibble(path = list.files(dir_in, pattern = "^vwc_\\d{4}-\\d{2}-\\d{2}\\.tif$", full.names = TRUE)) |>
    mutate(date = as.Date(str_extract(basename(path), "\\d{4}-\\d{2}-\\d{2}")),
           yr = year(date), mo = month(date)) |>
    filter(mo %in% 6:8) |>
    arrange(date)

  # keep only FULL JJA years (all 3 months present, no gaps) so partial end-years don't bias the trend
  yr_counts = files_tbl |> count(yr)
  full_years = yr_counts |> filter(n >= 90) |> pull(yr)   # 92 nominal; allow tiny tolerance for missing days
  files_tbl = files_tbl |> filter(yr %in% full_years)
  message(glue("[{dep}] {nrow(files_tbl)} JJA daily files across {length(full_years)} full years ({min(full_years)}-{max(full_years)})"))

  cache_csv = glue("{cache_dir}/{dep}_jja_daily_means.csv")
  done = if (file.exists(cache_csv)) read_csv(cache_csv, show_col_types = FALSE) else tibble(date = as.Date(character()), mean_vwc = numeric())
  todo = files_tbl |> filter(!date %in% done$date)
  message(glue("[{dep}] {nrow(todo)} days remaining to compute (", nrow(files_tbl) - nrow(todo), " cached)"))

  if (nrow(todo) > 0) {
    plan(multisession, workers = min(32, max(1, parallel::detectCores() - 4)))
    new_vals = future_map_dbl(todo$path, daily_mean_cached, w_vals = w_vals,
                               .options = furrr_options(seed = NULL, globals = c("w_vals"), packages = "terra"),
                               .progress = TRUE)
    plan(sequential)
    new_rows = tibble(date = todo$date, mean_vwc = new_vals)
    done = bind_rows(done, new_rows) |> distinct(date, .keep_all = TRUE) |> arrange(date)
    write_csv(done, cache_csv)
  }

  done |> filter(date %in% files_tbl$date) |> mutate(depth = dep, yr = year(date))
}

daily_all = map_dfr(names(dirs), compute_depth)

# ---- annual JJA mean per depth ----
annual = daily_all |>
  group_by(depth, yr) |>
  summarise(n_days = n(), jja_mean = mean(mean_vwc, na.rm = TRUE), .groups = "drop") |>
  filter(n_days >= 90) |>
  arrange(depth, yr)
write_csv(annual, glue("{tabs_dir}/growing_season_jja_trend.csv"))

# ---- trend stats: OLS + Mann-Kendall / Sen's slope (robust, non-parametric) ----
have_trend_pkg = requireNamespace("trend", quietly = TRUE)
trend_stats = map_dfr(names(dirs), function(dep) {
  d = annual |> filter(depth == dep)
  ols = lm(jja_mean ~ yr, data = d)
  ols_s = summary(ols)
  row = tibble(depth = dep, n_years = nrow(d),
               ols_slope_per_decade = coef(ols)[2] * 10, ols_p = ols_s$coefficients[2, 4], ols_r2 = ols_s$r.squared)
  if (have_trend_pkg) {
    mk = trend::mk.test(d$jja_mean)
    sens = trend::sens.slope(d$jja_mean)
    row = row |> mutate(mk_p = mk$p.value, sens_slope_per_decade = as.numeric(sens$estimates) * 10)
  }
  row
})
write_csv(trend_stats, glue("{tabs_dir}/growing_season_jja_trend_stats.csv"))
cat("\n=== JJA trend statistics ===\n")
print(as.data.frame(trend_stats |> mutate(across(where(is.numeric), \(x) round(x, 5)))))

# ---- figure ----
annual2 = annual |> mutate(depth_lab = factor(depth_lab[depth], levels = depth_lab))
p = ggplot(annual2, aes(x = yr, y = jja_mean)) +
  geom_line(color = "grey50", linewidth = 0.4) +
  geom_point(color = "#341539", size = 2) +
  geom_smooth(method = "lm", formula = y ~ x, color = "#D55E00", fill = "#D55E00", alpha = 0.15) +
  facet_wrap(~depth_lab, scales = "free_y") +
  labs(title = "Growing-Season (June-August) Mean Soil Moisture, 1981-2025",
       subtitle = "CONUS area-weighted daily mean, averaged over June-July-August each year",
       x = "Year", y = expression("Mean JJA VWC (m"^3*" m"^-3*")")) +
  theme_bw(base_size = 14) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, size = 11),
        strip.text = element_text(face = "bold", size = 13))
ggsave(glue("{figs_dir}/growing_season_jja_trend.png"), p, width = 12, height = 5.5, dpi = 300, bg = "white")
cat(glue("\nWrote figs/growing_season_jja_trend.png\n"))
cat(glue("Wrote tables/growing_season_jja_trend.csv, growing_season_jja_trend_stats.csv\n"))
