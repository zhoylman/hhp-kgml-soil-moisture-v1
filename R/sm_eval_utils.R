# ============================================================================
#  sm_eval_utils.R — shared evaluation helpers for the validation scripts
#  (sourced by 6_1-check-results-smoothed.R and 6_2-validation-out-of-sample.R)
#
#  Single source of truth for: per-site metric computation, the cached raster
#  extraction at obs points, and the two figure functions (skill boxplots +
#  spatial difference maps). Depth-agnostic; callers pass paths/labels.
# ============================================================================

# ---- Drought standardization (beta SMI) + classification ------------------
# Standardize a single daily series against ITS OWN moving day-of-year beta
# climatology (the soil-moisture-validation flavor). For each calendar day t,
# pool all values within +/- `window` days (circular, 366) across the last
# `max_years` years, fit a Beta by MLE (MoM start; ecdf fallback), take the
# fitted CDF at each day-t value, and map to a standard normal via qnorm.
# Returns the input df with an added `z` column (NA where the window has
# fewer than `min_n` samples, e.g. <6 yr). Input: df with `date`, `value`.
standardize_doy_beta <- function(df, window = 15L, max_years = 30L, min_n = 31L * 6L) {
  df <- df |>
    dplyr::transmute(date = as.Date(date), value = as.numeric(value)) |>
    dplyr::filter(is.finite(value)) |>
    dplyr::distinct(date, .keep_all = TRUE)
  if (nrow(df) == 0) return(dplyr::mutate(df, z = numeric(0)))

  df  <- dplyr::filter(df, date >= (max(date) - 365.25 * max_years))
  v   <- df$value
  if (any(v > 1, na.rm = TRUE)) v <- v / 100              # % -> fraction
  df$v01  <- pmin(pmax(v, 1e-6), 1 - 1e-6)
  df$yday <- as.integer(format(df$date, "%j"))
  df$z    <- NA_real_

  for (t in sort(unique(df$yday))) {
    dd   <- pmin(abs(df$yday - t), 366L - abs(df$yday - t))   # circular distance
    pool <- df$v01[dd <= window]
    if (length(pool) < min_n || length(unique(pool)) < 3L) next
    m <- mean(pool); vr <- stats::var(pool)
    if (!is.finite(vr) || vr <= 0) next
    th <- max(m * (1 - m) / vr - 1, 2); a0 <- m * th; b0 <- (1 - m) * th
    fit <- try(MASS::fitdistr(pool, function(x, a, b) dbeta(x, a, b, log = TRUE),
                              start = list(a = a0, b = b0)), silent = TRUE)
    tgt <- which(df$yday == t)
    p <- if (!inherits(fit, "try-error"))
           pbeta(df$v01[tgt], fit$estimate[["a"]], fit$estimate[["b"]])
         else stats::ecdf(pool)(df$v01[tgt])
    df$z[tgt] <- stats::qnorm(pmin(pmax(p, 1e-12), 1 - 1e-12))
  }
  dplyr::select(df, date, value, z)
}

# 11-class USDM-style drought classes from a standardized anomaly (z), clamped
# to +/-2. Bin 1 = driest (D4) ... bin 11 = wettest (W4). Returns integer 1..11.
smi_to_class <- function(z) {
  # Thresholds = exact qnorm(percentile) breakpoints from the NSAEM modified
  # USDM drought classification schema (30/20/10/5/2 percentile -> D0-D4),
  # mirrored symmetrically for the wet side (W0-W4). Clamp at +/-3.09 (the
  # 99.9th-percentile guardrail used elsewhere in this pipeline for capped SMI)
  # -- wide enough that it never coincides with the +/-2.054 D4/W4 boundary,
  # so D4/W4 remain reachable (the prior +/-2.0 clamp coincided with the old
  # rounded +/-2.0 boundary and would make D4/W4 structurally empty here).
  z <- pmin(pmax(z, -3.09), 3.09)
  brks <- rev(c(Inf, 2.054, 1.644, 1.281, 0.842, 0.524, -0.524, -0.842, -1.281, -1.644, -2.054, -Inf))
  .bincode(z, breaks = brks, include.lowest = TRUE)
}

# ---- Per-site metrics: KGE, Pearson r, percent bias -----------------------
compute_metrics <- function(df, site_id, fold = "all", min_n = 5) {
  df <- df %>% dplyr::filter(!is.na(obs), !is.na(ml))
  tibble::tibble(
    fold    = fold,
    site_id = site_id,
    n       = nrow(df),
    KGE     = if (nrow(df) >= min_n) hydroGOF::KGE(df$ml, df$obs)   else NA_real_,
    r       = if (nrow(df) >= min_n) cor(df$ml, df$obs)            else NA_real_,
    pbias   = if (nrow(df) >= min_n) hydroGOF::pbias(df$ml, df$obs) else NA_real_
  )
}

# ---- Cached extraction of daily rasters at obs points ---------------------
# Reads ONLY the rasters whose date is in `obs_dates` (bit-identical to reading
# all, since downstream joins keep only obs dates), extracts at `meta_xy`
# (a data frame with site_id/longitude/latitude), and returns a long table
# (date, site_id, ml). If `cache_file` is given, the result is cached keyed on
# the requested obs_dates + site_ids and reused when it covers a later request.
extract_at_sites <- function(raster_dir, obs_dates, site_ids, meta_xy,
                             cache_file = NULL, label = "extract") {
  if (!is.null(cache_file) && file.exists(cache_file)) {
    cached <- readRDS(cache_file)
    if (all(obs_dates %in% cached$dates) && all(site_ids %in% cached$sites)) {
      cli::cli_inform("{label}: cache hit ({length(cached$dates)} dates).")
      return(dplyr::filter(cached$smoothed_data, date %in% obs_dates, site_id %in% site_ids))
    }
    cli::cli_inform("{label}: cache miss/stale; re-extracting.")
  }

  files_tbl <- tibble::tibble(path = list.files(raster_dir, full.names = TRUE)) |>
    dplyr::mutate(date = as.Date(stringr::str_extract(basename(path), "\\d{4}-\\d{2}-\\d{2}"))) |>
    dplyr::filter(!is.na(date), date %in% obs_dates) |>
    dplyr::arrange(date)

  if (nrow(files_tbl) == 0) return(tibble::tibble())

  smoothed_stack <- terra::rast(files_tbl$path)
  dates_vec      <- files_tbl$date                       # materialize dates now
  xy             <- as.matrix(meta_xy[, c("longitude", "latitude")])

  ext_df <- terra::extract(smoothed_stack, xy)           # rows = sites, cols = layers
  vals_t <- t(ext_df)

  if (nrow(vals_t) != length(dates_vec))
    stop(glue::glue("Layer/date mismatch: n_layers={nrow(vals_t)} vs n_dates={length(dates_vec)}"))
  if (ncol(vals_t) != nrow(meta_xy))
    stop(glue::glue("Site mismatch: n_sites(vals)={ncol(vals_t)} vs n_sites(meta)={nrow(meta_xy)}"))

  df_wide <- tibble::as_tibble(vals_t, .name_repair = "minimal")
  df_wide <- rlang::set_names(df_wide, meta_xy$site_id)
  df_wide <- tibble::add_column(df_wide, date = dates_vec, .before = 1)
  smoothed_data <- tidyr::pivot_longer(df_wide, cols = -date, names_to = "site_id", values_to = "ml")

  rm(smoothed_stack, ext_df, vals_t, df_wide); gc()

  if (!is.null(cache_file)) {
    dir.create(dirname(cache_file), showWarnings = FALSE, recursive = TRUE)
    # Store the REQUESTED obs_dates as coverage (not just dates found) so a rerun
    # with the same obs doesn't re-extract just because some dates lacked a raster.
    saveRDS(list(smoothed_data = smoothed_data, dates = obs_dates, sites = site_ids), cache_file)
    cli::cli_inform("{label}: cached {nrow(smoothed_data)} rows over {length(obs_dates)} obs dates.")
  }
  smoothed_data
}

# ---- Figure: skill boxplots (KGE / Pearson r / |% bias|), 2 regions -------
plot_skill_boxes_2x2 = function(difference, site_meta, depth_label = NULL, save_path = NULL) {
  suppressPackageStartupMessages({
    library(dplyr); library(tidyr); library(ggplot2); library(forcats); library(purrr)
  })

  umrb_ids = site_meta %>% filter(network == "UMRB Mesonet") %>% pull(site_id) %>% unique()

  long = difference %>%
    filter(Metric %in% c("KGE","r","pbias")) %>%
    select(site_id, Metric, KGML, `SPoRT-LIS`) %>%
    pivot_longer(cols = c(KGML, `SPoRT-LIS`), names_to = "Model", values_to = "Score") %>%
    mutate(
      Model  = factor(Model, levels = c("SPoRT-LIS","KGML")),
      Metric = factor(Metric, levels = c("KGE","r","pbias")),
      # KGE floored at 0; pbias shown in absolute terms (closer to 0 = better)
      Score  = dplyr::case_when(
        Metric == "KGE"   ~ pmax(Score, 0),
        Metric == "pbias" ~ abs(Score),
        TRUE              ~ Score
      )
    )

  panels = bind_rows(
    long %>% mutate(Region = "All sites"),
    long %>% filter(site_id %in% umrb_ids) %>% mutate(Region = "UMRB Network")
  ) %>% mutate(Region = factor(Region, levels = c("All sites","UMRB Network")))

  # Wilcoxon + annotation heights
  wdf = panels %>%
    select(site_id, Region, Metric, Model, Score) %>%
    pivot_wider(names_from = Model, values_from = Score) %>%
    drop_na()

  pvals = wdf %>%
    group_by(Region, Metric) %>%
    summarize(
      p    = tryCatch(stats::wilcox.test(`SPoRT-LIS`, KGML, paired = TRUE)$p.value, error = function(e) NA_real_),
      ymax = max(c(`SPoRT-LIS`, KGML), na.rm = TRUE) + ifelse(unique(Metric) == "KGE", 0.05, 0.03),
      .groups = "drop"
    ) %>%
    mutate(stars = case_when(is.na(p) ~ "ns", p < 0.05 ~ "*", TRUE ~ "ns"),
           x1 = 1, x2 = 2, xmid = 1.5)

  pal = c(`SPoRT-LIS` = "#1AFF1A", KGML = "#4B0092")
  metric_lab = function(x) dplyr::recode(x, r = "Pearson's r", pbias = "|% Bias|")
  lab_fun = labeller(Metric = metric_lab, Region = label_value)

  # Minimal baseline rows to force y>=0 in KGE facets
  kge_zero = panels %>%
    filter(Metric == "KGE") %>%
    distinct(Region, Metric, Model) %>%
    mutate(Score = 0)

  p = ggplot(panels, aes(x = Model, y = Score, fill = Model)) +
    geom_boxplot(width = 0.6, outlier.alpha = 0.15, color = "black", linewidth = 0.3) +
    geom_blank(data = kge_zero) +                                  # <- forces 0 into KGE y-scale
    # independent y per panel: KGE/r (~0-1) and |% bias| (0-100+) are on very
    # different scales, so each metric column gets its own y-axis.
    ggh4x::facet_grid2(rows = vars(Region), cols = vars(Metric),
                       scales = "free_y", independent = "y", labeller = lab_fun) +
    scale_fill_manual(values = pal, guide = "none") +
    scale_color_manual(values = pal, guide = "none") +
    scale_y_continuous(expand
                       = expansion(mult = c(0.02, 0.06))) + # tiny headroom for stars
    labs(title = "Model Skill by Network, Metric, and Model", subtitle = depth_label, x = NULL, y = NULL) +
    theme_bw(base_size = 18) +
    theme(
      strip.text = element_text(face = "bold"),
      strip.background = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5),
      axis.text.x = element_text(face = "bold")
    ) +
    geom_segment(data = pvals, aes(x = x1, xend = x2, y = ymax, yend = ymax),
                 linewidth = 0.4, inherit.aes = FALSE) +
    geom_text(data = pvals, aes(x = xmid, y = ymax + ifelse(Metric == "KGE", 0.02, 0.015), label = stars),
              fontface = "bold", inherit.aes = FALSE)

  if (!is.null(save_path)) ggsave(filename = save_path, plot = p, width = 11, height = 6.5, dpi = 300, bg = "white")
  p
}

# ---- Figure: spatial difference map ---------------------------------------
plot_spatial_differences_by_metric = function(data, missouri_basin, save_path = NULL, depth,
                                               limit = 0.4,
                                               legend_title = "Difference\n(KGML - SPoRT-LIS)",
                                               plot_title = NULL) {
  suppressPackageStartupMessages({
    library(dplyr); library(ggplot2); library(sf); library(glue); library(scales)
  })

  states_sf = sf::st_read("https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_20m.json", quiet = TRUE) |>
    sf::st_make_valid() |>
    dplyr::filter(!NAME %in% c("Virgin Islands", "Hawaii", "Alaska", "Puerto Rico")) |>
    sf::st_transform("EPSG:5070")

  data_coords = data |>
    sf::st_transform(5070) |>
    dplyr::mutate(
      x = purrr::map_dbl(geometry, ~ sf::st_coordinates(.x)[1]),
      y = purrr::map_dbl(geometry, ~ sf::st_coordinates(.x)[2]),
      diff_clamped = pmax(pmin(diff, limit), -limit)
    ) |>
    tibble::as_tibble() |>
    tidyr::drop_na(diff_clamped)

  lab_fun = labeller(Metric = function(x) dplyr::recode(x, r = "Pearson's r", pbias = "|% Bias| advantage"))

  plot = ggplot() +
    # match old aesthetics (use size, not linewidth)
    geom_sf(data = states_sf, fill = NA, color = "black", size = 0.3) +
    geom_sf(data = missouri_basin, fill = NA, color = "darkgreen", linewidth = 1) +
    geom_point(
      data = data_coords,
      aes(x = x, y = y, fill = diff_clamped),
      shape = 21, color = "black", size = 3.5, alpha = 0.7
    ) +
    facet_wrap(~ Metric, ncol = 1, labeller = lab_fun) +
    scale_fill_gradient2(
      name = legend_title,
      low = "#1AFF1A", mid = "white", high = "#4B0092",
      midpoint = 0,
      limits = c(-limit, limit),
      oob = scales::squish,
      breaks = c(-limit, 0, limit),
      labels = c(glue::glue("< {-limit}\n(KGML Worse)"), "0\n(No Diff)", glue::glue("> {limit}\n(KGML Better)")),
      guide = ggplot2::guide_colorbar(
        barwidth = 12,
        barheight = 0.5,
        title.position = "top"
      )
    ) +
    theme_minimal(base_size = 16) +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "bottom",
      legend.title.align = 0.5,
      strip.text = element_text(face = "bold", size = 13),
      panel.spacing = grid::unit(1.5, "lines")
    ) +
    labs(
      title = if (is.null(plot_title))
                glue::glue("Spatial Difference in Accuracy (KGML – SPoRT-LIS)\n{depth}")
              else glue::glue("{plot_title}\n{depth}"),
      x = NULL, y = NULL
    )

  if (!is.null(save_path)) {
    ggsave(plot, file = save_path, width = 7, height = 10, dpi = 300, bg = "white")
  }
  plot
}

# ---- Figure: KGML-only skill boxplots by network --------------------------
# For the out-of-sample assessment: KGML standalone skill (KGE / r / |% bias|),
# one box per network. `per_site` is a wide per-site table with KGE_KGML,
# r_KGML, pbias_KGML, network. Dashed line at 0 (KGE/r reference; |bias| ideal).
plot_kgml_skill_boxes = function(per_site, depth_label = NULL, save_path = NULL) {
  suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(ggplot2) })

  net_order = per_site |> dplyr::count(network, sort = TRUE) |> dplyr::pull(network)

  long = per_site |>
    dplyr::transmute(network, site_id,
                     KGE = KGE_KGML, r = r_KGML, `|% Bias|` = abs(pbias_KGML)) |>
    tidyr::pivot_longer(c(KGE, r, `|% Bias|`), names_to = "Metric", values_to = "Score") |>
    dplyr::mutate(Metric  = factor(Metric, levels = c("KGE", "r", "|% Bias|")),
                  network = factor(network, levels = net_order),
                  # floor KGE and cap |% bias| for display (a few tiny-variance
                  # sites give pbias in the thousands and blow out the panel).
                  Score   = dplyr::case_when(
                    Metric == "KGE"      ~ pmax(Score, -1),
                    Metric == "|% Bias|" ~ pmin(Score, 100),
                    TRUE                 ~ Score
                  )) |>
    tidyr::drop_na(Score)

  p = ggplot(long, aes(network, Score, fill = network)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey55") +
    geom_boxplot(width = 0.6, outlier.alpha = 0.2, color = "black", linewidth = 0.3) +
    ggh4x::facet_grid2(cols = vars(Metric), scales = "free_y", independent = "y") +
    scale_fill_brewer(palette = "Set2", guide = "none") +
    labs(title = "KGML out-of-sample skill by network", subtitle = depth_label, x = NULL, y = NULL) +
    theme_bw(base_size = 16) +
    theme(strip.text = element_text(face = "bold"),
          strip.background = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 30, hjust = 1, face = "bold"))

  if (!is.null(save_path)) ggsave(save_path, p, width = 11, height = 5.5, dpi = 300, bg = "white")
  p
}

# ---- Figure: spatial map of a KGML skill value ----------------------------
# `data` is an sf of sites with `value_col` (e.g. KGE_KGML). Values clamped to
# `limits` and shown on a viridis scale (high = better for KGE/r; for |% bias|
# pass a reversed sense via `limits`/`legend_title`).
plot_kgml_skill_map = function(data, missouri_basin, value_col = "KGE_KGML",
                               limits = c(-1, 1), legend_title = "KGML KGE",
                               plot_title = "KGML out-of-sample skill",
                               depth = NULL, save_path = NULL) {
  suppressPackageStartupMessages({ library(dplyr); library(ggplot2); library(sf); library(scales) })

  states_sf = sf::st_read("https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_20m.json", quiet = TRUE) |>
    sf::st_make_valid() |>
    dplyr::filter(!NAME %in% c("Virgin Islands", "Hawaii", "Alaska", "Puerto Rico")) |>
    sf::st_transform("EPSG:5070")

  dat = data |>
    sf::st_transform(5070) |>
    dplyr::mutate(
      x   = sf::st_coordinates(geometry)[, 1],
      y   = sf::st_coordinates(geometry)[, 2],
      val = pmax(pmin(.data[[value_col]], max(limits)), min(limits))
    ) |>
    sf::st_drop_geometry() |>
    tibble::as_tibble() |>
    tidyr::drop_na(val)

  p = ggplot() +
    geom_sf(data = states_sf, fill = NA, color = "black", size = 0.3) +
    geom_sf(data = missouri_basin, fill = NA, color = "darkgreen", linewidth = 1) +
    geom_point(data = dat, aes(x = x, y = y, fill = val),
               shape = 21, color = "black", size = 3.2, alpha = 0.85) +
    scale_fill_viridis_c(name = legend_title, limits = limits, oob = scales::squish, option = "D") +
    theme_minimal(base_size = 15) +
    theme(axis.text = element_blank(), axis.ticks = element_blank(),
          panel.grid = element_blank(), legend.position = "bottom",
          legend.title.align = 0.5,
          plot.title = element_text(hjust = 0.5, face = "bold")) +
    labs(title = if (is.null(depth)) plot_title else glue::glue("{plot_title}\n{depth}"),
         x = NULL, y = NULL)

  if (!is.null(save_path)) ggsave(p, file = save_path, width = 7, height = 6, dpi = 300, bg = "white")
  p
}
