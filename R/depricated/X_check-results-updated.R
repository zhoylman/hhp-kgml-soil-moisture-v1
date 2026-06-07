library(tidyverse)
library(hydroGOF)
library(sf)
library(magrittr)
library(furrr)
library(glue)
library(progressr)

d = 1

depth = c('shallow','middle')[d]
depth_2 = c('Shallow','Middle')[d]
sport_depth = c('SPoRT_raw_0-10cm','SPoRT_raw_10-40cm')[d]
depth_name = c('Shallow Soil Moisture (0-10cm)',
               'Mid-depth Soil Moisture (10-50cm)')[d]

site_meta = read_csv("/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-meta.csv") |>
  sf::st_as_sf(coords = c('longitude', 'latitude')) %>%
  sf::st_set_crs('EPSG:4326')

# load observations
obs = readr::read_csv("/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-no-frozen.csv") |>
  dplyr::filter(generalized_depth == depth_2) |>
  dplyr::select(site_id, date, obs = soil_moisture)

obs_sport = readr::read_csv("/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-sport-specific-no-frozen.csv") |>
  dplyr::filter(generalized_depth == depth_2) |>
  dplyr::select(site_id, date, obs_sport_depth = soil_moisture)

# load base SPoRT data
SPoRT = readr::read_csv("/data/ssd2/soil-moisture-ml/observations/observational-sites-raw-sport.csv") |>
  dplyr::filter(var == sport_depth) |>
  tidyr::pivot_longer(cols = -c(var, time)) |>
  dplyr::select(site_id = name, date = time, SPoRT = value)

# compute baseline sport accuracy
merged = dplyr::left_join(obs, SPoRT) |>
  dplyr::left_join(obs_sport) |>
  mutate(obs = ifelse(obs > 0.7, NA, obs),
         obs_sport_depth = ifelse(obs_sport_depth > 0.7, NA, obs_sport_depth))

compute_kge = function(actual, predicted) {
  if (length(actual) != length(predicted)) {
    stop("Actual and predicted vectors must be of the same length.")
  }
  
  actual_mean = mean(actual)
  predicted_mean = mean(predicted)
  
  r = cor(actual, predicted)
  alpha = sd(predicted) / sd(actual)
  beta = predicted_mean / actual_mean
  
  kge = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
  return(kge)
}

site_metrics = merged |>
  drop_na()|>
  group_by(site_id) |>
  summarise(
    n = n(),
    KGE = if (n >= 5) compute_kge(obs_sport_depth, SPoRT) else NA_real_,
    #KGE = if (n >= 5) hydroGOF::KGE(SPoRT, obs_sport_depth) else NA_real_,
    r   = if (n >= 5) cor(SPoRT, obs_sport_depth) else NA_real_,
    .groups = "drop"
  )

site_metrics |>
  left_join(site_meta) |>
  group_by(network) |>
  mutate(KGE = ifelse(KGE < 0, 0, KGE)) |>
  summarise(mean = mean(KGE, na.rm = T),
            sd = sd(KGE, na.rm = T))

# Define root directory containing the fold subdirectories
results_dir = glue::glue("/data/ssd2/soil-moisture-ml/results-kfold-{depth}")

# List all `data_*.csv` paths inside each fold's hydrograph_data/
all_csvs = list.files(results_dir, pattern = "data_.*\\.csv$", recursive = TRUE, full.names = TRUE)

# Helper: safely compute KGE and r
compute_metrics = function(df, site_id, fold) {
  df = df |> filter(!is.na(yobs), !is.na(yhat))
  
  tibble(
    fold = fold,
    site_id = site_id,
    n = nrow(df),
    KGE = if (nrow(df) >= 5) hydroGOF::KGE(df$yhat, df$yobs) else NA_real_,
    r   = if (nrow(df) >= 5) cor(df$yhat, df$yobs) else NA_real_
  )
}

# Set up parallel backend
plan(multisession, workers = 30)

# Process all files in parallel
all_metrics = future_map_dfr(all_csvs, function(path) {
  df = suppressMessages(read_csv(path, show_col_types = FALSE))
  
  # Extract site_id from path column inside file
  site_id = df$path[1] |>
    str_extract("basin-group-([^_]+)") |>
    str_remove("basin-group-")
  
  # Extract fold number from file path
  fold = path |>
    str_extract("fold_[0-9]+") |>
    str_remove("fold_") |>
    as.integer()
  
  compute_metrics(df, site_id, fold)
}, .options = furrr_options(seed = TRUE))

plan(sequential)

ml_results = all_metrics |>
  pivot_longer(cols = -c(fold, site_id)) |>
  group_by(site_id, name) |>
  summarise(median = median(value)) |>
  pivot_wider(values_from = median)

median(ml_results$KGE, na.rm = T)
median(site_metrics$KGE, na.rm = T)

(median(ml_results$KGE, na.rm = T) - 
    median(site_metrics$KGE, na.rm = T))/
  median(site_metrics$KGE, na.rm = T) *100

median(ml_results$r, na.rm = T)
median(site_metrics$r, na.rm = T)

# compute difference in accuracy
difference = ml_results |>
  dplyr::select(-c(n)) |>
  tidyr::pivot_longer(cols = -site_id, values_to = 'KGML') |>
  dplyr::left_join(
    site_metrics |>
      dplyr::select(-c(n)) |>
      tidyr::pivot_longer(cols = -site_id, values_to = 'SPoRT')
  ) |>
  dplyr::mutate(diff = KGML - SPoRT)

KGE = difference |> 
  filter(name == 'KGE')

sum(KGE$diff > 0, na.rm = T) / sum(!is.na(KGE$diff))

r = difference |> 
  filter(name == 'r')

sum(r$diff > 0, na.rm = T) / sum(!is.na(r$diff))

#just the UMRB
KGE_UMRB = difference |> 
  filter(name == 'KGE' &
           site_id %in% (site_meta |>
                           filter(network == 'UMRB Mesonet') %$%
                           site_id))

median(KGE_UMRB$KGML, na.rm = TRUE)
median(KGE_UMRB$SPoRT, na.rm = TRUE)

sum(KGE_UMRB$diff > 0, na.rm = T) / sum(!is.na(KGE_UMRB$diff))
# Compute percent improvement in KGE
( (median(KGE_UMRB$KGML, na.rm = TRUE) - median(KGE_UMRB$SPoRT, na.rm = TRUE)) /
    median(KGE_UMRB$SPoRT, na.rm = TRUE) ) * 100


r_UMRB = difference |> 
  filter(name == 'r' &
           site_id %in% (site_meta |>
                           filter(network == 'UMRB Mesonet') %$%
                           site_id))

sum(r_UMRB$diff > 0, na.rm = T) / sum(!is.na(r_UMRB$diff))
( (median(r_UMRB$KGML, na.rm = TRUE) - median(r_UMRB$SPoRT, na.rm = TRUE)) /
    median(r_UMRB$SPoRT, na.rm = TRUE) ) * 100
# Spatial plots 

difference_spatial = difference |>
  left_join(site_meta) |>
  st_as_sf() |>
  mutate(name = ifelse(name == 'r', "Pearson's R", name))


difference |> 
  filter(name == 'KGE' &
           site_id %in% (site_meta |>
                           filter(network == 'UMRB Mesonet') %$%
                           site_id))

difference = difference |>
  rename(`SPoRT-LIS` = SPoRT)

plot_skill_boxes_2x2 <- function(difference, site_meta, depth_label = NULL, save_path = NULL) {
  suppressPackageStartupMessages({
    library(dplyr); library(tidyr); library(ggplot2); library(forcats); library(purrr)
  })
  
  # --- Regions
  umrb_ids <- site_meta %>% filter(network == "UMRB Mesonet") %>% pull(site_id) %>% unique()
  
  # --- Long form for both metrics
  long <- difference %>%
    filter(name %in% c("KGE","r")) %>%
    rename(Metric = name) %>%
    select(site_id, Metric, KGML, `SPoRT-LIS`) %>%
    pivot_longer(cols = c(KGML, `SPoRT-LIS`), names_to = "Model", values_to = "Score") %>%
    mutate(
      Model  = factor(Model, levels = c("SPoRT-LIS","KGML")),
      Metric = factor(Metric, levels = c("KGE","r")),
      Score  = ifelse(Metric == "KGE", pmax(Score, 0), Score)  # clamp only KGE
    )
  
  panels <- bind_rows(
    long %>% mutate(Region = "All sites"),
    long %>% filter(site_id %in% umrb_ids) %>% mutate(Region = "UMRB Network")
  ) %>%
    mutate(Region = factor(Region, levels = c("All sites","UMRB Network")))
  
  # --- Paired significance (Wilcoxon) on matched sites
  wdf <- panels %>%
    select(site_id, Region, Metric, Model, Score) %>%
    pivot_wider(names_from = Model, values_from = Score) %>%
    drop_na()
  
  pvals <- wdf %>%
    group_by(Region, Metric) %>%
    summarize(
      p = tryCatch(stats::wilcox.test(`SPoRT-LIS`, KGML, paired = TRUE)$p.value, error = function(e) NA_real_),
      ymax = max(c(`SPoRT-LIS`, KGML), na.rm = TRUE) + ifelse(unique(Metric) == "KGE", 0.05, 0.03),
      .groups = "drop"
    ) %>%
    mutate(
      stars = dplyr::case_when(
        is.na(p)  ~ "ns",
        p < 0.05 ~ "*",
        TRUE      ~ "ns"
      ),
      x1 = 1, x2 = 2, xmid = 1.5
    )
  
  # --- Colors (fix double-# bug)
  pal <- c(`SPoRT-LIS` = "#1AFF1A", KGML = "#4B0092")
  
  # --- Facet labeler: show full "Pearson's r"
  lab_fun <- labeller(
    Metric = function(x) ifelse(x == "r", "Pearson's r", x),
    Region = label_value
  )
  
  p <- ggplot(panels, aes(x = Model, y = Score, fill = Model)) +
    geom_boxplot(width = 0.6, outlier.alpha = 0.15, color = "black", linewidth = 0.3) +
    facet_grid(rows = vars(Region), cols = vars(Metric), scales = "fixed", labeller = lab_fun) +
    scale_fill_manual(values = pal, guide = "none") +
    scale_color_manual(values = pal, guide = "none") +
    labs(
      title = "Model Skill by Network, Metric, and Model",
      subtitle = depth_label,
      x = NULL, y = NULL
    ) +
    theme_bw(base_size = 18) +
    theme(
      strip.text = element_text(face = "bold"),
      strip.background = element_blank(),   # remove gray strip background
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5),
      axis.text.x = element_text(face = "bold")
    ) +
    geom_segment(data = pvals, aes(x = x1, xend = x2, y = ymax, yend = ymax), linewidth = 0.4, inherit.aes = FALSE) +
    geom_text(data = pvals, aes(x = xmid, y = ymax + ifelse(Metric == "KGE", 0.02, 0.015), label = stars),
              fontface = "bold", inherit.aes = FALSE)
  
  if (!is.null(save_path)) {
    ggsave(filename = save_path, plot = p, width = 8, height = 6.5, dpi = 300, bg = "white")
  }
  p
}

p <- plot_skill_boxes_2x2(
  difference = difference,
  site_meta  = site_meta,
  depth_label = depth_name,
  save_path = glue::glue("~/soil-moisture-ml/figs/kge_boxes_4panel_{depth}.png")
)

p

plot_spatial_differences_by_metric = function(data, missouri_basin, save_path = NULL, depth) {
  
  # Load states
  states_sf = sf::st_read("https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_20m.json", quiet = TRUE) %>%
    sf::st_make_valid() %>%
    dplyr::filter(!NAME %in% c("Virgin Islands", "Hawaii", "Alaska", "Puerto Rico")) %>%
    sf::st_transform("EPSG:5070")
  
  # Clamp and extract coordinates
  data_coords = data %>%
    sf::st_transform(5070) %>%
    dplyr::mutate(
      longitude = purrr::map_dbl(geometry, ~ sf::st_coordinates(.x)[1]),
      latitude  = purrr::map_dbl(geometry, ~ sf::st_coordinates(.x)[2]),
      diff_clamped = pmax(pmin(diff, 0.4), -0.4)
    ) %>%
    tibble::as_tibble() %>%
    tidyr::drop_na(diff_clamped)
  
  # Plot
  plot = ggplot() +
    geom_sf(data = states_sf, fill = NA, color = "black", size = 0.3) +
    geom_sf(data = missouri_basin, fill = NA, color = "darkgreen", linewidth = 1) +
    geom_point(
      data = data_coords,
      aes(x = longitude, y = latitude, fill = diff_clamped),
      shape = 21, color = "black", size = 3.5, alpha = 0.7
    ) +
    facet_wrap(~name, ncol = 1) +
    scale_fill_gradient2(
      name = "Difference\n(KGML - SPoRT-LIS)",
      low = "#1AFF1A", mid = "white", high = "#4B0092",
      midpoint = 0,
      limits = c(-0.4, 0.4),
      oob = scales::squish,
      breaks = c(-0.4, 0, 0.4),
      labels = c("< -0.4\n(KGML Worse)", "0\n(No Diff)", "> 0.4\n(KGML Better)"),
      guide = guide_colorbar(
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
      title = glue::glue("Spatial Difference in Accuracy (KGML – SPoRT-LIS)\n{depth}"),
      x = NULL, y = NULL
    )
  
  if (!is.null(save_path)) {
    ggsave(plot, file = save_path, width = 7, height = 10)
  }
  
  return(plot)
}

missouri_basin <- sf::read_sf("~/temp/WBD_10_HU2_Shape/Shape/WBDHU2.shp") |> 
  sf::st_transform(5070) |>
  select(-name)

plot_spatial_differences_by_metric(
  data = difference_spatial,
  missouri_basin = missouri_basin,
  depth = depth_name,
  save_path = glue::glue("~/soil-moisture-ml/figs/kgml_vs_sport_diff_map-{depth}.png")
)


# # stations curce
# ---- Define root directory of runs ----
# ---- Flatten all data_*.csv files across all site-count runs ----
results_dir = glue::glue("/data/ssd2/soil-moisture-ml/results-{depth}")
all_csvs = list.files(results_dir, pattern = "data_.*\\.csv$", recursive = TRUE, full.names = TRUE)

# ---- Fast KGE/R computation like k-fold version ----

# ---- Process all CSVs quickly with progress ----
# Setup parallel plan
plan(multisession, workers = 30)

# Enable progress handler
handlers(global = TRUE)
handlers("txtprogressbar")

# Cache length outside of future scope
n_files = length(all_csvs)

results_kge = with_progress({
  p <- progressor(along = all_csvs)
  
  n_files <- length(all_csvs)  # force eval BEFORE entering future
  
  future_imap_dfr(all_csvs, function(path, i) {
    
    df = suppressMessages(readr::read_csv(path, show_col_types = FALSE))
    
    site_id = df$path[1] |>
      stringr::str_extract("basin-group-([^_]+)") |>
      stringr::str_remove("basin-group-")
    
    n_sites = path |>
      stringr::str_extract("uNET_\\d+_sites") |>
      stringr::str_extract("\\d+") |>
      as.integer()
    
    p()
    compute_metrics(df, site_id, n_sites)
  })
})

plan(sequential)

by_fold = results_kge |>
  group_by(fold, site_id) |>
  summarise(site_median = quantile(KGE, 0.5, na.rm = T)) |>
  ungroup()|>
  group_by(fold) |>
  summarise(median = quantile(site_median, 0.5, na.rm = T),
            upper = quantile(site_median, 0.75, na.rm = T),
            lower = quantile(site_median, 0.25, na.rm = T)) |>
  bind_rows(tibble(fold = 650,
                   upper = quantile(ml_results$KGE, 0.75, na.rm = T),
                   lower = quantile(ml_results$KGE, 0.25, na.rm = T),
                   median = median(ml_results$KGE, na.rm = T)))

# Reference model lines
sport_kge = median(site_metrics$KGE, na.rm = T)  # SPoRT median KGE
smap_kge = 0.48   # SMAP HydroBlocks median KGE (from visual reference)

# Plot
plot = ggplot(by_fold, aes(x = fold, y = median)) +
  geom_smooth(method = 'lm')+
  geom_point(size = 3, color = "#341539") +
  #geom_errorbar(aes(ymin = lower, ymax = upper), width = 10, color = "#2C7BB6", alpha = 0.4, linewidth = 1) +
  geom_hline(yintercept = sport_kge, linetype = "dashed", color = "darkgreen", linewidth = 1) +
  #geom_hline(yintercept = smap_kge, linetype = "dashed", color = "darkblue", linewidth = 1) +
  #annotate("text", x = 350, y = smap_kge + 0.01, label = "SMAP HydroBlocks", color = "darkblue", hjust = 1, size = 4) +
  annotate("text", x = 600, y = sport_kge - 0.01, label = "SPoRT-LIS", color = "darkgreen", hjust = 1, size = 4) +
  labs(
    title = "KGE vs. Number of Sites in Training",
    subtitle = depth_name,
    x = "Number of Sites",
    y = "Median KGE",
    caption = "KGE = Out-of-Sample\nKling-Gupta Efficiency"
  ) +
  theme_bw(base_size = 15) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(color = "black")
  )

plot

ggsave(plot, file =  glue::glue("~/soil-moisture-ml/figs/kge_by_fold-{depth}-no-smap.png"), height = 6, width = 6)

# --- Site map only (CONUS) ---

library(sf)
library(dplyr)
library(ggplot2)
library(viridis)
library(rnaturalearth)
library(rnaturalearthdata)

# 1) Load sites (lon/lat in columns 'longitude', 'latitude')
roi <- readr::read_csv("/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-meta.csv") |>
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326) |>
  st_transform(5070) |>   # CONUS Albers Equal Area
  filter(!network %in% 'OK Mesonet') |>
  mutate(network = ifelse(network == 'SNTL', 'SNOTEL', network)) |>
  mutate(network = ifelse(network == 'UMRB Mesonet', 'UMRB Network', network))
  
# 2) Get US states, filter to CONUS, project to 5070
states_conus <- rnaturalearth::ne_states(country = "United States of America",
                                         returnclass = "sf") |>
  filter(!name %in% c("Alaska","Hawaii","Puerto Rico",
                      "American Samoa","Guam",
                      "Commonwealth of the Northern Mariana Islands",
                      "United States Virgin Islands")) |>
  st_transform(5070)

# 3) Map
p <- ggplot() +
  geom_sf(data = states_conus, fill = NA, color = "grey20", linewidth = 0.3) +
  geom_sf(data = roi, aes(fill = network), shape = 21, size = 1.6, alpha = 0.9, color = "black") +
  scale_fill_viridis_d(option = "plasma", name = NULL) +
  coord_sf(expand = FALSE) +
  labs(title = "Soil Moisture Observation Sites (CONUS)",
       x = NULL, y = NULL) +
  theme_bw(base_size = 13) +
  theme(
    panel.grid = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# 4) Save
ggsave(filename = "~/soil-moisture-ml/figs/site_map.png", plot = p, width = 8, height = 5.5, dpi = 300)

pretraining = sf::read_sf("/data/ssd2/soil-moisture-ml/random-pretraining-roi/pretraining-roi.geojson")

# 1) Project sites to CONUS Albers (EPSG:5070)
pretraining_5070 <- st_transform(pretraining, 5070)

# 3) Plot
p_pre <- ggplot() +
  geom_sf(data = states_conus, fill = NA, color = "grey20", linewidth = 0.3) +
  geom_sf(data = pretraining_5070, shape = 21, size = 1.4, alpha = 0.85,
          fill = viridis::viridis(3)[2], color = "black") +
  coord_sf(expand = FALSE) +
  labs(title = "Pretraining Sites",
       x = NULL, y = NULL) +
  theme_bw(base_size = 13) +
  theme(
    panel.grid = element_blank(),
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# 4) Save
ggsave("~/soil-moisture-ml/figs/pretraining_site_map.png",
       plot = p_pre, width = 8, height = 5.5, dpi = 300)
