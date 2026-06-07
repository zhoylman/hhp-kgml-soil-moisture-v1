# ----
# Function: compute_kge_from_csv
# Description: Reads a CSV file, extracts 'yhat' and 'yobs' columns,
#              and computes the Kling-Gupta Efficiency (KGE).
# Author: Zachary H. Hoylman
# Date: 4-18-2025
# ----

LSTM try again with pretraining
Pump up params to 1-10mil
take out climatology!
  
SITES TO REMOVE ANT2 MTHE
Try increasing seq length
Visualize timeseries for bad ones!
bring in topofire 
best so far is simple u net with L1
compute true shallow sport accuracy with shallow sims


library(tidyverse)
library(hydroGOF)
library(magrittr)

`%notin%` = Negate(`%in%`)

compute_kge_from_csv = function(csv_path) {
  # Load necessary library
  if (!requireNamespace("stats", quietly = TRUE)) {
    stop("The 'stats' package is required but not available.")
  }
  
  # Define population standard deviation
  pop_sd = function(x) {
    sqrt(mean((x - mean(x))^2))  # divide by N
  }
  
  # Read CSV
  data = read.csv(csv_path)
  site = str_extract(data$path[1], "(?<=basin-group-)[^_]+")
  # Check for required columns
  if (!all(c("yhat", "yobs") %in% names(data))) {
    stop("Input CSV must contain 'yhat' and 'yobs' columns.")
  }
  
  # Extract and filter NA pairs
  yhat = data$yhat
  yobs = data$yobs
  valid = complete.cases(yhat, yobs)
  yhat = yhat[valid]
  yobs = yobs[valid]
  
  # Compute KGE components using population std dev
  r = stats::cor(yhat, yobs)
  alpha = pop_sd(yhat) / pop_sd(yobs)
  beta = mean(yhat) / mean(yobs)
  
  # Compute KGE
  kge = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
  return(tibble(site,kge,r))
}

files = list.files("/data/ssd2/soil-moisture-ml/results/LSTM_740_sites_04_24_2025-09_57_16_565617/hydrograph_data", full.names = T)
kge_results = map(files,compute_kge_from_csv) %>%
  bind_rows

site_results = kge_results %>% 
  group_by(site) %>%
  summarise(median_kge = median(kge, na.rm = T),
            median_r = median(r, na.rm = T)) %>%
  ungroup() %>%
  mutate(median_kge = ifelse(median_kge < 0, 0, median_kge))

site_results %>%
  filter(site %notin% c('ANT2','MTHE'))%$%
  median_r %>%
  median(., na.rm = T)

hist(site_results$median_kge)  
  

library(terra)
library(exactextractr)
library(tidyverse)
library(sf)

roi = read_csv("/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-meta.csv")%>%
  st_as_sf(., coords = c('longitude', 'latitude')) %>%
  st_set_crs('EPSG:4326') 

# Load raster and timestamp data for SPoRT
sport = terra::rast("/data/ssd2/soil-moisture-models/nc/SPoRT_mean_soil_moisture_0-10cm.nc")
time = time(sport)

# extraction
sport_extraction = terra::extract(sport, terra::vect(roi%>%
                                                       filter(site_id == 'SCAN:2113'))) %>%
  t() %>%
  tibble::as_tibble() 
colnames(sport_extraction) = roi$site_id

sport_long = sport_extraction %>%
  slice(-1) %>%
  mutate(time = time) %>%
  pivot_longer(cols = -c(time)) %>%
  rename(date = time, sport = value, site_id = name)

obs = read_csv("/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized.csv")

binded = left_join(sport_long, obs, by = c('site_id', 'date')) %>%
  drop_na() %>%
  filter(site_id %in% site_results$site,
         generalized_depth == 'Shallow')

# Define population standard deviation function
pop_sd = function(x) {
  sqrt(mean((x - mean(x))^2))
}

# KGE computation function
compute_kge = function(data){
  # Extract and filter NA pairs
  yhat = data$sport
  yobs = data$soil_moisture
  valid = complete.cases(yhat, yobs)
  yhat = yhat[valid]
  yobs = yobs[valid]
  
  # Compute KGE components
  r = stats::cor(yhat, yobs)
  alpha = pop_sd(yhat) / pop_sd(yobs)
  beta = mean(yhat) / mean(yobs)
  
  # Compute KGE
  kge = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
  return(kge)
}

kge_by_site = binded %>%
  select(site_id, sport, soil_moisture) %>%
  group_by(site_id) %>%
  summarise(kge = compute_kge(cur_data_all()), .groups = "drop")
  
sport_val = median(kge_by_site$kge, na.rm = T)

n_sites = c(seq(100,700,100),740)
base_path = list.dirs("/data/ssd2/soil-moisture-ml/NCSMMN-initial-results", recursive = FALSE)

compute_site_results_from_dir = function(base_dir) {
  print(base_dir)
  files = base::list.files(glue::glue('{base_dir}/hydrograph_data'), full.names = TRUE)
  
  kge_results = purrr::map(files, compute_kge_from_csv) %>%
    dplyr::bind_rows()
  
  site_results = kge_results %>%
    dplyr::group_by(site) %>%
    dplyr::summarise(
      median_kge = median(kge, na.rm = TRUE),
      median_r   = median(r, na.rm = TRUE)
    ) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(median_kge = ifelse(median_kge < 0, 0, median_kge))
  
  return(site_results)
}

overall_median_kge = function(site_results) {
  site_results %>%
    summarise(ave_kge = median(median_kge, na.rm = TRUE),
              upper_kge = quantile(median_kge, 0.75, na.rm = TRUE),
              lower_kge = quantile(median_kge, 0.25, na.rm = TRUE),
              ave_r = median(median_r, na.rm = TRUE),
              upper_r = quantile(median_r, 0.75, na.rm = TRUE),
              lower_r = quantile(median_r, 0.25, na.rm = TRUE)) 
    
}

all_results = purrr::map(base_path, compute_site_results_from_dir) 

summary = all_results %>%
  purrr::map(., overall_median_kge) %>%
  bind_rows() %>%
  mutate(`Number of Sites` = c(seq(100,700,100),740))

kge_plot = ggplot(summary, aes(x = `Number of Sites`, y = ave_kge)) +
  geom_point(size = 3, color = "steelblue") +
  geom_errorbar(
    aes(ymin = lower_kge, ymax = upper_kge),
    width = 10, color = "steelblue", alpha = 0.6
  ) +
  geom_hline(yintercept = sport_val, linetype = "dashed", color = "darkgreen", linewidth = 1) +
  geom_hline(yintercept = 0.48, linetype = "dashed", color = "purple", linewidth = 1) +
  annotate("text", x = median(summary$`Number of Sites`), y = sport_val - 0.015,
           label = "SPoRT-LIS", hjust = 0.5, color = "darkgreen", size = 4) +
  annotate("text", x = median(summary$`Number of Sites`), y = 0.48 - 0.015,
           label = "SMAP HydroBlocks", hjust = 0.5, color = "purple", size = 4) +
  labs(
    title = "Median KGE vs. Number of Sites",
    subtitle = "With interquartile range (25th–75th percentile) as error bars",
    x = "Number of Sites",
    y = "Median KGE",
    caption = "KGE = Kling-Gupta Efficiency"
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    plot.title.position = "plot",
    legend.position = "none"
  )

ggsave(kge_plot, file =  '~/soil-moisture-ml/figs/kge_plot.png', height = 6, width = 7)


library(viridis)

# Read states from GeoJSON
states = sf::st_read("https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_20m.json", quiet = TRUE)

# Filter to CONUS (remove Alaska, Hawaii, Puerto Rico, etc.)
states_conus = states %>%
  filter(!NAME %in% c("Alaska", "Hawaii", "Puerto Rico", "American Samoa",
                      "Guam", "Commonwealth of the Northern Mariana Islands",
                      "United States Virgin Islands"))%>% st_transform(., 'EPSG:5070')

# Ensure site_sf is projected to WGS84
site_sf = roi %>% st_transform(., 'EPSG:5070')

# Plot
map = ggplot() +
  geom_sf(data = states_conus, fill = 'transparent', color = "black", linewidth = 0.3) +
  geom_sf(data = site_sf, aes(fill = network), size = 1.5, alpha = 0.85, shape = 21, color = 'black') +
  scale_fill_viridis_d(option = "plasma", name = NULL) +
  labs(
    title = "Observation Sites across the Continental U.S.",
    x = NULL,
    y = NULL
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    legend.position = "bottom",
    panel.grid = element_blank()
  ) 

ggsave(map, file =  '~/soil-moisture-ml/figs/site_map.png', height = 6, width = 7)

# Plot
map_val = ggplot() +
  geom_sf(data = states_conus, fill = "gray95", color = "gray70", linewidth = 0.3) +
  geom_sf(data = site_sf, aes(fill = network), size = 1.5, alpha = 0.1, shape = 21, color = 'black') +
  geom_sf(data = site_sf_val, aes(fill = network), size = 1.5, alpha = 0.85, shape = 21, color = 'black') +
  scale_fill_viridis_d(option = "plasma", name = NULL) +
  labs(
    title = "Observation Sites across the Continental U.S.",
    x = NULL,
    y = NULL
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    legend.position = "bottom",
    panel.grid = element_blank()
  ) 

ggsave(map_val, file =  '~/soil-moisture-ml/figs/site_map_val.png', height = 6, width = 7)

all_data = list.files('/data/ssd2/soil-moisture-ml/results/uNET_729_sites_05_05_2025-18_27_28_780067/hydrograph_data', full.names = T) %>%
  map(~ suppressMessages(suppressWarnings(read_csv(.x, show_col_types = FALSE)))) %>%
  bind_rows()

all_data_filtered = all_data %>%
  mutate(site_id = str_extract(path, "(?<=basin-group-).*?(?=-train)")) %>%
  filter(site_id == 'SCAN:2113_88')

raw_forcing_data = read_csv('/data/ssd2/soil-moisture-ml/full-dataloader-shallow/basin-group-SCAN:2113_88-train.csv')

ml_and_obs = all_data_filtered %>%
  select(yhat,  yobs) %>%
  mutate(date = raw_forcing_data$time) %>%
  select(Date = date, `KGML` = yhat, Observations = yobs)

sport_example = sport_long %>%
  filter(site_id == 'SCAN:2113') %>% 
  mutate(date = as.Date(date)) |>
  filter(date %in% raw_forcing_data$time) %>%
  select(Date = date, `SPoRT-LIS` = sport)


 merged = left_join(ml_and_obs, sport_example) %>%
   pivot_longer(cols = -Date)
 
 time_series = ggplot() +
   # 1. SPoRT-LIS line first (in the back)
   geom_line(
     data = filter(merged, name == "SPoRT-LIS"),
     aes(x = Date, y = value, color = name, linetype = name),
     size = 0.8, alpha = 0.5
   ) +
   
   # 2. KGML on top
   geom_line(
     data = filter(merged, name == "KGML"),
     aes(x = Date, y = value, color = name, linetype = name),
     size = 1.2, alpha = 1
   ) +
   
   # 3. Observations on top
   geom_line(
     data = filter(merged, name == "Observations"),
     aes(x = Date, y = value, color = name, linetype = name),
     size = 1, alpha = 1
   ) +
   
   # Scales
   scale_color_manual(values = c(
     "KGML" = "steelblue",
     "Observations" = "black",
     "SPoRT-LIS" = "red"
   )) +
   scale_linetype_manual(values = c(
     "KGML" = "solid",
     "Observations" = "solid",
     "SPoRT-LIS" = "solid"
   )) +
   
   labs(
     title = "Daily Soil Moisture Comparison",
     subtitle = "KGML vs SPoRT-LIS and Observations",
     x = "Date",
     y = "Soil Moisture (m³/m³)",
     color = "Data Source",
     linetype = "Data Source"
   ) +
   theme_bw(base_size = 14) +
   theme(
     plot.title    = element_text(face = "bold", hjust = 0.5, size = 16),
     plot.subtitle = element_text(hjust = 0.5, size = 13),
     axis.title    = element_text(size = 13),
     axis.text     = element_text(size = 12),
     legend.position = "bottom",
     legend.title = element_text(size = 12, face = "bold"),
     legend.text  = element_text(size = 11),
     legend.key.width = unit(2, "cm")
   )+
   scale_x_date(
     date_breaks = "1 month",
     date_labels = "%b",  # short month names like Jan, Feb, etc.
     expand = c(0.01, 0.01)
   )
 time_series
 
 ggsave(time_series, file =  '~/soil-moisture-ml/figs/time_series.png', height = 6, width = 7)
 