library(tidyverse)
library(arrow)

#read in data 
umrb_mesonet = read_parquet("/data/ssd2/soil-moisture-ml/observations/umrb-mesonet/umrb_sm.parquet") %>%
  filter(variable == 'soil_moisture') %>%
  mutate(network = 'UMRB Mesonet',
         value = value/100) %>%
  dplyr::select(network, site_id = STID, date, depth, moisture_corrected = value) 

umrb_mesonet_no_frozen = read_parquet("/data/ssd2/soil-moisture-ml/observations/umrb-mesonet/umrb_sm.parquet") %>%
  group_by(STID, date, depth, variable) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = variable, values_from = value) %>%
  mutate(
    soil_moisture = if_else(soil_temp < 1.1111, NA_real_, soil_moisture) # ~34F
  ) %>%
  pivot_longer(cols = c(soil_moisture, soil_temp), names_to = "variable", values_to = "value") |>
  filter(variable == 'soil_moisture') %>%
  mutate(network = 'UMRB Mesonet',
         value = value/100) %>%
  dplyr::select(network, site_id = STID, date, depth, moisture_corrected = value) 

# ok_mesonet = read_csv("/data/ssd2/soil-moisture-ml/observations/hoylman-et-al-24/ok-mesonet-raw.csv") %>%
#   pivot_longer(cols = -c(site_id, date)) %>%
#   mutate(depth = parse_number(name),
#          network = 'OK Mesonet') %>%
#   dplyr::select(network, site_id, date, depth, moisture_corrected = value)

uscrn_nrcs = read_csv("/data/ssd2/soil-moisture-ml/observations/hoylman-et-al-24/uscrn-sntl-scan-raw.csv") %>%
  pivot_longer(cols = -c(site_id, date)) %>%
  mutate(depth = parse_number(name),
         var = str_sub(name, 6,6),
         var = ifelse(var == 't', 'temperature', 'moisture'),
         network = ifelse(str_detect(site_id, 'SCAN'), 'SCAN',
                          ifelse(str_detect(site_id, 'SNTL'), 'SNTL', 'USCRN'))) %>%
  filter(var == 'moisture') %>%
  dplyr::select(-name) %>%
  pivot_wider(names_from = var, values_from = value) %>%
  rowwise() %>%
  #filter for temperature threshold 
  mutate(#moisture_corrected = ifelse(temperature < 34, NA, moisture),
    moisture_corrected = moisture,
    moisture_corrected = ifelse(network != 'USCRN', moisture_corrected/100, moisture_corrected)) %>%
  dplyr::select(network, site_id, date, depth, moisture_corrected)

uscrn_nrcs_no_frozen = read_csv("/data/ssd2/soil-moisture-ml/observations/hoylman-et-al-24/uscrn-sntl-scan-raw.csv") %>%
  pivot_longer(cols = -c(site_id, date)) %>%
  mutate(depth = parse_number(name),
         var = str_sub(name, 6,6),
         var = ifelse(var == 't', 'temperature', 'moisture'),
         network = ifelse(str_detect(site_id, 'SCAN'), 'SCAN',
                          ifelse(str_detect(site_id, 'SNTL'), 'SNTL', 'USCRN'))) %>%
  dplyr::select(-name) %>%
  pivot_wider(names_from = var, values_from = value) %>%
  rowwise() %>%
  #filter for temperature threshold 
  mutate(moisture_corrected = ifelse(temperature < 34, NA, moisture),
         moisture_corrected = ifelse(network != 'USCRN', moisture_corrected/100, moisture_corrected)) %>%
  dplyr::select(network, site_id, date, depth, moisture_corrected)

#compute "generalized depth" data
all_data = list(umrb_mesonet, uscrn_nrcs) %>%
  bind_rows() %>%
  mutate(moisture_corrected = ifelse(moisture_corrected > 1, NA, moisture_corrected),
         moisture_corrected = ifelse(moisture_corrected < 0, NA, moisture_corrected))

#compute "generalized depth" data
all_data_no_frozen = list(umrb_mesonet_no_frozen, uscrn_nrcs_no_frozen) %>%
  bind_rows() %>%
  mutate(moisture_corrected = ifelse(moisture_corrected > 1, NA, moisture_corrected),
         moisture_corrected = ifelse(moisture_corrected < 0, NA, moisture_corrected))

all_data %>%
  group_by(network) %>%
  summarise(
    unique_depths = paste(sort(unique(depth)), collapse = ", ")
  )

generalized_depths = all_data %>%
  mutate(generalized_depth = ifelse(depth <= 10, 'Shallow',
                                    ifelse(depth > 10 & depth <= 50, 'Middle', 'Deep'))) %>%
  group_by(network, site_id, date, generalized_depth) %>%
  summarise(soil_moisture = median(moisture_corrected, na.rm = T))

generalized_depths_no_frozen = all_data_no_frozen %>%
  mutate(generalized_depth = ifelse(depth <= 10, 'Shallow',
                                    ifelse(depth > 10 & depth <= 50, 'Middle', 'Deep'))) %>%
  group_by(network, site_id, date, generalized_depth) %>%
  summarise(soil_moisture = median(moisture_corrected, na.rm = T))

full_profile = generalized_depths %>%
  group_by(network, site_id, date) %>%
  summarise(n = sum(!is.na(soil_moisture)),
            soil_moisture = median(soil_moisture, na.rm = T),
            soil_moisture = ifelse(n == 3, soil_moisture, NA)) %>%
  mutate(generalized_depth = 'Depth Averaged') %>%
  dplyr::select(-n)

full_profile_no_frozen = generalized_depths_no_frozen %>%
  group_by(network, site_id, date) %>%
  summarise(n = sum(!is.na(soil_moisture)),
            soil_moisture = median(soil_moisture, na.rm = T),
            soil_moisture = ifelse(n == 3, soil_moisture, NA)) %>%
  mutate(generalized_depth = 'Depth Averaged') %>%
  dplyr::select(-n)

# get spatial metadata
umrb_site_meta = read_csv("/data/ssd2/soil-moisture-ml/observations/umrb-mesonet/station_locations.csv") %>%
  mutate(network = 'UMRB Mesonet') %>%
  select(network, site_id = STID, latitude = lat, longitude = lon)

other_site_meta = read_csv("/data/ssd2/soil-moisture-ml/observations/hoylman-et-al-24/station-meta-conus-raw.csv") %>%
  filter(network %in% c('USCRN', 'SNTL', 'SCAN', 'OK Mesonet'))

final_data = bind_rows(generalized_depths, full_profile) %>%
  arrange(network, site_id, date, generalized_depth)

final_data_no_frozen = bind_rows(generalized_depths_no_frozen, full_profile_no_frozen) %>%
  arrange(network, site_id, date, generalized_depth)

all_site_meta = bind_rows(umrb_site_meta, other_site_meta) %>%
  filter(site_id %in% unique(final_data$site_id))

#write out all data
write_csv(final_data, "/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized.csv")
write_csv(final_data_no_frozen, "/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-no-frozen.csv")

write_csv(all_site_meta, "/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-meta.csv")

# sport specific generalized depths
generalized_depths_sport = all_data %>%
  mutate(generalized_depth = ifelse(depth <= 10, 'Shallow',
                                    ifelse(depth > 10 & depth <= 40, 'Middle', 'Deep'))) %>%
  group_by(network, site_id, date, generalized_depth) %>%
  summarise(soil_moisture = median(moisture_corrected, na.rm = T))

full_profile_sport = generalized_depths_sport %>%
  group_by(network, site_id, date) %>%
  summarise(n = sum(!is.na(soil_moisture)),
            soil_moisture = median(soil_moisture, na.rm = T),
            soil_moisture = ifelse(n == 3, soil_moisture, NA)) %>%
  mutate(generalized_depth = 'Depth Averaged') %>%
  dplyr::select(-n)

final_data_sport = bind_rows(generalized_depths_sport, full_profile_sport) %>%
  arrange(network, site_id, date, generalized_depth)

#write out all data
write_csv(final_data_sport, "/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-sport-specific.csv")

generalized_depths_sport_no_frozen = all_data_no_frozen %>%
  mutate(generalized_depth = ifelse(depth <= 10, 'Shallow',
                                    ifelse(depth > 10 & depth <= 40, 'Middle', 'Deep'))) %>%
  group_by(network, site_id, date, generalized_depth) %>%
  summarise(soil_moisture = median(moisture_corrected, na.rm = T))

full_profile_sport_no_frozen = generalized_depths_sport_no_frozen %>%
  group_by(network, site_id, date) %>%
  summarise(n = sum(!is.na(soil_moisture)),
            soil_moisture = median(soil_moisture, na.rm = T),
            soil_moisture = ifelse(n == 3, soil_moisture, NA)) %>%
  mutate(generalized_depth = 'Depth Averaged') %>%
  dplyr::select(-n)

final_data_sport_no_frozen = bind_rows(generalized_depths_sport_no_frozen, full_profile_sport_no_frozen) %>%
  arrange(network, site_id, date, generalized_depth)

#write out all data
write_csv(final_data_sport_no_frozen, "/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-sport-specific-no-frozen.csv")


# indivitual depths
all_data_final = all_data %>%
  mutate(depth = depth/100) %>%
  arrange(network, site_id, date) %>%
  dplyr::select(network, site_id, date, depth, soil_moisture = moisture_corrected) 

write_csv(all_data_final, "/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-raw-depths.csv")

all_site_meta = bind_rows(umrb_site_meta, other_site_meta) %>%
  filter(site_id %in% unique(all_data_final$site_id))

write_csv(all_site_meta, "/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-raw-depths-meta.csv")
