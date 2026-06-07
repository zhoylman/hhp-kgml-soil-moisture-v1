##############################################################
# Title: Extract GridMET and SPoRT Time Series for ROIs
# Description: Extracts GridMET, SPoRT climatology, and raw SPoRT time series for pretraining.
# Author: Dr. Zachary H. Hoylman
# Date: 4-23-2025
##############################################################

library(terra)
library(exactextractr)
library(tidyverse)
library(sf)
library(tictoc)
library(glue)

# ---- Import ROIs ----
roi = read_csv("/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-meta.csv") %>%
  st_as_sf(coords = c('longitude', 'latitude')) %>%
  st_set_crs('EPSG:4326')

roi_pretrain = read_sf("/data/ssd2/soil-moisture-ml/random-pretraining-roi/pretraining-roi.geojson")

# ---- GridMET Extraction Function ----
extract_gridmet = function(roi, site_id_col, tag) {
  vars = c('bi', 'fm100', 'fm1000', 'pet', 'pr', 'srad', 'tmmn', 'tmmx', 'vpd')
  out = list()
  
  for (i in seq_along(vars)) {
    tictoc::tic()
    message(glue("[{tag}] Extracting GridMET variable: {vars[i]}"))
    
    gridmet = rast(paste0('/data/ssd2/gridmet/gridmet_', vars[i], '.nc'))
    time = read_csv(paste0('/data/ssd2/gridmet/gridmet_', vars[i], '_time.csv'))
    
    extract = terra::extract(gridmet, vect(roi)) %>%
      t() %>%
      as_tibble()
    colnames(extract) = roi[[site_id_col]]
    
    out[[i]] = extract[-1, ] %>%
      mutate(time = time$datetime, var = vars[i]) %>%
      select(var, time, everything())
    tictoc::toc()
  }
  return(bind_rows(out))
}

# ---- SPoRT Climatology Extraction Function ----
extract_sport_climatology = function(roi, site_id_col, tag) {
  message(glue("[{tag}] Extracting SPoRT climatology"))
  
  sport = rast("/data/ssd2/soil-moisture-models/nc/SPoRT_mean_soil_moisture_0-100cm.nc")
  #time_vals = time(sport)
  time_vals = read_csv("/data/ssd2/soil-moisture-models/nc/SPoRT_mean_soil_moisture_0-100cm_time.csv")$time
  
  tictoc::tic()
  sport_extraction = terra::extract(sport, vect(roi)) %>%
    t() %>%
    as_tibble()
  colnames(sport_extraction) = roi[[site_id_col]]
  tictoc::toc()
  
  clim = sport_extraction[-1, ] %>%
    mutate(time = time_vals,
           var = 'SPoRT') %>%
    pivot_longer(cols = -c(var, time), names_to = "name", values_to = "value") %>%
    mutate(yday = lubridate::yday(time)) %>%
    group_by(name, yday) %>%
    summarise(
      SPoRT_mean = mean(value, na.rm = TRUE),
      SPoRT_median = median(value, na.rm = TRUE),
      SPoRT_sd = sd(value, na.rm = TRUE),
      SPoRT_iqr = IQR(value, na.rm = TRUE),
      SPoRT_upper = quantile(value, 0.95, na.rm = TRUE),
      SPoRT_lower = quantile(value, 0.05, na.rm = TRUE),
      .groups = "drop"
    )
  
  dates = tibble(date = seq(as.Date("2005-01-01"), as.Date("2023-12-31"), by = "day")) %>%
    mutate(yday = lubridate::yday(date), year = lubridate::year(date))
  
  clim_expanded = clim %>%
    inner_join(dates, by = "yday") %>%
    arrange(name, date) %>%
    select(-c(yday, year)) %>%
    pivot_longer(
      cols = c(SPoRT_mean, SPoRT_median, SPoRT_sd, SPoRT_iqr, SPoRT_upper, SPoRT_lower),
      names_to = "var", values_to = "value"
    ) %>%
    select(var, date, name, value) %>%
    pivot_wider(names_from = name, values_from = value) %>%
    rename(time = date)
  
  return(clim_expanded)
}

# ---- NEW: Raw SPoRT Extraction Function (WIDE format, matching GridMET) ----
extract_sport_raw = function(roi, site_id_col, tag) {
  message(glue::glue("[{tag}] Extracting raw SPoRT time series (wide format)"))
  
  # File paths and corresponding depths
  files = list(
    "0-10cm"   = "/data/ssd2/soil-moisture-models/nc/SPoRT_mean_soil_moisture_0-10cm.nc"
    # "10-40cm"  = "/data/ssd2/soil-moisture-models/nc/SPoRT_mean_soil_moisture_10-40cm.nc"
    # "40-100cm" = "/data/ssd2/soil-moisture-models/nc/SPoRT_mean_soil_moisture_40-100cm.nc"
    #"0-100cm" = "/data/ssd2/soil-moisture-models/nc/SPoRT_mean_soil_moisture_0-100cm.nc"
  )
  
  # Loop over files and extract rasters
  results = lapply(names(files), function(depth_label) {
    sport_rast = terra::rast(files[[depth_label]])
    time_vals = terra::time(sport_rast)
    #time_vals = read_csv("/data/ssd2/soil-moisture-models/nc/SPoRT_mean_soil_moisture_0-100cm_time.csv")$time
    
    tictoc::tic(glue::glue("Extracting depth {depth_label}"))
    raw = terra::extract(sport_rast, terra::vect(roi)) %>%
      t() %>%
      tibble::as_tibble()
    tictoc::toc()
    
    colnames(raw) = roi[[site_id_col]]
    
    # Build wide tibble: var, time, site_1, site_2, ...
    wide = raw[-1, ] %>%
      dplyr::mutate(
        time = time_vals,
        var = paste0("SPoRT_raw_", depth_label)
      ) %>%
      dplyr::select(var, time, everything())
    
    return(wide)
  })
  
  # Combine all variable-depths into one tibble
  sport_wide = dplyr::bind_rows(results)
  return(sport_wide)
}

# ---- Run for Observational ROI ----
gridmet_operational = extract_gridmet(roi, "site_id", "Observational")
sport_raw_operational = extract_sport_raw(roi, "site_id", "Observational")

write_csv(gridmet_operational, "/data/ssd2/soil-moisture-ml/seq-data/observational-sites-seq-data.csv")
write_csv(sport_raw_operational, "/data/ssd2/soil-moisture-ml/observations/observational-sites-raw-sport-depth-10-40cm.csv")

rm(gridmet_operational, sport_raw_operational)
gc(); gc()

# ---- Run for Pretraining ROI ----
gridmet_pretrain = extract_gridmet(roi_pretrain, "site_id", "Pretraining")
sport_raw_pretrain = extract_sport_raw(roi_pretrain, "site_id", "Pretraining")

write_csv(gridmet_pretrain, "/data/ssd2/soil-moisture-ml/seq-data/pretraining-sites-seq-data.csv")
write_csv(sport_raw_pretrain, "/data/ssd2/soil-moisture-ml/observations/pretraining-sites-raw-sport-depth-0-10cm.csv")
