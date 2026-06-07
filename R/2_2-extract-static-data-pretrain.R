##############################################################
# Title: Extract Static Characteristics from Earth Engine
# Description: Batched static point feature extraction from GEE
# Author: Dr. Zachary H. Hoylman
# Date: 4-23-2025
##############################################################

library(reticulate)
library(rgee)
library(sf)
library(tidyverse)
library(geojsonio)

# ---- Initialize Earth Engine ----
reticulate::use_condaenv("gee-base", conda = "auto", required = TRUE)
ee = reticulate::import("ee")
rgee::ee_Initialize(drive = TRUE)

# ---- Import Large ROI Dataset ----
roi_pretrain = read_sf("/data/ssd2/soil-moisture-ml/random-pretraining-roi/pretraining-roi.geojson")

# ---- Define Static Layers ----

# SSURGO
ssurgo = ee$ImageCollection(c(
  ee$Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite')$rename('ssurgo_awc'),
  ee$Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_Clay_WTA_0to152cm_composite')$rename('ssurgo_clay'),
  ee$Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_Ksat_WTA_0to152cm_composite')$rename('ssurgo_ksat'),
  ee$Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_Sand_WTA_0to152cm_composite')$rename('ssurgo_sand')
))$toBands()

# PRISM
prism = ee$ImageCollection("OREGONSTATE/PRISM/Norm91m")$mean()

# Terrain
terrain = ee$ImageCollection(c(
  ee$Image("users/zhoylman/CONUS_TWI_epsg5072_30m"),
  ee$Image("CSP/ERGo/1_0/US/topoDiversity"),
  ee$Image("USGS/SRTMGL1_003")
))$toBands()

# POLARIS
polaris = ee$ImageCollection(c(
  ee$ImageCollection('projects/sat-io/open-datasets/polaris/bd_mean')$mean()$rename('bd_mean'),
  ee$ImageCollection('projects/sat-io/open-datasets/polaris/clay_mean')$mean()$rename('clay_mean'),
  ee$ImageCollection('projects/sat-io/open-datasets/polaris/ksat_mean')$mean()$rename('ksat_mean'),
  ee$ImageCollection('projects/sat-io/open-datasets/polaris/n_mean')$mean()$rename('n_mean'),
  ee$ImageCollection('projects/sat-io/open-datasets/polaris/om_mean')$mean()$rename('om_mean'),
  ee$ImageCollection('projects/sat-io/open-datasets/polaris/ph_mean')$mean()$rename('ph_mean'),
  ee$ImageCollection('projects/sat-io/open-datasets/polaris/sand_mean')$mean()$rename('sand_mean'),
  ee$ImageCollection('projects/sat-io/open-datasets/polaris/silt_mean')$mean()$rename('silt_mean'),
  ee$ImageCollection('projects/sat-io/open-datasets/polaris/theta_r_mean')$mean()$rename('theta_r_mean'),
  ee$ImageCollection('projects/sat-io/open-datasets/polaris/theta_s_mean')$mean()$rename('theta_s_mean'),
  ee$ImageCollection('projects/sat-io/open-datasets/polaris/lambda_mean')$mean()$rename('lambda_mean'),
  ee$ImageCollection('projects/sat-io/open-datasets/polaris/hb_mean')$mean()$rename('hb_mean'),
  ee$ImageCollection('projects/sat-io/open-datasets/polaris/alpha_mean')$mean()$rename('alpha_mean')
))$toBands()

# ---- Define Feature Extraction Function ----
extract_static_features = function(chunk_sf) {
  message(glue::glue("Extracting chunk with {nrow(chunk_sf)} points..."))
  
  ssurgo_out = ee_extract(ssurgo, chunk_sf, fun = ee$Reducer$mean(), scale = 90, sf = FALSE)
  prism_out  = ee_extract(prism, chunk_sf, fun = ee$Reducer$mean(), scale = 1000, sf = FALSE)
  terrain_out = ee_extract(terrain, chunk_sf, fun = ee$Reducer$mean(), scale = 30, sf = FALSE)
  polaris_out = ee_extract(polaris, chunk_sf, fun = ee$Reducer$mean(), scale = 30, sf = FALSE)
  
  chunk_final = prism_out %>%
    left_join(terrain_out, by = c("site_id")) %>%
    left_join(ssurgo_out, by = c("site_id")) %>%
    left_join(polaris_out, by = c("site_id"))
  
  return(chunk_final)
}

# ---- Loop Over Chunks ----
n_chunks = 10
chunk_list = split(roi_pretrain, cut(seq_len(nrow(roi_pretrain)), breaks = n_chunks, labels = FALSE))

results_list = vector("list", length = n_chunks)

for (i in seq_along(chunk_list)) {
  message(glue::glue("Processing chunk {i} of {n_chunks}"))
  results_list[[i]] = extract_static_features(chunk_list[[i]])
}

# ---- Combine and Save Output ----
final_pretrain_static = bind_rows(results_list) 
write_csv(final_pretrain_static, "/data/ssd2/soil-moisture-ml/static-data/pretrain-sites-static-data.csv")
