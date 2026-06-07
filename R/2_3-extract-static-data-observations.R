##############################################################
# Title: Extract Static Characteristics from Earth Engine
# Description: This script extracts static point features 
#   using Google Earth Engine (GEE).
# Author: Dr. Zachary H. Hoylman
# Date: 4-16-2025
##############################################################

# ---- Load Required Libraries ----
library(reticulate)
library(rgee)
library(sf)
library(tidyverse)
library(geojsonio)

# ---- Initialize Earth Engine ----
reticulate::use_condaenv("gee-base", conda = "auto", required = TRUE)
ee = reticulate::import("ee")
rgee::ee_Initialize(drive = TRUE)

# ---- Import Points of Interest ----
roi = readr::read_csv("/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-meta.csv") %>%
  sf::st_as_sf(coords = c('longitude', 'latitude')) %>%
  sf::st_set_crs('EPSG:4326')

# ---- SSURGO Soil Variables ----
ssurgo = ee$ImageCollection(c(
  ee$Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_AWC_WTA_0to152cm_composite')$rename('ssurgo_awc'),
  ee$Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_Clay_WTA_0to152cm_composite')$rename('ssurgo_clay'),
  ee$Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_Ksat_WTA_0to152cm_composite')$rename('ssurgo_ksat'),
  ee$Image('projects/earthengine-legacy/assets/projects/openet/soil/ssurgo_Sand_WTA_0to152cm_composite')$rename('ssurgo_sand')
))$toBands()

ssurgo_vars = rgee::ee_extract(
  x = ssurgo,
  y = roi,
  fun = ee$Reducer$mean(),
  scale = 90,
  sf = FALSE
)

# ---- PRISM Climate Normals ----
prism = ee$ImageCollection("OREGONSTATE/PRISM/Norm91m")$mean()

prism_vars = rgee::ee_extract(
  x = prism,
  y = roi,
  fun = ee$Reducer$mean(),
  scale = 1000,
  sf = FALSE
)

# # ---- OpenLandMap Soil Variables ----
# openland_vars = ee$ImageCollection(c(
#   ee$Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02"),
#   ee$Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02"),
#   ee$Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02"),
#   ee$Image("OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01")
# ))$toBands()
# 
# soil_vars = rgee::ee_extract(
#   x = openland_vars,
#   y = roi,
#   fun = ee$Reducer$mean(),
#   scale = 250,
#   sf = FALSE
# )

# ---- Terrain Variables ----
terrain = ee$ImageCollection(c(
  ee$Image("users/zhoylman/CONUS_TWI_epsg5072_30m"),
  ee$Image("CSP/ERGo/1_0/US/topoDiversity"),
  ee$Image("USGS/SRTMGL1_003")
))$toBands()

terrain_vars = rgee::ee_extract(
  x = terrain,
  y = roi,
  fun = ee$Reducer$mean(),
  scale = 30,
  sf = FALSE
)

# ---- POLARIS Soil Variables ----
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

polaris_extract = rgee::ee_extract(
  x = polaris,
  y = roi,
  fun = ee$Reducer$mean(),
  scale = 30,
  sf = FALSE
)

# ---- Combine All Extracted Variables ----
final = prism_vars %>%
  # dplyr::left_join(soil_vars, by = c("site_id", 'network')) %>%
  dplyr::left_join(terrain_vars, by = c("site_id", 'network')) %>%
  dplyr::left_join(ssurgo_vars, by = c("site_id", 'network')) %>%
  dplyr::left_join(polaris_extract, by = c("site_id", 'network'))

# ---- Export Final Dataset ----
readr::write_csv(final, "/data/ssd2/soil-moisture-ml/static-data/all-sites-static-data.csv")
