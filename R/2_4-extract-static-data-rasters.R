##############################################################
# Title: Export Static Covariates from Earth Engine at 4km
# Description: Downloads static covariates (soil, terrain, 
#              climate) as local GeoTIFFs using ee_as_rast().
# Author: Dr. Zachary H. Hoylman
# Date: 8-12-2025
##############################################################

# ---- Load Required Libraries ----
library(rgee)
library(reticulate)
library(glue)
library(terra)

# ---- Initialize Earth Engine ----
reticulate::use_condaenv("gee-base", conda = "auto", required = TRUE)
ee_Initialize(drive = FALSE)

# ---- Define Output Directory ----
out_dir = "/data/ssd2/soil-moisture-ml/static-data-rasters"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Define Local Export Function Using ee_as_rast ----
export_static_layer = function(image, name, scale = 4000) {
  cat(glue("Starting: {name} at {scale}m\n"))
  
  # Clip to CONUS bounding box
  region = ee$Geometry$Rectangle(
    coords = c(-125, 24, -66.5, 50),
    proj = "EPSG:4326",
    geodesic = FALSE
  )
  
  # Convert and download as terra SpatRaster
  rast = ee_as_rast(
    image = image,
    region = region,
    scale = scale,
    via = "drive",
    container = "earthengine",
    dsn = file.path(out_dir, glue("{name}.tif"))
  )
  
  cat(glue("✓ Exported: {name}.tif\n"))
  invisible(rast)
}

# ---- PRISM Normals ----
prism_collection = ee$ImageCollection("OREGONSTATE/PRISM/Norm91m")$mean()
prism_bands = c('ppt','solclear','solslope','soltotal','soltrans','tdmean','tmax','tmean','tmin','vpdmax','vpdmin')

for (band in prism_bands) {
  image = prism_collection$
    select(band)$
    rename(glue("prism_{band}"))
  
  export_static_layer(image, glue("prism_{band}_4km"))
}

# ---- SSURGO Soil Variables ----
ssurgo = list(
  awc = "ssurgo_AWC_WTA_0to152cm_composite",
  clay = "ssurgo_Clay_WTA_0to152cm_composite",
  ksat = "ssurgo_Ksat_WTA_0to152cm_composite",
  sand = "ssurgo_Sand_WTA_0to152cm_composite"
)

for (name in names(ssurgo)) {
  image = ee$Image(glue("projects/earthengine-legacy/assets/projects/openet/soil/{ssurgo[[name]]}"))$
    rename(glue("ssurgo_{name}"))
  
  export_static_layer(image, glue("ssurgo_{name}_4km"))
}

# ---- POLARIS Soil Variables ----
polaris_vars = c(
  "bd_mean", "clay_mean", "ksat_mean", "n_mean", "om_mean", "ph_mean",
  "sand_mean", "silt_mean", "theta_r_mean", "theta_s_mean",
  "lambda_mean", "hb_mean", "alpha_mean"
)

for (var in polaris_vars) {
  image = ee$ImageCollection(glue("projects/sat-io/open-datasets/polaris/{var}"))$
    mean()$
    rename(glue("polaris_{var}"))
  
  export_static_layer(image, glue("polaris_{var}_4km"))
}

# ---- Terrain Variables ----
terrain_layers = list(
  twi = "users/zhoylman/CONUS_TWI_epsg5072_30m",
  elev = "USGS/SRTMGL1_003",
  topo_div = "CSP/ERGo/1_0/US/topoDiversity"
)

for (name in names(terrain_layers)) {
  image = ee$Image(terrain_layers[[name]])$
    rename(glue("terrain_{name}"))
  
  export_static_layer(image, glue("terrain_{name}_4km"))
}
