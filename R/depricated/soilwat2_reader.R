library(ncdf4)
library(lubridate)
library(tidyverse)

read_soilwat_site = function(nc_path, site_idx = 1) {
  # Open NetCDF
  nc = nc_open(nc_path)
  on.exit(nc_close(nc), add = TRUE)
  
  # --- Coordinates and domain ---
  lat = ncvar_get(nc, "latitude")[site_idx]
  lon = ncvar_get(nc, "longitude")[site_idx]
  site_id = if ("domain" %in% names(nc$var)) ncvar_get(nc, "domain")[site_idx] else NA_integer_
  soilwat2_name = paste0("vwc_site=", site_idx)
  
  site_meta = tibble(
    soilwat2_name = soilwat2_name,
    site_id = site_id,
    lat = lat,
    lon = lon
  )
  
  # --- Time axis ---
  time_num = ncvar_get(nc, "time")
  origin = sub("^days since ", "", ncatt_get(nc, "time", "units")$value)
  time = as.Date(time_num, origin = origin)
  
  # --- Depths ---
  if ("vertical" %in% names(nc$var)) {
    depths = ncvar_get(nc, "vertical")
  } else if ("vertical_bnds" %in% names(nc$var)) {
    vb = ncvar_get(nc, "vertical_bnds")
    depths = colMeans(vb)
  } else {
    stop("No vertical or vertical_bnds variable found in file.")
  }
  
  # --- VWC extraction [vertical, time, site] ---
  vwc = ncvar_get(nc, "vwc",
                  start = c(1, 1, site_idx),
                  count = c(-1, -1, 1))
  fill = ncatt_get(nc, "vwc", "_FillValue")$value
  vwc[vwc == fill] = NA_real_
  
  # Transpose -> [time, depth]
  vwc_t = t(vwc)
  
  # --- Build tibble ---
  data = as_tibble(vwc_t)
  names(data) = paste0("depth_", seq_along(depths))
  data = data |> mutate(time = time, soilwat2_name = soilwat2_name, site_id = site_id) |> relocate(soilwat2_name, site_id, time)
  
  # Return as list
  list(
    data = data,
    site_meta = site_meta
  )
}

fn = "/data/ssd2/SOILWAT2/VWCBULK_1979-2024_day.nc"
res = read_soilwat_site(fn, site_idx = 1)

res$site_meta
res$data |> glimpse()
