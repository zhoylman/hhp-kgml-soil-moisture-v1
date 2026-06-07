library(terra)
library(tidyverse)
library(doParallel)
library(magrittr)
library(sf)

tmp_folder = "/data/ssd2/soil-moisture-models/temp"
processed_10cm_folder = '/data/ssd2/soil-moisture-models/SPoRT-processed_0-10cm'
processed_40cm_folder = '/data/ssd2/soil-moisture-models/SPoRT-processed_10-40cm'
processed_100cm_folder = '/data/ssd2/soil-moisture-models/SPoRT-processed_40-100cm'

raw_files = list.files("/data/ssd2/soil-moisture-models/raw-SPoRT", full.names = T)

dirs = c(tmp_folder, processed_10cm_folder, processed_40cm_folder, processed_100cm_folder)

# Create directories if they don't exist
for (dir in dirs) {
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
    message(glue::glue("Created: {dir}"))
  } else {
    message(glue::glue("Exists: {dir}"))
  }
}

cl = makeCluster(30)
registerDoParallel(cl)

tictoc::tic()

####
out = foreach(i = 1:length(raw_files), .packages = c('terra', 'tidyverse')) %dopar% {
  tmp_folder_i = paste0(tmp_folder, '/', i)
  
  temp_file = raw_files[i]
  
  untar(temp_file,exdir=tmp_folder_i)
  
  internal_files_temp = list.files(tmp_folder_i) %>%
    paste0(tmp_folder_i, '/', .) %>%
    tibble(full_path = list.files(., full.names = T)) %>%
    mutate(short_name = list.files(tmp_folder_i) %>%
             paste0(tmp_folder_i, '/', .) %>%
             list.files(., full.names = F)) %>%
    dplyr::select(-.) %>%
    mutate(time = substr(short_name, 10, 17))
  
  #for file
  for(f in 1:length(internal_files_temp$time)){
    temp_rast = rast(internal_files_temp$full_path[f])
    
    temp_sm_1 = temp_rast$`0-10[cm] DBLY (layer between 2 depths below land surface); Soil moisture content [kg/m^2]`
    temp_sm_2 = temp_rast$`10-40[cm] DBLY (layer between 2 depths below land surface); Soil moisture content [kg/m^2]`
    temp_sm_3 = temp_rast$`40-100[cm] DBLY (layer between 2 depths below land surface); Soil moisture content [kg/m^2]`
    #temp_sm_4 = temp_rast$`100-200[cm] DBLY (layer between 2 depths below land surface); Soil moisture content [kg/m^2]`
    
    writeRaster(temp_sm_1, paste0(processed_10cm_folder, '/SPoRT_mean_sm_0-10cm_', internal_files_temp$time[f], '.tif'), overwrite=TRUE) 
    writeRaster(temp_sm_2, paste0(processed_40cm_folder, '/SPoRT_mean_sm_10-40cm_', internal_files_temp$time[f], '.tif'), overwrite=TRUE)
    writeRaster(temp_sm_3, paste0(processed_100cm_folder, '/SPoRT_mean_sm_40-100cm_', internal_files_temp$time[f], '.tif'), overwrite=TRUE) 
    
  }
  
  unlink(tmp_folder_i, recursive = TRUE)
  gc()
}

stopCluster(cl)

tictoc::toc()


#######################################################
#######################################################
#        clip, merge and export as nc by year
#######################################################
#######################################################

`%notin%` = Negate(`%in%`) 

# Output directories
output_dir_10 = '/data/ssd2/soil-moisture-models/SPoRT-temp_nc_0-10cm'
output_dir_40 = '/data/ssd2/soil-moisture-models/SPoRT-temp_nc_10-40cm'
output_dir_100 = '/data/ssd2/soil-moisture-models/SPoRT-temp_nc_40-100cm'

final_dir = '/data/ssd2/soil-moisture-models/nc'

# Ensure output folders exist
dir.create(output_dir_10, recursive = TRUE, showWarnings = FALSE)
dir.create(output_dir_40, recursive = TRUE, showWarnings = FALSE)
dir.create(output_dir_100, recursive = TRUE, showWarnings = FALSE)
dir.create(final_dir, recursive = TRUE, showWarnings = FALSE)

# Load CONUS mask
conus = read_sf('https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_20m.json') %>%
  filter(NAME %notin% c('Alaska', 'Hawaii', 'Vigin Islands'))

import_project_clip = function(img){
  rast(img) %>%
    project(., crs('EPSG:4326')) %>%
    terra::mask(., conus)
}

process_depth_group = function(base_dir, output_dir, prefix, date_start_pos) {
  file_names = list.files(base_dir, pattern = ".tif$", full.names = FALSE)
  file_info = tibble(
    name = file_names,
    full_path = file.path(base_dir, file_names),
    time_char = substr(file_names, date_start_pos, date_start_pos + 7),
    time = as.Date(substr(file_names, date_start_pos, date_start_pos + 7), format = "%Y%m%d"),
    groups = rep(1:100, each = 100)[1:length(file_names)]
  )
  
  for (grp in unique(file_info$groups)) {
    temp_time = file_info %>% filter(groups == grp)
    
    rasters = temp_time$full_path %>%
      purrr::map(import_project_clip) %>%
      rast()
    
    names(rasters) = temp_time$time_char
    
    writeCDF(
      rasters,
      filename = file.path(output_dir, paste0(prefix, "_", grp, ".nc")),
      varname = prefix,
      overwrite = TRUE
    )
    
    write_csv(temp_time,
              file.path(output_dir, paste0(prefix, "_", grp, ".csv")))
    
    rm(rasters)
    gc()
  }
  
  return(file_info)
}

# Process each depth
names_10 = process_depth_group(
  base_dir = processed_10cm_folder,
  output_dir = output_dir_10,
  prefix = "SPoRT_mean_soil_moisture_0-10cm",
  date_start_pos = 22
)

names_40 = process_depth_group(
  base_dir = processed_40cm_folder,
  output_dir = output_dir_40,
  prefix = "SPoRT_mean_soil_moisture_10-40cm",
  date_start_pos = 23
)

names_100 = process_depth_group(
  base_dir = processed_100cm_folder,
  output_dir = output_dir_100,
  prefix = "SPoRT_mean_soil_moisture_40-100cm",
  date_start_pos = 24
)

merge_netcdf = function(output_dir, prefix, final_path){
  nc_files = tibble(
    nc_path = list.files(output_dir, pattern = ".nc$", full.names = TRUE),
    nc_name = basename(nc_path),
    time_path = list.files(output_dir, pattern = ".csv$", full.names = TRUE)
  ) %>%
    mutate(id = str_extract(nc_name, "\\d+$") %>% as.numeric()) %>%
    arrange(id)
  
  command = paste0(
    "cdo mergetime ",
    paste(nc_files$nc_path, collapse = " "),
    " ", final_path
  )
  system(command)
}

# Merge outputs
merge_netcdf(
  output_dir = output_dir_10,
  prefix = "SPoRT_mean_soil_moisture_0-10cm",
  final_path = file.path(final_dir, "SPoRT_mean_soil_moisture_0-10cm.nc")
)

merge_netcdf(
  output_dir = output_dir_40,
  prefix = "SPoRT_mean_soil_moisture_10-40cm",
  final_path = file.path(final_dir, "SPoRT_mean_soil_moisture_10-40cm.nc")
)

merge_netcdf(
  output_dir = output_dir_100,
  prefix = "SPoRT_mean_soil_moisture_40-100cm",
  final_path = file.path(final_dir, "SPoRT_mean_soil_moisture_40-100cm.nc")
)
