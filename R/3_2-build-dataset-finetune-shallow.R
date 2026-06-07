##############################################################
# Title: Normalize and Transform soil moisture Model Input Data
# Description: This script loads soil moisture data and covariates,
#   computes rolling climate metrics, performs normalization, 
#   and prepares the full dataset for model training and evaluation.
# Author: Dr. Zachary H. Hoylman
# Date: 4-15-2025
##############################################################

# Load required libraries
library(tidyverse)     # For data wrangling
library(sf)            # For spatial operations
library(zoo)           # For rolling means
library(doSNOW)        # For parallel processing with progress bars
library(magrittr)      # For piping
library(furrr)         # For future-based mapping
library(future)        # For parallel plans

# Define custom operator
`%notin%` = Negate(`%in%`)

# Min-max normalization function
min_max_normalize = function(x){
  if (is.Date(x) || is.character(x)) return(x)
  norm = (x - min(x)) / (max(x) - min(x))
  return(norm)
}

# Function to extract min and max values
min_max_normalize_params = function(x){
  if (is.Date(x) || is.character(x)) {
    return(tibble::tibble(var = colnames(x), min = NA, max = NA))
  } else {
    return(tibble::tibble(var = colnames(x), min = min(x, na.rm = TRUE), max = max(x, na.rm = TRUE)))
  }
}

# Load drought-related helper functions
source('https://raw.githubusercontent.com/mt-climate-office/mco-drought-indicators/master/processing/ancillary-functions/R/drought-functions.R')

roi = readr::read_csv("/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-meta.csv") %>%
  sf::st_as_sf(coords = c('longitude', 'latitude')) %>%
  sf::st_set_crs('EPSG:4326') %>%
  filter(site_id %notin% c("1529", "1085")) # sites outside of SPoRT Domain

# Set up parallel backend
cl = makeSOCKcluster(20)
doSNOW::registerDoSNOW(cl)
pb = utils::txtProgressBar(min = 1, max = length(unique(roi$site_id)), style = 3)
progress = function(n) utils::setTxtProgressBar(pb, n)
opts = list(progress = progress)

# Load raw sequential covariates
seq_covariates_raw = readr::read_csv('/data/ssd2/soil-moisture-ml/seq-data/all-sites-seq-data.csv') %>%
  dplyr::filter(var %notin% c('SPoRT_max', 'SPoRT_min'))

# Compute rolling features in parallel
tictoc::tic()
temp_out = foreach::foreach(i = 1:length(unique(roi$site_id)),
                            .packages = c('tidyverse', 'sf', 'zoo'),
                            .options.snow = opts) %dopar% {
                              
                              seq_covariates_out = seq_covariates_raw %>%
                                dplyr::select(time, var, unique(roi$site_id)[i]) %>%
                                tidyr::pivot_longer(cols = -c(time, var), names_to = 'site_id', values_to = 'val') %>%
                                tidyr::pivot_wider(id_cols = c(time, site_id), names_from = 'var', values_from = 'val') %>%
                                dplyr::mutate(year = lubridate::year(time)) %>%
                                dplyr::arrange(site_id, year) %>%
                                dplyr::group_by(site_id) %>%
                                dplyr::mutate(
                                  precip_roll_sum_short_short = zoo::rollsum(pr, align = 'right', k = 7, fill = NA),
                                  pet_roll_sum_short_short    = zoo::rollsum(pet, align = 'right', k = 7, fill = NA),
                                  precip_roll_sum_short       = zoo::rollsum(pr, align = 'right', k = 15, fill = NA),
                                  pet_roll_sum_short          = zoo::rollsum(pet, align = 'right', k = 15, fill = NA),
                                  precip_roll_sum             = zoo::rollsum(pr, align = 'right', k = 30, fill = NA),
                                  pet_roll_sum                = zoo::rollsum(pet, align = 'right', k = 30, fill = NA),
                                  precip_roll_sum_mid         = zoo::rollsum(pr, align = 'right', k = 365, fill = NA),
                                  pet_roll_sum_mid            = zoo::rollsum(pet, align = 'right', k = 365, fill = NA),
                                  precip_roll_sum_long        = zoo::rollsum(pr, align = 'right', k = 730, fill = NA),
                                  pet_roll_sum_long           = zoo::rollsum(pet, align = 'right', k = 730, fill = NA),
                                  precip_roll_sum_long_long   = zoo::rollsum(pr, align = 'right', k = 1095, fill = NA),
                                  pet_roll_sum_long_long      = zoo::rollsum(pet, align = 'right', k = 1095, fill = NA),
                                  def_short                   = precip_roll_sum_short - pet_roll_sum_short,
                                  def                         = precip_roll_sum - pet_roll_sum,
                                  def_mid                     = precip_roll_sum_mid - pet_roll_sum_mid,
                                  def_long                    = precip_roll_sum_long - pet_roll_sum_long
                                ) %>%
                                dplyr::ungroup() %>%
                                dplyr::arrange(site_id, year)
                              
                            }
tictoc::toc()
parallel::stopCluster(cl)

# Combine all processed output
seq_covariates = dplyr::bind_rows(temp_out)
#------------------------------------------------------------
# Compute min-max ranges for sequential covariates
#------------------------------------------------------------

min_max_seq = seq_covariates %>%
  dplyr::summarise(across(everything(), min, na.rm = TRUE)) %>%
  dplyr::bind_rows(
    seq_covariates %>%
      dplyr::summarise(across(everything(), max, na.rm = TRUE))
  ) %>%
  dplyr::mutate(min_max_id = c('min', 'max')) %>%
  dplyr::select(-c(time, site_id)) %>%
  tidyr::pivot_longer(cols = -min_max_id)

readr::write_csv(min_max_seq, '/data/ssd2/soil-moisture-ml/min-max-definitions/seq-min-max-definitions.csv')

#------------------------------------------------------------
# Load and transform target variable (streamflow)
#------------------------------------------------------------

target = readr::read_csv("/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized.csv") %>% 
  dplyr::filter(generalized_depth == 'Shallow') %>%
  dplyr::select(site_id, time = date, soil_moisture)

#------------------------------------------------------------
# Load gage locations for USGS and StAGE datasets
#------------------------------------------------------------

site_loc = readr::read_csv("/data/ssd2/soil-moisture-ml/observations/final-soil-moisture-data-generalized-meta.csv")  %>%
  dplyr::select(-network)

#------------------------------------------------------------
# Import and normalize static covariates
#------------------------------------------------------------

static_covariates = readr::read_csv("/data/ssd2/soil-moisture-ml/static-data/all-sites-static-data.csv") %>%
  dplyr::select(-network) %>%
  tidyr::drop_na() %>%
  dplyr::mutate_all(~ replace(., is.infinite(.), NA)) %>%
  dplyr::left_join(site_loc, by = 'site_id') %>%
  purrr::map_df(min_max_normalize)

#------------------------------------------------------------
# Compute min-max definitions for static covariates
#------------------------------------------------------------

min_max_static = readr::read_csv("/data/ssd2/soil-moisture-ml/static-data/all-sites-static-data.csv") %>%
  dplyr::select(-network) %>%
  tidyr::drop_na() %>%
  dplyr::left_join(site_loc, by = 'site_id') %>%
  dplyr::summarise(across(everything(), min, na.rm = TRUE)) %>%
  dplyr::bind_rows(
    readr::read_csv("/data/ssd2/soil-moisture-ml/static-data/all-sites-static-data.csv") %>%
      dplyr::select(-network) %>%
      tidyr::drop_na() %>%
      dplyr::left_join(site_loc, by = 'site_id') %>%
      dplyr::summarise(across(everything(), max, na.rm = TRUE))
  ) %>%
  dplyr::mutate(min_max_id = c('min', 'max')) %>%
  dplyr::select(-site_id) %>%
  tidyr::pivot_longer(cols = -min_max_id) %>%
  dplyr::mutate(value = ifelse(is.infinite(value), 1, value))

readr::write_csv(min_max_static, '/data/ssd2/soil-moisture-ml/min-max-definitions/static-min-max-definitions.csv')
#------------------------------------------------------------
# Combine and normalize final input dataset
#------------------------------------------------------------

tictoc::tic()

out = seq_covariates %>%
  dplyr::mutate_all(~ replace(., is.infinite(.), NA)) %>%
  tidyr::drop_na() %>%
  purrr::map_df(min_max_normalize) %>%
  dplyr::as_tibble() %>%
  dplyr::left_join(target, by = c('time', 'site_id')) %>%
  dplyr::left_join(static_covariates, by = 'site_id') %>%
  dplyr::select(site_id, time, soil_moisture, everything()) %>%
  tidyr::drop_na()

tictoc::toc()

#------------------------------------------------------------
# Helper function: Assign temporal group ID with optional overlap
#------------------------------------------------------------

assign_group_id <- function(x, stride_length = 90, group_size = 365) {
  # Determine number of groups (not strictly used)
  num_groups <- ceiling(nrow(x) / (group_size - stride_length))
  
  # Random start day to introduce data variety
  random_start = sample(1:365, 1)
  
  # Identify valid start indices for group segments
  start_indices <- seq(random_start, nrow(x), by = stride_length) %>%
    tibble::as_tibble() %>%
    dplyr::mutate(neg_id = nrow(x) - value) %>%
    dplyr::filter(neg_id > 0) %$%
    value
  
  # Create rolling window groupings
  group_indices <- lapply(start_indices, function(start) {
    seq(start, min(start + group_size - 1, nrow(x)))
  })
  
  # Apply group indices and return grouped dataset
  x %>%
    dplyr::slice(unlist(group_indices)) %>%
    dplyr::mutate(
      group_id = rep(1:length(group_indices), times = lengths(group_indices))
    ) %>%
    dplyr::group_by(group_id) %>%
    dplyr::group_split()
}

#------------------------------------------------------------
# Helper function: Convert day-of-year to circular (sinusoidal) representation
#------------------------------------------------------------

convert_julian_to_circular <- function(yday) {
  angle = 2 * pi * yday / 365
  sin_day_norm = (sin(angle) + 1) / 2  # Normalize to [0,1]
  return(sin_day_norm)
}
#------------------------------------------------------------
# Generate full dataset (entire record, no holdout)
#------------------------------------------------------------

# Example test of grouping (disabled for now)
# test = out_raw %>%
#   mutate(yday = lubridate::yday(time)) %>%
#   filter(site_id == c('01013500')) %>%
#   group_by(site_id) %>%
#   do(assign_group_id(., stride_length = 90, group_size = 365)) %>%
#   ungroup() %>%
#   unnest(cols = c(data))

# Set seed for reproducibility
set.seed(10)

# Initialize parallel backend
cl = makeSOCKcluster(20)
registerDoSNOW(cl)

# Define progress bar
pb = txtProgressBar(min = 1, max = length(unique(out$site_id)), style = 3)
progress = function(n) setTxtProgressBar(pb, n)
opts = list(progress = progress)

# Write out full sequence datasets (entire record, no train/test split)
temp_out = foreach(b = unique(out$site_id),
                   .packages = c('tidyverse', 'magrittr'),
                   .options.snow = opts) %dopar% {
                     
                     group_size = 180
                     stride_length = 45
                       
                     train_temp = out %>%
                       filter(site_id == b) %>%
                       dplyr::select(-c(site_id)) %>%
                       mutate(circular_yday = convert_julian_to_circular(lubridate::yday(time))) %>%
                       drop_na() %>%
                       assign_group_id(stride_length = stride_length, group_size = group_size)
                     
                     for (i in seq_along(train_temp)) {
                       if (nrow(train_temp[[i]]) == group_size && # if length is group_size
                           all(diff(train_temp[[i]]$time) == 1)) { # and if time is complete and sequential
                         write_csv(train_temp[[i]] %>% dplyr::select(-group_id),
                                   paste0('/data/ssd2/soil-moisture-ml/full-dataloader/basin-group-',
                                          b, '_', i, '-train.csv'))
                       }
                     }
                   }

stopCluster(cl)

#------------------------------------------------------------
# Random splits for training and validation
#------------------------------------------------------------
all_sites = unique(out$site_id)
all_files = list.files("/data/ssd2/soil-moisture-ml/full-dataloader", full.names = T) %>%
  as_tibble() %>%
  mutate(site_id = str_extract(value, "(?<=basin-group-)[^_]+")) %>%
  rename(path = value)



# ----
# Function: generate_site_splits
# Description: Generate training/validation splits for modeling.
#              Writes CSVs of site metadata for each split size.
# Author: Zachary H. Hoylman
# Date: 4-18-2025
# ----

generate_site_splits = function(all_sites,
                                all_files,
                                validation_n = 100,
                                train_step = 100,
                                out_dir = '/data/ssd2/soil-moisture-ml/split-definitions',
                                seed = 69) {
  
  # Load necessary packages
  requireNamespace("dplyr", quietly = TRUE)
  requireNamespace("readr", quietly = TRUE)
  requireNamespace("fs", quietly = TRUE)
  
  # Reproducibility
  set.seed(seed)
  
  # Create output directory if needed
  fs::dir_create(out_dir)
  
  # Sample validation sites
  validation_sites = sample(all_sites, validation_n)
  
  # Write validation CSV
  readr::write_csv(
    all_files %>% dplyr::filter(site_id %in% validation_sites),
    file = file.path(out_dir, paste0("validation_", validation_n, ".csv"))
  )
  
  # Get pool of training sites (everything not in validation)
  train_pool = setdiff(all_sites, validation_sites)
  n_available = length(train_pool)
  
  # Build train sizes (step-based sequence + full pool)
  train_sizes = seq(train_step, n_available - 1, by = train_step)
  train_sizes = unique(c(train_sizes, n_available))  # ensure full pool is included
  
  for (n_train in train_sizes) {
    train_sites = sample(train_pool, n_train)
    
    readr::write_csv(
      all_files %>% dplyr::filter(site_id %in% train_sites),
      file = file.path(out_dir, paste0("train_", n_train, ".csv"))
    )
  }
  
  message("Site splits generated and saved to: ", out_dir)
}

generate_site_splits(
  all_sites = unique(all_files$site_id),
  all_files = all_files,
  validation_n = 100,
  train_step = 100,
  out_dir = '/data/ssd2/soil-moisture-ml/split-definitions'
)

# 
# set.seed(69)
# #define splits
# validation_100 = sample(all_sites, 100)
# train_100 = sample(all_sites[all_sites %notin% validation_100], 100)
# train_500 = sample(all_sites[all_sites %notin% validation_100], 500)
# train_rest = sample(all_sites[all_sites %notin% validation_100], 839)
# 
# write_csv(
#   all_files %>%
#   filter(site_id %in% validation_100),
#   file = '/data/ssd2/soil-moisture-ml/split-definitions/validation_100.csv'
#   )
# 
# write_csv(
#   all_files %>%
#     filter(site_id %in% train_100),
#   file = '/data/ssd2/soil-moisture-ml/split-definitions/train_100.csv'
# )
# 
# write_csv(
#   all_files %>%
#     filter(site_id %in% train_500),
#   file = '/data/ssd2/soil-moisture-ml/split-definitions/train_500.csv'
# )
# 
# write_csv(
#   all_files %>%
#     filter(site_id %in% train_rest),
#   file = '/data/ssd2/soil-moisture-ml/split-definitions/train_rest.csv'
# )

#------------------------------------------------------------
# Random 10% station holdout for test dataset
#------------------------------------------------------------
# 
# # Set seed for reproducibility
# set.seed(11)
# 
# # Determine number of gages to hold out for test
# test_percent = 0.1
# test_gage_number = round(length(unique(out$site_id)) * test_percent)
# 
# # Randomly select test IDs
# test_ids = sample(unique(out$site_id), test_gage_number)
# 
# # Split into train/test based on IDs
# train = out %>% filter(site_id %notin% test_ids)
# test = out %>% filter(site_id %in% test_ids)
# 
# #------------------------------------------------------------
# # Write train set sequences (excluding test gages)
# #------------------------------------------------------------
# 
# # Initialize parallel backend
# cl = makeSOCKcluster(20)
# registerDoSNOW(cl)
# 
# # Define progress bar
# pb = txtProgressBar(min = 1, max = length(unique(train$site_id)), style = 3)
# progress = function(n) setTxtProgressBar(pb, n)
# opts = list(progress = progress)
# 
# # Write train sequences
# temp_out = foreach(b = unique(train$site_id),
#                    .packages = c('tidyverse', 'magrittr'),
#                    .options.snow = opts) %dopar% {
#                      
#                      train_temp = train %>%
#                        filter(site_id == b) %>%
#                        dplyr::select(-c(site_id, basin_year)) %>%
#                        mutate(circular_yday = convert_julian_to_circular(lubridate::yday(time))) %>%
#                        assign_group_id(stride_length = 90, group_size = 365)
#                      
#                      for (i in seq_along(train_temp)) {
#                        if (nrow(train_temp[[i]]) == 365) {
#                          write_csv(train_temp[[i]] %>% dplyr::select(-group_id),
#                                    paste0('/data/ssd2/soil-moisture-ml/train-dataloader/basin-year-',
#                                           b, '_', i, '-train.csv'))
#                        }
#                      }
#                    }
# 
# stopCluster(cl)
# #------------------------------------------------------------
# # Write test set sequences (10% random holdout)
# #------------------------------------------------------------
# 
# for (b in unique(test$site_id)) {
#   print(b)
#   tryCatch({
#     test_temp = test %>%
#       filter(site_id == b) %>%
#       dplyr::select(-c(site_id, basin_year)) %>%
#       mutate(circular_yday = convert_julian_to_circular(lubridate::yday(time))) %>%
#       assign_group_id(stride_length = 90, group_size = 365)
#     
#     for (i in seq_along(test_temp)) {
#       if (nrow(test_temp[[i]]) == 365) {
#         write_csv(test_temp[[i]] %>% dplyr::select(-group_id),
#                   paste0('/data/ssd2/soil-moisture-ml/test-dataloader/basin-year-',
#                          b, '_', i, '-test.csv'))
#       }
#     }
#   }, error = function(e) {
#     # Continue on error silently
#   })
# }
# 
# #------------------------------------------------------------
# # Begin K-Fold Cross-Validation Setup
# #------------------------------------------------------------
# 
# tictoc::tic()
# 
# # Set seed for reproducibility
# set.seed(11)
# #------------------------------------------------------------
# # K-Fold Cross-Validation: Partition and Write Dataloaders
# #------------------------------------------------------------
# 
# # Number of folds
# k_folds = 10
# 
# # Get unique station IDs and randomly shuffle them
# unique_ids = unique(out$site_id)
# shuffled_ids = sample(unique_ids)
# 
# # Split into k roughly equal groups
# folds = split(shuffled_ids, cut(seq_along(shuffled_ids), breaks = k_folds, labels = FALSE))
# 
# # Loop over each fold
# for (k in seq_along(folds)) {
#   cat("Fold:", k, "\n")
#   
#   # Create directory structure for fold k
#   target_dir = c(glue::glue('/data/ssd2/soil-moisture-ml/k-fold-dataloader/fold_{k}'),
#                  glue::glue('/data/ssd2/soil-moisture-ml/k-fold-dataloader/fold_{k}/train-dataloader'),
#                  glue::glue('/data/ssd2/soil-moisture-ml/k-fold-dataloader/fold_{k}/test-dataloader'))
#   
#   for (dir in target_dir) {
#     if (!dir.exists(dir)) {
#       dir.create(dir, recursive = TRUE)
#       cat("Created directory:", dir, "\n")
#     } else {
#       cat("Directory already exists:", dir, "\n")
#     }
#   }
#   
#   # Get test/train site_ids for this fold
#   test_ids = folds[[k]]
#   train = out %>% filter(!site_id %in% test_ids)
#   test = out %>% filter(site_id %in% test_ids)
#   
#   # Write training data for fold
#   cat("Fold:", k, "train data writing in progress\n")
#   cl = makeSOCKcluster(20)
#   registerDoSNOW(cl)
#   pb = txtProgressBar(min=1, max=length(unique(train$site_id)), style=3)
#   progress = function(n) setTxtProgressBar(pb, n)
#   opts = list(progress=progress)
#   
#   temp_out = foreach(b = unique(train$site_id),
#                      .packages = c('tidyverse', 'magrittr'),
#                      .export = "k",
#                      .options.snow = opts) %dopar% {
#                        tryCatch({
#                          train_temp = train %>%
#                            filter(site_id == b) %>%
#                            dplyr::select(-c(site_id, basin_year)) %>%
#                            mutate(circular_yday = convert_julian_to_circular(lubridate::yday(time))) %>%
#                            assign_group_id(stride_length = 90, group_size = 365)
#                          
#                          for (i in seq_along(train_temp)) {
#                            if (nrow(train_temp[[i]]) == 365) {
#                              write_csv(train_temp[[i]] %>% dplyr::select(-group_id),
#                                        glue::glue('/data/ssd2/soil-moisture-ml/k-fold-dataloader/fold_{k}/train-dataloader/basin-year-{b}_{i}-train.csv'))
#                            }
#                          }
#                        }, error = function(e) {})
#                      }
#   
#   stopCluster(cl)
#   
#   # Write test data for fold
#   cat("Fold:", k, "test data writing in progress\n")
#   for (b in unique(test$site_id)) {
#     print(b)
#     tryCatch({
#       test_temp = test %>%
#         filter(site_id == b) %>%
#         dplyr::select(-c(site_id, basin_year)) %>%
#         mutate(circular_yday = convert_julian_to_circular(lubridate::yday(time))) %>%
#         assign_group_id(stride_length = 90, group_size = 365)
#       
#       for (i in seq_along(test_temp)) {
#         if (nrow(test_temp[[i]]) == 365) {
#           write_csv(test_temp[[i]] %>% dplyr::select(-group_id),
#                     glue::glue('/data/ssd2/soil-moisture-ml/k-fold-dataloader/fold_{k}/test-dataloader/basin-year-{b}_{i}-test.csv'))
#         }
#       }
#     }, error = function(e) {})
#   }
# }
# 
# tictoc::toc()
# #------------------------------------------------------------
# # Special Case Dataloader: Spatiotemporal Holdout (e.g., Idaho)
# #------------------------------------------------------------
# 
# # Load state boundaries
# states = st_read('https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_20m.json') %>%
#   st_make_valid() %>%
#   filter(NAME %notin% c('Virgin Islands', 'Hawaii', 'Alaska', 'Puerto Rico'))
# 
# # Identify gages and intersect with state geometries
# gage_loc_states = roi %>%
#   st_transform('EPSG:4326') %>%
#   st_intersection(states %>% dplyr::select(NAME))
# 
# # Define test/train IDs for Idaho holdout
# test_ids = gage_loc_states %>%
#   filter(NAME %in% c('Idaho')) %$%
#   site_id %>%
#   unique()
# 
# train_ids = gage_loc_states %>%
#   filter(site_id %notin% test_ids) %$%
#   site_id %>%
#   unique()
# 
# # Split training and testing based on time and space
# train = out %>%
#   filter((time < as.Date('2011-01-01') | time >= as.Date('2014-01-01')) & site_id %in% train_ids)
# 
# test = out %>%
#   filter(time >= as.Date('2011-01-01') & time < as.Date('2014-01-01') & site_id %in% test_ids)
# 
# # Write training data
# cl = makeSOCKcluster(20)
# registerDoSNOW(cl)
# pb = txtProgressBar(min=1, max=length(unique(train$site_id)), style=3)
# progress = function(n) setTxtProgressBar(pb, n)
# opts = list(progress=progress)
# 
# temp_out = foreach(b = unique(train$site_id),
#                    .packages = c('tidyverse', 'magrittr'),
#                    .options.snow = opts) %dopar% {
#                      train_temp = train %>%
#                        filter(site_id == b) %>%
#                        dplyr::select(-c(site_id, basin_year)) %>%
#                        mutate(circular_yday = convert_julian_to_circular(lubridate::yday(time))) %>%
#                        assign_group_id(stride_length = 90, group_size = 365)
#                      
#                      for (i in seq_along(train_temp)) {
#                        if (nrow(train_temp[[i]]) == 365) {
#                          write_csv(train_temp[[i]] %>% dplyr::select(-group_id),
#                                    glue::glue('/data/ssd2/soil-moisture-ml/special-case-dataloader/spatiotemporal-holdout/train-dataloader/basin-year-{b}_{i}-train.csv'))
#                        }
#                      }
#                    }
# 
# stopCluster(cl)
# 
# # Write test data
# for (b in unique(test$site_id)) {
#   print(b)
#   tryCatch({
#     test_temp = test %>%
#       filter(site_id == b) %>%
#       dplyr::select(-c(site_id, basin_year)) %>%
#       mutate(circular_yday = convert_julian_to_circular(lubridate::yday(time))) %>%
#       assign_group_id(stride_length = 90, group_size = 365)
#     
#     for (i in seq_along(test_temp)) {
#       if (nrow(test_temp[[i]]) == 365) {
#         write_csv(test_temp[[i]] %>% dplyr::select(-group_id),
#                   glue::glue('/data/ssd2/soil-moisture-ml/special-case-dataloader/spatiotemporal-holdout/test-dataloader/basin-year-{b}_{i}-test.csv'))
#       }
#     }
#   }, error = function(e) {})
# }