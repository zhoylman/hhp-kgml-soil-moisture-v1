# Load necessary libraries
library(sf)
library(dplyr)
library(ggplot2)
library(terra)

# Read CONUS shapefile and filter out non-CONUS regions
conus = read_sf('https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_20m.json') %>%
  st_make_valid() %>%
  filter(!NAME %in% c('Alaska', 'Hawaii', 'Virgin Islands', 'Puerto Rico'))  # Fixed spelling of "Virgin"

# Combine all states into a single geometry
conus_union = st_union(conus)

#template data
sport = rast("/data/ssd2/soil-moisture-models/SPoRT-processed_0-10cm/SPoRT_mean_sm_0-10cm_20050101.tif") %>%
  project('EPSG:4326') %>%
  mask(conus)

# Get raster cells with non-NA values
valid_cells = which(!is.na(values(sport)))

# Get cell coordinates (centroids)
cell_coords = xyFromCell(sport, valid_cells)

# Convert to sf points
cell_points_sf = st_as_sf(as.data.frame(cell_coords), coords = c("x", "y"), crs = crs(sport))

# Sample 20,000 centroids randomly
set.seed(123)
n_centroids = 20000
sampled_centroids = cell_points_sf %>%
  slice_sample(n = n_centroids) %>%
  mutate(site_id = paste0("pretrain_", 1:n()))

# Save to GeoJSON
system('rm /data/ssd2/soil-moisture-ml/random-pretraining-roi/pretraining-roi.geojson')
write_sf(sampled_centroids, '/data/ssd2/soil-moisture-ml/random-pretraining-roi/pretraining-roi.geojson')

# Optional: Plot
ggplot() +
  geom_sf(data = conus_union, fill = NA, color = "black") +
  geom_sf(data = sampled_centroids, color = "darkgreen", size = 0.4) +
  theme_minimal() +
  labs(title = "Randomly Sampled Centroids from Valid Raster Pixels")
