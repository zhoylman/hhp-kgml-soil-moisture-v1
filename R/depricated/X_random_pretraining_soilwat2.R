library(sf)
library(dplyr)

# 1. Get U.S. boundary polygon
us <- read_sf('https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_20m.json') |> 
  st_make_valid() |>
  filter(!NAME %in% c("Alaska", "Hawaii", "Puerto Rico")) |> 
  st_union() |> 
  st_transform(5070)

set.seed(59801)
# 2. Sample N random points inside the U.S. (equal-area)
points <- st_sample(us, size = 10000, type = "random") |> 
  st_as_sf() |> 
  mutate(site_id = paste0('pretrain_',row_number())) |>
  st_transform(4326)

# 3. Plot them
library(ggplot2)
ggplot(test) +
  geom_sf(alpha = 0.4, color = "dodgerblue") +
  theme_minimal()
