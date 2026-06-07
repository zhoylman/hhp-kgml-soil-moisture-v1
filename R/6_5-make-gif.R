library(terra)
library(dplyr)
library(tidyr)
library(ggplot2)
library(gganimate)
library(rnaturalearth)
library(sf)
library(lubridate)
library(viridisLite)  # for turbo()

vwc <- rast("/data/ssd2/soil-moisture-ml/inference-rasters/inference-stack/current-inference-shallow.tif") |>
  terra::project('EPSG:5070')

# downsample for plotting speed
vwc_plot <- terra::aggregate(vwc, fact = 3, fun = mean, na.rm = TRUE) 
vwc_plot <- vwc 

nb <- nlyr(vwc_plot)

# label bands as calendar dates: 179 days before 2025-08-11 through 2025-08-11
end_date   <- as.Date("2025-08-11")
start_date <- end_date - 179
dates <- seq(start_date, by = "1 day", length.out = nb)
names(vwc_plot) <- format(dates, "%Y-%m-%d")  # these will become your time labels

# to df
df <- as.data.frame(vwc_plot, xy = TRUE) |>
  pivot_longer(cols = -c(x, y), names_to = "time", values_to = "vwc") |>
  mutate(time = factor(time, levels = names(vwc_plot)))  # preserve order

states <- rnaturalearth::ne_states(country = "United States of America", returnclass = "sf") |>
  st_transform('EPSG:5070')

vmin <- max(0, min(df$vwc, na.rm = TRUE))
vmax <- min(1, max(df$vwc, na.rm = TRUE))

g <- ggplot(df, aes(x, y, fill = vwc)) +
  geom_raster(interpolate = TRUE) +
  geom_sf(
    data = states,
    fill = NA, color = "grey30", linewidth = 0.25,
    inherit.aes = FALSE
  ) +
  coord_sf(xlim = ext(vwc_plot)[1:2], ylim = ext(vwc_plot)[3:4], expand = FALSE) +
  scale_fill_gradientn(
    colours = rev(viridisLite::turbo(256)),
    limits  = c(vmin, vmax),
    na.value = "grey95",
    name = "VWC",
    breaks = scales::pretty_breaks(6),
    labels = scales::label_number(accuracy = 0.01)
  ) +
  labs(
    title = "Volumetric Water Content (0-10cm)",
    subtitle = "{closest_state}",   # gganimate will update this per frame
    x = NULL, y = NULL,
    caption = "KGML uNET Model inference\nMontana Climate Office"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 26, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5, margin = margin(b = 6)),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.background  = element_rect(fill = "white", color = NA),
    legend.position  = "right",
    legend.key.height = unit(4, "lines"),
    legend.key.width  = unit(0.6, "lines"),
    legend.title      = element_text(size = 11, face = "bold"),
    legend.text       = element_text(size = 10),
    plot.caption      = element_text(color = "grey40", size = 9, hjust = 1),
    axis.text         = element_blank(),
    axis.ticks        = element_blank(),
    axis.title        = element_blank()
  ) +
  transition_states(time, transition_length = 1, state_length = 0)

anim <- animate(g, nframes = nb, fps = 12, width = 1000, height = 700, renderer = gifski_renderer())
anim_save("/data/ssd2/soil-moisture-ml/inference-rasters/inference-stack/vwc_sequence-shallow.gif", anim)
