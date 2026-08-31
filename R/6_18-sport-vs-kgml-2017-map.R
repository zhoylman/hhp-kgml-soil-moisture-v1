##############################################################
# Title: SPoRT-LIS vs. KGML — 2017 mean shallow (0-10 cm) soil moisture
# Description:
#   Companion to R/6_11's 30-year climatology maps. SPoRT-LIS's raw record
#   (2005-2022) is too short to build a matched 30-year (1991-2020)
#   climatology, so instead this compares a single shared year (2017) mean
#   VWC between SPoRT-LIS (native product) and KGML (year-frozen ensemble
#   median), both reprojected to EPSG:5070 (bilinear) and cropped to the
#   same domain, in the same house map style as R/6_10/6_11.
##############################################################

suppressPackageStartupMessages({
  library(terra); library(tidyverse); library(glue); library(sf); library(patchwork)
})

repo      = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
figs_dir  = file.path(repo, "figs")
YEAR      = 2017L

sport_dir = "/data/ssd2/soil-moisture-models/SPoRT-processed_0-10cm"
kgml_dir  = "/data/ssd3/soil-moisture-ml-inference/ensemble-smoothed-daily-shallow-yearfrozen/median"

# ---------------------------------------------------------------------
# 1. Annual mean, each product's NATIVE grid (cheap: average before reprojecting)
# ---------------------------------------------------------------------
mean_of_year = function(dir, pattern, date_regex, cache_tif, date_format = "%Y-%m-%d") {
  if (file.exists(cache_tif)) return(rast(cache_tif))
  files = tibble(path = list.files(dir, pattern = pattern, full.names = TRUE)) |>
    mutate(date = as.Date(str_extract(basename(path), date_regex), format = date_format)) |>
    filter(!is.na(date), year(date) == YEAR) |>
    arrange(date)
  message(glue("{cache_tif}: {nrow(files)} daily files (expect 365)"))
  stopifnot(nrow(files) >= 300)   # sanity floor; don't silently average a mostly-empty year
  r = terra::app(terra::rast(files$path), fun = mean, na.rm = TRUE)
  writeRaster(r, cache_tif, overwrite = TRUE)
  r
}

cache_dir = glue("{repo}/cache/sport-vs-kgml-{YEAR}")
dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

sport_mean = mean_of_year(sport_dir, "^SPoRT_mean_sm_0-10cm_\\d{8}\\.tif$", "(?<=0-10cm_)\\d{8}",
                           glue("{cache_dir}/sport_mean_{YEAR}_native.tif"), date_format = "%Y%m%d")
kgml_mean  = mean_of_year(kgml_dir, "^vwc_\\d{4}-\\d{2}-\\d{2}\\.tif$", "\\d{4}-\\d{2}-\\d{2}",
                           glue("{cache_dir}/kgml_mean_{YEAR}_native.tif"), date_format = "%Y-%m-%d")

# ---------------------------------------------------------------------
# 2. Reproject both to EPSG:5070 (bilinear), crop to the SAME domain
#    (intersection of both native extents, transformed)
# ---------------------------------------------------------------------
proj_out = "EPSG:5070"
sport_5070 = terra::project(sport_mean, proj_out, method = "bilinear")
kgml_5070  = terra::project(kgml_mean,  proj_out, method = "bilinear")

# Resample SPoRT onto KGML's exact grid (bilinear) -- this puts both products
# on one shared, evenly-spaced grid, which geom_raster requires (combining
# two independently-reprojected native grids caused a silent mis-render --
# ggplot warned "pixels placed at uneven horizontal intervals and will be
# shifted"). Then build ONE combined mask that is NA wherever EITHER product
# is NA: KGML has no water mask of its own (it predicts, meaninglessly, over
# lakes/ocean), so SPoRT's water/lake mask is needed; SPoRT's native grid
# also extends beyond KGML's real domain (e.g. further into Canada), so
# KGML's domain mask is needed too. Applying the combined mask to both
# rasters gives both panels an identical domain AND identical water bodies.
sport_on_kgml  = terra::resample(sport_5070, kgml_5070, method = "bilinear")
combined_mask  = kgml_5070
combined_mask[is.na(sport_on_kgml)] = NA
sport_masked = terra::mask(sport_on_kgml, combined_mask)
kgml_masked  = terra::mask(kgml_5070,     combined_mask)

states = st_read("https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_20m.json", quiet = TRUE) |>
  st_make_valid() |>
  filter(!NAME %in% c("Virgin Islands", "Hawaii", "Alaska", "Puerto Rico")) |>
  st_transform(proj_out)

rast_df = function(r) as.data.frame(r, xy = TRUE, na.rm = TRUE) |> rlang::set_names(c("x", "y", "vwc"))
sport_df = rast_df(sport_masked) |> mutate(product = "SPoRT-LIS")
kgml_df  = rast_df(kgml_masked)  |> mutate(product = "KGML")
df = bind_rows(sport_df, kgml_df) |> mutate(product = factor(product, c("SPoRT-LIS", "KGML")))

# ---------------------------------------------------------------------
# 3. Map — same house style as R/6_11 (viridis, shared scale, EPSG:5070)
# ---------------------------------------------------------------------
vwc_lims = c(0.05, 0.45)
map_theme = theme_minimal(base_size = 12) +
  theme(panel.grid = element_blank(), axis.title = element_blank(), axis.text = element_blank(),
        plot.title = element_text(hjust = 0.5, face = "bold", size = 22),
        plot.subtitle = element_text(hjust = 0.5),
        strip.text = element_text(size = 18, face = "bold"),
        legend.position = "bottom", legend.title.position = "top",
        legend.title = element_text(hjust = 0.5))
vwc_scale = scale_fill_viridis_c(
  name = expression("Mean VWC (m"^3*" m"^-3*")"),
  option = "viridis", direction = 1, limits = vwc_lims, oob = scales::squish,
  guide = guide_colorbar(barwidth = 14, barheight = 0.5))

p = ggplot(df) +
  geom_raster(aes(x, y, fill = vwc)) +
  geom_sf(data = states, fill = NA, color = "grey25", linewidth = 0.15) +
  coord_sf(crs = proj_out, expand = FALSE) +
  facet_wrap(~product) +
  vwc_scale +
  labs(title = glue("{YEAR} Mean Shallow (0–10 cm) Soil Moisture")) +
  map_theme
ggsave(glue("{figs_dir}/vwc_{YEAR}_sport_vs_kgml_shallow.png"), p, width = 12, height = 5.5, dpi = 300, bg = "white")
message(glue("Wrote figs/vwc_{YEAR}_sport_vs_kgml_shallow.png"))

# ---------------------------------------------------------------------
# 4. Difference map (KGML - SPoRT-LIS), same domain/grid/mask as above
# ---------------------------------------------------------------------
diff_r = kgml_masked - sport_masked
diff_df = rast_df(diff_r) |> rename(diff = vwc)
diff_lims = c(-1, 1) * max(abs(quantile(diff_df$diff, c(0.01, 0.99), na.rm = TRUE)))

p_diff = ggplot(diff_df) +
  geom_raster(aes(x, y, fill = diff)) +
  geom_sf(data = states, fill = NA, color = "grey25", linewidth = 0.15) +
  coord_sf(crs = proj_out, expand = FALSE) +
  scale_fill_distiller(name = expression(Delta*" VWC (KGML − SPoRT-LIS)"), palette = "RdBu", direction = 1,
                        limits = diff_lims, oob = scales::squish,
                        guide = guide_colorbar(barwidth = 14, barheight = 0.5)) +
  labs(title = glue("{YEAR} Mean Shallow VWC Difference: KGML − SPoRT-LIS")) +
  map_theme
ggsave(glue("{figs_dir}/vwc_{YEAR}_sport_vs_kgml_shallow_diff.png"), p_diff, width = 7, height = 5.5, dpi = 300, bg = "white")
message(glue("Wrote figs/vwc_{YEAR}_sport_vs_kgml_shallow_diff.png"))
