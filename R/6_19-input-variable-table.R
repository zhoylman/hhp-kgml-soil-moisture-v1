##############################################################
# Title: SI Table -- full list of model input predictors (excludes `year`)
# Description:
#   Enumerates all 60 non-year predictors from py/exp_point_centerkeep_eval.py's
#   FEATURE_ORDER (61 total incl. year), grouped by category, with a
#   human-readable description and source dataset for each -- matching the
#   descriptions already given in prose in Section 2.2.
##############################################################

suppressPackageStartupMessages({ library(tidyverse); library(glue); library(gt) })
try(chromote::set_chrome_args(c("--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--headless=new")), silent = TRUE)

repo = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
figs_dir = glue("{repo}/figs"); tabs_dir = glue("{repo}/tables")

vars = tribble(
  ~category, ~variable, ~description, ~source,
  "Dynamic meteorological", "pr", "Precipitation", "gridMET",
  "Dynamic meteorological", "tmmx", "Maximum air temperature", "gridMET",
  "Dynamic meteorological", "tmmn", "Minimum air temperature", "gridMET",
  "Dynamic meteorological", "vpd", "Vapor pressure deficit", "gridMET",
  "Dynamic meteorological", "srad", "Downward shortwave radiation", "gridMET",
  "Dynamic meteorological", "pet", "Potential evapotranspiration", "gridMET",
  "Dynamic meteorological", "fm100", "100-hour dead fuel moisture", "gridMET",
  "Dynamic meteorological", "fm1000", "1000-hour dead fuel moisture", "gridMET",
  "Dynamic meteorological", "bi", "Burning index (NFDRS)", "gridMET",

  "Rolling accumulation (antecedent moisture)", "precip_roll_sum_short_short", "7-day precipitation sum", "Derived (gridMET)",
  "Rolling accumulation (antecedent moisture)", "pet_roll_sum_short_short", "7-day PET sum", "Derived (gridMET)",
  "Rolling accumulation (antecedent moisture)", "precip_roll_sum_short", "15-day precipitation sum", "Derived (gridMET)",
  "Rolling accumulation (antecedent moisture)", "pet_roll_sum_short", "15-day PET sum", "Derived (gridMET)",
  "Rolling accumulation (antecedent moisture)", "precip_roll_sum", "30-day precipitation sum", "Derived (gridMET)",
  "Rolling accumulation (antecedent moisture)", "pet_roll_sum", "30-day PET sum", "Derived (gridMET)",
  "Rolling accumulation (antecedent moisture)", "precip_roll_sum_mid", "60-day precipitation sum", "Derived (gridMET)",
  "Rolling accumulation (antecedent moisture)", "pet_roll_sum_mid", "60-day PET sum", "Derived (gridMET)",
  "Rolling accumulation (antecedent moisture)", "precip_roll_sum_long", "365-day precipitation sum", "Derived (gridMET)",
  "Rolling accumulation (antecedent moisture)", "pet_roll_sum_long", "365-day PET sum", "Derived (gridMET)",
  "Rolling accumulation (antecedent moisture)", "precip_roll_sum_long_long", "730-day precipitation sum", "Derived (gridMET)",
  "Rolling accumulation (antecedent moisture)", "pet_roll_sum_long_long", "730-day PET sum", "Derived (gridMET)",

  "Climatic water balance (P − PET)", "def_short", "15-day water balance", "Derived (gridMET)",
  "Climatic water balance (P − PET)", "def", "30-day water balance", "Derived (gridMET)",
  "Climatic water balance (P − PET)", "def_mid", "60-day water balance", "Derived (gridMET)",
  "Climatic water balance (P − PET)", "def_long", "365-day water balance", "Derived (gridMET)",
  "Climatic water balance (P − PET)", "def_long_long", "730-day water balance", "Derived (gridMET)",

  "PRISM 1991-2020 climate normals", "ppt", "Mean precipitation", "PRISM",
  "PRISM 1991-2020 climate normals", "tmax", "Maximum air temperature", "PRISM",
  "PRISM 1991-2020 climate normals", "tmean", "Mean air temperature", "PRISM",
  "PRISM 1991-2020 climate normals", "tmin", "Minimum air temperature", "PRISM",
  "PRISM 1991-2020 climate normals", "tdmean", "Mean dew-point temperature", "PRISM",
  "PRISM 1991-2020 climate normals", "vpdmax", "Maximum vapor pressure deficit", "PRISM",
  "PRISM 1991-2020 climate normals", "vpdmin", "Minimum vapor pressure deficit", "PRISM",
  "PRISM 1991-2020 climate normals", "solclear", "Clear-sky solar radiation", "PRISM",
  "PRISM 1991-2020 climate normals", "solslope", "Slope-corrected solar radiation", "PRISM",
  "PRISM 1991-2020 climate normals", "soltotal", "Total solar radiation", "PRISM",
  "PRISM 1991-2020 climate normals", "soltrans", "Atmospheric transmittance", "PRISM",

  "Terrain", "X2_elevation", "Elevation", "NED",
  "Terrain", "X0_b1", "Topographic wetness index", "Hoylman (2021)",
  "Terrain", "X1_constant", "Topographic diversity / ruggedness", "Theobald et al. (2015)",

  "SSURGO soil (0-152 cm depth-weighted)", "X0_ssurgo_awc", "Available water capacity", "SSURGO",
  "SSURGO soil (0-152 cm depth-weighted)", "X1_ssurgo_clay", "Clay content", "SSURGO",
  "SSURGO soil (0-152 cm depth-weighted)", "X2_ssurgo_ksat", "Saturated hydraulic conductivity", "SSURGO",
  "SSURGO soil (0-152 cm depth-weighted)", "X3_ssurgo_sand", "Sand content", "SSURGO",

  "POLARIS soil properties", "X0_bd_mean", "Bulk density", "POLARIS",
  "POLARIS soil properties", "X6_sand_mean", "Sand fraction", "POLARIS",
  "POLARIS soil properties", "X7_silt_mean", "Silt fraction", "POLARIS",
  "POLARIS soil properties", "X1_clay_mean", "Clay fraction", "POLARIS",
  "POLARIS soil properties", "X5_ph_mean", "pH", "POLARIS",
  "POLARIS soil properties", "X4_om_mean", "Organic matter content", "POLARIS",
  "POLARIS soil properties", "X2_ksat_mean", "Saturated hydraulic conductivity", "POLARIS",
  "POLARIS soil properties", "X8_theta_r_mean", "Van Genuchten residual water content (θr)", "POLARIS",
  "POLARIS soil properties", "X9_theta_s_mean", "Van Genuchten saturated water content (θs)", "POLARIS",
  "POLARIS soil properties", "X12_alpha_mean", "Van Genuchten shape parameter (α)", "POLARIS",
  "POLARIS soil properties", "X3_n_mean", "Van Genuchten shape parameter (n)", "POLARIS",
  "POLARIS soil properties", "X10_lambda_mean", "Brooks-Corey pore-size distribution index", "POLARIS",
  "POLARIS soil properties", "X11_hb_mean", "Brooks-Corey air-entry (bubbling) pressure", "POLARIS",

  "Positional / seasonal", "latitude", "Station/grid-cell latitude", "Site metadata",
  "Positional / seasonal", "longitude", "Station/grid-cell longitude", "Site metadata",
  "Positional / seasonal", "circular_yday", "Normalized sinusoidal day-of-year (Eq. 1)", "Derived"
) |>
  mutate(category = factor(category, levels = unique(category)))

stopifnot(nrow(vars) == 60)
write_csv(vars, glue("{tabs_dir}/si_input_variables.csv"))
cat(glue("n = {nrow(vars)} predictors (excl. year) across {n_distinct(vars$category)} categories\n\n"))
print(vars |> count(category), n = Inf)

g = vars |> gt(groupname_col = "category") |>
  tab_header(title = md("**Model Input Predictors**"), subtitle = "All 60 predictors excluding the frozen `year` channel (Section 2.11)") |>
  cols_label(variable = "Variable", description = "Description", source = "Source") |>
  tab_style(cell_text(weight = "bold"), cells_row_groups()) |>
  opt_table_outline() |>
  tab_options(table.font.size = px(12), data_row.padding = px(3), column_labels.font.weight = "bold",
              row_group.font.weight = "bold")
gtsave(g, glue("{figs_dir}/si_input_variables_table.png"), expand = 30, zoom = 2.5, vwidth = 900)
cat(glue("\nWrote figs/si_input_variables_table.png, tables/si_input_variables.csv\n"))

# ---------------------------------------------------------------------
# Two-column layout (to fit one page): split categories into two halves by
# row count (26 vs 34 -- can't split a category itself without breaking the
# group headers), render each as its own narrower gt table, stitch side by
# side with magick.
# ---------------------------------------------------------------------
suppressPackageStartupMessages(library(magick))

left_cats  = c("Dynamic meteorological", "Rolling accumulation (antecedent moisture)", "Climatic water balance (P − PET)")
right_cats = setdiff(levels(vars$category), left_cats)

make_half = function(cats) {
  vars |> filter(category %in% cats) |>
    gt(groupname_col = "category") |>
    cols_label(variable = "Variable", description = "Description", source = "Source") |>
    tab_style(cell_text(weight = "bold"), cells_row_groups()) |>
    opt_table_outline() |>
    tab_options(table.font.size = px(11), data_row.padding = px(2.5), column_labels.font.weight = "bold",
                row_group.font.weight = "bold")
}
tmp_left  = tempfile(fileext = ".png")
tmp_right = tempfile(fileext = ".png")
gtsave(make_half(left_cats),  tmp_left,  expand = 15, zoom = 2.5, vwidth = 520)
gtsave(make_half(right_cats), tmp_right, expand = 15, zoom = 2.5, vwidth = 520)

img = image_append(c(image_read(tmp_left), image_read(tmp_right)), stack = FALSE)
title_img = image_blank(image_info(img)$width, 70, color = "white") |>
  image_annotate("Model Input Predictors", size = 42, weight = 700, gravity = "north", location = "+0+10")
img_full = image_append(c(title_img, img), stack = TRUE)
image_write(img_full, glue("{figs_dir}/si_input_variables_table_2col.png"))
cat(glue("Wrote figs/si_input_variables_table_2col.png ({image_info(img_full)$width}x{image_info(img_full)$height}px)\n"))
