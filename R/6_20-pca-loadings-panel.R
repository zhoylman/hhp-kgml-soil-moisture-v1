##############################################################
# Title: PCA loadings panel -- quantitative backing for SI Figure 1's
#   "soil-hydraulic (PC1)" / "heat-aridity (PC2)" axis labels
# Description:
#   Adds a top-10-loadings bar chart per PC (from the same CONUS-standardized
#   PCA used in R/6_16), then stacks it below the existing standalone PCA
#   scatter (figs/covariate_coverage_pca_standalone.png) so SI Figure 1
#   shows the axis interpretation directly, not just in prose.
##############################################################

suppressPackageStartupMessages({
  library(tidyverse); library(glue); library(patchwork); library(magick)
})

repo = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
figs_dir = glue("{repo}/figs")

load = readRDS(glue("{repo}/cache/covariate-coverage/pca_loadings.rds")) |> as_tibble()

# human-readable labels (matches SI predictor table, R/6_19)
# Every label carries its source dataset, consistently -- not just where it
# happens to disambiguate (e.g. POLARIS "sand fraction" vs. SSURGO "sand
# content" both appear in PC1's top loadings and must stay distinguishable).
label_map = c(
  X3_n_mean = "van Genuchten n (POLARIS)", X10_lambda_mean = "Brooks-Corey lambda (POLARIS)",
  X6_sand_mean = "Sand fraction (POLARIS)", X11_hb_mean = "Brooks-Corey air-entry pressure (POLARIS)",
  X12_alpha_mean = "van Genuchten alpha (POLARIS)", X3_ssurgo_sand = "Sand content (SSURGO)",
  X1_clay_mean = "Clay fraction (POLARIS)", X8_theta_r_mean = "van Genuchten theta_r (POLARIS)",
  X7_silt_mean = "Silt fraction (POLARIS)", X2_ksat_mean = "Saturated hydraulic conductivity (POLARIS)",
  tmax = "Max. air temperature (PRISM)", vpdmax = "Max. VPD (PRISM)", tmean = "Mean air temperature (PRISM)",
  solslope = "Slope-corrected solar radiation (PRISM)", solclear = "Clear-sky solar radiation (PRISM)",
  soltotal = "Total solar radiation (PRISM)", tmin = "Min. air temperature (PRISM)",
  soltrans = "Atmospheric transmittance (PRISM)", X4_om_mean = "Organic matter content (POLARIS)",
  vpdmin = "Min. VPD (PRISM)"
)

top_n = 8
mk_panel = function(pc_col, pc_label, fill_color) {
  d = load |> transmute(variable, loading = .data[[pc_col]]) |>
    arrange(desc(abs(loading))) |> slice_head(n = top_n) |>
    mutate(label = label_map[variable], label = if_else(is.na(label), variable, label),
           label = fct_reorder(label, abs(loading)))
  ggplot(d, aes(x = loading, y = label, fill = loading > 0)) +
    geom_col(width = 0.7) +
    geom_vline(xintercept = 0, color = "grey40", linewidth = 0.3) +
    scale_fill_manual(values = c(`TRUE` = "#D55E00", `FALSE` = "#0072B2"), guide = "none") +
    labs(title = pc_label, x = "Loading", y = NULL) +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(face = "bold", hjust = 0.5, size = 13))
}

p1 = mk_panel("PC1", "PC1 top loadings:\nsoil texture & hydraulic properties")
p2 = mk_panel("PC2", "PC2 top loadings:\ntemperature & aridity")
loadings_panel = p1 | p2
ggsave(glue("{figs_dir}/covariate_coverage_pca_loadings.png"), loadings_panel, width = 11, height = 4.2, dpi = 300, bg = "white")
message("Wrote figs/covariate_coverage_pca_loadings.png")

# ---- stack below the existing standalone PCA scatter to form the final SI Figure 1 ----
scatter = image_read(glue("{figs_dir}/covariate_coverage_pca_standalone.png"))
loadings = image_read(glue("{figs_dir}/covariate_coverage_pca_loadings.png"))
loadings_resized = image_resize(loadings, glue("{image_info(scatter)$width}x"))
combined = image_append(c(scatter, loadings_resized), stack = TRUE)
image_write(combined, glue("{figs_dir}/covariate_coverage_pca_standalone_with_loadings.png"))
message(glue("Wrote figs/covariate_coverage_pca_standalone_with_loadings.png ({image_info(combined)$width}x{image_info(combined)$height}px)"))
