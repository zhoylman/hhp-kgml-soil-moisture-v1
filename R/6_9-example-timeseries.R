library(tidyverse)
library(glue)

# ============================================================================
#  CONSUMER — example time series: Obs vs KGML (held-out fold) vs SPoRT-LIS.
#  Reads kfold_matched.rds (obs + KGML) and the SPoRT sims (raw). Default:
#  SCAN:2011 at both depths, 2016-2017 window. Edit `examples` to swap.
# ============================================================================

repo = "/home/zhoylman/hhp-kgml-soil-moisture-v1"
obs_dir = "/data/ssd2/soil-moisture-ml/observations"
figs_dir = glue("{repo}/figs")
window = as.Date(c("2016-01-01", "2018-01-01"))   # 2016-2017, readable trace

examples = list(list(site = "1122", depth = "Shallow"),   # USCRN 1122
                list(site = "1122", depth = "Middle"))

matched = readRDS(glue("{repo}/cache/datasets/kfold_matched.rds"))
sims_all = read_csv(glue("{obs_dir}/observational-sites-raw-sport.csv"), show_col_types = FALSE) |>
  mutate(date = as.Date(time))
kfold = read_csv(glue("{repo}/tables/kfold_validation.csv"), show_col_types = FALSE)
pal = c(Obs = "black", KGML = "#4B0092", `SPoRT-LIS` = "#1AA34A")

make_ts = function(site, depth) {
  sport_var  = c(Shallow = "SPoRT_raw_0-10cm", Middle = "SPoRT_raw_10-40cm")[[depth]]
  depth_name = c(Shallow = "Shallow Soil Moisture (0-10 cm)", Middle = "Mid-depth Soil Moisture (10-50 cm)")[[depth]]

  m = matched |> filter(site_id == site, depth == !!depth) |> transmute(date, Obs = obs, KGML = ml)
  sport = sims_all |> filter(var == sport_var) |> transmute(date, `SPoRT-LIS` = .data[[site]])
  ts = full_join(m, sport, by = "date") |> filter(date >= window[1], date <= window[2]) |>
    pivot_longer(c(Obs, KGML, `SPoRT-LIS`), names_to = "Source", values_to = "sm") |>
    mutate(Source = factor(Source, levels = c("Obs", "KGML", "SPoRT-LIS")))
  kge = kfold |> filter(site_id == site, depth == !!depth)

  p = ggplot(ts, aes(date, sm, color = Source, linewidth = Source, alpha = Source)) +
    geom_line(na.rm = TRUE) +
    scale_color_manual(values = pal) +
    scale_linewidth_manual(values = c(Obs = 0.9, KGML = 0.7, `SPoRT-LIS` = 0.7), guide = "none") +
    scale_alpha_manual(values = c(Obs = 1, KGML = 0.9, `SPoRT-LIS` = 0.9), guide = "none") +
    labs(title = glue("Example soil-moisture time series — {site}"),
         subtitle = glue("{depth_name}   |   KGE: KGML = {round(kge$KGE,2)}, SPoRT-LIS = {round(kge$KGE_sport,2)}"),
         x = NULL, y = expression("Volumetric soil moisture (m"^3*" m"^-3*")"), color = NULL) +
    guides(color = guide_legend(override.aes = list(linewidth = 1.5))) +
    theme_bw(base_size = 15) +
    theme(legend.position = "bottom", plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5), axis.title.y = element_text(size = 12))

  out = glue("{figs_dir}/example_timeseries_{gsub(':','_',site)}_{tolower(depth)}.png")
  ggsave(out, p, width = 11, height = 4.5, dpi = 300, bg = "white")
  message(glue("Wrote {out}"))
}

for (e in examples) make_ts(e$site, e$depth)
