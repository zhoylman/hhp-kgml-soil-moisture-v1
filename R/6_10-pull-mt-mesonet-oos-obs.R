library(tidyverse)
library(httr)
library(jsonlite)
library(glue)

# ============================================================================
#  Re-pull Montana Mesonet obs for the OOS stations (longer records) and
#  process into our generalized-depth, frozen-filtered obs schema.
#
#  Source: https://mesonet.climate.umt.edu/api/v2 (public). Uses premade=true
#  (precomputed daily) + station batching via httr::RETRY (the single bulk call
#  502s at the gateway; on-the-fly aggregation times out). Of 184 UMRB OOS
#  stations, 69 are MT Mesonet (matched by nwsli_id). Shallow = mean VWC at
#  5 & 10 cm; Middle = mean VWC at 20 & 50 cm; each reading frozen-filtered by
#  its own soil temp >= 34 F (matches 1_1). VWC % -> fraction.
# ============================================================================

base    = "https://mesonet.climate.umt.edu/api/v2/observations/daily/"
out_dir = "/home/zhoylman/hhp-kgml-soil-moisture-v1/data"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- match OOS UMRB sites to MT Mesonet station keys via nwsli_id ----
oos = read_csv("/home/zhoylman/hhp-kgml-soil-moisture-v1/tables/oos_sites_to_download.csv",
               show_col_types = FALSE) |> filter(network == "UMRB Mesonet")
st  = fromJSON("https://mesonet.climate.umt.edu/api/v2/stations/?type=json")
st$nwsli_up = toupper(as.character(st$nwsli_id))
matched = oos |>
  mutate(nwsli_up = toupper(site_id)) |>
  inner_join(st |> select(station, name, nwsli_up), by = "nwsli_up") |>
  transmute(site_id, station, name)
message(glue("Matched {nrow(matched)} OOS sites to MT Mesonet stations."))

# ---- batched fetch (premade daily, soil_vwc + soil_temp = all depths) ----
fetch_batch = function(station_ids) {
  resp = httr::RETRY("GET", url = base,
    query = list(na_info = "false", premade = "true", latest = "false", type = "csv",
                 rm_na = "true", active = "true", public = "true", wide = "true",
                 units = "us", tz = "America/Denver", simple_datetime = "true",
                 agg_func = "avg", start_time = "2000-01-01T00:00:00", level = "1",
                 elements = "soil_vwc,soil_temp", stations = paste(station_ids, collapse = ",")),
    times = 4, pause_base = 2, pause_cap = 30, httr::timeout(600))
  httr::stop_for_status(resp)
  readr::read_csv(I(httr::content(resp, as = "text", encoding = "UTF-8")), show_col_types = FALSE)
}

batches = split(matched$station, ceiling(seq_along(matched$station) / 15))
raw = purrr::map_dfr(seq_along(batches), function(i) {
  message(sprintf("Fetching batch %d/%d (%d stations)...", i, length(batches), length(batches[[i]])))
  fetch_batch(batches[[i]])
})
message(glue("Pulled {nrow(raw)} station-days across {length(unique(raw$station))} stations."))

# ---- tidy: parse depth/variable from unit-laden column names ----
key = matched |> transmute(station = station, site_id)
long = raw |>
  pivot_longer(cols = -c(station, datetime), names_to = "col", values_to = "value") |>
  filter(!is.na(value)) |>
  mutate(
    date  = as.Date(substr(as.character(datetime), 1, 10)),
    depth = as.integer(str_extract(col, "(?<=-)\\d+")),                 # cm below surface
    var   = if_else(str_detect(col, "Temperature"), "temp_F", "vwc_pct")
  ) |>
  select(station, date, depth, var, value) |>
  pivot_wider(names_from = var, values_from = value) |>
  inner_join(key, by = "station")

# ---- frozen filter (per-depth soil temp >= 34 F) + generalized depth ----
final = long |>
  filter(!is.na(vwc_pct), !is.na(temp_F), temp_F >= 34) |>
  mutate(generalized_depth = case_when(depth <= 10 ~ "Shallow",
                                       depth > 10 & depth <= 50 ~ "Middle",
                                       TRUE ~ NA_character_)) |>
  filter(!is.na(generalized_depth)) |>
  group_by(network = "UMRB Mesonet", site_id, date, generalized_depth) |>
  summarise(soil_moisture = mean(vwc_pct, na.rm = TRUE) / 100, .groups = "drop")  # % -> fraction

out_csv = file.path(out_dir, "oos-mt-mesonet-obs.csv")
readr::write_csv(final, out_csv)

# ---- report record lengths ----
cat(sprintf("\nWrote %s : %d rows, %d sites.\n", out_csv, nrow(final), length(unique(final$site_id))))
summ = final |> group_by(site_id, generalized_depth) |>
  summarise(n = n(), first = min(date), last = max(date),
            yrs = round(as.numeric(last - first) / 365.25, 1), .groups = "drop")
cat("Obs days per site x depth:\n"); print(summary(summ$n))
cat("Years of record:\n"); print(summary(summ$yrs))
cat(sprintf("Sites with >= 365 obs days (Shallow / Middle): %d / %d\n",
            sum(summ$n >= 365 & summ$generalized_depth == "Shallow"),
            sum(summ$n >= 365 & summ$generalized_depth == "Middle")))
