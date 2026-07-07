suppressPackageStartupMessages({
  library(terra)
  library(lubridate)
  library(glue)
  library(dplyr)
  library(lmomco)
})

# If you haven't already sourced it:
source('https://raw.githubusercontent.com/mt-climate-office/mco-drought-indicators/refs/heads/master/processing/ancillary-functions/R/drought-functions.R')

beta_fit_smi <- function(x,
                         climatology_length = 30L,
                         return_latest = TRUE) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  x <- utils::tail(x, climatology_length)
  
  # clip to (0,1) for Beta
  eps <- 1e-6
  x <- pmin(pmax(x, eps), 1 - eps)
  if (length(unique(x)) < 3L) return(NA_real_)
  
  # Method-of-moments start
  m <- mean(x); v <- stats::var(x)
  if (!is.finite(v) || v <= 0) return(NA_real_)
  t <- m * (1 - m) / v - 1
  t <- max(t, 2)
  alpha0 <- m * t
  beta0  <- (1 - m) * t
  
  # custom density for fitdistr
  dbeta_custom <- function(x, a, b, log = TRUE) {
    dbeta(x, shape1 = a, shape2 = b, log = log)
  }
  
  fit <- try(
    MASS::fitdistr(x,
                   densfun = dbeta_custom,
                   start   = list(a = alpha0, b = beta0)),
    silent = TRUE
  )
  
  if (!inherits(fit, "try-error")) {
    # Beta fit succeeded
    a <- unname(fit$estimate[["a"]])
    b <- unname(fit$estimate[["b"]])
    Fvals <- pbeta(x, shape1 = a, shape2 = b)
  } else {
    # Fallback: empirical CDF
    Fvals <- ecdf(x)(x)
  }
  
  out <- if (return_latest) utils::tail(Fvals, 1L) else Fvals
  
  # clamp away from 0/1 to avoid +/-Inf
  out_clamped <- pmin(pmax(out, 1e-12), 1 - 1e-12)
  stats::qnorm(out_clamped)
}

spi_for_day = function(
    date_of_interest,
    dir_in,
    clim_years = 30L,
    fallback_feb29 = c("02-28","03-01"),
    min_years_required = 20L,
    cores = max(1L, parallel::detectCores() - 1L),
    memfrac = 0.7,                    # fraction of RAM terra can use
    tempdir = NULL                    # optionally point to a fast SSD scratch
){
  stopifnot(inherits(date_of_interest, "Date"))
  stopifnot(dir.exists(dir_in))
  
  if (!is.null(tempdir)) terraOptions(tempdir = tempdir)
  terraOptions(progress = 1, memfrac = memfrac)
  
  # handle Feb 29 fallback
  md = format(date_of_interest, "%m-%d")
  if (md == "02-29") {
    for (alt in fallback_feb29) {
      candidate = as.Date(paste0(lubridate::year(date_of_interest), "-", alt))
      if (!is.na(candidate)) { date_of_interest = candidate; md = alt; break }
    }
  }
  
  yr        = lubridate::year(date_of_interest)
  years     = seq(yr - (clim_years - 1L), yr, by = 1L)
  target_m  = format(date_of_interest, "%m")
  target_d  = format(date_of_interest, "%d")
  dates_need = as.Date(sprintf("%04d-%s-%s", years, target_m, target_d))
  files_need = file.path(dir_in, paste0("vwc_", dates_need, ".tif"))
  
  exists_mask = file.exists(files_need)
  files_ok    = files_need[exists_mask]
  years_ok    = years[exists_mask]
  
  if (length(files_ok) < min_years_required) {
    stop(glue("Only {length(files_ok)} of {clim_years} years found; need >= {min_years_required}."))
  }
  
  message(glue("Using {length(files_ok)} years (", paste(range(years_ok), collapse = "–"),
               ") for {format(date_of_interest, '%Y-%m-%d')} on {cores} cores"))
  
  # Read stack (terra will process in blocks)
  r_stack = rast(files_ok)
  
  # Per-pixel SPI using your beta_fit_smi (returns the latest value by default)
  spi_fun = function(v) {
    if (all(is.na(v))) return(NA_real_)
    beta_fit_smi(
      as.numeric(v),
      return_latest = TRUE,
      climatology_length = min(clim_years, length(v))
    )
  }
  
  # Otherwise it returns an in-memory SpatRaster.
    spi_fun = function(v, clim_len = 30L) {
      if (all(is.na(v))) return(NA_real_)
      beta_fit_smi(as.numeric(v), return_latest = TRUE,
                    climatology_length = min(clim_len, length(v)))
    }
    
    # 4) Start a cluster and make workers aware of needed objects/packages
    cl = parallel::makeCluster(cores)                # or terra::initCluster(cores)
    parallel::clusterEvalQ(cl, { library(lmomco) })   # load package on workers
    parallel::clusterExport(cl, c("beta_fit_smi"))  # send functions
    # 5) Run in parallel (pass the cluster instead of an integer)
    spi_rast = app(r_stack, beta_fit_smi, cores = cl)
    parallel::stopCluster(cl)
  
  names(spi_rast) = paste0("soil_moisture_anomaly_", format(date_of_interest, "%Y_%m_%d"))
  spi_rast
}

missouri_basin <- sf::read_sf("~/temp/WBD_10_HU2_Shape/Shape/WBDHU2.shp") |> 
  sf::st_transform(5070) |>
  select(-name)

plot_spi_conus_binned = function(
    r_spi,
    states_url = "https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_20m.json",
    crs_out = "EPSG:5070",
    title = NULL, subtitle = NULL
){
  stopifnot(inherits(r_spi, "SpatRaster"))
  
  suppressPackageStartupMessages({
    library(terra); library(sf); library(dplyr); library(ggplot2)
  })
  
  # --- Classes (breaks, labels, colors) ---
  brks  = c(-Inf, -2, -1.6, -1.3, -0.8, -0.5, 0.5, 0.8, 1.3, 1.6, 2, Inf)
  lbls  = c("< -2 (D4)",
            "-2 – -1.6 (D3)",
            "-1.6 – -1.3 (D2)",
            "-1.3 – -0.8 (D1)",
            "-0.8 – -0.5 (D0)",
            "-0.5 – 0.5",
            "0.5 – 0.8",
            "0.8 – 1.3",
            "1.3 – 1.6",
            "1.6 – 2",
            "> 2")
  
  pal = c(
    "< -2 (D4)"       = "#730000",
    "-2 – -1.6 (D3)"  = "#E60000",
    "-1.6 – -1.3 (D2)"= "#FFAA00",
    "-1.3 – -0.8 (D1)"= "#FCD37F",
    "-0.8 – -0.5 (D0)"= "#FFFF00",
    "-0.5 – 0.5"      = "#FFFFFF",
    "0.5 – 0.8"       = "#82FCF9",
    "0.8 – 1.3"       = "#32E1FA",
    "1.3 – 1.6"       = "#325CFE",
    "1.6 – 2"         = "#4030E3",
    "> 2"             = "#303B83"
  )
  
  # --- States (lower 48) ---
  states = st_read(states_url, quiet = TRUE)
  name_col = intersect(names(states), c("name","NAME","state","STUSPS","STATE_NAME"))[1]
  lower48 = states |>
    mutate(STNAME = .data[[name_col]]) |>
    filter(!STNAME %in% c("Alaska","Hawaii","Puerto Rico","AK","HI","PR"))
  
  lower48_5070 = st_transform(lower48, crs_out)
  
  # --- Raster -> EPSG:5070, crop/mask to CONUS ---
  r_5070 = terra::project(r_spi, crs_out, method = "bilinear")
  conus_union = st_union(lower48_5070) |> st_as_sf()
  r_5070 = crop(r_5070, vect(conus_union))
  r_5070 = mask(r_5070, vect(conus_union))
  
  # --- Raster to df, bin to classes ---
  df = as.data.frame(r_5070, xy = TRUE, na.rm = TRUE)  # drop NA here!
  val_col = names(r_5070)[1]
  
  df$cat = cut(
    df[[val_col]], breaks = brks, labels = lbls, right = TRUE, include.lowest = TRUE
  )
  
  # Ensure legend order (from driest to wettest)
  df$cat = factor(df$cat, levels = names(pal))
  
  # --- Titles ---
  if (is.null(title)) {
    cleaned = gsub("^spi_|soil_moisture_anomaly_", "", val_col)
    cleaned = gsub("_", "-", cleaned)
    title = paste("Soil moisture standardized anomaly:", cleaned)
  }
  
  # --- Plot ---
  p = ggplot() +
    geom_raster(data = df, aes(x = x, y = y, fill = cat)) +
    geom_sf(data = lower48_5070, fill = NA, color = "grey20", linewidth = 0.25) +
    geom_sf(data = missouri_basin, fill = NA, color = "black", linewidth = 0.5) +
    coord_sf(crs = crs_out, expand = FALSE) +
    scale_fill_manual(
      values = pal,
      drop = FALSE,
      name = "Soil Moisture\nIndex (SMI)",
      na.translate = FALSE  # = removes NA from legend
    ) +
    labs(title = title, subtitle = subtitle,
         caption = "KGML Soil Moisture Anomaly (Montana Climate Office)") +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid = element_blank(),
      axis.title = element_blank(),
      axis.text = element_blank(),
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.subtitle = element_text(face = "bold", hjust = 0.5),
      legend.key.height = unit(0.5, "cm"),
      legend.key.width  = unit(0.7, "cm"),
      panel.background = element_rect(fill = "transparent", color = NA) # transparent bg
    )
  
  p
}

# ------------------
# Example
# ------------------
date0 = as.Date("2017-06-15")
dir0  = "/data/ssd4/soil-moisture-ml-inference/predictions-smoothed-daily-shallow"
r_spi = spi_for_day(
  date0, dir0, clim_years = 30L,
  cores = 8,                      # tune for your box
  memfrac = 0.8,                  # let terra use more RAM
  tempdir = "/data/ssd4/fast_scratch"   # optional: fast SSD scratch
)

p = plot_spi_conus_binned(r_spi,
                           title = glue::glue("Soil Moisture Anomaly (0-10 cm)"),
                          subtitle = glue::glue('{format(date0, "%m-%d-%Y")}'))
fig_dir = "/home/zhoylman/hhp-kgml-soil-moisture-v1/figs/soil_moisture_anom"
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)
ggsave(p, file = glue::glue('{fig_dir}/shallow_{date0}_attr.png'), width = 8, height = 6)
