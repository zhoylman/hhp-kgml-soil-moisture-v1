library(terra)
library(sf)
library(dplyr)
library(ggplot2)
library(lubridate)

# path to your 180-band prediction raster
vwc <- list.files("/data/ssd4/soil-moisture-ml-inference/predictions-smoothed-daily-shallow", full.names = T)|>
  map(rast)

vwc = rast(vwc)

extract_ts <- function(r,
                       lon = -109.43757, lat = 43.05685,           # Missoula, MT
                       end_date = as.Date("2024-10-08"),       # date of the last band
                       method = "near") {
  
  nb <- nlyr(r)
  dates <- seq(end_date - (nb - 1), end_date, by = "day")
  
  pt <- st_sfc(st_point(c(lon, lat)), crs = 4326)
  vals <- terra::extract(r, terra::vect(pt), method = method)[, -1] |> as.numeric()
  
  tibble(date = dates, vwc = vals)
}

ts_missoula <- extract_ts(vwc)

# quick look
print(head(ts_missoula))

# plot
ggplot(ts_missoula, aes(date, vwc)) +
  geom_line() +
  labs(title = "Missoula, MT — VWC (model inference)",
       x = NULL, y = "Volumetric Water Content") +
  theme_minimal()


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

test = ts_missoula |>
  mutate(year = year(date),
         month = month(date),
         day = day(date)) |>
  filter(month == 5 & day == 1 & year <= 2011 & year > 1981)

test2 = ts_missoula |>
  filter(date >= as.Date('2011-01-01') &
           date < as.Date('2011-10-01'))

plot(test2$vwc)
points(test2$vwc, col = 'red')


beta = beta_fit_smi(test$vwc, return_latest = F)
plot(test$vwc, beta)
points(test$vwc[30], beta[30], col = 'red')
