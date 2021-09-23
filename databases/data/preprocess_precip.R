library(raster)
library(ncdf4)

leap_year <- function(year) {
  return(ifelse((year %%4 == 0 & year %%100 != 0) | year %%400 == 0, TRUE, FALSE))
}
filename_template <- "precip/precip.YYYY.nc"

years <- 2007:2017
days <- seq(from = 5, to = 365, by = 1)
idx <- c(sapply(1:45, function(i) rep(i, 8)))

dates <- seq(from = as.Date("2007-01-01"), to = as.Date("2017-12-31", ), by = 1)
length(dates)

leaps <- years[leap_year(years)]
days29 <- sapply(leaps, function(yyy){ paste0("X",yyy,".02.29")} )

tmp <- format(dates, "X%Y.%m.%d")
tmp <- (tmp[!(tmp %in% days29)])

precip.list <- lapply(years, function(yyyy) 
         stack(sub("YYYY", yyyy, filename_template))
         )

precip <- stack(precip.list)
precip <- rotate(precip)
precip <- subset(precip, tmp)

seasonality = stackApply(precip, indices = 1:365, fun = mean, na.rm = TRUE)
#sd_seasonality = stackApply(precip, indices = 1:365, fun = sd, na.rm = TRUE)


precip_ns <- (precip - seasonality) 

africa_precip <- crop(precip, extent(-20 , 55 , -37, 38))

africa_precip_ns <- crop(precip_ns, extent(-20 , 55 , -37, 38))

dir.create("africa_precip", recursive = TRUE, showWarnings = FALSE)
raster::writeRaster(africa_precip, 
                    filename = "africa_precip/precip.nc", 
                    bylayer = TRUE,
                    varname = 'precip',
                    overwrite = TRUE,
                    NAflag = -999)

dir.create("africa_precip_ns", recursive = TRUE, showWarnings = FALSE)
raster::writeRaster(africa_precip_ns, 
                    filename = "africa_precip_ns/precip.nc", 
                    bylayer = TRUE,
                    varname = 'precip',
                    overwrite = TRUE,
                    NAflag = -999)


