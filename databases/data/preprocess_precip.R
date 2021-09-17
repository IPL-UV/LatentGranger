library(raster)
library(ncdf4)

filename_template <- "precip/precip.YYYY.nc"

years <- 2007:2017
days <- seq(from = 5, to = 365, by = 1)
idx <- c(sapply(1:45, function(i) rep(i, 8)))

precip.list <- lapply(years, function(yyyy) 
  stackApply(subset(stack(sub("YYYY", yyyy, filename_template)), days), 
             indices =  idx,
             fun = mean, na.rm = TRUE))

precip <- stack(precip.list)
precip <- rotate(precip)

africa_precip <- crop(precip, extent(-20 , 55 , -37, 38))
plot(subset(africa_precip, 1))

dir.create("africa_precip", recursive = TRUE)
raster::writeRaster(africa_precip, 
                    filename = "africa_precip/precip.nc", 
                    bylayer = TRUE,
                    varname = 'precip',
                    overwrite = TRUE,
                    NAflag = -999)