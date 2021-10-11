library(raster)
library(ncdf4)

path <- "../../../neural_granger/databases/experiment1_NDVI_SIF_ENSO/dataFinal/"
filename_template <- paste0(path,"MODIS_MCD43C4_05deg_16d_TMSTMP.nc")

years <- 2007:2017
days <- formatC(seq(from =5, to = 365, by = 8), width = 3, flag = "0")
timestamps <- apply(expand.grid(days, years)[,2:1], 1, paste0, collapse = "")
length(timestamps) #506

red.list <- lapply(timestamps, function(tmstmp) stack(sub("TMSTMP", tmstmp, filename_template), varname = "red"))
nir.list <- lapply(timestamps, function(tmstmp) stack(sub("TMSTMP", tmstmp, filename_template), varname = "nir"))
red <- stack(red.list)
nir <- stack(nir.list)
ndvi = (nir-red)/(nir+red)
ndvi = approxNA(ndvi, rule = 2)  ### use linear interpolation to fill NA
seasonality = stackApply(ndvi, indices = 1:46, fun = mean, na.rm = TRUE)

dir.create("ndvi_seasonality")
raster::writeRaster(seasonality, 
                    filename = "ndvi_seasonality/ndvi_seasonality.nc", 
                    bylayer = TRUE,
                    varname = 'ndvi',
                    suffix = days,
                    overwrite = TRUE,
                    NAflag = -999)

dir.create("africa_ndvi_seasonality")
raster::writeRaster(crop(seasonality, extent(-20, 55, -37, 38)), 
                    filename = "africa_ndvi_seasonality/ndvi_seasonality.nc", 
                    bylayer = TRUE,
                    varname = 'ndvi',
                    suffix = days,
                    overwrite = TRUE,
                    NAflag = -999)

ndvi_noseasonality <- ndvi - seasonality ### ok values are recycled

dir.create("ndvi", recursive = TRUE)
raster::writeRaster(ndvi, 
                    filename = "ndvi/ndvi.nc", 
                    bylayer = TRUE,
                    varname = 'ndvi',
                    suffix = timestamps,
                    overwrite = TRUE,
                    NAflag = -999)

dir.create("ndvi_noseasonality", recursive = TRUE)
raster::writeRaster(ndvi_noseasonality, 
                    filename = "ndvi_noseasonality/ndvi_noseasonality.nc", 
                    bylayer = TRUE,
                    varname = 'ndvi',
                    suffix = timestamps,
                    overwrite = TRUE,
                    NAflag = -999)

## save NDVI - seasonality for africa
africa_ndvi_noseasonality <- crop(ndvi_noseasonality, extent(-20, 55, -37, 38))
dir.create("africa_ndvi_noseasonality", recursive = TRUE)
raster::writeRaster(africa_ndvi_noseasonality, 
                    filename = "africa_ndvi_noseasonality/ndvi_noseasonality.nc", 
                    bylayer = TRUE,
                    varname = 'ndvi',
                    suffix = timestamps,
                    overwrite = TRUE,
                    NAflag = -999)

### save NDVI for Africa
africa_ndvi <- crop(ndvi, extent(-20, 55, -37, 38))
dir.create("africa_ndvi", recursive = TRUE)
raster::writeRaster(africa_ndvi, 
                    filename = "africa_ndvi/ndvi.nc", 
                    bylayer = TRUE,
                    varname = 'ndvi',
                    suffix = timestamps,
                    overwrite = TRUE,
                    NAflag = -999)


#### crop land cover
LC <- raster("LC_hd_global_2012_full.tif")
LC.africa <- crop(LC, extent(-2*20 + 360, 2*55 + 360, -2*37 + 180, 2*38 + 180))
writeRaster(LC.africa, filename = "africa_LC.tif", overwrite = TRUE)
