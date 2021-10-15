n <- 128
message("image size set to: ", n) 
tt <- 2000
message("number of time steps set to: ", tt) 
d <- n / 2

dir.create("databases/data/toy2/", showWarnings = FALSE, recursive = TRUE)
cube <- array(data =0, dim = c(n,n,tt))
true <- sin(1:tt / 10) + runif(tt)
hidden <- sin( 1:tt / 3) + runif(tt)
m1 <- mean(cube[1:d, 1:d, 1])
target <-vector(mode = 'double', length = tt)
target[1] <- m1
for (t in 2:tt){
  cube[1:d, 1:d, t] <-  0.5*cube[1:d, 1:d, t-1] + 0.5 * true[t-1] + runif(d*d, min = -1, max = 1)
  cube[(n+1-d):n, (n+1-d):n, t] <- 0.5*cube[(n+1-d):n, (n+1-d):n, t - 1] + 0.5 * hidden[t-1] + runif(d*d, min = -1, max = 1)
}

library(raster)
library(rgdal)

message("saving target time series")
write.table(true, file = "databases/data/toy2/target.txt", append = FALSE, row.names = FALSE, col.names = FALSE)

message("saving slices") 
cube <- 255 * (cube - min(cube)) / (max(cube) - min(cube)) 
for (t in 1:tt){
  writeGDAL(as(raster(cube[,,t]), Class = "SpatialPixelsDataFrame"), 
            fname = paste0("databases/data/toy2/cube_slice",
                           formatC(t, width = 3, flag = "0"),".png"), 
            drivername = "PNG",type = "Byte")
  file.remove(paste0("databases/data/toy2/cube_slice", formatC(t, width = 3, flag = "0"), ".png.aux.xml"))
}
message("done")
