# LJ 3/15/2025
# Rscript to turn cmaq output of NO2/NOx ratio into isrm gridded a polygon file

rm(list=ls())

mywd = "~/Dropbox/Research/SmartGrid_Behavioral/TransportationInitiative/ATLAS/BEAM_AQM/Rscripts"
setwd(mywd)



# load packages
library(pacman)
library(dplyr)
library(mlogit)
library(tidyverse)
library(stringi)
library(parallel)
library(foreach)
library(iterators)
library(doParallel)
library(stringr)
p_load(splitstackshape)
library(data.table)
p_load(fastDummies) # for step34
p_load(R.utils) # for reading gz files
p_load(readr) # for read files from a zip foler

p_load(data.table)
p_load(dplyr)
p_load(tidyr)
p_load(ggplot2)
p_load(stringr)
p_load(vroom)
p_load(tictoc)
p_load(stargazer)
p_load(Hmisc) # weighted quantile
p_load(stringr)
p_load(readr) # read multiple csv
p_load(bit64) # convert char to integer64
p_load(sf)
p_load(rgeoda) # for spatial clustering
p_load(geodaData) # sample data for spatial clustering
p_load(units) # to drop units
# p_load(rmapshaper) # to remove isolated polygons, not working. so I write my own code to do it
# #p_load(geojson_sf) # convert to geojson and sf, but this is not available
p_load(tmap) # visualize spatial features
p_load(clusterCrit) # validate cluster
p_load(tigris)
p_load(rgdal)
p_load(raster)
p_load(ggmap)
p_load(maptiles)
p_load(tidyterra)
p_load(hexbin) # viridis color
p_load(viridisLite)
p_load(grid)
p_load(gridExtra) # plot multiple plots
#remotes::install_github("https://github.com/dkahle/ggmap")
library(ggmap) # remotes::install_github("dkahle/ggmap") # devtools::install_github("dkahle/ggmap")

##### define cmaq modeling domain
proj.baaqmd = "+proj=lcc +lat_1=60 +lat_2=30 +lon_0=-120.5 +lat_0=37 +datum=NAD83"
Ncol = 164 # number of col
Nrow = 224 # number of rows
x0= -220000 # lower left corner coordinate on the lcc projection
y0= -16000 #  lower left corner coordinate on the lcc projection
dx = 1000 # grid resolution (1km)
dy = 1000 # m (1km resolution)
dt = 3600 # s (hourly resolution)
x1= x0+Ncol*dx  # x coordinate the upper right corner
y1= y0+Nrow*dy  # y coordinate the upper right corner

CR_XY <- function(col,row){
  # function to convert baaqmd Col and Row in the center (can be the fraction of the col or row) to LCC X, Y 
  x = (col-1)*dx + x0 +dx/2
  y = (row-1)*dy + y0 +dy/2
  return(data.frame(x = x, y=y))
}


# # isrm polygon
isrm <- st_read(file.path('../Data/fromYuhan/isrm_polygon/isrm_polygon.shp'))


###########
# 1. load cmaq NOx ratio

# # the ratio is obtained by 
# pdat =  (nox.mean[,,'NO2']/nox.mean[,,'NOx']) %>%
#   reshape2::melt() %>% setNames(c('col','row','value')) 


load(file.path('../RData/cmaqtestNO2_ratio.RData')) # loaded 'pdat'
dim(pdat) 
pdat$col = pdat$x
pdat$row = pdat$y # in col, row, value
pdat[,c('x','y')] = CR_XY(pdat$col,pdat$row) # change col, row into projected x, y coordinates


# 2. turn x,y, (midpoint of the grid) into raster of 1kmx1km resolution, then to polygon
kk = rasterFromXYZ(pdat[,c('x','y','value','col','row')],res = c(1000,1000),crs = CRS(proj.baaqmd))
rtp <- rasterToPolygons(kk)

mm = st_as_sf(rtp) # turn into a sf polygon object
mm$value<-NULL

# save to shp file
st_write(mm, file.path('../RData/baaqmd.shp'), delete_layer = T)

# 3. intersect with ISRM and compute area weighted average of NO2/NOx ratio
# first make mm the same projection as isrm
mm = st_transform(mm, crs=st_crs(isrm))
kk = st_intersection(mm, isrm)
kk$area = drop_units(st_area(kk$geometry)) # in m^2
kkk = kk %>%
  left_join(pdat)%>%
  st_drop_geometry() %>%
  group_by(isrm) %>%
  summarise(value = sum(value * area)/sum(area))

#pp = asleft_join(kkk,isrm)

pp = isrm %>%
  left_join(kkk) %>%
  filter(!is.na(value)) %>%
  rename(`NO2/NOx` = value)
ggplot() +
  geom_sf(pp%>%st_transform(crs = "epsg:4326"), mapping=aes(geometry=geometry, fill = `NO2/NOx`, color = `NO2/NOx`))

no2ratio = kkk %>%
  rename( NO2_NOx_ratio = value )


save(no2ratio, file = file.path('../RData/sfb.no2ratio_isrmGRID.RData'))


