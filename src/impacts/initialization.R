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
register_stadiamaps('9c32701b-7c88-498c-b03a-0c14ac817192') ## you can request this key yourself

p_load(tidycensus)

census_api_key("2e9a585abf38b5ee3cb67040b334942f82ec38e1")
