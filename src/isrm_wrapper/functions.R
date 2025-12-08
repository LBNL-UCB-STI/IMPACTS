# functions used

##################################################
# global parameters
##################################################
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



#+++++++++++++++++++++++++++++++++++++++++++
# Functions
#+++++++++++++++++++++++++++++++++++++++++++
strlen_fit <-function(x,len=16) {
  if (nchar(x)<len){
    y= ifelse(len==16 , sprintf("%-16s", x), 
              (ifelse(len==80, sprintf("%-80s", x), sprintf("%-16s", x))))
  }else{
    y=substr(x , start = 1, stop = len)
  }
  return(y)
}


CR_XY <- function(col,row){
  # function to convert baaqmd Col and Row in the center (can be the fraction of the col or row) to LCC X, Y 
  x = (col-1)*dx + x0 +dx/2
  y = (row-1)*dy + y0 +dy/2
  return(data.frame(x = x, y=y))
}

XY_CR <- function(x,y){
  # function to convert baaqmd LCC X, Y to baaqmd Col and Row 
  col = (x-x0)/dx
  row = (y-y0)/dy
  return(data.frame(COL = col, ROW = row))
}


LCCXY_LonLat <- function(lccx,lccy){
  # function to convert baaqmd xy to lon and lat
  rsytm = proj.baaqmd
  lccxy = data.frame(x = lccx, y = lccy)
  lccxy = SpatialPoints(lccxy, proj4string=CRS(rsytm))  
  domain_geo = spTransform(lccxy, CRS("+proj=longlat +datum=NAD83"))
  domain_geo <- as.data.frame(domain_geo)
  colnames(domain_geo)<-c("lon","lat")
  return(domain_geo)
}

LonLat_LCCXY <- function(lon,lat){
  # function to convert lon and lat to baaqmd x and y
  rsytm = proj.baaqmd
  latlon = data.frame(x = lon, y = lat)
  latlon = SpatialPoints(latlon, proj4string=CRS("+proj=longlat +datum=NAD83"))  
  lccxy = spTransform(latlon, CRS(rsytm))
  lccxy <- as.data.frame(lccxy)
  colnames(lccxy)<-c("lccx","lccy")
  return(lccxy)
}

get.wknd.dates <- function(year){
  # function to return dates of weekends of given year
  require(timeDate)
  dates = as.Date(paste0(year,'0101'), format = '%Y%m%d') +seq(0,ifelse(year%%4==0, 365, 364), by = 1)
  return(dates[isWeekend(dates)])
}

clearspace<-function(){
  # remove all the object from global environment
  rm(list = ls(envir = .GlobalEnv),envir = .GlobalEnv)
}


# global variables and function to map the inmap run output on gob grids to the old ISRM grids

# global variables: find out the SF bounding box
library(ggplot2)
library(sf)
sfb.counties <- map_data("county") %>% 
  filter(region == 'california') %>%
  filter(subregion %in% c(
    'alameda',
    "contra costa",
    "marin",
    "napa" ,
    "san francisco" ,
    "san mateo"  ,
    "santa clara"  ,
    "solano" ,
    "sonoma" 
  ))

# bounding box:
ylims <- c(min(sfb.counties$lat)-0.02, max(sfb.counties$lat)+0.02)
xlims <- c(min(sfb.counties$long)-0.02, max(sfb.counties$long)+0.02)
box_coords <- tibble(x = xlims, y = ylims) %>% 
  st_as_sf(coords = c("x", "y")) %>% 
  #st_set_crs(st_crs(pisrm))
  st_set_crs("EPSG:4326") #you can use the EPSG code "4326", which is the standard identifier for WGS84. 




bounding_box <- st_bbox(box_coords) %>% st_as_sfc()


map_to_isrm <- function(output_dir, isrm_source_grid, isrm, bounding_box){
  #function to map gob gridded output to isrm grid within the bounding_box
  # input: isrm_source_grid = gob gridded NOx output from perturbing NOx from this isrm grid , 
  #        isrm = isrm shp file
  # output: a dataframe with row name = isrm_source_grid, col name = sorted isrm receptor grid
  # load packages
  require(tidyverse)
  require(sf)
  require(units) # to drop units
  require(tigris)
  require(rgdal)
  require(raster)
  require(ggmap)
  require(maptiles)
  require(tidyterra)
  require(hexbin) # viridis color
  require(viridisLite)
  require(grid)
  require(gridExtra) # plot multiple plots
  #remotes::install_github("https://github.com/dkahle/ggmap")
  require(ggmap) # remotes::install_github("dkahle/ggmap") # devtools::install_github("dkahle/ggmap")
 # register_stadiamaps('9c32701b-7c88-498c-b03a-0c14ac817192') ## you can request this key yourself
  
 
  tic()
  # 1. load the test output to find out output grids. 
  dat = st_read(file.path(output_dir, paste0('isrm_', isrm_source_grid, '_geopoint.shp')))
  if(dim(dat)[1]==0)return(NA)
  
  # transform to lat and long for now
  pdat = dat %>%sf::st_transform("+proj=longlat +ellps=WGS84 +datum=WGS84")
  pisrm = isrm %>%sf::st_transform("+proj=longlat +ellps=WGS84 +datum=WGS84")
  
  #  3. get the SF portion of the isrm and dat. 
  sf.isrm = st_intersection(pisrm, bounding_box) 
  sf.dat = st_intersection(pdat, bounding_box) 
  
  
  # 4. now create xwalk between the gob grid to isrm grid. 
  
  # use projected values rather than lat and long can speed the instersection
  psf.isrm = sf.isrm %>% st_transform(isrm, crs=st_crs(isrm))
  psf.dat = sf.dat %>%
    mutate(gobid = 1:n())%>% st_transform(isrm, crs=st_crs(isrm))
  
  
  kk = st_intersection(psf.dat, psf.isrm)
  kk = kk %>%
    st_collection_extract(type = 'POLYGON') # remove points and line intersects
  
  kk$area = drop_units(st_area(kk$geometry)) # in m^2
  
  # 5. map the output to isrm grid by area weighted average of gob concentrations within each isrm grid. 
  
  kk.isrm = kk %>%
    st_drop_geometry() %>%
    group_by(isrm) %>%
    summarise(NOx = sum(NOx * area)/sum(area)) %>%
    arrange(isrm)
  
  res = data.frame(matrix(kk.isrm$NOx, nrow = 1))
  rownames(res) = isrm_source_grid
  colnames(res) = kk.isrm$isrm
  toc()
  return(res)
}

