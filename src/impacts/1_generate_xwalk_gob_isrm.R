# LJ 3/15/2025
# Rscript to read in the inMap NOx output and generate NOx to NO2 ISRM for Bay area

rm(list=ls())

mywd = "~/Dropbox/Research/SmartGrid_Behavioral/TransportationInitiative/ATLAS/BEAM_AQM/Rscripts"
setwd(mywd)


datdir <- '../InMap/NOx_SFB_InMapRun/sfbay_isrm_geopoints_inmap_1.9.6'
rdatdir <- '../RData/'
source('./inmap_postprocess/initialization.R')
source('./inmap_postprocess/functions.R')
# p_load(archive) # read tar.gz
# p_load(readr) # read tar.gz
library(tictoc)

# # isrm polygon
isrm <- st_read(file.path('../Data/fromYuhan/isrm_polygon/isrm_polygon.shp'))


################
# 1. read in the file list
#filelist = untar(tarfile = file.path(datdir,"inmap1.9.6.output.tar.gz"), list = TRUE)
idlist = list.files(datdir) 
idlist = unique(substr(idlist, nchar('isrm_')+1, nchar('isrm_00843')))

# res = NULL
# 
# for(i in 1:length(idlist)){
# #for(i in 1:10){
#     sid = idlist[i]
#   isrm_source_grid = sid
#   tmp =   map_to_isrm(output_dir=datdir, isrm_source_grid=sid, isrm=isrm, bounding_box=bounding_box)
#   #tmp$isrm = sid
#   res = rbind(res, tmp)
#   print(i/length(idlist))
# }



tic()
Npe = 6

registerDoParallel(cores = Npe) 

# parallel
res <- foreach(sid = idlist, 
               .combine=rbind,
               .packages = c('tidyverse','ggplot2','sf')
)  %dopar% {
  map_to_isrm(output_dir=datdir, isrm_source_grid=sid, isrm=isrm, bounding_box=bounding_box)

}

stopImplicitCluster()

toc()


save(res, file = file.path(rdatdir, 'SFB_NOX_NOX_ISRM.RData'))

# load NO2/NOx ratio on ISRM grid

