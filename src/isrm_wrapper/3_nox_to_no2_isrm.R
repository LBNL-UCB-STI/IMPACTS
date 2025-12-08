# LJ 3/15/2025
# Rscript to turn NOx to NOx ISRM to NOx to NO2 ISRM

mywd = "~/Dropbox/Research/SmartGrid_Behavioral/TransportationInitiative/ATLAS/BEAM_AQM/Rscripts"
setwd(mywd)


datdir <- '../InMap/NOx_SFB_InMapRun/sfbay_isrm_geopoints_inmap_1.9.6'
rdatdir <- '../RData/'
source('./inmap_postprocess/initialization.R')
source('./inmap_postprocess/functions.R')

load(file = file.path(rdatdir, 'SFB_NOX_NOX_ISRM.RData')) # res
load( file = file.path('../RData/sfb.no2ratio_isrmGRID.RData')) # no2ratio

# 1. preproc NOx to NOx to the same dimension
rownames(res) = as.numeric(rownames(res))
s.isrm = as.numeric(rownames(res)) # the 910 output has NA, not in the rowname
s.isrm = s.isrm[s.isrm >3 & s.isrm != 3554]#  3354 is outside of bay area, and 910 receptor not in the receptor domain
cnames = as.numeric(colnames(res))
setdiff(s.isrm, cnames)

# keep the receptor grid of the same list of source grid
res = res[as.character(s.isrm),as.character(s.isrm)]

# find out if any isrm not in the ratio data
setdiff(s.isrm, no2ratio$isrm) # 843 not in the ratio data because of projection used in cmaq

# mannually set the 843 as background ratio of 0.94
tmp = data.frame(isrm = 843, NO2_NOx_ratio = 0.94)
no2ratio = rbind(no2ratio, tmp) 


# now multiply the receptor NOx with NO2/NOx ratio at that locaiton

dat = as.data.frame(t(res)) 
dat$isrm = as.numeric(rownames(dat))

dat = dat %>% 
  left_join(no2ratio) 
datt = dat %>%
  mutate(
    across(c(`843`:`3706`), ~ . * NO2_NOx_ratio))

# transpose back to source (rownames) to receptor (colnames)
res.dat = t(datt%>%select(`843`:`3706`))
colnames(res.dat) = dat$isrm

save(res.dat, file = file.path(rdatdir, 'NOx_to_NO2_ISRM.RData'))

# make a plot to check
library(plot.matrix)

s.idx = sample(1:dim(res.dat)[1] , 50)
pp = res.dat[s.idx, s.idx]
pp[pp<0.001] = NA
plot(pp)
