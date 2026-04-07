## This code applies 1KM X 1KM NO2/NOx ratios derivedfrom CMAQ to estimate 100m x 100m grid NO2 concentrations 
## from SFBAY ISRM-predicted NOx concentrations and 100m x 100m grid NOx concentrations.
library(tidyverse)
load(file = "NOx_to_NO2_ISRM.RData")
load(file="sfb.no2ratio_isrmGRID.RData")

####################################
# 1. Calculate NO2 for ISRM grid 
####################################
### Assume that ISRM-grid NOx emissions have already been obtained:
ISRM_NOX_emission = data.frame(isrm_id = c(), nox_us_tons = c())

# make sure source IDs in emission table match the rownames of res.dat
ISRM_NOX_emission$isrm_id <- as.character(ISRM_NOX_emission$isrm_id)
rownames(res.dat) <- as.character(rownames(res.dat))
colnames(res.dat) <- as.character(colnames(res.dat))

# keep only source grids that exist in both emission table and response matrix
common_ids <- intersect(ISRM_NOX_emission$isrm_id, rownames(res.dat))

# build emission vector in the same order as response matrix rows
emis_vec <- ISRM_NOX_emission$`nox_us_ton`[match(common_ids, ISRM_NOX_emission$isrm_id)]

# subset response matrix
res_sub <- res.dat[common_ids, , drop = FALSE]

# calculate receptor concentrations
conc_vec <- as.numeric(t(emis_vec) %*% as.matrix(res_sub))

# output
ISRM_NOx_conc <- data.frame(
  isrm_id = colnames(res_sub),
  isrm_nox_conc = conc_vec
)


####################################
# 2. Calculate NO2 for 100m x 100m grid
####################################
### Use the calculated outputs from the AERMOD + InMAP PM2.5 results
result_out$isrm = as.character(result_out$isrm)
result_out = left_join(result_out,ISRM_NOx_conc,by=c("isrm"="isrm_id")) ##### background NO2 from ISRM

no2ratio$isrm = as.character(no2ratio$isrm)
result_out = result_out %>%
  left_join(no2ratio, by = "isrm") %>%
  mutate(
    grid_NO2_conc = emis_NOx * NO2_NOx_ratio
  )

# if no nox in 100m x 100m grid, replace by the ISRM NO2 derived in last step
result_out$integrated_NO2 = ifelse(result_out$grid_NO2_conc==0,isrm_nox_conc,result_out$grid_NO2_conc) 





