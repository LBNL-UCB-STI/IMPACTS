library(data.table)
library(sf)
library(RANN)

############################################################
# 0. USER SETTINGS
############################################################

# ---- study area projection ----
utm_crs <- 26911   # SOCAL; Bay Area use 26910

# ---- kernel settings ----
KERNEL_RADIUS <- 1000
GRID_SIZE <- 100

# ---- unit conversion settings ----
emis_rate <- 0.1          # g/s/m2 in AERMOD unit pattern
wd_annual <- 330
us_shorttons <- 907184.74
gridarea <- 100 * 100

# ---- source filtering ----
USE_NONZERO_SOURCE_ONLY <- FALSE

# ---- pattern-selection switches ----
USE_NEAREST_SITE      <- FALSE   # FALSE = always use default site
USE_URBAN_CLASS       <- FALSE   # FALSE = always use default urban/rural
USE_TEMPORAL_CLASS    <- FALSE   # FALSE = always use default emission type
USE_RELEASE_HEIGHT    <- FALSE   # FALSE = always use default release height

# ---- default scenario ----
DEFAULT_SITE          <- "LIVERMORE_2015"
DEFAULT_URBAN_CLASS   <- 0
DEFAULT_TEMPORAL      <- "CITYSTREET"
DEFAULT_HEIGHT        <- 1

# ---- file paths ----
grid_file <- "K:/BREATHE/ISRM/MyProject-riv/SubIEGrid_new_SummarizeWithin_TableToExcel_1.csv"
pattern_file <- "K:/Cassidy/YejiaLiao/RegressionModel/full_aggregate_data_wgs.csv"
isrm_file <- "K:/BREATHE/ISRM/MyProject-riv/SubIEGrid_new_with__ISRM.csv"

output_file_main <- "K:/BREATHE/ISRM/MyProject-riv/IE_full_concentration_all_grids_dynamicPattern.csv"
output_file_with_isrm <- "K:/BREATHE/ISRM/MyProject-riv/IE_full_concentration_all_grids_dynamicPattern_withISRM.csv"

############################################################
# 1. SITE REFERENCE TABLE
############################################################

pat <- fread(pattern_file)
site_ref <- pat[
  !is.na(DataSet_ID) & !is.na(Grid_X) & !is.na(Grid_Y),
  .(
    centroid_x = mean(Grid_X, na.rm = TRUE),
    centroid_y = mean(Grid_Y, na.rm = TRUE)
  ),
  by = DataSet_ID
]

# join centroid back to all points
site_pts <- merge(
  pat[, .(DataSet_ID, Grid_X, Grid_Y)],
  site_ref,
  by = "DataSet_ID",
  all.x = TRUE
)

# distance from each point to centroid
site_pts[, dist_to_centroid := sqrt((Grid_X - centroid_x)^2 + (Grid_Y - centroid_y)^2)]

# choose the one closest to centroid for each site
site_ref <- site_pts[
  order(DataSet_ID, dist_to_centroid)
][
  , .SD[1], by = DataSet_ID
][
  , .(
    DataSet_ID,
    site_x = Grid_X,
    site_y = Grid_Y
  )
]

print(site_ref)
cat("Number of unique sites:", nrow(site_ref), "\n")
############################################################
# 2. READ GRID
############################################################

grid <- fread(grid_file)

if (!("gridID" %in% names(grid))) {
  stop("gridID does not exist in input grid.")
}

grid[, `:=`(
  lon = as.numeric(point_x),
  lat = as.numeric(point_y)
)]

grid <- grid[!is.na(lon) & !is.na(lat)]

if (!("pop" %in% names(grid))) {
  grid[, pop := 10000]
}

############################################################
# 3. READ EMISSIONS AND CONVERT TO ANNUAL SHORT TONS
############################################################

pollutants <- c("PM25", "NOx", "ROG", "SOx", "NH3")

need_cols <- c(
  "sum_totalbylink0316_csv_PM25_truck",
  "sum_totalbylink0316_csv_PM25_car",
  "sum_totalbylink0316_csv_NOx_truck",
  "sum_totalbylink0316_csv_NOx_car",
  "sum_totalbylink0316_csv_ROG_truck",
  "sum_totalbylink0316_csv_ROG_car",
  "sum_totalbylink0316_csv_SOx_truck",
  "sum_totalbylink0316_csv_SOx_car",
  "sum_totalbylink0316_csv_NH3_truck",
  "sum_totalbylink0316_csv_NH3_car"
)

for (cc in need_cols) {
  if (!(cc %in% names(grid))) grid[, (cc) := 0]
  grid[, (cc) := as.numeric(get(cc))]
  grid[is.na(get(cc)), (cc) := 0]
}

# daily grams -> annual short tons
grid[, emis_PM25 := (sum_totalbylink0316_csv_PM25_truck + sum_totalbylink0316_csv_PM25_car) * wd_annual / us_shorttons]
grid[, emis_NOx  := (sum_totalbylink0316_csv_NOx_truck  + sum_totalbylink0316_csv_NOx_car ) * wd_annual / us_shorttons]
grid[, emis_ROG  := (sum_totalbylink0316_csv_ROG_truck  + sum_totalbylink0316_csv_ROG_car ) * wd_annual / us_shorttons]
grid[, emis_SOx  := (sum_totalbylink0316_csv_SOx_truck  + sum_totalbylink0316_csv_SOx_car ) * wd_annual / us_shorttons]
grid[, emis_NH3  := (sum_totalbylink0316_csv_NH3_truck  + sum_totalbylink0316_csv_NH3_car ) * wd_annual / us_shorttons]

for (p in pollutants) {
  col <- paste0("emis_", p)
  grid[is.na(get(col)), (col) := 0]
}

############################################################
# 4. PROJECT GRID AND SITE COORDINATES
############################################################

g_sf <- st_as_sf(grid, coords = c("lon", "lat"), crs = 4326, remove = FALSE)
g_m <- st_transform(g_sf, utm_crs)
xy <- st_coordinates(g_m)

grid[, `:=`(
  gid_internal = .I,
  xm = as.numeric(xy[,1]),
  ym = as.numeric(xy[,2])
)]

site_sf <- st_as_sf(site_ref, coords = c("site_lon", "site_lat"), crs = 4326, remove = FALSE)
site_m <- st_transform(site_sf, utm_crs)
site_xy <- st_coordinates(site_m)

site_ref[, `:=`(
  site_xm = as.numeric(site_xy[,1]),
  site_ym = as.numeric(site_xy[,2])
)]

############################################################
# 5. ASSIGN SITE / URBAN / TEMPORAL / HEIGHT TO EACH GRID
############################################################

# ---- nearest site ----
site_mat <- as.matrix(site_ref[, .(site_xm, site_ym)])
grid_mat <- as.matrix(grid[, .(xm, ym)])

nn_site <- nn2(data = site_mat, query = grid_mat, k = 1)
grid[, nearest_site := site_ref$DataSet_ID[nn_site$nn.idx[,1]]]

# ---- urban class ----
# replace with actual population-density logic if available
grid[, urban_class := fifelse(pop < 1000, 0,
                              fifelse(pop < 10000, 1000, 10000))]

# ---- temporal class ----
grid[, temporal_class := "CITYSTREET"]

# ---- release height ----
grid[, release_height := 1]

# ---- apply switches ----
grid[, selected_site := if (USE_NEAREST_SITE) nearest_site else DEFAULT_SITE]
grid[, selected_urban := if (USE_URBAN_CLASS) urban_class else DEFAULT_URBAN_CLASS]
grid[, selected_temporal := if (USE_TEMPORAL_CLASS) temporal_class else DEFAULT_TEMPORAL]
grid[, selected_height := if (USE_RELEASE_HEIGHT) release_height else DEFAULT_HEIGHT]

grid[, pattern_key := paste(selected_site, selected_urban, selected_temporal, selected_height, sep = "__")]

############################################################
# 6. Process pattern library
############################################################

pat[, Concentration := as.numeric(Concentration)]
pat[, Concentration := Concentration / (emis_rate * gridarea * wd_annual * 24 * 3600 / us_shorttons)]

pat[, `:=`(
  Distance = as.numeric(Distance),
  Grid_X = as.numeric(Grid_X),
  Grid_Y = as.numeric(Grid_Y),
  Concentration = as.numeric(Concentration),
  Height = as.numeric(Height),
  Urban_Rural = as.numeric(Urban_Rural),
  DataSet_ID = as.character(DataSet_ID),
  Emissions = as.character(Emissions)
)]

pat <- pat[
  !is.na(Distance) & !is.na(Grid_X) & !is.na(Grid_Y) & !is.na(Concentration) &
    !is.na(Height) & !is.na(Urban_Rural) & !is.na(DataSet_ID) & !is.na(Emissions)
]

pat[, pattern_key := paste(DataSet_ID, Urban_Rural, Emissions, Height, sep = "__")]

############################################################
# 7. BUILD KERNEL FOR EACH PATTERN
############################################################

build_kernel_one_pattern <- function(pat_sub, kernel_radius, grid_size) {
  center_row <- pat_sub[order(abs(Distance))][1]
  
  cx <- center_row$Grid_X
  cy <- center_row$Grid_Y
  
  kernel <- pat_sub[, .(
    dx = Grid_X - cx,
    dy = Grid_Y - cy,
    resp = Concentration
  )]
  
  kernel <- kernel[!is.na(dx) & !is.na(dy) & !is.na(resp)]
  
  kernel <- kernel[, .(
    resp = mean(resp, na.rm = TRUE)
  ), by = .(dx, dy)]
  
  kernel[, dist := sqrt(dx^2 + dy^2)]
  kernel <- kernel[dist <= kernel_radius]
  
  kernel[, `:=`(
    dix = as.integer(round(dx / grid_size)),
    diy = as.integer(round(dy / grid_size))
  )]
  
  kernel <- kernel[, .(
    resp = mean(resp, na.rm = TRUE)
  ), by = .(dix, diy)]
  
  kernel
}

kernel_list <- lapply(split(pat, by = "pattern_key", keep.by = FALSE), function(pp) {
  build_kernel_one_pattern(pp, KERNEL_RADIUS, GRID_SIZE)
})

kernel_keys_available <- names(kernel_list)

cat("Number of available kernels:", length(kernel_keys_available), "\n")

############################################################
# 8. GRID INDEX FOR EXACT (x,y) GRID MATCHING
############################################################

x0 <- min(grid$xm, na.rm = TRUE)
y0 <- min(grid$ym, na.rm = TRUE)

grid[, `:=`(
  ix = as.integer(round((xm - x0) / GRID_SIZE)),
  iy = as.integer(round((ym - y0) / GRID_SIZE))
)]

target_index <- grid[, .(
  gid_internal,
  gridID,
  lon,
  lat,
  pop,
  xm,
  ym,
  ix,
  iy
)]

setkey(target_index, ix, iy)

############################################################
# 9. PREPARE SOURCE GRIDS
############################################################

if (USE_NONZERO_SOURCE_ONLY) {
  src <- grid[
    emis_PM25 != 0 | emis_NOx != 0 | emis_ROG != 0 | emis_SOx != 0 | emis_NH3 != 0,
    .(gid_internal, gridID, ix, iy,
      emis_PM25, emis_NOx, emis_ROG, emis_SOx, emis_NH3,
      nearest_site, urban_class, temporal_class, release_height,
      selected_site, selected_urban, selected_temporal, selected_height,
      pattern_key)
  ]
} else {
  src <- grid[, .(
    gid_internal, gridID, ix, iy,
    emis_PM25, emis_NOx, emis_ROG, emis_SOx, emis_NH3,
    nearest_site, urban_class, temporal_class, release_height,
    selected_site, selected_urban, selected_temporal, selected_height,
    pattern_key
  )]
}

if (nrow(src) == 0) stop("No source grids found.")

############################################################
# 10. FALLBACK RULES
############################################################

# raw requested pattern
src[, pattern_key_raw := pattern_key]

# exact pattern unavailable -> fallback to same site with defaults
src[!(pattern_key %in% kernel_keys_available),
    pattern_key := paste(selected_site, DEFAULT_URBAN_CLASS, DEFAULT_TEMPORAL, DEFAULT_HEIGHT, sep = "__")]

# still unavailable -> fallback to LIVERMORE default scenario
src[!(pattern_key %in% kernel_keys_available),
    pattern_key := paste(DEFAULT_SITE, DEFAULT_URBAN_CLASS, DEFAULT_TEMPORAL, DEFAULT_HEIGHT, sep = "__")]

missing_after_fallback <- src[!(pattern_key %in% kernel_keys_available)]
if (nrow(missing_after_fallback) > 0) {
  stop("Some source cells still have no matched pattern even after fallback.")
}

############################################################
# 11. SOURCE × CORRESPONDING KERNEL
############################################################

expanded_list <- vector("list", length = length(unique(src$pattern_key)))
names(expanded_list) <- unique(src$pattern_key)

src_keys <- unique(src$pattern_key)

for (k in src_keys) {
  src_sub <- src[pattern_key == k]
  ker_sub <- copy(kernel_list[[k]])
  
  src_sub[, tmp_join_key := 1L]
  ker_sub[, tmp_join_key := 1L]
  
  exp_sub <- merge(
    src_sub,
    ker_sub,
    by = "tmp_join_key",
    allow.cartesian = TRUE
  )
  
  exp_sub[, tmp_join_key := NULL]
  
  exp_sub[, `:=`(
    target_ix = ix + dix,
    target_iy = iy + diy
  )]
  
  expanded_list[[k]] <- exp_sub
}

expanded <- rbindlist(expanded_list, use.names = TRUE, fill = TRUE)

############################################################
# 12. MATCH TARGET GRID BY EXACT GRID COORDINATES
############################################################

setkey(expanded, target_ix, target_iy)

expanded <- target_index[
  expanded,
  on = .(ix = target_ix, iy = target_iy),
  nomatch = 0L,
  allow.cartesian = TRUE
]

setnames(
  expanded,
  old = c("gid_internal", "gridID", "lon", "lat", "pop", "xm", "ym", "ix", "iy"),
  new = c("target_gid_internal", "target_gridID", "target_lon", "target_lat",
          "target_pop", "target_xm", "target_ym", "target_ix_real", "target_iy_real")
)

setnames(
  expanded,
  old = c("i.gid_internal", "i.gridID", "i.ix", "i.iy"),
  new = c("source_gid_internal", "source_gridID", "source_ix", "source_iy")
)

############################################################
# 13. CALCULATE CONTRIBUTION FROM EACH SOURCE TO TARGET
############################################################

expanded[, `:=`(
  add_PM25 = emis_PM25 * resp,
  add_NOx  = emis_NOx  * resp,
  add_ROG  = emis_ROG  * resp,
  add_SOx  = emis_SOx  * resp,
  add_NH3  = emis_NH3  * resp
)]

############################################################
# 14. ACCUMULATE CONCENTRATIONS TO TARGET GRID
############################################################

conc <- expanded[, .(
  concentration_PM25 = sum(add_PM25, na.rm = TRUE),
  concentration_NOx  = sum(add_NOx,  na.rm = TRUE),
  concentration_ROG  = sum(add_ROG,  na.rm = TRUE),
  concentration_SOx  = sum(add_SOx,  na.rm = TRUE),
  concentration_NH3  = sum(add_NH3,  na.rm = TRUE)
), by = .(target_gid_internal, target_gridID)]

setnames(conc, c("target_gid_internal", "target_gridID"), c("gid_internal", "gridID"))

############################################################
# 15. WRITE BACK TO ALL GRID CELLS
############################################################

result <- copy(grid)

result[, `:=`(
  concentration_PM25 = 0.0,
  concentration_NOx  = 0.0,
  concentration_ROG  = 0.0,
  concentration_SOx  = 0.0,
  concentration_NH3  = 0.0
)]

setkey(result, gid_internal)
setkey(conc, gid_internal)

result[conc, `:=`(
  concentration_PM25 = i.concentration_PM25,
  concentration_NOx  = i.concentration_NOx,
  concentration_ROG  = i.concentration_ROG,
  concentration_SOx  = i.concentration_SOx,
  concentration_NH3  = i.concentration_NH3
)]

############################################################
# 16. OUTPUT MAIN RESULT
############################################################

options(scipen = 999)

result[, `:=`(
  concentration_PM25 = round(as.numeric(concentration_PM25), 6),
  concentration_NOx  = round(as.numeric(concentration_NOx),  6),
  concentration_ROG  = round(as.numeric(concentration_ROG),  6),
  concentration_SOx  = round(as.numeric(concentration_SOx),  6),
  concentration_NH3  = round(as.numeric(concentration_NH3),  6)
)]

result_out <- result[, .(
  gridID,
  gid_internal,
  lon,
  lat,
  pop,
  xm,
  ym,
  ix,
  iy,
  nearest_site,
  urban_class,
  temporal_class,
  release_height,
  selected_site,
  selected_urban,
  selected_temporal,
  selected_height,
  pattern_key,
  emis_PM25,
  emis_NOx,
  emis_ROG,
  emis_SOx,
  emis_NH3,
  concentration_PM25,
  concentration_NOx,
  concentration_ROG,
  concentration_SOx,
  concentration_NH3
)]

num_cols <- c(
  "emis_PM25","emis_NOx","emis_ROG","emis_SOx","emis_NH3",
  "concentration_PM25","concentration_NOx","concentration_ROG",
  "concentration_SOx","concentration_NH3"
)

for (cc in num_cols) {
  result_out[, (cc) := as.numeric(get(cc))]
}

fwrite(result_out, output_file_main)

cat("Output complete:\n")
cat("1)", output_file_main, "\n")
cat("Input grid rows:", nrow(grid), "\n")
cat("Output rows:", nrow(result_out), "\n")

############################################################
# 17. MERGE ISRM CONCENTRATION PRODUCT
############################################################

ISRM_Product <- fread(isrm_file)

setnames(ISRM_Product, c(
  "SubIEGrid_new.OID",
  "SubIEGrid_new.Shape_Length",
  "SubIEGrid_new.Shape_Area",
  "SubIEGrid_new.gid",
  "SubIEGrid_new_AddSpatialJoin.OBJECTID",
  "SubIEGrid_new_AddSpatialJoin.Join_Count",
  "SubIEGrid_new_AddSpatialJoin.TARGET_FID",
  "SubIEGrid_new_AddSpatialJoin.isrm",
  "SubIEGrid_new_AddSpatialJoin.ISRM_1",
  "SubIEGrid_new_AddSpatialJoin.SOA",
  "SubIEGrid_new_AddSpatialJoin.pNO3",
  "SubIEGrid_new_AddSpatialJoin.pNH4",
  "SubIEGrid_new_AddSpatialJoin.pSO4",
  "SubIEGrid_new_AddSpatialJoin.PrimaryPM25",
  "SubIEGrid_new_AddSpatialJoin.TotalPM25",
  "SubIEGrid_new_AddSpatialJoin.Shape_Length",
  "SubIEGrid_new_AddSpatialJoin.Shape_Area"
), c(
  "OID",
  "Shape_Length",
  "Shape_Area",
  "gid",
  "join_OBJECTID",
  "join_count",
  "target_fid",
  "isrm",
  "ISRM_dup",
  "SOA",
  "pNO3",
  "pNH4",
  "pSO4",
  "PrimaryPM25",
  "TotalPM25",
  "join_Shape_Length",
  "join_Shape_Area"
))

ISRM_Product[, secondaryPM25 := TotalPM25 - PrimaryPM25]
ISRM_Product <- ISRM_Product[, .(
  gid, pNO3, pNH4, SOA, pSO4, PrimaryPM25, secondaryPM25, TotalPM25
)]

result_out <- merge(result_out, ISRM_Product, by.x = "gridID", by.y = "gid", all.x = TRUE)

result_out[, PrimaryPM25_integrated_ISRM := fifelse(
  concentration_PM25 == 0, PrimaryPM25, concentration_PM25
)]

result_out[, newTotalPM25 := PrimaryPM25_integrated_ISRM + secondaryPM25]

result_out[, PrimaryPM25_diff := fifelse(
  is.na(PrimaryPM25) | PrimaryPM25 == 0,
  NA_real_,
  (-PrimaryPM25_integrated_ISRM + PrimaryPM25) / PrimaryPM25
)]

fwrite(result_out, output_file_with_isrm)

cat("2)", output_file_with_isrm, "\n")