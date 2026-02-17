#!/usr/bin/env python
# Auto-generated from EMFAC-InMAP_GRID .ipynb

# %% [code cell 1]
proj_string = "+proj=lcc +lat_0=40 +lon_0=-97 +lat_1=33 +lat_2=45 +x_0=0 +y_0=0 +ellps=sphere +units=m +no_defs +type=crs"

emis_shapefile_filepath = '../BEAM_to_EMFACT/grid_polygon/grid_polygon.shp'

emissionType = 'All' #['onNetwork','offNetwork','All']


# scenarios = [
#     'sfbay-baseline_20240725_2',
#             ]
# scenarios2 = [
#     'sfbay-cordon_20240726',
#              ]
# inexus_filepaths = [
#     'sfbay_repo_transitCapacity-1.5_2020__20230608.csv.gz',
#             ]
# inexus_filepaths2 = [
#     'sfbay_baseline_default-1.0_2020__20230702.csv.gz',
#              ]

# scenario_labels = [
#     'SFMTA Cordon Policy', 
# ]

scenarios = [

             'sfbay-baseline3_20240728',
             'sfbay-baseline3_20240728',
             
#              'sfbay-tr_capacity_1_5-20230608',
#              'sfbay-tr_capacity_1_5-20230608',
#              'sfbay-tr_capacity_1_5-20230608',
             
#              'sfbay-telecommuting-baseline-20230616',
             

            ]
scenarios2 = [
        'sfbay-cordon_flatrate_20241023',
        'sfbay-cordon_income_20241023',
    
        'sfbay-tr_capacity_1_5-20230608',
        'sfbay-wb-incentives-200-20230630',
        'sfbay-tr-discount-100-20230703',
    
        'sfbay-telecommuting-8p60-20230620',
             ]
#noFutureScenario means only first scenario emissions

inexus_filepaths = [

    'sfbay-baseline3_20240728/inexus/sfbay_baseline_default-1.0_2020__20240728.csv.gz',
    'sfbay-baseline3_20240728/inexus/sfbay_baseline_default-1.0_2020__20240728.csv.gz',
    
     'sfbay-tr_capacity_1_5-20230608/inexus/sfbay_repo_transitCapacity-1.5_2020__20230608.csv.gz',
     'sfbay-tr_capacity_1_5-20230608/inexus/sfbay_repo_transitCapacity-1.5_2020__20230608.csv.gz',
     'sfbay-tr_capacity_1_5-20230608/inexus/sfbay_repo_transitCapacity-1.5_2020__20230608.csv.gz',

     'sfbay-telecommuting-baseline-20230616/inexus/sfbay_baseline_default-1.0_2020__20230616.csv.gz',



            ]

inexus_filepaths2 = [
        'sfbay-cordon_flatrate_20241023/inexus/output/sfbay_baseline_default-1.0_2020__20241024.csv.gz',
        'sfbay-cordon_income_20241023/inexus/output/sfbay_baseline_default-1.0_2020__20241024.csv.gz',
    
     'sfbay-tr_capacity_1_5-20230608/inexus/sfbay_repo_transitCapacity-1.5_2020__20230608.csv.gz',
     'sfbay-wb-incentives-200-20230630/inexus/sfbay_incentives_walk and bike-1000_2020__20230630.csv.gz', 
     'sfbay-tr-discount-100-20230703/inexus/sfbay_price_transit_price-0_2020__20230702.csv.gz',

     'sfbay-telecommuting-8p60-20230620/inexus/sfbay_baseline_default-1.0_2020__20230620.csv.gz',


             ]


scenario_labels = [
    'SFMTA Cordon Policy Flat Rate', 
    'SFMTA Cordon Policy Income-Based', 
    
    'Baseline',
    'Active Modes Incentives',
    'Transit Incentives',

    'Telecommuting'
    
    
#     'Cruise_a1b3c3d1',
#     'Cruise_a1b4c5d1',
#     'Cruise_a1b4c3d1',
#     'Comparing Baselines'

]
# descriptive_scenario_labels = ['Cordon Zone', 'Active Modes Incentives', 'Transit Incentives', 'Telecommuting']
# descriptive_scenario_labels = ['Cordon Zone - Flat Rate', 'Cordon Zone - Income Based']

# scenario = 'sfbay-baseline-20230526'
# scenario = 'sfbay-tr_capacity_1_5-20230608'
# inexus_filepath = 'sfbay_repo_transitCapacity-1.5_2020__20230608.csv.gz'
# scenario = 'sfbay-telecommuting-baseline-20230616'
# inexus_filepath = 'sfbay_baseline_default-1.0_2020__20230616.csv.gz'
# scenario = 'sfbay-baseline2018-30pct-20230825'
# scenario = 'sfbay_cruise_SAVBaseline_1_phase2_97'
# inexus_filepath = 'output/sfbay_baseline_default-1.0_2020__20240520.csv.gz'

# scenario2 = 'sfbay-cordon-MTA-20230703'
# inexus_filepath2 = 'sfbay_baseline_default-1.0_2020__20230702.csv.gz'
# scenario2 = 'sfbay-tr-30pct-20231014'
# inexus_filepath2 = 'sfbay_baseline_TR-1.0_2018__20231014.csv.gz'
# scenario2 = 'sfbay-wb-incentives-200-20230630'
# inexus_filepath2 = 'sfbay_incentives_walk and bike-1000_2020__20230630.csv.gz'
# scenario2 = 'sfbay-telecommuting-8p60-20230620'
# inexus_filepath2 = 'sfbay_baseline_default-1.0_2020__20230620.csv.gz'
# scenario2 = 'sfbay_cruise_phase2_a1b3c3d1'
# scenario2 = 'sfbay_cruise_phase2_a1b4c5d1'
# scenario2 = 'sfbay_cruise_phase2_a1b4c3d1'
# inexus_filepath2 = 'sfbay_baseline_default-1.0_2020__20240522.csv.gz'

bucket = 'beam-core-outputs'
# bucket = 'cruise-outputs'

is_NEVI = False
is_BC = False
# DAC_filepath = 'dacs_nevi_joint_May2022/dacs_nevi_joint_May2022.shp'
# DAC_crs = 'EPSG:4326'

DAC_crs = 'EPSG:3310'
is_enviro = True
DAC_filepath = 'calenviroscreen40shpf2021shp/CES4 Final Shapefile.shp'

pop_filepath = '/Users/cpoliziani/Documents/repo/beam-data-sfbay/activitysim/2018/persons.csv.gz'
hs_filepath = '/Users/cpoliziani/Documents/repo/beam-data-sfbay/activitysim/2018/households.csv.gz'


# edu_mapping = {
#     0: "No formal education", 1: "Less than 1st grade", 2: "1st, 2nd, 3rd, or 4th grade", 3: "5th or 6th grade",
#     4: "7th or 8th grade", 5: "9th grade", 6: "10th grade", 7: "11th grade", 8: "12th grade, no diploma",
#     9: "High school graduate (includes equivalency)", 10: "Some college, less than 1 year", 
#     11: "Some college, 1 or more years, no degree", 12: "Associate degree", 13: "Bachelor's degree",
#     14: "Master's degree", 15: "Professional school degree", 16: "Doctorate degree", 17: "Unknown",
#     18: "Elementary school", 19: "Middle school", 20: "High school", 21: "Technical/vocational school",
#     22: "Trade school", 23: "Online degree", 24: "Other formal education not classified elsewhere",
#     25: "Continuing education"
# }
edu_mapping = {
    0: "Other Ed.", 1: "Other Ed.", 
    2: "Other Ed.", 3: "Other Ed.", 
    4: "Other Ed.", 5: "Other Ed.", 6: "Other Ed.",
    7: "Other Ed.", 8: "Other Ed.", 
    9: "HS or higher", 10: "HS or higher",
    11: "HS or higher", 12: "HS or higher", 
    13: "HS or higher", 14: "HS or higher",
    15: "HS or higher", 16: "HS or higher", 
    17: "Other Ed.", 18: "Other Ed.", 19: "Other Ed.", 
    20: "Other Ed.", 21: "Other Ed.", 22: "Other Ed.",
    23: "Other Ed.", 24: "Other Ed.", 25: "Other Ed."
}

cordon_zones = ['060750615006',
'060750615005',
'060750615004',
'060750615003',
'060750615002',
'060750615001',
'060750611003',
'060750611002',
'060750611001',
'060750607003',
'060750607002',
'060750607001',
'060750202001',
'060750201004',
'060750201003',
'060750201002',
'060750201001',
'060750180002',
'060750180001',
'060750178022',
'060750178021',
'060750178012',
'060750178011',
'060750177002',
'060750177001',
'060750176015',
'060750176014',
'060750176013',
'060750176012',
'060750176011',
'060750168023',
'060750168021',
'060750162003',
'060750162002',
'060750162001',
'060750125022',
'060750125021',
'060750125012',
'060750125011',
'060750124023',
'060750124022',
'060750124021',
'060750124012',
'060750124011',
'060750123022',
'060750123021',
'060750123012',
'060750123011',
'060750122021',
'060750122012',
'060750122011',
'060750121002',
'060750121001',
'060750120002',
'060750120001',
'060750119022',
'060750119021',
'060750119012',
'060750119011',
'060750118001',
'060750117002',
'060750117001',
'060750113002',
'060750113001',
'060750112003',
'060750112002',
'060750112001',
'060750111003',
'060750111002',
'060750111001',
'060750110003',
'060750110002',
'060750110001',
'060750109003',
'060750109002',
'060750109001',
'060750108003',
'060750108002',
'060750108001',
'060750107004',
'060750107003',
'060750107002',
'060750107001',
'060750106003',
'060750106002',
'060750106001',
'060750105002',
'060750105001',
'060750104004',
'060750104003',
'060750104002',
'060750104001',
'060750103003',
'060750103002',
'060750103001',
'060750102003',
'060750102002',
'060750102001',
'060750101002',
'060750101001']

# Custom colormap with white at the center
def custom_colormap():
    colors = ["#0000ff", "#ffffff", "#ff0000"]
    colors = ["#0000ff", "#00aaff", "#ffff00", "#ffffff"]
    colors = ["#006837", "#00a884", "#33cfff", "#b3e5ff"]
    colors = ["#00429d", "#4caf50", "#fff176", "#ffffff"]
#     colors = ["#006400", "#32cd32", "#adff2f", "#ffff66"]
#     colors = ["#007f3f", "#00b3b3", "#4dc4ff", "#fff176", "#ffff33"] 
    n_bins = 100  # Discretizes the interpolation into bins
    cmap_name = 'custom_cmap'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

def custom_colormap2():
    colors = ["#ff0000", "#ffffff", "#0000ff"]
    n_bins = 100  # Discretizes the interpolation into bins
    cmap_name = 'custom_cmap'
    return LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# %% [code cell 2]
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from shapely.geometry import Polygon
import time
import numpy as np
import zarr
import s3fs
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import FancyArrowPatch
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from PIL import Image
import io
import numpy as np
from contextily.tile import _fetch_tile
from matplotlib.colors import LinearSegmentedColormap

def load_emis_data(emis_filepath, emis_filepath2=None):
    print('load_emis_data ...')
    emis = pd.read_csv(emis_filepath, nrows=None)
    
    if emissionType == 'onNetwork':

        emis['tons_per_year_ROG'] = emis['tons_per_year_RUNEX_ROG'] + emis['tons_per_year_RUNLOSS_ROG']  
        emis['tons_per_year_NOx'] = emis['tons_per_year_RUNEX_NOx'] 
        emis['tons_per_year_NH3'] = emis['tons_per_year_RUNEX_NH3']
        emis['tons_per_year_SOx'] = emis['tons_per_year_RUNEX_SOx'] 
        emis['tons_per_year_PM2_5'] = emis['tons_per_year_RUNEX_PM2_5'] + emis['tons_per_year_PMBW_PM2_5'] + emis['tons_per_year_PMTW_PM2_5'] 
        emis['tons_per_year_CO2'] = emis['tons_per_year_RUNEX_CO2']

    elif emissionType == 'offNetwork':

        emis['tons_per_year_ROG'] =  emis['tons_per_year_DIURN_ROG'] + emis['tons_per_year_HOTSOAK_ROG'] + emis['tons_per_year_STREX_ROG']  
        emis['tons_per_year_NOx'] = emis['tons_per_year_STREX_NOx'] 
        emis['tons_per_year_NH3'] = 0.0
        emis['tons_per_year_SOx'] = emis['tons_per_year_STREX_SOx'] 
        emis['tons_per_year_PM2_5'] = emis['tons_per_year_STREX_PM2_5']
        emis['tons_per_year_CO2'] = emis['tons_per_year_STREX_CO2']

    
    emis = emis[['GRID', 'tons_per_year_ROG', 'tons_per_year_NOx', 'tons_per_year_NH3', 'tons_per_year_SOx', 'tons_per_year_PM2_5', 'tons_per_year_CO2']]
    emis = emis.groupby('GRID').sum().reset_index()
    
    #Transfor to short tons
    emis['tons_per_year_ROG'] = emis['tons_per_year_ROG'] * 1000000 / 907185
    emis['tons_per_year_NOx'] = emis['tons_per_year_NOx'] * 1000000 / 907185
    emis['tons_per_year_NH3'] = emis['tons_per_year_NH3'] * 1000000 / 907185
    emis['tons_per_year_SOx'] = emis['tons_per_year_SOx'] * 1000000 / 907185
    emis['tons_per_year_PM2_5'] = emis['tons_per_year_PM2_5'] * 1000000 / 907185
    emis['tons_per_year_CO2'] = emis['tons_per_year_CO2'] * 1000000 / 907185

    if emis_filepath2 != emis_filepath:
        
        emis2 = pd.read_csv(emis_filepath2, nrows=None)
        
        if emissionType == 'onNetwork':
            
            emis2['tons_per_year_ROG'] = emis2['tons_per_year_RUNEX_ROG'] + emis2['tons_per_year_RUNLOSS_ROG']  
            emis2['tons_per_year_NOx'] = emis2['tons_per_year_RUNEX_NOx'] 
            emis2['tons_per_year_NH3'] = emis2['tons_per_year_RUNEX_NH3']
            emis2['tons_per_year_SOx'] = emis2['tons_per_year_RUNEX_SOx'] 
            emis2['tons_per_year_PM2_5'] = emis2['tons_per_year_RUNEX_PM2_5'] + emis2['tons_per_year_PMBW_PM2_5'] + emis2['tons_per_year_PMTW_PM2_5'] 
            emis2['tons_per_year_CO2'] = emis2['tons_per_year_RUNEX_CO2']

        elif emissionType == 'offNetwork':

            emis2['tons_per_year_ROG'] =  emis2['tons_per_year_DIURN_ROG'] + emis2['tons_per_year_HOTSOAK_ROG'] + emis2['tons_per_year_STREX_ROG']  
            emis2['tons_per_year_NOx'] = emis2['tons_per_year_STREX_NOx'] 
            emis2['tons_per_year_NH3'] = 0.0
            emis2['tons_per_year_SOx'] = emis2['tons_per_year_STREX_SOx'] 
            emis2['tons_per_year_PM2_5'] = emis2['tons_per_year_STREX_PM2_5']
            emis2['tons_per_year_CO2'] = emis2['tons_per_year_STREX_CO2']

        emis2 = emis2[['GRID', 'tons_per_year_ROG', 'tons_per_year_NOx', 'tons_per_year_NH3', 'tons_per_year_SOx', 'tons_per_year_PM2_5', 'tons_per_year_CO2']]
        emis2 = emis2.groupby('GRID').sum().reset_index()
        merged_emis = pd.merge(emis, emis2, on='GRID', how='outer', suffixes=('_1', '_2')).fillna(0)
        merged_emis['tons_per_year_ROG'] = (merged_emis['tons_per_year_ROG_2'] - merged_emis['tons_per_year_ROG_1'])
        merged_emis['tons_per_year_NOx'] = (merged_emis['tons_per_year_NOx_2'] - merged_emis['tons_per_year_NOx_1'])
        merged_emis['tons_per_year_NH3'] = (merged_emis['tons_per_year_NH3_2'] - merged_emis['tons_per_year_NH3_1'])
        merged_emis['tons_per_year_SOx'] = (merged_emis['tons_per_year_SOx_2'] - merged_emis['tons_per_year_SOx_1'])
        merged_emis['tons_per_year_PM2_5'] = (merged_emis['tons_per_year_PM2_5_2'] - merged_emis['tons_per_year_PM2_5_1'])
        merged_emis['tons_per_year_CO2'] = (merged_emis['tons_per_year_CO2_2'] - merged_emis['tons_per_year_CO2_1'])
        
        #Transfor to short tons
        merged_emis['tons_per_year_ROG'] = merged_emis['tons_per_year_ROG'] * 1000000 / 907185
        merged_emis['tons_per_year_NOx'] = merged_emis['tons_per_year_NOx'] * 1000000 / 907185
        merged_emis['tons_per_year_NH3'] = merged_emis['tons_per_year_NH3'] * 1000000 / 907185
        merged_emis['tons_per_year_SOx'] = merged_emis['tons_per_year_SOx'] * 1000000 / 907185
        merged_emis['tons_per_year_PM2_5'] = merged_emis['tons_per_year_PM2_5'] * 1000000 / 907185
        merged_emis['tons_per_year_CO2'] = merged_emis['tons_per_year_CO2'] * 1000000 / 907185
        
        calculate_emissions_differences(emis, emis2)
        
        return merged_emis
    
    else:

        return emis
907185

def load_shape_data(emis_shapefile_filepath):
    print('load_shape_data ...')
    return gpd.read_file(emis_shapefile_filepath)


def calculate_emissions_differences(emis, emis2):
    print('calculate_emissions_differences ...')
    
    def calculate_change(old, new):
        return (new / old - 1) * 100
    
    # Define GRID cordon zone and SF ranges
    cordon_ranges = [(1346, 1350), (1378, 1382), (1393, 1397), (1402, 1402), (1412, 1415)]
    SF_ranges = [(983, 986), (988, 991), (1002, 1005), (1039, 1048), 
                 (1064, 1073), (1084, 1093), (1129, 1138), (1176, 1185), 
                 (1221, 1230), (1253, 1264), (1291, 1302), (1340, 1345), 
                 (1351, 1351), (1372, 1377), (1383, 1383), (1388, 1392), 
                 (1407, 1411), (1416, 1416), (1053, 1053), (1113, 1113), 
                 (1193, 1193)]
    
    # Function to check if GRID is in the cordon zone or SF
    def is_in_cordon(grid):
        return any(start <= grid <= end for start, end in cordon_ranges)
    
    def is_in_SF(grid):
        return any(start <= grid <= end for start, end in SF_ranges)
    
    # Filter emis and emis2 for cordon zone, SF, and the rest
    emis_cordon = emis[emis['GRID'].apply(is_in_cordon)]
    emis2_cordon = emis2[emis2['GRID'].apply(is_in_cordon)]
    
    emis_SF = emis[emis['GRID'].apply(is_in_SF)]
    emis2_SF = emis2[emis2['GRID'].apply(is_in_SF)]
    
    emis_rest = emis[~emis['GRID'].apply(lambda x: is_in_cordon(x) or is_in_SF(x))]
    emis2_rest = emis2[~emis2['GRID'].apply(lambda x: is_in_cordon(x) or is_in_SF(x))]
    
    # Calculate sum of emissions for cordon, SF, and rest
    def calculate_totals(emis):
        return {
            'ROG': emis['tons_per_year_ROG'].sum(),
            'NOx': emis['tons_per_year_NOx'].sum(),
            'NH3': emis['tons_per_year_NH3'].sum(),
            'SOx': emis['tons_per_year_SOx'].sum(),
            'PM2.5': emis['tons_per_year_PM2_5'].sum(),
            'CO2': emis['tons_per_year_CO2'].sum(),
        }
    
    totals_cordon = calculate_totals(emis_cordon)
    totals_cordon2 = calculate_totals(emis2_cordon)
    
    totals_SF = calculate_totals(emis_SF)
    totals_SF2 = calculate_totals(emis2_SF)
    
    totals_rest = calculate_totals(emis_rest)
    totals_rest2 = calculate_totals(emis2_rest)
    
    # Calculate percentage changes
    def calculate_percentage_changes(totals1, totals2):
        return {key: calculate_change(totals1[key], totals2[key]) for key in totals1}
    
    delta_cordon = calculate_percentage_changes(totals_cordon, totals_cordon2)
    delta_SF = calculate_percentage_changes(totals_SF, totals_SF2)
    delta_rest = calculate_percentage_changes(totals_rest, totals_rest2)
    
    # Print the results
    print("\n--- Total Emissions ---")
    print("Cordon Zone:")
    for pollutant, value in totals_cordon.items():
        print(f"  {pollutant}: {value:.2f} tons/year")
    
    print("San Francisco (SF):")
    for pollutant, value in totals_SF.items():
        print(f"  {pollutant}: {value:.2f} tons/year")
    
    print("Rest of the Area:")
    for pollutant, value in totals_rest.items():
        print(f"  {pollutant}: {value:.2f} tons/year")
    
    print("\n--- Percentage Change in Emissions ---")
    print("Cordon Zone:")
    for pollutant, value in delta_cordon.items():
        print(f"  {pollutant}: {value:.2f}% change")
    
    print("San Francisco (SF):")
    for pollutant, value in delta_SF.items():
        print(f"  {pollutant}: {value:.2f}% change")
    
    print("Rest of the Area:")
    for pollutant, value in delta_rest.items():
        print(f"  {pollutant}: {value:.2f}% change")


def merge_emis_with_shape(merged_emis, gdf):
    print('merge_emis_with_shape ...')
    merged_emis['GRID'] = merged_emis['GRID'].astype(str).str.upper()
    gdf['grid'] = gdf['grid'].astype(str).str.upper()
    merged_emis = merged_emis.merge(gdf[['grid', 'geometry']], left_on='GRID', right_on='grid')
    emis = gpd.GeoDataFrame(merged_emis, geometry='geometry')
    emis['GRID'] = emis['GRID'].astype(int)
    emis['area'] = emis.geometry.area
    return emis[['GRID', 'tons_per_year_ROG', 'tons_per_year_NOx', 'tons_per_year_NH3', 'tons_per_year_SOx', 'tons_per_year_PM2_5', 'tons_per_year_CO2','geometry']]

def rect(i, w, s, e, n):
    x = [w[i], e[i], e[i], w[i], w[i]]
    y = [s[i], s[i], n[i], n[i], s[i]]
    return x, y

def poly(sr):
    ret = []
    w = sr["W"][:]
    s = sr["S"][:]
    e = sr["E"][:]
    n = sr["N"][:]
    for i in range(52411):
        x, y = rect(i, w, s, e, n)
        ret.append(Polygon([[x[0], y[0]], [x[1], y[1]], [x[2], y[2]], [x[3], y[3]], [x[4], y[4]]]))
    return ret

def load_inmap_data(url):
    print('load_inmap_data ...')
    fs = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name='us-east-2'))
    sr = zarr.open(s3fs.S3Map(url, s3=fs, check=False), mode="r")
    return sr, poly(sr)

def process_emission_data(emis, sr, p):
    print('process_emission_data ...')
    TotalPop = sr['TotalPop'][0:52411]
    MortalityRate = sr['MortalityRate'][0:52411]
    df = pd.DataFrame({'Location': range(52411)})
    df['Location'] = df['Location'].astype(int)
    emis['GRID'] = emis['GRID'].astype(int)
    join_right_df = df.merge(emis, left_on='Location', right_on='GRID', how='right')
    index = join_right_df.Location.tolist()
    ppl = np.unique(join_right_df.Location.tolist())
    num = range(0, len(ppl))
    dictionary = dict(zip(ppl, num))
    print(list(sr.keys()))
    SOA = sr['SOA'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("SOA data is allocated.")
    pNO3 = sr['pNO3'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("pNO3 data is allocated.")
    pNH4 = sr['pNH4'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("pNH4 data is allocated.")
    pSO4 = sr['pSO4'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("pSO4 data is allocated.")
    PM25 = sr['PrimaryPM25'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("PrimaryPM25 data is allocated.")
    BCV1 = sr['PrimaryPM25'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("BCV1 data is allocated.")
    BCV3 = sr['PrimaryPM25'].get_orthogonal_selection(([0], ppl, slice(None)))
    print("BCV2 data is allocated.")
    SOA_data, pNO3_data, pNH4_data, pSO4_data, PM25_data, BCV1_data, BCV3_data = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for i in range(len(index)):
        SOA_data += SOA[0, dictionary[index[i]], :] * emis.tons_per_year_ROG[i]
        pNO3_data += pNO3[0, dictionary[index[i]], :] * emis.tons_per_year_NOx[i]
        pNH4_data += pNH4[0, dictionary[index[i]], :] * emis.tons_per_year_NH3[i]
        pSO4_data += pSO4[0, dictionary[index[i]], :] * emis.tons_per_year_SOx[i]
        PM25_data += PM25[0, dictionary[index[i]], :] * emis.tons_per_year_PM2_5[i]
        if is_BC:
            BCV1_data += BCV1[0, dictionary[index[i]], :] * emis.tons_per_year_BCV1[i]
            BCV3_data += BCV3[0, dictionary[index[i]], :] * emis.tons_per_year_BCV3[i]
    
    data = SOA_data + pNO3_data + pNH4_data + pSO4_data + PM25_data
    
    if is_BC:
        data += BCV1_data + BCV3_data
        
    fact = 28766.639
    TotalPM25 = fact * data
#     deathsK = (np.exp(np.log(1.06) / 10 * TotalPM25) - 1) * TotalPop * 1.0465819687408728 * MortalityRate / 100000 * 1.025229357798165
#     deathsL = (np.exp(np.log(1.14) / 10 * TotalPM25) - 1) * TotalPop * 1.0465819687408728 * MortalityRate / 100000 * 1.025229357798165#     
    # Update form 2016 to 2018
#     1.0465819687408728 is the ratio between year-2016 population (what we want) and year-2010 population (what the model has). 2018 is  1.096163
#     1.025229357798165 is the ratio between year-2016 mortality rate (what we want) and year-2005 mortality rate (what the model has). 2018 is 0.960899254
    deathsK = (np.exp(np.log(1.06) / 10 * TotalPM25) - 1) * TotalPop * 1.096163 * MortalityRate / 100000 * 0.960899254
    deathsL = (np.exp(np.log(1.14) / 10 * TotalPM25) - 1) * TotalPop * 1.096163 * MortalityRate / 100000 * 0.960899254


    if is_BC:
        resultsGRID = gpd.GeoDataFrame(pd.DataFrame({
                'SOA': fact * SOA_data, 'pNO3': fact * pNO3_data, 'pNH4': fact * pNH4_data,
                'pSO4': fact * pSO4_data, 'PrimaryPM25': fact * PM25_data,
             'BCV1': fact * BCV1_data, 'BCV3': fact * BCV3_data,
            'TotalPM25': TotalPM25,
                'deathsK': deathsK, 'deathsL': deathsL
            }), geometry=p[0:52411])
    else:
        resultsGRID = gpd.GeoDataFrame(pd.DataFrame({
                'SOA': fact * SOA_data, 'pNO3': fact * pNO3_data, 'pNH4': fact * pNH4_data,
                'pSO4': fact * pSO4_data, 'PrimaryPM25': fact * PM25_data, 'TotalPM25': TotalPM25,
                'deathsK': deathsK, 'deathsL': deathsL
            }), geometry=p[0:52411])
        
    deaths = pd.DataFrame.from_dict({
    "Model": ["GRID"],
    "Krewski Deaths": [resultsGRID.deathsK.sum()],
    "LePeule Deaths": [resultsGRID.deathsL.sum()],
    })
    print(deaths)
    vsl = 9.0e6
    print(pd.DataFrame.from_dict({
        "Model": ["GRID"],
        "Krewski Damages": deaths["Krewski Deaths"] * vsl,
        "LePeule Damages": deaths["LePeule Deaths"] * vsl,
    }))

    return resultsGRID

# def plot_emis(emis, scenario, scenario2):
#     print('plot_emis ...')
#     emis = emis.to_crs(epsg=3857)
#     emis['area'] = emis.geometry.area
#     emis['tons_per_year_PM2_5/area_square_meters'] = emis['tons_per_year_PM2_5'] / emis['area'] * 2589988.11
#     emis.to_file(f'{scenario2}_{scenario}_delta_emis.shp')

#     fig, ax = plt.subplots(figsize=(15, 10))
#     emis[(emis['tons_per_year_PM2_5/area_square_meters'] > 0.001) | (emis['tons_per_year_PM2_5/area_square_meters'] < -0.001)].plot(
#         ax=ax, column='tons_per_year_PM2_5/area_square_meters', cmap='BrBG_r', legend=True,
#         legend_kwds={'label': "ΔPM$_{2.5}$ Delta Emission (Tons per Year per Square Mile)", 'orientation': "vertical"},
#         vmin=-max(abs(emis['tons_per_year_PM2_5/area_square_meters'])),
#         vmax=max(abs(emis['tons_per_year_PM2_5/area_square_meters'])), alpha=0.75
#     )
#     ax.set_xlim(-13662750, -13552000)
#     ax.set_ylim(4465000, 4585000)
#     ctx.add_basemap(ax, crs=emis.crs.to_string(), source=ctx.providers.CartoDB.Positron, alpha=1)
#     scalebar = ScaleBar(1, location='lower right', box_color='white', box_alpha=0.55, color='black', scale_loc='top')
#     ax.add_artist(scalebar)
#     north_arrow = FancyArrowPatch((0.1, 0.85), (0.1, 0.95), facecolor='black', edgecolor='black', transform=ax.transAxes, arrowstyle='-|>', mutation_scale=20)
#     ax.add_patch(north_arrow)
#     ax.text(0.1, 0.95, 'N', transform=ax.transAxes, fontsize=20, ha='center', va='bottom')
#     ax.axis('off')
#     plt.savefig(f'{scenario2}_{scenario}_PM25EmissionMap.png', dpi=600)
#     plt.show()

def plot_emis(emis, scenario, scenario2, emissionType, detail_net, is_zoom = False, poll = 'PM2_5'):
    print('plot_emis ...')
    emis = emis.to_crs(epsg=3857)
    detail_net = detail_net.to_crs(epsg=3857)
    emis['area'] = emis.geometry.area
    emis[f'tons_per_year_{poll}/area_square_meters'] = emis[f'tons_per_year_{poll}'] / emis['area'] * 2589988.11
    emis.to_file(f'{scenario2}_{scenario}_{emissionType}_delta_emis.shp')

    fig, ax = plt.subplots(figsize=(15, 10))

    detail_net.plot(ax=ax, color='grey', alpha=0.05)

    ctx.add_basemap(ax, crs=emis.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.65)
    cmap = custom_colormap()
    
    emis[(emis[f'tons_per_year_{poll}/area_square_meters'] > 0.001) | (emis[f'tons_per_year_{poll}/area_square_meters'] < -0.001)].plot(
        ax=ax, column=f'tons_per_year_{poll}/area_square_meters', cmap=cmap, legend=True,
        legend_kwds={'label': f"Δ{poll} Delta Emission (Tons per Year per Square Mile)", 'orientation': "vertical"},
        vmin=-max(abs(emis[f'tons_per_year_{poll}/area_square_meters'])),vmax=max(abs(emis[f'tons_per_year_{poll}/area_square_meters'])), 
#         vmin=-0.2,vmax=0.2, 
        alpha=0.65
    )

    if is_zoom:
        
        ax.set_xlim(-13642750, -13592000)
        ax.set_ylim(4527000, 4565000)
        
    else:
        
        ax.set_xlim(-13662750, -13552000)
        ax.set_ylim(4465000, 4585000)
        
    scalebar = ScaleBar(1, location='lower right', box_color='white', box_alpha=1, color='black', scale_loc='top')
    ax.add_artist(scalebar)
    north_arrow = FancyArrowPatch((0.1, 0.85), (0.1, 0.95), facecolor='black', edgecolor='black', transform=ax.transAxes, arrowstyle='-|>', mutation_scale=20)
    ax.add_patch(north_arrow)
    ax.text(0.1, 0.95, 'N', transform=ax.transAxes, fontsize=20, ha='center', va='bottom')
    ax.axis('off')
    plt.savefig(f'{scenario2}_{scenario}_{emissionType}_{is_zoom}_{poll}EmissionMap.png', dpi=600)
    plt.show()
    
def plot_concentrations(resultsGRID, scenario, scenario2, emissionType, proj_string, detail_net, is_zoom = False):

    print('plot_concentrations ...')
    try:
        resultsGRID = resultsGRID.set_crs(crs=proj_string)
    except:
        None


    resultsGRID = resultsGRID.to_crs(epsg=3857)
    detail_net = detail_net.to_crs(epsg=3857)  

    fig, ax = plt.subplots(figsize=(15, 10))
    detail_net.plot(ax=ax, color='grey', alpha=0.05)

    ctx.add_basemap(ax, crs=resultsGRID.crs.to_string(), source=ctx.providers.CartoDB.PositronNoLabels, alpha=0.9)
    cmap = custom_colormap()
    
    resultsGRID[(resultsGRID['TotalPM25'] > 0.005) | (resultsGRID['TotalPM25'] < -0.005)].plot(
        ax=ax, column='TotalPM25', cmap=cmap, legend=False,
        legend_kwds={'label': "ΔPM$_{2.5}$ concentration (μg m$^{-3}$)", 'orientation': "vertical"},
#         vmin=-max(abs(resultsGRID['TotalPM25'])), vmax=max(abs(resultsGRID['TotalPM25'])), alpha=0.65
        vmin=-0.08, vmax=0, 
        alpha=0.5
    )

    if is_zoom:
        
        ax.set_xlim(-13642750, -13592000)
        ax.set_ylim(4527000, 4565000)
        
    else:
        
        ax.set_xlim(-13662750, -13552000)
        ax.set_ylim(4465000, 4585000)

    # custom colorbar (full control)
    norm = mpl.colors.Normalize(vmin=-0.08, vmax=0)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
    cbar.set_label("ΔPM$_{2.5}$ concentration (μg m$^{-3}$)")

    # make the colorbar itself transparent
    if cbar.solids is not None:              # usual case (QuadMesh)
        cbar.solids.set_alpha(0.5)
    for im in cbar.ax.images:                 # fallback if rendered as an image
        im.set_alpha(0.5)

    # background box & outline
    cbar.ax.set_facecolor((1, 1, 1, 0))       # transparent bg
    cbar.outline.set_alpha(0.5)
    
    north_arrow = FancyArrowPatch((0.1, 0.85), (0.1, 0.95), facecolor='black', edgecolor='black', transform=ax.transAxes, arrowstyle='-|>', mutation_scale=20)
    ax.add_patch(north_arrow)
    ax.text(0.1, 0.95, 'N', transform=ax.transAxes, fontsize=20, ha='center', va='bottom')
    ax.axis('off')
    plt.savefig(f'{scenario2}_{scenario}_{emissionType}_{is_zoom}_PM25Map.png', dpi=600)
    plt.show()


def run_emissions(run_plots=True):
    print('main ...') 

    for scenario, scenario2 in zip(scenarios, scenarios2):

        print('####SCENARIO#####')
        print(f'####{scenario}#####')
        print(f'####{scenario2}#####')

        emis_filepath = f'../BEAM_to_EMFACT/BEAM_INMAP_detail_GRID_{scenario}.csv'
        emis_filepath2 = f'../BEAM_to_EMFACT/BEAM_INMAP_detail_GRID_{scenario2}.csv'

        emis = load_emis_data(emis_filepath, emis_filepath2)
        shape_data = load_shape_data(emis_shapefile_filepath)
        emis = merge_emis_with_shape(emis, shape_data)
        emis.to_csv(f'{scenario2}_{scenario}_{emissionType}_deltaEmis.csv')

        detail_net = gpd.read_file('/Users/cpoliziani/Downloads/toUse/InMAP/BEAM_to_EMFACT/sfbay-unclassified-unsimplified-unprojected.osm.shp')
        url = 's3://inmap-model/grid_v1.2.1.zarr/'
        sr, p = load_inmap_data(url)
        resultsGRID = process_emission_data(emis, sr, p)
        resultsGRID.to_file(f'{scenario2}_{scenario}_{emissionType}_resultsGRID.shp')
        resultsGRID.to_csv(f'{scenario2}_{scenario}_{emissionType}_resultsGRID.csv')
    if run_plots:
        plot_emis(emis, scenario, scenario2, emissionType, detail_net, poll = 'PM2_5')
    if run_plots:
        plot_emis(emis, scenario, scenario2, emissionType, detail_net, poll = 'CO2')
    if run_plots:
        plot_concentrations(resultsGRID, scenario, scenario2, emissionType, proj_string, detail_net)        
    if run_plots:
        plot_emis(emis, scenario, scenario2, emissionType, detail_net, is_zoom = True, poll = 'PM2_5')
    if run_plots:
        plot_emis(emis, scenario, scenario2, emissionType, detail_net, is_zoom = True, poll = 'CO2')
    if run_plots:
        plot_concentrations(resultsGRID, scenario, scenario2, emissionType, proj_string, detail_net, is_zoom = True)
        

# %% [code cell 3]
# Filter when there are not both inexus data for same person?Delete nans

# %% [code cell 4]
# Scatter plot of mortality rates

# url = 's3://inmap-model/grid_v1.2.1.zarr/'
# sr, p = load_inmap_data(url)
# MortalityRate = sr['MortalityRate'][0:52411]

# plt.scatter(range(len(MortalityRate)), MortalityRate, color='blue', marker='o')

# # Add title and labels
# plt.title('Scatter Plot of Mortality Rate Values')
# plt.xlabel('Index')
# plt.ylabel('Mortality Rate')
# plt.grid(True)

# # Show the plot
# plt.show()

# %% [code cell 5]
# for scenario, scenario2 in zip(scenarios, scenarios2):
#     def plot_emis(emis, scenario, scenario2, detail_net):
#         print('plot_emis ...')
#         emis = emis.to_crs(epsg=3857)
#         detail_net = detail_net.to_crs(epsg=3857)
#         emis['area'] = emis.geometry.area
#         emis['tons_per_year_PM2_5/area_square_meters'] = emis['tons_per_year_PM2_5'] / emis['area'] * 2589988.11
#         emis.to_file(f'{scenario2}_{scenario}_delta_emis.shp')

#         fig, ax = plt.subplots(figsize=(15, 10))
        
#         detail_net.plot(ax=ax, color='grey', alpha=0.05)

#         ctx.add_basemap(ax, crs=emis.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.5)
        
#         emis[(emis['tons_per_year_PM2_5/area_square_meters'] > 0.001) | (emis['tons_per_year_PM2_5/area_square_meters'] < -0.001)].plot(
#             ax=ax, column='tons_per_year_PM2_5/area_square_meters', cmap='coolwarm', legend=True,
#             legend_kwds={'label': "ΔPM$_{2.5}$ Delta Emission (Tons per Year per Square Mile)", 'orientation': "vertical"},
#             vmin=-max(abs(emis['tons_per_year_PM2_5/area_square_meters'])),
#             vmax=max(abs(emis['tons_per_year_PM2_5/area_square_meters'])), alpha=0.84
#         )
        
#         ax.set_xlim(-13662750, -13552000)
#         ax.set_ylim(4465000, 4585000)



#         scalebar = ScaleBar(1, location='lower right', box_color='white', box_alpha=0.84, color='black', scale_loc='top')
#         ax.add_artist(scalebar)
#         north_arrow = FancyArrowPatch((0.1, 0.85), (0.1, 0.95), facecolor='black', edgecolor='black', transform=ax.transAxes, arrowstyle='-|>', mutation_scale=20)
#         ax.add_patch(north_arrow)
#         ax.text(0.1, 0.95, 'N', transform=ax.transAxes, fontsize=20, ha='center', va='bottom')
#         ax.axis('off')
#         plt.savefig(f'{scenario2}_{scenario}_PM25EmissionMap.png', dpi=600)
#         plt.show()
#     def plot_concentrations(resultsGRID, scenario, scenario2, proj_string, detail_net):
        
#         print('plot_concentrations ...')
#         try:
#             resultsGRID = resultsGRID.set_crs(crs=proj_string)
#         except:
#             None
            
        
#         resultsGRID = resultsGRID.to_crs(epsg=3857)
#         detail_net = detail_net.to_crs(epsg=3857)  
        
#         fig, ax = plt.subplots(figsize=(15, 10))
#         detail_net.plot(ax=ax, color='grey', alpha=0.05)
        
#         ctx.add_basemap(ax, crs=resultsGRID.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.5)
        
#         resultsGRID[(resultsGRID['TotalPM25'] > 0.005) | (resultsGRID['TotalPM25'] < -0.005)].plot(
#             ax=ax, column='TotalPM25', cmap='coolwarm', legend=True,
#             legend_kwds={'label': "ΔPM$_{2.5}$ concentration (μg m$^{-3}$)", 'orientation': "vertical"},
#             vmin=-max(abs(resultsGRID['TotalPM25'])), vmax=max(abs(resultsGRID['TotalPM25'])), alpha=0.84
#         )
        
        
        
#         ax.set_xlim(-13662750, -13552000)
#         ax.set_ylim(4465000, 4585000)
        
#         scalebar = ScaleBar(1, location='lower right', box_color='white', box_alpha=0.84, color='black', scale_loc='top')
#         ax.add_artist(scalebar)
#         north_arrow = FancyArrowPatch((0.1, 0.85), (0.1, 0.95), facecolor='black', edgecolor='black', transform=ax.transAxes, arrowstyle='-|>', mutation_scale=20)
#         ax.add_patch(north_arrow)
#         ax.text(0.1, 0.95, 'N', transform=ax.transAxes, fontsize=20, ha='center', va='bottom')
#         ax.axis('off')
#         plt.savefig(f'{scenario2}_{scenario}_PM25Map.png', dpi=600)
#         plt.show()

#     print('####SCENARIO#####')
#     print(f'####{scenario}#####')
#     print(f'####{scenario2}#####')

#     emis_filepath = f'../BEAM_to_EMFACT/BEAM_INMAP_GRID_{scenario}.csv'
#     emis_filepath2 = f'../BEAM_to_EMFACT/BEAM_INMAP_GRID_{scenario2}.csv'

#     emis = load_emis_data(emis_filepath, emis_filepath2)
#     shape_data = load_shape_data(emis_shapefile_filepath)
#     emis = merge_emis_with_shape(emis, shape_data)

#     detail_net = gpd.read_file('/Users/cpoliziani/Downloads/toUse/InMAP/BEAM_to_EMFACT/sfbay-unclassified-unsimplified-unprojected.osm.shp')
#     url = 's3://inmap-model/grid_v1.2.1.zarr/'
#     sr, p = load_inmap_data(url)
#     resultsGRID = process_emission_data(emis, sr, p)
#     resultsGRID.to_file(f'{scenario2}_{scenario}_resultsGRID.shp')
#     plot_emis(emis, scenario, scenario2, detail_net)
#     plot_concentrations(resultsGRID, scenario, scenario2, proj_string, detail_net)
    

# %% [code cell 6]
import geopandas as gpd
import pandas as pd

def load_shape_data(emis_shapefile_filepath):
    print('load_shape_data ...')
    return gpd.read_file(emis_shapefile_filepath)

def add_geometry_id_to_dataframe(df, gdf, xcol, ycol, id_column="geometry", df_geom='epsg:32610', column='blkgrpid', df_geom2='epsg:4326'):
    print('add_geometry_id_to_dataframe ...')
    gdf.crs = df_geom2
    gdf_data = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[xcol], df[ycol]))
    gdf_data.crs = {'init': df_geom}
    joined = gpd.sjoin(gdf_data.to_crs('epsg:26910'), gdf.to_crs('epsg:26910'))
    print(joined.keys())
    gdf_data = gdf_data.merge(joined[column], left_index=True, right_index=True, how="left")
    gdf_data.rename(columns={column: id_column}, inplace=True)
    df = pd.DataFrame(gdf_data.drop(columns='geometry'))
    return df.loc[~df.index.duplicated(keep='first'), :]

def load_data(DAC_filepath, pop_filepath, bucket, scenario, inexus_filepath, scenario2, inexus_filepath2):
    print('load_data ...')
    DAC = gpd.read_file(DAC_filepath)
    
    pop = pd.read_csv(pop_filepath,
                      usecols=['person_id', 'household_id', 'earning', 'age', 'home_x', 'home_y', 'race', 'edu', 'value_of_time', 'worker', 'person_sex', 'edu', 'distance_to_work'], nrows = None)
    inexus = pd.read_csv(f'gs://{bucket}/{inexus_filepath}', usecols=['IDMerged', 'mode_choice_actual_BEAM', 'logsum_trip_Potential_INEXUS', 'distance_travelling', 'duration_travelling', 'mode_choice_actual_BEAM', 'distance_walking', 'distance_bike', 'distance_ridehail', 'distance_privateCar', 'distance_transit','fuel_marginal'], nrows = None)
    inexus2 = pd.read_csv(f'gs://{bucket}/{inexus_filepath2}', usecols=['IDMerged', 'mode_choice_actual_BEAM','logsum_trip_Potential_INEXUS', 'distance_travelling', 'duration_travelling', 'mode_choice_actual_BEAM', 'distance_walking', 'distance_bike', 'distance_ridehail', 'distance_privateCar', 'distance_transit','fuel_marginal'], nrows = None)
    car_modes = ['car','car_hov2', 'car_hov3','hov2_teleportation','hov3_teleportation','ride_hail','ride_hail_pooled', 'drive_transit', 'ride_hail_transit']
    inexus['is_car'] = inexus['mode_choice_actual_BEAM'].isin(car_modes)
    inexus2['is_car'] = inexus2['mode_choice_actual_BEAM'].isin(car_modes)


    return DAC, pop, inexus, inexus2

def process_population_data(DAC, DAC_crs, pop, shape_data, block_info, BGs):
    print('process_population_data ...')
    pop = add_geometry_id_to_dataframe(pop, shape_data, 'home_x', 'home_y', 'GRID',       df_geom='epsg:4623', column='grid',     df_geom2='+proj=lcc +lat_0=40 +lon_0=-97 +lat_1=33 +lat_2=45 +x_0=0 +y_0=0 +ellps=sphere +units=m +no_defs +type=crs')
    pop = add_geometry_id_to_dataframe(pop, DAC,        'home_x', 'home_y', 'DAC',        df_geom='epsg:4623', column='DAC',  df_geom2=DAC_crs)
    if is_enviro:
        pop = add_geometry_id_to_dataframe(pop, DAC,        'home_x', 'home_y', 'CIscoreP',        df_geom='epsg:4623', column='CIscoreP',  df_geom2=DAC_crs)
    elif is_NEVI:
        pop = add_geometry_id_to_dataframe(pop, DAC,        'home_x', 'home_y', 'DOE_DAC',        df_geom='epsg:4623', column='DOE_DAC',  df_geom2=DAC_crs)
    pop = add_geometry_id_to_dataframe(pop, BGs,        'home_x', 'home_y', 'BlockGroup', df_geom='epsg:4623', column='blkgrpid', df_geom2='epsg:4326')
    pop = pd.merge(pop, block_info, how='left', left_on='BlockGroup', right_on='bgid')
    print('Len Pop: ',len(pop))
    pop.drop_duplicates(subset='person_id', keep='first', inplace=True)
    print('Len Pop after dropping duplicates: ',len(pop))
    return pop

def replace_edu_column(df, mapping):
    df['edu'] = df['edu'].replace(mapping)
    return df


def aggregate_inexus_data(inexus, inexus2):
    print('aggregate_inexus_data ...')

    inexus_agg = inexus.groupby('IDMerged').agg({
        'logsum_trip_Potential_INEXUS': 'mean', 
        'IDMerged': 'count',
        'distance_travelling': 'sum',
        'duration_travelling': 'sum',
        'mode_choice_actual_BEAM': list,
        'distance_walking': 'sum',
        'distance_bike': 'sum',
        'distance_ridehail': 'sum',
        'distance_privateCar': 'sum',
        'distance_transit': 'sum',
        'fuel_marginal': 'sum',
        'is_car': 'sum',
    }).rename(columns={'IDMerged': 'number_of_trips'})
    
    inexus_agg2 = inexus2.groupby('IDMerged').agg({
        'logsum_trip_Potential_INEXUS': 'mean', 
        'IDMerged': 'count',
        'distance_travelling': 'sum',
        'duration_travelling': 'sum',
        'mode_choice_actual_BEAM': list,
        'distance_walking': 'sum',
        'distance_bike': 'sum',
        'distance_ridehail': 'sum',
        'distance_privateCar': 'sum',
        'distance_transit': 'sum',
        'fuel_marginal': 'sum',
        'is_car': 'sum',
    }).rename(columns={'IDMerged': 'number_of_trips'})
    
    inexus_agg['logsum_trip_Potential_INEXUS_perTrip'] = inexus_agg['logsum_trip_Potential_INEXUS'] / inexus_agg['number_of_trips']
    inexus_agg['number_active_people'] = 1

    inexus_agg2['logsum_trip_Potential_INEXUS_perTrip'] = inexus_agg2['logsum_trip_Potential_INEXUS'] / inexus_agg2['number_of_trips']
    inexus_agg2['number_active_people'] = 1
    
    return inexus_agg, inexus_agg2

def merge_population_data(pop, inexus_agg, inexus_agg2):
    print('merge_population_data ...')
    pop_merged = pop.merge(inexus_agg, right_on='IDMerged', left_on='person_id', how='left')
    pop_merged = pop_merged.merge(inexus_agg2, right_on='IDMerged', left_on='person_id', how='left', suffixes=('_1', '_2'))
    return pop_merged

# def calculate_exposure_and_density(pop, resultsGRID):
#     print('calculate_exposure_and_density ...')
# #     grouped = pop.groupby('GRID').agg({'person_id': 'count'}).reset_index()
# #     print('sum person id',grouped['person_id'].sum())
#     resultsGRID['GRID'] = resultsGRID.index.astype(int)
#     grouped['GRID'] = grouped['GRID'].astype(int)
#     resultsGRID_pop = resultsGRID.merge(grouped, on='GRID')
# #     resultsGRID_pop['area'] = resultsGRID_pop.geometry.area
# #     resultsGRID_pop['deathsK/person'] = resultsGRID_pop['deathsK']/resultsGRID_pop['person_id']*resultsGRID_pop['person_id'].sum()
#     resultsGRID_pop['GRID'] = resultsGRID_pop['GRID'].astype(int)
#     return resultsGRID_pop

# def merge_exposure_data(pop_merged, resultsGRID_pop):
#     print('merge_exposure_data ...')
#     pop_merged['GRID'] = pop_merged['GRID'].astype(int)
#     pop_merged = pop_merged.merge(resultsGRID_pop[['GRID', 'TotalPM25', 'deathsK/person']], on='GRID', how='left')
#     return pop_merged

def merge_with_households(pop_merged, households):
    print('merge_exposure_data with households...')
    pop_merged = pop_merged.merge(households[['income','household_id' ]], on='household_id',how='left')
    return pop_merged

def group_variables(pop_merged):
    print('group_variables ...')
    age_bins = [18, 65, float('inf')]
    age_labels = ['18-65', '65+']
    pop_merged['age_group'] = pd.cut(pop_merged['age'], bins=age_bins, labels=age_labels, right=False)

    dtw_bins = [-999, 0, 5, 40, float('inf')]
    dtw_labels = ['Non Worker','0-5mi', '5-40mi', '40+mi']
    pop_merged['distance_to_work'] = pop_merged['distance_to_work'].fillna(-99)
    pop_merged['dtw_group'] = pd.cut(pop_merged['distance_to_work'], bins=dtw_bins, labels=dtw_labels, right=False)

    income_bins = [-999, 90000, float('inf')]
    income_labels = ['<90k', '>=90k']
    pop_merged['income'] = pop_merged['income'].fillna(-99)

    pop_merged['income_group'] = pd.cut(pop_merged['income'], bins=income_bins, labels=income_labels, right=False)
    
    
    race_groups = {'asian': 'Other', 'white': 'Other', 'black': 'Black', 'other': 'Other'}
    print(race_groups)
    pop_merged['race_group'] = pop_merged['race'].map(race_groups)
    
    
#     print(pop_merged.keys())
    
    pop_merged['zone'] = 'Outside'
    pop_merged.loc[pop_merged['county_name'] == 'San Francisco County', 'zone'] = 'Outside'
    pop_merged.loc[pop_merged['BlockGroup'].isin(cordon_zones), 'zone'] = 'Cordon SF'

    print('Out SF', len(pop_merged[pop_merged['zone']=='Out SF']))
    print('SF', len(pop_merged[pop_merged['zone']=='SF']))
    print('Cordon SF', len(pop_merged[pop_merged['zone']=='Cordon SF']))
    

    return pop_merged

#xxx add qui la variabile Cordon, SF o outside, prima add a column wehere you calculate that
def calculate_results(pop_merged, resultsGRID, scenario2, scenario, emissionType):
    print('calculate_results ...')
    pop_merged['index'] = pop_merged.index
    pop_merged = replace_edu_column(pop_merged, edu_mapping)
    
    print(pop_merged.DAC.value_counts())
    print(pop_merged.race_group.value_counts())
    print(pop_merged.age_group.value_counts())
    print(pop_merged.income_group.value_counts())
    print(pop_merged.edu.value_counts())
    print(pop_merged.dtw_group.value_counts())
    print(pop_merged.zone.value_counts())
    print(resultsGRID.deathsL.sum())
    
#     pop_merged = pop_merged[pop_merged['age']>=18]
    
    resultsGRID['GRID'] = resultsGRID.index.astype(int)
    # Initial sum of deathsL in resultsGRID
    initial_sum_deathsL = resultsGRID['deathsL'].sum()
    print('Initial sum of deathsL:', initial_sum_deathsL)

    # Merge the dataframes
    pop_merged = pop_merged.merge(resultsGRID[['GRID', 'deathsL', 'TotalPM25']], on='GRID')
    sum_after_merge = pop_merged['deathsL'].sum()
    print('Sum after merge:', sum_after_merge)

    # Count occurrences of each GRID
    grid_counts = pop_merged['GRID'].value_counts()
    print("GRID counts:", grid_counts)

    # Divide deathsK values by the count of each GRID to distribute them evenly
    pop_merged['deathsL'] = pop_merged.apply(lambda row: row['deathsL'] / grid_counts[row['GRID']], axis=1)
    sum_after_division = pop_merged['deathsL'].sum()
    print('Sum after division:', sum_after_division)

    pop_merged.to_csv(f'{scenario2}_{scenario}_{emissionType}_PopOutcomes_disaggr.csv', index=False)
#     print(pop_merged['deathsK'].sum())
          
    results = pop_merged.groupby(['DAC', 'race_group', 'age_group', 'income_group', 'edu', 'dtw_group', 'zone']).agg({
#             'index': 'count', # one per person
            'number_active_people_1': 'sum', # one per active person
            'logsum_trip_Potential_INEXUS_perTrip_1': 'sum',
            'distance_walking_1': 'sum',
            'distance_bike_1': 'sum',
            'distance_ridehail_1': 'sum',
            'distance_privateCar_1': 'sum',
            'distance_travelling_1': 'sum',
            'duration_travelling_1': 'sum',
            'number_of_trips_1': 'sum',
            'distance_transit_1': 'sum',
            'fuel_marginal_1': 'sum',
        'is_car_1': 'sum',
            'number_active_people_2': 'sum',
            'logsum_trip_Potential_INEXUS_perTrip_2': 'sum',
            'distance_walking_2': 'sum',
            'distance_bike_2': 'sum',
            'distance_ridehail_2': 'sum',
            'distance_privateCar_2': 'sum',
            'distance_travelling_2': 'sum',
            'duration_travelling_2': 'sum',
            'number_of_trips_2': 'sum',
            'distance_transit_2': 'sum',
            'fuel_marginal_2': 'sum',
        'is_car_2': 'sum',
            'TotalPM25': 'sum',
            'deathsL': 'sum'
        }).reset_index()
    
    total_entries = pop_merged.groupby(['DAC', 'race_group', 'age_group', 'income_group', 'edu', 'dtw_group', 'zone']).size().reset_index(name='total_number_of_people')
    results = results.merge(total_entries, on=['DAC', 'race_group', 'age_group', 'income_group', 'edu', 'dtw_group', 'zone'])

    print(results.number_active_people_1.sum())
    print(results.number_of_trips_1.sum())
    print(results.total_number_of_people.sum())
    
    results = results[(results['number_active_people_1']>1)&(results['number_active_people_2']>1)]
    
    print(results.number_active_people_1.sum())
    print(results.number_of_trips_1.sum())
    print(results.total_number_of_people.sum())
    
    print('aooo', results.deathsL.sum())
    
    return results


def aggregate_results(results, groups, labels):
    label_results = {}
    
    for group in groups:
        grouped = results.groupby(group).agg({
            'total_number_of_people': 'sum',
            'number_active_people_1': 'sum',
            'logsum_trip_Potential_INEXUS_perTrip_1': 'sum',
            'distance_walking_1': 'sum',
            'distance_bike_1': 'sum',
            'distance_ridehail_1': 'sum',
            'distance_privateCar_1': 'sum',
            'distance_travelling_1': 'sum',
            'duration_travelling_1': 'sum',
            'number_of_trips_1': 'sum',
            'distance_transit_1': 'sum',
            'fuel_marginal_1': 'sum',
        'is_car_1': 'sum',
            'number_active_people_2': 'sum',
            'logsum_trip_Potential_INEXUS_perTrip_2': 'sum',
            'distance_walking_2': 'sum',
            'distance_bike_2': 'sum',
            'distance_ridehail_2': 'sum',
            'distance_privateCar_2': 'sum',
            'distance_travelling_2': 'sum',
            'duration_travelling_2': 'sum',
            'number_of_trips_2': 'sum',
            'distance_transit_2': 'sum',
            'fuel_marginal_2': 'sum',
        'is_car_2': 'sum',
            'TotalPM25': 'sum',
            'deathsL': 'sum'
        }).reset_index()
        
        print('aooo', group, grouped.deathsL.sum())
#         grouped.rename(columns={'edu': 'total_number_of_people'}, inplace=True)

        def calculate_relative_difference(col_1, col_2, num_active_1, num_active_2):
            return (grouped[col_2] / grouped[num_active_2] - grouped[col_1] / grouped[num_active_1]) / (grouped[col_1] / grouped[num_active_1])
        def calculate_average_baseline(col_1, num_active_1):
            return (grouped[col_1] / grouped[num_active_1])
        
        grouped['logsum_trip_Potential_INEXUS_perTrip'] = calculate_relative_difference('logsum_trip_Potential_INEXUS_perTrip_1', 'logsum_trip_Potential_INEXUS_perTrip_2', 'number_active_people_1', 'number_active_people_2')
        grouped['distance_walking'] = calculate_relative_difference('distance_walking_1', 'distance_walking_2', 'number_active_people_1', 'number_active_people_2')
        grouped['distance_bike'] = calculate_relative_difference('distance_bike_1', 'distance_bike_2', 'number_active_people_1', 'number_active_people_2')
        grouped['distance_ridehail'] = calculate_relative_difference('distance_ridehail_1', 'distance_ridehail_2', 'number_active_people_1', 'number_active_people_2')
        grouped['distance_privateCar'] = calculate_relative_difference('distance_privateCar_1', 'distance_privateCar_2', 'number_active_people_1', 'number_active_people_2')
        grouped['distance_travelling'] = calculate_relative_difference('distance_travelling_1', 'distance_travelling_2', 'number_active_people_1', 'number_active_people_2')
        grouped['duration_travelling'] = calculate_relative_difference('duration_travelling_1', 'duration_travelling_2', 'number_active_people_1', 'number_active_people_2')
        grouped['number_of_trips'] = calculate_relative_difference('number_of_trips_1', 'number_of_trips_2', 'number_active_people_1', 'number_active_people_2')
        grouped['number_of_car_trips'] = calculate_relative_difference('is_car_1', 'is_car_2', 'number_active_people_1', 'number_active_people_2')
        grouped['distance_transit'] = calculate_relative_difference('distance_transit_1', 'distance_transit_2', 'number_active_people_1', 'number_active_people_2')
        grouped['fuel_marginal'] = calculate_relative_difference('fuel_marginal_1', 'fuel_marginal_2', 'number_active_people_1', 'number_active_people_2')
        grouped['TotalPM25'] = grouped['TotalPM25'] / grouped['total_number_of_people']
#         grouped['deathsK/sqmi'] = grouped['deathsK/sqmi'] / grouped['total_number_of_people']
        print('grouped[total_number_of_people].sum()', grouped['total_number_of_people'].sum() )
    
#         print('grouped', grouped)
#         print('deathsK',grouped['deathsK'])
#         print('total_number_of_people',grouped['total_number_of_people'])
        grouped['deathsL'] = - grouped['deathsL'] / grouped['total_number_of_people'] * 100000
#         print('deathsK',grouped['deathsK'])
        
        grouped['speed_1'] = grouped['distance_travelling_1']/grouped['duration_travelling_1']
        grouped['speed_2'] = grouped['distance_travelling_2']/grouped['duration_travelling_2']
        grouped['speed'] = (grouped['speed_2']-grouped['speed_1'])/grouped['speed_1']
#         grouped['number_active_people'] = (results['number_active_people_1'] + results['number_active_people_2']) / 2

        grouped['AV_bs_logsum_trip_Potential_INEXUS_perTrip'] = calculate_average_baseline('logsum_trip_Potential_INEXUS_perTrip_1', 'number_active_people_1',)
        grouped['AV_bs_distance_walking'] = calculate_average_baseline('distance_walking_1', 'number_active_people_1', )
        grouped['AV_bs_distance_bike'] = calculate_average_baseline('distance_bike_1', 'number_active_people_1', )
        grouped['AV_bs_distance_ridehail'] = calculate_average_baseline('distance_ridehail_1', 'number_active_people_1', )
        grouped['AV_bs_distance_privateCar'] = calculate_average_baseline('distance_privateCar_1', 'number_active_people_1',)
        grouped['AV_bs_distance_travelling'] = calculate_average_baseline('distance_travelling_1', 'number_active_people_1', )
        grouped['AV_bs_duration_travelling'] = calculate_average_baseline('duration_travelling_1', 'number_active_people_1', )
        grouped['AV_bs_number_of_trips'] = calculate_average_baseline('number_of_trips_1', 'number_active_people_1', )
        grouped['AV_bs_number_of_car_trips'] = calculate_average_baseline('is_car_1', 'number_active_people_1', )
        grouped['AV_bs_distance_transit'] = calculate_average_baseline('distance_transit_1', 'number_active_people_1', )
        grouped['AV_bs_fuel_marginal'] = calculate_average_baseline('fuel_marginal_1', 'number_active_people_1', )
        grouped['AV_bs_speed'] = grouped['distance_travelling_1']/grouped['duration_travelling_1']

        
        grouped['Group Type'] = group
        label_results[group] = grouped[['Group Type', group] + labels + AV_bs_labels + ['total_number_of_people']]
#         label_results[group].rename(columns={group: 'group'}, inplace=True)
        for group in label_results:
            label_results[group] = label_results[group].rename(columns={group: 'group'})
    return label_results

# def calculate_delta_car_distance(pop_merged):
#     car_distance_per_grid = pop_merged.groupby('GRID').agg({
#             'distance_privateCar_1': 'sum',
#             'distance_privateCar_2': 'sum',})
#     car_distance_per_grid['distance_privateCar'] = car_distance_per_grid['distance_privateCar_2']-car_distance_per_grid['distance_privateCar_1']
    
#     return car_distance_per_grid
    
def save_label_results_to_csv(label_results, scenario, scenario2, emissionType):
    combined_df = pd.concat(label_results.values(), ignore_index=True)
    combined_df.to_csv(f'{scenario2}_{scenario}_{emissionType}_PopOutcomes_aggr.csv', index=False)

# def main():
#     print('main ...')

def run_population():
    groups = ['DAC', 'race_group', 'age_group', 'income_group', 'edu', 'dtw_group', 'zone']
    labels = [
        'TotalPM25', 'deathsL', 'distance_travelling', 'duration_travelling', 'number_of_trips', 'number_of_car_trips',
        'logsum_trip_Potential_INEXUS_perTrip', 'fuel_marginal', 'distance_walking', 'distance_bike', 'distance_ridehail',
        'distance_privateCar', 'distance_transit', 'speed'
    ]
    AV_bs_labels = [
         'AV_bs_distance_travelling', 'AV_bs_duration_travelling', 'AV_bs_number_of_trips', 'AV_bs_number_of_car_trips',
        'AV_bs_logsum_trip_Potential_INEXUS_perTrip', 'AV_bs_fuel_marginal', 'AV_bs_distance_walking', 'AV_bs_distance_bike', 'AV_bs_distance_ridehail',
        'AV_bs_distance_privateCar', 'AV_bs_distance_transit',  'AV_bs_speed'
    ]

    block_info = pd.read_csv('https://github.com/LBNL-UCB-STI/beam-core-analysis/raw/main/Users/Nazanin/JoeFish_BlockGroup_Labels/bg_w_geog_labels.csv',
                             dtype={'bgid': str, 'tractid': str})
    block_info['bgid'] = '0' + block_info['bgid'].astype(str) 
    BGs = gpd.read_file('/Users/cpoliziani/Documents/repo/beam-core-analysis/Users/Nazanin/Shapefile2010/641aa0d4-ce5b-4a81-9c30-8790c4ab8cfb202047-1-wkkklf.j5ouj.shp')

    for scenario, scenario2, inexus_filepath, inexus_filepath2 in zip(scenarios, scenarios2, inexus_filepaths, inexus_filepaths2):
        print('####SCENARIO#####')
        print(f'####{scenario}#####')
        print(f'####{scenario2}#####')

        households  = pd.read_csv(hs_filepath,
                          usecols=['household_id', 'income'], 
                      nrows = None)
        shape_data = load_shape_data(emis_shapefile_filepath)
        resultsGRID = load_shape_data(f'{scenario2}_{scenario}_{emissionType}_resultsGRID.shp')
        DAC, pop, inexus, inexus2 = load_data(DAC_filepath, pop_filepath, bucket, scenario, inexus_filepath, scenario2, inexus_filepath2)
        if is_enviro:
    #         first_quartile = DAC['CIscore'].quantile(0.25)
    #         third_quartile = DAC['CIscore'].quantile(0.75)
            DAC['DAC'] = DAC['CIscoreP'].apply(lambda x: 2 if x < 25 else (1 if x <= 75 else 0)).astype(float)
        elif is_NEVI:
            DAC['DAC'] = DAC['DOE_DAC']
        pop = process_population_data(DAC, DAC_crs, pop, shape_data, block_info, BGs)
        inexus_agg, inexus_agg2 = aggregate_inexus_data(inexus, inexus2)
        pop_merged = merge_population_data(pop, inexus_agg, inexus_agg2)
    #     resultsGRID_pop = calculate_exposure_and_density(pop, resultsGRID)
    #     pop_merged = merge_exposure_data(pop_merged, resultsGRID_pop)
        pop_merged = merge_with_households(pop_merged, households)
        pop_merged = group_variables(pop_merged)
        results = calculate_results(pop_merged, resultsGRID, scenario2, scenario, emissionType)
    #     car_distance_per_grid = calculate_delta_car_distance(pop_merged)
    #     car_distance_per_grid.to_csv(f'{scenario2}_{scenario}_DeltaCarVMT.csv')
        results.to_csv(f'{scenario2}_{scenario}_{emissionType}_PopOutcomes.csv')

        label_results = aggregate_results(results, groups, labels)
        save_label_results_to_csv(label_results, scenario, scenario2, emissionType)



def run_main(run_plots=True):
    run_emissions(run_plots=run_plots)
    run_population()


# Backward-compatible aliases
run_emissions_workflow = run_emissions
run_population_workflow = run_population
run_main_workflow = run_main

if __name__ == "__main__":
    run_main(run_plots=True)
