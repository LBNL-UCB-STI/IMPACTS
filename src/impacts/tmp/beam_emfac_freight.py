#!/usr/bin/env python
# Auto-generated from BEAM_EMFAC-3-withFreight.ipynb

# %% [code cell 1]
########### AUTHOR: CRISTIAN POLIZIANI, PhD ################
######## Lawrence Berkeley National Laboratory #############
############################################################
###################### May 2, 2024 #########################
############################################################
######SCOPE: Provide a BEAM CORE output link and ###########
######## prepare pollutant emissions for INMAP #############
############################################################

# Match beam freight vehicles to the EMFAC
# Add additional processes/pollutants for freight

# %% [code cell 2]
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import bisect
import time
import gc

try:
    from impacts.network2grid.network_grid_clipping import (
        intersect_beam_osm_with_grid,
        load_mapping_config,
        map_beam_network_to_osm,
    )
except ImportError:
    from network2grid.network_grid_clipping import (
        intersect_beam_osm_with_grid,
        load_mapping_config,
        map_beam_network_to_osm,
    )

 

#############################################################
########################## INPUTS ###########################
#############################################################

# other_county = 'San Francisco County' # When a link falls in the water or outside county boundaries, like on bridges, consider it under this EMFAC county in order not to be excluded for emisison calculation
                                      # However, this is replaced with Oter for plots on gridToCountyDict

# corr_VMT_by_county_dict = {('Alameda County', 'LDA'): 0.8382584178761276, ('Alameda County', 'LDT1'): 0.8382955189879107, ('Alameda County', 'LDT2'): 0.8384195964699777, ('Alameda County', 'MDV'): 0.8384078994409043, ('Alameda County', 'UBUS'): 1.0, ('Contra Costa County', 'LDA'): 1.0364590034773495, ('Contra Costa County', 'LDT1'): 1.0354872989789385, ('Contra Costa County', 'LDT2'): 1.036609238618241, ('Contra Costa County', 'MDV'): 1.036809838645456, ('Contra Costa County', 'UBUS'): 1.0, ('Marin County', 'LDA'): 0.8350251329301085, ('Marin County', 'LDT1'): 0.8364646611758678, ('Marin County', 'LDT2'): 0.8354799512452032, ('Marin County', 'MDV'): 0.8354492129920658, ('Marin County', 'UBUS'): 1.0, ('Napa County', 'LDA'): 0.8461647920705302, ('Napa County', 'LDT1'): 0.8458556105737716, ('Napa County', 'LDT2'): 0.8464635131944629, ('Napa County', 'MDV'): 0.8467133218850701, ('Napa County', 'UBUS'): 1.0, ('San Francisco County', 'LDA'): 0.9632673134665494, ('San Francisco County', 'LDT1'): 0.9611247721072494, ('San Francisco County', 'LDT2'): 0.9635388441890713, ('San Francisco County', 'MDV'): 0.963717170418973, ('San Francisco County', 'UBUS'): 1.0, ('San Mateo County', 'LDA'): 0.8665326830803317, ('San Mateo County', 'LDT1'): 0.8668840511668878, ('San Mateo County', 'LDT2'): 0.866152509448014, ('San Mateo County', 'MDV'): 0.8671667712889382, ('San Mateo County', 'UBUS'): 1.0, ('Santa Clara County', 'LDA'): 0.8280820774305037, ('Santa Clara County', 'LDT1'): 0.8278832876939534, ('Santa Clara County', 'LDT2'): 0.8278219637335007, ('Santa Clara County', 'MDV'): 0.8284479156732447, ('Santa Clara County', 'UBUS'): 1.0, ('Solano County', 'LDA'): 1.295980878198694, ('Solano County', 'LDT1'): 1.294765565973038, ('Solano County', 'LDT2'): 1.295497537235339, ('Solano County', 'MDV'): 1.2963054635632012, ('Solano County', 'UBUS'): 1.0, ('Sonoma County', 'LDA'): 1.4011986791374236, ('Sonoma County', 'LDT1'): 1.4010809579562558, ('Sonoma County', 'LDT2'): 1.4010050148776163, ('Sonoma County', 'MDV'): 1.4019799515123095, ('Sonoma County', 'UBUS'): 1.0}
# corr_Trips_by_county_dict = {('Alameda County', 'LDA'): 0.7047680690771602, ('Alameda County', 'LDT1'): 0.689314393083685, ('Alameda County', 'LDT2'): 0.7267671182103814, ('Alameda County', 'MDV'): 0.6907161567740274, ('Alameda County', 'UBUS'): 1.0, ('Contra Costa County', 'LDA'): 0.802898225985577, ('Contra Costa County', 'LDT1'): 0.7919667293056307, ('Contra Costa County', 'LDT2'): 0.843262307240611, ('Contra Costa County', 'MDV'): 0.7991784461558805, ('Contra Costa County', 'UBUS'): 1.0, ('Marin County', 'LDA'): 0.5510642938592881, ('Marin County', 'LDT1'): 0.5328783005592662, ('Marin County', 'LDT2'): 0.5743851117136962, ('Marin County', 'MDV'): 0.5523760316395268, ('Marin County', 'UBUS'): 1.0, ('Napa County', 'LDA'): 0.626636904959386, ('Napa County', 'LDT1'): 0.5632007875904346, ('Napa County', 'LDT2'): 0.6022609869579018, ('Napa County', 'MDV'): 0.5708766475232429, ('Napa County', 'UBUS'): 1.0, ('San Francisco County', 'LDA'): 1.2883514823468911, ('San Francisco County', 'LDT1'): 1.168491928257591, ('San Francisco County', 'LDT2'): 1.2189083712998492, ('San Francisco County', 'MDV'): 1.2984693895455655, ('San Francisco County', 'UBUS'): 1.0, ('San Mateo County', 'LDA'): 0.7523217307547722, ('San Mateo County', 'LDT1'): 0.7008868316076761, ('San Mateo County', 'LDT2'): 0.7411828603484104, ('San Mateo County', 'MDV'): 0.7427945003041081, ('San Mateo County', 'UBUS'): 1.0, ('Santa Clara County', 'LDA'): 0.9091343513181342, ('Santa Clara County', 'LDT1'): 0.839523566483252, ('Santa Clara County', 'LDT2'): 0.8689378108899927, ('Santa Clara County', 'MDV'): 0.8388883504691426, ('Santa Clara County', 'UBUS'): 1.0, ('Solano County', 'LDA'): 1.1853792192875006, ('Solano County', 'LDT1'): 1.0342434543771066, ('Solano County', 'LDT2'): 1.1050388141925946, ('Solano County', 'MDV'): 1.0675685971050919, ('Solano County', 'UBUS'): 1.0, ('Sonoma County', 'LDA'): 1.0796644236379482, ('Sonoma County', 'LDT1'): 0.9756880678880802, ('Sonoma County', 'LDT2'): 1.0448094446183394, ('Sonoma County', 'MDV'): 1.0265947791976873, ('Sonoma County', 'UBUS'): 1.0}

cruise_class_type = 'LDA'
cruise_fuel_type = 'Elec'

scenarios = [
#     'sfbay-baseline3_20240728',
#     'sfbay-cordon_flatrate_20241023',
#     'sfbay-cordon_income_20241023',
    
#             'sfbay-tr_capacity_1_5-20230608',
#             'sfbay-wb-incentives-200-20230630',
#             'sfbay-tr-discount-100-20230703',
             
#             'sfbay-telecommuting-baseline-20230616',
#             'sfbay-telecommuting-8p60-20230620',
             
#             'sfbay_cruise_SAVBaseline_1_phase2_97',
#             'sfbay_cruise_phase2_a1b3c3d1',
#             'sfbay_cruise_phase2_a1b4c5d1',
#             'sfbay_cruise_phase2_a1b4c3d1',
             
#             'sfbay-baseline2018-30pct-20230825',
#             'sfbay-tr-30pct-20231014'
    
#             'sfbay-emissions--20240123--2018-Baseline__2025-04-10_03-11-41_sjh',
            'sfbay-emissions--20240123--2018-Baseline__2025-04-13_21-05-01_iek'
    
            ]


#Scenarios must be in order: when is baseline, calculate 
# the correction factors to be used for the next scenarios, 
# until there is a next baseline
are_baseline = [
#                 True,
#                 False,
#                 False,
    
#                 True,
#                 False,
#                 False,

#                 True,
#                 False,

#                 True,
#                 False,
#                 False,
#                 False,
    
#                 True,
#                 False,
    
#     True,
    True
               ]
years = [
#     '2020',
#     '2020',
#     '2020',
    
#         '2020',
#         '2020',
#         '2020',
         
#         '2020',
#         '2020',
         
#         '2020',
#         '2020',
#         '2020',
#         '2020',
        
#         '2018',
#         '2018'
    
#     '',
    ''
                ]
iterations = [
#     '3',
#     '3',
#     '3',
    
#             '4',
#              '4',
#              '4',
              
#              '4',
#              '4',
              
#              '3',
#              '3',
#              '3',
#              '3',
                
#             '10',
#             '10',
    
#     '',
    ''
                    ]

scales = [
#     0.1,
#     0.1,
#     0.1,
    
#             0.1,
#          0.1,
#          0.1,
          
#          0.1,
#          0.1,
          
#          0.1,
#          0.1,
#          0.1,
#          0.1,
         
#          0.3,
#          0.3,
    
#     0.1,
    0.1
            ]

buckets = [
#             'beam-core-outputs',
#             'beam-core-outputs',
#             'beam-core-outputs',
    
#             'beam-core-outputs',
#           'beam-core-outputs',
#           'beam-core-outputs',
           
#           'beam-core-outputs',
#           'beam-core-outputs',
           
#           'cruise-outputs',
#           'cruise-outputs',
#           'cruise-outputs',
#           'cruise-outputs',
           
#           'beam-core-outputs',
#           'beam-core-outputs',
    
#     'beam-core-outputs/output/testing/job_67f72d0565516f2aaa14d443'
    'beam-core-outputs/output/testing/job_67fc238a350864f86e0f2c0b'
          ]

are_cruise = [
#     False, 
#     False, 
#     False, 
    
#             False,
#               False,
#               False,
              
#               False,
#               False,
              
#               True,
#               True,
#               True,
#               True, 
            
#               False,
#               False
    
#     False,
    False
]

is_HaitamTest = [
#     False, 
#     False, 
#     False, 
    
#             False,
#               False,
#               False,
              
#               False,
#               False,
              
#             False,
#               False,
#               False,
#               False,
            
#               False,
#               False
    
#     True,
    True
]

is_baseline = True # Recalculate correction factors

is_cruise = False # Correct Cruise vehicle types

# scenario = 'sfbay-baseline-20230526'
# scenario = 'sfbay-tr_capacity_1_5-20230608'
# scenario = 'sfbay-cordon-MTA-20230703'
# scenario = 'sfbay-wb-incentives-200-20230630'

# scenario = 'sfbay-baseline2018-30pct-20230825'
# scenario = 'sfbay-tr-30pct-20231014'

# scenario = 'sfbay-telecommuting-baseline-20230616'
# scenario = 'sfbay-telecommuting-8p60-20230620'

# scenario = 'sfbay_cruise_SAVBaseline_1_phase2_97'
# scenario = 'sfbay_cruise_phase2_a1b3c3d1'
# scenario = 'sfbay_cruise_phase2_a1b4c5d1'
# scenario = 'sfbay_cruise_phase2_a1b4c3d1'


# year = '2018'
# iteration = '10'
# scale = 0.3

# year = '2020'
# iteration = '4'
# scale = 0.1
# bucket = 'beam-core-outputs'



nrows_events = None

geoAggregationType = 'GRID' 
correction_factor_curvature = 1.0
block_info = pd.read_csv('/Users/cpoliziani/Documents/repo/beam-core-analysis/Users/Nazanin/JoeFish_BlockGroup_Labels/bg_w_geog_labels.csv',
                         dtype={'bgid': str, 'tractid': str})
block_info['bgid'] = '0' + block_info['bgid'].astype(str) 
BGs = gpd.read_file('/Users/cpoliziani/Documents/repo/beam-core-analysis/Users/Nazanin/Shapefile2010/641aa0d4-ce5b-4a81-9c30-8790c4ab8cfb202047-1-wkkklf.j5ouj.shp')
GRID_grid = gpd.read_file('grid_polygon/grid_polygon.shp')
counties_shapefile = gpd.read_file('/Users/cpoliziani/Downloads/region_county_5910830457166027147/region_county.shp')

# Optional BEAM-OSM-GRID mapping configuration.
WORKFLOW_CONFIG_PATH = "src/impacts/settings.yaml"
MAPPING_CFG = load_mapping_config(WORKFLOW_CONFIG_PATH)

EMFACT_VMT_filepath = 'Default_MTC_2018_Annual_vmt_20240313111517.csv'
EMFACT_trips_filepath = 'Default_MTC_2018_Annual_trips_20240313111517.csv'
EMFACTemfactFilepath = 'imputed_MTC_emission_rate_agg_NH3_added.csv'
EMFACTemissionsFilepath = 'Default_MTC_2018_Annual_emission_20240313111517.csv'
segment_length = 300  # meters
 
days = 330
default_speed = 40 #mph when not specified
parking_limit = 60
end_day_parking = 29*3600
beginning_day_parking = (5*3600)
seed = 105

correction_factor_duration = 1.0
   

# #Baseline
# corr_VMT_by_county_dict =   {('Alameda County', 'LDA'): 0.8309246072433601, ('Alameda County', 'LDT1'): 0.8310758447862574, ('Alameda County', 'LDT2'): 0.8313088623777907, ('Alameda County', 'MDV'): 0.8310083342522308, ('Alameda County', 'UBUS'): 1.0, ('Contra Costa County', 'LDA'): 1.033735832908966, ('Contra Costa County', 'LDT1'): 1.0338351289963696, ('Contra Costa County', 'LDT2'): 1.0333044923754449, ('Contra Costa County', 'MDV'): 1.0343219194927749, ('Contra Costa County', 'UBUS'): 1.0, ('Marin County', 'LDA'): 0.8592816168500973, ('Marin County', 'LDT1'): 0.8600807360838104, ('Marin County', 'LDT2'): 0.8604278805799475, ('Marin County', 'MDV'): 0.8601664274778759, ('Marin County', 'UBUS'): 1.0, ('Napa County', 'LDA'): 0.8482688339393221, ('Napa County', 'LDT1'): 0.8488538654672606, ('Napa County', 'LDT2'): 0.8474093148764735, ('Napa County', 'MDV'): 0.8482049183258689, ('Napa County', 'UBUS'): 1.0, ('San Francisco County', 'LDA'): 0.9549358551340453, ('San Francisco County', 'LDT1'): 0.9551103178545439, ('San Francisco County', 'LDT2'): 0.9544763590079611, ('San Francisco County', 'MDV'): 0.9539689509490054, ('San Francisco County', 'UBUS'): 1.0, ('San Mateo County', 'LDA'): 0.8747048647328706, ('San Mateo County', 'LDT1'): 0.8745750344012946, ('San Mateo County', 'LDT2'): 0.8748760656998817, ('San Mateo County', 'MDV'): 0.874774762534271, ('San Mateo County', 'UBUS'): 1.0, ('Santa Clara County', 'LDA'): 0.8295052204636604, ('Santa Clara County', 'LDT1'): 0.829417216568818, ('Santa Clara County', 'LDT2'): 0.8294559098635733, ('Santa Clara County', 'MDV'): 0.829646215560378, ('Santa Clara County', 'UBUS'): 1.0, ('Solano County', 'LDA'): 1.3250414194391953, ('Solano County', 'LDT1'): 1.3263127547052709, ('Solano County', 'LDT2'): 1.3247367057668684, ('Solano County', 'MDV'): 1.3258652425693782, ('Solano County', 'UBUS'): 1.0, ('Sonoma County', 'LDA'): 1.4237514900884463, ('Sonoma County', 'LDT1'): 1.4230662335837894, ('Sonoma County', 'LDT2'): 1.4241932671740207, ('Sonoma County', 'MDV'): 1.424221619330168, ('Sonoma County', 'UBUS'): 1.0}
# corr_Trips_by_county_dict = {('Alameda County', 'LDA'): 0.7091597506569293, ('Alameda County', 'LDT1'): 0.6936323597508934, ('Alameda County', 'LDT2'): 0.7312968301659862, ('Alameda County', 'MDV'): 0.695013598925902, ('Alameda County', 'UBUS'): 1.0, ('Contra Costa County', 'LDA'): 0.8122153579138889, ('Contra Costa County', 'LDT1'): 0.8011148598421644, ('Contra Costa County', 'LDT2'): 0.8530516728370202, ('Contra Costa County', 'MDV'): 0.8084326383470113, ('Contra Costa County', 'UBUS'): 1.0, ('Marin County', 'LDA'): 0.5577073646904817, ('Marin County', 'LDT1'): 0.5391600746601661, ('Marin County', 'LDT2'): 0.5813144784963997, ('Marin County', 'MDV'): 0.5590469075196355, ('Marin County', 'UBUS'): 1.0, ('Napa County', 'LDA'): 0.6357106636630795, ('Napa County', 'LDT1'): 0.5711720562226346, ('Napa County', 'LDT2'): 0.6111369616248149, ('Napa County', 'MDV'): 0.5790350554407057, ('Napa County', 'UBUS'): 1.0, ('San Francisco County', 'LDA'): 1.2904878233981214, ('San Francisco County', 'LDT1'): 1.1706782047854523, ('San Francisco County', 'LDT2'): 1.2209977032428923, ('San Francisco County', 'MDV'): 1.300734984990792, ('San Francisco County', 'UBUS'): 1.0, ('San Mateo County', 'LDA'): 0.7566152007867873, ('San Mateo County', 'LDT1'): 0.7049478035053239, ('San Mateo County', 'LDT2'): 0.7454133561266547, ('San Mateo County', 'MDV'): 0.7470495206515292, ('San Mateo County', 'UBUS'): 1.0, ('Santa Clara County', 'LDA'): 0.9147138591092338, ('Santa Clara County', 'LDT1'): 0.8446889699427009, ('Santa Clara County', 'LDT2'): 0.8742638501011555, ('Santa Clara County', 'MDV'): 0.8440384932282214, ('Santa Clara County', 'UBUS'): 1.0, ('Solano County', 'LDA'): 1.2027290804944388, ('Solano County', 'LDT1'): 1.049380804461785, ('Solano County', 'LDT2'): 1.121255444037017, ('Solano County', 'MDV'): 1.0832065938854636, ('Solano County', 'UBUS'): 1.0, ('Sonoma County', 'LDA'): 1.0979105108539893, ('Sonoma County', 'LDT1'): 0.9921343975310387, ('Sonoma County', 'LDT2'): 1.0624719218248007, ('Sonoma County', 'MDV'): 1.0438817373596945, ('Sonoma County', 'UBUS'): 1.0}
# #Baseline Telecommuting
# corr_VMT_by_county_dict =   {('Alameda County', 'LDA'): 0.785363107733696, ('Alameda County', 'LDT1'): 0.7853564165979886, ('Alameda County', 'LDT2'): 0.785390797817228, ('Alameda County', 'MDV'): 0.7849956777912425, ('Alameda County', 'UBUS'): 1.0, ('Contra Costa County', 'LDA'): 0.9872432530089924, ('Contra Costa County', 'LDT1'): 0.987048936093546, ('Contra Costa County', 'LDT2'): 0.9876268814954524, ('Contra Costa County', 'MDV'): 0.9872268057399457, ('Contra Costa County', 'UBUS'): 1.0, ('Marin County', 'LDA'): 0.8172879096205063, ('Marin County', 'LDT1'): 0.8185103988462367, ('Marin County', 'LDT2'): 0.8174120228282623, ('Marin County', 'MDV'): 0.8182681812704706, ('Marin County', 'UBUS'): 1.0, ('Napa County', 'LDA'): 0.8458142553951628, ('Napa County', 'LDT1'): 0.8467611067417001, ('Napa County', 'LDT2'): 0.8457382327681194, ('Napa County', 'MDV'): 0.8457574922686403, ('Napa County', 'UBUS'): 1.0, ('San Francisco County', 'LDA'): 0.8993073501587944, ('San Francisco County', 'LDT1'): 0.9005953463714442, ('San Francisco County', 'LDT2'): 0.899192784589715, ('San Francisco County', 'MDV'): 0.8992279905520532, ('San Francisco County', 'UBUS'): 1.0, ('San Mateo County', 'LDA'): 0.8124041598123923, ('San Mateo County', 'LDT1'): 0.8109831615917596, ('San Mateo County', 'LDT2'): 0.812035815114161, ('San Mateo County', 'MDV'): 0.8128831432733896, ('San Mateo County', 'UBUS'): 1.0, ('Santa Clara County', 'LDA'): 0.7857616435243143, ('Santa Clara County', 'LDT1'): 0.7857972813436149, ('Santa Clara County', 'LDT2'): 0.7858681090323005, ('Santa Clara County', 'MDV'): 0.7858562582893761, ('Santa Clara County', 'UBUS'): 1.0, ('Solano County', 'LDA'): 1.2742151495807013, ('Solano County', 'LDT1'): 1.2733758023012685, ('Solano County', 'LDT2'): 1.2747147253340616, ('Solano County', 'MDV'): 1.2743458197808435, ('Solano County', 'UBUS'): 1.0, ('Sonoma County', 'LDA'): 1.3656619432175308, ('Sonoma County', 'LDT1'): 1.3656938407153925, ('Sonoma County', 'LDT2'): 1.3668753119486865, ('Sonoma County', 'MDV'): 1.366124572683926, ('Sonoma County', 'UBUS'): 1.0}
# corr_Trips_by_county_dict = {('Alameda County', 'LDA'): 0.6826851725666049, ('Alameda County', 'LDT1'): 0.6677245597476429, ('Alameda County', 'LDT2'): 0.7039856109116475, ('Alameda County', 'MDV'): 0.6690661641149621, ('Alameda County', 'UBUS'): 1.0, ('Contra Costa County', 'LDA'): 0.7670701948598858, ('Contra Costa County', 'LDT1'): 0.756624262263867, ('Contra Costa County', 'LDT2'): 0.8056481451499258, ('Contra Costa County', 'MDV'): 0.7635208244086293, ('Contra Costa County', 'UBUS'): 1.0, ('Marin County', 'LDA'): 0.5392965683868883, ('Marin County', 'LDT1'): 0.5213872503746932, ('Marin County', 'LDT2'): 0.5620706370312918, ('Marin County', 'MDV'): 0.5406848058702638, ('Marin County', 'UBUS'): 1.0, ('Napa County', 'LDA'): 0.6207311395962061, ('Napa County', 'LDT1'): 0.5576822169989116, ('Napa County', 'LDT2'): 0.5966917479512112, ('Napa County', 'MDV'): 0.5656849333939483, ('Napa County', 'UBUS'): 1.0, ('San Francisco County', 'LDA'): 1.2424132583226901, ('San Francisco County', 'LDT1'): 1.1268069224597026, ('San Francisco County', 'LDT2'): 1.1755842994997672, ('San Francisco County', 'MDV'): 1.2522134825388531, ('San Francisco County', 'UBUS'): 1.0, ('San Mateo County', 'LDA'): 0.7229052386151321, ('San Mateo County', 'LDT1'): 0.6735675661144088, ('San Mateo County', 'LDT2'): 0.7121928313838096, ('San Mateo County', 'MDV'): 0.7137063008601003, ('San Mateo County', 'UBUS'): 1.0, ('Santa Clara County', 'LDA'): 0.8734394028335127, ('Santa Clara County', 'LDT1'): 0.8065304612880395, ('Santa Clara County', 'LDT2'): 0.834829214293623, ('Santa Clara County', 'MDV'): 0.8059331433404102, ('Santa Clara County', 'UBUS'): 1.0, ('Solano County', 'LDA'): 1.1430680793970225, ('Solano County', 'LDT1'): 0.9975726907916888, ('Solano County', 'LDT2'): 1.0656208637467885, ('Solano County', 'MDV'): 1.0294647457951722, ('Solano County', 'UBUS'): 1.0, ('Sonoma County', 'LDA'): 1.0468909755527545, ('Sonoma County', 'LDT1'): 0.945855190861318, ('Sonoma County', 'LDT2'): 1.0131414788781201, ('Sonoma County', 'MDV'): 0.9952778260056453, ('Sonoma County', 'UBUS'): 1.0}
# #Baseline Transit Rich

# #Baseline Cruise
# corr_VMT_by_county_dict = {('Alameda County', 'LDA'): 0.7470757280963015, ('Alameda County', 'LDT1'): 0.7472140594014864, ('Alameda County', 'LDT2'): 0.7472380715194904, ('Alameda County', 'MDV'): 0.7471262895070857, ('Alameda County', 'UBUS'): 1.0, ('Contra Costa County', 'LDA'): 0.9847653531214324, ('Contra Costa County', 'LDT1'): 0.984509971601675, ('Contra Costa County', 'LDT2'): 0.9845308289181104, ('Contra Costa County', 'MDV'): 0.9847239637405645, ('Contra Costa County', 'UBUS'): 1.0, ('Marin County', 'LDA'): 0.7213519223772172, ('Marin County', 'LDT1'): 0.7212999867229127, ('Marin County', 'LDT2'): 0.7213014923777741, ('Marin County', 'MDV'): 0.721263316901008, ('Marin County', 'UBUS'): 1.0, ('Napa County', 'LDA'): 0.8131699350289362, ('Napa County', 'LDT1'): 0.8127364757621987, ('Napa County', 'LDT2'): 0.8124096211841729, ('Napa County', 'MDV'): 0.8129467981345975, ('Napa County', 'UBUS'): 1.0, ('San Francisco County', 'LDA'): 0.690365277153231, ('San Francisco County', 'LDT1'): 0.6911573784822581, ('San Francisco County', 'LDT2'): 0.690634537403338, ('San Francisco County', 'MDV'): 0.6898464226378359, ('San Francisco County', 'UBUS'): 1.0, ('San Mateo County', 'LDA'): 0.755619339380792, ('San Mateo County', 'LDT1'): 0.7565534468039801, ('San Mateo County', 'LDT2'): 0.7556177981134615, ('San Mateo County', 'MDV'): 0.7562688858176352, ('San Mateo County', 'UBUS'): 1.0, ('Santa Clara County', 'LDA'): 0.7550948471743995, ('Santa Clara County', 'LDT1'): 0.7549179970888578, ('Santa Clara County', 'LDT2'): 0.7549617843328084, ('Santa Clara County', 'MDV'): 0.7548916110007055, ('Santa Clara County', 'UBUS'): 1.0, ('Solano County', 'LDA'): 1.3281306684274687, ('Solano County', 'LDT1'): 1.3264322018480532, ('Solano County', 'LDT2'): 1.3286481555145708, ('Solano County', 'MDV'): 1.327944123036664, ('Solano County', 'UBUS'): 1.0, ('Sonoma County', 'LDA'): 1.3192556630301941, ('Sonoma County', 'LDT1'): 1.319402335782789, ('Sonoma County', 'LDT2'): 1.3199072435124304, ('Sonoma County', 'MDV'): 1.3190121792896632, ('Sonoma County', 'UBUS'): 1.0}
# corr_Trips_by_county_dict = {('Alameda County', 'LDA'): 0.5930602881693966, ('Alameda County', 'LDT1'): 0.5800952362072366, ('Alameda County', 'LDT2'): 0.6115585952640538, ('Alameda County', 'MDV'): 0.5812290510410403, ('Alameda County', 'UBUS'): 1.0, ('Contra Costa County', 'LDA'): 0.7379894497503067, ('Contra Costa County', 'LDT1'): 0.7279298155498949, ('Contra Costa County', 'LDT2'): 0.7751301383370086, ('Contra Costa County', 'MDV'): 0.7345590004670559, ('Contra Costa County', 'UBUS'): 1.0, ('Marin County', 'LDA'): 0.5059431763954676, ('Marin County', 'LDT1'): 0.48921230985788877, ('Marin County', 'LDT2'): 0.5274633994993898, ('Marin County', 'MDV'): 0.5070553388045606, ('Marin County', 'UBUS'): 1.0, ('Napa County', 'LDA'): 0.6084893875519989, ('Napa County', 'LDT1'): 0.5463384885607809, ('Napa County', 'LDT2'): 0.584770095702514, ('Napa County', 'MDV'): 0.5543479249891623, ('Napa County', 'UBUS'): 1.0, ('San Francisco County', 'LDA'): 0.8405743980089847, ('San Francisco County', 'LDT1'): 0.7624275011494895, ('San Francisco County', 'LDT2'): 0.7954441499381745, ('San Francisco County', 'MDV'): 0.8473326965147943, ('San Francisco County', 'UBUS'): 1.0, ('San Mateo County', 'LDA'): 0.639439247829707, ('San Mateo County', 'LDT1'): 0.5958553311639659, ('San Mateo County', 'LDT2'): 0.6299430871478379, ('San Mateo County', 'MDV'): 0.631283622233763, ('San Mateo County', 'UBUS'): 1.0, ('Santa Clara County', 'LDA'): 0.7968017355126182, ('Santa Clara County', 'LDT1'): 0.7358153604065697, ('Santa Clara County', 'LDT2'): 0.7615898446656352, ('Santa Clara County', 'MDV'): 0.7352292444096753, ('Santa Clara County', 'UBUS'): 1.0, ('Solano County', 'LDA'): 1.0953976674750743, ('Solano County', 'LDT1'): 0.9564246828150279, ('Solano County', 'LDT2'): 1.0212832615504275, ('Solano County', 'MDV'): 0.9865887358433549, ('Solano County', 'UBUS'): 1.0, ('Sonoma County', 'LDA'): 0.9987039280764861, ('Sonoma County', 'LDT1'): 0.9023807845958227, ('Sonoma County', 'LDT2'): 0.9664565170989271, ('Sonoma County', 'MDV'): 0.9495801279078178, ('Sonoma County', 'UBUS'): 1.0}

processes = ['RUNEX', 'PMBW', 'PMTW', 'RUNLOSS', 'HOTSOAK', 'DIURN', 'STREX']

processDataframes = ['PT', 'PT', 'PT', 'PT', 'PAn', 'PA', 'DP']

pollutants_list = [
    ['PM2_5', 'SOx', 'NOx', 'ROG', 'NH3', 'CO2'],  # RUNEX
    ['PM2_5'],                              # PMBW
    ['PM2_5'],                              # PMTW
    ['ROG'],                                # RUNLOSS
    ['ROG'],                                # HOTSOAK
    ['ROG'],                                # DIURN
    ['PM2_5', 'SOx', 'NOx', 'ROG', 'CO2']          # STREX
]

merging_lists = [
    ['fuel_type_emfact', 'average_speed_mph', 'county_name', 'vehicle_class_emfact'],  # RUNEX
    ['fuel_type_emfact', 'average_speed_mph', 'county_name', 'vehicle_class_emfact'],  # PMBW
    ['fuel_type_emfact', 'county_name', 'vehicle_class_emfact'],                       # PMTW
    ['fuel_type_emfact', 'county_name', 'vehicle_class_emfact'],                       # RUNLOSS
    ['fuel_type_emfact', 'county_name', 'vehicle_class_emfact'],                       # HOTSOAK
    ['fuel_type_emfact', 'county_name', 'vehicle_class_emfact'],                       # DIURN
    ['fuel_type_emfact', 'average_speed_mph', 'county_name', 'vehicle_class_emfact']   # STREX
]

busesFuelType = 'Dsl'
busVehicleClassDict = {'UBUS':'UBUS',
                       'OBUS':'UBUS',
#                        'SBUS':'UBUS',
#                        'All Other Buses':'UBUS',
                      }
pathTraversalModesDict = {'car':'car', 
                               'car_hov2':'car', 
                               'car_hov3':'car', 
                               'bus':'UBUS',
                               'freight':'freight',
                         }
                               
EMFACT_classes_car = ['LDA','LDT1', 'LDT2', 'MDV']
EMFACT_classes_freight = [
    'LHD1', 'LHD2', 'MCY', 'MH',
    'T6 CAIRP Class 4', 'T6 CAIRP Class 5', 'T6 CAIRP Class 6', 'T6 CAIRP Class 7',
    'T6 Instate Delivery Class 4', 'T6 Instate Delivery Class 5', 'T6 Instate Delivery Class 6', 'T6 Instate Delivery Class 7',
    'T6 Instate Other Class 4', 'T6 Instate Other Class 5', 'T6 Instate Other Class 6', 'T6 Instate Other Class 7',
    'T6 Instate Tractor Class 6', 'T6 Instate Tractor Class 7',
    'T6 OOS Class 4', 'T6 OOS Class 5', 'T6 OOS Class 6', 'T6 OOS Class 7',
    'T6 Public Class 4', 'T6 Public Class 5', 'T6 Public Class 6', 'T6 Public Class 7',
    'T6 Utility Class 5', 'T6 Utility Class 6', 'T6 Utility Class 7',
    'T6TS',
    'T7 CAIRP Class 8', 'T7 NNOOS Class 8', 'T7 NOOS Class 8', 'T7 Other Port Class 8', 'T7 POAK Class 8',
    'T7 Public Class 8',
    'T7 Single Concrete/Transit Mix Class 8', 'T7 Single Dump Class 8', 'T7 Single Other Class 8',
    'T7 SWCV Class 8', 'T7 Tractor Class 8', 'T7 Utility Class 8', 'T7IS']
EMFACT_classes_bus = list(busVehicleClassDict.keys())
EMFACT_classes = EMFACT_classes_car + EMFACT_classes_bus + EMFACT_classes_freight

county_dict = {
        'Alameda (SF)': 'Alameda County',
        'Contra Costa (SF)': 'Contra Costa County',
        'Marin (SF)': 'Marin County',
        'Napa (SF)': 'Napa County',
        'San Francisco (SF)': 'San Francisco County',
        'San Mateo (SF)': 'San Mateo County', 
        'Santa Clara (SF)': 'Santa Clara County',
        'Solano (SF)': 'Solano County',
        'Sonoma (SF)': 'Sonoma County'
    }

events_column_types = {
    'type': 'category',
    'vehicle': 'str',
    'links': 'str',  
    'linkTravelTime': 'str',
    'mode': 'str',
    'vehicleType': 'category',
    'departureTime': 'Int64', 
    'length': 'float'
}




#############################################################
########################## FUNCTIONS ########################
#############################################################

def addGeometryIdToDataFrame(df, gdf, xcol, ycol, idColumn="geometry", df_geom='EPSG:32610', column='blkgrpid', df_geom2='EPSG:4326'):
    start_time = time.time() 
    
    print(' - Function:  addGeometryIdToDataFrame ...')

    # Spatial join of points with polygons
    
    gdf.crs = df_geom2
    gdf_data = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[xcol], df[ycol]))
    gdf_data.crs = df_geom  # Updated to use EPSG code directly
    joined = gpd.sjoin(gdf_data.to_crs('EPSG:26910'), gdf.to_crs('EPSG:26910'), how='inner')
    gdf_data = gdf_data.merge(joined[column], left_index=True, right_index=True, how="left")
    gdf_data.rename(columns={column: idColumn}, inplace=True)
    df = pd.DataFrame(gdf_data.drop(columns='geometry'))

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return df.loc[~df.index.duplicated(keep='first'), :]

def split_row(row):
    
    # Split a Network link into smaller segments 
    # - Adjust link length and coordinates
    # - Add segment part code
    
    global segment_length

    num_segments = int(np.ceil(row['linkLength'] / segment_length))
    rows = []
    for i in range(num_segments):
        segment_length_eff = row['linkLength'] / num_segments
        rows.append({
            **row,
            'fromLocationX': row['fromLocationX'] + (row['toLocationX'] - row['fromLocationX']) * (i / num_segments),
            'fromLocationY': row['fromLocationY'] + (row['toLocationY'] - row['fromLocationY']) * (i / num_segments),
            'toLocationX': row['fromLocationX'] + (row['toLocationX'] - row['fromLocationX']) * ((i + 1) / num_segments),
            'toLocationY': row['fromLocationY'] + (row['toLocationY'] - row['fromLocationY']) * ((i + 1) / num_segments),
            'linkLength': segment_length_eff,
            'segment_part': i + 1
        })

    return rows

#############################################################
########################## NETWORK ##########################
#############################################################

def split_network(network):
    start_time = time.time() 
    
    print(' - Function:  split_network ...')
    
    # Split network database on smaller links (segment_length) to better associate with GRID borders

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return pd.DataFrame([y for x in network.apply(split_row, axis=1) for y in x])

def calculate_midpoints(network):
    start_time = time.time() 
    
    print(' - Function:  calculate_midpoints ...')

    # Get link centroids for all network links

    network['X'] = (network['fromLocationX'] + network['toLocationX']) / 2
    network['Y'] = (network['fromLocationY'] + network['toLocationY']) / 2

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return network

def enrich_geo_data(network, GRID_grid, BGs, block_info, counties_shapefile):
    start_time = time.time() 
    
    print(' - Function:  enrich_geo_data ...')

    # Add geographic information to the network
    # - GRID
    # - Block ID
    # - Block info: county, city, etc.

    network = addGeometryIdToDataFrame(network, GRID_grid, 'X', 'Y', 'GRID', column='grid', 
                                       df_geom2='+proj=lcc +lat_0=40 +lon_0=-97 +lat_1=33 +lat_2=45 +x_0=0 +y_0=0 +ellps=sphere +units=m +no_defs +type=crs')
    network = addGeometryIdToDataFrame(network, BGs, 'X', 'Y', 'BlockGroup', column='blkgrpid')
    network = pd.merge(network, block_info, how='left', left_on='BlockGroup', right_on='bgid')
    network['linkId'] = network['linkId'].astype(int)

    def recalculate_counties(network, counties_shapefile):
    # Print the initial dataframe and its keys
        print(network.county_name.value_counts())

        # Count NaN values in the county_name column before the update
        if 'county_name' in network.columns:
            network = network.drop(columns=['county_name'])
        else:
            print("'county_name' column does not exist before update.")

        network = addGeometryIdToDataFrame(network, counties_shapefile, xcol='X', ycol='Y', 
                                           idColumn='county_name', df_geom='EPSG:26910', 
                                           column='coname', df_geom2='EPSG:4326')

        # Print the updated dataframe and its keys
        print(network.county_name.value_counts())


        # Count NaN values in the county_name column after the update
        print(f"Number of NaN values in 'county_name' after update: {network['county_name'].isna().sum()}")
    
        return network
    network = recalculate_counties(network, counties_shapefile)
    network['county_name'] = network['county_name'] + ' County'


    
#     columns_to_fill = ['bgid', 'tractid', 'juris_name', 'mpo']
#     network[columns_to_fill] = network[columns_to_fill].fillna('Other')
#     network['county_name'] = network['county_name'].fillna(other_county)

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return network

#############################################################
########################## PATHTRAVERSAL ####################
#############################################################

def filter_events(events):
    start_time = time.time() 
    
    print(' - Function:  filter_pathtraversal ...')
    
    # Filter PathTraversal events from the event file
    
    initial_len = len(events)
    pathTraversal = events[events['type'] == 'PathTraversal'].copy()
    pathTraversal.loc[:, 'PTID'] = pathTraversal.index
    print('     ... Len Events', initial_len)
    print('     ... Len pathTraversal', len(pathTraversal))
    print('     ... Total Length in Meters PathTraversal', pathTraversal.length.sum())

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return pathTraversal

def drop_na_links(pathTraversal):
    start_time = time.time() 
    
    print(' - Function:  drop_na_links ...')

    # Drop PathTraversal with no links

    initial_len = pathTraversal.length.sum()
    initial_count = len(pathTraversal)
    pathTraversal = pathTraversal.dropna(subset=['links'])
    print('     ... Total Length in Meters PathTraversal', initial_len)
    print('     ... Len pathTraversal', initial_count)
    print('     ... Total Length in Meters PathTraversal after dropping Nan links', pathTraversal.length.sum())
    print('     ... Len pathTraversal after dropping Nan links',len(pathTraversal))

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return pathTraversal

def filter_modes(pathTraversal, scenario):
    start_time = time.time() 
    
    print(' - Function:  filter_modes ...')
    
    # Filter PathTraversal modes
    
    global pathTraversalModesDict
    
    print('     ... Modes of pathTraversal', pathTraversal['mode'].value_counts().keys())

    pathTraversal['vehicleType'].value_counts().to_csv(f'vehicletypes_{scenario}.csv')
    pathTraversal.loc[
        pathTraversal['vehicleType'].str.contains(r'T6|T7', na=False),
        'mode'
        ] = 'freight'
    pathTraversal = pathTraversal[pathTraversal['mode'].isin(pathTraversalModesDict.keys())].copy()
    print('     ... Len pathTraversal after dropping active modes', len(pathTraversal))
    print('     ... Total Length in Meters PathTraversal after dropping active modes',pathTraversal.length.sum())
 
    pathTraversal.loc[:, 'mode'] = pathTraversal['mode'].map(pathTraversalModesDict)
    
    print('     ... Total Length in Meters PathTraversal of freight vehicles',
          pathTraversal[pathTraversal['mode']=='freight'].length.sum())
    print('     ... Total Length in Meters PathTraversal of car vehicles',
          pathTraversal[pathTraversal['mode']=='car'].length.sum())
    print('     ... Total Length in Meters PathTraversal of bus vehicles',
          pathTraversal[pathTraversal['mode']=='UBUS'].length.sum())
    print('     ... Len pathTraversal Freight', len(pathTraversal[pathTraversal['mode'].isin(['freight'])]))
    print('     ... Len pathTraversal Car', len(pathTraversal[pathTraversal['mode'].isin(['car'])]))
    print('     ... Len pathTraversal Bus', len(pathTraversal[pathTraversal['mode'].isin(['UBUS'])]))
    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return pathTraversal

def process_links(pathTraversal):
    start_time = time.time() 
    
    print(' - Function:  process_links ...')
    
    # - Adjust format of linkIDs and their travel times for each PathTraversal
    # - Get first and last link for each PathTraversal and filter Nan
    
    global correction_factor_duration

    pathTraversal['links'] = pathTraversal['links'].str.split(',')
    pathTraversal['linkTravelTime'] = pathTraversal['linkTravelTime'].str.split(',')
    pathTraversal['linkTravelTime'] = pathTraversal['linkTravelTime'].apply(
        lambda times: [str(float(time) * correction_factor_duration) for time in times])
    pathTraversal['firstLink'] = pathTraversal['links'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    pathTraversal['lastLink'] = pathTraversal['links'].apply(lambda x: x[-1] if isinstance(x, list) and len(x) > 0 else None)
    pathTraversal.dropna(subset=['firstLink', 'lastLink'], inplace=True)
    print('     ... Total Length in Meters PathTraversal after dropping rows with None in firstLink or lastLink', pathTraversal.length.sum())
    print('     ... Len pathTraversal after dropping rows with None in firstLink or lastLink', len(pathTraversal))

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return pathTraversal

def calculate_link_departure_times(departureTime, linkTravelTimes):
    
    # For a specific PathTraversal calculate the departure time of each traveled link based
    # on trip departure and link travel times
    
    link_departure_times = [departureTime]
    current_time = departureTime
    for travel_time in linkTravelTimes[:-1]:
        current_time += float(travel_time) 
        link_departure_times.append(current_time) 

    return link_departure_times

def link_times(pathTraversal):
    start_time = time.time() 
    
    print(' - Function:  link_times ...')

    # - Calculate departing times for each link in a PathTraversal
    # - For each PathTraversal, get a list comprehending the sequence of linkIDs, Travel times and departing times
    
    pathTraversal['link_departure_times'] = pathTraversal.apply(
        lambda row: calculate_link_departure_times(row['departureTime'], row['linkTravelTime']), axis=1)
    
    pathTraversal['link_with_time'] = pathTraversal.apply(
        lambda row: list(zip(row['links'], row['linkTravelTime'], row['link_departure_times'])), axis=1)
    
    pathTraversal = pathTraversal.drop(['links', 'linkTravelTime', 'link_departure_times'], axis=1)

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return pathTraversal

#############################################################
########################## PARKING ##########################
#############################################################

def prepare_park_departDatabase(pathTraversal):
    start_time = time.time() 
    
    print(' - Function:  prepare_park_departDatabase ...')

    # For each PathTraversal, get parking time
    # - Last one will go until end_day_parking
    
    global end_day_parking
    
    pathTraversal['trip_duration'] = pathTraversal['link_with_time'].apply(
        lambda x: sum(float(triplet[1]) for triplet in x)
    )
    pathTraversal['start_parking_time'] = pathTraversal['departureTime'] + pathTraversal['trip_duration'] 
    pathTraversal.sort_values(by=['vehicle', 'departureTime'], inplace=True)
    pathTraversal['next_departureTime'] = pathTraversal.groupby('vehicle')['departureTime'].shift(-1)
    pathTraversal['next_vehicle'] = pathTraversal['vehicle'].shift(-1)
    pathTraversal['parking_time'] = pathTraversal.apply(
        lambda x: (x['next_departureTime'] - x['start_parking_time']) if x['vehicle'] == x['next_vehicle'] else max(0, end_day_parking - x['start_parking_time'] ), axis=1
    )
    pathTraversal = pathTraversal.dropna(subset=['parking_time', 'departureTime'])

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return pathTraversal

def prepare_parkingDatabase(pathTraversal, networkParking):
    start_time = time.time() 
    
    print(' - Function:  prepare_parkingDatabase ...')

    # Add parking time prior to first trip to cover the 24h parking time
    # - Filter low parking times, due to small drop offs or when vehicles start looking for parking
    # - Note: first row for each vehicle can be easily overlapped since it would be filtered anyway
    # - Drop last parking time to get the prior-trip parking database
    
    global beginning_day_parking
    global parking_limit
    
    pathTraversal.loc[pathTraversal.groupby('vehicle').cumcount() == 0, 'parking_time'] = (pathTraversal['departureTime'] - beginning_day_parking).clip(lower=0)
    parkDepartDatabase = pathTraversal[pathTraversal['parking_time'] >= parking_limit].copy()
    parkDepartDatabase.loc[:, 'lastLink']  = parkDepartDatabase['lastLink'].astype(int)
    parkDepartDatabase = pd.merge(parkDepartDatabase, networkParking[['linkId','linkLength','GRID','county_name']], how='left', left_on='lastLink', right_on='linkId')
    
    departDatabase = parkDepartDatabase.drop(parkDepartDatabase.groupby('vehicle').tail(1).index)
    parkingDatabase_no_first = parkDepartDatabase.drop(parkDepartDatabase.groupby('vehicle').head(1).index)

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return parkDepartDatabase, departDatabase, parkingDatabase_no_first


#############################################################
########################## EXPLODE LINKS ####################
#############################################################

def explode_path_traversal(pathTraversal):
    start_time = time.time() 
    
    print(' - Function:  explode_path_traversal ...')

    # Explode PathTraversals link by link
    
    explodedPathTraversal = pathTraversal.explode('link_with_time')
    explodedPathTraversal['identifier'] = range(len(explodedPathTraversal))
    explodedPathTraversal[['link', 'travelTime', 'departureTime']] = pd.DataFrame(
        explodedPathTraversal['link_with_time'].tolist(), index=explodedPathTraversal.index)
    explodedPathTraversal = explodedPathTraversal.drop(['link_with_time'], axis=1)
    explodedPathTraversal['travelTime'] = explodedPathTraversal['travelTime'].astype(float)
    explodedPathTraversal['departureTime'] = explodedPathTraversal['departureTime'].astype(float)
    explodedPathTraversal['link'] = explodedPathTraversal['link'].astype(int)

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return explodedPathTraversal

def merge_with_network(explodedPathTraversal, networkExploded):
    start_time = time.time() 
    
    print(' - Function:  merge_with_network ...')

    # Merge PathTraversal with network to further explode links on smaller ones (see function split_network)
    
    networkExploded['linkId'] = networkExploded['linkId'].astype(int)
    explodedPathTraversal = pd.merge(
        explodedPathTraversal, networkExploded[['linkId', 'linkLength', 'bgid','GRID', 'county_name', 'segment_part']],
        how='left', left_on='link', right_on='linkId')
    explodedPathTraversal = explodedPathTraversal.drop(['linkId'], axis=1)
    
    print('     ... Len explodedPathTraversal', len(explodedPathTraversal))
    explodedPathTraversal = explodedPathTraversal.dropna(subset=['county_name'])
    print('     ... Len explodedPathTraversal after dropping trip outside the simulated counties: ', len(explodedPathTraversal))

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return explodedPathTraversal

def adjust_travel_times(explodedPathTraversal):
    start_time = time.time() 
    
    print(' - Function:  adjust_travel_times ...')

    # Edit travel and departing times accordingly to the link disaggregation
    
    explodedPathTraversal['total_length'] = explodedPathTraversal.groupby(
        ['vehicle', 'firstLink', 'lastLink', 'identifier', 'PTID'])['linkLength'].transform('sum')
    explodedPathTraversal['length_proportion'] = explodedPathTraversal['linkLength'] / explodedPathTraversal['total_length']
    explodedPathTraversal['travelTime'] *= explodedPathTraversal['length_proportion']
    explodedPathTraversal['cumulativeTravelTime'] = explodedPathTraversal.groupby(
        ['vehicle', 'firstLink', 'lastLink', 'identifier', 'PTID'])['travelTime'].cumsum()
    explodedPathTraversal['departureTime'] += explodedPathTraversal.groupby(
        ['vehicle', 'firstLink', 'lastLink', 'identifier', 'PTID'])['cumulativeTravelTime'].shift(fill_value=0)
    explodedPathTraversal = explodedPathTraversal.drop(['total_length', 'length_proportion', 'cumulativeTravelTime'], axis=1)

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return explodedPathTraversal

def calculate_speeds(explodedPathTraversal):
    start_time = time.time() 
    
    print(' - Function:  calculate_speeds ...')

    # Add Speed column and adjust units
    
    global default_speed
    
    explodedPathTraversal['linkLength'] /= 1000 * 1.60934
    explodedPathTraversal['travelTime'] /= 3600
    explodedPathTraversal['average_speed_mph'] = explodedPathTraversal['linkLength'] / explodedPathTraversal['travelTime']
    print('     ... Len explodedPathTraversal', len(explodedPathTraversal))
    print('     ... Len of undetermined speeds:', len(explodedPathTraversal.loc[(explodedPathTraversal['travelTime'] == 0) | (explodedPathTraversal['linkLength'] == 0), 'average_speed_mph']))
    explodedPathTraversal.loc[(explodedPathTraversal['travelTime'] == 0) | (explodedPathTraversal['linkLength'] == 0), 'average_speed_mph'] = default_speed

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return explodedPathTraversal

#############################################################
################# ASSIGN EMFACT VEHICLE TYPES ###############
#############################################################


def load_EMFACT_data(filepath, EMFACT_classes):
    start_time = time.time()
    print(' - Function:  load_EMFACT_data ...')

    global county_dict

    EMFACT_file = pd.read_csv(filepath)
    print(f"   Loaded {len(EMFACT_file):,} rows from {filepath}")

    # Check unaccounted BEFORE filtering
    unaccounted = EMFACT_file.loc[~EMFACT_file['vehicle_class'].isin(EMFACT_classes), 'vehicle_class'].unique()
    if len(unaccounted):
        print(f"   Unaccounted vehicle classes ({len(unaccounted)}): {unaccounted}")
    else:
        print("   All vehicle classes are accounted for.")

    # Filter
    EMFACT_file = EMFACT_file[EMFACT_file['vehicle_class'].isin(EMFACT_classes)]
    print(f"   Remaining after mode filter: {len(EMFACT_file):,} rows")

    # Map county names
    EMFACT_file['sub_area'] = EMFACT_file['sub_area'].map(county_dict)
    EMFACT_file = EMFACT_file[EMFACT_file['sub_area'].isin(list(county_dict.values()))]
    print(f"   After county filter: {len(EMFACT_file):,} rows in {EMFACT_file['sub_area'].nunique()} counties")

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return EMFACT_file

def calculate_vmt_distribution(EMFACT_vmt):
    start_time = time.time() 
    
    print(' - Function:  calculate_vmt_distribution by county ...')

    # Get EMFACT VMT distribution per vehicle type and county, weighted on VMT
    
    total_vmt_by_category = EMFACT_vmt.groupby(['sub_area', 'vehicle_class', 'fuel']).agg({'total_vmt': 'sum'}).reset_index()
    totalVMTbyCounty = total_vmt_by_category.groupby('sub_area').agg({'total_vmt': 'sum'}).rename(columns={'total_vmt': 'total_vmt_county'})
    totalVMTbyCounty = total_vmt_by_category.merge(totalVMTbyCounty, left_on='sub_area', right_index=True)
    totalVMTbyCounty['vmt_distribution'] = totalVMTbyCounty['total_vmt'] / totalVMTbyCounty['total_vmt_county']
#     totalVMTbyCounty['cdf'] = totalVMTbyCounty.groupby('sub_area')['vmt_distribution'].cumsum()
    print(f"   Created VMT distribution for {totalVMTbyCounty['sub_area'].nunique()} counties")
    print("   Example distribution rows:")
    print(totalVMTbyCounty.head())
    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return totalVMTbyCounty


def create_vmtLookup(totalVMTbyCounty):
    start_time = time.time() 
    
    print(' - Function:  create_vmtLookup ...')

    # Create a dictionary of VMT distribution per vehicle type and county
    
    vmtLookup = {}
    for sub_area, group in totalVMTbyCounty.groupby('sub_area'):
        vmtLookup[sub_area] = {
            'vmt_distribution': group['vmt_distribution'].values,
            'vehicle_classes': group['vehicle_class'].values,
            'fuels': group['fuel'].values
        }
    print(f"   Total counties in lookup: {len(vmtLookup)}")
    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return vmtLookup


def assign_subset(sub_area_data, vmtLookup):
    
#     print(' - Function:  assign_subset ...')

    # For each car-related explodedPathTraversal of a specific county, assign vehicles based 
    # on the EMFACT VMT distribution
    
    global seed
    
    sub_area = sub_area_data['county_name'].iloc[0]
    lookup = vmtLookup[sub_area]
    total_rows = len(sub_area_data)
    print(f"   Assigning {total_rows} rows for county '{sub_area}' from {len(lookup['vehicle_classes'])} EMFACT classes")
    assignments = []
    for threshold, vehicle_class, fuel in zip(lookup['vmt_distribution'], lookup['vehicle_classes'], lookup['fuels']):
        num_rows = int(round(threshold * total_rows+1))
        assignments += [(vehicle_class, fuel)] * num_rows
    np.random.seed(seed)
    np.random.shuffle(assignments)
    assignments = assignments[:total_rows]
    sub_area_data.loc[:, 'vehicle_class_emfact'], sub_area_data.loc[:, 'fuel_type_emfact'] = zip(*assignments)
    
    return sub_area_data

def add_emfact_vehicle_type_buses(bus_group):
    
#     print(' - Function:  assign_subset ...')

    # For each car-related explodedPathTraversal of a specific county, assign vehicles based 
    # on the EMFACT VMT distribution

    bus_group.loc[:, 'vehicle_class_emfact'] = bus_group.loc[:, 'mode']
    bus_group.loc[:, 'fuel_type_emfact'] = busesFuelType
    
    return bus_group

def assign_vehicle_classes(dataframe, car_vmtLookup, freight_vmtLookup):
    start_time = time.time() 
    
    print(' - Function:  assign_vehicle_classes ...')

    # Update car-related modes based on EMFACT distribution (vmtLookup)
            
    final_result = pd.DataFrame()
    print(f'len dataframe: {len(dataframe)}')
    for name, group in dataframe.groupby('county_name'):
        car_group = group[group['mode'] == 'car'].copy()
        bus_group = group[group['mode'].isin(EMFACT_classes_bus)].copy()
        freight_group = group[group['mode'] == 'freight'].copy()
        print(f'{name} - len car dataframe: {len(car_group)}')
        print(f'{name} - len freight dataframe: {len(freight_group)}')
        print(f'{name} - len bus dataframe: {len(bus_group)}')
        bus_group = add_emfact_vehicle_type_buses(bus_group)
        final_result = pd.concat([final_result, assign_subset(car_group, car_vmtLookup),
                                  assign_subset(freight_group, freight_vmtLookup), 
                                  bus_group])

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return final_result

def correct_cruise_vehicle_classes(dataframe):
    start_time = time.time() 
    
    print(' - Function:  correct_cruise_vehicle_classes ...')

    # Update Cruise vehicle type
    dataframe.loc[dataframe.vehicle.str.contains('@Cruise'), 'fuel_type_emfact'] = cruise_fuel_type
    dataframe.loc[dataframe.vehicle.str.contains('@Cruise'), 'vehicle_class_emfact'] = cruise_class_type

    
    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return dataframe

#############################################################
############### ACTIVITIES CORRECTION FACTORS ###############
#############################################################

def calculate_correction_factors(explodedPathTraversal, departDatabase, EMFACT_vmt, EMFACT_trips, scale, scenario):
    start_time = time.time() 
    
    print(' - Function:  calculate_correction_factors ...')

    # Get activity-based correction factors by county and vehicle type, based on VMT and #Trips activities
    
    global EMFACT_classes_bus
#     explodedPathTraversal.loc[explodedPathTraversal['bgid'] == 'Other', 'county_name'] = 'Other'
    print(type(scale), scale)

    vmt_by_county_BEAM = explodedPathTraversal.groupby(['county_name', 'vehicle_class_emfact'])['linkLength'].sum() / scale
    vmt_by_county_EMFACT = EMFACT_vmt.groupby(['sub_area', 'vehicle_class'])['total_vmt'].sum()
    vmt_by_county_BEAM.index.rename(['sub_area', 'vehicle_class'], inplace=True)
    corr_VMT_by_county = (vmt_by_county_BEAM).div(vmt_by_county_EMFACT, fill_value=np.nan).fillna(1)
    corr_VMT_by_county_dict = corr_VMT_by_county.to_dict()

    trips_by_county_BEAM_origin = departDatabase.groupby(['county_name','vehicle_class_emfact']).county_name.count() / scale
    trips_by_county_EMFACT = EMFACT_trips.groupby(['sub_area', 'vehicle_class'])['trips'].sum()
    trips_by_county_BEAM_origin.index.rename(['sub_area', 'vehicle_class'], inplace=True)
    corr_Trips_by_county = (trips_by_county_BEAM_origin).div(trips_by_county_EMFACT, fill_value=np.nan).fillna(1)
    corr_Trips_by_county_dict = corr_Trips_by_county.to_dict()
    
#     for (county, vehicle_class), factor in corr_VMT_by_county_dict.items():
#         if vehicle_class in EMFACT_classes_bus:
#             corr_VMT_by_county_dict[(county, vehicle_class)] *= scale

#     for (county, vehicle_class), factor in corr_Trips_by_county_dict.items():
#         if vehicle_class in EMFACT_classes_bus:
#             corr_Trips_by_county_dict[(county, vehicle_class)] *= scale
        
    print('corr_VMT_by_county_dict', corr_VMT_by_county_dict )
    print('corr_Trips_by_county_dict', corr_Trips_by_county_dict )
    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    
    pd.DataFrame.from_dict(corr_VMT_by_county_dict, orient='index').to_csv(f'{scenario}_corr_VMT_by_county.csv')
    pd.DataFrame.from_dict(corr_Trips_by_county_dict, orient='index').to_csv(f'{scenario}_corr_Trips_by_county.csv')
    return corr_VMT_by_county_dict, corr_Trips_by_county_dict


def apply_correction_factors(df, correction_factors, correctionFactorName):
    start_time = time.time() 
    
    print(' - Function:  apply_correction_factors ...')

    # Assign correction factors to the databases
    
    correction_factors_series = pd.Series(correction_factors)
#     print(correction_factors_series)
#     df['county_vehicle_key'] = df['county_name'] + '_' + df['vehicle_class_emfact']
#     df[correctionFactorName] = df['county_vehicle_key'].map(correction_factors_series).fillna(1)
#     df.drop('county_vehicle_key', axis=1, inplace=True)
    
    
    df['combined_key'] = list(zip(df['county_name'], df['vehicle_class_emfact']))
    df[correctionFactorName] = df['combined_key'].map(correction_factors_series).fillna(1)
    df.drop('combined_key', axis=1, inplace=True)
    
#         df['corr_VMT_by_county'] = df.apply(lambda row: corr_VMT_by_county_dict.get((row['county_name'], row['vehicle_class_emfact']), 1), axis=1)
#         df['corr_trips_by_county'] = df.apply(lambda row: corr_Trips_by_county_dict.get((row['county_name'], row['vehicle_class_emfact']), 1), axis=1)
#     print(df[correctionFactorName].value_counts())
    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return df



#############################################################
####################### EMISSION FACTORS ####################
#############################################################

def rename_emfact_columns(emfactEmisRates):
    start_time = time.time() 
    
    print(' - Function:  rename_emfact_columns ...')

    # Rename EMFACT columns
    
    emfactEmisRates.rename(columns={'fuel': 'fuel_type_emfact', 'speed_time': 'average_speed_mph', 'sub_area': 'county_name', 'vehicle_class': 'vehicle_class_emfact'}, inplace=True)

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return emfactEmisRates

def categorize_speed_data(explodedPathTraversal):
    
    def categorize_speed(speed):

#         print(' - Function:  categorize_speed ...')

        #Categorize speed on EMFACT speed bins

        category = min((speed // 5) * 5 + 5, 90)
        return max(category, 5)
    
    explodedPathTraversal['average_speed_mph'] = explodedPathTraversal['average_speed_mph'].apply(categorize_speed)
    
    return explodedPathTraversal


def adjust_departDatabase_speed(departDatabase):
    
    def categorize_park(park):

#         print(' - Function:  categorize_park ...')

        #Categorize park time on EMFACT time bins

        intervals = [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 480.0, 540.0, 600.0, 660.0, 720.0]
        pos = bisect.bisect_left(intervals, park)
        return intervals[min(pos, len(intervals) - 1)]

    departDatabase['average_speed_mph'] = departDatabase['parking_time'] / 60  # Parking time is on average speed column for STREX
    departDatabase['average_speed_mph'] = departDatabase['average_speed_mph'].apply(categorize_park)
    
    return departDatabase

#############################################################
################## ASSIGN EMISSION FACTORS ##################
#############################################################

def process_emission_rates(emfactEmisRates, explodedPathTraversal, parkingDatabase, departDatabase, parkingDatabase_no_first, scale):
    start_time = time.time() 
    
    print(' - Function:  process_emission_rates ...')

    # Associate emisison rates to the right dataframe
    
    global processDataframes 
    global processes 
    global pollutants_list 
    global merging_lists 
    
    for process, pollutants, merging_list, processdf in zip(processes, pollutants_list, merging_lists, processDataframes):
        print(' + Process:', process)
        
        for pollutant in pollutants:
            print('        - Pollutant:', pollutant)
            process_emission_rates = emfactEmisRates[emfactEmisRates.process == process]
            pollutant_emission_rates = process_emission_rates[process_emission_rates['pollutant'] == pollutant]
            print('                 ... LEN emfact emis factors: ', len(pollutant_emission_rates))
            pollutant_emission_rates = pollutant_emission_rates.drop_duplicates(subset=merging_list)
            print('                 ... LEN emfact emis factors after dropping duplicates: ', len(pollutant_emission_rates))
            pollutant_emission_rates.rename(columns={'emission_rate': f'emission_rate_{process}_{pollutant}'}, inplace=True)
            
            def process_emissions_for_df(df, process, pollutant, merging_list, pollutant_emission_rates, scale):
                start_time = time.time() 
    
                print(' - Function:  process_emissions_for_df ...')
                df = pd.merge(df, pollutant_emission_rates[merging_list + [f'emission_rate_{process}_{pollutant}']],
                              on=merging_list, how='left')
                df = calculate_emissions(df, process, pollutant, scale)
                print(f'           ... Emission added:', sum(df[f'tons_per_year_{process}_{pollutant}'].fillna(0)), 'tons per year')
                
                print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
                return df
            
            if processdf == 'PT':
                explodedPathTraversal = process_emissions_for_df(explodedPathTraversal, process, pollutant, merging_list, pollutant_emission_rates, scale)
            elif processdf == 'PAn':
                parkingDatabase_no_first = process_emissions_for_df(parkingDatabase_no_first, process, pollutant, merging_list, pollutant_emission_rates, scale)
            elif processdf == 'PA':
                parkingDatabase = process_emissions_for_df(parkingDatabase, process, pollutant, merging_list, pollutant_emission_rates, scale)
            elif processdf == 'DP':
                departDatabase = process_emissions_for_df(departDatabase, process, pollutant, merging_list, pollutant_emission_rates, scale)
    


    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return explodedPathTraversal, parkingDatabase, departDatabase, parkingDatabase_no_first

def calculate_emissions(df, process, pollutant, scale):
    start_time = time.time() 
    
    print(' - Function:  calculate_emissions ...')

    # Muliply Emission Factors for the specific process and pollutant
    
    global days
    global EMFACT_classes_bus
    
    df[f'tons_per_year_{process}_{pollutant}'] = df[f'emission_rate_{process}_{pollutant}'] * days / scale / 1000000 
                                                      
    if process in ['RUNEX', 'PMBW', 'PMTW']:
        df[f'tons_per_year_{process}_{pollutant}'] *= (df['linkLength'] / df['corr_VMT_by_county'])
    elif process == 'RUNLOSS':
        df[f'tons_per_year_{process}_{pollutant}'] *= (df['travelTime'] / df['corr_VMT_by_county'])
    elif process == 'HOTSOAK':
        df[f'tons_per_year_{process}_{pollutant}'] /= df['corr_trips_by_county']
    elif process == 'DIURN':
        df[f'tons_per_year_{process}_{pollutant}'] *= (df['parking_time'] / 3600 / df['corr_trips_by_county'])
    elif process == 'STREX':
        df[f'tons_per_year_{process}_{pollutant}'] /=  df['corr_trips_by_county']
        
    df.loc[df['vehicle_class_emfact'].isin(EMFACT_classes_bus), f'tons_per_year_{process}_{pollutant}'] *= scale

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return df

#############################################################
#################### AGGREGATE RESULTS ######################
#############################################################

def aggregate_emissions(explodedPathTraversal, parkingDatabase, departDatabase, parkingDatabase_no_first):
    start_time = time.time()
#     def aggregate_df_emissions_by_grid(dataframe, processes, pollutants_list):
         
    
#         print(' - Function:  aggregate_emissions_by_grid ...')

#         # Aggregate emissions by GRID

#         emissions_aggregated = dataframe.groupby('GRID')[
#             [f'tons_per_year_{process}_{pollutant}' 
#              for pollutants, process in zip(pollutants_list, processes)
#              for pollutant in pollutants 
#             ]
#         ].sum().reset_index()
#         return emissions_aggregated
    
    def aggregate_df_emissions_by_grid_and_mode(dataframe, processes, pollutants_list):
            print(' - Function:  aggregate_emissions_by_grid_and_mode ...')
            group_cols = ['GRID', 'mode', 'county_name', 'vehicle_class_emfact']
            emissions_aggregated = dataframe.groupby(group_cols)[
                [f'tons_per_year_{process}_{pollutant}'
                 for pollutants, process in zip(pollutants_list, processes)
                 for pollutant in pollutants]
            ].sum().reset_index()
            return emissions_aggregated
    
    BEAM_emis = aggregate_df_emissions_by_grid_and_mode(explodedPathTraversal, 
                                        ['RUNEX', 'PMBW', 'PMTW', 'RUNLOSS'], 
                                        [['PM2_5', 'SOx', 'NOx', 'ROG', 'NH3', 'CO2'], ['PM2_5'], ['PM2_5'], ['ROG']])
    BEAM_emis_park_no_first = aggregate_df_emissions_by_grid_and_mode(parkingDatabase_no_first, 
                                                          ['HOTSOAK'], 
                                                          [['ROG']])
    BEAM_emis_park = aggregate_df_emissions_by_grid_and_mode(parkingDatabase, 
                                                 ['DIURN'], 
                                                 [['ROG']])
    BEAM_emis_dep = aggregate_df_emissions_by_grid_and_mode(departDatabase, 
                                                ['STREX'], 
                                                [['PM2_5', 'SOx', 'NOx', 'ROG', 'CO2']])

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return BEAM_emis, BEAM_emis_park, BEAM_emis_park_no_first, BEAM_emis_dep

def merge_emission_dataframes(df_list):
    start_time = time.time() 
    
    print(' - Function:  merge_emission_dataframes ...')

    # Merge emissions from different dataframes
    
    result = df_list[0]
    for df in df_list[1:]:
        result = pd.merge(result, df, on=['GRID', 'mode', 'county_name', 'vehicle_class_emfact'], how='left')

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return result


def calculate_total_emissions(emissions_df):
    start_time = time.time() 
    
    print(' - Function:  calculate_total_emissions ...')

    # Combining specific pollutants and processes into total emissions
    
    emissions_df['tons_per_year_ROG'] = emissions_df[['tons_per_year_RUNEX_ROG', 'tons_per_year_STREX_ROG', 'tons_per_year_DIURN_ROG', 'tons_per_year_HOTSOAK_ROG', 'tons_per_year_RUNLOSS_ROG']].sum(axis=1)
    emissions_df['tons_per_year_PM2_5'] = emissions_df[['tons_per_year_RUNEX_PM2_5', 'tons_per_year_STREX_PM2_5', 'tons_per_year_PMBW_PM2_5', 'tons_per_year_PMTW_PM2_5']].sum(axis=1)
    emissions_df['tons_per_year_SOx'] = emissions_df[['tons_per_year_RUNEX_SOx', 'tons_per_year_STREX_SOx']].sum(axis=1)
    emissions_df['tons_per_year_NOx'] = emissions_df[['tons_per_year_RUNEX_NOx', 'tons_per_year_STREX_NOx']].sum(axis=1)
    emissions_df['tons_per_year_NH3'] = emissions_df['tons_per_year_RUNEX_NH3']
    emissions_df['tons_per_year_CO2'] = emissions_df[['tons_per_year_RUNEX_CO2', 'tons_per_year_STREX_CO2']].sum(axis=1)

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return emissions_df

def calculate_total_emissions_noCO2(emissions_df):
    start_time = time.time() 
    
    print(' - Function:  calculate_total_emissions ...')

    # Combining specific pollutants and processes into total emissions
    
    emissions_df['tons_per_year_ROG'] = emissions_df[['tons_per_year_RUNEX_ROG', 'tons_per_year_STREX_ROG', 'tons_per_year_DIURN_ROG', 'tons_per_year_HOTSOAK_ROG', 'tons_per_year_RUNLOSS_ROG']].sum(axis=1)
    emissions_df['tons_per_year_PM2_5'] = emissions_df[['tons_per_year_RUNEX_PM2_5', 'tons_per_year_STREX_PM2_5', 'tons_per_year_PMBW_PM2_5', 'tons_per_year_PMTW_PM2_5']].sum(axis=1)
    emissions_df['tons_per_year_SOx'] = emissions_df[['tons_per_year_RUNEX_SOx', 'tons_per_year_STREX_SOx']].sum(axis=1)
    emissions_df['tons_per_year_NOx'] = emissions_df[['tons_per_year_RUNEX_NOx', 'tons_per_year_STREX_NOx']].sum(axis=1)
    emissions_df['tons_per_year_NH3'] = emissions_df['tons_per_year_RUNEX_NH3']
#     emissions_df['tons_per_year_CO2'] = emissions_df[['tons_per_year_RUNEX_CO2', 'tons_per_year_STREX_CO2']].sum(axis=1)

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
    return emissions_df
#############################################################
################### SAVE FILES FOR INMAP ####################
#############################################################

def save_emissions(emissions_df, scenario):
    start_time = time.time() 
    
    print(' - Function:  save_emissions ...')

    #
    
    global geoAggregationType
    
    detail_file_name = f'BEAM_INMAP_detail_{geoAggregationType}_{scenario}.csv'
    summary_file_name = f'BEAM_INMAP_{geoAggregationType}_{scenario}.csv'
    emissions_df.to_csv(detail_file_name)
    emissions_df[['GRID','tons_per_year_ROG','tons_per_year_PM2_5','tons_per_year_SOx','tons_per_year_NOx','tons_per_year_NH3','tons_per_year_CO2']].to_csv(summary_file_name)
    emissions_df[emissions_df['mode']=='car'][['GRID','tons_per_year_ROG','tons_per_year_PM2_5','tons_per_year_SOx','tons_per_year_NOx','tons_per_year_NH3','tons_per_year_CO2']].to_csv(f'BEAM_INMAP_{geoAggregationType}_{scenario}_car.csv')
    emissions_df[emissions_df['mode']=='freight'][['GRID','tons_per_year_ROG','tons_per_year_PM2_5','tons_per_year_SOx','tons_per_year_NOx','tons_per_year_NH3','tons_per_year_CO2']].to_csv(f'BEAM_INMAP_{geoAggregationType}_{scenario}_freight.csv')
    emissions_df[emissions_df['mode']=='UBUS'][['GRID','tons_per_year_ROG','tons_per_year_PM2_5','tons_per_year_SOx','tons_per_year_NOx','tons_per_year_NH3','tons_per_year_CO2']].to_csv(f'BEAM_INMAP_{geoAggregationType}_{scenario}_bus.csv')

    print(' - Time taken: {:.2f} seconds'.format(time.time() - start_time))
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################

# def main():


def run_main():
    for scenario, year, iteration, scale, bucket, is_baseline, is_cruise in zip(scenarios, years, iterations, scales, buckets, are_baseline, are_cruise):

        abs_start_time = time.time()
        #############################################################
        print('########################## READ #############################')
        #############################################################

        print('Read Files ... ')
        if is_HaitamTest:
            events_fp = f'gs://{bucket}/{scenario}/ITERS/it.0/0.events.csv.gz'
            netwrok_fp = 'gs://'+bucket+'/'+scenario+'/network.csv.gz'
        else:
            events_fp = f'gs://{bucket}/{scenario}/beam/year-{year}-iteration-{iteration}/ITERS/it.0/0.events.csv.gz'
            netwrok_fp = 'gs://'+bucket+'/'+scenario+'/beam/year-'+year+'-iteration-'+iteration+'/network.csv.gz'

        print('########################## OSM-CHORDIFY #####################')
        osm_tmp_dir = Path(MAPPING_CFG.output_dir)
        osm_tmp_dir.mkdir(parents=True, exist_ok=True)
        beam_osm_mapped_out = osm_tmp_dir / f'{scenario}_beam_osm_mapped.geojson'
        beam_osm_grid_out = osm_tmp_dir / f'{scenario}_beam_osm_grid_intersection.geojson'

        beam_osm_path = MAPPING_CFG.precomputed_beam_osm_path
        if MAPPING_CFG.osm_links_path:
            try:
                map_beam_network_to_osm(
                    osm_path=MAPPING_CFG.osm_links_path,
                    beam_network_path=MAPPING_CFG.beam_network_path or netwrok_fp,
                    output_path=str(beam_osm_mapped_out),
                    network_osm_id_col=MAPPING_CFG.beam_osm_id_col,
                )
                beam_osm_path = str(beam_osm_mapped_out)
            except FileNotFoundError as exc:
                print(f"Skipping BEAM->OSM mapping: {exc}")

        if beam_osm_path:
            try:
                intersect_beam_osm_with_grid(
                    beam_osm_path=beam_osm_path,
                    grid_cells_path=MAPPING_CFG.grid_cells_path,
                    output_path=str(beam_osm_grid_out),
                    beam_osm_epsg=MAPPING_CFG.beam_osm_epsg,
                    grid_epsg=MAPPING_CFG.grid_epsg,
                    output_epsg=MAPPING_CFG.output_epsg,
                    beam_length_col=MAPPING_CFG.beam_length_col,
                )
            except FileNotFoundError as exc:
                print(f"Skipping BEAM+OSM->GRID intersection: {exc}")
        else:
            print("Skipping BEAM+OSM->GRID intersection: mapped BEAM+OSM path not available.")
        
        # vTypes = pd.read_csv('vehicletypes-baseline.csv',nrows = None)
        events = pd.read_csv(
            events_fp,
            usecols=events_column_types.keys(),
            nrows=nrows_events,
            dtype=events_column_types)
        network = pd.read_csv(netwrok_fp, usecols = ['linkId','fromLocationX','toLocationX','fromLocationY','toLocationY','linkLength'])
        network['linkLength'] = network['linkLength'] * correction_factor_curvature

        #############################################################
        print('########################## NETWORK ##########################')
        #############################################################

        networkExploded = split_network(network)
        networkExploded = calculate_midpoints(networkExploded)
        networkExploded = enrich_geo_data(networkExploded, GRID_grid, BGs, block_info, counties_shapefile)

        #############################################################
        print('########################## PATHTRAVERSAL ####################')
        #############################################################

        pathTraversal = filter_events(events)
        pathTraversal = drop_na_links(pathTraversal)
        pathTraversal = filter_modes(pathTraversal, scenario)
        pathTraversal = process_links(pathTraversal)  
        pathTraversal = link_times(pathTraversal)

        #############################################################
        print('########################## NETWORK ##########################')
        #############################################################

        networkParking = calculate_midpoints(network)
        networkParking = enrich_geo_data(networkParking, GRID_grid, BGs, block_info, counties_shapefile)

        #############################################################
        print('########################## PARK #############################')
        #############################################################

        parkDepartDatabase = prepare_park_departDatabase(pathTraversal)
        parkingDatabase, departDatabase, parkingDatabase_no_first = prepare_parkingDatabase(parkDepartDatabase, networkParking)

        # print(parkingDatabase[parkingDatabase['vehicle']=='rideHailVehicle-663229@Lyft'][['vehicle','parking_time','departureTime']])
        # print(parkingDatabase_no_first[parkingDatabase_no_first['vehicle']=='rideHailVehicle-663229@Lyft'][['vehicle','parking_time','departureTime']])
        # print(departDatabase[departDatabase['vehicle']=='rideHailVehicle-663229@Lyft'][['vehicle','parking_time','departureTime']])
        # print(parkDepartDatabase[parkDepartDatabase['vehicle']=='rideHailVehicle-663229@Lyft'][['vehicle','parking_time','departureTime']])

        #############################################################
        print('########################## EXPLODE ##########################')
        #############################################################

        explodedPathTraversal = explode_path_traversal(pathTraversal)
        explodedPathTraversal = merge_with_network(explodedPathTraversal, networkExploded)
        explodedPathTraversal = adjust_travel_times(explodedPathTraversal)
        explodedPathTraversal = calculate_speeds(explodedPathTraversal)

        print(len(explodedPathTraversal), '- len of explodedPathTraversal after processing')

        #############################################################
        print('########################## ASSIGN VEHICLE CLASS #############')
        #############################################################

        car_EMFACT_vmt = load_EMFACT_data(EMFACT_VMT_filepath, EMFACT_classes_car)
        freight_EMFACT_vmt = load_EMFACT_data(EMFACT_VMT_filepath, EMFACT_classes_freight)
        car_totalVMTbyCounty = calculate_vmt_distribution(car_EMFACT_vmt)
        freight_totalVMTbyCounty = calculate_vmt_distribution(freight_EMFACT_vmt)
        car_vmtLookup = create_vmtLookup(car_totalVMTbyCounty)
        freight_vmtLookup = create_vmtLookup(freight_totalVMTbyCounty)

        explodedPathTraversal = assign_vehicle_classes(explodedPathTraversal, car_vmtLookup, freight_vmtLookup)
        parkingDatabase = assign_vehicle_classes(parkingDatabase, car_vmtLookup, freight_vmtLookup)
        departDatabase = assign_vehicle_classes(departDatabase, car_vmtLookup, freight_vmtLookup)
        parkingDatabase_no_first = assign_vehicle_classes(parkingDatabase_no_first, car_vmtLookup, freight_vmtLookup)
        if is_cruise:
            explodedPathTraversal = correct_cruise_vehicle_classes(explodedPathTraversal)
            parkingDatabase = correct_cruise_vehicle_classes(parkingDatabase)
            departDatabase = correct_cruise_vehicle_classes(departDatabase)
            parkingDatabase_no_first = correct_cruise_vehicle_classes(parkingDatabase_no_first)
        print(len(explodedPathTraversal), len(parkingDatabase), len(departDatabase),  len(parkingDatabase_no_first),)
        
        explodedPathTraversal_grouped = explodedPathTraversal[["vehicle_class_emfact", "fuel_type_emfact","linkLength"]].groupby(["vehicle_class_emfact", "fuel_type_emfact"]).sum(numeric_only=True).reset_index().to_csv(f'VMT_by_EMFAC_vehicletype_{scenario}.csv')


        #############################################################
        print('########################## CORRECTION FACTORS ###############')
        #############################################################

        EMFACT_trips = load_EMFACT_data(EMFACT_trips_filepath, EMFACT_classes)
        EMFACT_vmt = load_EMFACT_data(EMFACT_VMT_filepath, EMFACT_classes)

        # EMFACT_trips.loc[EMFACT_trips['vehicle_class'].isin(EMFACT_classes_bus) , 'fuel'] = busesFuelType
        # EMFACT_vmt.loc[EMFACT_vmt['vehicle_class'].isin(EMFACT_classes_bus) , 'fuel'] = busesFuelType

        # EMFACT_trips['vehicle_class'].replace(busVehicleClassDict, inplace=True)
        # EMFACT_vmt['vehicle_class'].replace(busVehicleClassDict, inplace=True)

        if is_baseline:
            corr_VMT_by_county_dict, corr_Trips_by_county_dict = calculate_correction_factors(explodedPathTraversal, departDatabase, EMFACT_vmt, EMFACT_trips, scale, scenario)
        explodedPathTraversal = apply_correction_factors(explodedPathTraversal, corr_VMT_by_county_dict, 'corr_VMT_by_county')
        parkingDatabase= apply_correction_factors(parkingDatabase, corr_Trips_by_county_dict, 'corr_trips_by_county')
        departDatabase= apply_correction_factors(departDatabase, corr_Trips_by_county_dict, 'corr_trips_by_county')
        parkingDatabase_no_first = apply_correction_factors(parkingDatabase_no_first, corr_Trips_by_county_dict, 'corr_trips_by_county')

        #############################################################
        print('########################## EMFACT ###########################')
        #############################################################

        emfactEmisRates = load_EMFACT_data(EMFACTemfactFilepath, EMFACT_classes)
        emfactEmisRates = rename_emfact_columns(emfactEmisRates)
        explodedPathTraversal= categorize_speed_data(explodedPathTraversal)
        departDatabase = adjust_departDatabase_speed(departDatabase)

        #############################################################
        print('########################## ASSIGN EMFACT ####################')
        #############################################################
        explodedPathTraversal, parkingDatabase, departDatabase, parkingDatabase_no_first = process_emission_rates(emfactEmisRates, explodedPathTraversal, parkingDatabase, departDatabase, parkingDatabase_no_first, scale)

        #############################################################
        print('########################## AGGREGATE ########################')
        #############################################################
        print(explodedPathTraversal.keys())
    #         explodedPathTraversal.loc[explodedPathTraversal['bgid'] == 'Other', 'county_name'] = 'Other'
        county_counts = (
            explodedPathTraversal.groupby(['GRID', 'county_name'])
            .size()
            .reset_index(name='county_counts')
        )

        # Determine the most frequent county_tag for each GRID
        most_frequent_county = (
            county_counts.sort_values('county_counts', ascending=False)
            .drop_duplicates(subset=['GRID'])
            .sort_values('GRID')
        )
        gridToCountyDict = most_frequent_county.set_index('GRID')['county_name'].to_dict()
    #         gridToCountyDict = explodedPathTraversal[['GRID', 'county_name']].drop_duplicates().set_index('GRID')['county_name'].to_dict()

        BEAM_emis, BEAM_emis_park, BEAM_emis_park_no_first, BEAM_emis_dep = aggregate_emissions(explodedPathTraversal, parkingDatabase, departDatabase, parkingDatabase_no_first )
        BEAM_emis = merge_emission_dataframes([BEAM_emis, BEAM_emis_park, BEAM_emis_park_no_first, BEAM_emis_dep])
        BEAM_emis = calculate_total_emissions(BEAM_emis)

        BEAM_emis['county_name'] = BEAM_emis['GRID'].map(gridToCountyDict)

        #############################################################
        print('########################## SAVE ########################')
        #############################################################

        # Saving emission data
        save_emissions(BEAM_emis, scenario)

        explodedPathTraversal.groupby(['link','segment_part']).agg(
            {'linkLength':sum}
        ).reset_index().merge(
            networkExploded[['linkId','X', 'Y','segment_part']], 
            left_on = ['link', 'segment_part'], 
            right_on = ['linkId', 'segment_part']
           ).to_csv(f'VMT_{geoAggregationType}_{scenario}.csv')

        explodedPathTraversal.groupby(['link','segment_part']).agg(
            {'travelTime':sum}
        ).reset_index().merge(
            networkExploded[['linkId','X', 'Y','segment_part']], 
            left_on = ['link', 'segment_part'], 
            right_on = ['linkId', 'segment_part']
           ).to_csv(f'VHT_{geoAggregationType}_{scenario}.csv')

        explodedPathTraversal.groupby(['link', 'segment_part']).agg(
            linkLength_sum=('linkLength', 'sum'),
            count=('linkLength', 'size')
            ).reset_index().merge(
            networkExploded[['linkId', 'X', 'Y', 'segment_part']],
            left_on=['link', 'segment_part'],
            right_on=['linkId', 'segment_part']
            ).to_csv(f'Flow_{geoAggregationType}_{scenario}.csv')



def plot_workflow_diagram(output_path="src/impacts/tmp/impacts-workflow-generated.png"):
    """Generate a workflow diagram aligned with the current impacts pipeline."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f0f0f0")
    ax.set_facecolor("#f0f0f0")

    def box(x, y, w, h, text, fc="#ffffff", ec="#222222", lw=2, fs=12, ls="-"):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)
        return rect

    def arrow(x1, y1, x2, y2, color="#222222", lw=1.8, style="->"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle=style, color=color, lw=lw))

    # Inputs
    box(0.02, 0.82, 0.13, 0.08, "BEAM Events", fc="#cbe9ee", ec="#3f646d", fs=11)
    box(0.02, 0.70, 0.13, 0.08, "BEAM Emissions", fc="#cbe9ee", ec="#3f646d", fs=11)
    box(0.02, 0.58, 0.13, 0.08, "BEAM Network", fc="#b9d89a", ec="#58784d", fs=11)
    box(0.02, 0.46, 0.13, 0.08, "OSM Network", fc="#b9d89a", ec="#58784d", fs=11)
    box(0.02, 0.34, 0.13, 0.08, "SR/GRID Grid", fc="#cbe9ee", ec="#3f646d", fs=11)
    box(0.02, 0.14, 0.13, 0.08, "InMAP/AERMOD\nSR Matrices", fc="#cbe9ee", ec="#3f646d", fs=11)

    # Main left pipeline
    box(
        0.20,
        0.76,
        0.20,
        0.12,
        "Calculate BEAM\nVehicle Activity Emissions\n(run_main)",
        fc="#f8f8f8",
        ec="#777777",
        ls="--",
    )
    box(0.20, 0.59, 0.20, 0.09, "Distribute emissions\nacross OSM links", fc="#ffffff")
    box(0.20, 0.43, 0.20, 0.09, "Map OSM links\nto BEAM links\n[osm-chordify placeholder]", fc="#ffffff")
    box(0.20, 0.27, 0.20, 0.12, "Chordify OSM links\nwithin grid cells\n[osm-chordify placeholder]", fc="#ffffff")

    # Main right pipeline
    box(0.49, 0.70, 0.31, 0.13, "Assign emissions to SR cells\n(SOx, NOx, PM2.5, VOC, NH3, BC)", fc="#ffffff")
    box(0.52, 0.44, 0.35, 0.18, "Apply AERMOD and InMAP marginal impacts\n(primary + secondary PM species)\nto compute pollutant concentrations", fc="#ffffff")
    box(0.56, 0.24, 0.32, 0.11, "Integrate SR vectors/matrices\nto compute TotalPM2.5", fc="#ffffff")
    box(0.90, 0.08, 0.09, 0.12, "Exposure\noutputs", fc="#cbe9ee", ec="#3f646d", fs=10)
    box(0.34, 0.08, 0.11, 0.08, "model_data\n(.h5)", fc="#b8dde5", ec="#3f646d", fs=10)

    # Section marker
    box(0.80, 0.85, 0.12, 0.06, "IMPACTS", fc="#e9eef6", ec="#2d4f7a", fs=13)

    # Arrows
    arrow(0.15, 0.86, 0.20, 0.82, color="#9acfd8", lw=1.6)
    arrow(0.15, 0.74, 0.20, 0.635, color="#9acfd8", lw=1.6)
    arrow(0.15, 0.62, 0.20, 0.475, color="#9ac26d", lw=1.6)
    arrow(0.15, 0.50, 0.20, 0.315, color="#9ac26d", lw=1.6)
    arrow(0.15, 0.38, 0.20, 0.315, color="#9acfd8", lw=1.6)
    arrow(0.40, 0.635, 0.49, 0.765)
    arrow(0.40, 0.475, 0.49, 0.765)
    arrow(0.40, 0.315, 0.49, 0.765)
    arrow(0.65, 0.70, 0.70, 0.62)
    arrow(0.70, 0.44, 0.72, 0.35)
    arrow(0.88, 0.29, 0.94, 0.20)
    arrow(0.15, 0.18, 0.52, 0.53, color="#9acfd8", lw=1.6)
    arrow(0.45, 0.12, 0.56, 0.295, color="#9acfd8", lw=1.6)

    ax.text(
        0.02,
        0.02,
        "Code status: core BEAM/EMFAC workflow implemented. "
        "OSM chordify + OSM-BEAM mapping steps are explicit placeholders.",
        fontsize=10,
        color="#444444",
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved workflow diagram: {out}")


def run_plots(output_path="src/impacts/tmp/impacts-workflow-generated.png"):
    """Generate workflow diagram for documentation."""
    plot_workflow_diagram(output_path=output_path)


# Backward-compatible aliases
run_main_workflow = run_main
run_plot_workflow = run_plots
placeholder_osm_grid_intersections = intersect_beam_osm_with_grid
placeholder_osm_beam_mapping = map_beam_network_to_osm
osm_chordify_grid_intersections_placeholder = intersect_beam_osm_with_grid
osm_chordify_beam_mapping_placeholder = map_beam_network_to_osm
generate_impacts_workflow_diagram = plot_workflow_diagram

if __name__ == "__main__":
    run_main()
