import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import os
import sys

base_path = 'C:/Users/Nayeli/Documents/Thesis/'
sys.path.append(os.path.join(os.getcwd(), '0_Params')) # Append the path
import calculate_mortality_noadap   # Import function to calculate mortality without adaptation
import climate_models_info

''' Define important variables'''
scenarios = ['SSP126', 'SSP245', 'SSP370', 'SSP585']
scenarios_SSP = ['SSP1', 'SSP2', 'SSP3', 'SSP5']
years = np.arange(2015,2105,5)
groups = ['oldest', 'older', 'young']
index = pd.MultiIndex.from_product([groups, scenarios], names=['Age group', 'Scenario'])
results_noadapt = pd.DataFrame(index=index, columns=years)
climate_models = ['AWI-CM-1-1-MR', 'BCC-CSM2-MR', 'CAMS-CSM1-0', 'CESM2', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'CNRM-CM6-1-HR', 'EC-Earth3',
                  'EC-Earth3-Veg', 'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NorESM2-MM', 'TaiESM1']  
tmin = pd.read_csv(os.path.join(base_path, 'Response functions\\No adaptation\\Tmin.csv'))