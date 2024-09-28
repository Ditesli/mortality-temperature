import numpy as np
import xarray as xr
import pandas as pd
from shapely.geometry import Point, Polygon
import geopandas as gpd
import sys
import os

sys.path.append(os.path.join(os.getcwd(), '0_Params')) # Append the path
import relationship_temp_to_ir # Import relationship to match the 
import climate_models_info


''' Load .shp Carleton's file with impact regions '''
carleton_path = 'D:\\data'   ### Set path to Carleton et al.'s folder
ir = gpd.read_file(f'{carleton_path}'+'\\2_projection\\1_regions\\ir_shp\\impact-region.shp')


'''Calculate Present Day NSAT from reanalysis data for each impact region and save file'''
temp = xr.open_dataset('D:ERA5\\ERA5_Amon_tas_1990-2020.nc') ### This script uses ERA5 reanalysis data (see README)
temperatura_ERA5 = temp.mean('time').t2m  ### Select correct variable

latitud_ERA5 = -((temperatura_ERA5.latitude.values[1]-temperatura_ERA5.latitude.values[0]) + (temperatura_ERA5.latitude.values[2]-temperatura_ERA5.latitude.values[1]))/2
longitud_ERA5 = ((temperatura_ERA5.longitude.values[1]-temperatura_ERA5.longitude.values[0]) + (temperatura_ERA5.longitude.values[2]-temperatura_ERA5.longitude.values[1]))/2

result_ERA5 = relationship_temp_to_ir.relationship(ir, longitud_ERA5, latitud_ERA5, temperatura_ERA5, extended=True) ### Use relationship function

df_copy = ir.copy()
df_copy['T_mean'] = result_ERA5
df = df_copy[['hierid','T_mean']]
df.to_csv(os.path.join(os.getcwd(), '1_Climate_Data', 'Intermediate_ERA5_NSAT_PresentDay.csv')) ### Optional: Save the file with ERA5 NSAT per impact region


'''Calculate Present Day NSAT (1990-2020) for climate models'''

# This requires having the .nc data of the desired model ensemble to calculate the GSAT. See example below

scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585'] ### Data from 2015 to 2020 is generated per scenario so we use all the scenarios

def generate_present_nsat(climate_model, label, grid):  ### The entries of this function are related to the file names. Can be changed
    temp_hist = xr.open_dataset(f'D:\\Climate Models - Present Day NSAT\\tas_Amon_{climate_model}_historical_{label}_{grid}_1990-2014.nc') # Open historical data (1990-2014)
    temp_hist = temp_hist.sel(time=slice('1990', '2014'))
    temp_ssp = []
    for scenario in scenarios:
        temp_ssp_scenario = xr.open_dataset(f'D:\\Climate Models - Future NSAT\\tas_Amon_{climate_model}_{scenario}_{label}_{grid}_2015-2100.nc') # Open scenario data from 2015 to 2020
        temp_ssp_scenario = temp_ssp_scenario.sel(time=slice('2015', '2020'))
        temp_ssp.append(temp_ssp_scenario) # Add the xarray of the current scenario to the list temp_ssp

    temp_ssp_concat = xr.concat(temp_ssp, dim='scenario') # Concatenates xarrays along a new 'scenario' dimension
    temp_ssp_mean = temp_ssp_concat.mean(dim='scenario') # Calculate the average temperature for all scenarios
    temp = xr.concat([temp_hist, temp_ssp_mean], dim='time') # Concatenate historical and scenario data to have the period 1990-2020
    temp = temp.tas
    temp.to_netcdf(f'D:\\Climate Models - Present Day NSAT\\tas_Amon_{climate_model}_present_{label}_{grid}_1990-2020.nc') # Save the result as a new netCDF file

for climate_model in climate_models_info.climate_models_dic: ### Run the function for every climate model
    generate_present_nsat(climate_model, climate_models_info.climate_models_dic[climate_model][0], climate_models_info.climate_models_dic[climate_model][1])


''' Calculate NSAT bias between Climate Models and ERA5 data '''
# df = pd.read_csv('Intermediate_ERA5_NSAT_PresentDay.csv', index_col=[0])
bias = pd.DataFrame(columns=climate_models_info.climate_models_dic.keys())
bias['hierid'] = df['hierid'] 

def calculate_bias(climate_model, label, grid):
    temp = xr.open_dataset(f'D:\\Climate Models - Present Day NSAT\\tas_Amon_{climate_model}_present_{label}_{grid}_1990-2020.nc')
    latitud = ((temperatura.lat.values[1]-temperatura.lat.values[0]) + (temperatura.lat.values[2]-temperatura.lat.values[1]))/2
    longitud = ((temperatura.lon.values[1]-temperatura.lon.values[0]) + (temperatura.lon.values[2]-temperatura.lon.values[1]))/2
    print(latitud, longitud)
    result = relationship(longitud, latitud, temperatura)
    bias = df['T_mean'] - result
    return bias

for climate_model in climate_models_info.climate_models_dic:
    bias.loc[:,climate_model] = calculate_bias(climate_model, climate_models_info.climate_models_dic[climate_model][0], climate_models_info.climate_models_dic[climate_model][1])

#bias.to_csv('Bias_Correction.csv') ### Save file


'''Correct daily data per impact region and every climate model'''

years = np.arange(2015,2101)
bias = pd.read_csv('Bias_Correction.csv')
bias.set_index('hierid', inplace=True)

def process_file(climate_model, scenario, year):  ### This function needs to run for all the files previously generated in step 1.1
    df = pd.read_csv(f'D:\\Climate models\\{climate_model}\\{scenario}\\{climate_model}_{scenario}_{year}.csv')
    df.set_index('hierid', inplace=True)
    df = df.drop('Unnamed: 0', axis=1)
    resultado = df.add(bias[climate_model], axis='index').round(1)
    resultado.to_csv(f'D:\\Climate models - Bias Corrected\\{climate_model}\\{scenario}\\BC_{climate_model}_{scenario}_{year}.csv') 