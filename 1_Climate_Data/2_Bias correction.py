import numpy as np
import xarray as xr
import pandas as pd
from shapely.geometry import Point, Polygon
import geopandas as gpd
import sys, os
sys.path.append(os.path.join(os.getcwd(), '0_Params')) # Append the path
import relationship_temp_to_ir # Import relationship to match the 
import climate_models_info

base_path = 'C:/Users/Nayeli/Documents/' ### Select main path to store output 
carleton_path = 'D:\\data'   ### Set path to Carleton et al.'s folder
ir = gpd.read_file(f'{carleton_path}'+'\\2_projection\\1_regions\\ir_shp\\impact-region.shp') #Load .shp Carleton's file with impact regions


'''Calculate Present Day NSAT from reanalysis data ERA5'''
temp = xr.open_dataset('D:ERA5\\ERA5_Amon_tas_1990-2020.nc') ### This script uses ERA5 reanalysis data (see README)
temperatura_ERA5 = temp.mean('time').t2m  ### Select correct variable
latitud_ERA5 = -((temperatura_ERA5.latitude.values[1]-temperatura_ERA5.latitude.values[0]) + (temperatura_ERA5.latitude.values[2]-temperatura_ERA5.latitude.values[1]))/2
longitud_ERA5 = ((temperatura_ERA5.longitude.values[1]-temperatura_ERA5.longitude.values[0]) + (temperatura_ERA5.longitude.values[2]-temperatura_ERA5.longitude.values[1]))/2
relation = relationship_temp_to_ir.relationship(ir, longitud_ERA5, latitud_ERA5, temperatura_ERA5, extended=True) ### Use relationship function to generate indices to merge
temperatures = temperatura_ERA5.values.flatten() - 273.15
relation['temperature'] = temperatures[relation.index]
result_ERA5 = relation.groupby('index_right')['temperature'].mean() # Generate temp at impact region level
df_copy = ir.copy() # Append the results in a new dataframe
df_copy['T_mean'] = result_ERA5
df_rounded = df_copy.round(2)
df = df_rounded[['hierid','T_mean']] 
folder_path = f'{base_path}/Main folder/Climate Data/' ### Optional: Save the file with ERA5 NSAT per impact region
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
df.to_csv(f'{folder_path}/ERA5_NSAT_PresentDay.csv') 


'''Calculate Present Day NSAT (1990-2020) for climate models'''
# This requires having the .nc data of the desired model ensemble from 1990 to 2020 to calculate the GSAT
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585'] ### Data from 2015 to 2020 is generated per scenario so we use all the scenarios

def generate_present_nsat(climate_model, label, grid):  ### The entries of this function are related to the file names. Can be changed
    temp_hist = xr.open_dataset(f'D:\\Climate Models - Present Day NSAT\\tas_Amon_{climate_model}_historical_{label}_{grid}_1984-2014.nc') # Open historical data (1990-2014)
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
    folder_path2 = f'{base_path}/Main folder/Climate Data/Models_PresentDayNSAT'
    if not os.path.exists(folder_path2): # Save the result as a new netCDF file
        os.makedirs(folder_path2)
    temp.to_netcdf(f'{folder_path2}/tas_Amon_{climate_model}_present_{label}_{grid}_1990-2020.nc') 

for climate_model in climate_models_info.climate_models_dic: ### Run the function for every climate model
    generate_present_nsat(climate_model, climate_models_info.climate_models_dic[climate_model][0], climate_models_info.climate_models_dic[climate_model][1])


''' Calculate NSAT bias between Climate Models and ERA5 data '''
df = pd.read_csv(f'{base_path}/Main folder/Climate Data/ERA5_NSAT_PresentDay.csv', index_col=[0])
bias = pd.DataFrame(columns=climate_models_info.climate_models_dic.keys())
bias['hierid'] = df['hierid'] 

def calculate_bias(climate_model, label, grid): # This function involves generating present day NSAT at impact region level and substract it with the one of ERA5
    temp = xr.open_dataset(f'{base_path}/Main folder/Climate Data/Models_PresentDayNSAT/tas_Amon_{climate_model}_present_{label}_{grid}_1990-2020.nc')
    temperatura_model = temp.mean('time').tas
    latitud_model = ((temperatura_model.lat.values[1]-temperatura_model.lat.values[0]) + (temperatura_model.lat.values[2]-temperatura_model.lat.values[1]))/2
    longitud_model = ((temperatura_model.lon.values[1]-temperatura_model.lon.values[0]) + (temperatura_model.lon.values[2]-temperatura_model.lon.values[1]))/2
    relation_model = relationship_temp_to_ir.relationship(ir, longitud_model, latitud_model, temperatura_model, extended=False)
    temperatures_model = temperatura_model.values.flatten() - 273.15
    relation_model['temperature'] = temperatures_model[relation_model.index]
    result_model = relation_model.groupby('index_right')['temperature'].mean()
    bias = df['T_mean'] - result_model
    print(climate_model)
    return bias

for climate_model in climate_models_info.climate_models_dic:  ## Apply the function to every climate model
    bias.loc[:,climate_model] = calculate_bias(climate_model, climate_models_info.climate_models_dic[climate_model][0], climate_models_info.climate_models_dic[climate_model][1])
bias.to_csv(f'{base_path}/Main folder/Climate Data/Bias_Correction.csv') ### Save file of bias per impact region and climate model


'''Correct daily data per impact region and every climate model'''

bias = pd.read_csv(f'{base_path}/Main folder/Climate Data/Bias_Correction.csv')
bias.set_index('hierid', inplace=True)

def process_file(climate_model, scenario, year):  ### This function needs to run for all the files previously generated in step 1.1
    folder_path_ensemble = f'{base_path}/Climate data/Climate ensemble/{climate_model}/{scenario}/' # Open files created in step 1.1
    df = pd.read_csv(f'{folder_path_ensemble}/{climate_model}_{scenario}_{year}.csv')
    df.set_index('hierid', inplace=True)
    df = df.drop('Unnamed: 0', axis=1)
    resultado = df.add(bias[climate_model], axis='index').round(1)  # Apply bias correction
    folder_path_ensemble_corrected = f'{base_path}/Climate data/Climate ensemble corrected/{climate_model}/{scenario}'
    if not os.path.exists(folder_path_ensemble_corrected): # Save corrected data in a new folder
        os.makedirs(folder_path_ensemble_corrected)
    resultado.to_csv(f'{folder_path_ensemble_corrected}/BC_{climate_model}_{scenario}_{year}.csv') 

#process_file('AWI-CM-1-1-MR', 'SSP126', 2015) ### EXAMPLE. Run for every model, scenario and year. Takes around 5s