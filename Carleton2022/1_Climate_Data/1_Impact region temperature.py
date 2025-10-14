import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np 
from shapely.geometry import Point, Polygon
import sys, os
sys.path.append(os.path.join(os.getcwd(), '0_Params')) # Append the path
import relationship_temp_to_ir 

''' Load .shp Carleton's file with impact regions '''
carleton_path = 'D:\\data'   ### Set path to Carleton et al.'s folder
ir = gpd.read_file(f'{carleton_path}'+'\\2_projection\\1_regions\\ir_shp\\impact-region.shp')

''' Load temperature data file and define model specs'''
model = 'CESM2' # Set specific model. See example
year = 2015 # Set year in int format
scenario = 'SSP126' # Set the scenario here
model_path =  'C:/Users/Nayeli/Downloads/tas_day_AWI-CM-1-1-MR_ssp126_r1i1p1f1_gn_20150101-20151231.nc'   # Set the path to the climate model data here
tas_SSP = xr.open_dataset(f'{model_path}')
temperatura = tas_SSP['tas']

''' Calculate manually lat and lon dimensions of climate file '''
latitud = ((temperatura.lat.values[1]-temperatura.lat.values[0]) + (temperatura.lat.values[2]-temperatura.lat.values[1]))/2
longitud = ((temperatura.lon.values[1]-temperatura.lon.values[0]) + (temperatura.lon.values[2]-temperatura.lon.values[1]))/2
relation = relationship_temp_to_ir.relationship(ir, longitud, latitud, temperatura, extended=False)


''' Select a day to merge the temperature data into the impact region level '''
def generate_temp_per_day(day):
    temperatures = temperatura.sel(time=day).values.flatten() - 273.15
    relation['temperature'] = temperatures[relation.index] # Associate the temperatures with the relationship precalculated
    result = relation.groupby('index_right')['temperature'].mean()
    print(day)
    return result

''' Function to merge temperature data for all days in a file '''
def create_file(model, year, scenario):
    fechas = temperatura['time'].values
    date_list = fechas[np.isin(fechas.astype('datetime64[Y]'), np.datetime64(f'{year}', 'Y'))]
    date_list = date_list.astype('datetime64[D]').astype(str)
    df = pd.DataFrame(ir['hierid'])
    for item in date_list:
        df[item] = generate_temp_per_day(item)
    df_rounded = df.round(1)
    folder_path = f'{base_path}/Climate ensemble preliminar/{model}/{scenario}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    df_rounded.to_csv(f'{folder_path}/{model}_{scenario}_{year}.csv') ### Requires to create subfolders per model and scenario

base_path = 'C:/Users/Nayeli/Documents/Climate data' ### Select the folder for the output
create_file(model, year, scenario)  ### This function can be put in a for loop to include more years or scenario