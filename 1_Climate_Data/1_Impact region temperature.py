import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np 
from shapely.geometry import Point, Polygon

''' Load .shp Carleton's file with impact regions '''
carleton_path = 'D:\\data'   ### Set path to Carleton et al.'s folder
ir = gpd.read_file(f'{carleton_path}'+'\\2_projection\\1_regions\\ir_shp\\impact-region.shp')

''' Load temperature data file '''
model = ''
year = 
scenario = '' ### Set the climate model name here
model_path =  ''   # Set the path to the climate model data here
tas_SSP = xr.open_dataset(f'{model_path}')
temperatura = tas_SSP['tas']

''' Calculate manually lat and lon dimensions of climate file '''
latitud = ((temperatura.lat.values[1]-temperatura.lat.values[0]) + (temperatura.lat.values[2]-temperatura.lat.values[1]))/2
longitud = ((temperatura.lon.values[1]-temperatura.lon.values[0]) + (temperatura.lon.values[2]-temperatura.lon.values[1]))/2
print(latitud, longitud)

def create_square(lon, lat): ### function only works for squared grids
    return Polygon([
        (lon, lat),
        (lon + longitud, lat),
        (lon + longitud, lat + latitud),
        (lon, lat + latitud)
    ])

'''Get indices of the spatial join between the climate data and '''
new_lon = temperatura.lon.values ### Obtaining the lon and lat values
new_lat = temperatura.lat.values
new_lon = np.where(new_lon > 180, new_lon - 360, new_lon) ### Converting lon to right -180 to 180 degrees. Otherwise map is shifted 180 deg.
lon2d, lat2d = np.meshgrid(new_lon, new_lat) ### Create meshgrid
lon_flatten = lon2d.flatten() ### Flatten lon and lat values
lat_flatten = lat2d.flatten()
points_df = pd.DataFrame({ 'longitude': lon_flatten, 'latitude': lat_flatten}) ### Create dataframe with lat and lon
points_df['geometry'] = [create_square(lon, lat) for lon, lat in zip(points_df.longitude, points_df.latitude)] ### Make the geometry with squares for the temperature values
points_gdf = gpd.GeoDataFrame(points_df) ### Convert the DataFrame to GeoDataFrame
points_gdf = points_gdf.reset_index()
points_gdf.crs = ir.crs  ### Set same CRS
relationship = gpd.sjoin(points_gdf, ir, how='inner', predicate='intersects') ### Make the spatial join only once and recover the indices
relationship = relationship[['geometry','index_right']]


''' Select a day to merge the temperature data into the impact region level '''
def new_response_function(day):
    temperatures = temperatura.sel(time=day).values.flatten() - 273.15
    # Associate the temperatures with the relationship precalculated
    relationship['temperature'] = temperatures[relationship.index]
    result = relationship.groupby('index_right')['temperature'].mean()
    print(day)
    return result

''' Function to merge temperature data for all days in a file '''
def create_file(model, year, scenario):
    fechas = temperatura['time'].values
    date_list = fechas[np.isin(fechas.astype('datetime64[Y]'), np.datetime64(f'{year}', 'Y'))]
    date_list = date_list.astype('datetime64[D]').astype(str)

    df = pd.DataFrame(ir['hierid'])
    for item in date_list:
        df[item] = new_response_function(item)
    df_rounded = df.round(1)
    df_rounded.to_csv(f'{base_path}\\{model}\\SSP{scenario}\\{model}_SSP{scenario}_{year}.csv') ### Requires to create subfolders per model and scenario

base_path = '' ### Select the folder for the output
create_file(model, year, scenario)