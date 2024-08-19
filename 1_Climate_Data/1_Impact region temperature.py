import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np 
from shapely.geometry import Point, Polygon

base_path = '' ### Set your wd here

# Load .shp Carleton's file with impact regions 
ir = gpd.read_file(f'{base_path}'+'Carleton\\data\\2_projection\\1_regions\\ir_shp\\impact-region.shp')

# Load temperature data file
model = # Set the model name here
model_path = # Set the path to the model data here
tas_SSP = xr.open_dataset(f'{file_path}')
temperatura = tas_SSP['tas']

# Calculate manually lat and lon dimensions
latitud = ((temperatura.lat.values[1]-temperatura.lat.values[0]) + (temperatura.lat.values[2]-temperatura.lat.values[1]))/2
longitud = ((temperatura.lon.values[1]-temperatura.lon.values[0]) + (temperatura.lon.values[2]-temperatura.lon.values[1]))/2
print(latitud, longitude)

def create_square(lon, lat):
    return Polygon([
        (lon, lat),
        (lon + longitud, lat),
        (lon + longitud, lat + latitud),
        (lon, lat + latitud)
    ])

# Obtaining the lon and lat values
new_lon = temperatura.lon.values
new_lat = temperatura.lat.values
# Converting lon to right -180 to 180 degrees
new_lon = np.where(new_lon > 180, new_lon - 360, new_lon)
# Create meshgrid
lon2d, lat2d = np.meshgrid(new_lon, new_lat)
# Flatten the values
lon_flatten = lon2d.flatten()
lat_flatten = lat2d.flatten()
# Create dataframe with lat and lon
points_df = pd.DataFrame({ 'longitude': lon_flatten, 'latitude': lat_flatten})
# Make the geometry with squares for the temperature values
points_df['geometry'] = [create_square(lon, lat) for lon, lat in zip(points_df.longitude, points_df.latitude)]
# Convert the DataFrame to GeoDataFrame
points_gdf = gpd.GeoDataFrame(points_df)
points_gdf = points_gdf.reset_index()
# Set same CRS
points_gdf.crs = ir.crs  

#Make the spatial join only once and recover the indices
relationship = gpd.sjoin(points_gdf, ir, how='inner', predicate='intersects')
relationship = relationship[['geometry','index_right']]


# Select a day to merge the temperature data into the impact region level
def new_response_function(day):
    temperatures = temperatura.sel(time=day).values.flatten() - 273.15
    # Associate the temperatures with the relationship precalculated
    relationship['temperature'] = temperatures[relationship.index]
    result = relationship.groupby('index_right')['temperature'].mean()
    print(day)
    return result

# Function to merge temperature data for all days in a file
def create_file(model, year, scenario):
    fechas = temperatura['time'].values
    date_list = fechas[np.isin(fechas.astype('datetime64[Y]'), np.datetime64(f'{year}', 'Y'))]
    date_list = date_list.astype('datetime64[D]').astype(str)

    df = pd.DataFrame(ir['hierid'])
    for item in date_list:
        df[item] = new_response_function(item)
    df_rounded = df.round(1)
    df_rounded.to_csv(f'{base_path}\\{model}\\SSP{scenario}\\{model}_SSP{scenario}_{year}.csv')
