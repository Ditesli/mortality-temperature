import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import geopandas as gpd

def relationship(ir, longitud, latitud, temperatura, extended=False):
    
    def create_square(lon, lat): ### function only works for climate data with squared grids
        return Polygon([
            (lon, lat),
            (lon + longitud, lat),
            (lon + longitud, lat + latitud),
            (lon, lat + latitud)
        ])

    if extended:
        new_lon = temperatura.longitude.values   # Obtaining the lon and lat values, format varies depending of the file
        new_lat = temperatura.latitude.values
    else:
        new_lon = temperatura.lon.values   # Obtaining the lon and lat values
        new_lat = temperatura.lat.values  

    new_lon = np.where(new_lon > 180, new_lon - 360, new_lon)   # Converting lon to right -180 to 180 degrees
    lon2d, lat2d = np.meshgrid(new_lon, new_lat)    # Create meshgrid
    lon_flatten = lon2d.flatten()    # Flatten the values
    lat_flatten = lat2d.flatten()
    points_df = pd.DataFrame({ 'longitude': lon_flatten, 'latitude': lat_flatten})    # Create dataframe with lat and lon
    points_df['geometry'] = [create_square(lon, lat) for lon, lat in zip(points_df.longitude, points_df.latitude)]  # Make the geometry with squares for the temperature values
    points_gdf = gpd.GeoDataFrame(points_df) # Convertir el DataFrame a un GeoDataFrame
    points_gdf = points_gdf.reset_index()
    points_gdf.crs = ir.crs  # Set same .crs
    relationship = gpd.sjoin(points_gdf, ir, how='inner', predicate='intersects') #Make spatial join only once
    relationship = relationship[['geometry','index_right']]
    
    #temperatures = temperatura.values.flatten() - 273.15
    #relationship['temperature'] = temperatures[relationship.index]
    #result = relationship.groupby('index_right')['temperature'].mean()

    return relationship