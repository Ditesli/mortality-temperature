import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import utils_common.population as pop
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt



def calculate_temperature_zones(wdir, era5_path):
    
    '''
    Calculate the Temperature zones (tz) per each grid cell, according to GBD methodology 
    by calculating the mean from 1980-2019.
    Following their method, monthly mean ERA5 data was retreived from the website: 
    cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means
    Sea values were set to NaN using the mask provided by ERA5:
    confluence.ecmwf.int/pages/viewpage.action?pageId=140385202#ERA5Land:datadocumentation-parameterlistingParameterlistings
    '''
    
    print('Calculating temperature zones...')

    # Open and preprocess population data and set mask for ALL cells with population
    mask_pop = pop.get_all_population_data(wdir, return_pop=False)
    
    print('[1] Population data loaded and mask created.')
    
    # Open ERA5 t2m data and discard dummy var 'number'
    era5_t2m = xr.open_dataset(era5_path+'era5_t2m_mean_1980-2019.nc') 
    era5_t2m = era5_t2m.drop_vars('number')
    
    # Calculate 1980-2019 mean
    era5_t2m_mean = era5_t2m.mean(dim='valid_time')
    
    # Convert to Celius
    era5_t2m_mean -= 273.15

    # Shift coordinates
    era5_t2m_mean = era5_t2m_mean.assign_coords(longitude=((era5_t2m_mean.longitude + 180) % 360 - 180)).sortby("longitude")

    # Interpolate to align temperature and population latitude
    era5_t2m_mean = era5_t2m_mean.interp(longitude=mask_pop.longitude, 
                                         latitude=mask_pop.latitude, 
                                         method="nearest")
    
    print('[2] ERA5 historical temperature data loaded and preprocessed.')

    # Open ERA5 land-sea mask, discard dummy var 'time' and rename variable to 't2m' for easier handling
    era5_land_sea = xr.open_dataset(era5_path+'lsm_1279l4_0.1x0.1.grb_v4_unpack.nc') 
    era5_land_sea = era5_land_sea.drop_vars('time')
    era5_land_sea = era5_land_sea.rename({'lsm': 't2m'})

    # Shift coordinates
    era5_land_sea = era5_land_sea.assign_coords(longitude=((era5_land_sea.longitude + 180) % 360 - 180)).sortby("longitude")

    ### Interpolate land-sea data to 0.25 degree resolution to match population grids
    era5_land_sea_15min = era5_land_sea.interp(longitude=mask_pop.longitude, 
                                               latitude=mask_pop.latitude, 
                                               method="nearest")

    ### Set sea and Antarctica to NaN and ensure that ANY cell (of any scenario and year) with POP data has a temperature zone
    era5_w_nan = xr.where(
        ((mask_pop.GPOP) | (era5_land_sea_15min.t2m > 0)) & (era5_t2m_mean.latitude >= -60),
        era5_t2m_mean,
        np.nan
        )
    
    print('[3] Land-sea mask applied and population mask enforced.')
    
    # Round data to get temperature zones, clip values between 6 and 28
    era5_rounded = era5_w_nan.where(era5_w_nan.isnull(), era5_w_nan.round())
    temp_zone_rounded = np.clip(era5_rounded.t2m, 6, 28).mean(dim='time')

    # Save in project data folder
    temp_zone_rounded.to_netcdf(wdir+'data/temperature_zones/ERA5_mean_1980-2019_land_t2m_tz.nc')
    
    print('[4] Temperature zones calculation completed and saved.')

    
    
def generate_raster_gbd_locations(wdir):
    
    '''
    Convert the level 3 GBD shapefile to a raster format (nmupy array)
    '''

    # Load population xarray and mask
    mask_pop = pop.get_all_population_data(wdir, return_pop=False)

    # Load GBD shapefile with level 3 locations
    # TODO: replace later with level 4 locations
    gbd_locations_level3 = gpd.read_file(wdir+'data/gbd_locations/GBD_shapefile.shp')

    # Convert geopandas dataframe to xarray using location ID label for the pixel values
    gbd_rasterized_xarray = convert_vector_to_raster(mask_pop, gbd_locations_level3, 'loc_id', set_zero_to_nan=True)

    # Interpolate TMREL to areas with NaN but population is positive (maily coasts and islands)
    gbd_rasterized_xarray = fill_missing_with_nearest(gbd_rasterized_xarray, mask_pop.GPOP)

    # Save file in GBD_Data folder
    gbd_rasterized_xarray.to_netcdf(f'{wdir}/data/regions_classification/GBD/GBD_locations_level3.nc')

    

def convert_vector_to_raster(mask_pop: xr.Dataset, vector_data: gpd.GeoDataFrame, 
                             column: str, set_zero_to_nan=True):
    
    '''
    This function uses the GBD level3 locations shapefile and converts it into 
    an xarray setting data from a selected column of the shapefile
    '''
    
    # Extract coordinate information
    y_coords = mask_pop.coords['latitude'].values
    x_coords = mask_pop.coords['longitude'].values

    # Calculate resolution
    resolution_y = np.abs(y_coords[1] - y_coords[0])
    resolution_x = np.abs(x_coords[1] - x_coords[0])

    # Define origin (upper left corner)
    origin_x = x_coords.min()
    origin_y = y_coords.max()

    # Create affine transform
    transform = from_origin(origin_x, origin_y, resolution_x, resolution_y)

    # Create a list of shapes (geometry, loc_id) tuples
    shapes = [(geom, loc_id) for geom, loc_id in zip(vector_data.geometry, vector_data[column])]

    # Rasterize the shapefile to the array dimensions
    height, width = len(y_coords), len(x_coords)
    rasterized_data = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0)
    
    if set_zero_to_nan:
        rasterized_data[rasterized_data == 0] = np.nan  # Set fill value to NaN
    
    # Convert to xarray 
    rasterized_xarray = xr.DataArray(rasterized_data, coords=[y_coords, x_coords], dims=['latitude', 'longitude'])
    rasterized_xarray.name = column

    # Combine with original xarray to include other non-NaN values but with loc_id data
    rasterized_xarray = xr.Dataset({column:rasterized_xarray})
        
    return rasterized_xarray



def fill_missing_with_nearest(xarray_dataset, mask_positive_pop):
    
    '''
    Fill missing values (NaN) in an xarray dataset using the nearest
    valid value, but only in locations where the population mask indicates
    positive population.
    '''
    
    filled_xarray = xarray_dataset.copy()

    # Check if the dataset has 2 or 3 dimensions
    
    if len(tuple(xarray_dataset.sizes.values())) == 3:
        
        # 3D array with multiple draws (TMRELs)
        for draw in range(xarray_dataset.shape[2]):
            xarray_draw = xarray_dataset[:, :, draw].values
            missing_mask = np.isnan(xarray_draw) & mask_positive_pop.values

            # Mask where there are valid values
            valid_mask = ~np.isnan(xarray_draw)

            # Get coordinates of the closest valid value
            distance, indices = distance_transform_edt(~valid_mask, return_indices=True)

            # Fill NaN
            xarray_draw[missing_mask] = xarray_draw[indices[0][missing_mask], indices[1][missing_mask]]

            # Store in the new xarray
            filled_xarray[:, :, draw] = xarray_draw
            
    if len(tuple(xarray_dataset.sizes.values())) == 2:
        
        # Single 2D array (e.g., GBD locations)
        xarray_draw = xarray_dataset.loc_id.values
        missing_mask = np.isnan(xarray_draw) & mask_positive_pop.values
        # Mask where there are valid values
        valid_mask = ~np.isnan(xarray_draw)
        
        # Get coordinates of the closest valid value
        distance, indices = distance_transform_edt(~valid_mask, return_indices=True)

        # Fill NaN
        xarray_draw[missing_mask] = xarray_draw[indices[0][missing_mask], indices[1][missing_mask]]

        # Store in the new xarray
        filled_xarray['loc_id'][:, :] = xarray_draw

    return filled_xarray



def import_tmrels_files(wdir, year):
    
    ''' 
    Open TMREL files and concat them into a single DataFrame with multi-index
    level 1: loc_id, level 2: temperature zone.
    The code needs to import the raster file with the GBD locations first to 
    get the unique loc_ids.
    '''
    
    # Open rasterized GBD level 3 locations and convert to numpy array
    gbd_rasterized = xr.open_dataset(f'{wdir}/data/regions_classification/GBD/GBD_locations_level3.nc')
    gbd_rasterized = gbd_rasterized.loc_id.values

    # Get unique location IDs, excluding -1 and NaN and covert to integers
    loc_ids = np.unique(gbd_rasterized)[1:-1].astype(int)

    # Create an empty list to store filtered DataFrames
    df_list = []

    # Loop through each location ID level 3
    for loc_id in loc_ids:
        file_path = wdir+f"data/exposure_response_functions/TMRELs/tmrel_{loc_id}.csv"
        
        # Read the CSV file into a DataFrame with a multi-index
        df = pd.read_csv(file_path, index_col=[0, 1, 2])

        # Filter the DataFrame for the year -----> This can be 1990, 2010, 2020
        df_filtered = df.xs(year, axis=0, level=1)

        # Append the filtered DataFrame to the list
        df_list.append(df_filtered)

    # Concatenate all filtered DataFrames into a single DataFrame
    gbd_tmrel = pd.concat(df_list, axis=0)
    
    return gbd_rasterized, gbd_tmrel



def read_temperature_zones(wdir):
    
    '''
    Read ERA5 temperature zones raster file and convert to numpy array
    '''
        
    # Import ERA5 temperature zones
    era5_tz = xr.open_dataset(wdir+'data/temperature_zones/ERA5_mean_1980-2019_land_t2m_tz.nc')

    # Convert file to numpy array
    era5_tz = era5_tz.t2m.values
    
    return era5_tz



def generate_tmrels_rasters(wdir, year):

    ''' 
    Create a new 3D array with the same dimensions as the population data and a 
    third dimension for the TMREL draws
    '''
    
    print(f'Generating TMRELs for year {year}...')
    
    # Import GBD rasterized locations, TMRELs and temperature zones
    gbd_rasterized, gbd_tmrel = import_tmrels_files(wdir, year)
    mask_pop = pop.get_all_population_data(wdir, return_pop=False)
    temperature_zones = read_temperature_zones(wdir)
    
    print('[1] GBD locations, TMRELs and temperature zones imported.')
    
    # Create empty 3D array for TMRELs
    tmrel_array = np.empty((720, 1440, 100))

    for i in range(gbd_rasterized.shape[0]):
        
        for j in range(gbd_rasterized.shape[1]):
            
            loc_id = gbd_rasterized[i, j]
            temp_zone = temperature_zones[i, j]
            
            # Skip if either value is NaN
            if np.isnan(loc_id) or np.isnan(temp_zone):
                tmrel_array[i, j, :] = np.nan
                
            # Handle valid loc_id and temp_zone
            else:
                # Check if the index exists in the DataFrame
                if (loc_id, temp_zone) in gbd_tmrel.index:
                    tmrel_array[i, j, :] = gbd_tmrel.loc[(loc_id, temp_zone)].values
                # Handle missing indices
                else:
                    tmrel_array[i, j, :] = np.nan  
                    
    print('[2] TMRELs array populated.')
                    
    # Convert to xarray DataArray
    tmrel_xarray = xr.DataArray(tmrel_array, dims=("latitude", "longitude", "draw"), 
                                coords=dict(latitude=mask_pop.latitude.values,
                                            longitude=mask_pop.longitude.values, 
                                            draw=np.arange(1, 101)))
    tmrel_xarray.name = "tmrel"

    # Interpolate TMREL to areas with NaN but population is positive (maily coasts and islands)
    interpolated_tmrel = fill_missing_with_nearest(tmrel_xarray, mask_pop.GPOP)

    # Save file in GBD_Data folder
    interpolated_tmrel.to_netcdf(wdir+f'data/exposure_response_functions/TMRELs_{year}.nc')
    
    print(f'[3] TMRELs for year {year} generated and saved.')
    
    
    
def generate_gdp_rasters(wdir, gdp_dir, year):
    
    '''
    Convert GDP and GDPPC to xarray for future analysis and calculations
    '''

    # Load GBD shapefile with level 3 locations
    gbd_locations_level3 = gpd.read_file(f'{wdir}\\GBD_Data\\GBD_locations\\Shapefile\\GBD_shapefile.shp')
    
    # Open and preprocess population data and set mask for ALL cells with population
    mask_pop = pop.get_all_population_data(wdir, return_pop=False)

    # Load files
    gdp = pd.read_excel(gdp_dir+'iamc_db gdp.xlsx')
    gdppc = pd.read_excel(gdp_dir+'iamc_db gdppc.xlsx')

    # Select closest year to 2019 and SSP2 data
    gdp_year = gdp[gdp['Scenario'] == 'SSP2'][['Region', year]].rename(columns={year:'gdp'})
    gdppc_year = gdppc[gdppc['Scenario'] == 'SSP2'][['Region', year]].rename(columns={year:'gdppc'})

    # Merge to gbd locations shapefile
    gbd_locations_level3_gdp = gbd_locations_level3.merge(gdp_year, right_on='Region', left_on='ihme_lc_id', how='left')
    gbd_locations_level3_gdppc = gbd_locations_level3.merge(gdppc_year, right_on='Region', left_on='ihme_lc_id', how='left')

    # Convert geopandas dataframe to xarray using location ID label for the pixel values
    gbd_rasterized_gdp = convert_vector_to_raster(mask_pop, gbd_locations_level3_gdp, 'gdp', set_zero_to_nan=True)
    gbd_rasterized_gdppc = convert_vector_to_raster(mask_pop, gbd_locations_level3_gdppc, 'gdppc', set_zero_to_nan=True)

    gbd_rasterized_gdp_gdppc = xr.merge([gbd_rasterized_gdp, gbd_rasterized_gdppc])
    gbd_rasterized_gdp_gdppc.to_netcdf(f'{wdir}\\SocioeconomicData\\EconomicData\\GDP_SSP2_{year}.nc')

    # Select only data from original countries 
    original_countries = ['BRA', 'CHL', 'CHN', 'COL', 'GTM', 'MEX', 'NZL', 'ZAF', 'USA']
    gdp_countries = gdp[(gdp['Region'].isin(original_countries)) & (gdp['Scenario'] == 'SSP2')][['Region', year]].rename(columns={year:'gdp'})
    gdppc_countries = gdppc[(gdppc['Region'].isin(original_countries)) & (gdppc['Scenario'] == 'SSP2')][['Region', year]].rename(columns={year:'gdp'})

    # Merge to gbd locations shapefile
    gbd_locations_countries_gdp = gbd_locations_level3.merge(gdp_countries, right_on='Region', left_on='ihme_lc_id', how='left')
    gbd_locations_countries_gdppc = gbd_locations_level3.merge(gdppc_countries, right_on='Region', left_on='ihme_lc_id', how='left')

    # Convert geopandas dataframe to xarray using location ID label for the pixel values
    gbd_rasterized_countries_gdp = convert_vector_to_raster(mask_pop, gbd_locations_countries_gdp, 'gdp', set_zero_to_nan=True)
    gbd_rasterized_countries_gdppc = convert_vector_to_raster(mask_pop, gbd_locations_countries_gdppc, 'gdppc', set_zero_to_nan=True)

    gbd_rasterized_countries_gdp_gdppc = xr.merge([gbd_rasterized_countries_gdp, gbd_rasterized_countries_gdppc])
    gbd_rasterized_countries_gdp_gdppc.to_netcdf(f'{wdir}\\SocioeconomicData\\EconomicData\\GDP_SSP2_{year}_OriginalCountries.nc')