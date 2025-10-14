import numpy as np
import xarray as xr
import glob
import re
import pandas as pd
import geopandas as gpd
from scipy.interpolate import griddata
from rasterio.features import rasterize
from rasterio.transform import from_origin
from read_files import get_annual_pop, read_IMAGEregions_and_TempZone
from scipy.ndimage import distance_transform_edt


wdir = 'X:\\user\\liprandicn\\Health Impacts Model'
year = 2020


### ---------------------------------------------------------------------------------------------------
'''
Functions to run the script
'''

# Create a mask from SSP population data that contains all cells that have a positive population (in any year and ssp)

def get_mask_positive_pop():
    
    '''
    Open all four SSP population data to get a mask of all the possible grid cells where population occurs
    '''
    
    pop_ssp_1 = get_annual_pop('SSP1_M')
    pop_ssp_2 = get_annual_pop('SSP2_CP')
    pop_ssp_3 = get_annual_pop('SSP3_H')
    pop_ssp_5 = get_annual_pop('SSP5_H')
    
    # Create xarray with all the SSPs scenarios
    pop_all_ssp = xr.concat([pop_ssp_1, pop_ssp_2, pop_ssp_3, pop_ssp_5], dim = 'ssp')
    pop_all_ssp['ssp'] = ['ssp1', 'ssp2', 'ssp3', 'ssp5']
    # Mask with positive pop values
    mask_pop_positive = (pop_all_ssp > 0).any(dim=['time', 'ssp'])
    
    return mask_pop_positive, pop_ssp_2


def convert_vector_to_raster(pop_ssp: xr.Dataset, mask_pop_positive: xr.Dataset, vector_data: gpd.GeoDataFrame, column: str, set_zero_to_nan=True):
    
    '''
    This function uses the GBD level3 locations shapefile and converts it into an xarray setting data from a selected column of the shapefile
    '''
    
    # Extract coordinate information
    y_coords = pop_ssp.coords['latitude'].values
    x_coords = pop_ssp.coords['longitude'].values

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
    
    # Interpolate loc_id cells where there are NaN values but population is positive
    interpolated = rasterized_xarray.where(mask_pop_positive).interpolate_na(
        dim=["latitude", "longitude"], method="linear")

    # Combine with original xarray to include other non-NaN values but with loc_id data
    rasterized_xarray = rasterized_xarray.combine_first(interpolated)

    rasterized_xarray = rasterized_xarray.rename({"GPOP": column})
        
    
    return rasterized_xarray


def fill_missing_with_nearest(xarray_dataset, mask_positive_pop):
    filled_xarray = xarray_dataset.copy()

    if len(tuple(xarray_dataset.sizes.values())) == 3:

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



### -------------------------------------------------------------------------
'''
Convert the level 3 GBD shapefile to a raster format (nmupy array)
'''

### Load population xarray from ssp2 and mask
mask_pop_positive, pop_ssp = get_mask_positive_pop()

# Select "Present Day" population data
pop_ssp_year = pop_ssp.sel(time=f'{year}').mean('time')

# Load GBD shapefile with level 3 locations, replace later with level 4 locations
gbd_locations_level3 = gpd.read_file(f'{wdir}\\GBD_Data\\GBD_locations\\Shapefile\\GBD_shapefile.shp')

# Convert geopandas dataframe to xarray using location ID label for the pixel values
gbd_rasterized_xarray = convert_vector_to_raster(pop_ssp, mask_pop_positive, gbd_locations_level3, 'loc_id', set_zero_to_nan=True)

# Interpolate TMREL to areas with NaN but population is positive (maily coasts and islands)
gbd_rasterized_xarray = fill_missing_with_nearest(gbd_rasterized_xarray, mask_pop_positive.GPOP)

# Save file in GBD_Data folder
gbd_rasterized_xarray.to_netcdf(f'{wdir}\\GBD_Data\\GBD_locations\\GBD_locations_level3.nc')

# Convert xarray back to numpy.ndarray for TMREL xarray generation
gbd_rasterized = gbd_rasterized_xarray.loc_id.values



### -------------------------------------------------------------------------
''' Open TMREL files and concat them into a single DataFrame with multi-index
level 1: loc_id, level 2: temperature zone'''

# Get unique location IDs, excluding -1 and NaN and covert to integers
loc_ids = np.unique(gbd_rasterized).astype(int)[1:-1]

# Create an empty list to store filtered DataFrames
df_list = []

# Loop through each location ID level 3
for loc_id in loc_ids:
    file_path = f"{wdir}\\GBD_Data\\Exposure_Response_Functions\\TMRELs\\tmrel_{loc_id}.csv"
    
    # Read the CSV file into a DataFrame with a multi-index
    df = pd.read_csv(file_path, index_col=[0, 1, 2])

    # Filter the DataFrame for the year -----> This can be 1990, 2010, 2020
    df_filtered = df.xs(year, axis=0, level=1)

    # Append the filtered DataFrame to the list
    df_list.append(df_filtered)

# Concatenate all filtered DataFrames into a single DataFrame
gbd_tmrel = pd.concat(df_list, axis=0)



### -------------------------------------------------------------------------
''' Create a new 3D array with the same dimensions as the population data and a 
third dimension for the TMREL draws'''

image_regions, temperature_zone = read_IMAGEregions_and_TempZone()
tmrel_array = np.empty((720, 1440, 100))

for i in range(gbd_rasterized.shape[0]):
    for j in range(gbd_rasterized.shape[1]):
        loc_id = gbd_rasterized[i, j]
        temp_zone = temperature_zone[i, j]
        # Skip if either value is NaN
        if np.isnan(loc_id) or np.isnan(temp_zone):
            tmrel_array[i, j, :] = np.nan
        else:
            # Check if the index exists in the DataFrame
            if (loc_id, temp_zone) in gbd_tmrel.index:
                tmrel_array[i, j, :] = gbd_tmrel.loc[(loc_id, temp_zone)].values
            else:
                tmrel_array[i, j, :] = np.nan  # Handle missing indices
                
# Save the 3D array as a NetCDF file
tmrel_xarray = xr.DataArray(tmrel_array, dims=("latitude", "longitude", "draw"), 
                            coords=dict(latitude=pop_ssp_year.coords['latitude'].values,
                                        longitude=pop_ssp_year.coords['longitude'].values, 
                                        draw=np.arange(1, 101)))
tmrel_xarray.name = "tmrel"

# Interpolate TMREL to areas with NaN but population is positive (maily coasts and islands)
interpolated_tmrel = fill_missing_with_nearest(tmrel_xarray, mask_pop_positive.GPOP)

interpolated_tmrel.to_netcdf(f'{wdir}\\GBD_Data\\Exposure_Response_Functions\\TMRELs_{year}.nc')



### ---------------------------------------------------------------------------------------
'''
Convert GDP and GDPPC to xarray for future analysis and calculations
'''

gbd_locations_level3 = gpd.read_file(f'{wdir}\\GBD_Data\\GBD_locations\\Shapefile\\GBD_shapefile.shp')

# Load files
gdp = pd.read_excel('X:\\user\\hooijschue\\Projects\\GDP_POP\\source_data update (elena)\\GDP\\iamc_db gdp.xlsx')
gdppc = pd.read_excel('X:\\user\\hooijschue\\Projects\\GDP_POP\\source_data update (elena)\\GDP\\iamc_db gdppc.xlsx')

# Select closest year to 2019 and SSP2 data
gdp_year = gdp[gdp['Scenario'] == 'SSP2'][['Region', year]].rename(columns={year:'gdp'})
gdppc_year = gdppc[gdppc['Scenario'] == 'SSP2'][['Region', year]].rename(columns={year:'gdppc'})

# Merge to gbd locations shapefile
gbd_locations_level3_gdp = gbd_locations_level3.merge(gdp_year, right_on='Region', left_on='ihme_lc_id', how='left')
gbd_locations_level3_gdppc = gbd_locations_level3.merge(gdppc_year, right_on='Region', left_on='ihme_lc_id', how='left')

# Convert geopandas dataframe to xarray using location ID label for the pixel values
gbd_rasterized_gdp = convert_vector_to_raster(pop_ssp, mask_pop_positive, gbd_locations_level3_gdp, 'gdp', set_zero_to_nan=True)
gbd_rasterized_gdppc = convert_vector_to_raster(pop_ssp, mask_pop_positive, gbd_locations_level3_gdppc, 'gdppc', set_zero_to_nan=True)

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
gbd_rasterized_countries_gdp = convert_vector_to_raster(pop_ssp, mask_pop_positive, gbd_locations_countries_gdp, 'gdp', set_zero_to_nan=True)
gbd_rasterized_countries_gdppc = convert_vector_to_raster(pop_ssp, mask_pop_positive, gbd_locations_countries_gdppc, 'gdppc', set_zero_to_nan=True)

gbd_rasterized_countries_gdp_gdppc = xr.merge([gbd_rasterized_countries_gdp, gbd_rasterized_countries_gdppc])

gbd_rasterized_countries_gdp_gdppc.to_netcdf(f'{wdir}\\SocioeconomicData\\EconomicData\\GDP_SSP2_{year}_OriginalCountries.nc')