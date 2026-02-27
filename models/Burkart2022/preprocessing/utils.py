import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import utils_common.population as pop
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt



def GenerateTemperatureZones(wdir, era5_path):
    
    """
    Generate files with the temeprature zones as defined by Burkart et al., 2022.
    The code reads in ERA5 monthly temperature data from 1980-2019, aligns the 
    grid to the population grid and calculates the mean. 
    A mask with population data and sea-land data are loaded to ensure that all
    sea pixels are NaN and all pixels with population have assigned a temperature
    zone. Temperature is also interpolated to match population grid,
    """
    
    print("Calculating temperature zones...")
    
    print("[1] Loading population data to create mask...")

    # Open and preprocess population data and set mask for ALL cells with population
    mask_pop = pop.get_all_population_data(wdir, return_pop=False)    
    
    print("[2] Loading ERA5 historical temperature data and computing climatologies...")

    era5_t2m = (
        xr.open_dataset(era5_path + "/era5_t2m_mean_1980-2019.nc")
        .drop_vars("number") # Discard dummy var "number"
        .mean(dim="valid_time")  # Calculate 1980-2019 mean
        .assign_coords(longitude=lambda x: ((x.longitude + 180) % 360 - 180))
        .sortby("longitude") # Shift coordinates
        .interp(
            longitude=mask_pop.longitude, # Align latitudes with mask_pop
            latitude=mask_pop.latitude, 
            method="nearest"
            )
        - 273.15 # Convert to Celius
    )
    
    print("[3] Loading ERA5 land sea-mask...")

    # Open ERA5 land-sea mask, discard dummy var "time" and rename variable to "t2m" for easier handling
    era5_land_sea = (
        xr.open_dataset(era5_path+"/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc") 
        .drop_vars("time")
        .rename({"lsm": "t2m"})
        .assign_coords(longitude=lambda x: ((x.longitude + 180) % 360 - 180))
        .sortby("longitude") # Shift coordinates
        .interp(
            longitude=mask_pop.longitude, # Align latitudes with mask_pop
            latitude=mask_pop.latitude, 
            method="nearest"
            )
    )
    
    print("[4] Applying land-sea and population mask and clipping temperatures to tz...")
    
    output_dir = wdir + "/data/temperature_zones"
    os.makedirs(output_dir, exist_ok=True)
    
    (
        era5_t2m
        .where(
            ((mask_pop.GPOP) | (era5_land_sea.t2m > 0)) & # Keep only land or cells with POP
            (era5_t2m.latitude >= -60) # Exclude Antarctica to save memory
        )
        .where(lambda x: x.isnull(), lambda x: x.round()) # Round data to integers
        .pipe(lambda x: np.clip(x.t2m, 6, 28)) # Clip values between 6 and 28
        .mean(dim="time")
        .to_netcdf(output_dir+"/ERA5_mean_1980-2019_land_t2m_tz.nc") # Save
    )
    
    print("[5] Temperature zones calculated and saved.")

    
    
def GenerateRasterGBDLocations(wdir):
    
    """
    Convert the level 3 GBD shapefile to a raster format (numpy array)
    """

    print("[1] Loading population data to create mask...")
    
    # Load population xarray and mask
    mask_pop = pop.get_all_population_data(wdir, return_pop=False)
    
    print("[2] Loading GBD location shapefile and rasterize it...")

    # Load GBD shapefile with level 3 locations
    gbd_locations_level3 = gpd.read_file(wdir+"/data/gbd_locations/gbd_shapefiles/GBD_shapefile.shp")

    # Convert geopandas dataframe to xarray using location ID label for the pixel values
    gbd_rasterized_xarray = ConvertVectorizedToRaster(mask_pop, gbd_locations_level3, "loc_id", set_zero_to_nan=True)

    print("[3] Filling in missing pixels but with poulation..")

    # Interpolate to areas with NaN but with population (maily coasts and islands)
    gbd_rasterized_xarray = FillMissingWithNearest(gbd_rasterized_xarray, mask_pop.GPOP)

    # Save file in GBD_Data folder
    output_dir = wdir + "/data/gbd_locations"
    os.makedirs(output_dir, exist_ok=True)
    gbd_rasterized_xarray.to_netcdf(output_dir+"/GBD_locations_level3.nc")
    
    print("[4] Raster file generated and saved.")

    

def ConvertVectorizedToRaster(mask_pop: xr.Dataset, vector_data: gpd.GeoDataFrame, 
                             column: str, set_zero_to_nan=True):
    
    """
    This function uses the GBD level3 locations shapefile and converts it into 
    an xarray setting data from a selected column of the shapefile
    """
    
    # Extract coordinate information
    y_coords = mask_pop.coords["latitude"].values
    x_coords = mask_pop.coords["longitude"].values

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
    rasterized_xarray = xr.DataArray(rasterized_data, coords=[y_coords, x_coords], dims=["latitude", "longitude"])
    rasterized_xarray.name = column

    # Combine with original xarray to include other non-NaN values but with loc_id data
    rasterized_xarray = xr.Dataset({column:rasterized_xarray})
        
    return rasterized_xarray



def FillMissingWithNearest(xarray_dataset, mask_positive_pop):
    
    """
    Fill missing values (NaN) in an xarray dataset using the nearest valid
    value, but only in locations where the population mask is positive.
    """
    
    filled_xarray = xarray_dataset.copy()
    
    
    def fill_nearest_2d(xarray_draw):
        # Get missing pixerls mask
        missing_mask = np.isnan(xarray_draw) & mask_positive_pop.values
        # Get coordinates of the closest valid value
        _,indices = distance_transform_edt(np.isnan(xarray_draw), return_indices=True)
        # Fill NaN
        xarray_draw[missing_mask] = xarray_draw[indices[0][missing_mask], indices[1][missing_mask]]
        return xarray_draw
    
    
     # Single 2D array (GBD locations)   
    if len(tuple(xarray_dataset.sizes.values())) == 2:
        # Get 2d-values
        xarray_draw = xarray_dataset.loc_id.values
        # Fill missing and store in the new xarray
        filled_xarray["loc_id"][:, :] = fill_nearest_2d(xarray_draw)

    
    # 3D array with multiple draws (TMRELs)
    if len(tuple(xarray_dataset.sizes.values())) == 3:
        # Iterate over draws
        for draw in range(xarray_dataset.shape[2]):
            # Get 2d-values
            xarray_draw = xarray_dataset[:, :, draw].values
            # Fill missing and store in the new xarray
            filled_xarray[:, :, draw] = fill_nearest_2d(xarray_draw)
            
    
    return filled_xarray



def ImportTMRELsFiles(wdir, year):
    
    """ 
    Open TMREL files and concat them into a single DataFrame with multi-index
    level 1: loc_id, level 2: temperature zone.
    The code needs to import the raster file with the GBD locations first to 
    get the unique loc_ids.
    """
    
    # Open rasterized GBD level 3 locations 
    gbd_rasterized = (
        xr.open_dataset(f"{wdir}/data/gbd_locations/GBD_locations_level3.nc")
        .loc_id.values # Convert to numpy array
    )

    # Unique location IDs, excluding -1 and NaN, as integers
    loc_ids = np.unique(gbd_rasterized)[1:-1].astype(int)
    
    # Loop through each location ID level 3
    df_list = []
    for loc_id in loc_ids:

        tmrel = (
            pd.read_csv(wdir+f"/data/burkart_sm/TMRELs/tmrel_{loc_id}.csv",
                        index_col=[0, 1, 2])
            .xs(year, axis=0, level=1) # Filter selected year ---> 1990, 2010, 2020
        )

        # Append the filtered DataFrame to the list
        df_list.append(tmrel)

    # Concatenate all filtered DataFrames into a single DataFrame
    gbd_tmrel = pd.concat(df_list, axis=0)
    
    return gbd_rasterized, gbd_tmrel



def GenerateTMRELsRasters(wdir):

    """ 
    Create a 3D raster with the same dimensions as the population data and a 
    third dimension for the TMREL draws.
    """
    
    print("Generating TMRELs rasters...")
    
    print("[1] Importing GBD locations, TMRELs and temperature zones...")
    
    # Import population mask
    mask_pop = pop.get_all_population_data(wdir, return_pop=False)
    
    # Import Temperature Zones
    temperature_zones = (
        xr.open_dataset(wdir+"/data/temperature_zones/ERA5_mean_1980-2019_land_t2m_tz.nc")
        .t2m.values # Convert to numpy array
    )
    
    for year in [1990, 2010, 2020]:
        
        print(f"[2] TMRELs for year {year}...")
    
        # Import TMRELs
        gbd_rasterized, gbd_tmrel = ImportTMRELsFiles(wdir, year)    
        
        print("[2.1] Assigning each GBD location and temeprature zone its corresponding TMREL...")
        
        # Create empty 3D array for TMRELs with predefined resolution
        tmrel_array = np.empty((720, 1440, 100))

        # Scan every grid cell
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
                        
        print(f"[2.2] Converting {year} TMRELs to xarray and saving them as nc file...")
                        
        # Convert to xarray DataArray
        tmrel_xarray = xr.DataArray(
            tmrel_array, 
            dims=("latitude", "longitude", "draw"), 
            coords=dict(
                latitude=mask_pop.latitude.values,
                longitude=mask_pop.longitude.values, 
                draw=np.arange(1, 101)
            ),
        name= "tmrel"
        )

        # Interpolate TMREL to areas with NaN but population is positive (maily coasts and islands)
        interpolated_tmrel = FillMissingWithNearest(tmrel_xarray, mask_pop.GPOP)

        # Save file in GBD_Data folder
        interpolated_tmrel.to_netcdf(wdir+f"/data/TMRELs_nc/TMRELs_{year}.nc")
        
    print("TMRELs generated and saved.")