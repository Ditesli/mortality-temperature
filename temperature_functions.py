from shapely.geometry import Point, mapping
import pandas as pd
import geopandas as gpd ### 
import xarray as xr
# import rioxarray
import numpy as np
from scipy.stats import skew, kurtosis
from affine import Affine
from rasterio.features import rasterize



### -----------------------------------------------------------------------------------------------------
### Define functions

def get_country_mask(pop_ssp, gdf_countries):
    
    '''
    Generate country mask based on population data resolution (from LPJml) and country geometries.
    '''
    
    pop_ssp_year = pop_ssp.isel(time=0)
    
    lats = pop_ssp_year["latitude"].values
    lons = pop_ssp_year["longitude"].values

    # Determine resolution from coordinates
    res_lat = np.abs(lats[1] - lats[0])
    res_lon = np.abs(lons[1] - lons[0])

    # Use first value as reference for the transformation
    transform = Affine.translation(lons[0] - res_lon / 2, lats[0] - res_lat / 2) * Affine.scale(res_lon, -res_lat)

    # Raster shape
    shape = (len(lats), len(lons))
    
    # Value to rasterize
    shapes = ((geom, idx) for geom, idx in zip(gdf_countries.geometry, gdf_countries['loc_id']))

    country_mask_np = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=-1,  # Value for the ocean an unclassified zones
        dtype="int16", 
        all_touched=False #  Only the center of the pixel is considered (avoids overlaping geometries)
    )
    
    country_mask = xr.DataArray(
        data=country_mask_np,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        name="loc_id"
        )
    
    return country_mask



def compute_country_statistics(climate_var, pop_ssp, country_mask, year, country_statistics, final_name):
    
    '''
    Compute the spatially aggregated population-weighted mean by country for a given year.
    '''
    
    # Select population data of the corresponding year
    population = pop_ssp.sel(time=f'{year}').mean(dim='time').GPOP
    
    # Stack dimensions to allign with country mask
    climate_var_stacked = climate_var.stack(stacked=('latitude', 'longitude'))
    country_mask_stacked = country_mask.stack(stacked=('latitude', 'longitude'))
    population_stacked = population.stack(stacked=('latitude', 'longitude'))

    # Create xarray with pop*climate_var
    weighted_var = climate_var_stacked * population_stacked

    # Group per country and compute the weighted sum and population sum
    sum_weighted_temp = weighted_var.groupby(country_mask_stacked).sum(dim='stacked')
    sum_population = population_stacked.groupby(country_mask_stacked).sum(dim='stacked')

    # Calculate the population-weighted mean by country
    weighted_mean_temp_by_country = sum_weighted_temp / sum_population

    # Convert to dataframe
    df_weighted = weighted_mean_temp_by_country.to_dataframe(name=final_name).reset_index()
    
    country_statistics = country_statistics.merge(df_weighted, on='loc_id')

    return(country_statistics)



def compute_temperature_statistics(climate_var, pop_ssp, country_mask, year, country_statistics, climate_variable,
                                time_name, mean=False, std=False, skewness=False, kurtosis_=False, 
                                image_degree_days=False, base_degree_days=False, bins=False):
    
    if mean:
        # Calculate mean temperature
        climate_var_processed = climate_var.mean(dim=time_name)
        country_statistics = compute_country_statistics(climate_var_processed, pop_ssp, country_mask, year, country_statistics, f'{year}_{climate_variable}_mean')
        climate_var_processed = climate_var_processed**2
        country_statistics = compute_country_statistics(climate_var_processed, pop_ssp, country_mask, year, country_statistics, f'{year}_{climate_variable}_mean_sq')

    if std:
        # Calculate standard deviation of temperature
        climate_var_processed = climate_var.std(dim=time_name)
        country_statistics = compute_country_statistics(climate_var_processed, pop_ssp, country_mask, year, country_statistics, f'{year}_{climate_variable}_std')

    if skewness:
        # Calculate skewness of temperature
        climate_var_array = skew(climate_var.values, axis=0, nan_policy='omit')
        climate_var_processed = xr.DataArray(climate_var_array, dims=climate_var.dims, coords=climate_var.coords)
        country_statistics = compute_country_statistics(climate_var_processed, pop_ssp, country_mask, year, country_statistics, f'{year}_{climate_variable}_skewness')
    
    if kurtosis_:
        # Calculate kurtosis of temperature
        climate_var_array = kurtosis(climate_var.values, axis=0, nan_policy='omit')
        climate_var_processed = xr.DataArray(climate_var_array, dims=climate_var.dims, coords=climate_var.coords)
        country_statistics = compute_country_statistics(climate_var_processed, pop_ssp, country_mask, year, country_statistics, f'{year}_{climate_variable}_kurtosis')
        
    if base_degree_days:
        climate_var_processed = (climate_var.where(climate_var > 20) - 20).sum(dim=time_name)
        country_statistics = compute_country_statistics(climate_var_processed, pop_ssp, country_mask, year, country_statistics, f'{year}_CDD_20')
        
        climate_var_processed = np.abs(climate_var.where(climate_var < 20) - 20).sum(dim=time_name)
        country_statistics = compute_country_statistics(climate_var_processed, pop_ssp, country_mask, year, country_statistics, f'{year}_HDD_20')
        
        climate_var_processed = ((climate_var.where(climate_var > 20) - 20)**2).sum(dim=time_name)
        country_statistics = compute_country_statistics(climate_var_processed, pop_ssp, country_mask, year, country_statistics, f'{year}_CDD_20_sq')
        
        climate_var_processed = ((climate_var.where(climate_var < 20) - 20) ** 2).sum(dim=time_name)
        country_statistics = compute_country_statistics(climate_var_processed, pop_ssp, country_mask, year, country_statistics, f'{year}_HDD_20_sq')
        
    if image_degree_days:    
        # Compute degree days
        climate_var_processed = (climate_var.where(climate_var > 20) - 20).t2m.sum(dim=time_name) + np.abs(climate_var.where(climate_var < 20) - 20).sum(dim=time_name)
        country_statistics = compute_country_statistics(climate_var_processed, pop_ssp, country_mask, year, country_statistics, f'{year}_degree_days')
        
        # Compute Cooling Degree Days (CDD) and Heating Degree Days (HDD) for different setpoints
        setpoints_cdd = [23.3, 18.,  20., 22., 24., 26., 25.] # First, default value. Last, found in literature
        setpoints_hdd = [18.3, 18.,  20., 22., 24., 26., 15.] # First, default value. Last, found in literature

        names_cdd = ['CDD_23_3', 'CDD_18', 'CDD_22', 'CDD_24', 'CDD_26', 'CDD_25'] 
        names_hdd = ['HDD_18_3', 'HDD_18', 'HDD_22', 'HDD_24', 'HDD_26', 'HDD_15']

        for setpoint, name in zip(setpoints_cdd, names_cdd):
            climate_var_processed = (climate_var.where(climate_var > setpoint) - setpoint).sum(dim=time_name)
            country_statistics = compute_country_statistics(climate_var_processed, pop_ssp, country_mask, year, country_statistics, f'{year}_{name}')
            
        for setpoint, name in zip(setpoints_hdd, names_hdd):
            climate_var_processed = np.abs(climate_var.where(climate_var < setpoint) - setpoint).sum(dim=time_name)
            country_statistics = compute_country_statistics(climate_var_processed, pop_ssp, country_mask, year, country_statistics, f'{year}_{name}')
        
    if bins:
        initial_condition = climate_var <= -20
        climate_var_processed  = initial_condition.sum(dim=time_name)
        country_statistics = compute_country_statistics(climate_var_processed, pop_ssp, country_mask, year, country_statistics, f'{year}_bin_-20')

        for t0 in [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]:
            condition = (t0 < climate_var) & (climate_var <= (t0 + 5))
            climate_var_processed  = condition.sum(dim=time_name)
            country_statistics = compute_country_statistics(climate_var_processed, pop_ssp, country_mask, year, country_statistics, f'{year}_bin_{t0}_{t0+5}')
            
        final_condition = climate_var > 40
        climate_var_processed = final_condition.sum(dim=time_name)
        country_statistics = compute_country_statistics(climate_var_processed, pop_ssp, country_mask, year, country_statistics, f'{year}_bin_40')
    
    return country_statistics
    


def compute_rh_statistics(calc_type, daily_temp_era5, pop_ssp, country_mask, year, country_statistics):
    
    '''
    Calculate annual mean relative humidity and relative humidity times CDD_25
    '''
    
    # Load temperature files
    max_temperature = get_data_era5(calc_type, pop_ssp, year, daily_temp_era5, 't2m', 'max')
    dewpoint_temperature = get_data_era5(calc_type, pop_ssp, year, daily_temp_era5, 'd2m', 'mean')
    mean_temperature = get_data_era5(calc_type, pop_ssp, year, daily_temp_era5, 't2m', 'mean')

    # Magnus approximation for relative humidity
    relative_humidity = 100 * (
        np.exp((17.625 * dewpoint_temperature) / (243.04 + dewpoint_temperature)) / 
        np.exp(17.625 * max_temperature / (243.04 + max_temperature))
        )
    
    mean_relative_humidity = relative_humidity.mean(dim='valid_time')
    country_statistics = compute_country_statistics(mean_relative_humidity, pop_ssp, country_mask, year, country_statistics, f'{year}_mean_relative_humidity')
    
    del dewpoint_temperature, max_temperature, mean_relative_humidity
    
    # Calculate CDD (Cooling Degree Days) for relative humidity
               
    CDD_offset = 25
    cdd_temperature = (mean_temperature.where(mean_temperature > CDD_offset) - CDD_offset)
    temperature = cdd_temperature * relative_humidity
    temperature = temperature.sum(dim='valid_time')
    
    del mean_temperature, cdd_temperature, relative_humidity
    
    # Compute country statistics for CDD with relative humidity
    country_statistics = compute_country_statistics(temperature, pop_ssp, country_mask, year, country_statistics, f'{year}_CDD_25_rh')
    
    return country_statistics



def melt_columns(country_statistics):
    
    '''
    Melt dataframe from wide to long
    '''
    
    # Melt columns
    df_melted = country_statistics.melt(
        id_vars=['loc_id', 'loc_name'],
        var_name='year_variable',
        value_name='value'
    )

    # Separate 'year_variable' into 'year' and 'variable'
    df_melted[['year', 'variable']] = df_melted['year_variable'].str.extract(r'(\d{4})_(.*)')

    # Pivot to have columns per variable and not per year
    clim_long = df_melted.pivot_table(
        index=['loc_id', 'loc_name', 'year'],
        columns='variable',
        values='value'
    ).reset_index()

    # if function_type == 'mean':
    #     # Add third columnd with mean temperature in case pop weighted temperature is NaN
    #     clim_long[f'all_temp'] = clim_long[f'{function_type}_temp'].where(clim_long[f'pop_weighted_{function_type}_temp'].isna(), clim_long[f'pop_weighted_{function_type}_temp'])
    
    # Rename loc_id --> location_id
    clim_long = clim_long.rename(columns={'loc_id':'location_id', 'loc_name':'location_name'})
    
    # Sort values by year
    clim_long = clim_long.sort_values('year')
    
    return clim_long



def get_data_era5(calc_type, pop_ssp, year, daily_temp_era5, variable, variable_type):
        
    if calc_type == 'clim':
        # Open ERA5 yearly data
        temperature = xr.open_dataset('X:\\user\\liprandicn\\Data\\ERA5\\ERA5_t2m_yearly_1940-2024.nc')
        temperature -= 273.15  # Convert to Celsius
        # Select temperature dataframe
        temperature = temperature.sel(valid_time=slice(f'{year-29}', f'{year}')).mean('valid_time').t2m
        # Spatial interplation to match population resolution
        temperature = temperature.interp(latitude=pop_ssp.latitude, longitude=pop_ssp.longitude, method="nearest")

    if calc_type == 'stats':
        # Open era5 daily data
        temperature, _ = daily_temp_era5(year, pop_ssp, variable, variable_type, to_array=False, interp=True)
        temperature = temperature[f'{variable}']
        
    return temperature



def get_data_isimip(calc_type, pop_ssp, model, scenario, year):
    
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        num_days = 366
    else:
        num_days = 365

    temperature = xr.open_dataset(f'X:\\user\\krugerc\\data\\climate_data\\{model}\\{scenario}_tas_1970_2100.nc')
    temperature = temperature.sortby("time") 
    temperature = temperature.rename({'lon': 'longitude', 'lat': 'latitude'})
    
    if calc_type == 'clim':
        # Select 30 year window
        temperature = temperature.sel(time=slice(f"{year-29}-01-01", f"{year}-12-31"))
        temperature = temperature.groupby('time.year').mean(dim='time')
        temperature = temperature.mean(dim='year')  .tas  
        
    if calc_type == 'stats':
        temperature = temperature.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).tas
        
    temperature -= 273.15 
    temperature = temperature.interp(longitude=pop_ssp.longitude, latitude=pop_ssp.latitude, method="nearest")
    
    return temperature, num_days



def melt_and_save(country_statistics, calc_type, data_type, climate_variable, wdir, years, model=None, scenario=None):
    
    # Melt columns from wide to long format
    temp_long = melt_columns(country_statistics)
    
    # Save
    if calc_type=='clim':
        if model is not None and scenario is not None:
            temp_long.to_csv(f'{wdir}\\Climate_Data\\ERA5\\country_data\\{data_type}_{model}_{scenario}_climatology_{years[0]}-{years[-1]}.csv')
        else:
            temp_long.to_csv(f'{wdir}\\Climate_Data\\ERA5\\country_data\\{data_type}_climatology_{years[0]}-{years[-1]}.csv')

    if calc_type=='stats':
        if model is not None and scenario is not None:
            temp_long.to_csv(f'{wdir}\\Climate_Data\\ERA5\\country_data\\{data_type}_{climate_variable}_{model}_{scenario}_statistics_{years[0]}-{years[-1]}.csv', index=False)
        else:
            temp_long.to_csv(f'{wdir}\\Climate_Data\\ERA5\\country_data\\{data_type}_{climate_variable}_statistics_{years[0]}-{years[-1]}.csv', index=False)
        
    print(f'Saved statistics data to csv file')