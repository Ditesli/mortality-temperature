'''
This script contains he functions necessary to calculate the daily relative risk for all grid
cells in a year.
'''

import pandas as pd
import numba as nb
import numpy as np


@nb.njit
def temperature_to_location(t, min_temp, max_temp, is_day):
    
    '''METHOD 2
    Retrieve the location (row and column index) of a given daily temperature
    (if is_day=True) or temperature zone (if is_day=False)'''
    
    # Return -1 for NaN cells (sea)
    if np.isnan(t): return -1

    # If temperature is outside limits, take first/last column or row
    if t < min_temp: return 0
    if t > max_temp: t = max_temp

    # Calculate location index according to temperature range and decimals 
    return (np.round(t, 1) - min_temp) * 10 if is_day else np.round(t) - min_temp


@nb.njit
def get_daily_rr(temperature_zone_data, daily_temperature_data, erf_data):
    
    '''Method 2
    Calculate daily relative risk for all grid cells using tmmperature_to_location
    function and temperature_zone map and daily temperature data'''
    
    # Get size of the temperature files
    rows, cols = temperature_zone_data.shape
    # array of daily relative risks
    rr_data_day = np.full((rows, cols), np.nan, dtype=np.float64) 

    # Iterate over the latitude and longitude of the temperature arrays
    for i in nb.prange(rows*cols):
        lat = i // cols
        lon = i % cols
        
        # Get the temperature zone and daily temperature per grid cell
        daily_temp = daily_temperature_data[lat, lon]
        zone_temp = temperature_zone_data[lat, lon]

        # Discard points with NaN
        if np.isfinite(daily_temp) or np.isfinite(zone_temp): 
            row_idx = temperature_to_location(daily_temp, -25.0, 35.0, is_day=True)  
            col_idx = temperature_to_location(zone_temp, 6.0, 28.0, is_day=False)
            
            # Retrieve corresponding RR value from erf data
            if row_idx >= 0 and col_idx >= 0:  # Avoid invalid indices
                rr_data_day[lat, lon] = erf_data[int(row_idx), int(col_idx)] 

    return rr_data_day


@nb.njit
def get_yearly_rr(temperature_zone_data, erf, daily_temperature_data):
    
    '''METHOD 2
    Generate yearly rr using get_daily_rr and putting the result in one 3d-array'''
    
    # Get data shape
    rows, cols = daily_temperature_data.shape[0], daily_temperature_data.shape[1]
    num_days = daily_temperature_data.shape[2]
    # Start with NaN arrayy
    rr_year = np.full((num_days, rows, cols), np.nan, dtype=np.float64)

    # Get relative risk for every day of the year
    for day in nb.prange(num_days):
        rr_year[day] = get_daily_rr(temperature_zone_data, daily_temperature_data[:,:,day], erf)
    
    return rr_year


#@nb.njit
def get_spatial_mean(rr_temporal_mean, pop_ssp_year, greg, region):
    
    '''Method 2
    Caclulate the population weighted mean of the relative risk values for a given IMAGE region'''
    # Get the mask for the specific region
    rr_mask = rr_temporal_mean[greg == region]
    pop_mask = pop_ssp_year[greg == region]
    # Calculate the population weighted mean
    weighted_mean = np.nansum(rr_mask * pop_mask) / np.nansum(pop_mask)
    
    return weighted_mean


### ------------------------------------------------------------------------------------------
### METHOD 1


#@nb.njit
def get_array_from_mask(data, valid_mask, num_days):
    
    ''' Method 1
    Converts GREG, yearly population and temperature zone xarrays into 1-D numpy arrays 
    by keeping only the entries where there is population data (data is more than 0)
    '''
    # Convert xarray to numpy and get the values for the valid mask
    data_masked = data[valid_mask]
    # Repeat the same values for the number of days in a year
    data_array = np.concatenate([data_masked] * num_days)
    
    return data_array


def get_temp_array_from_mask(daily_temp, valid_mask, pop_array, num_days):
    
    '''METHOD 1
    Creates a 1-D array from the daily temperature data by masking first cells with
    population and the flattening'''
    
    # Create an empty array to store the daily temperatures
    dayTemp_array = np.empty(len(pop_array), dtype=np.float32) 
    index = 0
    # Iterate over the number of days in the year
    for day in range(num_days):
        # Get the daily temperature
        dayTemp_np = daily_temp[:,:,day]
        # Mask the values to get only the ones with POP data
        dayTemp_values = dayTemp_np[valid_mask]
        # Append the values to the array
        dayTemp_array[index:index+len(dayTemp_values)] = dayTemp_values
        index += len(dayTemp_values) # or len(pop_array)
        
    return dayTemp_array


def get_all_data_masked(pop_ssp, year, num_days, greg, era5_tz, daily_temp):
    
    '''METHOD 1
    Creates a mask for the population data and then creates 1-D arrays for the 
    population, temperature zone and daily temperature data'''
    
    # Select the specific year from the population data
    pop_ssp_year = pop_ssp.sel(time=f'{year}').mean('time').GPOP
    # Set a mask of pixels with population data
    valid_mask = pop_ssp_year.values > 0.
    # Get arrays for the data using the functions defined above
    greg_array = get_array_from_mask(greg, valid_mask, num_days)
    pop_array = get_array_from_mask(pop_ssp_year.values, valid_mask, num_days)
    meanTemp_array = get_array_from_mask(era5_tz, valid_mask, num_days)
    dayTemp_array = get_temp_array_from_mask(daily_temp, valid_mask, pop_array, num_days)
    
    # print(f'{year} - Data masked')
    
    return greg_array, pop_array, meanTemp_array, dayTemp_array                 


def get_pop_weighted_factors(greg_array, pop_array, meanTemp_array, dayTemp_array):
    
    '''METHOD 1
    Calculates the population weighted factors for the relative risk calculation
    (slow function as it uses pandas!!!)'''
    
    # Convert arrays to a single dataframe
    df = pd.DataFrame({'dailyTemp': dayTemp_array, 'greg': greg_array, 'pop': pop_array, 'meanTemp': meanTemp_array})
    # For a year and tz, sum all the people that was exposed to a given daily temperature
    df = df.groupby(['dailyTemp', 'greg', 'meanTemp'], as_index=False).sum()
    # Sum all the populations for a given IMAGE region and Normalize the population for each region
    df['pop'] /= df.groupby('greg')['pop'].transform('sum')
    
    print(f'Population weighted factors calculated')
    
    return df

def get_paf_to_rr(df, min_dict, max_dict, erf, diseases):
    
    '''METHOD 1'''
    # Truncate min and max Temperature values according to availability in ERF
    df['dailyTemp'] = df['dailyTemp'].clip(lower=df['meanTemp'].map(min_dict), upper=df['meanTemp'].map(max_dict))
    # Merge the ERF with the grouped data to assign rr
    df = pd.merge(df, erf, on=['meanTemp', 'dailyTemp'], how='left')
    # Calculate the PAF for each disease
    df[[f'{col}' for col in diseases]] = np.where(df[diseases] < 1, df['pop'].values[:, None] * (df[diseases] - 1), df['pop'].values[:, None] * (1 - 1 / df[diseases]))
    # Sum all PAFs for each disease
    df = df.groupby('greg').sum()
    # Calculate RR
    df[[f'{col}' for col in diseases]] = 1 / (1 - df[[f'{col}' for col in diseases]])
    # Append to final dataframe
    df = df.iloc[:,3:]
    
    return df
    


def get_yearly_pop_weighted_values(pop_ssp_year, image_regions, num_days):  
    
    """
    Calculate population-weighted values for a given year according to the image regions.

     Parameters:
    - pop_ssp_year: Population data for the specified year.
    - image_regions: Array with indices as IMAGE regions.

    Returns:
    - pop_weighted_ssp_year: Population-weighted values for the specified year.

    """

    # Handle NaN by assigning them index 0 and convert to integers
    image_regions_nan_mask = np.isnan(image_regions)
    image_regions[image_regions_nan_mask] = 0
    image_regions = image_regions.astype(int)

    # Define the range of image regions (1 to 26) and include 0 for NaNs
    unique_image_regions = np.arange(0, 27)
    
    # Handle NaN in population array by setting 0 value
    pop_ssp_year[np.isnan(pop_ssp_year)] = 0
    
    # Sum populations for each region, including index 0 for NaNs
    total_population_per_region = np.bincount(image_regions.flat, weights=pop_ssp_year.flat, minlength=len(unique_image_regions))
    
    # Create a mask to avoid division by zero
    valid_mask = total_population_per_region[image_regions] != 0
    
    # Create empty arrays for the results
    pop_weighted_ssp_year = np.zeros_like(pop_ssp_year, dtype=float)
    
    # Calculate the population-weighted values
    pop_weighted_ssp_year[valid_mask] = pop_ssp_year[valid_mask] / (total_population_per_region[image_regions][valid_mask] * num_days)
    
    return pop_weighted_ssp_year


def tz_tmrel_combinations(pop_ssp, tmrel, temperature_zones):
    
    '''
    M1-1
    This will produce a dataframe with the unique combinations of temperature zones and TMREL
    '''
    
    # Get any cell with population data for the selected scenario
    mask_pop = (pop_ssp.GPOP > 0).any(dim='time')
    
    # Mask TMREL and temperature_zones arrays
    tmrel_valid_pop = tmrel[mask_pop.values]
    tz_valid_pop = temperature_zones[mask_pop.values]
    
    # Create dataframe with these data
    df_tz_tmrel = pd.DataFrame({'temperature_zone': tz_valid_pop, 'tmrel': np.round(tmrel_valid_pop,1)})
    
    # Remove duplicated rows
    df_tz_tmrel = df_tz_tmrel.drop_duplicates()
    
    return df_tz_tmrel



def shift_rr(df_erf, df_tz_tmrel, diseases):
    
    '''
    For every temperature zone, the merging assings all the possible TMREL. 
    This implies that we will have repeated rows for the daily temperature and relative risks 
    '''

    # Merge df_tz_tmrel with ERF data
    df_erf_tmrel = pd.merge(df_erf, df_tz_tmrel, on=['temperature_zone'], how='left') 
    
    # Set temperature_zone as index
    df_erf_tmrel = df_erf_tmrel.set_index('temperature_zone')
    
    # For each temperature zone, divide the RR by the TMREL
    # df_erf_tmrel = df_erf_tmrel.groupby('temperature_zone', group_keys=False).apply(divide_by_tmrel)
    df_erf_tmrel = df_erf_tmrel.groupby('temperature_zone', group_keys=False).apply(
        lambda group: divide_by_tmrel(group, diseases)
    )

    # Reset index
    df_erf_tmrel = df_erf_tmrel.reset_index()

    return df_erf_tmrel



def divide_by_tmrel(group, diseases):
    
    '''
    This function works per temperature zone groups. It locates the row whose daily temperature equals 
    the TMREL and substracts this row for the rest of them
    '''    
    
    # Locate the rows whose daily temperature equals the TMREL
    fila_tmrel = group.loc[group['daily_temperature'] == group['tmrel'].iloc[0]]
    
    # Select first row
    reference = fila_tmrel.iloc[0][diseases]
    
    # Divide to shift the RR vertically
    group[diseases] = group[diseases] / reference
    
    return group



def get_data_masked_per_region(valid_mask, num_days, pop, era5_tz, daily_temp, tmrel): 
    
    '''
    Use the mask for the population data to mask the population temperature zone, tmrel map and 
    daily temperature data. The first three maps are repreated 365/366 times depending the 
    number of days in the specific year. This process creates 1-D arrays for the data representing
    the different combinations.
    '''
    
    # Get arrays for the data using the functions defined above
    pop_array = get_array_from_mask(pop, valid_mask, num_days)
    meanTemp_array = get_array_from_mask(era5_tz, valid_mask, num_days)
    dayTemp_array = get_temp_array_from_mask(daily_temp, valid_mask, pop_array, num_days)
    tmrel_array = get_array_from_mask(tmrel, valid_mask, num_days)
    
    # print(f'Data masked')
    
    return pop_array, meanTemp_array, dayTemp_array, tmrel_array



def create_population_df(mask, pop_ssp_year, temperature_zones, daily_temp, tmrel, num_days, min_dict, max_dict):
    
    population_array, temperature_zones_array, daily_temperatures_array, tmrel_array = get_data_masked_per_region(mask, num_days, pop_ssp_year, 
                                                                                                              temperature_zones, daily_temp, tmrel)
    
    #Change array type for posterior merging
    daily_temperatures_array = np.array(daily_temperatures_array, dtype=np.float64)
    
    # Create a dataframe that includes data on temperature zone, daily temperature, population and tmrel
    df_pop = pd.DataFrame({'temperature_zone': temperature_zones_array, 'daily_temperature': np.round(daily_temperatures_array,1),
                           'population': np.round(population_array,1), 'tmrel':np.round(tmrel_array,1)})

    # Truncate min and max Temperature values according to availability in ERF
    df_pop['daily_temperature'] = df_pop['daily_temperature'].clip(lower=df_pop['temperature_zone'].map(min_dict), 
                                                                   upper=df_pop['temperature_zone'].map(max_dict))
    
    # Group per temperature_zone and daily_temperature, calculating the fraction of population per each combination
    df_pop = df_pop.groupby(['temperature_zone', 'daily_temperature', 'tmrel'], as_index=False).sum()
    df_pop['population'] /= df_pop['population'].sum()
    
    return df_pop



def rr_paf_rr(df, rr_year, diseases, year, region, temp_type):
    
    # Convert the RR to PAF
    df[[f'{col}' for col in diseases]] = np.where(df[diseases] < 1, 
                                                  df['population'].values[:, None] * (df[diseases] - 1),
                                                  df['population'].values[:, None] * (1 - 1 / df[diseases]))
    
    # Aggregate PAFs
    df_aggregated = df.sum(axis=0)
    
    # Convert aggregated PAF to RR and locate in annual RR dataframe
    rr_year.loc[region, (year, diseases, temp_type)] = [1/(1- df_aggregated[f'{d}']) for d in diseases]