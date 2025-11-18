import numpy as np
import pandas as pd
import xarray as xr



def load_temperature_type(temp_dir, temp_source, temp_type, year, pop_ssp):
    
    '''
    Select the temperature data type to use (ERA5 or monthly statistics)
    '''
    
    if temp_source == 'ERA5':
        daily_temp, num_days = daily_temp_era5(temp_dir, year, temp_type, pop_ssp, to_array=True)
        
    elif temp_source == 'MS':
        daily_temp, num_days = daily_from_monthly_temp(temp_dir, year, temp_type.upper())
        
    return daily_temp, num_days



def daily_temp_era5(era5_dir, year, type, pop_ssp=None, to_array=False):
    
    '''
    Read daily ERA5 temperature data for a specific year, shift longitude coordinates,
    convert to Celsius, and match grid with population data.
    Parameters:
    - era5_dir: directory where ERA5 daily temperature data is stored
    - year: year to read
    - pop_ssp: population data xarray dataset to match grid
    - to_array: boolean, if True return numpy array, if False return xarray dataset
    Returns:
    - daily_temp: daily temperature data for the year, either as numpy array or xarray dataset
    - num_days: number of days in the year (365 or 366)
    '''
    
    # Read file and shift longitude coordinates
    era5_daily = xr.open_dataset(era5_dir+f'/era5_t2m_{type}_day_{year}.nc')
    
    # Shift longitudinal coordinates  
    era5_daily = era5_daily.assign_coords(longitude=((era5_daily.coords['longitude'] + 180) % 360 - 180)).sortby("longitude")
    
    # Convert to Celsius 
    era5_daily -= 273.15
    
    if pop_ssp is not None:
        # Match grid with population data. Nearest neighbor interpolation
        era5_daily = era5_daily.interp(latitude=np.clip(pop_ssp.latitude, 
                                                        era5_daily.latitude.min().item(), 
                                                        era5_daily.latitude.max().item()), 
                                    method='nearest')
        
    # Swap axes to match required format
    if to_array:
        daily_temp = era5_daily.t2m.values.swapaxes(1,2).swapaxes(0,2)
    else: 
        daily_temp = era5_daily.drop_vars('number')
    
    # Define num_days for leap year/non-leap year
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        num_days = 366
    else:
        num_days = 365
    
    return daily_temp, num_days
        
    

def daily_from_monthly_temp(temp_dir, year, temp_type, to_xarray=False):
    
    '''
    Generate daily temperature data from monthly statistics assuming normal distribution.
    Parameters:
    - temp_dir: directory where monthly statistics files are stored  
    - year: year to generate daily data for
    - temp_type: type of temperature statistic ('MEAN', 'MAX', 'MIN')
    Returns:
    - daily_temp: generated daily temperature data for the year as numpy array
    '''
    
    # Read monthly statistics
    temp_mean, temp_std = open_montlhy_stats(temp_dir, year, temp_type)
    
    # Define num_days for leap year/non-leap year
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        num_days = 366
    else:
        num_days = 365
        
    # Generate daily temperature data
    daily_temp = daily_temp_normal_dist(year, num_days, temp_mean, temp_std)
    
    # print(f'Error statistics for year {year}:', error_daily_stats(year, daily_temp, temp_mean, temp_std))
    
    if to_xarray == True:
        # Convert to xarray DataArray
        daily_dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')
        
        daily_temp = xr.DataArray(daily_temp,
                               coords={'latitude':temp_mean.latitude,
                                       'longitude':temp_mean.longitude,
                                       'valid_time':daily_dates},
                              dims=['latitude', 'longitude', 'valid_time'])
    
    return daily_temp, num_days



def open_montlhy_stats(temp_dir, year, temp_type):
    
    '''
    Read monthly statistics of daily temperature data and reconstruct daily data
    assuming a normal distribution.
    '''
    
    # Read file
    temp_mean = xr.open_dataset(temp_dir+f'GTMP_{temp_type}_30MIN.nc')
    temp_std = xr.open_dataset(temp_dir+f'GTMP_{temp_type}_STD_30MIN.nc')
    
    if temp_type == 'MAX':
        temp_mean = temp_mean.sel(time=f'{year}-01-01')[f'GTMP_MAX_30MIN']
        temp_std = temp_std.sel(time=f'{year}-01-01')[f'GTMPMAX_STD_30MIN']
    
    return temp_mean, temp_std



def daily_temp_normal_dist(year, num_days, temp_mean, temp_std):
    
    '''
    Generate daily temperature data from monthly statistics assuming normal distribution.
    Parameters:
    - year: year to generate daily data for
    - num_days: number of days in the year (365 or 366)
    - temp_mean: xarray DataArray of monthly mean temperatures
    - temp_std: xarray DataArray of monthly standard deviation of temperatures
    Returns:
    - synthetic_daily: generated daily temperature data for the year as numpy array
    '''
    
    temp_mean = temp_mean.sel(time=f'{year}-01-01').drop_vars('time')
    temp_std = temp_std.sel(time=f'{year}-01-01').drop_vars('time')
    
    # Generate daily dates for the year
    daily_dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')

    lats = 360
    lons = 720
    day_idx=0
    
    synthetic_daily = np.empty((lats, lons, num_days))
    
    for i, m in enumerate(temp_mean.NM.values):
        
        # Find number of days in the month
        month = i + 1
        n_days = np.sum(daily_dates.month == month)
        
        # Get mean and std for the month
        mu = temp_mean['GTMP_MEAN_30MIN'].sel(NM= m).values
        sigma = temp_std['GTMP_STD_30MIN'].sel(NM= m).values
        
        # Generate random daily values from normal distribution
        vals = np.random.normal(mu, sigma, size=(n_days, lats, lons))
        
        # Standardize generated values and rescale to original mean and std
        vals_mean = np.mean(vals, axis=0)
        vals_std = np.std(vals, axis=0)
        
        vals_std[vals_std == 0] = 1.0
        
        vals = (vals - vals_mean) / vals_std
        vals = vals * sigma + mu 

        # Assign generated values to the correct days in the year
        synthetic_daily[..., day_idx:day_idx+n_days] = vals.swapaxes(0,1).swapaxes(1,2)
        day_idx += n_days
        
    return synthetic_daily



def error_daily_stats(year, daily_temp, temp_mean, temp_std):
    
    '''
    Calculate error between generated daily temperature statistics and original monthly statistics.
    Parameters:
    - daily_temp: generated daily temperature data as numpy array
    - temp_mean: xarray DataArray of original monthly mean temperatures
    - temp_std: xarray DataArray of original monthly standard deviation of temperatures
    Returns:
    - mean_error: error in mean temperature between generated daily data and original monthly data
    - std_error: error in standard deviation between generated daily data and original monthly data
    '''
    
    daily_dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')
    
    daily_temp_xr = xr.DataArray(daily_temp,
                               coords={'latitude':temp_mean.latitude,
                                       'longitude':temp_mean.longitude,
                                       'time':daily_dates},
                              dims=['latitude', 'longitude', 'time'])
    # Calculate monthly mean and std from generated daily data
    monthly_mean = daily_temp_xr.resample(time='1M').mean()
    monthly_std = daily_temp_xr.resample(time='1M').std()
    
    # Calculate error between generated monthly statistics and original monthly statistics
    mean_error = (monthly_mean.mean(dim='time') - temp_mean).mean().item()
    std_error = (monthly_std.mean(dim='time') - temp_std).mean().item()
    
    np.set_printoptions(suppress=True, precision=2)
    print('Percentage error per month:', 
          np.nanmean(np.nanmean((monthly_mean.values - temp_mean.values) / temp_mean.values, axis=0), axis=0) * 100)
    
    return mean_error, std_error