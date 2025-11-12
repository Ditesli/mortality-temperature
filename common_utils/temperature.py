import numpy as np
import xarray as xr



def daily_temp_era5(era5_dir, year, pop_ssp, to_array=False):
    
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
    era5_daily = xr.open_dataset(era5_dir+f'era5_t2m_max_day_{year}.nc')
    
    # Shift longitudinal coordinates  
    era5_daily = era5_daily.assign_coords(longitude=((era5_daily.longitude + 180) % 360 - 180))
    era5_daily = era5_daily.sel(longitude=np.unique(era5_daily.longitude)).sortby("longitude")
    
    # Convert to Celsius 
    era5_daily -= 273.15
    
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
        
    print(f'ERA5 {year} daily temperatures imported')
    
    return daily_temp, num_days