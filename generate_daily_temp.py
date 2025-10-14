import xarray as xr
import pandas as pd
import numpy as np


###---------------------------------------------------------------------------------------------
### Generate daily temperature data for one year and one C-Category --- Interpolate within months and adding noise

def get_montlhy_nc(ccategory, year, month_start, month_end, cc_path_mean, gcm_diff, gcm_start, gcm_end):
    
    ### Use IMAGE land interpolation method
    nsat_month_cc = gcm_start + (cc_path_mean.at[ccategory, f'{year}'] / gcm_diff) * gcm_end
    
    ### Slice selected months and delete empty variable  "time"
    return nsat_month_cc.sel(NM=slice(month_start, month_end)).mean(dim='time', skipna=True)


def daily_temp_interp(ccategory, year, cc_path_mean, gcm_diff, gcm_start, gcm_end, noise, noise_leap):
    
    ### Apply get_montlhy_nc function to near years
    nsat_month_cc = get_montlhy_nc(ccategory, year, 1, 12, cc_path_mean, gcm_diff, gcm_start, gcm_end)
    nsat_month_cc_pre = get_montlhy_nc(ccategory, year-1, 12, 12, cc_path_mean, gcm_diff, gcm_start, gcm_end) if year > 2000 else nsat_month_cc.sel(NM=1)
    nsat_month_cc_pos = get_montlhy_nc(ccategory, year+1, 1, 1, cc_path_mean, gcm_diff, gcm_start, gcm_end) if year < 2100 else nsat_month_cc.sel(NM=12)

    ### concat data in one xarray
    dec_year_jan = xr.concat([nsat_month_cc_pre, nsat_month_cc, nsat_month_cc_pos], dim='NM')

    ### Generate monthly dates
    monthly_time = pd.date_range(start=f'15/12/{year-1}', end=f'15/2/{year+1}', freq='ME') - pd.DateOffset(days=15)
    
    ### Change NM data to monthly data and rename variable
    dec_year_jan = dec_year_jan.assign_coords(NM=monthly_time).rename({'NM': 'time'})
    
    ### Interpolation (slinear for now)
    dec_year_jan_daily = dec_year_jan.resample(time='1D').interpolate('slinear')
    
    ### Make Antarctica values NaN
    dec_year_jan_daily = dec_year_jan_daily.where(dec_year_jan_daily.latitude >= -60, np.nan)
    
    ### Keep only data within the selected year in numpy array format
    daily_1year = dec_year_jan_daily.sel(time=slice(f'{year}/1/1', f'{year}/12/31')).tas.values#.round(1)
    
    ### Add random noise
    
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        noise = noise_leap
        num_days = 366
    else:
        noise = noise
        num_days = 365
    
    ### Shuffle noise array
    np.random.shuffle(noise)
    ### Add to daily temperature
    daily_1year += noise
    
    print(f'{ccategory} - {year} daily temperatures generated')

    return daily_1year, num_days


def create_noise(std_value):
    
    np.random.seed(0)
    noise_leap = np.random.normal(scale=std_value, size=(720, 1440, 366))#.round(1) 
    ### Remove noise for 29th of February
    noise = np.delete(noise_leap, 59, axis=2)  
    
    print('Temperature noise generated')

    return noise, noise_leap


### ---------------------------------------------------------------------------------------------
### Read and process daily ERA5 Reanalysis data

def daily_temp_era5(year, pop_ssp, variable, variable_type, to_array=False, interp=False):
    
    # Read file and shift longitude coordinates
    era5_daily = xr.open_dataset(f'X:\\user\\liprandicn\\Data\\ERA5\\{variable}_daily\\era5_{variable}_{variable_type}_day_{year}.nc')

    # Shift longitudinal coordinates
    era5_daily = era5_daily.assign_coords(longitude=((era5_daily.coords['longitude'] + 180) % 360 - 180)).sortby("longitude")
    
    # Convert to Celsius 
    era5_daily -= 273.15
    
    if interp:
        # Match grid with population data
        era5_daily = era5_daily.interp(longitude=pop_ssp.longitude, latitude=pop_ssp.latitude, method="nearest")
    
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
        
    print(f'{year} ERA5 daily {variable_type} {variable} imported')
    
    return daily_temp, num_days