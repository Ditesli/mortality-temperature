import numpy as np
import pandas as pd
import xarray as xr
import re, os
from calendar import isleap



def LoadDailyTemperatures(temp_dir, scenario, temp_type, year, pop_map, std_factor):
    
    """
    Select the temperature data type to use (ERA5 or monthly statistics)
    """
    
    if "ERA5" in scenario:
        daily_temp, num_days = DailyTemperatureERA5(temp_dir, year, temp_type, pop_map, to_array=True)
        
    else:
        daily_temp, num_days = DailyFromMonthlyTemperature(temp_dir, year, temp_type.upper(), std_factor)
        
    return daily_temp, num_days



def DailyTemperatureERA5(era5_dir, year, temp_type, pop_map=None, to_array=False):
    
    """
    Read daily ERA5 temperature data for a specific year, shift longitude coordinates,
    convert to Celsius, and match grid with population data.
    """
    
    if year < 1970 or year > 2025:
        raise ValueError(
            f"ERA5 data for year {year} is not available "
            f"(valid range: 1970–2025)."
      )        
    
    # Read file and shift longitude coordinates
    era5_daily = xr.open_dataset(era5_dir+f"/era5_t2m_{temp_type}_day_{year}.nc")
    
    # Shift longitudinal coordinates  
    era5_daily = era5_daily.assign_coords(longitude=((era5_daily.coords["longitude"] + 180) % 360 - 180)).sortby("longitude")
    
    # Convert to Celsius 
    era5_daily -= 273.15
    
    if pop_map is not None:
        # Match grid with population data. Nearest neighbor interpolation
        era5_daily = era5_daily.interp(latitude=np.clip(pop_map.latitude, 
                                                        era5_daily.latitude.min().item(), 
                                                        era5_daily.latitude.max().item()), 
                                    method="nearest")
        
    # Swap axes to match required format
    if to_array:
        daily_temp = era5_daily.t2m.values.swapaxes(1,2).swapaxes(0,2)
    else: 
        daily_temp = era5_daily.drop_vars("number")
    
    # Define num_days for leap year/non-leap year        
    num_days = 366 if isleap(year) else 365
    
    return daily_temp, num_days
        
    

def DailyFromMonthlyTemperature(temperature_mean, temperature_std, years_in, random_vals, to_xarray=False):

    """
    Generate daily temperature data fro a given year from monthly statistics assuming 
    a normal distribution.
    """
    
    # -------------- Define params for leap year/non-leap year ------------------
    
    # Importing single year
    if isinstance(years_in, int):
        mid_year = years_in
        years = [years_in]
        NUMBER_DAYS = 366 if isleap(mid_year) else 365
            
    # Importing mean of multiple years
    else: 
        NUMBER_DAYS = 365
        mid_year = 2005
        years = years_in
        
        
    # Select std for defined period
    temperature_std = (
        temperature_std
        .sel(time=slice(f"{years[0]}-01-01", f"{years[-1]}-01-01"))
        .mean(dim="time")
    )
        
        
    # Calculate the monthly climatology (mean) for the selected years
    if isinstance(years_in, int):
        monthly_climatology = temperature_mean.sel(time=f"{mid_year}-01-01").drop_vars("time")
    else:
        monthly_climatology = (
            temperature_mean
            .sel(time=slice(f"{years[0]-1}-01-01", f"{years[-1]}-01-01"))
            .mean(dim="time")
        )

    # Pad the December and January data to ensure smooth transition between years
    december_pad = monthly_climatology.isel(NM=11)
    january_pad = monthly_climatology.isel(NM=0)
    # Concatenate the padded December and January data with the monthly climatology
    padded_climatology = xr.concat([december_pad, monthly_climatology, january_pad], dim="NM")

    monthly_dates = pd.date_range(
        start=f"{mid_year-1}-12-15", 
        end=f"{mid_year+1}-02-15",
        freq="ME"
        ) - pd.DateOffset(days=15)

    # Interpolate the padded climatology to daily resolution using linear interpolation
    temperature_interpolated = (
        padded_climatology
        .assign_coords(NM=monthly_dates)
        .rename({"NM": "time"})
        .resample(time="1D")
        .interpolate("slinear")
        .sel(time=slice(f"{mid_year}-01-01", f"{mid_year}-12-31"))
    )

    # Generate daily temperature data from monthly STD statistics
    daily_temperature = DailyTemperatureFromNormalPDF(
        year=mid_year, 
        temp_daily_mean=temperature_interpolated, 
        temp_std=temperature_std, 
        random_vals=random_vals
        )
    
    if to_xarray == True:
        pass
    else:
        daily_temperature = daily_temperature.astype(np.float32).values
    
    return daily_temperature, NUMBER_DAYS



def OpenMonthlyTemperatures(temp_dir, temp_type):
    
    """
    Read monthly statistics of daily temperature data (mean and standard deviation)
    according to the temperature type (temp_type).
    """
    
    temp_type_upper = temp_type.upper()

    if temp_type_upper == "MEAN":
        mean_file = os.path.join(temp_dir, "GTMP_30MIN.nc")
        mean_var = "GTMP_30MIN"
        std_var = "GTMP_STD_30MIN"
    elif temp_type_upper == "MAX":
        mean_file = os.path.join(temp_dir, f"GTMP_{temp_type_upper}_30MIN.nc")
        mean_var = f"GTMP_{temp_type_upper}_30MIN"
        std_var = "GTMPMAX_STD_30MIN"
        
    std_file = os.path.join(temp_dir, "GTMP_STD_30MIN.nc")

    with xr.open_dataset(mean_file) as ds_mean:
        temp_mean = ds_mean[mean_var].astype("float32").load()
        
    with xr.open_dataset(std_file) as ds_std:
        temp_std = ds_std[std_var].astype("float32").load()

    return temp_mean, temp_std



def DailyTemperatureFromNormalPDF(year, temp_daily_mean, temp_std, random_vals):
    
    """
    Generate daily temperature data from monthly statistics assuming normal distribution.
    """    
    
    # Get corresponding month per day
    months_per_day = temp_daily_mean.time.dt.month - 1
    
    # Map std file to daily std values replicating the same value per month
    temp_std = (
        temp_std
        .assign_coords(NM=np.arange(0, 12))
        .isel(NM=months_per_day)
        .drop_vars("NM")
    )
    
    # Cut random_vals distribution if needed (e.g. 366 days to 365)
    random_vals = random_vals[:,:,:temp_daily_mean.shape[2]]
    
    # Add random noise from std to daily temperature
    final_result = temp_daily_mean + (random_vals * temp_std)
    
    return final_result