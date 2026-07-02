import numpy as np
import pandas as pd
import xarray as xr
import re



def LoadDailyTemperatures(temp_dir, scenario, temp_type, year, pop_map, std_factor):
    
    """
    Select the temperature data type to use (ERA5 or monthly statistics)
    """
    
    if re.search(r"SSP[1-5]_ERA5", scenario):
        daily_temp, num_days = DailyTemperatureERA5(temp_dir, year, temp_type, pop_map, to_array=True)
        
    else:
        daily_temp, num_days = DailyFromMonthlyTemperature(temp_dir, year, temp_type.upper(), std_factor)
        
    return daily_temp, num_days



def DailyTemperatureERA5(era5_dir, year, temp_type, pop_map=None, to_array=False):
    
    """
    Read daily ERA5 temperature data for a specific year, shift longitude coordinates,
    convert to Celsius, and match grid with population data.
    Parameters:
    - era5_dir: directory where ERA5 daily temperature data is stored
    - year: year to read
    - pop_map: population data xarray dataset to match grid
    - to_array: boolean, if True return numpy array, if False return xarray dataset
    Returns:
    - daily_temp: daily temperature data for the year, either as numpy array or xarray dataset
    - num_days: number of days in the year (365 or 366)
    """
    
    if year < 1980 or year > 2025:
        raise ValueError(
            f"ERA5 data for year {year} is not available "
            f"(valid range: 1980–2025)."
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
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        num_days = 366
    else:
        num_days = 365
    
    return daily_temp, num_days
        
    

def DailyFromMonthlyTemperature(temperature_mean, temperature_std, years, std_factor, to_xarray=False):
    
    """
    Generate daily temperature data fro a given year from monthly statistics assuming 
    a normal distribution.
    
    ------------
    Parameters:
    - temp_dir: directory where monthly statistics files are stored  
    - year: year to generate daily data for
    - temp_type: type of temperature statistic ("MEAN", "MAX", "MIN")
    - std_factor: factor to adjust daily variability (standard deviation)
    - to_xarray: boolean, if True return xarray DataArray, if False return numpy array
    
    ------------
    Returns:
    - daily_temperature: generated daily temperature data for the year as numpy array or xarray DataArray
    - NUMBER_DAYS: number of days in the year (365 or 366)
    """
    
    # Define num_days for leap year/non-leap year
    # ---------- Importing single year -------------
    if isinstance(years, int):
        mid_year = years
        years = [years]
        if (mid_year % 4 == 0 and mid_year % 100 != 0) or (mid_year % 400 == 0):
            NUMBER_DAYS = 366
        else:
            NUMBER_DAYS = 365
            
    # ---------- Importing mean of multiple years ---------
    else: 
        NUMBER_DAYS = 366
        mid_year = 2000
        years = years
        
    # # Read monthly statistics
    # temperature_mean, temperature_std = OpenMonthlyTemperatures(
    #     temp_dir=temp_dir, 
    #     temp_type=temp_type
    #     )
    
    temperature_std = (
        temperature_std
        .sel(time=slice(f"{years[0]}-01-01", f"{years[-1]}-01-01"))
        .mean(dim="time")
    )
    
    # Select std data and get the mean of the specific year
    final_year = years[-1] if years[-1] == 2100 else years[-1] + 1
    temp_core = temperature_mean.sel(time=slice(f"{years[0]-1}-01-01", f"{final_year}-01-01"))
    
    dec_years_jan = (
        temp_core
        .stack(valid_time=("time", "NM"))
        .drop_vars(["time", "NM"], errors="ignore")
    )
    
    # Select only December of previous year and January of next year for smooth transition
    dec_years_jan = dec_years_jan.isel(
        valid_time=slice(11, -11)  
    )
    
    monthly_dates = pd.date_range(
        start=f"15/12/{years[0]-1}", 
        end=f"15/2/{years[-1]+1}",
        freq="ME"
        ) - pd.DateOffset(days=15)
    
    
    temperature_interpolated = (
        dec_years_jan
        .assign_coords(valid_time=monthly_dates)
        .resample(valid_time="1D")
        .interpolate("slinear")
        .sel(valid_time=slice(f"{years[0]}-01-01", f"{years[-1]}-12-31"))
        .groupby("valid_time.dayofyear")
        .mean("valid_time")
    )
    
    # Generate daily temperature data from monthly STD statistics
    daily_temperature = DailyTemperatureFromNormalPDF(
        year=mid_year, 
        number_days=NUMBER_DAYS, 
        temp_daily_mean=temperature_interpolated, 
        temp_std=temperature_std, 
        std_factor=std_factor
        )
    
    if to_xarray == True:
        # Convert to xarray DataArray
        daily_dates = pd.date_range(f"{mid_year}-01-01", f"{mid_year}-12-31", freq="D")
        
        # Create xarray DataArray with original coordinates
        daily_temperature = xr.DataArray(
            daily_temperature,
            coords={"latitude": temperature_interpolated.latitude,
                    "longitude": temperature_interpolated.longitude,
                    "valid_time":daily_dates},
            dims=["latitude", "longitude", "valid_time"]
            )
    
    return daily_temperature, NUMBER_DAYS



def OpenMonthlyTemperatures(temp_dir, temp_type):
    
    """
    Read monthly statistics of daily temperature data (mean and standard deviation)
    according to the temperature type (temp_type):
    - temp_type = "MEAN": mean and std of daily mean temperatures
    - temp_type = "MAX": mean and std of daily maximum temperatures
    
    -----------
    Parameters:
    - temp_dir: directory where monthly statistics files are stored (IMAGE folder)
    - temp_type: type of temperature statistic ("MEAN", "MAX")
    
    ----------
    Returns:
    - temp_mean: xarray DataArray of monthly mean temperatures
    - temp_std: xarray DataArray of monthly standard deviation of temperatures
    """
    
    # Read temperature mean and std files of from scenario 
    if temp_type.upper() == "MEAN":
        temp_mean = xr.open_dataset(temp_dir+f"/GTMP_30MIN.nc")
    else: 
        temp_mean = xr.open_dataset(temp_dir+f"/GTMP_{temp_type}_30MIN.nc")
    temp_std = xr.open_dataset(temp_dir+f"/GTMP_STD_30MIN.nc")
    
    # Select temperature variable depending on type
    if temp_type.upper() == "MEAN":
        temp_mean = temp_mean[f"GTMP_30MIN"]
        temp_std = temp_std[f"GTMP_STD_30MIN"]
    
    if temp_type.upper() == "MAX":
        temp_mean = temp_mean[f"GTMP_MAX_30MIN"]
        temp_std = temp_std[f"GTMPMAX_STD_30MIN"]
    
    return temp_mean, temp_std



def DailyTemperatureFromNormalPDF(year, number_days, temp_daily_mean, temp_std, std_factor):
    
    """
    Generate daily temperature data from monthly statistics assuming normal distribution.
    Parameters:
    - year: year to generate daily data for
    - num_days: number of days in the year (365 or 366)
    - temp_mean: xarray DataArray of monthly mean temperatures
    - temp_std: xarray DataArray of monthly standard deviation of temperatures
    Returns:
    - synthetic_daily: generated daily temperature data for the year as numpy array
    """
    
    # Generate daily dates for the year
    daily_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    
    # Prepare std data and adjust with std_factor
    sigma_months = np.maximum(temp_std.values * std_factor, 0.1)

    # Expand sigma_months to match the number of days in the year
    month_indices = daily_dates.month.values - 1
    sigma_daily = sigma_months[...,month_indices]

    # Generate daily temperature data from normal distribution with mean and std
    vals = np.random.normal(0.0, sigma_daily)

    return vals + temp_daily_mean.values



def error_daily_stats(year, daily_temp, temp_mean, temp_std):
    
    """
    Calculate error between generated daily temperature statistics and original monthly statistics.
    Parameters:
    - daily_temp: generated daily temperature data as numpy array
    - temp_mean: xarray DataArray of original monthly mean temperatures
    - temp_std: xarray DataArray of original monthly standard deviation of temperatures
    Returns:
    - mean_error: error in mean temperature between generated daily data and original monthly data
    - std_error: error in standard deviation between generated daily data and original monthly data
    """
    
    daily_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    
    daily_temp_xr = xr.DataArray(daily_temp,
                               coords={"latitude":temp_mean.latitude,
                                       "longitude":temp_mean.longitude,
                                       "time":daily_dates},
                              dims=["latitude", "longitude", "time"])
    # Calculate monthly mean and std from generated daily data
    monthly_mean = daily_temp_xr.resample(time="1M").mean()
    monthly_std = daily_temp_xr.resample(time="1M").std()
    
    # Calculate error between generated monthly statistics and original monthly statistics
    mean_error = (monthly_mean.mean(dim="time") - temp_mean).mean().item()
    std_error = (monthly_std.mean(dim="time") - temp_std).mean().item()
    
    np.set_printoptions(suppress=True, precision=2)
    print("Percentage error per month:", 
          np.nanmean(np.nanmean((monthly_mean.values - temp_mean.values) / temp_mean.values, axis=0), axis=0) * 100)
    
    return mean_error, std_error