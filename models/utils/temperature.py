import numpy as np
import pandas as pd
import xarray as xr
import re



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
    Parameters:
    - era5_dir: directory where ERA5 daily temperature data is stored
    - year: year to read
    - pop_map: population data xarray dataset to match grid
    - to_array: boolean, if True return numpy array, if False return xarray dataset
    Returns:
    - daily_temp: daily temperature data for the year, either as numpy array or xarray dataset
    - num_days: number of days in the year (365 or 366)
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
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        num_days = 366
    else:
        num_days = 365
    
    return daily_temp, num_days
        
    

def DailyFromMonthlyTemperature(temperature_mean, temperature_std, years_in, random_vals, to_xarray=False):
    #temp_dir, temp_type, years_in, random_vals, to_xarray=False):
    
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
    if isinstance(years_in, int):
        mid_year = years_in
        years = [years_in]
        if (mid_year % 4 == 0 and mid_year % 100 != 0) or (mid_year % 400 == 0):
            NUMBER_DAYS = 366
        else:
            NUMBER_DAYS = 365
            
    # ---------- Importing mean of multiple years ---------
    else: 
        NUMBER_DAYS = 366
        mid_year = 2000
        years = years_in
        
    # Open monthly temperature statistics (mean and std) for the given years
    # temperature_mean, temperature_std = OpenMonthlyTemperatures(temp_dir, temp_type)
    
    temperature_std = (
        temperature_std
        .sel(time=slice(f"{years[0]}-01-01", f"{years[-1]}-01-01"))
        .mean(dim="time")
    )
        
        
    # Calculate the monthly climatology (mean) for the selected years
    if isinstance(years_in, int):
        monthly_climatology = temperature_mean.sel(time=f"{mid_year}-01-01")
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
        start=f"15/12/{mid_year-1}", 
        end=f"15/2/{mid_year+1}",
        freq="ME"
        ) - pd.DateOffset(days=15)

    # Interpolate the padded climatology to daily resolution using linear interpolation
    temperature_interpolated = (
        padded_climatology
        .assign_coords(NM=monthly_dates)
        .rename({"NM": "dayofyear"})
        .resample(dayofyear="1D")
        .interpolate("slinear")
        .sel(dayofyear=slice(f"{mid_year}-01-01", f"{mid_year}-12-31"))
        .values
    )

    # Generate daily temperature data from monthly STD statistics
    daily_temperature = DailyTemperatureFromNormalPDF(
        year=mid_year, 
        temp_daily_mean=temperature_interpolated, 
        temp_std=temperature_std, 
        random_vals=random_vals
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
    
    return temp_mean.astype(np.float32), temp_std.astype(np.float32)



def DailyTemperatureFromNormalPDF(year, temp_daily_mean, temp_std, random_vals):
    
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

    # Generate daily dates for the year and get the corresponding month for each day
    months_per_day = pd.date_range(f"{year}-01-01", f"{year}-12-31").month -1

    # Expand the standard deviation data to match the number of days in the year
    std_expanded = (
        temp_std
        .assign_coords(NM=np.arange(0, 12))
        .isel(NM=months_per_day)
        .values
    )

    # Cut random vals to match 366 or 365 days
    random_vals = random_vals[:,:,:temp_daily_mean.shape[2]]

    # Calculate the final daily temperature data by adding the mean and scaled standard deviation
    final_result = temp_daily_mean + (random_vals * std_expanded)
    
    return final_result



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