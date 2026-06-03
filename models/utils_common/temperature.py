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
    era5_daily = era5_daily.assign_coords(
        longitude=((era5_daily.coords["longitude"] + 180) % 360 - 180)
        ).sortby("longitude")
    
    # Convert to Celsius 
    era5_daily -= 273.15
    
    if pop_map is not None:
        # Match grid with population data. Nearest neighbor interpolation
        era5_daily = era5_daily.interp(
            latitude=np.clip(pop_map.latitude, 
            era5_daily.latitude.min().item(), 
            era5_daily.latitude.max().item()), 
            method="nearest"
            )
        
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



def DailyTemperatureFromEmulator(em, year):
    
    """
    Generate daily temperature data for a given year from the monthly statistics (std) 
    that the EERIE emulator produces for that year, assuming a normal distribution.
    """
    
    
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        NUMBER_DAYS = 366
    else:
        NUMBER_DAYS = 365
    
    # Run emulator to get monthly mean and std data for the year
    temperature_mean, temperature_std = em.run_emulator(year)
    
    # Generate monthly dates (15th OR 16th of each month)
    monthly_dates = pd.date_range(
        start=f"15/12/{year-1}", 
        end=f"15/2/{year+1}", freq="ME"
        ) - pd.DateOffset(days=15)
    
    # Interpolate to daily data
    temperature_mean = (
        temperature_mean
        .assign_coords(month=monthly_dates) # Change monthly coordinates to datetime
        .rename({"month": "valid_time"}) 
        .drop_vars("time") # Drop original time coordinate
        .resample(valid_time="1D")
        .interpolate("slinear") # Interpolate to daily data
        .sel(valid_time=slice(f"{year}-01-01", f"{year}-12-31")) # Keep selected year
    )

    temperature_std = temperature_std.rename({"month":"NM"})

    daily_temperature = DailyTemperatureFromNormalPDF(
        time=year, 
        number_days=NUMBER_DAYS, 
        temp_daily_mean=temperature_mean, 
        temp_std=temperature_std, 
        std_factor=1
        )
    
    return daily_temperature
    
    

def DailyFromMonthlyTemperature(temp_dir, year, temp_type, std_factor, to_xarray=False):
    
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
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        NUMBER_DAYS = 366
    else:
        NUMBER_DAYS = 365
    
    # Read monthly statistics
    temperature_mean, temperature_monthly_std = OpenMontlhyTemperatures(
        temp_dir=temp_dir, 
        temp_type=temp_type
        )
    
    # Select std data and get the mean of the specific year
    temperature_monthly_std = (
        temperature_monthly_std
        .sel(time=slice(f"{year}-01-01", f"{year}-01-01"))
        .mean(dim="time")
    )
    
    # Prepare monthly mean data including December of previous year and January of next year
    temperature_mean_present = []
    # Repeat 2100 for January 2101
    final_year = 2100 if year == 2100 else year+1

    # Append December of previous year, January of selected year, and January of next year for interpolation
    temperature_mean_present.append(
        temperature_mean.sel(time=f"{year-1}-01-01").isel(NM=-1)
    ).append(
        temperature_mean.sel(time=f"{year}-01-01")
    ).append(
        temperature_mean.sel(time=f"{final_year}-01-01").isel(NM=0)
    )

    # Concatenate December of previous year and January of next year for smooth transition
    dec_years_jan = xr.concat(
        temperature_mean_present, 
        dim="NM",
        coords="different",
        compat="equals"
    )
    
    # Generate monthly dates (15th OR 16th of each month)
    monthly_dates = pd.date_range(
        start=f"15/12/{year-1}", 
        end=f"15/2/{year+1}", freq="ME"
        ) - pd.DateOffset(days=15)
    
    # Change NM data to monthly data and rename variable
    temperature_interpolated = (
        dec_years_jan
        .assign_coords(NM=monthly_dates)  # Change monthly coordinates to datetime
        .rename({"NM": "valid_time"})
        .drop_vars("time") # Drop original time coordinate
        .resample(valid_time="1D")
        .interpolate("slinear") # Interpolate to daily data
        .sel(valid_time=slice(f"{year}-01-01", f"{year}-12-31")) # Keep selcted year
    )
    
    
    # Generate daily temperature data from monthly STD statistics
    daily_temperature = DailyTemperatureFromNormalPDF(
        time=year, 
        number_days=NUMBER_DAYS, 
        temp_daily_mean=dec_years_jan, 
        temp_std=temperature_monthly_std, 
        std_factor=std_factor
        )
    
    if to_xarray == True:
        # Convert to xarray DataArray
        daily_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        
        # Create xarray DataArray with original coordinates
        daily_temperature = xr.DataArray(
            daily_temperature,
            coords={"latitude": temperature_interpolated.latitude,
                    "longitude": temperature_interpolated.longitude,
                    "valid_time":daily_dates},
            dims=["latitude", "longitude", "valid_time"]
            )
    
    return daily_temperature, NUMBER_DAYS



def OpenMontlhyTemperatures(temp_dir, temp_type):
    
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



def DailyTemperatureFromNormalPDF(time, number_days, temp_daily_mean, temp_std, std_factor):
    
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
    daily_dates = pd.date_range(f"{time}-01-01", f"{time}-12-31", freq="D")

    lats = 360
    lons = 720
    day_idx = 0

    synthetic_daily = np.empty((lats, lons, number_days))
    
    for i, m in enumerate(temp_std.NM.values):
        
        # Find number of days in the month
        month = i + 1
        n_days = np.sum(daily_dates.month == month)
        
        # Get mean and std for the month
        sigma = temp_std.sel(NM=m).values
        mean = temp_daily_mean.values[..., day_idx:day_idx+n_days]
        
        #Create array of zeros with same shape as mean to use as mu
        mu = np.zeros([lats,lons])
        
        # Adjust std to change daily variability
        sigma = std_factor*sigma 
        
        # Change any negative and nan std values to a small positive number to avoid issues with normal distribution
        sigma = np.where(sigma <= 0, 0.1, sigma)     

        # Generate random daily variability from normal distribution
        vals = np.random.normal(mu, sigma, size=(n_days, lats, lons))
        vals = vals + mean.swapaxes(1,2).swapaxes(1,0)

        # Assign generated values to the correct days in the year
        synthetic_daily[..., day_idx:day_idx+n_days] = vals.swapaxes(0,1).swapaxes(1,2)
        day_idx += n_days
        
    return synthetic_daily



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
    
    daily_temp_xr = xr.DataArray(
        daily_temp,
        coords={"latitude":temp_mean.latitude,
                "longitude":temp_mean.longitude,
                "time":daily_dates},
        dims=["latitude", "longitude", "time"]
        )
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