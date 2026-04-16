import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..")))

import numpy as np
import pandas as pd
import xarray as xr
import pyreadr


def ERA5TemperaturePercentiles(wdir, era5_dir, years):
    
    """
    Calculate a mapping per grid cell with the temperature percentile
    per grid cell for the historical 30 year period given by the parameters
    from ERA5 temperature data.
    """
    
    print(f"Calculating ERA5 temperature percentiles for range {years[0]}-{years[-1]}...")
    
    # Load in ERF from Scovronick to get the percentiles to calculate
    erf = pyreadr.read_r(wdir + "/data/Scovronick_SM/Fig2_20Nov2025.Rdata")
    percentiles = erf["cvd"]["Percentile"].values/100
    
    # Calculate the percentiles, processing in latitude bands to optimize memory usage
    percentile_final = CalculateBandPercentiles(era5_dir, years, percentiles, step=25)
    
    # Convert Kelvin to Celsius
    percentile_final -= 273.15
    
    print("Saving percentiles to NetCDF file...")
    percentile_final.to_netcdf(
        wdir + f"/data/Temperature_Percentiles/ERA5_Tmean_Percentiles_{years[0]}-{years[-1]+1}.nc")
    
    
    
def CalculateBandPercentiles(data_path: str, years: np.ndarray, percentiles: np.ndarray, step: int):
    
    """
    Calculate the Nth percentile of daily temperature from ERA5 reanalysis data.
    File resolution is 0.25 degrees (720x1440 dimension).
    ERA5 files were renamed following the format: 'era5_t2m_mean_day_<year>.nc'.
    The data is processed in bands of latitude to avoid memory allocation issues.
    Depending on the available memory, a step of 20 or 30 is recommended. 
    If unable to allocate memory to process, decrease the step size.

    Parameters:
    - data_path: str
        Path to the directory containing the ERA5 data files.
    - years: np.ndarray 
        Array of years for which to calculate percentiles, usually a 30 year period.
    - percentiles: np.ndarray
        Array of percentiles to calculate, prescribed by the R data file.
    - step: int
        Step size for latitude bands to optimize memory usage.

    Returns:
    - None, saves the results as NetCDF files.
    """
    
    # Process data in latitude bands to optimize memory usage
    lats = range(0, 720, step)  

    # Array to append the calculated percentiles of each latitude band range
    percentile_bands = [] 

    # Iterate over each latitude band
    for lat in lats:
        
        print(f"Processing latitudes: {lat} - {lat + step}")
        
        # Array to append the temperature data for the current latitude band across all years
        temporal_lat_data = []
        
        for year in years:
            
            # Load the dataset for the specific year and latitude
            data_name = f'era5_t2m_mean_day_{year}.nc'
            with xr.open_dataset(data_path + "/" + data_name) as data:
                data = data['t2m'].isel(latitude=slice(lat, lat + step))
                temporal_lat_data.append(data.load())
            
        # Concatenate the data for the current latitudes across all years
        temporal_data = xr.concat(temporal_lat_data, dim='valid_time')
        
        # Array to append the calculated percentiles for the current latitudes band
        p_bands = []
        for p in percentiles:
            
            print(f"Lat: {lat}-{lat + step}, p{np.round(p*100,1)}")
            
            # Calculate the nth percentile for the latitudes band 
            p_band = temporal_data.quantile(p, dim='valid_time')
            p_bands.append(p_band)
        
        # Concatenate the percentiles for the current latitudes band and append to the final array
        p_bands_lat = xr.concat(p_bands, dim='quantile')
        percentile_bands.append(p_bands_lat)
    
    # Concatenate the percentiles for all latitude bands to create the final xarray
    percentile_final = xr.concat(percentile_bands, dim='latitude')
    
    return percentile_final