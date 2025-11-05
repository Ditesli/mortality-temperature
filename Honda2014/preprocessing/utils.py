import numpy as np
import xarray as xr



def calculate_optimal_temperature(data_path: str, final_path:str,  years: np.ndarray, step: int):
    
    """
    Calculate the optimal temperature defined in Honda et al., (2014) as the 
    83.6th percentile of daily mean temperature from ERA5 reanalysis data.

    Parameters:
    - data_path: str
          Path to the directory containing the ERA5 data files at daily temporal resolution
          and spatial resolution of 0.25 degrees (720x1440 dimension). Files are renamed 
          following the format: 'era5_t2m_mean_day_<start_year>_<final_year>.nc'
    - final_path: str
          Path to the directory where the results will be saved.
    - years: np.ndarray
          Array of years for which to calculate percentiles, usually a 30 year period.
    - step: int
          Step size for latitude bands to optimize memory usage.
          The data is processed in bands of latitude to avoid memory allocation issues.
          Depending on the available memory, a step of 20 is recommended. 
          If unable to allocate memory to process, decrease the step size.

    Returns:
    - None
          The code saves the results as NetCDF files in the data directory.
    """
    
    # Process data in latitude bands to optimize memory usage
    lats = range(0, 720, step)  

    percentile_bands = [] 

    # Iterate over each latitude band
    for lat in lats:
        
        print(f"Processing latitudes: {lat} - {lat + step}")
        
        temporal_lat_data = []
        
        for year in years:
            
            # Load the dataset for the specific year and latitude
            data_name = f'era5_t2m_max_day_{year}.nc'
            with xr.open_dataset(data_path + data_name) as data:
                data = data['t2m'].isel(latitude=slice(lat, lat + step))
                temporal_lat_data.append(data.load())
            
        # Concatenate the data for the current latitudes across all years
        temporal_data = xr.concat(temporal_lat_data, dim='valid_time')
        
        # Calculate the 95th percentile for the latitude band and append it to the list
        percentile_band = temporal_data.quantile(0.836, dim='valid_time')
        percentile_bands.append(percentile_band)
        
    percentile_final = xr.concat(percentile_bands, dim='latitude')
    percentile_final.name = 't2m_p84'
    
    # Convert from Kelvin to Celsius
    percentile_final -= 273.15
    
    # Shift longitude coordinates to -180 - 180 range
    percentile_final = percentile_final.assign_coords(longitude=((percentile_final.coords['longitude'] + 180) % 360 - 180)).sortby("longitude")
    
    # Drop quantile coordinate
    percentile_final = percentile_final.drop_vars('quantile')
    
    # Round to one decimal place
    percentile_final = percentile_final.round(1)
    
    percentile_final.to_netcdf(final_path + f'era5_t2m_max_{years[0]}-{years[-1]}_p84.nc')
