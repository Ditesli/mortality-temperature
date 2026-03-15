import numpy as np
import xarray as xr



def calculate_optimal_temperature(
      temp_dir: str, 
      out_dir:str,  
      years: np.ndarray, 
      step: int,
      temp_type: str,
      percentile: float
      ) -> None:
    
    """
    Calculate the optimal temperature defined in Honda et al., (2014) as the 
    83.6th percentile of daily mean temperature from ERA5 reanalysis data.

    Parameters:
    - temp_dir: str
          Path to the directory containing the ERA5 data files at daily temporal resolution
          and spatial resolution of 0.25 degrees (720x1440 dimension). Files are renamed 
          following the format: 'era5_t2m_mean_day_<start_year>_<final_year>.nc'
    - out_dir: str
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
            data_name = f'era5_t2m_{temp_type}_day_{year}.nc'
            with xr.open_dataset(temp_dir + data_name) as data:
                data = data['t2m'].isel(latitude=slice(lat, lat + step))
                temporal_lat_data.append(data.load())
            
        # Concatenate the data for the current latitudes across all years
        temporal_data = xr.concat(temporal_lat_data, dim='valid_time')
        # Calculate the percentile for the latitude band and append it to the list
        percentile_band = temporal_data.quantile(percentile, dim='valid_time')
        percentile_bands.append(percentile_band)
        
    (
      xr.concat(percentile_bands, dim='latitude')
     .rename(f't2m_p{np.round(percentile*100,0)}')
     .assign_coords(longitude=lambda x: ((x.longitude + 180) % 360 - 180))
     .sortby("longitude") # Shift longitude coordinates to -180 - 180 range
     .drop_vars('quantile') # Drop quantile coordinate
     .pipe(lambda x: x - 273.15) # Convert to Celsius
     .round(1)
     .to_netcdf(out_dir + 
                     f'era5_t2m_{temp_type}_{years[0]}-{years[-1]}_p{np.round(percentile*100,0)}.nc')
    )