'''
Calculate the Temperature zones (tz) per each grid cell, according to GBD methodology by calculating the mean from 1980-2019
Follwing their methods, data was retreived from the ERA5 website: 
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means
Sea values were set to NaN using the mask provided by ERA5:
https://confluence.ecmwf.int/pages/viewpage.action?pageId=140385202#ERA5Land:datadocumentation-parameterlistingParameterlistings

'''

import numpy as np
import xarray as xr
from new_health_module.read_files import get_all_population_data


### ------------------------------------------------------------------------------------------
# Open and preprocess population data and set mask for ALL cells with population

pop_all_ssp, mask_positive_pop = get_all_population_data()


### ------------------------------------------------------------------------------------------
# Open and preprocess ERA5 t2m data

### Path to ERA5 t2m data
era5_t2m = xr.open_dataset('X:\\user\\liprandicn\\Data\\ERA5\\ERA5_t2m_mean_1980-2019.nc') 

### Calculate 1980-2019 mean
era5_t2m_mean = era5_t2m.mean(dim='valid_time')

### Convert to Celius and round to integers
era5_t2m_mean = era5_t2m_mean - 273.15

### Shift coordinates
era5_t2m_mean = era5_t2m_mean.assign_coords(longitude=((era5_t2m_mean.coords['longitude'] + 180) % 360 - 180)).sortby("longitude")

### Interpolate to align temperature and population grids
era5_t2m_mean = era5_t2m_mean.interp(longitude=pop_all_ssp.longitude, latitude=pop_all_ssp.latitude, method="nearest")

### Discard dummy var
era5_t2m_mean = era5_t2m_mean.drop_vars('number')


### ------------------------------------------------------------------------------------------
# Open and preprocess ERA5 land-sea data

### Parh to ERA5 land-sea mask
era5_land_sea = xr.open_dataset('X:\\user\\liprandicn\\Data\\ERA5\\lsm_1279l4_0.1x0.1.grb_v4_unpack.nc') 

### Delete single time value and rename lsm to t2m to keep naming
era5_land_sea = era5_land_sea.drop_vars('time')
era5_land_sea = era5_land_sea.rename({'lsm': 't2m'})

### Shift coordinates
era5_land_sea = era5_land_sea.assign_coords(longitude=((era5_land_sea.coords['longitude'] + 180) % 360 - 180)).sortby("longitude")

### Interpolate land-sea data to 0.25 degree resolution to match population grids
era5_land_sea_15min = era5_land_sea.interp(longitude=pop_all_ssp.longitude, latitude=pop_all_ssp.latitude, method="nearest")


### ------------------------------------------------------------------------------------------
### Asign NaN values to sea cells and Antarctica, and ensure that ANY cell (of any scenario and year) 
### with POP data has associated a temperature zone

### Set sea cells and no-pop cells to NaN
era5_t2m_mean_land_pop = xr.where((mask_positive_pop.GPOP) | (era5_land_sea_15min.t2m > 0), era5_t2m_mean, np.nan)
### Set Antarctica to NaN
era5_wo_antarctica = era5_t2m_mean_land_pop.where(era5_t2m_mean_land_pop.latitude >= -60, np.nan)


### ------------------------------------------------------------------------------------------
# Create temperature zones by rounding and clipping temperature values and save file

### Round data to get temperature zones
era5_rounded = era5_wo_antarctica.where(era5_wo_antarctica.isnull(), era5_wo_antarctica.round())

### Clip tz to 6 to 28 values
temp_zone_rounded = np.clip(era5_rounded.t2m, 6, 28)

### Remove time dimension
temp_zone_rounded=temp_zone_rounded.mean(dim='time')

### Save in project folder
temp_zone_rounded.to_netcdf('X:\\user\\liprandicn\\Health Impacts Model\\ClimateData\\ERA5\\ERA5_mean_1980-2019_land_t2m_tz.nc')