'''
Using ISIMIP "dummy" data provided by the land team:
1. Calculate the 30 year everage for the beggining and end of the century for the RCP8.5 scenario 
    for 5 models: GFDL, IPSL, MRI, MPI, UKESM.
2. Calculate the difference between the beginning and end of the century for each model (\Delta T_{GCM}).
3. Calculate the ensemble mean
4. Downscale to 0.25 degree resolution and save in project folder

'''
import numpy as np
import xarray as xr

### ------------------------------------------------------------------------------------------------------------
### Bring 30-year climate data from ISIMIP RCP8.5 scenario for the beginning and end of the century for 5 models 

models = ['GFDL', 'IPSL', 'MRI', 'MPI', 'UKESM']
models_diff = []; models_0 = []; models_1 = []
variable_name= 'tas'

for model in models:
    nc_file_0 = xr.open_dataset(f'X:/user\\krugerc\\code\\ClimateData\\output\\{model}/30_year_average_ssp585_{variable_name}_0.nc')
    nc_file_0 = nc_file_0.mean(dim='time')
    models_0.append(nc_file_0)
    nc_file_1 = xr.open_dataset(f'X:/user\\krugerc\\code\\ClimateData\\output\\{model}/30_year_average_ssp585_{variable_name}_1.nc')
    nc_file_1 = nc_file_1.mean(dim='time')
    models_1.append(nc_file_1)
    
    ### Calculate the difference between the beginning and end of the century for each model
    models_diff.append(nc_file_1 - nc_file_0) 
    
    
### ------------------------------------------------------------------------------------------------------------
### Calculate model mean

variables_diff = [ds[variable_name] for ds in models_diff]
variables_0 = [ds[variable_name] for ds in models_0]
variables_1 = [ds[variable_name] for ds in models_1]

ensemble_diff = xr.concat(variables_diff, dim='model')
ensemble_0 = xr.concat(variables_0, dim='model')
ensemble_1 = xr.concat(variables_1, dim='model')

ensemble_diff_mean = ensemble_diff.mean(dim='model')
ensemble_0_mean = ensemble_0.mean(dim='model')
ensemble_1_mean = ensemble_1.mean(dim='model')


### ------------------------------------------------------------------------------------------------------------
### Simple downscaling using nearest neighbor interpolation

latitude_15min = np.arange(89.75, -90, -0.25)
longitude_15min = np.arange(-179.75, 180, 0.25)

ensemble_diff_mean_15min = ensemble_diff_mean.interp(longitude=longitude_15min, latitude=latitude_15min)#, method="nearest")
ensemble_0_mean_15min = ensemble_0_mean.interp(longitude=longitude_15min, latitude=latitude_15min)#, method="nearest")
ensemble_1_mean_15min = ensemble_1_mean.interp(longitude=longitude_15min, latitude=latitude_15min)#, method="nearest")


### ------------------------------------------------------------------------------------------------------------
### Use land-sea mask to set sea cells to NaN. Map retrieved from ERA5 datasets: 
### https://confluence.ecmwf.int/pages/viewpage.action?pageId=140385202#ERA5Land:datadocumentation-parameterlistingParameterlistings

era5_land_sea = xr.open_dataset('X:\\user\\liprandicn\\Health Module\\ClimateData\\ERA5\\lsm_1279l4_0.1x0.1.grb_v4_unpack.nc')

# Delete single time value and rename lsm to tas to keep naming
era5_land_sea = era5_land_sea.drop_vars('time')
era5_land_sea = era5_land_sea.rename({'lsm': 'tas'})

# Rearrange lon coordinate
era5_land_sea.coords['longitude'] = (era5_land_sea.coords['longitude'] + 180) % 360 - 180
era5_land_sea = era5_land_sea.sortby('longitude')

# Interpolate to 0.25 degree resolution to match grids
era5_land_sea_15min = era5_land_sea.interp(longitude=longitude_15min, latitude=latitude_15min, method="nearest")

# Asign NaN values to sea cells
ensemble_diff_mean_15min_land = xr.where(era5_land_sea_15min > 0, ensemble_diff_mean_15min, np.nan)
ensemble_0_mean_15min_land = xr.where(era5_land_sea_15min > 0, ensemble_0_mean_15min, np.nan)
ensemble_1_mean_15min_land = xr.where(era5_land_sea_15min > 0, ensemble_1_mean_15min, np.nan)

### ------------------------------------------------------------------------------------------------------------
### Regrid to match POP map 

pop = xr.open_dataset('X:\\user\\doelmanj\\ScenDevelopment\\ScenarioMIP\\3_IMAGE_land\\scen\\SSP5_H\\netcdf\\GPOP.nc')
pop_coarse = pop.coarsen(latitude=3, longitude=3, boundary='pad').sum(skipna=False)

ensemble_diff_mean_15min_land = ensemble_diff_mean_15min_land.interp(longitude=pop_coarse.longitude, latitude=pop_coarse.latitude)
ensemble_0_mean_15min_land = ensemble_0_mean_15min_land.interp(longitude=pop_coarse.longitude, latitude=pop_coarse.latitude)
ensemble_1_mean_15min_land = ensemble_1_mean_15min_land.interp(longitude=pop_coarse.longitude, latitude=pop_coarse.latitude)


### ------------------------------------------------------------------------------------------------------------
### Save in personal folder

ensemble_diff_mean_15min_land.to_netcdf(f'X:\\user\\liprandicn\\Health Module\\ClimateData\\GCM\\ensemble_mean_ssp585_tas_15min_diff.nc')
ensemble_0_mean_15min_land.to_netcdf(f'X:\\user\\liprandicn\\Health Module\\ClimateData\\GCM\\ensemble_mean_ssp585_tas_15min_start.nc')
ensemble_1_mean_15min_land.to_netcdf(f'X:\\user\\liprandicn\\Health Module\\ClimateData\\GCM\\ensemble_mean_ssp585_tas_15min_end.nc')