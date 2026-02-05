import utils

'''
This script calculates temperature zones following Burkart et al. methodology.
The script also generates raster files for the GBD locations at level 3 using the shapefile 
available at the GBD repository: 
and TMRELs (Theoretical Minimum Risk Exposure Levels) as raster files.
'''

# Path to main working directory
wdir = 'X:/user/liprandicn/mt-comparison/burkart2022/'

# Path to ERA5 t2m data and land-sea mask
path_era5 = 'X:/user/liprandicn/Data/ERA5/'


### ---------------------------------------------------------------------------------------------------
### Calculate temperature zones for a specific period

utils.calculate_temperature_zones(wdir, path_era5)


### ---------------------------------------------------------------------------------------------------
### Generate raster file with GBD level 3 locations shapefile

utils.generate_raster_gbd_locations(wdir)

### ---------------------------------------------------------------------------------------------------
### Calculate TMRELs for a specific year

for year in [1990,2010,2020]:
    utils.generate_tmrels_rasters(wdir, year)