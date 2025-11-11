import utils

'''
This script calculates temperature zones and TMRELs (Theoretical Minimum Risk Exposure Levels) 
for a specific year based on ERA5 temperature data and GBD location data.
'''

# Path to main working directory
wdir = 'X:/user/liprandicn/mt-comparison/burkart2022/'


### ---------------------------------------------------------------------------------------------------
### Calculate temperature zones for a specific period

# Path to ERA5 t2m data and land-sea mask
# path_era5 = 'X:/user/liprandicn/Data/ERA5/'

# utils.calculate_temperature_zones(wdir, path_era5)


### ---------------------------------------------------------------------------------------------------
### Generate raster file with GBD level 3 locations shapefile

# utils.generate_raster_gbd_locations(wdir)

### ---------------------------------------------------------------------------------------------------
### Calculate TMRELs for a specific year

for year in [1990,2010,2020]:
    utils.generate_tmrels_rasters(wdir, year)