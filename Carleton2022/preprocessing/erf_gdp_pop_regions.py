import utils

"""
Preprocessing file that generates relevant data for the mortality calculation.
The code uses data downloaded from the supplementary material of Carleton et al., (2022).
This data has the same folder hierarchy as the original data but only the relevant data 
files are kept.
The calculations done in this script are:
- Generate a smaller file with the covariates for the no adpatation case.
- Generate population and gdp projections at impact region level as csv files.
- Generate file that contains the region classification.
- Generate files with historical population for the years not included in the s.m.
  the approach is the same as in the paper described in the appendix B.3.3.
"""


### ------------------------------------------------------------------------------

# Path to working directory where all data is stored
wdir = "X:/user/liprandicn/mt-comparison/carleton2022/data"
# Path to IMAGE regions classification folder produced manually
regions_file = "X:/user/liprandicn/mt-comparison/regions_comparisson.csv"
# Open LandScan population raster 
landscan_file = f"{wdir}"+"/LandScan_Global/landscan-global-2000-assets/landscan-global-2000.tif"
# Open impact regions shapefile
impact_regions = f"{wdir}"+"/carleton_sm/ir_shp/impact-region.shp"


### ------------------------------------------------------------------------------

"""
Generate Exposure Response Functions without Adaptation using covaraiates from 
Carleton et al. (2022) and the Tmin (temperature at which the response functions is
minimized per impact region).
"""

utils.condense_erf_files(wdir)


### ----------------------------------------------------------------------
""" 
Generate files per scenario and age group for all impact regions 
"""

utils.gdp_pop_ssp_projections(wdir)
    
    
### ----------------------------------------------------------------------
""" 
Generate file that contains impact region codes, names and their corresponding 
IMAGE and GBD region
"""

utils.region_classification_file(wdir, regions_file)

    
### ----------------------------------------------------------------------
""" 
Generate files that contains historical population per impact region
"""

utils.generate_historical_pop(wdir, landscan_file, impact_regions)


### ----------------------------------------------------------------------
"""
  Generate files with ERA5 daily temperature data at the impact region 
level from 2000 to 2010, defined by Carleton et al as T_0 (present day)
temeprature levels and the baseline of the analysis.
"""
  
wdir = "X:/user/liprandicn/mt-comparison/carleton2022/"
era5_dir = "X:/user/liprandicn/Data/ERA5/t2m_daily/"

utils.import_present_day_temperature(wdir, era5_dir)


### ---------------------------------------------------------------------
"""
Generate GDP relationship for downscaling national GDP to impact region level
"""

utils.generate_gdp_relationship(wdir)



