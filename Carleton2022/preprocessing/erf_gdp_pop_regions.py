import utils

"""
Preprocessing file that generates relevant data for the mortality calculation.
The code uses data downloaded from the supplementary material of Carleton et al., (2022).
This data has the same folder hierarchy as the original data but only the relevant data 
files are kept.
The calculations done in this script are:
- Generate file that contains the region classification.
- Generate files with historical population for the years not included in the s.m.
  the approach is the same as in the paper described in the appendix B.3.3.
- Generate files with the daily temperature for 2000-2010 at the impact region level.
"""


### ------------------------------------------------------------------------------

# Path to working directory where all data is stored
wdir = "X:/user/liprandicn/mt-comparison/carleton2022/data/"

# Path to ERA5 daily temeprature data
era5_dir = "X:/user/liprandicn/DATA/ERA5/t2m_daily/"

# Path to IMAGE regions classification folder produced manually
regions_file = "X:/user/liprandicn/mt-comparison/regions_comparisson.csv"

# Open LandScan population raster 
landscan_file = "X:/user/liprandicn/DATA/POPULATION/LandScan_Global/"

# Open impact regions shapefile
impact_regions = f"{wdir}"+"carleton_sm/ir_shp/impact-region.shp"

# Path to population projections from IMAGE nc files
pop_dir = "X:/user/liprandicn/DATA/POPULATION"


    
### ----------------------------------------------------------------------
""" 
Generate file that contains impact region codes, names and their corresponding 
IMAGE and GBD region
"""

utils.RegionClassificationFile(wdir, regions_file)

    
### ----------------------------------------------------------------------
""" 
Generate files that contains historical population per impact region
"""

utils.PopulationHistorical(wdir, landscan_file, impact_regions)


### ----------------------------------------------------------------------
"""
Generate population projection files from IMAGE nc files
"""
utils.PopulationProjections(wdir, pop_dir)


### ----------------------------------------------------------------------
"""
Generate files with ERA5 daily temperature data at the impact region 
level from 2000 to 2010, defined by Carleton et al as T_0 (present day)
temeprature levels and the baseline of the analysis.
"""

utils.DailyTemperaturesERA5PresentDay(wdir, era5_dir)


### ----------------------------------------------------------------------
"""
Generate files with ERA5 climatology (defined here as the 30-year running mean) 
at the impact region level. This is used to generate Exposure Response Functions
(ERFs) with adaptation.
"""

utils.ClimatologiesERA5(wdir, era5_dir, range(2000,2026))

