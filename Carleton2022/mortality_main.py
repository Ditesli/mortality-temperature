import mortality_functions as mf


### --------------------------------------------------------------------------------------
### Define paths and variables

# Working directory (root)
wdir = 'X:/user/liprandicn/mt-comparison/Carleton2022'

# Climate data type and path
climate_type = 'ERA5' # ERA5, AR6
climate_path = 'X:/user/liprandicn/Data/ERA5/t2m_daily'
# climate_path =  D:\\Climate Models - Bias Corrected from CMIP6 precalculated data

# Set years range
years = range(2000,2019,1)

# Define scenarios
scenarios_RCP = [] # Climate scenarios: ['SSP126', 'SSP245', 'SSP370', 'SSP585'] 
                    # or [] if present day climate data
scenarios_SSP = ['SSP2'] # Socioeconomic scenarios: ['SSP1', 'SSP2', 'SSP3', 'SSP5']

# Define region definitions
regions = 'gbd_level3' #  impact_regions, ISO3, gbd_level3, UN_M49_level1,	IMAGE26, continents


### --------------------------------------------------------------------------------------
### Run main model

era5 = mf.calculate_mortality(wdir, # Working directory
                             years, # Years range
                             climate_type, # Climate data type
                             climate_path, # Path to climate data files
                             scenarios_SSP, # Socioeconomic scenarios
                             scenarios_RCP, # Climate scenarios if projections are done
                             regions, # Region classification
                             IAM_format=False, # If True, use IAM output format
                             )