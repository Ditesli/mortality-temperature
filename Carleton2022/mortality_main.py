import mortality_functions as mf


### --------------------------------------------------------------------------------------
### Define paths and variables

# Working directory (root)
wdir = 'X:\\user\\liprandicn\\mt-comparison\\Carleton2022'

# Climate data type and path
climate_type = 'ERA5' # ERA5, CMIP6, AR6
climate_path = 'X:\\user\\liprandicn\\Data\\ERA5\\t2m_daily'
# climate_path =  D:\\Climate Models - Bias Corrected from CMIP6 precalculated data

# Set years range
years = range(2010,2012,1)

# Define scenarios
scenarios_RCP = [] #['SSP126', 'SSP245', 'SSP370', 'SSP585'] or []
scenarios_SSP = ['SSP2'] #['SSP1', 'SSP2', 'SSP3', 'SSP5']

# Define region definitions
regions = 'IMAGE26' #  impact_regions, ISO3, gbd_level3, uN_M49_level1,	IMAGE26, continents



### --------------------------------------------------------------------------------------
### Run main model
era5 = mf.mortality_scenario(wdir, # Working directory
                             years, # Years range
                             climate_type, 
                             climate_path, 
                             scenarios_SSP, 
                             scenarios_RCP,
                             regions,
                             IAM_format=True, # If True, use IAM output format
                             )