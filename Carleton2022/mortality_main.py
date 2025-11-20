import mortality_functions as mf


### --------------------------------------------------------------------------------------
### Define paths and variables

# Working directory (root)
wdir = 'X:/user/liprandicn/mt-comparison/Carleton2022/'

# Climate data type and path
temp_source = 'MS' # ERA5, MS, AR6

# temp_dir = 'X:/user/liprandicn/Data/ERA5/t2m_daily/'
temp_dir = 'X:/user/scherrenbm/ModelDevelopment/IMAGE_Development/IMAGE_Daily_Indicators/SSP2/netcdf/'

# Set years range
years = range(2050,2051,1)

# Define SSP scenario
SSP = 'SSP2' # Socioeconomic scenarios: 'SSP1', 'SSP2', 'SSP3', 'SSP5'

# Define region definitions
regions = 'countries' #  IMAGE26 or 'countries'

# Turn on or off adaptation
adaptation = True # True or False


### --------------------------------------------------------------------------------------
### Run main model

era5 = mf.calculate_mortality(wdir, # Working directory
                             years, # Years range
                             temp_source, # Climate data type
                             temp_dir, # Path to climate data files
                             SSP, # Socioeconomic scenarios
                             regions, # Region classification
                             adaptation=adaptation, # Adaptation on or off
                             IAM_format=False # If True, use IAM output format
                             )