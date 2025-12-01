import mortality_functions as mf


### --------------------------------------------------------------------------------------""
### Define paths and variables

# Working directory (root)
wdir = "X:/user/liprandicn/mt-comparison/carleton2022/"

# Climate data type and path
temp_source = "MS" # ERA5, MS (monthly statistics)

# temp_dir = "X:/user/liprandicn/Data/ERA5/t2m_daily/"
temp_dir = "X:/user/scherrenbm/ModelDevelopment/IMAGE_Development/IMAGE_Daily_Indicators/SSP2/netcdf/"

# Set years range
years = range(2020,2025,1)

# Define SSP scenario
ssp = "SSP2" # Socioeconomic scenarios: "SSP1", "SSP2", "SSP3", "SSP5"

# Define region definitions
regions = "IMAGE26" #  "IMAGE26" or "countries"

# Turn adaptation on or off 
adaptation = {"tmean": temp_dir, "loggdppc": "test"}#None #{"tmean": temp_dir, "loggdppc": "default"} 
# {"tmean": "default" or "temp_dir", "loggdppc": "default" or "/path/to/gdp-loop/file.csv"} OR None (No adaptation)


### --------------------------------------------------------------------------------------
### Run main model

era5 = mf.calculate_mortality(wdir, # Working directory
                             years, # Years range
                             temp_source, # Climate data type
                             temp_dir, # Path to climate data files
                             ssp, # Socioeconomic scenarios
                             regions, # Region classification
                             adaptation=adaptation, # Adaptation on or off
                             IAM_format=False # If True, use IAM output format
                             )