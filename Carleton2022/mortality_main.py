import mortality_functions as mf


### --------------------------------------------------------------------------------------""
### Define paths and variables

# Working directory (root)
wdir = "X:/user/liprandicn/mt-comparison/carleton2022/"

# Climate data type 
temp_source = "ERA5" # ERA5, MS (monthly statistics)

# Climate data path (use ERA5 data 1980-2024 or put path to monthly statistics files)
era5_dir = "X:/user/liprandicn/Data/ERA5/t2m_daily/"
temp_dir = "X:/user/scherrenbm/ModelDevelopment/IMAGE_Development/IMAGE_Daily_Indicators/SSP2/netcdf/"

# Set years range
years = range(2000,2025,1)

# Define SSP scenario
scenario = "SSP2_default" #  OR SSP1_default, SSP2_default, SSP3_default, SSP4_default, SSP5_default

# Define region definitions
regions = "IMAGE26" #  "IMAGE26" or "countries"

# Turn adaptation on or off 
# !!! Set here the path to the project for GDP
# !!! I'll change this later to make it a more default option (as it reads any gdp data from TIMER)
adaptation = {"tmean": temp_dir, "loggdppc": "default"}
            # {"tmean": temp_dir, "loggdppc": "X:\\user\\harmsenm\\Projects\\PRISMA\\PRISMA53"}
            # {"tmean": temp_dir, "loggdppc": "X:/user/dekkerm/IMAGE_environments/IMPACTS"}
# {"tmean": "temp_dir", "loggdppc": "default" or "/path/to/project/folder"} OR None (Default No adaptation)


### --------------------------------------------------------------------------------------
### Run main model

era5 = mf.CalculateMortality(wdir, # Working directory
                             years, # Years range
                             temp_source, # Climate data type
                             era5_dir, # Path to climate data files
                             scenario, # Scenario (can only run 1)
                             regions, # Region classification
                             adaptation=adaptation, # Adaptation on or off
                             IAM_format=True # If True, use IAM output format
                             )