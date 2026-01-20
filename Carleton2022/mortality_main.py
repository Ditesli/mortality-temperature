import mortality_functions as mf


### --------------------------------------------------------------------------------------""
### Define paths and variables

# Working directory (root)
wdir = "X:/user/liprandicn/mt-comparison/carleton2022/"

# Climate data path (use ERA5 data 1980-2024 or put path to monthly statistics files)
era5_dir = "X:/user/liprandicn/Data/ERA5/t2m_daily/"
temp_dir = "X:/user/dekkerm/IMAGE_environments/IMPACTS/3_IMAGE_land/scen/SSP2/netcdf/"

# Set years range
years = range(2015,2020,1)

# Define SSP scenario
scenario = "SSP2_M_CP" #  SSP#_carleton OR SSP#_ERA5 OR other

# Define region definitions
regions = "IMAGE26" #  "IMAGE26" or "countries"

# Turn adaptation on or off 
# !!! I'll change this later to make it a more default option (as it reads any gdp data from TIMER)
# {"climtas": "temp_dir", "loggdppc": "default" or "/path/to/project/folder"} OR None (Default No adaptation)
adaptation = {"climtas": temp_dir, "loggdppc": "X:\\user\\harmsenm\\Projects\\PRISMA\\PRISMA53"}
            # {"climtas": temp_dir, "loggdppc": "X:/user/dekkerm/IMAGE_environments/IMPACTS"}



### --------------------------------------------------------------------------------------
### Run main model

era5 = mf.CalculateMortality(wdir, # Working directory
                             years, # Years range
                             temp_dir, # Path to climate data files
                             scenario, # Scenario (can only run 1)
                             regions, # Region classification
                             adaptation=adaptation, # Adaptation on or off
                             IAM_format=True # If True, use IAM output format
                             )