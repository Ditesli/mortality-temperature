import mortality_functions as mf


### --------------------------------------------------------------------------------------
### Define paths and variables

# Working directory (root)
wdir = ""

# Climate data path (use ERA5 data 1980-2024 or put path to monthly statistics files)
era5_dir = ""
temp_dir = ""

# GDP data path (use Carleton GDP data or put path to GDP data files)
gdp_dir = ""
# "X:\\user\\harmsenm\\Projects\\PRISMA\\PRISMA53"

# Set years range
years = []

# Define project name and scenario
project = ""
scenario = "" #  SSP#_carleton OR SSP#_ERA5 OR IMAGE-scenario-name

# Define whether to run with adaptation or not
adaptation = True



### --------------------------------------------------------------------------------------
### Run main model

mf.CalculateMortality(wdir, # Working directory
                      years, # Years range
                      gdp_dir, # Path to climate data files
                      gdp_dir, # Path to GDP data files
                      project, # Project name
                      scenario, # Scenario name
                      regions="IMAGE26", # "IMAGE26" or "countries"
                      adaptation=adaptation, # Adaptation on or off
                      IAM_format=True # If True, use IAM output format
                      )
