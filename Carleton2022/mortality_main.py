import mortality_functions as mf


### --------------------------------------------------------------------------------------""
### Define paths and variables

# Working directory (root)
wdir = ""

# Climate data path 
era5_dir = ""
temp_dir = ""

# Set years range
years = []

# Define SSP scenario
scenario = "" #  SSP#_carleton OR SSP#_ERA5 OR other

# Define region definitions
regions = "" #  "IMAGE26" or "countries"

# Turn adaptation on or off (and define path to GDPpc data if on)
# {"climtas": "temp_dir", "loggdppc": "default" or "/path/to/project/folder"} OR None (Default No adaptation)
adaptation = ""
            


### --------------------------------------------------------------------------------------
### Run main model

era5 = mf.CalculateMortality(wdir, # Working directory
                             years, # Years range
                             temp_dir, # Path to climate data files
                             scenario, # Scenario name
                             regions, # Region classification
                             adaptation=adaptation, # Adaptation on or off
                             IAM_format=True # If True, use IAM output format
                             )