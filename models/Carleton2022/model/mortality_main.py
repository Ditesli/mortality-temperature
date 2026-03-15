import mortality_functions as mf
import yaml
from pathlib import Path

config_file = Path(__file__).parent / "config.yaml"
with open(config_file) as f:
    config = yaml.safe_load(f)
    

# mf.CalculateMortality(
#     wdir=config["wdir"], # Working directory
#     years=range(config["start_year"], config["end_year"]), # Years range
#     temp_dir=config["temp_dir"], # Path to climate data files
#     gdp_dir=config["gdp_dir"], # Path to GDP data files
#     project=config["project"], # Project name
#     scenario=config["scenario"], # Scenario name
#     regions=config["regions"],  # "IMAGE26" or "countries"
#     adaptation=config["adaptation"], # Adaptation on or off
#     reporting_tool=config["reporting_tool"], # Report on or off
# )

# ssp_s = ["SSP2_M", "SSP3_H", "SSP5_H"]

# for ssp in ssp_s:
#     mf.CalculateMortality(
#         wdir="X:\\user\\liprandicn\\mt-comparison\\carleton2022\\", # Working directory
#         years=range(2000,2101), # Years range
#         temp_dir=f"X:/user/dekkerm/IMAGE_environments/IMPACTS/3_IMAGE_land/scen/{ssp}/netcdf/", # Path to climate data files
#         gdp_dir=None, # Path to GDP data files
#         project="default", # Project name
#         scenario=f"{ssp}_Carleton", # Scenario name
#         regions="IMAGE26",  # "IMAGE26" or "countries"
#         adaptation=True, # Adaptation on or off
#         reporting_tool=False # Report on or off
# )


ssp_s = ["IMP-SSP2-REF-noGDPIMP-15", "IMP-SSP3-REF-noGDPIMP-15", "IMP-SSP2-REF-GDPIMP-15", "IMP-SSP3-REF-GDPIMP-15", "IMP-SSP2-REF-noGDPIMP-CP"]

for ssp in ssp_s:
    mf.CalculateMortality(
        wdir="X:\\user\\liprandicn\\mt-comparison\\carleton2022\\", # Working directory
        years=range(2000,2101), # Years range
        temp_dir=f"X:/user/dekkerm/IMAGE_environments/", # Path to climate data files
        gdp_dir="X:/user/dekkerm/IMAGE_environments/", # Path to GDP data files
        project="IMPACTS", # Project name
        scenario=f"{ssp}", # Scenario name
        regions="IMAGE26",  # "IMAGE26" or "countries"
        adaptation=True, # Adaptation on or off
        reporting_tool=True # Report on or off
)