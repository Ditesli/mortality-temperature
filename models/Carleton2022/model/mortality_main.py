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


for scenario in ["IMP-SSP2-REF-noGDPIMP-15", "IMP-SSP3-REF-noGDPIMP-15", "IMP-SSP5-REF-noGDPIMP-15", "IMP-SSP1-REF-GDPIMP-15", "IMP-SSP2-REF-GDPIMP-15"]:
    mf.CalculateMortality(
        wdir="X:/user/liprandicn/mt-comparison/carleton2022/",
        years=range(2000,2101),
        temp_dir="X:\\user\\dekkerm\\IMAGE_environments",
        gdp_dir="X:\\user\\dekkerm\\IMAGE_environments",
        project="IMPACTS",
        scenario=scenario,
        regions="IMAGE26",
        adaptation=True,
        reporting_tool=True
    )