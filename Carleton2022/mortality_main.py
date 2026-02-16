import mortality_functions as mf
import yaml
from pathlib import Path

config_file = Path(__file__).parent / "config.yaml"
with open(config_file) as f:
    config = yaml.safe_load(f)
    

mf.CalculateMortality(wdir=config["wdir"], # Working directory
                    years=range(config["start_year"], config["end_year"]), # Years range
                    temp_dir=config["temp_dir"], # Path to climate data files
                    gdp_dir=config["gdp_dir"], # Path to GDP data files
                    project=config["project"], # Project name
                    scenario=config["scenario"], # Scenario name
                    regions="IMAGE26", # "IMAGE26" or "countries"
                    adaptation=config["adaptation"], # Adaptation on or off
                    IAM_format=True # If True, use IAM output format
                    )