import mortality_functions as mf
import yaml
from pathlib import Path

config_file = Path(__file__).parent / "config.yaml"
with open(config_file) as f:
    config = yaml.safe_load(f)


mf.CalculatePAF(
    wdir=config["wdir"],   # Working directory
    temp_dir=config["temp_dir"], # ERA5 directory
    project=config["project"],
    scenario=config["scenario"],   # Scenario
    years=range(config["start_year"], config["end_year"]),  #  Years range
    regions=config["region_class"],   # Region classification 
    optimal_range=config["optimal_range"],
    extrap_erf=config["extrap_erf"],   # Extrapolate ERF(s)
    temp_max=config["temp_max"],   # Minimum temperature to extrapolate to
    )