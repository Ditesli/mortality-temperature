import PAF_calculations as paf
import yaml
from pathlib import Path

config_file = Path(__file__).parent / "config.yaml"
with open(config_file) as f:
    config = yaml.safe_load(f)
    

paf.CalculatePAF(wdir=config["wdir"],   # Working directory
                 temp_dir=config["temp_dir"], # Temperature data directory
                 project=config["project"], # Temperature data source
                 scenario=config["scenario"],   # SSP scenario
                 years=range(config["start_year"],config["end_year"]),  # Years range
                 regions=config["region_class"],   # Region classification 
                 draw=config["draw_type"],  # Mean RR or specific/random draw
                 single_erf=config["single_erf"],   # Single ERF or use temperature zones
                 extrap_erf=config["extrap_erf"]   # Extrapolate ERF(s)
                 )

