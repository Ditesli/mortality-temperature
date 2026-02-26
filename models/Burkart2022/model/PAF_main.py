import PAF_calculations as paf
import yaml
from pathlib import Path

config_file = Path(__file__).parent / "config.yaml"
with open(config_file) as f:
    config = yaml.safe_load(f)
    

paf.run_main(wdir=config["wdir"],   # Working directory
             temp_dir=config["temp_dir"], # Temperature data directory
             temp_source=config["temp_source"], # Temperature data source
             ssp=config["ssp"],   # SSP scenario
             years=range(config["start_year"],config["end_year"]),  #  Years range
             region_class=config["region_class"],   # Region classification 
             draw_type=config["draw_type"],  # Mean RR or specific/random draw
             single_erf=config["single_erf"],   # Single ERF or use temperature zones
             extrap_erf=config["extrap_erf"]   # Extrapolate ERF(s)
             )

