import PAF_calculations as paf
import yaml
from pathlib import Path

config_file = Path(__file__).parent.parent / "examples" / "ERA5_default.yaml"
with open(config_file) as f:
    config = yaml.safe_load(f)
    

paf.CalculatePAF(
    wdir=config["wdir"],   # Working directory
    temp_dir=config["temp_dir"], # Temperature data directory
    project=config["project"], # Temperature data source
    scenario=config["scenario"],   # SSP scenario
    years=range(config["start_year"],config["end_year"]),  # Years range
    )