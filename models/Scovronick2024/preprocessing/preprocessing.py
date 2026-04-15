import utils
from pathlib import Path
import yaml


def load_config():
    config_path = Path(__file__).parent.parent / "examples" / "config_preprocessing.yaml"
    with config_path.open() as f:
        return yaml.safe_load(f)

        
def main():
    config = load_config()
    wdir = config["wdir"]

    utils.ERA5TemperaturePercentiles(wdir, config["path_era5"], years=range(1980,2010))
    
        
if __name__ == "__main__":
    main()