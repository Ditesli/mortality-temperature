import utils
from pathlib import Path
import yaml


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as f:
        return yaml.safe_load(f)

        
def main():
    config = load_config()
    wdir = config["wdir"]

    utils.GenerateTemperatureZones(wdir, config["path_era5"])
    utils.GenerateRasterGBDLocations(wdir)

    for year in [1990, 2000, 2020]:
        utils.GenerateTMRELsRasters(wdir, year)
        
        
if __name__ == "__main__":
    main()