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

    utils.calculate_temperature_zones(wdir, config["path_era5"])
    utils.generate_raster_gbd_locations(wdir)

    for year in [1990, 2000, 2020]:
        utils.generate_tmrels_rasters(wdir, year)
        
        
if __name__ == "__main__":
    main()