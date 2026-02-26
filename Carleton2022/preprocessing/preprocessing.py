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
    print(wdir)

    tasks = [
        lambda: utils.RegionClassificationFile(wdir, config["regions_class"]),
        lambda: utils.PopulationHistorical(wdir, config["landscan_path"]),
        lambda: utils.PopulationProjections(wdir, config["pop_dir"]),
        lambda: utils.DailyTemperaturesERA5PresentDay(wdir, config["era5_dir"]),
        lambda: utils.ClimatologiesERA5(wdir, config["era5_dir"], range(2000, 2026)),
    ]

    for task in tasks:
        task()
        
        
if __name__ == "__main__":
    main()
