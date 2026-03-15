
import utils
from pathlib import Path
import yaml


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as f:
        return yaml.safe_load(f)

        

def main():
    config = load_config()
    utils.calculate_optimal_temperature(
        temp_dir=config["temp_dir"],
        out_dir=config["out_dir"],
        years=range(config["start_year"],config["end_year"]), 
        step=config["step"],
        temp_type="max",
        percentile=config["percentile"]
    )
        
        


if __name__ == "__main__":
    main()