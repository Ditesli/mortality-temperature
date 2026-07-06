import mortality_functions as mf
import yaml
from pathlib import Path

from pyinstrument import Profiler


scenarios = [
    # "SSP3_H_STS3_AllImpacts",
    # "SSP3_H_ERA_AllImpacts",
    # "SSP3_H_STS3_NoEcon",
    # "SSP3_H_ERA_NoEcon",
    # "SSP1_M_CP_ERA_AllImpacts",
    # "SSP1_M_CP_ERA_NoImpacts",
    # "SSP1_M_CP_ERA_NoEcon",
    # "SSP1_ML_ERA_NoImpacts",
    # "SSP2_ML_ERA_NoImpacts",
    # "SSP1_ML_ERA_AllImpacts",
    # "SSP2_ML_ERA_AllImpacts",
    # "SSP1_ML_ERA_NoEcon",
    # "SSP2_ML_ERA_NoEcon",
    # "SSP2_VLLO_ERA_NoImpacts",
    # "SSP1_VLLO_STS1_AllImpacts",
    # "SSP2_VLLO_ERA_AllImpacts",
    # "SSP1_VLLO_ERA_NoEcon"
    ]

# for scenario in scenarios:
    
config_file = Path(__file__).parent.parent / "settings" / f"SPARCCLE/test.yaml"
with open(config_file) as f:
    config = yaml.safe_load(f)

profiler = Profiler()
profiler.start()
    
mf.CalculateMortality(
    wdir=config["wdir"], # Working directory
    years=range(config["start_year"], config["end_year"]), # Years range
    temp_dir=config["temp_dir"],#+f"{clim_var[i]}", # Path to climate data files
    gdp_dir=config["gdp_dir"], # Path to GDP data files
    project=config["project"],#+f"{i}", # Project name
    scenario=config["scenario"], # Scenario name
    adaptation=config["adaptation"], # Adaptation on or off
    counterfactual=config["counterfactual"], # Counterfactual climate scenario
    draw=config["draw"], # Mean or specific/random draw
    reporting_tool=config["reporting_tool"], # Report on or off
)

profiler.stop()
profiler.write_html("profile.html")