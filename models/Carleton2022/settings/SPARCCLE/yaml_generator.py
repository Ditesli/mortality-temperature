import os
import yaml


def generate_config_yaml(scenario):


    folder_script = os.path.dirname(os.path.abspath(__file__))
    filename = f"{scenario}.yaml"
    path = os.path.join(folder_script, filename)

    # Define data
    data = {
        "wdir": "X:/user/liprandicn/Projects/mt-comparison/models/carleton2022",
        "temp_dir": f"X:/user/dekkerm/IMAGE_environments/IMPACTS/Z_Emulator_Standalone_Tool/{scenario}/netcdf",
        "gdp_dir": f"X:/user/dekkerm/IMAGE_environments/IMPACTS/2_TIMER/outputlib/TIMER_3_5/IMPACTS/{scenario}/indicators/Economy/GDPpc_incl_impacts.out",
        "start_year": 2000,
        "end_year": 2100,
        "project": "SPARCCLE",
        "scenario": scenario,
        "adaptation": True,
        "counterfactual": True,
        "draw": "mean",
        "reporting_tool": "X:/user/dekkerm/IMAGE_environments/IMPACTS/7_Reporting_Tool/outxlsx",
    }

    # Save YAML
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(
            data, file, default_flow_style=False, sort_keys=False
        )

    print(f"{scenario}.yaml file generated")


scenarios = [
    "SSP2_M_CP_ERA_NoImpacts",
    "SSP2_M_CP_ERA_AllImpacts",
    "SSP2_M_CP_ERA_NoEcon",
    "SSP2_M_CP_Default_NoEcon",
    "SSP3_H_STS3_AllImpacts",
    "SSP3_H_ERA_AllImpacts",
    "SSP3_H_STS3_NoEcon",
    "SSP3_H_ERA_NoEcon",
    "SSP1_M_CP_ERA_AllImpacts",
    "SSP1_M_CP_ERA_NoImpacts",
    "SSP1_M_CP_ERA_NoEcon",
    "SSP1_ML_ERA_NoImpacts",
    "SSP2_ML_ERA_NoImpacts",
    "SSP1_ML_ERA_AllImpacts",
    "SSP2_ML_ERA_AllImpacts",
    "SSP1_ML_ERA_NoEcon",
    "SSP2_ML_ERA_NoEcon",
    "SSP1_VLLO_ERA_NoImpacts",
    "SSP2_VLLO_ERA_NoImpacts",
    "SSP1_VLLO_STS1_AllImpacts",
    "SSP1_VLLO_ERA_AllImpacts",
    "SSP2_VLLO_STS1_AllImpacts",
    "SSP2_VLLO_ERA_AllImpacts",
    "SSP1_VLLO_ERA_NoEcon"
    ]


for scenario in scenarios:
    generate_config_yaml(scenario)