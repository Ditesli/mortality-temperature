import os
import yaml


def generate_config_yaml(path, scenario, project):


    filename = f"{scenario}.yaml"
    path = os.path.join(path+f"\\{project}", filename)
    
    if "extra" in scenario:
        scenario_image_land = scenario[:-6]
    else:
        scenario_image_land=scenario

    # Define data
    data = {
        "wdir": "X:/user/liprandicn/Projects/mt-comparison/models/carleton2022",
        "temp_dir":f"X:\\user\\dekkerm\\IMAGE_environments\\IMPACTS\\3_IMAGE_land\\scen\\{scenario}\\netcdf",
            # f"X:/user/dekkerm/IMAGE_environments/IMPACTS/Z_Emulator_Standalone_Tool/{scenario_image_land}/netcdf",
        "gdp_dir": f"X:/user/dekkerm/IMAGE_environments/IMPACTS/2_TIMER/outputlib/TIMER_3_5/IMPACTS/{scenario[:9]}_ERA_AllImpacts/indicators/Economy/GDPpc_incl_impacts.out",
            # f"X:/user/dekkerm/IMAGE_environments/IMPACTS/2_TIMER/outputlib/TIMER_3_5/IMPACTS/{scenario}/indicators/Economy/GDPpc_incl_impacts.out",
        "start_year": 2000,
        "end_year": 2050,
        "project": project,
        "scenario": scenario,
        "adaptation": True,
        "counterfactual": True,
        "draw": "mean",
        "reporting_tool": False,#"X:/user/dekkerm/IMAGE_environments/IMPACTS/7_Reporting_Tool/outxlsx",
        "dask_on":True
    }

    # Save YAML
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(
            data, file, default_flow_style=False, sort_keys=False
        )

    print(f"{scenario}.yaml file generated")




project="SPARCCLE"
path="X:\\user\\liprandicn\\projects\\mt-comparison\\data\\yaml_settings"
for scenario in SPARCCLE_extra:
    generate_config_yaml(path, scenario, project)


# SPARCCLE_2round = [
#     "SSP2_M_CP_ERA_NoImpacts",
#     "SSP2_M_CP_ERA_AllImpacts",
#     "SSP2_M_CP_ERA_NoEcon",
#     "SSP2_M_CP_Default_NoEcon",
#     "SSP3_H_STS3_AllImpacts",
#     "SSP3_H_ERA_AllImpacts",
#     "SSP3_H_STS3_NoEcon",
#     "SSP3_H_ERA_NoEcon",
#     "SSP1_M_CP_ERA_AllImpacts",
#     "SSP1_M_CP_ERA_NoImpacts",
#     "SSP1_M_CP_ERA_NoEcon",
#     "SSP1_ML_ERA_NoImpacts",
#     "SSP2_ML_ERA_NoImpacts",
#     "SSP1_ML_ERA_AllImpacts",
#     "SSP2_ML_ERA_AllImpacts",
#     "SSP1_ML_ERA_NoEcon",
#     "SSP2_ML_ERA_NoEcon",
#     "SSP1_VLLO_ERA_NoImpacts",
#     "SSP2_VLLO_ERA_NoImpacts",
#     "SSP1_VLLO_STS1_AllImpacts",
#     "SSP1_VLLO_ERA_AllImpacts",
#     "SSP2_VLLO_STS1_AllImpacts",
#     "SSP2_VLLO_ERA_AllImpacts",
#     "SSP1_VLLO_ERA_NoEcon"
#     ]


# SPARCCLE_extra = [
#     'SSP1_ML_ERA_AllImpacts',
#     'SSP1_ML_ERA_AllImpacts_extra',
#     'SSP1_ML_ERA_NoEcon',
#     'SSP1_ML_ERA_NoEcon_extra',
#     'SSP1_ML_ERA_NoImpacts',
#     'SSP1_ML_ERA_NoImpacts_extra',
#     'SSP1_M_CP_ERA_AllImpacts',
#     'SSP1_M_CP_ERA_AllImpacts_extra',
#     'SSP1_M_CP_ERA_NoEcon',
#     'SSP1_M_CP_ERA_NoEcon_extra',
#     'SSP1_M_CP_ERA_NoImpacts',
#     'SSP1_M_CP_ERA_NoImpacts_extra',
#     'SSP1_VLLO_ERA_AllImpacts',
#     'SSP1_VLLO_ERA_NoEcon',
#     'SSP1_VLLO_ERA_NoImpacts',
#     'SSP1_VLLO_ERA_NoImpacts_extra',
#     'SSP1_VLLO_STS1_AllImpacts',
#     'SSP1_VLLO_STS1_AllImpacts_extra',
#     'SSP2_ML_ERA_AllImpacts',
#     'SSP2_ML_ERA_AllImpacts_extra',
#     'SSP2_ML_ERA_NoEcon',
#     'SSP2_ML_ERA_NoEcon_extra',
#     'SSP2_ML_ERA_NoImpacts',
#     'SSP2_ML_ERA_NoImpacts_extra',
#     'SSP2_M_CP_ERA_AllImpacts',
#     'SSP2_M_CP_ERA_AllImpacts_extra',
#     'SSP2_M_CP_ERA_NoEcon',
#     'SSP2_M_CP_ERA_NoEcon_extra',
#     'SSP2_M_CP_ERA_NoImpacts',
#     'SSP2_M_CP_ERA_NoImpacts_extra',
#     'SSP2_VLLO_ERA_AllImpacts',
#     'SSP2_VLLO_ERA_AllImpacts_extra',
#     'SSP2_VLLO_ERA_NoImpacts',
#     'SSP2_VLLO_ERA_NoImpacts_extra', 
#     'SSP2_VLLO_STS1_AllImpacts',
#     'SSP2_VLLO_STS1_AllImpacts_extra',
#     'SSP3_H_ERA_AllImpacts',
#     'SSP3_H_ERA_AllImpacts_extra',
#     'SSP3_H_ERA_NoEcon',
#     'SSP3_H_ERA_NoEcon_extra',
#     'SSP3_H_STS3_NoEcon',
#     'SSP3_H_STS3_NoEcon_extra',
#     'SSP2_M_CP_ERA_NoImpacts'
#     ]


# EERIE = [
#     "SSP2_M_CP_ENSO_r2",
#     "SSP2_M_CP_idealised_r3",
#     "SSP2_M_CP_idealised_r1",
#     "SSP2_VLLO_ENSO_r2",
#     "SSP2_VLLO_idealised_r3",
#     "SSP2_M_CP_idealised_r1"
# ]