import os, yaml


def generate_config_yaml(path, scenario, project):

    if "extra" in scenario:
        scenario_image_land = scenario[:-6]
    else:
        scenario_image_land=scenario

    for climvar in ["ssp245_r1", "ssp245_r2", "ssp245_r3", "ssp370_r1", "ssp370_r2", "ssp370_r3", "ssp585_r1", "ssp585_r2", "ssp585_r3"]:
        
        # Define data
        data = {
            "wdir": "X:/user/liprandicn/Projects/mt-comparison/models/carleton2022",
            "temp_dir": f"X:\\user\\liprandicn\\data\\IMAGE_Temperatures\\ScenarioMIP7\\ACCESS-CM2_{climvar}i1p1f1\\{scenario}\\netcdf",
            # f"X:/user/dekkerm/IMAGE_environments/IMPACTS/Z_Emulator_Standalone_Tool/{scenario_image_land}/netcdf", 
            "gdp_dir": f"X:\\user\\boerdhs\\Projects\\ScenarioMIP\\2_TIMER\\outputlib\\TIMER_3_4\\ScenarioMIP\\SSP1_VLLO\\tuss\\global/GDP_ppp.scn",
            "start_year": 2000,
            "end_year": 2100,
            "project": project,
            "scenario": scenario+"_"+climvar,#+"_NoAdap",
            "adaptation": True,
            "counterfactual": True,
            "draw": "LHScut_50_0_p25-p75",
            "reporting_tool": False,
                # "X:/user/dekkerm/IMAGE_environments/IMPACTS/7_Reporting_Tool/outxlsx",
            "dask_on":True,
            "stochastic": True
        }
        

        file_path = os.path.join(path, project, f"{data['scenario']}.yaml")

        # Save YAML
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(
                data, file, default_flow_style=False, sort_keys=False
            )

        print(f"{scenario}.yaml file generated")
        
        

SCENARIOS = []

project="Project"
path="Path"
for scenario in SCENARIOS:
    generate_config_yaml(path, scenario, project)
    
    

# SCENARIOMIP7 = [
#   "SSP1_L",
#   "SSP1_M",
#   "SSP1_M_CP",
#   "SSP1_ML",
#   "SSP1_VLHO",
#   "SSP1_VLLO",
#   "SSP2_L",
#   "SSP2_M",
#   "SSP2_M_CP",
#   "SSP2_ML",
#   "SSP2_VLHO",
#   "SSP2_VLLO",
#   "SSP3_H",
#   "SSP5_H",
#   "SSP5_HL",
# ]


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


# EERIE = [
#     "SSP2_M_CP_ENSO_r2",
#     "SSP2_M_CP_idealised_r3",
#     "SSP2_M_CP_idealised_r1",
#     "SSP2_VLLO_ENSO_r2",
#     "SSP2_VLLO_idealised_r3",
#     "SSP2_M_CP_idealised_r1"
# ]