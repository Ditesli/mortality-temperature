import os, yaml


def generate_config_yaml(path, scenario, project):

    if "extra" in scenario:
        scenario_image_land = scenario[:-6]
    else:
        scenario_image_land=scenario

    for climvar in ["ssp245_r1", "ssp245_r2", "ssp245_r3", "ssp370_r1", "ssp370_r2", "ssp370_r3", "ssp585_r1", "ssp585_r2", "ssp585_r3"]:
        
        # Define data
        data = {
            "wdir": "",
            "temp_dir": f"",
            "gdp_dir": f"",
            "start_year": 2000,
            "end_year": 2100,
            "monthly_output": False,
            "impact_regions": False,
            "project": project,
            "scenario": scenario+"_"+climvar,
            "adaptation": False,
            "counterfactual": True,
            "draw": "LHScut_50_0_p25-p75",
            "reporting_tool": False,
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

project=""
path=""
for scenario in SCENARIOS:
    generate_config_yaml(path, scenario, project)



# scenariomip = [
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
#   "SSP3_M_CP"
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