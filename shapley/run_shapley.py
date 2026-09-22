import shapley_decomposition as sd

SCENARIOS = [
    "SSP1_L", "SSP1_M", "SSP1_M_CP", "SSP1_ML", "SSP1_VLHO", "SSP1_VLLO",
    "SSP2_L", "SSP2_M", "SSP2_M_CP", "SSP2_ML", "SSP2_VLHO", "SSP2_VLLO",
    "SSP3_H", "SSP3_M_CP",
    "SSP5_H", "SSP5_HL",
]
# SCENARIOS = [f"{scen}_NoAdap" for scen in SCENARIOS]

predictors =  ["ssp", "climate", "variability", "erf"] 
#["ssp", "gdp_impacts", "climate", "variability"]
# ["variability", "gdppc", "population", "climate", "erf_draw"] 
#[("cumulative_emissions", "squared_cumulative_emissions"), "model", "scenario"]
variable="mortality"
#"Primary_Energy"
age_group = "All ages"
region = "CAN"
gdp_dir =  "X:\\user\\boerdhs\\Projects\\ScenarioMIP\\2_TIMER\\outputlib\\TIMER_3_4\\ScenarioMIP\\{scenario}\\tuss\\global/GDP_ppp.scn"
# "X:/user/dekkerm/IMAGE_environments/IMPACTS/2_TIMER/outputlib/TIMER_3_5/IMPACTS/{scenario}/indicators/Economy/GDPpc_incl_impacts.out"
wdir = "X:\\user\\liprandicn\\projects\\mt-comparison\\models\\carleton2022\\output\\ScenarioMIP7\\"
# "X:\\user\\liprandicn/Projects\\mt-comparison\\models/Carleton2022/output/SPARCCLE_climvar\\"




for region in list(sd.IMAGE_REGIONS.keys()):
    for age_group in ["All ages"]:
        print(region)
        sd.ComputeShapleyOwen(wdir+"Shapley/", region, age_group, variable, predictors, show_plot=False, adap=True)
        # sd.GenerateDataframe4Shapley(wdir, gdp_dir, SCENARIOS, region, age_group, variable)

