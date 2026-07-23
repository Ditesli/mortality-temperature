import mortality_functions as mf
import yaml
from pathlib import Path
import dask
import time
from dask.distributed import Client, LocalCluster
from pyinstrument import Profiler

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"



scenarios = [
    "SSP3_H_ERA_AllImpacts",
    "SSP3_H_ERA_NoEcon",
    "SSP1_M_CP_ERA_AllImpacts",
    "SSP1_M_CP_ERA_NoEcon",
    "SSP1_VLLO_ERA_AllImpacts",
    "SSP2_VLLO_ERA_AllImpacts",
    "SSP1_VLLO_ERA_NoEcon",
    "SSP1_ML_ERA_AllImpacts",
    "SSP2_ML_ERA_AllImpacts",
    "SSP1_ML_ERA_NoEcon",
    "SSP2_ML_ERA_NoEcon",
    "SSP2_M_CP_ERA_AllImpacts",
    "SSP2_M_CP_ERA_NoEcon",
]


clim_vars = ["ssp245", "ssp370", "ssp585"]



if __name__ == "__main__":
    
    cluster = LocalCluster(
        n_workers=24,          # workers = (# CPUs / # threads)
        threads_per_worker=4, # 4 threads per process 
        memory_limit= '28GB', #'32GB' (memory_limit = #GB RAM/n_workers)
    )
    client = Client(cluster)
    print(f"Dask started succesfully. Dashboard in: {client.dashboard_link}")


    draw=10
    for scenario in scenarios:
        # for clim_var in clim_vars:
        for j in range(0,draw):
            
            print(f"{scenario} - draw:{j}")
                
            start = time.time()
            
            # config_file = Path(__file__).parent.parent / "settings" / f"SPARCCLE/test.yaml"
            # with open(config_file) as f:
            #     config = yaml.safe_load(f)

            # profiler = Profiler()
            # profiler.start()
                
            mf.CalculateMortality(
                wdir="X:/user/liprandicn/Projects/mt-comparison/models/carleton2022",#config["wdir"], # Working directory
                years=range(2000,2100),#range(config["start_year"], config["end_year"]), # Years range
                temp_dir=f"X:/user/dekkerm/IMAGE_environments/IMPACTS/Z_Emulator_Standalone_Tool/{scenario}/netcdf",#f"X:\\user\\liprandicn\\data\\IMAGE_Temperatures\\SPARCCLE_scenarios\\ACCESS-CM2_{clim_var}_r1i1p1f1\\{scenario}\\netcdf",#config["temp_dir"],#+f"{clim_var[i]}", # Path to climate data files
                gdp_dir=f"X:/user/dekkerm/IMAGE_environments/IMPACTS/2_TIMER/outputlib/TIMER_3_5/IMPACTS/{scenario}/indicators/Economy/GDPpc_incl_impacts.out",#config["gdp_dir"], # Path to GDP data files
                project="SPARCCLE_uncertainty",#config["project"],#+f"{i}", # Project name
                scenario=f"{scenario}",#_{clim_var}",#config["scenario"], # Scenario name
                adaptation=True,#config["adaptation"], # Adaptation on or off
                counterfactual=True,#config["counterfactual"], # Counterfactual climate scenario
                draw=f"TRAP_{j}",#f"LHS_{LHS}_{i}",#config["draw"], # Mean or specific/random draw
                reporting_tool=False#config["reporting_tool"], # Report on or off
            )

            # profiler.stop()
            # profiler.write_html("profile.html")
            
            end=time.time()
            print(end-start)


    client.close()
    cluster.close()