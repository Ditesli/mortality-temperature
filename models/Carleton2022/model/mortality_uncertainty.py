import mortality_functions as mf
import yaml
from pathlib import Path
import dask
import time
from dask.distributed import Client, LocalCluster
from pyinstrument import Profiler



if __name__ == "__main__":
    
    cluster = LocalCluster(
        n_workers=24,          # workers = (# CPUs / # threads)
        threads_per_worker=4, # 4 threads per process 
        memory_limit= '28GB', #'32GB' (memory_limit = #GB RAM/n_workers)
    )
    client = Client(cluster)
    print(f"Dask started succesfully. Dashboard in: {client.dashboard_link}")


    # draw=10
    # for scenario in scenarios:
        # for clim_var in clim_vars:
    for i in range(0,50):
            
        print(i)
            
        start = time.time()
        
        config_file = Path(__file__).parent.parent / "settings" / f"example/test.yaml"
        # config_file = Path(f"X:\\user\\liprandicn\\projects\\mt-comparison\\data\\yaml_settings\\EERIE\\{scenario}.yaml")
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # profiler = Profiler()
        # profiler.start()
            
        mf.CalculateMortality(
            wdir=config["wdir"], #"X:/user/liprandicn/Projects/mt-comparison/models/carleton2022",## Working directory
            years=range(config["start_year"], config["end_year"]), # Years range
            temp_dir=config["temp_dir"],
            #f"X:\\user\\liprandicn\\data\\IMAGE_Temperatures\\SPARCCLE_scenarios\\ACCESS-CM2_{clim_var}_r1i1p1f1\\{scenario}\\netcdf",#f"X:/user/dekkerm/IMAGE_environments/IMPACTS/Z_Emulator_Standalone_Tool/{scenario}/netcdf",###+f"{clim_var[i]}", # Path to climate data files
            gdp_dir=config["gdp_dir"],
            #f"X:/user/dekkerm/IMAGE_environments/IMPACTS/2_TIMER/outputlib/TIMER_3_5/IMPACTS/{scenario}/indicators/Economy/GDPpc_incl_impacts.out",#, # Path to GDP data files
            project= "SPARCCLE_uncertainty_1scenario_dask", #config["project"],
            #"SPARCCLE_climvar",##+f"{i}", # Project name
            scenario=config["scenario"],
            #f"{scenario}_{clim_var}",# # Scenario name
            adaptation=config["adaptation"],
            # True,#, # Adaptation on or off
            counterfactual=config["counterfactual"],
            # True,# # Counterfactual climate scenario
            draw=f"LHScut_50_{i}_p25-p75", #config["draw"],
            # f"mean",#f"LHS_{LHS}_{i}",# # Mean or specific/random draw
            reporting_tool=False, #config["reporting_tool"],
            # =False,# # Report on or off
            dask_on=config["dask_on"]
        )

            # profiler.stop()
            # profiler.write_html("profile.html")
        
        end=time.time()
        print(end-start)


    client.close()
    cluster.close()