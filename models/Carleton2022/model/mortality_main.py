import mortality_functions as mf
import yaml, time, dask, tempfile
from pathlib import Path
from dask.distributed import Client, LocalCluster
from pyinstrument import Profiler



scenarios = [
#   "SSP1_L_ssp370_r3",
#   "SSP1_M_ssp370_r3",
  "SSP1_ML_ssp370_r3",
  "SSP1_VLHO_ssp370_r3",
  "SSP1_VLLO_ssp370_r3",
  "SSP2_L_ssp370_r3",
  "SSP2_M_ssp370_r3",
  "SSP2_ML_ssp370_r3",
  "SSP2_VLHO_ssp370_r3",
  "SSP2_VLLO_ssp370_r3",
  "SSP3_H_ssp370_r3",
  "SSP5_H_ssp370_r3",
  "SSP5_HL_ssp370_r3",
]


if __name__ == "__main__":
    

    # cluster = LocalCluster(
    #     n_workers=48,  #  workers = (# CPUs / # threads), 48, 40 (1)
    #     threads_per_worker=2,   # threads per worker, 2, 1 (1)
    #     memory_limit="14GB", # (memory_limit = #GB RAM/n_workers), 14, 12 (1)
    #     local_directory=tempfile.gettempdir(),
    #     scheduler_port=8786,
    #     dashboard_address=":8787"
    # )
    # client = Client(cluster)
    # print(f"Dask started succesfully. Dashboard in: {client.dashboard_link}")
    

    for scenario in scenarios:
        
        config_file = Path(__file__).parent.parent / "settings" / f"example/test.yaml"
        # config_file = f"X:\\user\\liprandicn\\projects\\mt-comparison\\models\\carleton2022\\data\\yaml\\ScenarioMIP7\\{scenario}.yaml"
        
        with open(config_file) as f:
            config = yaml.safe_load(f)

        # profiler = Profiler()
        # profiler.start()
        start = time.time()
            
        mf.CalculateMortality(
            wdir=config["wdir"], # Working directory
            years=range(config["start_year"], config["end_year"]), # Years range
            monthly_output=config["monthly_output"], # Monthly output on or off
            impact_regions=config["impact_regions"], # Impact regions on or off
            temp_dir=config["temp_dir"],#+f"{clim_var[i]}", # Path to climate data files
            gdp_dir=config["gdp_dir"], # Path to GDP data files
            project=config["project"],#+f"{i}", # Project name
            scenario=config["scenario"], # Scenario name
            adaptation=config["adaptation"], # Adaptation on or off
            counterfactual=config["counterfactual"], # Counterfactual climate scenario
            draw=config["draw"], # Mean or specific/random draw
            reporting_tool=config["reporting_tool"], # Report on or off
            dask_on=config["dask_on"],
            stochastic=config["stochastic"],
        )
        end=time.time()
        print(end-start)
        
    # profiler.stop()
    # profiler.write_html("profile.html")
        
        
    # client.close()
    # cluster.close()