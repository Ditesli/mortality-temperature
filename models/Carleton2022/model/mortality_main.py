import mortality_functions as mf
import yaml, time, dask
from pathlib import Path
from dask.distributed import Client, LocalCluster
from pyinstrument import Profiler


if __name__ == "__main__":
    
    
    # cluster = LocalCluster(
    #         n_workers=1,          # workers = (# CPUs / # workers)
    #         threads_per_worker=4, # 4 threads per process 
    #         memory_limit= '28GB' # (memory_limit = #GB RAM/n_workers) 1 worker ~ 28GB
    #     )
    # client = Client(cluster)
    # print(f"Dask started succesfully. Dashboard in: {client.dashboard_link}")
    

    # for scenario in scenarios:
        
    config_file = Path(__file__).parent.parent / "settings" / f"SPARCCLE/test.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # profiler = Profiler()
    # profiler.start()
    start = time.time()
        
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
        dask_on=config["dask_on"],
    )
    end=time.time()
    print(end-start)
    
    # profiler.stop()
    # profiler.write_html("profile.html")
        
        
    # client.close()
    # cluster.close()

