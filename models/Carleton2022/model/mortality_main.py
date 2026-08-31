import mortality_functions as mf
import yaml, time, dask, tempfile
from pathlib import Path
from dask.distributed import Client, LocalCluster
from pyinstrument import Profiler




if __name__ == "__main__":
    

    cluster = LocalCluster(
        n_workers=40,  #  workers = (# CPUs / # threads), 48, 40 (1)
        threads_per_worker=1,   # threads per worker, 2, 1 (1)
        memory_limit="14GB", # (memory_limit = #GB RAM/n_workers), 14, 12 (1)
        local_directory=tempfile.gettempdir(),
        scheduler_port=8886,
        dashboard_address=":8887"
    )
    client = Client(cluster)
    print(f"Dask started succesfully. Dashboard in: {client.dashboard_link}")
    
    
    config_file = Path(__file__).parent.parent / "settings" / f"example/test.yaml"
    
    
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # profiler = Profiler()
    # profiler.start()
    
    start = time.time()
        
    mf.CalculateMortality(**config)
    
    end=time.time()
    print(end-start)
    
    # profiler.stop()
    # profiler.write_html("profile.html")
    
        
    client.close()
    cluster.close()