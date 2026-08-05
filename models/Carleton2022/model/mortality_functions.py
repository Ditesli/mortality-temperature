import dask.delayed
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from dataclasses import dataclass, field
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import re, sys, os, prism, dask, shapely, shutil, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils import temperature as tmp
import numpy_groupies as npg
from scipy.stats import qmc, norm, truncnorm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from dask.delayed import delayed
from dask.distributed import get_client
from typing import Any, Optional
from pathlib import Path


### ------------------------------------------------------------------------------


def CalculateMortality(**config):

    sets = ModelSettings(**config)

    model = MortalityModel(sets=sets)

    if sets.stochastic:
        model.runs_stochastic()
    else:
        model.run()



@dataclass
class ModelSettings:
    temp_dir: str
    gdp_dir: any
    wdir: str
    project: any
    scenario: str
    years: list
    adaptation: bool
    counterfactual: bool
    draw: any
    reporting_tool: any
    dask_on: bool
    stochastic: bool
    base_years: list = field(default_factory=lambda: list(range(2000, 2010)))
    age_groups: list = field(
        default_factory=lambda: ["young", "older", "oldest"]
        )
    T: np.ndarray = field(
        default_factory=lambda: np.arange(
            -20, 40.1, 0.1, dtype=np.float32
        ).round(1)
    )
    
    def __post_init__(self):
        self.years = self.validate_years()

        
    def __post_init__(self):
        
        # Include last year 
        if isinstance(self.years, range):
            self.years = range(self.years.start, self.years.stop + 1)
            
        if "comparison" in str(self.project).lower():
            self.base_years = list(range(1980, 1990))
        
        # Reduce range years if working with ERA5 data
        ERA5_END_YEAR = 2025

        if "ERA5" in self.scenario:
            self.years = [
                y for y in self.years
                if y <= ERA5_END_YEAR
            ]
            


@dataclass
class MortalityModel:
    sets: ModelSettings
    
    """
    Model to calculate mortality projections for the given parameters.
    1. The model will first validate the input years.
    2. It will load the input data from the wdir/data/ folder and load some parameters for the ERFs.
    2. Then will calculate mortality per year form the years range, calculating first marginal mortality 
    and substracting the counterfactual mortality used to isolate the role of climate change 
    from the benefits of income growth.
    3. Postprocess results by calculating total and relative mortality of all age groups and
    regions; and save them in the output folder.
    
    Parameters:
    ----------
    sets : ModelSettings
        Paths to main working directory, climate data and income data. This folder must contain two folders:
        data (used for calculations) and output (where results are stored)
    years : list
        Provide the range of years and step the model will run.
    project : str
        Name of the project, used to locate the data in the right folder. It can be any of the projects
        included in wdir/data/IMAGE_land/scen/ AND/OR to use in the output file name.
    scenario : str
        - SSP#_carleton:
        Scenarios that use the economic data (GDP) from Carleton et al. 
        - SSP#_ERA5:
        This scenario uses historical socioeconomic data (GDP from World Bank and pop data from LandScan) 
        and ERA5 temperature data records. Scenario runs from 2000 to 2025.
        - IMAGE scenarios:
        These scenarios use population and GDP data from the IMAGE model. Run from 2000 until 2100.
    regions : str
        Region classification to use (e.g., "IMAGE26", "countries", "impact regions").
    adaptation: bool
        If True, the model will take into account adaptation to generate future ERFs. If False, the 
        model will use the "present day" ERFs from the paper for all years and scenarios.
    """

        
        
        
        print("----------------------------------------------------------------")
        print(f"Running Mortality-Temperature model (Carleton et al., 2022 version)")
        print(f"-----> Project: {self.sets.project}")
        print(f"-----> Scenario: {self.sets.scenario}")
        print(f"-----> Years: {self.sets.years[0]} to {self.sets.years[-1]}")
        if self.sets.adaptation == True:
            print("-----> Adaptation is ON: ERFs will be generated with adaptation.")
        print("----------------------------------------------------------------")
            
        
        self.base = LoadInputData.for_baseline(sets=self.sets)        
        self.tempe = LoadInputData.for_temperature(sets=self.sets, base=self.base)
        self.scen = LoadInputData.for_scenario(sets=self.sets, base=self.base)
        self.erf = LoadInputData.for_erf(sets=self.sets, tempe=self.tempe, scen=self.scen, base=self.base)

        print("[2] Starting mortality calculations...")
        
        ### -------------------------- Code WITH dask -------------------------------
        if self.sets.dask_on == True: 
            
            client = get_client() 

            sets_future = client.scatter(self.sets)
            base_future = client.scatter(self.base)
            tempe_future = client.scatter(self.tempe)
            scen_future = client.scatter(self.scen)
            erf_future = client.scatter(self.erf)

            tasks = []
            for year in self.sets.years:
                task = dask.delayed(CalculateMortalityEffects)(
                    sets_future,     
                    base_future, 
                    tempe_future, 
                    scen_future, 
                    erf_future, 
                    year
                )
                tasks.append(task)
                
            rel_mor = dask.compute(*tasks)
            
            

        ### ----------------------- Code WITHOUT dask ------------------------------
        else:                    
            rel_mor = []
            for year in self.sets.years:
                rel_mor_year = CalculateMortalityEffects(self.sets, self.base, self.tempe, self.scen, self.erf, year)
                rel_mor.append(rel_mor_year)


        # ---------------------- POSTPROCESSING -----------------------

        # Stack all yearly results
        rel_mor_scenario = np.stack(rel_mor, axis=-1)
        
        PostprocessResults(self.sets, self.base, self.scen, rel_mor_scenario)
                    
          


@dataclass
class LoadInputData:
    
    """
    Container for all input data required to run the model.
    """

    spatial_relation: Optional[gpd.GeoDataFrame] = None
    ir: Optional[pd.DataFrame] = None
    region_class: Optional[pd.DataFrame] = None
    gammas: Any = None
    pop: Optional[pd.DataFrame] = None
    random_vals: Optional[np.ndarray] = None
    temp_mean: Any = None
    temp_std: Any = None
    daily_temp_t0: Any = None
    climtas_ir: Any = None
    climtas_base: Any = None
    image_shares: Any = None
    country_shares: Any = None
    image_gdppc: Any = None
    gamma: Any = None
    erfs_t0: Any = None
    tmin_t0: Any = None
    

    @classmethod
    def for_baseline(cls, sets):
        
        """
        Read and load all input files required for mortality calculations. The 
        necessary data is located in the wdir/data folder.  
        """
        
        print("[1.1] Loading input files and defining parameters...")


        region_class = GenerateRegionClassification(sets)
        
        cache_dir = sets.wdir+f"/cache"
        name = "IMAGE" if "ERA5" not in sets.scenario else "ERA5"
        
        
        # Avoid computing many times the spatial relationship and impacts regions order (ir)
        if (Path(cache_dir)/f"{name}_spatial_relation.parquet").is_file():
            spatial_relation = pd.read_parquet(cache_dir+f"/{name}_spatial_relation.parquet")
            ir = pd.read_parquet(cache_dir+f"/{name}_ir.parquet")[0].values
            
        else:
            temp_mean = xr.open_dataset(os.path.join(sets.temp_dir, "GTMP_30MIN.nc"))
            spatial_relation, ir = GridRelationship(sets, temp_mean)
            
            spatial_relation.to_parquet(cache_dir+f"/{name}_spatial_relation.parquet")
            pd.DataFrame(ir).to_parquet(cache_dir+f"/{name}_ir.parquet")
            
        
        # Avoid generating same random values that generate daily temperature 
        if (Path(cache_dir)/f"{name}_random_vals.npy").is_file():
            random_vals = np.load(cache_dir+f"/{name}_random_vals.npy")
        else:
            random_vals = RandomValues4Temperature()
            np.save(cache_dir+f"/{name}_random_vals.npy", random_vals)
    
    
        return cls(
            spatial_relation=spatial_relation,
            ir=ir,
            region_class=region_class,
            random_vals=random_vals
        )
        
        
    @classmethod
    def for_temperature(cls, sets, base: "LoadInputData" = None):
        
        """
        Read and load all input files required for mortality calculations. The 
        necessary data is located in the wdir/data folder.  
        """
        
        print(f"[1.2] Loading temperature files for scenario {sets.scenario}...")    

        print("[1.2.1] Loading temperature data from IMAGE")
        if "ERA5" not in sets.scenario:
            temp_month_mean, temp_month_std = tmp.OpenMonthlyTemperatures(sets.temp_dir, "MEAN")
        else:
            temp_month_mean, temp_month_std = None, None
            
        daily_temp_t0 = ImportBaselineTemperatures(
            sets=sets, 
            base=base, 
            temp_mean=temp_month_mean,
            temp_std=temp_month_std
            )
        
        # Read GDP shares for scenarios that do not use Carleton's socioeconomic data.
        if sets.adaptation and "ERA5" not in sets.scenario:
                climtas_ir, climtas_base = ImportClimtas(
                    sets=sets,
                    base=base,
                    temp_mean=temp_month_mean
                    )
        else:  
            climtas_base=None; climtas_ir=None

        return cls(
            temp_mean=temp_month_mean,
            temp_std=temp_month_std,
            daily_temp_t0=daily_temp_t0,
            climtas_ir=climtas_ir,
            climtas_base=climtas_base
        )
        
        
    @classmethod
    def for_scenario(cls, sets, base: "LoadInputData" = None):
        
        """
        Read and load all input files required for mortality calculations. The 
        necessary data is located in the wdir/data folder.  
        """
        
        print("[1.3] Loading scenario files...") 
        
        population = ImportPopulationData(sets, base.ir, sets.scenario)
        
        # Read GDP shares for scenarios that do not use Carleton's socioeconomic data.
        if sets.adaptation:
                
            print("[1.3.2] Loading GDPpc shares at the impact region level...")
            image_shares, country_shares = GenerateGDPpcShares(sets, base, sets.scenario)
            image_gdppc = None
            
            if not re.search(r"ERA5", sets.scenario):# and "carleton" not in sets.scenario.lower():
                
                print("[1.3.3] Loading GDP data from IMAGE...")
                image_gdppc = ReadTIMERFiles(sets)
                
        # Set to None when adaptation is off        
        else:  
            image_shares = None; image_gdppc = None; country_shares = None
        
        return cls(
            pop=population,
            image_shares=image_shares,
            country_shares=country_shares,
            image_gdppc=image_gdppc
        )
        
        
    @classmethod
    def for_erf(cls, sets, tempe, scen, base: "LoadInputData" = None):
        
        """
        Read and load all input files required for mortality calculations. The 
        necessary data is located in the wdir/data folder.  
        """
        
        gamma_coeffs = ImportGammaCoefficients(sets, sets.draw)
        
        erfs_t0, tmin_t0 = GenerateERFAll(
            sets=sets, 
            base=base,
            tempe=tempe,
            scen=scen,
            erf=None,
            gammas=gamma_coeffs,
            year=None, 
            adaptation=False, 
            counterfactual=None
            ) 

        return cls(
            gammas=gamma_coeffs,
            erfs_t0=erfs_t0,
            tmin_t0=tmin_t0
        )
    
        


def RandomValues4Temperature():
    
    # Initialize random number generator
    rng = np.random.default_rng(seed=42)
    
    # Generate normal distribution with std = 1
    vals = rng.standard_normal(size=(360,720,366)).astype(np.float32)
    
    return vals



def GenerateRegionClassification(sets):
    
    print(f"[1.1.1] Loading region classification...")
    
    region_class = pd.read_csv(
        os.path.dirname(os.path.dirname(sets.wdir)) +
        f"/data/RegionClassification/region_classification.csv"
        )[["hierid", "ISO3", "IMAGE26"]].iloc[:24378].rename(columns={"IMAGE26":"IMAGE"})
    
    return region_class



def GridRelationship(sets, grid):
    
    """
    Create a DataFrame with the spatial relationship between temperature data points 
    and impact regions. It will assign each grid cell to the impact region it intersects with.
    If a grid cell it will be assigned multiple times. The function can work with any resolution 
    of temperature data. 
    Create a pandas series of the impact regions to align the order of the regions in the rest 
    of the dataframes with the same order as the spatial relationship dataframe. 
    """
    
    print("[1.1.2] Creating spatial relationship between temperature grid and impact regions...")
    
    # ---------- If ERA5 data ----------
    if re.search(r"ERA5", sets.scenario):
        # Use function located in the utils_common folder to import ERA5 data in the right format
        grid,_ = tmp.DailyTemperatureERA5(
            era5_dir=sets.temp_dir, 
            year=sets.years[0], 
            temp_type="mean", 
            pop_map=None, 
            to_array=False
            )

    # Extract coordinates
    def FindCoordinateName(possible_names, coord_names, temperature):
    
        for name in possible_names:
            if name in coord_names:
                return temperature[name].values
        raise KeyError(f"No coordinate was found among: {possible_names}")

    coord_names = grid.coords.keys()
    lon_vals = FindCoordinateName(["lon", "longitude", "x"], coord_names, grid)
    lat_vals = FindCoordinateName(["lat", "latitude", "y"], coord_names, grid)

     # Calculate grid cell size (assuming uniform grid)
    lon_size = np.abs(np.diff(lon_vals)[0])
    lat_size = np.abs(np.diff(lat_vals)[0])

    # Create meshgrid 
    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)  
    lon_flat = lon2d.ravel()
    lat_flat = lat2d.ravel()
    
    lon_min = lon_flat - (lon_size / 2)
    lat_min = lat_flat - (lat_size / 2)


    # Create vectorize GeoDataFrame
    points_gdf = gpd.GeoDataFrame(
        {"longitude": lon_flat, "latitude": lat_flat},
        geometry=shapely.box(lon_min, lat_min, lon_min + lon_size, lat_min + lat_size)
    )
        
    # Load .shp file with impact regions and set the same coordinate reference system (CRS)
    ir = gpd.read_file(sets.wdir + "/data/CarletonSM/ir_shp/impact-region.shp")
    points_gdf = points_gdf.set_crs(ir.crs, allow_override=True)
    
    # Make spatial join
    relationship = gpd.sjoin(points_gdf, ir, how="inner", predicate="intersects")

    # Return corresponding ir per pixel (relationship) and order of regions to align imported data
    return relationship[["index_right", "hierid"]], ir["hierid"].values



def ImportGammaCoefficients(sets, draw):    
    
    """
    Import gamma coefficients from the paper's Suplementary Material and convert 
    them to the right format to be multiplied later on by the covariates
    (climtas y loggdppc).
    The settings have the option to choose between the mean estimations or to draw
    from the distribution of the gamma coefficients using the variance-covariance matrix.
    
    Returns:
    ----------
    gamma_g : numpy.ndarray
        2d-array with the 36 gamma coefficients. Each 12 coefficients correspond
        to the younger, older and oldest group, respectively.
    cov_g : numpy.ndarray
        2d-array with the corresponding position that shoul multiply each coefficient.
        0 --> constant, multiply by 1
        1 --> multiply by the covariate climtas
        2 --> multiply by the covariate loggdppc
    """
    
    ### -------------- Read gamma coefficients file ------------------------------
    
    with open(sets.wdir+"/data/CarletonSM/Agespec_interaction_response.csvv") as f:
        
        # Initialize nupt array of 36x36
        vcv = np.zeros((36,36))
        
        # Extract relevant lines
        for i, line in enumerate(f, start=1):

            if i == 21:
                # Extract 1, climtas, loggdppc
                covar_names = [x for x in line.strip().split(", ")]
                # Convert to indices and to array
                covar_map = {"1":0, "climtas":1, "loggdppc":2}
                covar_idx = np.array([covar_map[str(x)] for x in covar_names])
                
            if i == 23:
                # Extract gamma coefficients
                gammas = np.array([float(x) for x in line.strip().split(", ")])
                
            if i in range(25,61):
                vcv[i-25] = np.array([float(x) for x in line.strip().split(", ")])
                

    ### -----------------------------------------------------------------------------
    # Apply conditions to select a random draw a draw from LHS, or mean estimations
    
    if str(draw).lower() == "mean":
        print("[1.4.1] Loading gamma coefficients - Mean estimates...")
        pass

    elif "LHS_" in str(draw):

        print("[1.4.1] Loading gamma coefficients using Latin Hypercube Sampling")
        # n_draws = 100 # Fixed number based on elbow error analysis
        
        n_draws = int(re.search(r"LHS_(\d+)_(\d+)", draw).group(1))
        sample = int(re.search(r"LHS_(\d+)_(\d+)", draw).group(2))

        # Latin hypercube object for the 3x12 gammas, including seed for replication
        sampler = qmc.LatinHypercube(d=36, seed=42, scramble=True)
        sample_uniform = sampler.random(n=n_draws)

        # Transform to a normal standard distribution
        sample_normal = norm.ppf(sample_uniform)

        # Apply the Cholesky factorization, matrix is now triangular inferior 
        L = np.linalg.cholesky(vcv)

        # Generate samples
        gammas_draws = gammas + np.dot(sample_normal, L.T)

        # Select sample given by LHS_## from draw
        gammas = gammas_draws[sample]
        
        
    elif "LHScut_" in str(draw):
        print("[1.4.1] Loading gamma coefficients using Latin Hypercube Sampling and 25-75 percentile")
        
        n_draws = int(re.search(r"LHScut_(\d+)_(\d+)_p(\d+)-p(\d+)", draw).group(1))
        sample = int(re.search(r"LHScut_(\d+)_(\d+)_p(\d+)-p(\d+)", draw).group(2))
        p_low = int(re.search(r"LHScut_(\d+)_(\d+)_p(\d+)-p(\d+)", draw).group(3))
        p_high = int(re.search(r"LHScut_(\d+)_(\d+)_p(\d+)-p(\d+)", draw).group(4))
        
        # Latin hypercube object for the 3x12 gammas, including seed for replication
        sampler = qmc.LatinHypercube(d=36, seed=42, scramble=True)
        sample_uniform = sampler.random(n=n_draws)     
        
        # Transform to a truncated normal standard distribution
        z_inferior = norm.ppf(p_low/100)
        z_superior = norm.ppf(p_high/100)
        sample_normal = truncnorm.ppf(sample_uniform, z_inferior, z_superior, loc=0, scale=1)
        
        # Apply the Cholesky factorization, vcv matrix is now triangular inferior 
        L = np.linalg.cholesky(vcv)
        
        # Generate samples
        gammas_draws = gammas + np.dot(sample_normal, L.T)
        
        # Select sample given by LHS_## from draw
        gammas = gammas_draws[sample]  
        

    elif "MonteCarlo" in str(draw):
        print(f"[1.4.1] Loading gamma coefficients - Random draw from the normal distribution...")
        gammas = np.random.multivariate_normal(mean=gammas, cov=vcv, size=1) 
        
    
    return gammas.reshape(3,12).astype(np.float32), covar_idx.reshape(3,12).astype(int)



def ImportPopulationData(sets, ir, scenario):
    
    # Extract SSP from scenario string
    ssp = re.search(r"(?i)ssp\d+", scenario).group().upper()
    
    print(f"[1.3.1] Loading Population data for {ssp} scenario at the impact regions level...")
    
    # Include ALWAYS population data from 2000 to 2010 (used in the counterfactual part)
    year = sorted(set(sets.years).union(range(2000, 2010)))
        
    # Import population data based on scenario type
    if 'carleton' in scenario.lower():
        # Import population data from the paper
        population = ImportDefaultPopulationData(sets, ssp, year, ir)
        
    else:
        # Import IMAGE population data nc4 file and calculate population per impact region
        population = ImportIMAGEPopulationData(sets, ssp, year, ir)
    
    return population



def ImportDefaultPopulationData(sets, ssp, years, ir): 
    
    """
    Read default population file for a given SSP scenario and age group
    and convert it to the right format (dataframe) to be used later on in the model in the 
    spatial aggregation of mortality. 
    """
    
    population_groups = []
    age_pop_names = ['pop0to4', 'pop5to64', 'pop65plus']
    
    for age_group, age_name in zip(sets.age_groups, age_pop_names):
        
        # Read 'present-day' population data
        pop_historical = (
            pd.read_csv(sets.wdir+f"/data/Population/PopulationHistorical/pop_historical_{age_group}.csv")
            .set_index("hierid")
        )

        pop_ssp = (
            xr.open_dataset(sets.wdir+f"/data/CarletonSM/econ_vars/{ssp.upper()}.nc4")[age_name]
            .sel(model="low") # Select any GDP model
            .to_dataframe() # Convert to dataframe
            .drop(columns=['ssp', 'model'])
            .unstack('year') # Reshape to have years as columns
            .pipe(lambda df: df.set_axis(df.columns.get_level_values(-1), axis=1))
            .pipe(lambda df: df.rename_axis("hierid"))
            [[y for y in years if y >= 2023]] # Keep only years from 2023 onwards 
            .merge(pop_historical, left_index=True, right_index=True) # Merge with historical population 
            .reindex(ir) # Align to impact regions order
            .pipe(lambda df: df.set_axis(df.columns.astype(int), axis=1))
            .pipe(lambda df: df.reindex(sorted(df.columns, key=int), axis=1))
        )
        
        population_groups.append(pop_ssp.loc[:, sets.years].to_numpy().astype(np.float32))
    
    return np.stack(population_groups, axis=0) # Shape: (3, 24378, len(years))



def ImportIMAGEPopulationData(sets, ssp, years, ir):
    
    """
    Read precalculated IMAGE population data at the impact region level for a
    given SSP.
    """
        
    pop_ssp = []

    for age_group in sets.age_groups:
        pop_ssp_group = (
            pd.read_csv(
                sets.wdir +
                f"/data/Population/PopulationIMAGE/pop_{ssp.lower()}_{age_group}.csv")
            .pipe(
                lambda df: df.filter(
                ["hierid"] +
                [c for c in df.columns if c.isdigit() and int(c) in years]
            ))
            .set_index("hierid")
            .reindex(ir) # Align to impact regions orders
            .pipe(lambda df: df.set_axis(df.columns.astype(int), axis=1))
        )

        pop_ssp.append(pop_ssp_group.loc[:, sets.years].to_numpy().astype(np.float32))
    
    return np.stack(pop_ssp, axis=0) # Shape: (3, 24378, len(years))



def ImportBaselineTemperatures(sets, base, temp_mean, temp_std):
    
    """
    The function will import the daily temperatures from 2000 to 2010, either precalculated
    ERA5 data or climate data from prescribed scenario. The output is a dictionary of numpy 
    arrays with the daily temperature per impact region and year.
    """
     
    print("[1.2.2] Generating 'present-day' temperature data...")
     
    # ------------------ ERA5 ------------------
    if "ERA5" in sets.scenario:
        
        t0_mean = {}
        for year in sets.base_years:
            
            # Load daily temperature files from ERA5 at ir level            
            t0_mean[year]  = xr.open_dataset(
                sets.wdir +
                f"/data/ClimateData/BaselineTemperatures/ERA5_tmean0_{year}.nc"
                ).tmean0.values.astype(np.float32)
            
    # -------------- Scenario data --------------
    else: 
        
        daily_temperature,_ = tmp.DailyFromMonthlyTemperature(
            temperature_mean=temp_mean,
            temperature_std=temp_std,
            years_in=sets.base_years,
            random_vals=base.random_vals, 
            to_xarray=False
        )

        t0_mean = MSTemperature2IR(
            temp=daily_temperature,
            spatial_relation=base.spatial_relation
            ).astype(np.float32)
    
    return t0_mean



def GenerateERFAll(sets, base, tempe, scen, erf, year, gammas, adaptation, counterfactual):
    
    """
    The code imports the gamma coefficients and the covariates (climtas and loggdppc) to 
    generate the Exposure Response Functions (ERFs) and Minimum Mortality Temperature (MMT)
    values per impact region and group. 

    Returns:
    ----------
    mor_np: 
        Dictionary with the three 2-d arrays corresponding to each age group.
    tmin:
        Dictionary with the three 1-d arrays corresponding to the minimum temperature.
    """
    
    # Import covariates with or without adaptation
    climtas, loggdppc = ImportCovariates(
        sets=sets,
        base=base,
        tempe=tempe,
        scen=scen,
        year=year, 
        adaptation=adaptation,
        counterfactual=counterfactual
        )

    # Covariates matrix
    covariates = np.column_stack(
        [np.ones(len(climtas)), climtas, loggdppc]
    ).astype(np.float32)
    
    
    if erf is None:
        erfs_t0 = None; tmin_t0 = None
    else:
        erfs_t0 = erf.erfs_t0; tmin_t0 = erf.tmin_t0 
            
    # Extract gammas and covariates position
    g = gammas[0]; cov = gammas[1]

    # Multiply each covariate by its corresponding gamma
    base = covariates[:, cov] * g
    
    # Compute the sum of the covariates to get polynomial coefficients
    tas = base[:, :, 0:3].sum(axis=2)  # Shape (24378, 3) for the three age groups
    tas2 = base[:, :, 3:6].sum(axis=2)  
    tas3 = base[:, :, 6:9].sum(axis=2) 
    tas4 = base[:, :, 9:12].sum(axis=2)

    # Generate raw Exposure Response Function
    erf = (
        tas[:, :, None] * sets.T[None, None, :]**1 +
        tas2[:, :, None] * sets.T[None, None, :]**2 +
        tas3[:, :, None] * sets.T[None, None, :]**3 +
        tas4[:, :, None] * sets.T[None, None, :]**4
    )
    
    # Impose zero mortality at tmin by vertically shifting erf
    erf, tmin = ShiftERFToTmin(erf, sets.T, tas, tas2, tas3, tas4, tmin_t0)
    
    #  # Ensure ERFs do not exceed no-adaptation ERFs 
    if erfs_t0 is not None:
        erf = np.minimum(erf, erfs_t0)
    
    # Impose weak monotonicity to the left and the right of the erf
    erf = MonotonicityERF(sets.T, erf, tmin)

    return erf, tmin



def ImportCovariates(sets, base, tempe, scen, year, adaptation, counterfactual):
    
    """
    Import the covariates climtas and loggdppc of the corresponding year as numpy arrays.
    If adaptation is False, the function will import the covariates defined in the paper 
    as the "present day" covariates. If adaptation is True the model can import them
    from different sources depending on the scenario.

    Returns:
    ----------
    climtas : np.ndarray
        1D array with the 30-year climate per impact region, the regions order is given by ir.
    loggdppc : np.ndarray 
        1D array with the log of the 13-year log(GDPpc) per impact region.
    """
    
    # NO ADAPTATION -----------------------------------------
    if adaptation==False:

        # Open covariates for "present day" (no adaptation) and reindex wrt ir dataframe
        covariates_t0 = (
             pd.read_csv(sets.wdir+"/data/CarletonSM/main_specification/mortality-allpreds.csv")
            .rename(columns={"region":"hierid"})
            .set_index("hierid")
            .reindex(base.ir)
        )
        
        # Extract only climtas and loggdppc as arrays
        climtas = covariates_t0["climtas"].values
        loggdppc = covariates_t0["loggdppc"].values
    
    # ADAPTATION ---------------------------------------------
    else:
        
        # climtas ---------------------------
        
        # Load ERA5 climatology
        if re.search(r"ERA5", sets.scenario):
            climtas = ImportClimtasERA5(sets.wdir, year, base.ir)
            
        # Load climatology of selected year and scenario
        else:
            # Load "present-day" climatology
            if counterfactual:
                climtas =  tempe.climtas_base
            else:
                climtas = tempe.climtas_ir[:,year-sets.years[0]]
                
        # log(GDPpc) ---------------------------    
        
        # Load historical log(GDPpc) from World Bank
        if re.search(r"ERA5", sets.scenario) or ("carleton" in sets.scenario.lower() and year < 2010):
            loggdppc = ImportHistoricalLogGDPpc(sets.wdir, base.ir, year, scen.country_shares)
        
        # Load log(GDPpc) from Carleton et al. (2022) for the selected year and scenario
        elif "carleton" in sets.scenario.lower() and year >= 2010:  
            loggdppc = ImportCarletonLogGDPpc(sets.wdir, sets.scenario, base.ir, year)
        
        # Load log(GDPpc) at the impact region level using the GDPpc output from IMAGE    
        else: 
            loggdppc = ImportIMAGEloggdppc(year, scen)
            
    return climtas.astype(np.float32), loggdppc.astype(np.float32)



def ImportHistoricalLogGDPpc(wdir, ir, year, country_shares):
    
    """
    Read historical GDP per capita data (GDP per capita (constant 2015 US$)) from
    the World Bank and calculate the 13 year running mean for the selected year at 
    the impact region level using the factors derived from the original paper that
    downscale national GDPpc to the regional one.
    """
    
    if year == 2025:
        year = 2024 # The latest year with GDPpc data available is 2024
    
    # Read GDPpc
    gdppc = (
        pd.read_csv(wdir + "/data/IncomeData/GDPpcHistorical/WB_WDI_NY_GDP_PCAP_KD.csv")
        [["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]] # Relevan columns
        .sort_values(["REF_AREA", "TIME_PERIOD"])
        .assign( # Calculate 13 year rolling mean of log(GDPpc) per country
            OBS_VALUE_13yr_mean=lambda x:
                x.groupby("REF_AREA")["OBS_VALUE"]
                .transform(lambda s: s.rolling(window=13).mean())
        )
        .loc[lambda x: x["TIME_PERIOD"] == year] # Keep only years from 2000 onwards
        .merge(country_shares, left_on="REF_AREA", right_on="ISO3", how="right") # Merge with impact regions
    )

    gdppc["gdppc"] = gdppc["OBS_VALUE_13yr_mean"] * gdppc["gdppc_share"]
    gdppc["loggdppc"] = np.log(gdppc["gdppc"])
    
    return gdppc.set_index("region").reindex(ir)["loggdppc"].values



def ImportCarletonLogGDPpc(wdir, scenario, ir, year):
    
    """
    Read GDP per capita files for a given SSP scenario from Carleton et al. (2022) 
    and calculate the 13-year running mean of the log(GDPpc) for the selected year 
    at the impact region level.
        
    Returns:
    ----------
    gdppc : np.ndarray
        GDP per capita data ordered by ir
    """
    
    #TODO: Fix to accecpt scenario
    
    scenario = re.search(r"(?i)\bssp\d+", scenario).group()
        
    # Read GDP per capita file
    gdppc = (
        xr.open_dataset(wdir+f"/data/CarletonSM/econ_vars/{scenario.upper()}.nc4")   
        .gdppc
        .mean(dim='model')  # Mean across models
        .rolling(year=13, min_periods=1)  # 13 year rolling mean
        .mean()
        .sel(year=year)  # Select relevant year
        .to_dataframe() # Convert to dataframe
        .reset_index()
        .drop(columns=["year", "ssp"])
        .rename(columns={"region":"hierid"})
        .set_index("hierid")
        .reindex(ir) # Reindex according to hierid
    )
    
    # Calculate log(GDPpc)
    gdppc["loggdppc"] = np.log(gdppc["gdppc"])
    
    # Return numpy array
    return gdppc["loggdppc"].values



def ImportIMAGEloggdppc(year, scen):
    
    """
    Calculate log(GDPpc) at the impact region level using the GDPpc output from a 
    TIMER run and the shares that downscale IMAGE GDPpc at the impact region level.
    """
    
    # Extract relevant year data (13 year rolling mean)
    image_gdppc = (
        scen
        .image_gdppc
        .sel(Time=slice(year-13,year))
        .mean(dim="Time")
        .mean(dim="Scenario")
        .mean(dim="Variable")
        .pint.dequantify() # Remove pint units and warning
        .to_dataframe()
        .reset_index()
    )
    
    # Merge IMAGE GDPpc with GDPpc shares
    gdppc_year = 2010 if year < 2010 else year
    
    gdppc = image_gdppc.merge(
        scen.image_shares[["region", "IMAGE", gdppc_year]], 
        right_on="IMAGE", 
        left_on="region", 
        how="right"
        )
    
    # Calculate share of log(GDPpc) based on regional GDPpc
    gdppc["gdppc"] = gdppc["Value"] * gdppc[gdppc_year] 
    gdppc["loggdppc"] = np.log(gdppc["gdppc"])
    
    return gdppc["loggdppc"].values



def GenerateGDPpcShares(sets, base, scenario):
    
    ssp = re.search(r"SSP\d", scenario).group()

    # Open scenario GDP data
    gdppc_shares = (
        xr.open_dataset(f"{sets.wdir}/data/CarletonSM/econ_vars/{ssp}.nc4")
        .mean(dim="model") # Mean between high and low economic models
        [["gdp", "pop", "gdppc"]]
        .to_dataframe() # Convert to dataframe
        .reset_index()
        .merge(base.region_class, left_on="region", right_on="hierid") # Merge with region classification to get ISO3 codes
        .assign( # Calculate GDPpc shares by dividing the regional GDPpc by the IMAGE GDPpc
            gdppc_iso3 = 
            lambda d: (d.groupby(['ISO3', "year"])["gdp"].transform("sum") / d.groupby(['ISO3', "year"])["pop"].transform("sum")),
            gdppc_image = 
            lambda d: (d.groupby(['IMAGE', "year"])["gdp"].transform("sum") / d.groupby(["IMAGE", "year"])["pop"].transform("sum")),
            gdppc_shares_ir = # Calculate GDPpc shares at impact region level
            lambda d: d["gdppc"] / d["gdppc_iso3"],
            gdppc_shares_iso3 =  # Calculate GDPpc shares at country level
            lambda d: d["gdppc_iso3"] / d["gdppc_image"],
            gdppc_shares = # Calculate total shares
            lambda d: d["gdppc_shares_ir"] * d["gdppc_shares_iso3"]
        )
    )

    image_shares = (
        gdppc_shares
        [["region", "IMAGE", "year", "gdppc_shares"]]
        .pivot(index=["region", "IMAGE"], columns="year", values="gdppc_shares")
        .reset_index()
        .set_index("region")
        .reindex(base.ir) # Reindex according to hierid
        .reset_index()
    )
    
    country_shares = (
        gdppc_shares
        .query("year==2010")
        [["region", "ISO3", "gdppc_shares_ir"]]
        .rename(columns={"gdppc_shares_ir":"gdppc_share"})
        .set_index("region")
        .reindex(base.ir) # Reindex according to hierid
        .reset_index()
    )

    return image_shares, country_shares



def ReadTIMERFiles(sets):
    
    """
    Read GDPpc data from TIMER output files from the selected scenario and project.
    """
    
    # Define dimensions and timeline for the xarray dataset
    _DIM_TIME = dict(start=1971, end=2100, stepsize=1)
    
    # Define timeline
    Timeline = prism.Timeline(
        start=_DIM_TIME['start'],
        end=_DIM_TIME['end'],
        stepsize=_DIM_TIME['stepsize']
        )
    
    # Define order of TIMER regions
    _DIM_IMAGE_REGIONS = [
        "CAN", "USA", "MEX", "RCAM", "BRA",
        "RSAM", "NAF", "WAF", "EAF", "SAF",
        "WEU", "CEU", "TUR", "UKR", "STAN",
        "RUS", "ME", "INDIA", "KOR", "CHN",
        "SEAS", "INDO", "JAP", "OCE", "RSAS",
        "RSAF"
        ]

    # Add extra regions depending on the file extension (always check if order is right with new files)
    extra_regions = ["dummy", "World"] if sets.gdp_dir[-3:] in ["scn", "dat"] else ["World"]
    prism_regions_world = prism.Dimension('region', _DIM_IMAGE_REGIONS + extra_regions)
    
    listy = []
    
    VAR = "GDPpc"
    
    # Create xarray dataset with the data from the OUT files. 
    datafile = prism.TimeVariable(
            timeline=Timeline,
            dims=[prism_regions_world],
            file=sets.gdp_dir,
        )
    
    listy.append(
        xr.merge(
            [
                datafile[i]
                .rename('Value')
                .expand_dims({"Time": [i]}) for i in np.arange(_DIM_TIME['start'], 2101)
                ]
            )
        .expand_dims({"Scenario": [sets.scenario], "Variable": [VAR]})
        )
    
    xr_vars = xr.merge(listy)
 
    return xr_vars



def ImportClimtasERA5(wdir, year, ir):
    
    # Read climatology data from ERA5 and reindex according to ir dataframe
    climtas = pd.read_csv(wdir+f"/data/ClimateData/Climatologies/ERA5_CLIMTAS_2000-2025.csv")
    climtas = climtas.set_index("hierid").reindex(ir)[str(year)].values

    return climtas



def ImportClimtas(sets, base, temp_mean):
    
    """
    Import climate data from montlhy statistics files. The code calculates the 30-year running
    mean from the mothly data at the grid cell level. The code will later calculate the mean 
    climatology per impact region using "spatial_relation" and will return the data as a numpy array
    ordered by "ir".
    """
    
    print("[1.2.3] Generating climatologies...")
    
    # Load monthly statistics data and calculate 30-year running mean at the grid cell level
    temp = (
        temp_mean
        .mean(dim="NM", skipna=False) # Annual temperature)
        .rolling(time=30, min_periods=30)
        .mean(dim="time", skipna=False) # Set to False assuming IMAGE temp are consistent
    )
    
    # Get the index of the impact regions and the spatial relationship between grid cells and impact regions
    group_idx = base.spatial_relation["index_right"].values
    spatial_idx = base.spatial_relation.index
    
    # Flatten temperature xarray and filter only impact regions index
    temp = temp.stack(grid_cell=("latitude", "longitude")).isel(grid_cell=spatial_idx)
    
    climtas_ir = (
        temp
        .sel(time=slice(f"{sets.years[0]}-01-01", f"{sets.years[-1]}-12-31"))
        .values
    )
    
    climtas_baseline = (
        temp
        .sel(time=slice(f"{sets.base_years[0]}-01-01", f"{sets.base_years[-1]}-12-31"))
        .mean(dim="time")
        .data
    )

    # Aggregate the 30-year running mean temperature at the impact region level using the spatial relationship
    climtas = npg.aggregate(group_idx, climtas_ir, func='nanmean', fill_value=20.0, axis=1)

    # Calculate baseline temperature at the impact region level using the same aggregation method
    temperature_ir_base = npg.aggregate(group_idx, climtas_baseline, func='nanmean', fill_value=20.0)

    return climtas.T, temperature_ir_base



def ShiftERFToTmin(erf, T, tas, tas2, tas3, tas4, tmin): 
    
    """   
    The code will apply the first constraint imposed by the paper (see more in Appendix pp. A62).
    The minimum of the ERF at Present Day temperatures is located between 20 and 30 degrees. 
    Later the function is shifted vertucally to ensure the minimum matches null mortality. 
    This procedure is done for all the ERF per age groups. The tmin remains fixed at future times.
     
    Parameters:
    ----------    
    erf : np.ndarray
        Raw ERFs array result of the fourth degree polynomial (see Appendix pp. A35)
    T : range
        Range of daily temperatures
    tas : np.ndarray
        first degree coefficients of the polynomial (result of 1 + gamma_1,1*climtas + gamma1,2*loggdppc)
    tas2 : np.ndarray
         second degree coefficients of the polynomial (result of 1 + gamma_2,1*climtas + gamma2,2*loggdppc)
    tas3 : np.ndarray
         third degree coefficients of the polynomial (result of 1 + gamma_3,1*climtas + gamma3,2*loggdppc)
    tas4 : np.ndarray
         fourth degree coefficients of the polynomial (result of 1 + gamma_4,1*climtas + gamma4,2*loggdppc)
        
    Returns:
    ----------
    erf : np.ndarray
        2D array with ERFs shifted vertically
    tmin_g : np.ndarray
        1D array with the daily temperature at which the ERF of an impact region are minimized at present values.
    """
    
    if tmin is None:
        # Locate idx of T (temperature array) between 20 and 30 degrees C
        idx_start = np.where(np.isclose(T, 10.0, atol=0.05))[0][0]
        idx_end = np.where(np.isclose(T, 30.0, atol=0.05))[0][0]
        segment = erf[:, :, idx_start:idx_end]
        
        # Find local minimum of erf between 20 and 30 degrees
        idx_local_min = np.argmin(segment, axis=2)
        tmin = T[idx_start + idx_local_min]
        
    # Calcualte mortality value at fixed tmin
    erf_at_tmin = tas*tmin + tas2*tmin**2 + tas3*tmin**3 + tas4*tmin**4
    
    # Shift vertical functions so tmin matches 0 deaths
    erf = erf - erf_at_tmin[:,:,None]
        
    return erf, tmin



def MonotonicityERF(T, erf, tmin):
    
    """
    The code applies the second constraint from the paper (see Appendix pp. A65), weak 
    monotonicity. It ensures that towards colder and hotter temperatures than the tmin, 
    the ERFs must be at least as harmful as temperatures closer to the MMT.
    
    Parameters:
    ----------
    T : range
        Range of daily temperatures
    erf : np.ndarray
        ERFs shifted (see Appendix pp. A35) 
    tmin_g :  np.ndarray
        1D array with the daily temperature at which the ERFs are minimized.
        
    Returns:
    ----------
    erf : np.ndarray
        Array witht the ERFs after weak monotonicity is imposed
    """
    
    # Find index of tmin in T
    idx_tmin = np.searchsorted(T, tmin)
    _, _, nT = erf.shape

    # Create index matrix to vectorize
    idx_matrix = np.arange(nT)[None, None :]
    
    # Mask for temperatures above and below tmin
    mask_left = idx_matrix < idx_tmin[:, :, None]
    mask_right = idx_matrix > idx_tmin[:, :, None]
    
    # Impose weak monotonicity to the left
    left_part = np.where(mask_left, erf, -np.inf)
    left_monotone = np.maximum.accumulate(left_part[:,:,::-1], axis=2)[:,:,::-1]
    
    # Impose weak monotonicity to the right
    right_part = np.where(mask_right, erf, -np.inf)
    right_monotone = np.maximum.accumulate(right_part, axis=2)
    
    # Generate final Exposure Response Function
    erf = np.where(
        mask_left, left_monotone,
        np.where(mask_right, right_monotone, erf)
        )
    
    # Ensure no negative values
    erf = np.maximum(erf, 0)
    
    return erf    
    
    
        
def DailyTemperature2IR(sets, base, tempe, year):
    
    """
    Convert daily temperature data of one year to temperature values at the impact region 
    level. All grid cells intersecting an impact region are considered. Return
    a dataframe with mean daily temperature per impact region for the given year.
    """
    
    print(f"[2.1] Loading daily temperature data for year {year}...")
    
    if "ERA5" in sets.scenario:
        
        # Open daily temperature data from ERA5
        daily_temperature = ERA5Temperature2IR(sets.temp_dir, year, base.spatial_relation)
        
    else:
                
        # Read daily temperature data generated from monthly statistics
        daily_temperature,_ = tmp.DailyFromMonthlyTemperature(
            temperature_mean=tempe.temp_mean,
            temperature_std=tempe.temp_std,
            years_in=year,
            random_vals=base.random_vals,
            to_xarray=False
            )
        
        # Aggregate daily temperature data to impact region level
        daily_temperature = MSTemperature2IR(
            temp=daily_temperature,
            spatial_relation=base.spatial_relation
            )

    return daily_temperature



def MSTemperature2IR(temp, spatial_relation):
    
    """
    Import gridded daily temperature data of one year from montlhy statistics and convert 
    it to the impact region level. Return a dataFrame with daily mean temperature per impact 
    region for the given year.
    """

    # Calculate mean temperature per impact region and round
    daily_temperatures = npg.aggregate(
        spatial_relation["index_right"].values, 
        temp.reshape(-1, temp.shape[-1])[spatial_relation.index], 
        func='nanmean', 
        axis=0, 
        fill_value=20.0 
    )
   
    return np.round(daily_temperatures, decimals=1)



def ERA5Temperature2IR(temp_dir, year, spatial_relation):
    
    """
    Import gridded daily temperature data of one year from ERA5 and convert it
    to the impact region level. Return a dataFrame with daily mean temperature 
    per impact region for the given year.
    """
    
    # Read ERA5 daily temperature data for a specific year
    daily_temperature, _ = tmp.DailyTemperatureERA5(
        era5_dir=temp_dir,
        year=year, 
        temp_type="mean", 
        pop_map=None, 
        to_array=False)
    
    daily_temperature = daily_temperature.t2m.values

    idx_points = spatial_relation.index.values

    daily_temperature = daily_temperature.reshape(daily_temperature.shape[0], -1)[:, idx_points]
    
    daily_temperature = npg.aggregate(
        spatial_relation["index_right"].values, 
        daily_temperature, 
        func='nanmean', 
        axis=1, 
        fill_value=np.nan
    ).T
    
    daily_temperature = np.round(daily_temperature, decimals=1)
    
    return daily_temperature



def CalculateMortalityEffects(sets, base, tempe, scen, erf, year):
    
    """
    The code calculates equation 2a or 2c from the paper, depending whether adaptation is on or off.
    1. It imports/generates the daily temperature data of the selected year at the ir level
    2. Calculates the first part of the equation (called Marginal Mortality here) 
    3. Claculates the second part (the Counterfactual Mortality).
    4. The substraction is done per impact region, age group, and type of temperature (all, heat, cold).
    3. Aggregate the results spatially to the selected region classification (IMAGE26, ISO3...) and 
       append the results in the DataFrame called results.
    """
    
    ### ---------------------- Import daily temperature -----------------------------------
    
    # Read daily temperature data from specified source
    daily_temperature = DailyTemperature2IR(sets, base, tempe, year)
    
    
    ### ---------------------- Calculate marginal mortality --------------------------------
    
    print(f"[2.2] Calculating marginal mortality for year {year}...")
    
    # Calculate marginal mortality (first term of equations 2' or 2a' from the paper)
    mor_heat_min, mor_cold_min = CalculateMarginalMortality(
        sets=sets, 
        base=base,
        tempe=tempe,
        scen=scen,
        erf=erf,
        year=year,  
        daily_temp=daily_temperature,
        counterfactual=False
        )
    
    
    ### ---------------------- Calculate counterfactual mortality -----------------------------
    
    print(f"[2.3] Calculating counterfactual mortality for year {year}...")
    
    if sets.counterfactual == True:

        # Calculate counterfactual mortality (second term of equations 2' or 2a' from the paper)
        if "ERA5" in sets.scenario:
            
            mor_heat_sub, mor_cold_sub = CalculateERA5baselineMortality(
                sets=sets, 
                base=base,
                erf=erf,
                temper=tempe
                )
        
        else:
            mor_heat_sub, mor_cold_sub = CalculateMarginalMortality(
                sets=sets,
                base=base,
                tempe=tempe,
                scen=scen,
                erf=erf,
                year=year,
                daily_temp=tempe.daily_temp_t0,
                counterfactual=True
            )
    
    # Results without counterfactual scenario will be substracted by zero
    elif sets.counterfactual == False:
        
        mor_heat_sub = np.zeros_like(mor_heat_min)
        mor_cold_sub = np.zeros_like(mor_cold_min)
        
        
    ### ---------------------- Locate annual results in array --------------------------------
   
    # Create temporal array
    mor_local = np.full((2, 3, 24378), np.nan, dtype=np.float32)
   
    # Locate mortality from heat in loc 0
    mor_local[0, :, :] = mor_heat_min - mor_heat_sub
    # Locate mortality from cold in loc 1
    mor_local[1, :, :] = mor_cold_min - mor_cold_sub

    return mor_local

        
        
def CalculateERA5baselineMortality(sets, base, erf, tempe):
    
    """
    Calculate "baseline" mortality for a 10-year period, calculating first the annual
    mortality and then averaging accross years. This approach avoids underestimating 
    heat and cold extremes if the mean temperature of the 10-year period was calculated.
    """
    
    # Initialize dics to store annual mortality
    mor_heat, mor_cold = [],[]

    # Calculate annual mortality using preloaded daily baseline temperatures
    for pd_year in sets.base_years:
        mor_heat_year, mor_cold_year = CalculateMarginalMortality(
            sets=sets, 
            base=base,
            erf=erf,
            year=pd_year,
            daily_temp=tempe.daily_temp_t0[pd_year],
            counterfactual=True
            )    
        mor_heat.append(mor_heat_year)
        mor_cold.append(mor_cold_year)

    # Calculate mean mortality of the 10-year period
    mor_heat_sub = np.mean(np.stack(mor_heat), axis=0)
    mor_cold_sub = np.mean(np.stack(mor_cold), axis=0)
    
    return mor_heat_sub, mor_cold_sub

            

def CalculateMarginalMortality(sets, base, tempe, scen, erf, year, daily_temp, counterfactual):
    
    """
    Calculate mortality effects from non optimal temperatures. Depending whether adaptation is on, 
    the code will either import the ERFs with no adaptation or generate ERFs with new income and climtas.
    Mortality per impact region will be calculated per age group and temperature type (all, heat and cold) 
    in the Mortality From Temperature Index function.
    """ 
    
    # Clip daily temperatures to the range of the ERFs
    min_temp = sets.T[0]
    max_temp = sets.T[-1]
    daily_temperature = np.clip(daily_temp, min_temp, max_temp)
    
    # Create rows array for indexing
    rows = np.arange(daily_temperature.shape[0])[:, None]
    
    # ------------------- Generate ERFs ------------------
    
    # Generate ERFs used when there is income growth and adaptation
    if sets.adaptation==True:    
        erfs_t, _ = GenerateERFAll(
            sets=sets,
            base=base,
            tempe=tempe,
            scen=scen,
            erf=erf,
            gammas=erf.gammas,
            year=year,
            adaptation=sets.adaptation,
            counterfactual=counterfactual
            )
    
    # Use pre-calculated ERFs with no adaptation
    else: 
        erfs_t = erf.erfs_t0
        
    # ------------------- Calculate annual mortality ------------------
    
    mor_heat, mor_cold = MortalityFromTemperatureIndex(
        daily_temp=daily_temperature, 
        rows=rows, 
        erf=erfs_t, 
        tmin=erf.tmin_t0,
        min_temp=min_temp, 
        )

    # Return mortality for heat and cold per age group        
    return mor_heat, mor_cold

    

def MortalityFromTemperatureIndex(daily_temp, rows, erf, tmin, min_temp):
    
    """
    The code gets the temperature indices for heat (temperatures above tmin) and 
    cold (temperatures below tmin) to locate the corresponding mortality value from 
    the ERF array and sums the daily mortality values to the annual level. All 
    non-optimal temperatures mortality is the sum of heat and cold mortality.

    Parameters:
    ----------
    daily_temp : np.ndarray
        Daily temperature data per impact region for a given year
        e.g. a daily temperature of -40 will have index 0, daily temperature of 10 will have index 500
    rows : np.ndarray
        Rows array for indexing
    erfs : dic
        Dictionary of the ERFs (store as numpy arrays) per age group
    tmin : dic
        Dictionary of the daily temperature at which the ERF are minimized per age group
    min_temp : float
        Minimum temperature from T: -40.0
    group : str
        Age group
    """

    # Expand tmin to shape (24378, 3, 1) for 3D broadcasting
    tmin = tmin[:, :, None]
    # Expand daily_temp to shape (24378, 1, col_temp) to align with categories
    daily_temp = daily_temp[:, None, :]

    # Align rows to shape (24378, 1, 1) for advanced 3D indexing
    rows_grid = rows[:, None]
    # Create category grid with shape (1, 3, 1) to match the second axis of erf
    cat_grid = np.arange(3)[None, :, None]

    # Calculate column indices (0-600) 
    idx_heat = np.round((np.maximum(daily_temp, tmin) - min_temp) * 10).astype(np.int16)
    idx_cold = np.round((np.minimum(daily_temp, tmin) - min_temp) * 10).astype(np.int16) 

    #  Extract values from erf using advanced indexing and sum along the days axis
    annual_mortality_heat = erf[rows_grid, cat_grid, idx_heat].sum(axis=2)
    annual_mortality_cold = erf[rows_grid, cat_grid, idx_cold].sum(axis=2)
    
    return annual_mortality_heat.T, annual_mortality_cold.T


       
def AggregateRegionalMortality(sets, base, scen, rel_mor):
    
    """
    Use numpy array where annual relative mortality results where store and population data,
    both at the impact region level to calculate first total mortality at the impact region
    level and then aggregate both mortality and population at a bigger region level
    (IMAGE regions and countries), including all ages results, all temperatures and global.
    Mortality and population xarrays for both classificaiton regions are merged into a single 
    dataset to recalculate relative mortality.
    """
    
    # Load population data
    pop = scen.pop[None, :, :, :]
        
    # Calculate total mortality from relative mortality and population
    mor = rel_mor * pop / 1e5
    
    region_datasets = []
        
    for region in ["ISO3", "IMAGE"]:
        
        # Define region characteristics
        regions, index_regions = np.unique(base.region_class[region], return_inverse=True)
        len_regions = len(regions)

        # Define coordinates and dimension of dataset
        coords = {
            "t_type": ["heat", "cold", "all"],
            "age_group": sets.age_groups + ["All ages"],
            "region": regions,
            "year": sets.years
        }

        dims = ["t_type", "age_group", "region", "year"]

        # Generate results at the region level and for all ages and all temperatures
        mor_region = GroupImpactRegions2LargerRegion(
            array=mor, 
            region=region,
            index_regions=index_regions, 
            len_regions=len_regions, 
            coords=coords, 
            dims=dims, 
            name="mortality")

        pop_region = GroupImpactRegions2LargerRegion(
            array=pop,
            region=region,
            index_regions=index_regions,
            len_regions=len_regions,
            coords=coords,
            dims=dims,
            name="population")

        region_datasets.append(xr.merge([pop_region, mor_region], compat="override"))
    
    # Merge all datasets from IMAGE and ISO3 regions
    xarray_unit = xr.concat(region_datasets, dim="region").set_index(geo=["region", "region_type"])
    
    # Recalculate relative mortality in the final step (xarrray now includes global, all ages and all temperatures)
    xarray_unit["relative_mortality"] = xarray_unit["mortality"] * 1e5 / xarray_unit["population"]
    
    # Return only mortality and relative mortality
    return xarray_unit.drop_vars(["population"])

 
 
def GroupImpactRegions2LargerRegion(array, region, index_regions, len_regions, coords, dims, name):
    
    """
    Take an array (mortality or populaiton) wiht annual results at the impact region level and
    aggregate results to a larger region classification (IMAGE or country level [ISO3])
    After merging to larger regions, results are aggregated to obtain mortality and
    population for all tempeeratures and all ages and all-age population. For the 
    IMAGE classification region, results are also merged to get global results. 
    The resulting arrays are converted to xarray format for posterior merging.
    """
    
    # Move regions axis at the end 
    array_T = np.transpose(np.nan_to_num(array, nan=0), (0, 1, 3, 2))
    shape_orig = array_T.shape

    # Flatten all axis but regions axis
    array_flatten = array_T.reshape(-1, shape_orig[-1])

    # Sum values within a region
    array_region = np.array([
        np.bincount(index_regions, weights=row, minlength=len_regions) 
        for row in array_flatten
    ])

    # Return to original shape
    array_shaped = np.transpose(
        array_region.reshape(shape_orig[:-1] + (len_regions,)),
        (0, 1, 3, 2)
    )

    # Generate "all ages" age group for both mortality and population
    array_shaped = np.concatenate(
        [array_shaped, np.sum(array_shaped, axis=1, keepdims=True)], 
        axis=1
    )
    
    if len_regions == 27:
        
        # Generate World results
        array_shaped = np.concatenate(
            [array_shaped, np.sum(array_shaped, axis=2, keepdims=True)], 
            axis=2
        )
        
        if "World" not in coords["region"]:
            coords["region"] = np.append(coords["region"], "World")
        
    
    # If data from mortality (heat and cold)
    if array.shape[0] == 2:
        
        # Generate mortality data for all temperatures
        array_shaped = np.concatenate(
            [array_shaped, np.sum(array_shaped, axis=0, keepdims=True)], 
            axis=0
        )
        
    else:
        # Remove t_type for population dataset
        dims=dims[1:]
        del coords["t_type"]
        array_shaped = array_shaped[0]
    
    # Convert to dataset
    var_dataset = xr.Dataset(
        data_vars={
            name: (dims, array_shaped)
        },
        coords=coords
    )
    
    return var_dataset.assign_coords(region_type=("region", [region] * len(var_dataset.region))) 

 

def PostprocessResults(sets, base, scen, rel_mor):
    
    """
    Postprocess final results and save to CSV file in output folder.
    1. Calculate total mortality and relative mortality for all-ages group, all temperatures and
    global results by summing the results of the iteration.
    2. If sets.reporting_tool is True, change the format of the results to match the IAM one for
    the IMAGE classification regions.
    3. Save results in main working directory.
    """
    
    print("[3] Postprocessing and saving results...")
    
    # Calculate total mortality and relative mortality for all-ages group
    results = AggregateRegionalMortality(sets, base, scen, rel_mor)
    
    
    if sets.reporting_tool != False:
        # Append results to reporting tool of corresponding scenario
        Export2ReportingTool(sets, results)


    # ------------------- Save results --------------------------------------

    # Define naming parameters    
    if sets.adaptation == True:
        adaptation = ""
    else:
        adaptation = "_NoAdap"
    if sets.project is not None:
        project = f"{sets.project}"
    else:
        project = ""
    if str(sets.draw).lower() == "mean":
        draw = "_mean"
    else: 
        draw = f"_{sets.draw}"
        
    # Create folder to sabe files if not there yet
    output_dir = sets.wdir + "/output/" + f"{sets.project}"  
    os.makedirs(output_dir, exist_ok=True)   

                    
    results = results.reset_index("geo")

    compresion_config= {
                "dtype": "float32",       
                "zlib": True,
                "complevel": 6,
    } 

    encoding_total = {
        var: compresion_config for var in results.data_vars
    }
        
    results.to_netcdf(
        output_dir +
        f"/mortality_{project}_{sets.scenario}{adaptation}_{sets.years[0]}-{sets.years[-1]}{draw}.nc",
        encoding=encoding_total
    )

    print("Scenario ran successfully!")
    
    
            
def Export2ReportingTool(sets, results):
    
    # Select only IMAGE regions + World
    results = (
        results
        .sel(region_type="IMAGE")
        .drop_vars("region_type")
        .rename_vars({"mortality":"Mortality", "relative_mortality": "Relative Mortality"})
        .rolling(year=5, center=True, min_periods=1)
        .mean() # Calculate 5-year rolling mean to smooth the results, avoiding aliasing effects
        .to_dataframe()
        .assign( # Convert mortality to thousands of deaths to match the format of the reporting tool
            Mortality = lambda d: d["Mortality"] / 1000
            ) 
        .reset_index()
        .melt( # Put Mortality and Relative Mortality as single column
            id_vars=["age_group", "year", "t_type", "region"],
            value_vars=["Mortality", "Relative Mortality"]
            )
        .pivot( # Put years in columns
            index=["age_group", "t_type", "region", "variable"], 
            columns="year", values="value"
            )
        .reset_index()
    )
    
    results = results.sort_values(by=[results.columns[0], results.columns[1], results.columns[3], results.columns[2]], 
                       ascending=[True, True, True, True])

    # Dictionary to map age groups to the format of the reporting tool
    map_age = {
        "young": "|Age 0-4",
        "older": "|Age 5-64",
        "oldest": "|Age 65+"
    }

    df_rt = pd.DataFrame(index=results.index)

    df_rt["Model"] = "IMAGE"
    df_rt["Scenario"] = sets.scenario
    df_rt["Region"] = results["region"]
    df_rt["Variable"] = (
        "Health|Mortality|Non-Optimal Temperatures"
        + np.where(results["t_type"].str.capitalize() != "All", "|" + results["t_type"].str.capitalize(), "")
        + results["age_group"].map(map_age).fillna("")
        + np.where(results["variable"] == "Relative Mortality", " [per 100,000 people]", "")
        ).str.rstrip("|")
    df_rt["Unit"] = np.where(results["variable"] == "Relative Mortality", "-", "thousand")


    # Load original reporting tool
    rt_path = f"{sets.reporting_tool}/{sets.scenario}.xlsx"
    wb = load_workbook(rt_path)
    rt_data = wb["data"]

    # Years reported
    years = [int(cell.value) for cell in rt_data[1][5:] if cell.value is not None]
    df_rt[[col for col in years if col in results.columns]]=results[[col for col in years if col in results.columns]]
    
    for row in dataframe_to_rows(df_rt, index=False, header=False):
        rt_data.append(row)
        
    wb.save(sets.reporting_tool+"/including_health_impacts/"+sets.scenario+".xlsx")