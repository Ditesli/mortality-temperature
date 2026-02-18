import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from dataclasses import dataclass
from shapely.geometry import Polygon
import re, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils_common import temperature as tmp
import prism



### ------------------------------------------------------------------------------
def CalculateMortality(
    wdir: str,
    years: list,
    temp_dir: str,
    gdp_dir: str,
    project: str,
    scenario: str,
    regions: str,
    adaptation: bool,
):

    paths = ModelPaths(
        temp_path=temp_dir,
        income_path=gdp_dir,
        wdir=wdir
    )

    model = MortalityModel(
        paths=paths,
        years=years,
        project=project,
        scenario=scenario,
        regions=regions,
        adaptation=adaptation,
    )

    model.run()
    


@dataclass
class ModelPaths:
    temp_path: str
    income_path: any
    wdir: str
    
    def climate_path(self, project: str, scenario: str) -> str:
        """
        Set path to climate data depending on the scenario type 
        (IMAGE or other scenarios)
        """
        if self.temp_path == self.income_path:
            return (
                f"{self.temp_path}/"
                f"{project}/3_IMAGE_land/scen/"
                f"{scenario}/netcdf/"
            )
        else:
            return self.temp_path



@dataclass
class MortalityModel:
    paths: ModelPaths
    years: list
    project: str
    scenario: str
    regions: str
    adaptation: bool
    
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
    paths : ModelPaths
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
        
    Returns:
    ----------
    None
    Saves the mortality results to CSV files in the output folder.
    """
    

    def validate_years(self):
        ERA5_START_YEAR = 2000
        ERA5_END_YEAR = 2025

        if re.search(r"SSP[1-5]_ERA5", self.scenario):
            self.years = [
                y for y in self.years
                if ERA5_START_YEAR <= y <= ERA5_END_YEAR
            ]


    def load_inputs(self):
        self.res = LoadInputData.from_files(
            paths=self.paths,
            regions=self.regions,
            project=self.project,
            scenario=self.scenario,
            years=self.years,
        )

        self.baseline = BaselineERFsInputs.fromPresentDay(
            paths=self.paths,
            project=self.project,
            scenario=self.scenario,
            adaptation=self.adaptation,
            res=self.res
        )
        
        
    def run(self):
        print(f"Starting mortality model for {self.scenario}")
        self.validate_years()
        self.load_inputs()

        print("[2] Starting mortality calculations...")

        for year in self.years:
            CalculateMortalityEffects(
                paths=self.paths,
                year=year,
                scenario=self.scenario,
                adaptation=self.adaptation,
                regions=self.regions,
                res=self.res,
                baseline=self.baseline
            )

        self.postprocess()
        
        
    def postprocess(self):
        PostprocessResults(
            wdir=self.paths.wdir,
            res=self.res,
            years=self.years,
            scenario=self.scenario,
            adaptation=self.adaptation
        )

    
    
@dataclass
class LoadInputData:
    
    """
    Container for all input data required to run the model.

    Attributes
    ----------
    age_groups : list
        Age group labels used in the model.
    T : np.ndarray
        Temperature range for exposure-response functions.
    spatial_relation : GeoDataFrame
        Spatial relationship between temperature grid cells and impact regions.
    ir : pd.DataFrame
        Impact region identifiers used to align new data.
    region_class : DataFrame
        DataFrame with region classification selected and corresponding ir.
    results : DataFrame
        DataFrame to store final results
    gammas : dict
        Dictionary with gamma coefficients to generate ERFs.
    pop : DataFrame
        Population data from selected SSP scenario and/or historical population.
    """
    
    age_groups: list
    T : np.ndarray
    spatial_relation: gpd.GeoDataFrame
    ir: pd.DataFrame
    region_class: pd.DataFrame
    results: pd.DataFrame
    gammas: any
    pop: pd.DataFrame
    base_years: list = range(2001,2011)

    @classmethod
    def from_files(cls, paths, regions, project, scenario, years):
        
        """
        Read and load all input files required for mortality calculations. The 
        necessary data is located in the wdir/data folder.  
        """
        
        print("[1] Loading input files and defining parameters...")
        
        AGE_GROUPS = ["young", "older", "oldest"]
        
        T = np.arange(-20, 40.1, 0.1).round(1)
    
        print(f"[1.2] Loading region classification: {regions}...")
        region_class = pd.read_csv(f"{paths.wdir}/data/regions/region_classification.csv")
        if regions == "impact_regions":
            REGION_CLASS = region_class[["hierid", "ISO3"]]
        if regions == "countries":
            REGION_CLASS = region_class[["hierid", "ISO3", "gbd_level3"]]
        else:
            REGION_CLASS = region_class[["hierid", "ISO3", regions]]
        
        SPATIAL_RELAITON, IR = GridRelationship(paths, project, scenario, years)
        
        RESULTS = FinalDataframe(regions, region_class, AGE_GROUPS, years)
        
        GAMMA_COEFFS = ImportGammaCoefficients(paths.wdir)
        
        POPULATION = ImportPopulationData(paths.wdir, scenario, years, AGE_GROUPS, IR)    
    
        return cls(
            age_groups=AGE_GROUPS,
            T = T,
            spatial_relation=SPATIAL_RELAITON,
            ir=IR,
            region_class=REGION_CLASS,
            results=RESULTS,
            gammas = GAMMA_COEFFS,
            pop = POPULATION
        )
    


@dataclass
class BaselineERFsInputs:
    
    """
    Container for all input data required to generate the baseline ERFs (ERFs for the 
    "present day", and as the basis to generate the ERFs with adaptation).
    
    Attributes
    ----------
    climtas_t0: np.ndarray
        1D array with the 30 year climate mean (2001-2010) per impact region.
    loggdppc_t0 : np.ndarray
        1D array with the log of the 13 year gdppc mean (2001-2010) per impact region.
    erfs_t0: any
        Dictionary with each age group's ERFs per impact region.
    tmin_t0: any
        Dictionary with the three 1-d arrays corresponding to the minimum temperature (MMT).
    image_shares: any
        GDPpc shares of regions within an IMAGE region. Used to downscale GDPpc.
    country_shares: any
        GDPpc shares of regions within a country. Used to downscale GDPpc.
    image_gdppc: any
        GDPpc per IMAGE region. Used only in IMAGE sscenarios.
    daily_temp_t0: pd.DataFrame
        DataFrame with daily "present day" temperature data for the counterfactual part.
    """
    
    climtas_t0: np.ndarray
    loggdppc_t0 : np.ndarray
    erfs_t0: any
    tmin_t0: any
    image_shares: any
    country_shares: any
    image_gdppc: any
    daily_temp_t0: pd.DataFrame
    
    
    def fromPresentDay(paths, project, scenario, adaptation, res):
        
        # Import present day covariates
        print("[1.4] Loading 'present day' covariates climtas and loggdppc...")
        
        CLIMTAS_T0, LOGGDPPC_T0 = ImportCovariates(
            paths=paths, 
            res = res,
            scenario=scenario, 
            year=None, 
            adaptation=False, 
            baseline = None,
            counterfactual=None
            )
        
        # Generate a single time 'present day' ERFs (no adaptation)
        ERFS_T0, TMIN_T0 = GenerateERFAll(
            paths=paths, 
            res = res,
            scenario=scenario, 
            year=None, 
            adaptation=False, 
            baseline = None,
            counterfactual=None
            ) 
        
        print("[1.5] Loading 'present-day' temperature data...")
        
        # Import present day temperatures
        DAILY_TEMP_T0 = ImportPresentDayTemperatures(
            paths=paths, 
            scenario=scenario, 
            base_years=range(2001,2011), 
            ir=res.ir, 
            spatial_relation=res.spatial_relation
            )
        
        #  Read GDP shares for scenarios that do not use Carleton's socioeconomic data.
        
        if adaptation:
                
            print("[1.6] Loading GDPpc shares at the imapct region level...")
            # Generate GDPpc shares of regions within a country and IMAGE region
            IMAGE_SHARES, COUNTRY_SHARES = GenerateGDPpcShares(
                wdir=paths.wdir, 
                ir=res.ir, 
                region_class=res.region_class)
            IMAGE_GDPPC = None
            
            if not re.search(r"SSP[1-5]", scenario) and "carleton" not in scenario.lower():
                
                print("[1.7] Loading GDP data from IMAGE...")
                # Open TIMER gdp file and calculate regional GDP from IMAGE-regional shares
                IMAGE_GDPPC = ReadOUTFiles(
                    gdp_dir=paths.income_path,
                    project=project, 
                    scenario=scenario)
                
        # Set to None when adaptation is off        
        else:  
            IMAGE_SHARES = None
            IMAGE_GDPPC = None
            COUNTRY_SHARES = None
            
            
        return BaselineERFsInputs(
            climtas_t0 = CLIMTAS_T0,
            loggdppc_t0 = LOGGDPPC_T0,
            erfs_t0 = ERFS_T0,
            tmin_t0 = TMIN_T0,
            image_shares = IMAGE_SHARES,
            country_shares=COUNTRY_SHARES,
            image_gdppc = IMAGE_GDPPC,
            daily_temp_t0 = DAILY_TEMP_T0
        )



def GridRelationship(paths, project, scenario, years):
    
    """
    Create a DataFrame with the spatial relationship between temperature data points 
    and impact regions. It will assign each grid cell to the impact region it intersects with.
    If a grid cell it will be assigned multiple times. The function can work with any resolution 
    of temeprature data. 
    Create a pandas series of the impact regions to align the order of the regions in the rest 
    of the dataframes with the same order as the spatial relationship dataframe. 
    """
    
    print("[1.1] Creating spatial relationship between temperature grid and impact regions...")
    
    # Define function that creates grid cells
    def CreateSquare(lon, lat, lon_size, lat_size): 
        
        """
        Return a square Polygon centered at (lon, lat).
        Function only works for climate data with squared grids.
        """
        
        return Polygon([
            (lon, lat),
            (lon + lon_size, lat),
            (lon + lon_size, lat + lat_size),
            (lon, lat + lat_size)
        ])

    # --------------------- Read climate data ----------------------
    
    # ---------- If ERA5 data ----------
    if re.search(r"ERA5", scenario):
        # Use function located in the utils_common folder to import ERA5 data in the right format
        GRID,_ = tmp.DailyTemperatureERA5(
            era5_dir=paths.temp_path, 
            year=years[0], 
            temp_type="mean", 
            pop_ssp=None, 
            to_array=False
            )
    
    # --------- If Monthly Statistics (MS) data ----------  
    else:
        
        # --------- IMAGE scenario climate data ----------
        if paths.temp_path == paths.income_path:
            PATH = paths.temp_path + "/" + project + "/3_IMAGE_land/scen/" + scenario + "/netcdf/"
        # --------- Other MS scenario data ----------
        else:
            PATH = paths.temp_path 
        
        # Use function to import monthly statistics (MS) of daily temperature data in the right format
        GRID,_ = tmp.DailyFromMonthlyTemperature(
            temp_dir=PATH, 
            years=years[0], 
            temp_type="MEAN", 
            std_factor=1, 
            to_xarray=True
            )
        
    
    # Extract coordinates
    def FindCoordinateName(possible_names, coord_names, temperature):
    
        for name in possible_names:
            if name in coord_names:
                return temperature[name].values
        raise KeyError(f"No coordinate was found among: {possible_names}")
    
    COORD_NAMES = GRID.coords.keys()
    LON_VALS = FindCoordinateName(["lon", "longitude", "x"], COORD_NAMES, GRID)
    LAT_VALS = FindCoordinateName(["lat", "latitude", "y"], COORD_NAMES, GRID)

    # Create meshgrid 
    LON2D, LAT2D = np.meshgrid(LON_VALS, LAT_VALS)  
    
    # Create GeoDataFrame with points and their corresponding square polygons
    POINTS_GDF = gpd.GeoDataFrame({
        "longitude": LON2D.ravel(),
        "latitude": LAT2D.ravel(),
        "geometry": [
            CreateSquare(lon, lat, np.abs(np.mean(np.diff(LON_VALS))),  np.abs(np.mean(np.diff(LAT_VALS))))
            for lon, lat in zip(LON2D.ravel(), LAT2D.ravel())
        ]
    })
    
    # Load .shp file with impact regions and set the same coordinate reference system (CRS)
    IR = gpd.read_file(paths.wdir + "/data/carleton_sm/ir_shp/impact-region.shp")
    POINTS_GDF = POINTS_GDF.set_crs(IR.crs, allow_override=True)
    
    # Make spatial join
    RELATIONSHIP = gpd.sjoin(POINTS_GDF, IR, how="inner", predicate="intersects")

    return RELATIONSHIP[["geometry", "index_right", "hierid"]], IR["hierid"]



def FinalDataframe(regions, region_class, age_groups, years):
    
    """
    Create results dataframe with multiindex for age groups, temperature types,
    mortality types (total mortality and relative mortality) and regions.
    """
    
    REGIONS = region_class[f"{regions}"].unique()
    REGIONS = REGIONS[~pd.isna(REGIONS)]
    REGIONS = np.append(REGIONS, "World")
    
    AGE_GROUPS = np.append(age_groups, "all population")
    TEMPERATURE_TYPES = ["Heat", "Cold", "All"]
    MORTALITY_TYPES = ["Total Mortality", "Relative Mortality"]
    
    # Create results multiindex dataframe
    RESULTS = (
        pd.DataFrame(
            index=pd.MultiIndex.from_product([AGE_GROUPS, TEMPERATURE_TYPES, MORTALITY_TYPES, REGIONS],
                                         names=["age_group", "t_type", "units", regions]), 
            columns=years
        )
    )
    
    return RESULTS 



def ImportGammaCoefficients(wdir):    
    
    """
    Import gamma coefficients from the paper's Suplementary Material and convert 
    them to the right format to be multiplied later on by the covariates
    (climtas y loggdppc).
    
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
    
    with open(wdir+"data/carleton_sm/Agespec_interaction_response.csvv") as f:
        
        # Extract relevant lines
        for i, line in enumerate(f, start=1):

            if i == 21:
                # Extract 1, climtas, loggdppc
                COVAR_NAMES = [x for x in line.strip().split(", ")]
                # Convert to indices and convert to array
                COVAR_MAP = {"1":0, "climtas":1, "loggdppc":2}
                COVAR_IDX = np.array([COVAR_MAP[str(x)] for x in COVAR_NAMES])
                
            if i == 23:
                # Extract gamma coefficients
                GAMMAS = np.array([float(x) for x in line.strip().split(", ")])
                
    return GAMMAS.reshape(3,12), COVAR_IDX.reshape(3,12)



def ImportPopulationData(wdir, scenario, years, age_groups, ir):
    
    print(f"[1.3] Loading Population data for {scenario} scenario at the impact regions level...")
    
    # Extract SSP from scenario string
    MATCH = re.search(r"(?i)\bssp\d+", scenario)
    # Extract corresponding SSP scenario
    SSP = MATCH.group().upper()
    
    # Include ALWAYS population data from 2000 to 2010 (used in the subtrahend part)
    YEARS = sorted(set(years).union(range(2000, 2011)))
        
    # Import population data based on scenario type
    if 'carleton' in scenario.lower() or re.search(r"SSP[1-5]_ERA5", scenario):
        # Import population data from the paper
        POPULATION = ImportDefaultPopulationData(wdir, SSP, YEARS, age_groups, ir)
        
    else:
        # Import IMAGE population data nc4 file and calculate population per impact region
        POPULATION = ImportIMAGEPopulationData(wdir, SSP, YEARS, age_groups, ir)
    
    return POPULATION



def ImportDefaultPopulationData(wdir, ssp, years, age_groups, ir): 
    
    """
    Read default population file for a given SSP scenario and age group
    and convert it to the right format (dataframe) to be used later on in the model in the 
    spatial aggregation of mortality. 
    
    Returns:
    ----------
    POPULATION_GROUPS : dict
        Dictionary with population dataframes per age group
    """
    
    POPULATION_GROUPS = {}
    age_pop_names = ['pop0to4', 'pop5to64', 'pop65plus']
    
    for age_group, age_name in zip(age_groups, age_pop_names):
        
        # Read 'present-day' population data
        POP_HISTORICAL = (
            pd.read_csv(f"{wdir}/data/population/pop_historical/POP_historical_{age_group}.csv")
            .set_index("hierid")
        )
    
        POP_SSP = (
            xr.open_dataset(wdir+f"data/carleton_sm/econ_vars/{ssp.upper()}.nc4")[age_name]
            .sel(model="low") # Select any GDP model
            .to_dataframe() # Convert to dataframe
            .drop(columns=['ssp', 'model'])
            .unstack('year') # Reshape to have years as columns
            .pipe(lambda df: df.set_axis(df.columns.get_level_values(-1), axis=1))
            [[y for y in years if y >= 2023]] # Keep only years from 2023 onwards 
            .merge(POP_HISTORICAL, left_index=True, right_index=True) # Merge with historical population 
            .pipe(lambda df: df.set_axis(df.columns.astype(str), axis=1)) # Convert year columns to string 
            .reindex(ir.values) # Align to impact regions order
            .reset_index()
            .rename(columns={"region":"hierid"})
        )
        
        POPULATION_GROUPS[age_group] = POP_SSP
    
    return POPULATION_GROUPS



def ImportIMAGEPopulationData(wdir, ssp, years, age_groups, ir):
    
    """
    Read precalculated IMAGE population data at the impact region level for a
    given SSP.

    Returns:
    ----------
    POP_SSP : dict
        Dictionary with population data per age group
    """
    
    POP_SSP = {}
    
    for age_group in age_groups:
        POP_SSP_GROUP = (
            pd.read_csv(f"{wdir}/data/population/pop_ssp/pop_{ssp.lower()}_{age_group}.csv")
            .pipe(lambda df: df.filter(
                ["hierid"] +
                [c for c in df.columns if c.isdigit() and int(c) in years]
            ))
            .set_index("hierid")
            .reindex(ir.values) # Align to impact regions orders
            .reset_index()
        )
    
        POP_SSP[age_group] = POP_SSP_GROUP
    
    return POP_SSP



def GenerateERFAll(paths, res, scenario, year, adaptation, baseline, counterfactual):
    
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
    CLIMTAS, LOGGDPPC = ImportCovariates(
        paths=paths,
        res = res,
        scenario=scenario, 
        year=year, 
        adaptation=adaptation, 
        baseline = baseline,
        counterfactual=counterfactual
        )

    # Covariates matrix
    COVARIATES = np.column_stack([np.ones(len(CLIMTAS)), CLIMTAS, LOGGDPPC])   

    # Generate arrays with erf and tmin per age group
    MOR_NP = {}; TMIN = {} 
    for i, group in enumerate(res.age_groups):
        
        if baseline is None:
            ERFS_T0 = None
            TMIN_T0 = None
        else:
            ERFS_T0 = baseline.erfs_t0[group]
            TMIN_T0 = baseline.tmin_t0[group]   
            
        # List of locations of gamma and covariates
        G = res.gammas[0][i]; C = res.gammas[1][i]

        # Multiply each covariate by its corresponding gamma
        BASE = COVARIATES[:, C] * G
        
        # Compute the sum of the covariates to get polynomial coefficients
        TAS = BASE[:, 0:3].sum(axis=1)  
        TAS2 = BASE[:, 3:6].sum(axis=1)  
        TAS3 = BASE[:, 6:9].sum(axis=1) 
        TAS4 = BASE[:, 9:12].sum(axis=1)

        # Generate raw Exposure Response Function
        ERF_RAW = (
            TAS[:,None] * res.T**1 +
            TAS2[:,None] * res.T**2 +
            TAS3[:,None] * res.T**3 +
            TAS4[:,None] * res.T**4
        )
        
        # Impose zero mortality at tmin by vertically shifting erf
        ERF_SHIFTED, TMIN_G = ShiftERFToTmin(ERF_RAW, res.T, TAS, TAS2, TAS3, TAS4, TMIN_T0)
        
        #  # Ensure ERFs do not exceed no-adaptation ERFs 
        if ERFS_T0 is not None:
            ERF_SHIFTED = np.minimum(ERF_SHIFTED, ERFS_T0)
        
        # Impose weak monotonicity to the left and the right of the erf
        MOR_NP[group] = MonotonicityERF(res.T, ERF_SHIFTED, TMIN_G)
        TMIN[group] = TMIN_G
        
    return MOR_NP, TMIN



def ImportCovariates(paths, res, scenario, year, adaptation, baseline, counterfactual):
    
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
        COVARIATES_T0 = (
             pd.read_csv(paths.wdir+"data/carleton_sm/main_specification/mortality-allpreds.csv")
            .rename(columns={"region":"hierid"})
            .set_index("hierid")
            .reindex(res.ir.values)
        )
        
        # Extract only climtas and loggdppc as arrays
        CLIMTAS = COVARIATES_T0["climtas"].values
        LOGGDPPC = COVARIATES_T0["loggdppc"].values
    
    # ADAPTATION ---------------------------------------------
    else:
        
        # CLIMTAS ---------------------------
        
        # Load "present-day" climatology
        if counterfactual:
            if re.search(r"ERA5", scenario):
                CLIMTAS = ImportClimtasERA5(paths.wdir, year, res.ir, present_day=True)
            else: 
                CLIMTAS = ImportClimtas(paths.temp_path, None, res.spatial_relation, present_day=True)
        
        # Load climatology of selected year 
        else:
            if re.search(r"ERA5", scenario):
                CLIMTAS = ImportClimtasERA5(paths.wdir, year, res.ir, present_day=False)
            else:
                CLIMTAS = ImportClimtas(paths.temp_path, year, res.spatial_relation, present_day=False)
                
        # log(GDPpc) ---------------------------    
        
        if re.search(r"ERA5", scenario) or ("carleton" in scenario.lower() and year < 2010):
            # Load historical log(GDPpc) from World Bank
            LOGGDPPC = ImportHistoricalLogGDPpc(paths.wdir, res.ir, year, baseline.country_shares)
        
        elif "carleton" in scenario.lower() and year >= 2010:  
            # Load log(GDPpc) from Carleton et al. (2022) for the selected year and scenario
            LOGGDPPC = ImportCarletonLogGDPpc(paths.wdir, scenario, res.ir, year)
            
        else: 
            # Load log(GDPpc) at the impact region level using the GDPpc output from IMAGE
            LOGGDPPC = ImportIMAGEloggdppc(year, baseline.image_gdppc, baseline.image_shares)
            
    return CLIMTAS, LOGGDPPC



def ImportHistoricalLogGDPpc(wdir, ir, year, country_shares):
    
    """
    Read historical GDP per capita data (GDP per capita (constant 2015 US$)) from
    the World Bank and calculate the 13 year running mean for the selected year at 
    the impact region level using the factors derived from the original paper that
    downscale national GDPpc to the regional one.
    """
    
    if year == 2025:
        year = 2024 # The latest year with GDPpc data available is 2024, so we will use that for 2025 as well.
    
    # Read GDPpc
    GDPPC = (
        pd.read_csv(wdir + "data/income_data/historical_gdppc/WB_WDI_NY_GDP_PCAP_KD.csv")
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

    GDPPC["gdppc"] = GDPPC["OBS_VALUE_13yr_mean"] * GDPPC["gdppc_share"]
    GDPPC["loggdppc"] = np.log(GDPPC["gdppc"])
    
    return GDPPC.set_index("region").reindex(ir)["loggdppc"].values



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
    
    SCENARIO = re.search(r"(?i)\bssp\d+", scenario).group()
        
    # Read GDP per capita file
    GDPPC = (
        xr.open_dataset(wdir+f"data/carleton_sm/econ_vars/{SCENARIO.upper()}.nc4")   
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
        .reindex(ir.values) # Reindex according to hierid
    )
    
    # Calculate log(GDPpc)
    GDPPC["loggdppc"] = np.log(GDPPC["gdppc"])
    
    # Return numpy array
    return GDPPC["loggdppc"].values



def ImportIMAGEloggdppc(year, image_gdppc, image_shares):
    
    """
    Calculate log(GDPpc) at the impact region level using the GDPpc output from a 
    TIMER run and the shares that downscale IMAGE GDPpc at the impact region level.
    """
    
    # Extract relevant year data (13 year rolling mean)
    IMAGE_GDPPC = (
        image_gdppc
        .sel(Time=slice(year-13,year))
        .mean(dim="Time")
        .mean(dim="Scenario")
        .mean(dim="Variable")
        .pint.dequantify() # Remove pint units and warning
        .to_dataframe()
        .reset_index()
    )
    
    # Merge IMAGE GDPpc with GDPpc shares
    GDPPC = image_shares.merge(IMAGE_GDPPC, left_on="IMAGE26", right_on="region", how="left")
    
    # Calculate share of log(GDPpc) based on regional GDPpc
    GDPPC["gdppc"] = GDPPC["Value"] * GDPPC["gdppc_share"] 
    GDPPC["loggdppc"] = np.log(GDPPC["gdppc"])
    
    return GDPPC["loggdppc"].values

 

def GenerateGDPpcShares(wdir, ir, region_class):
    
    """
    Generate the corresponding GDPpc shares per impact region within an IMAGE region and 
    country. The function will read the GDPpc data from Carleton et al. (2022) and calculate 
    the factor that downscales the GDPpc. The final output will be a dataframe with the GDPpc
    shares per impact region.
    """

    # Open GDP data (can be any SSP)
    GDPPC_CETAL_DF = (
        xr.open_dataset(f"{wdir}/data/carleton_sm/econ_vars/SSP2.nc4")
        .to_dataframe() # Convert to dataframe
        .reset_index()
        .merge(region_class, left_on="region", right_on="hierid")
        .query("model == 'high' and year == 2010") # Filter
        .drop(["model", "year", "ssp"], axis=1)
    )
    
    def compute_shares(df, group_var):
        return (
            df.assign(
                gdppc_ir_shares=lambda d: d["gdppc"] / d.groupby(group_var)["gdppc"].transform("sum"),
                gdppc_times_shares=lambda d: d["gdppc"] * d["gdppc_ir_shares"],
                gdppc_country=lambda d: d.groupby(group_var)["gdppc_times_shares"].transform("sum"),
                share_country_gdp=lambda d: d["gdppc_times_shares"] / d["gdppc_country"],
                gdppc_share =  lambda d: d["share_country_gdp"] / d["gdppc_ir_shares"]
            )
            [["region", group_var, "gdppc_share"]]
            .set_index("region")
            .reindex(ir.values)
            .reset_index()
        )
    
    # Calculate IMAGE and country shares
    IMAGE_SHARES = compute_shares(GDPPC_CETAL_DF, "IMAGE26")
    COUNTRY_SHARES = compute_shares(GDPPC_CETAL_DF, "ISO3")
        
    return IMAGE_SHARES, COUNTRY_SHARES



def ReadOUTFiles(gdp_dir, project, scenario):
    
    """
    Read GDPpc data from TIMER output files from the selected scenario and project.
    """
    
    # Define dimensions and timeline for the xarray dataset
    _DIM_TIME = dict(start=1971, end=2100, stepsize=1)
    _DIM_IMAGE_REGIONS = ["CAN", "USA", "MEX", "RCAM", "BRA",
                        "RSAM", "NAF", "WAF", "EAF", "SAF",
                        "WEU", "CEU", "TUR", "UKR", "STAN",
                        "RUS", "ME", "INDIA", "KOR", "CHN",
                        "SEAS", "INDO", "JAP", "OCE", "RSAS",
                        "RSAF"]
    Timeline = prism.Timeline(start=_DIM_TIME['start'],
                            end=_DIM_TIME['end'],
                            stepsize=_DIM_TIME['stepsize'])
    prism_regions_world = prism.Dimension('region', _DIM_IMAGE_REGIONS + ["World"])
    
    listy = []

    # Read GDPpc data path from specific scenario.
    path_clim = gdp_dir + "/" + project + "/2_TIMER/outputlib/TIMER_3_4/" + project + "/"+ scenario + "/indicators/Economy/"
    
    # TODO: Change variable name as it is GDPpc with or without impacts depending on scenario
    VAR = "GDPpc_incl_impacts"
    
    # Create xarray dataset with the data from the OUT files. 
    datafile = prism.TimeVariable(
            timeline=Timeline,
            dims=[prism_regions_world],
            file=path_clim+VAR+".OUT",
        )
    
    listy.append(xr.merge([datafile[i]
                           .rename('Value')
                           .expand_dims({"Time": [i]}) for i in np.arange(_DIM_TIME['start'], 2101)])
                 .expand_dims({"Scenario": [scenario], "Variable": [VAR]}))
    xr_vars = xr.merge(listy)
 
    return xr_vars


def ImportClimtasERA5(wdir, year, ir, present_day):
    
    CLIMTAS = pd.read_csv(wdir+f"data/climate_data/era5/climatologies/ERA5_CLIMTAS_2000-2025.csv")
    
    if present_day == False:
        CLIMTAS = CLIMTAS.set_index("hierid").reindex(ir)[str(year)].values
    
    else:
        COLS = [str(y) for y in range(2001, 2011)]
        CLIMTAS = CLIMTAS.set_index("hierid").reindex(ir)[COLS].mean(axis=1).values

    return CLIMTAS



def ImportClimtas(temp_path, year, spatial_relation, present_day):
    
    """
    Import climate data from montlhy statistics files. The code calculates the 30-year running
    mean from the mothly data at the grid cell level. The code will later calculate the mean 
    climatology per impact region using "spatial_relation" and will return the data as a numpy array
    ordered by "ir".
    """
    
    if present_day==True:
        YEAR = slice("2001-01-01", "2011-01-01") # The "present-day" climatology 
    else:
        YEAR = slice(f"{year}-01-01", f"{year}-12-31")
    
    # Read monthly mean of daily mean temperature data
    CLIMTAS_IR = (
        xr.open_dataset(temp_path+f"GTMP_MEAN_30MIN.nc")
        ["GTMP_MEAN_30MIN"]
        .mean(dim="NM") # Annual temperature
        .rolling(time=30, min_periods=1) 
        .mean() # Climatology
        .sel(time=YEAR)
        .mean(dim="time") # "present-day" climatology
        .values
        .ravel()
        [spatial_relation.index] # Assign pixels to every impact region using spatial relation
    )

    # Calculate mean temperature per impact region and round
    CLIMTAS = (
        pd.DataFrame(CLIMTAS_IR, index=spatial_relation["index_right"])
        .groupby("index_right")
        .mean()
        .fillna(20)
        [0].values
    )

    return CLIMTAS



def ShiftERFToTmin(raw, T, tas, tas2, tas3, tas4, tmin): 
    
    """   
    The code will apply the first constraint imposed by the paper (see more in Appendix pp. A62).
    The minimum of the ERF at Present Day temperatures is located between 20 and 30 degrees. 
    Later the function is shifted vertucally to ensure the minimum matches null mortality. 
    This procedure is done for all the ERF per age groups. The tmin remains fixed at future times.
     
    Parameters:
    ----------    
    raw : np.ndarray
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
        IDX_START = np.where(np.isclose(T, 10.0, atol=0.05))[0][0]
        IDX_END   = np.where(np.isclose(T, 30.0, atol=0.05))[0][0]
        SEGMENT = raw[:, IDX_START:IDX_END]
        
        # Find local minimum of erf between 20 and 30 degrees
        IDX_LOCAL_MIN = np.argmin(SEGMENT, axis=1)
        tmin = T[IDX_START + IDX_LOCAL_MIN]
        
    # Calcualte mortality value at fixed tmin
    ERF_AT_TMIN = tas*tmin + tas2*tmin**2 + tas3*tmin**3 + tas4*tmin**4
    
    # Shift vertical functions so tmin matches 0 deaths
    ERF_SHIFTED = raw - ERF_AT_TMIN[:,None]
        
    return ERF_SHIFTED, tmin



def MonotonicityERF(T, erf, tmin_g):
    
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
    erf_final : np.ndarray
        Array witht the ERFs after weak monotonicity is imposed
    """
    
    # Find index of tmin in T
    IDX_TMIN = np.searchsorted(T, tmin_g)
    _, NT = erf.shape

    # Create index matrix to vectorize
    IDX_MATRIX = np.arange(NT)[None, :]
    
    # Mask for temperatures above and below tmin
    MASK_LEFT = IDX_MATRIX < IDX_TMIN[:, None]
    MASK_RIGHT = IDX_MATRIX > IDX_TMIN[:, None]
    
    # Impose weak monotonicity to the left
    LEFT_PART = np.where(MASK_LEFT, erf, -np.inf)
    LEFT_MONOTONE = np.maximum.accumulate(LEFT_PART[:, ::-1], axis=1)[:, ::-1]
    
    # Impose weak monotonicity to the right
    RIGHT_PART = np.where(MASK_RIGHT, erf, -np.inf)
    RIGHT_MONOTONE = np.maximum.accumulate(RIGHT_PART, axis=1)
    
    # Generate final Exposure Response Function
    ERF_FINAL = np.where(
        MASK_LEFT, LEFT_MONOTONE,
        np.where(MASK_RIGHT, RIGHT_MONOTONE, erf)
        )
    
    # Ensure no negative values
    ERF_FINAL = np.maximum(ERF_FINAL, 0)
    
    return ERF_FINAL     
    
    
        
def DailyTemperature2IR(temp_path, year, ir, spatial_relation, scenario):
    
    """
    Convert daily temperature data of one year to temeprature values at the impact region 
    level. All grid cells intersecting an impact region are considered. Return
    a dataframe with mean daily temperature per impact region for the given year.
    """
    
    print(f"[2.1] Loading daily temperature data for year {year}...")
    
    if "ERA5" in scenario:
        
        # Open daily temperature data from ERA5
        DAILY_TEMPERATURE = ERA5Temperature2IR(temp_path, year, spatial_relation)
        
    else:
                
        # Read daily temperature data generated from monthly statistics
        DAILY_TEMPERATURE,_ = tmp.DailyFromMonthlyTemperature(
            temp_dir=temp_path, 
            years=year, 
            temp_type="MEAN", 
            std_factor=1,
            to_xarray=False)
        
        # Aggregate daily temperature data to impact region level
        DAILY_TEMPERATURE = MSTemperature2IR(
            temp=DAILY_TEMPERATURE, 
            year=year, 
            ir=ir, 
            spatial_relation=spatial_relation)
    
    # Convert dataframe to numpy array    
    DAILY_TEMPERATURE = DAILY_TEMPERATURE.to_numpy()
    
    return DAILY_TEMPERATURE



def MSTemperature2IR(temp, year, ir, spatial_relation):
    
    """
    Import gridded daily temperature data of one year from montlhy statistics and convert 
    it to the impact region level. Return a dataFrame with daily mean temperature per impact 
    region for the given year.
    """
    
    # Create a list of dates for the specified year
    DATE_LIST = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D").astype(str)
    
    # Temporarily store daily temperatures in a dictionary
    TEMPERATURE_DIC = {}
    for i, day in enumerate(DATE_LIST):
        TEMPERATURE_DIC[day] = temp[...,i].ravel()[spatial_relation.index]

    # Calculate mean temperature per impact region and round
    DAILY_TEMPERATURES_DF = (
        pd.DataFrame(TEMPERATURE_DIC, index=spatial_relation["index_right"])
        .groupby("index_right")
        .mean() # Calculate mean temperature per impact region
        .fillna(20) # Fill in nan with 20 degrees C (conservative choice)
        .round(1) # Round to 1 decimal place
    )
   
    return DAILY_TEMPERATURES_DF



def ERA5Temperature2IR(temp_path, year, spatial_relation):
    
    """
    Import gridded daily temperature data of one year from ERA5 and convert it
    to the impact region level. Return a dataFrame with daily mean temperature 
    per impact region for the given year.
    """
    
    # Read ERA5 daily temperature data for a specific year
    DAILY_TEMPERATURE, _ = tmp.DailyTemperatureERA5(
        era5_dir=temp_path,
        year=year, 
        temp_type="mean", 
        pop_ssp=None, 
        to_array=False)
    
    DAILY_TEMPERATURE = DAILY_TEMPERATURE.t2m
    
    # Create a list of dates for the specified year
    DATE_LIST =(
        DAILY_TEMPERATURE
        ["valid_time"]
        .values
        [np.isin(DAILY_TEMPERATURE["valid_time"]
                 .values
                 .astype("datetime64[Y]"),
                 np.datetime64(f"{year}", "Y"))]
        .astype("datetime64[D]")
        .astype(str)
    )
    
    # Temporarily store daily temperatures in a dictionary
    TEMPERATURE_DIC = {}
    for day in DATE_LIST:
        DAILY_TEMPERATURE_DAY = DAILY_TEMPERATURE.sel(valid_time=day).values.ravel()
        TEMPERATURE_DIC[day] = DAILY_TEMPERATURE_DAY[spatial_relation.index]
            
    # Calculate mean temperature per impact region and round
    DAILY_TEMPERATURE_DF = (
        pd.DataFrame(TEMPERATURE_DIC, index=spatial_relation["index_right"])
        .groupby("index_right")
        .mean()
        .round(1)
    )
    
    return DAILY_TEMPERATURE_DF



def CalculateMortalityEffects(paths, year, scenario, adaptation, regions, res, baseline):
    
    """
    The code calculates equation 2a or 2c from the paper, depending whether adaptation is on or off.
    1. It imports/generates the daily temperature data of the selected year at the ir level
    2. Calculates the first part of the equation (called Marginal Mortality here) 
    3. Claculates the second part (the Counterfactual Mortality).
    4. The substraction is done per impact region, age group, and type of temperature (all, heat, cold).
    3. Aggregate the results spatially to the selected region classification (IMAGE26, ISO3...) and 
       append the results in the DataFrame called results.
    """
    
    # Read daily temperature data from specified source
    DAILY_TEMP_T = DailyTemperature2IR(temp_path=paths.temp_path, 
                                       year=year, 
                                       ir=res.ir, 
                                       spatial_relation=res.spatial_relation, 
                                       scenario=scenario)
    
    print(f"[2.2] Calculating marginal mortality for year {year}...")
    
    # Calculate marginal mortality (first term of equations 2' or 2a' from the paper)
    MOR_ALL_MIN, MOR_HEAT_MIN, MOR_COLD_MIN =  CalculateMarginalMortality(
        paths=paths, 
        year=year, 
        scenario=scenario, 
        daily_temp=DAILY_TEMP_T, 
        adaptation=adaptation, 
        res=res,
        baseline=baseline,
        counterfactual=False
        )
    
    print(f"[2.3] Calculating counterfactual mortality for year {year}...")

    # Calculate counterfactual mortality (second term of equations 2' or 2a' from the paper)
    MOR_ALL_SUB, MOR_HEAT_SUB, MOR_COLD_SUB = CalculateMarginalMortality(
        paths=paths, 
        year=year,
        scenario=scenario, 
        daily_temp=baseline.daily_temp_t0, 
        adaptation=adaptation, 
        res=res,
        baseline=baseline,
        counterfactual=True
        )

    print("[2.4] Aggregating results to", regions, "regions and storing in results dataframe...")
    
    # Calculate mortality difference per impact region 
    for group in res.age_groups: 
        
        MOR_ALL = MOR_ALL_MIN[group] - MOR_ALL_SUB[group]
        MOR_HEAT = MOR_HEAT_MIN[group] - MOR_HEAT_SUB[group]
        MOR_COLD = MOR_COLD_MIN[group] - MOR_COLD_SUB[group]
        
        # Aggregate results to selected region classification and store in results dataframe
        MORTALITY = [MOR_ALL, MOR_HEAT, MOR_COLD]
        
        for mode, mor in zip(["All", "Heat", "Cold"], MORTALITY):
            Mortality2Regions(year, group, mor, regions, mode, res)  
            
            

def CalculateMarginalMortality(paths, year, scenario, daily_temp, adaptation, res, baseline, counterfactual):
    
    """
    Calculate mortality effects from non optimal temperatures. Depending whether adaptation is on, 
    the code will either import the ERFs with no adaptation or generate ERFs with new income and climtas.
    Mortality per impact region will be calculated per age group and temperature type (all, heat and cold) 
    in the Mortality From Temperature Index function.
    """
    
    # Clip daily temperatures to the range of the ERFs
    MIN_TEMP = res.T[0]
    MAX_TEMP = res.T[-1]
    DAILY_TEMP = np.clip(daily_temp, MIN_TEMP, MAX_TEMP)

    # Convert ALL daily temperatures to temperature indices with the min_temp as index 0
    TEMP_INDEX =  np.round(((DAILY_TEMP - MIN_TEMP) * 10)).astype(int)
    
    # Create rows array for indexing
    ROWS = np.arange(TEMP_INDEX.shape[0])[:, None]
    
    # ------------------- Generate ERFs ------------------
    
    # Generate ERFs used when there is income growth and adaptation
    if adaptation==True:    
        ERFS_T, _ = GenerateERFAll(
            paths=paths,
            res = res,
            scenario=scenario,
            year=year,
            adaptation=adaptation,
            baseline=baseline,
            counterfactual=counterfactual
            )
    
    # Use pre-calculated ERFs with no adaptation or income growth
    else: 
        ERFS_T = res.erfs_t0, 
        
    # ------------------- Calculate annual mortality ------------------
    
    MOR_ALL, MOR_HEAT, MOR_COLD = {}, {}, {}
    
    for group in res.age_groups:      
        MOR_ALL[group], MOR_HEAT[group], MOR_COLD[group] = MortalityFromTemperatureIndex(
            daily_temp=DAILY_TEMP, 
            rows=ROWS, 
            erfs=ERFS_T, 
            tmin=baseline.tmin_t0,
            min_temp=MIN_TEMP, 
            group=group)
            
    return MOR_ALL, MOR_HEAT, MOR_COLD



def ImportPresentDayTemperatures(paths, scenario, base_years, ir, spatial_relation):
    
    """
    The function will import the daily temperatures from 2000 to 2010, either precalculated
    ERA5 data or climate data from prescribed scenario. The output is a dictionary of numpy 
    arrays with the daily temperature per impact region and year.
    """
    
    # ------------------ ERA5 ------------------
    if re.search(r"SSP[1-5]_ERA5", scenario):
        
        T_0 = {}
        for year in base_years:
            
            ERA5_T0 = pd.read_csv(paths.wdir+
                                  f"data/climate_data/era5/present_day_temperatures/ERA5_T0_{year}.csv")
            # Store in dictionary as numpy arrays
            T_0[year] = ERA5_T0.iloc[:,2:].to_numpy()
          
        # Calculate mean ignoring 29th feb  
        YEARS_NO_LEAP = []
        for year, arr in T_0.items():
            if arr.shape[1] == 366:
                arr = np.delete(arr, 59, axis=1)
            YEARS_NO_LEAP.append(arr)
        T0_MEAN = np.mean(YEARS_NO_LEAP, axis=0)
            
            
    # -------------- Scenario data --------------
    else: 
        
        DAILY_TEMPERATURE,_ = tmp.DailyFromMonthlyTemperature(
            temp_dir=paths.temp_path, 
            years=base_years,
            temp_type="MEAN",
            std_factor=1, 
            to_xarray=False
        )

        T0_MEAN = MSTemperature2IR(
            temp=DAILY_TEMPERATURE, 
            year=2000, # Dummy year
            ir=ir, 
            spatial_relation=spatial_relation)
        
        # Convert "Present-day" temepratures dataframe to numpy array    
        T0_MEAN = T0_MEAN.iloc[:,1:].to_numpy()
        
    return T0_MEAN

    

def MortalityFromTemperatureIndex(daily_temp, rows, erfs, tmin, min_temp, group):
    
    """
    The code gets the temperature indices for heat (temepratures above tmin) and 
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

    # Extract tmin values for the given age group
    TMIN = tmin[group][:, None]

    # Calculate mortality for temperatures above tmin
    ANNUAL_MORTALITY_HEAT = (
        erfs[group][rows,
            np.round((np.maximum(daily_temp, TMIN) - min_temp) * 10).astype(int)
        ]
        .sum(axis=1)
    )
    
    # Calculate mortality for temperatures below tmin
    ANNUAL_MORTALITY_COLD = (
        erfs[group][rows,
            np.round((np.minimum(daily_temp, TMIN) - min_temp) * 10).astype(int)
        ]
        .sum(axis=1)
    )
    
    # Sum heat and cold mortality to get all-temperatures mortality
    ANNUAL_MORTALITY = ANNUAL_MORTALITY_COLD + ANNUAL_MORTALITY_HEAT
    
    return ANNUAL_MORTALITY, ANNUAL_MORTALITY_HEAT, ANNUAL_MORTALITY_COLD     



def Mortality2Regions(year, group, mor, regions, mode, res):
    
    """
    Aggregate spatially the annual relative mortality from the impact region 
    level to the region classification chosen and locate the results of mortality 
    from heat, cold and all-type mortality in the final results dataframe.
    """
    
    # Create a copy of region classification dataframe
    REGIONS_DF = res.region_class[["hierid", regions]]
    
    # Add mortality and population to df
    REGIONS_DF["mor"] = (mor * res.pop[group][f"{year}"] / 1e5)
    REGIONS_DF["pop"] = res.pop[group][f"{year}"]
    
    # Group total mortality per selected region definition
    REGIONS_DF = REGIONS_DF.drop(columns=["hierid"]).groupby(regions).sum()
    
    # Calculate relative mortality per 100,000 people
    REGIONS_DF["rel_mor"] = REGIONS_DF["mor"] * 1e5 / REGIONS_DF["pop"]
    
    # Locate results in dataframe
    REGIONS_IDX = res.results.loc[(group, mode, "Total Mortality"), year].index[:-1]
    res.results.loc[(group, mode, "Total Mortality", REGIONS_IDX), year] = (REGIONS_DF["mor"].reindex(REGIONS_IDX)).values
    res.results.loc[(group, mode, "Relative Mortality", REGIONS_IDX), year] = (REGIONS_DF["rel_mor"].reindex(REGIONS_IDX)).values
    
    # Locate global results in results dataframe
    res.results.loc[(group, mode, "Total Mortality", "World"), year] = REGIONS_DF["mor"].sum()
    res.results.loc[(group, mode, "Relative Mortality", "World"), year] = (REGIONS_DF["mor"].sum() * 1e5 / REGIONS_DF["pop"].sum())



def AddMortalityAllAges(results, pop, region_class, years, age_groups):
    
    """
    Calculate total mortality and relative mortality for all-ages group 
    by summing the results of the three age groups and dividing by the total 
    population of the three age groups respectively. The funciton will also 
    calculate the global relative mortality for the all-age group by dividing 
    the global total mortality by the global population of all ages.
    """
    
    region_class = region_class.set_index("hierid")
    
    # Prepare population dataframes and aggregate them
    pop = {age: df.set_index("hierid") for age, df in pop.items()}
    pop_all = pop["young"] + pop["older"] + pop["oldest"]
    
    pop_all = (
        pop_all
        .loc[:,[col for col in pop_all.columns if any(str(y) in col for y in years)]]
        .merge(region_class, right_index=True, left_index=True)
        .groupby("IMAGE26")
        .sum()   # Sum population per IMAGE26 region
        .loc[:, lambda df: df.columns.isin([str(y) for y in years])]
        .rename(columns=int)  # Convert column names to integers 
    )
    
    # Calculate total mortality and relative mortality for all-ages groups 
    for mode in ["All", "Heat", "Cold"]:
        
        # Calculate total mortality for all age groups
        results.loc[("all population", mode, "Total Mortality")] = (
            sum(results.loc[(age, mode, "Total Mortality")] for age in age_groups)
        ).values
        
        # Calculate relative mortality for all-age group        
        IMAGE26 = results.loc[("all population", mode, "Relative Mortality")].index[:-1]
        
        results.loc[("all population", mode, "Relative Mortality", IMAGE26)] = (
            results.loc[("all population", mode, "Total Mortality", IMAGE26)]
            .mul(1e5)
            .div(pop_all.where(pop_all.reindex(IMAGE26) != 0))
        ).values
        
        # Calculate global relative mortality for all-age group
        results.loc[("all population", mode, "Relative Mortality", "World")] = (
            results.loc[("all population", mode, "Total Mortality", "World")]
            .mul(1e5)
            .div(pop_all.sum(axis=0)) # Divide by global population of all ages
        )

    return results



def PostprocessResults(wdir, res, years, project, scenario, adaptation):
    
    """
    Postprocess final results and save to CSV file in output folder.
    1. Calculate total mortality and relative mortality for all-ages group by summing the results
    of the three age groups and dividing by the total population of the three age groups respectively.
    2. If IAM format is on, change the format of the results to match the IAM one.
    3. Save results in main working directory.
    """
    
    print("[3] Postprocessing and saving results...")
    
    # Calculate total mortality and relative mortality for all-ages group
    RESULTS = AddMortalityAllAges(res.results, res.pop, res.region_class, years, res.age_groups)
    
    # # Reset index and format results for IAMs if specified
    # if IAM_format==True:
    #     RESULTS = RESULTS.reset_index()
        
    #     # Asign mortality name according to units
    #     RESULTS.loc[RESULTS["units"] == "Relative Mortality", "var"] = "Relative Mortality"
    #     RESULTS.loc[RESULTS["units"] != "Relative Mortality", "var"] = "Mortality"

    #     # Rename all temperatures name
    #     RESULTS.loc[RESULTS["t_type"] == "All", "t_type"] = "All temperatures"

    #     # Rename 'age_group'
    #     RESULTS.loc[RESULTS["age_group"] == "all population", "age_group"] = "All ages"
    #     RESULTS.loc[RESULTS["age_group"] == "young", "age_group"] = "0-4 years"
    #     RESULTS.loc[RESULTS["age_group"] == "older", "age_group"] = "5-64 years"
    #     RESULTS.loc[RESULTS["age_group"] == "oldest", "age_group"] = "+65 years"

    #     # Create column 'Variable'
    #     RESULTS["Variable"] = (
    #         "Health|"
    #         + RESULTS["var"]
    #         + "|Non-optimal Temperatures|"
    #         + RESULTS["t_type"].str.capitalize()
    #         + "|"
    #         + RESULTS["age_group"].str.capitalize()
    #     )
            
    #     RESULTS = RESULTS[["IMAGE26", "Variable"] + list(RESULTS.columns[4:-2])]
            
    RESULTS = RESULTS.rename(columns={"IMAGE26": "region"})
    
    if adaptation == True:
        adapt = ""
    else:
        adapt = "_noadapt"
    if project is not None:
        project = f"_{project}"
    else:
        project = ""
        
    # Save results to CSV                
    RESULTS.to_csv(wdir +
                   f"output/mortality_carleton{project}_{scenario}{adapt}_{years[0]}-{years[-1]}.csv", 
                   index=False) 
    
    print("Scenario ran successfully!")