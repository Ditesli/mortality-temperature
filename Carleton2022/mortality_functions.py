import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from dataclasses import dataclass, field
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
    reporting_tool: bool
):

    sets = ModelSettings(
        temp_path=temp_dir,
        income_path=gdp_dir,
        wdir=wdir,
        project=project,
        scenario=scenario,
        regions=regions,
        years=years,
        adaptation=adaptation,
        reporting_tool=reporting_tool
    )

    model = MortalityModel(sets=sets)

    model.run()



@dataclass
class ModelSettings:
    temp_path: str
    income_path: any
    wdir: str
    project: any
    scenario: str
    regions: str
    years: list
    adaptation: bool
    reporting_tool: bool
    age_groups: list = field(default_factory=lambda: ["young", "older", "oldest"])
    T: np.ndarray = field(default_factory=lambda: np.arange(-20, 40.1, 0.1).round(1))
    
    def __post_init__(self):
        self.years = self.validate_years()
        self.temp_path = self.climate_path()
    
    
    def climate_path(self) -> str:
        """
        Set path to climate data depending on the scenario type 
        (IMAGE or other scenarios)
        """
        if self.temp_path == self.income_path:
            return (
                f"{self.temp_path}/"
                f"{self.project}/3_IMAGE_land/scen/"
                f"{self.scenario}/netcdf/"
            )
        else:
            return self.temp_path
        
    def validate_years(self):
        ERA5_START_YEAR = 2000
        ERA5_END_YEAR = 2025

        if re.search(r"SSP[1-5]_ERA5", self.scenario):
            return [
                y for y in self.years
                if ERA5_START_YEAR <= y <= ERA5_END_YEAR
            ]
        else:
            return self.years
            


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
        
    Returns:
    ----------
    None
    Saves the mortality results to CSV files in the output folder.
    """

    def load_inputs(self):
        self.fls = LoadInputData.from_files(sets=self.sets)
        self.baseline = BaselineERFsInputs.from_sets(sets=self.sets, fls=self.fls)
        
        
    def run(self):
        print(f"Starting mortality model for {self.sets.scenario}")
        self.load_inputs()

        print("[2] Starting mortality calculations...")

        for year in self.sets.years:
            CalculateMortalityEffects(
                sets=self.sets,
                year=year,
                fls=self.fls,
                baseline=self.baseline
            )

        self.postprocess()
        
        
    def postprocess(self):
        PostprocessResults(sets=self.sets, fls=self.fls)

    
    
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

    spatial_relation: gpd.GeoDataFrame
    ir: pd.DataFrame
    region_class: pd.DataFrame
    results: pd.DataFrame
    gammas: any
    pop: pd.DataFrame
    base_years: list = range(2001,2011)

    @classmethod
    def from_files(cls, sets):
        
        """
        Read and load all input files required for mortality calculations. The 
        necessary data is located in the wdir/data folder.  
        """
        
        print("[1] Loading input files and defining parameters...")    
        
        print(f"[1.2] Loading region classification: {sets.regions}...")
        region_class = pd.read_csv(f"{sets.wdir}/data/regions/region_classification.csv")
        if sets.regions == "impact_regions":
            region_class = region_class[["hierid", "ISO3"]]
        if sets.regions == "countries":
            region_class = region_class[["hierid", "ISO3", "gbd_level3"]]
        else:
            region_class = region_class[["hierid", "ISO3", sets.regions]]
        
        spatial_relation, ir = GridRelationship(sets)
        
        results = FinalDataframe(sets, region_class)
        
        gamma_coeffs = ImportGammaCoefficients(sets.wdir)
        
        population = ImportPopulationData(sets, ir)    
    
        return cls(
            spatial_relation=spatial_relation,
            ir=ir,
            region_class=region_class,
            results=results,
            gammas = gamma_coeffs,
            pop = population
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
    
    
    def from_sets(sets, fls):
        
        # Import present day covariates
        print("[1.4] Loading 'present day' covariates climtas and loggdppc...")
        
        climtas_t0, loggdppc_t0 = ImportCovariates(
            sets=sets, 
            fls=fls,
            year=None, 
            adaptation=False,
            baseline = None,
            counterfactual=None
            )
        
        # Generate a single time 'present day' ERFs (no adaptation)
        erfs_t0, tmin_t0 = GenerateERFAll(
            sets=sets, 
            fls=fls,
            year=None, 
            adaptation=False, 
            baseline = None,
            counterfactual=None
            ) 
        
        print("[1.5] Loading 'present-day' temperature data...")
        
        # Import present day temperatures
        daily_temp_t0 = ImportPresentDayTemperatures(
            sets=sets, 
            base_years=range(2001,2011), 
            ir=fls.ir, 
            spatial_relation=fls.spatial_relation
            )
        
        #  Read GDP shares for scenarios that do not use Carleton's socioeconomic data.
        
        if sets.adaptation:
                
            print("[1.6] Loading GDPpc shares at the impact region level...")
            # Generate GDPpc shares of regions within a country and IMAGE region
            image_shares, country_shares = GenerateGDPpcShares(wdir=sets.wdir, fls=fls)
            image_gdppc = None
            
            if not re.search(r"SSP[1-5]_ERA5", sets.scenario) and "carleton" not in sets.scenario.lower():
                
                print("[1.7] Loading GDP data from IMAGE...")
                image_gdppc = ReadOUTFiles(sets)
                
        # Set to None when adaptation is off        
        else:  
            image_shares = None
            image_gdppc = None
            country_shares = None
            
            
        return BaselineERFsInputs(
            climtas_t0 = climtas_t0,
            loggdppc_t0 = loggdppc_t0,
            erfs_t0 = erfs_t0,
            tmin_t0 = tmin_t0,
            image_shares = image_shares,
            country_shares=country_shares,
            image_gdppc = image_gdppc,
            daily_temp_t0 = daily_temp_t0
        )



def GridRelationship(sets):
    
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
    if re.search(r"ERA5", sets.scenario):
        # Use function located in the utils_common folder to import ERA5 data in the right format
        grid,_ = tmp.DailyTemperatureERA5(
            era5_dir=sets.temp_path, 
            year=sets.years[0], 
            temp_type="mean", 
            pop_ssp=None, 
            to_array=False
            )
    
    # --------- If Monthly Statistics (MS) data ----------  
    else:
        
        # Use function to import monthly statistics (MS) of daily temperature data in the right format
        grid,_ = tmp.DailyFromMonthlyTemperature(
            temp_dir=path, 
            years=sets.years[0], 
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
    
    coord_names = grid.coords.keys()
    lon_vals = FindCoordinateName(["lon", "longitude", "x"], coord_names, grid)
    lat_vals = FindCoordinateName(["lat", "latitude", "y"], coord_names, grid)

    # Create meshgrid 
    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)  
    
    # Create GeoDataFrame with points and their corresponding square polygons
    points_gdf = gpd.GeoDataFrame({
        "longitude": lon2d.ravel(),
        "latitude": lat2d.ravel(),
        "geometry": [
            CreateSquare(lon, lat, np.abs(np.mean(np.diff(lon_vals))),  np.abs(np.mean(np.diff(lat_vals))))
            for lon, lat in zip(lon2d.ravel(), lat2d.ravel())
        ]
    })
    
    # Load .shp file with impact regions and set the same coordinate reference system (CRS)
    ir = gpd.read_file(sets.wdir + "/data/carleton_sm/ir_shp/impact-region.shp")
    points_gdf = points_gdf.set_crs(ir.crs, allow_override=True)
    
    # Make spatial join
    relationship = gpd.sjoin(points_gdf, ir, how="inner", predicate="intersects")

    return relationship[["geometry", "index_right", "hierid"]], ir["hierid"]



def FinalDataframe(sets, region_class):
    
    """
    Create results dataframe with multiindex for age groups, temperature types,
    mortality types (total mortality and relative mortality) and regions.
    """
    
    regions = region_class[f"{sets.regions}"].unique()
    regions = regions[~pd.isna(regions)]
    regions = np.append(regions, "World")
    
    age_groups = np.append(sets.age_groups, "all population")
    temperature_types = ["Heat", "Cold", "All"]
    mortality_types = ["Total Mortality", "Relative Mortality"]
    
    # Create results multiindex dataframe
    results = (
        pd.DataFrame(
            index=pd.MultiIndex.from_product([age_groups, temperature_types, mortality_types, regions],
                                         names=["age_group", "t_type", "units", sets.regions]), 
            columns=sets.years
        ).sort_index()
    )
    
    return results 



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
                covar_names = [x for x in line.strip().split(", ")]
                # Convert to indices and convert to array
                covar_map = {"1":0, "climtas":1, "loggdppc":2}
                covar_idx = np.array([covar_map[str(x)] for x in covar_names])
                
            if i == 23:
                # Extract gamma coefficients
                gammas = np.array([float(x) for x in line.strip().split(", ")])
                
    return gammas.reshape(3,12), covar_idx.reshape(3,12)



def ImportPopulationData(sets, ir):
    
    print(f"[1.3] Loading Population data for {sets.scenario} scenario at the impact regions level...")
    
    # Extract SSP from scenario string
    match = re.search(r"(?i)\bssp\d+", sets.scenario)
    # Extract corresponding SSP scenario
    ssp = match.group().upper()
    
    # Include ALWAYS population data from 2000 to 2010 (used in the subtrahend part)
    year = sorted(set(sets.years).union(range(2000, 2011)))
        
    # Import population data based on scenario type
    if 'carleton' in sets.scenario.lower() or re.search(r"SSP[1-5]_ERA5", sets.scenario):
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
    
    Returns:
    ----------
    population_groups : dict
        Dictionary with population dataframes per age group
    """
    
    population_groups = {}
    age_pop_names = ['pop0to4', 'pop5to64', 'pop65plus']
    
    for age_group, age_name in zip(sets.age_groups, age_pop_names):
        
        # Read 'present-day' population data
        pop_historical = (
            pd.read_csv(f"{sets.wdir}/data/population/pop_historical/POP_historical_{age_group}.csv")
            .set_index("hierid")
        )

        pop_ssp = (
            xr.open_dataset(sets.wdir+f"data/carleton_sm/econ_vars/{ssp.upper()}.nc4")[age_name]
            .sel(model="low") # Select any GDP model
            .to_dataframe() # Convert to dataframe
            .drop(columns=['ssp', 'model'])
            .unstack('year') # Reshape to have years as columns
            .pipe(lambda df: df.set_axis(df.columns.get_level_values(-1), axis=1))
            [[y for y in years if y >= 2023]] # Keep only years from 2023 onwards 
            .merge(pop_historical, left_index=True, right_index=True) # Merge with historical population 
            .reindex(ir.values) # Align to impact regions order
            .pipe(lambda df: df.reindex(sorted(df.columns, key=int), axis=1))
            .pipe(lambda df: df.set_axis(df.columns.astype(int), axis=1))
            .reset_index()
            .rename(columns={"region":"hierid"})
        )
        
        population_groups[age_group] = pop_ssp
    
    return population_groups



def ImportIMAGEPopulationData(sets, ssp, years, ir):
    
    """
    Read precalculated IMAGE population data at the impact region level for a
    given SSP.

    Returns:
    ----------
    pop_ssp : dict
        Dictionary with population data per age group
    """
    
    pop_ssp = {}
    
    for age_group in sets.age_groups:
        pop_ssp_group = (
            pd.read_csv(f"{sets.wdir}/data/population/pop_ssp/pop_{ssp.lower()}_{age_group}.csv")
            .pipe(lambda df: df.filter(
                ["hierid"] +
                [c for c in df.columns if c.isdigit() and int(c) in years]
            ))
            .set_index("hierid")
            .reindex(ir.values) # Align to impact regions orders
            .reset_index()
        )
    
        pop_ssp[age_group] = pop_ssp_group
    
    return pop_ssp



def GenerateERFAll(sets, fls, year, adaptation, baseline, counterfactual):
    
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
        fls=fls,
        year=year, 
        adaptation=adaptation,
        baseline = baseline,
        counterfactual=counterfactual
        )

    # Covariates matrix
    covariates = np.column_stack([np.ones(len(climtas)), climtas, loggdppc])   

    # Generate arrays with erf and tmin per age group
    mor_np = {}; tmin = {} 
    for i, group in enumerate(sets.age_groups):
        
        if baseline is None:
            erfs_t0 = None
            tmin_t0 = None
        else:
            erfs_t0 = baseline.erfs_t0[group]
            tmin_t0 = baseline.tmin_t0[group]   
            
        # List of locations of gamma and covariates
        g = fls.gammas[0][i]; cov = fls.gammas[1][i]

        # Multiply each covariate by its corresponding gamma
        base = covariates[:, cov] * g
        
        # Compute the sum of the covariates to get polynomial coefficients
        tas = base[:, 0:3].sum(axis=1)  
        tas2 = base[:, 3:6].sum(axis=1)  
        tas3 = base[:, 6:9].sum(axis=1) 
        tas4 = base[:, 9:12].sum(axis=1)

        # Generate raw Exposure Response Function
        erf_raw = (
            tas[:,None] * sets.T**1 +
            tas2[:,None] * sets.T**2 +
            tas3[:,None] * sets.T**3 +
            tas4[:,None] * sets.T**4
        )
        
        # Impose zero mortality at tmin by vertically shifting erf
        erf_shifted, tmin_g = ShiftERFToTmin(erf_raw, sets.T, tas, tas2, tas3, tas4, tmin_t0)
        
        #  # Ensure ERFs do not exceed no-adaptation ERFs 
        if erfs_t0 is not None:
            erf_shifted = np.minimum(erf_shifted, erfs_t0)
        
        # Impose weak monotonicity to the left and the right of the erf
        mor_np[group] = MonotonicityERF(sets.T, erf_shifted, tmin_g)
        tmin[group] = tmin_g
        
    return mor_np, tmin



def ImportCovariates(sets, fls, year, adaptation, baseline, counterfactual):
    
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
             pd.read_csv(sets.wdir+"data/carleton_sm/main_specification/mortality-allpreds.csv")
            .rename(columns={"region":"hierid"})
            .set_index("hierid")
            .reindex(fls.ir.values)
        )
        
        # Extract only climtas and loggdppc as arrays
        climtas = covariates_t0["climtas"].values
        loggdppc = covariates_t0["loggdppc"].values
    
    # ADAPTATION ---------------------------------------------
    else:
        
        # climtas ---------------------------
        
        # Load "present-day" climatology
        if counterfactual:
            if re.search(r"ERA5", sets.scenario):
                climtas = ImportClimtasERA5(sets.wdir, year, fls.ir, present_day=True)
            else: 
                climtas = ImportClimtas(sets.temp_path, None, fls.spatial_relation, present_day=True)
        
        # Load climatology of selected year 
        else:
            if re.search(r"ERA5", sets.scenario):
                climtas = ImportClimtasERA5(sets.wdir, year, fls.ir, present_day=False)
            else:
                climtas = ImportClimtas(sets.temp_path, year, fls.spatial_relation, present_day=False)
                
        # log(GDPpc) ---------------------------    
        
        if re.search(r"ERA5", sets.scenario) or ("carleton" in sets.scenario.lower() and year < 2010):
            # Load historical log(GDPpc) from World Bank
            loggdppc = ImportHistoricalLogGDPpc(sets.wdir, fls.ir, year, baseline.country_shares)
        
        elif "carleton" in sets.scenario.lower() and year >= 2010:  
            # Load log(GDPpc) from Carleton et al. (2022) for the selected year and scenario
            loggdppc = ImportCarletonLogGDPpc(sets.wdir, sets.scenario, fls.ir, year)
            
        else: 
            # Load log(GDPpc) at the impact region level using the GDPpc output from IMAGE
            loggdppc = ImportIMAGEloggdppc(year, baseline)
            
    return climtas, loggdppc



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
    
    scenario = re.search(r"(?i)\bssp\d+", scenario).group()
        
    # Read GDP per capita file
    gdppc = (
        xr.open_dataset(wdir+f"data/carleton_sm/econ_vars/{scenario.upper()}.nc4")   
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
    gdppc["loggdppc"] = np.log(gdppc["gdppc"])
    
    # Return numpy array
    return gdppc["loggdppc"].values



def ImportIMAGEloggdppc(year, baseline):
    
    """
    Calculate log(GDPpc) at the impact region level using the GDPpc output from a 
    TIMER run and the shares that downscale IMAGE GDPpc at the impact region level.
    """
    
    # Extract relevant year data (13 year rolling mean)
    image_gdppc = (
        baseline
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
    gdppc =  baseline.image_shares.merge(image_gdppc, left_on="IMAGE26", right_on="region", how="left")
    
    # Calculate share of log(GDPpc) based on regional GDPpc
    gdppc["gdppc"] = gdppc["Value"] * gdppc["gdppc_share"] 
    gdppc["loggdppc"] = np.log(gdppc["gdppc"])
    
    return gdppc["loggdppc"].values

 

def GenerateGDPpcShares(wdir, fls):
    
    """
    Generate the corresponding GDPpc shares per impact region within an IMAGE region and 
    country. The function will read the GDPpc data from Carleton et al. (2022) and calculate 
    the factor that downscales the GDPpc. The final output will be a dataframe with the GDPpc
    shares per impact region.
    """

    # Open GDP data (can be any SSP)
    gdppc = (
        xr.open_dataset(f"{wdir}/data/carleton_sm/econ_vars/SSP2.nc4")
        .to_dataframe() # Convert to dataframe
        .reset_index()
        .merge(fls.region_class, left_on="region", right_on="hierid")
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
            .reindex(fls.ir.values)
            .reset_index()
        )
    
    # Calculate IMAGE and country shares
    image_shares = compute_shares(gdppc, "IMAGE26")
    country_shares = compute_shares(gdppc, "ISO3")
        
    return image_shares, country_shares



def ReadOUTFiles(sets):
    
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
    path_clim = (
        sets.gdp_dir + "/" 
        + sets.project + "/2_TIMER/outputlib/TIMER_3_4/" 
        + sets.project + "/"
        + sets.scenario + "/indicators/Economy/"
        )
    
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
                 .expand_dims({"Scenario": [sets.scenario], "Variable": [VAR]}))
    xr_vars = xr.merge(listy)
 
    return xr_vars



def ImportClimtasERA5(wdir, year, ir, present_day):
    
    climtas = pd.read_csv(wdir+f"data/climate_data/era5/climatologies/ERA5_CLIMTAS_2000-2025.csv")
    
    if present_day == False:
        climtas = climtas.set_index("hierid").reindex(ir)[str(year)].values
    
    else:
        cols = [str(y) for y in range(2001, 2011)]
        climtas = climtas.set_index("hierid").reindex(ir)[cols].mean(axis=1).values

    return climtas



def ImportClimtas(temp_path, year, spatial_relation, present_day):
    
    """
    Import climate data from montlhy statistics files. The code calculates the 30-year running
    mean from the mothly data at the grid cell level. The code will later calculate the mean 
    climatology per impact region using "spatial_relation" and will return the data as a numpy array
    ordered by "ir".
    """
    
    if present_day==True:
        year = slice("2001-01-01", "2011-01-01") # The "present-day" climatology 
    else:
        year = slice(f"{year}-01-01", f"{year}-12-31")
    
    # Read monthly mean of daily mean temperature data
    climtas_ir = (
        xr.open_dataset(temp_path+f"GTMP_MEAN_30MIN.nc")
        ["GTMP_MEAN_30MIN"]
        .mean(dim="NM") # Annual temperature
        .rolling(time=30, min_periods=1) 
        .mean() # Climatology
        .sel(time=year)
        .mean(dim="time") # "present-day" climatology
        .values
        .ravel()
        [spatial_relation.index] # Assign pixels to every impact region using spatial relation
    )

    # Calculate mean temperature per impact region and round
    climtas = (
        pd.DataFrame(climtas_ir, index=spatial_relation["index_right"])
        .groupby("index_right")
        .mean()
        .fillna(20)
        [0].values
    )

    return climtas



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
        idx_start = np.where(np.isclose(T, 10.0, atol=0.05))[0][0]
        idx_end   = np.where(np.isclose(T, 30.0, atol=0.05))[0][0]
        segment = raw[:, idx_start:idx_end]
        
        # Find local minimum of erf between 20 and 30 degrees
        idx_local_min = np.argmin(segment, axis=1)
        tmin = T[idx_start + idx_local_min]
        
    # Calcualte mortality value at fixed tmin
    erf_at_tmin = tas*tmin + tas2*tmin**2 + tas3*tmin**3 + tas4*tmin**4
    
    # Shift vertical functions so tmin matches 0 deaths
    erf_shifted = raw - erf_at_tmin[:,None]
        
    return erf_shifted, tmin



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
    idx_tmin = np.searchsorted(T, tmin_g)
    _, nT = erf.shape

    # Create index matrix to vectorize
    idx_matrix = np.arange(nT)[None, :]
    
    # Mask for temperatures above and below tmin
    mask_left = idx_matrix < idx_tmin[:, None]
    mask_right = idx_matrix > idx_tmin[:, None]
    
    # Impose weak monotonicity to the left
    left_part = np.where(mask_left, erf, -np.inf)
    left_monotone = np.maximum.accumulate(left_part[:, ::-1], axis=1)[:, ::-1]
    
    # Impose weak monotonicity to the right
    right_part = np.where(mask_right, erf, -np.inf)
    right_monotone = np.maximum.accumulate(right_part, axis=1)
    
    # Generate final Exposure Response Function
    erf_final = np.where(
        mask_left, left_monotone,
        np.where(mask_right, right_monotone, erf)
        )
    
    # Ensure no negative values
    erf_final = np.maximum(erf_final, 0)
    
    return erf_final     
    
    
        
def DailyTemperature2IR(sets, year, ir, spatial_relation):
    
    """
    Convert daily temperature data of one year to temeprature values at the impact region 
    level. All grid cells intersecting an impact region are considered. Return
    a dataframe with mean daily temperature per impact region for the given year.
    """
    
    print(f"[2.1] Loading daily temperature data for year {year}...")
    
    if "ERA5" in sets.scenario:
        
        # Open daily temperature data from ERA5
        daily_temperature = ERA5Temperature2IR(sets.temp_path, year, spatial_relation)
        
    else:
                
        # Read daily temperature data generated from monthly statistics
        daily_temperature,_ = tmp.DailyFromMonthlyTemperature(
            temp_dir=sets.temp_path, 
            years=year, 
            temp_type="MEAN", 
            std_factor=1,
            to_xarray=False)
        
        # Aggregate daily temperature data to impact region level
        daily_temperature = MSTemperature2IR(
            temp=daily_temperature, 
            year=year, 
            ir=ir, 
            spatial_relation=spatial_relation)
    
    # Convert dataframe to numpy array    
    daily_temperature = daily_temperature.to_numpy()
    
    return daily_temperature



def MSTemperature2IR(temp, year, ir, spatial_relation):
    
    """
    Import gridded daily temperature data of one year from montlhy statistics and convert 
    it to the impact region level. Return a dataFrame with daily mean temperature per impact 
    region for the given year.
    """
    
    # Create a list of dates for the specified year
    date_list = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D").astype(str)
    
    # Temporarily store daily temperatures in a dictionary
    temperature_dic = {}
    for i, day in enumerate(date_list):
        temperature_dic[day] = temp[...,i].ravel()[spatial_relation.index]

    # Calculate mean temperature per impact region and round
    daily_temperatures_df = (
        pd.DataFrame(temperature_dic, index=spatial_relation["index_right"])
        .groupby("index_right")
        .mean() # Calculate mean temperature per impact region
        .fillna(20) # Fill in nan with 20 degrees C (conservative choice)
        .round(1) # Round to 1 decimal place
    )
   
    return daily_temperatures_df



def ERA5Temperature2IR(temp_path, year, spatial_relation):
    
    """
    Import gridded daily temperature data of one year from ERA5 and convert it
    to the impact region level. Return a dataFrame with daily mean temperature 
    per impact region for the given year.
    """
    
    # Read ERA5 daily temperature data for a specific year
    daily_temperature, _ = tmp.DailyTemperatureERA5(
        era5_dir=temp_path,
        year=year, 
        temp_type="mean", 
        pop_ssp=None, 
        to_array=False)
    
    daily_temperature = daily_temperature.t2m
    
    # Create a list of dates for the specified year
    date_list =(
        daily_temperature
        ["valid_time"]
        .values
        [np.isin(daily_temperature["valid_time"]
                 .values
                 .astype("datetime64[Y]"),
                 np.datetime64(f"{year}", "Y"))]
        .astype("datetime64[D]")
        .astype(str)
    )
    
    # Temporarily store daily temperatures in a dictionary
    temperature_dic = {}
    for day in date_list:
        daily_temperature_day = daily_temperature.sel(valid_time=day).values.ravel()
        temperature_dic[day] = daily_temperature_day[spatial_relation.index]
            
    # Calculate mean temperature per impact region and round
    daily_temperature_df = (
        pd.DataFrame(temperature_dic, index=spatial_relation["index_right"])
        .groupby("index_right")
        .mean()
        .round(1)
    )
    
    return daily_temperature_df



def CalculateMortalityEffects(sets, year, fls, baseline):
    
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
    daily_temperature = DailyTemperature2IR(
        sets=sets, 
        year=year, 
        ir=fls.ir, 
        spatial_relation=fls.spatial_relation
        )
    
    print(f"[2.2] Calculating marginal mortality for year {year}...")
    
    # Calculate marginal mortality (first term of equations 2' or 2a' from the paper)
    mor_all_min, mor_heat_min, mor_cold_min = CalculateMarginalMortality(
        sets=sets, 
        year=year,  
        daily_temp=daily_temperature, 
        fls=fls,
        baseline=baseline,
        counterfactual=False
        )
    
    print(f"[2.3] Calculating counterfactual mortality for year {year}...")

    # Calculate counterfactual mortality (second term of equations 2' or 2a' from the paper)
    mor_all_sub, mor_heat_sub, mor_cold_sub = CalculateMarginalMortality(
        sets=sets, 
        year=year,
        daily_temp=baseline.daily_temp_t0,  
        fls=fls,
        baseline=baseline,
        counterfactual=True
        )

    print("[2.4] Aggregating results to", sets.regions, "regions...")
    
    # Calculate mortality difference per impact region 
    for group in sets.age_groups: 
        
        mor_all = mor_all_min[group] - mor_all_sub[group]
        mor_heat = mor_heat_min[group] - mor_heat_sub[group]
        mor_cold = mor_cold_min[group] - mor_cold_sub[group]
        
        # Aggregate results to selected region classification and store in results dataframe
        mortality = [mor_all, mor_heat, mor_cold]
        
        for mode, mor in zip(["All", "Heat", "Cold"], mortality):
            Mortality2Regions(year, group, mor, sets.regions, mode, fls)  
            
            

def CalculateMarginalMortality(sets, year, daily_temp, fls, baseline, counterfactual):
    
    """
    Calculate mortality effects from non optimal temperatufls. Depending whether adaptation is on, 
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
            fls=fls,
            year=year,
            adaptation=sets.adaptation,
            baseline=baseline,
            counterfactual=counterfactual
            )
    
    # Use pre-calculated ERFs with no adaptation or income growth
    else: 
        erfs_t = fls.erfs_t0, 
        
    # ------------------- Calculate annual mortality ------------------
    
    mor_all, mor_heat, mor_cold = {}, {}, {}
    
    for group in sets.age_groups:      
        mor_all[group], mor_heat[group], mor_cold[group] = MortalityFromTemperatureIndex(
            daily_temp=daily_temperature, 
            rows=rows, 
            erfs=erfs_t, 
            tmin=baseline.tmin_t0,
            min_temp=min_temp, 
            group=group)
            
    return mor_all, mor_heat, mor_cold



def ImportPresentDayTemperatures(sets, base_years, ir, spatial_relation):
    
    """
    The function will import the daily temperatures from 2000 to 2010, either precalculated
    ERA5 data or climate data from prescribed scenario. The output is a dictionary of numpy 
    arrays with the daily temperature per impact region and year.
    """
    
    # ------------------ ERA5 ------------------
    if re.search(r"SSP[1-5]_ERA5", sets.scenario):
        
        t_0 = {}
        for year in base_years:
            
            era5_t0 = pd.read_csv(sets.wdir+
                                  f"data/climate_data/era5/present_day_temperatures/ERA5_T0_{year}.csv")
            # Store in dictionary as numpy arrays
            t_0[year] = era5_t0.iloc[:,2:].to_numpy()
          
        # Calculate mean ignoring 29th feb  
        years_no_leap = []
        for year, arr in t_0.items():
            if arr.shape[1] == 366:
                arr = np.delete(arr, 59, axis=1)
        # years_no_leap = []
        # for year, arr in t_0.items():
        #     if arr.shape[1] == 366:
        #         arr = np.delete(arr, 59, axis=1)
        #     years_no_leap.append(arr)
            
            
    # -------------- Scenario data --------------
    else: 
        
        daily_temperature,_ = tmp.DailyFromMonthlyTemperature(
            temp_dir=sets.temp_path, 
            years=base_years,
            temp_type="MEAN",
            std_factor=1, 
            to_xarray=False
        )

        t0_mean = MSTemperature2IR(
            temp=daily_temperature, 
            year=2000, # Dummy year
            ir=ir, 
            spatial_relation=spatial_relation)
        
        # Convert "Present-day" temepratures dataframe to numpy array    
        t0_mean = t0_mean.iloc[:,1:].to_numpy()
        
    return t0_mean

    

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
    tmin = tmin[group][:, None]

    # Calculate mortality for temperatures above tmin
    annual_mortality_heat = (
        erfs[group][rows,
            np.round((np.maximum(daily_temp, tmin) - min_temp) * 10).astype(int)
        ]
        .sum(axis=1)
    )
    
    # Calculate mortality for temperatures below tmin
    annual_mortality_cold = (
        erfs[group][rows,
            np.round((np.minimum(daily_temp, tmin) - min_temp) * 10).astype(int)
        ]
        .sum(axis=1)
    )
    
    # Sum heat and cold mortality to get all-temperatures mortality
    annual_mortality = annual_mortality_cold + annual_mortality_heat
    
    return annual_mortality, annual_mortality_heat, annual_mortality_cold     



def Mortality2Regions(year, group, mor, regions, mode, fls):
    
    """
    Aggregate spatially the annual relative mortality from the impact region 
    level to the region classification chosen and locate the results of mortality 
    from heat, cold and all-type mortality in the final results dataframe.
    """
    
    # Create a copy of region classification dataframe
    regions_df = fls.region_class[["hierid", regions]]
    
    # Add mortality and population to df
    regions_df["mor"] = (mor * fls.pop[group][year] / 1e5)
    regions_df["pop"] = fls.pop[group][year]
    
    # Group total mortality per selected region definition
    regions_df = regions_df.drop(columns=["hierid"]).groupby(regions).sum()
    
    # Calculate relative mortality per 100,000 people
    regions_df["rel_mor"] = regions_df["mor"] * 1e5 / regions_df["pop"]
    
    # Locate results in dataframe
    regions_idx = fls.results.loc[(group, mode, "Total Mortality"), year].index[:-1]
    fls.results.loc[(group, mode, "Total Mortality", regions_idx), year] = (regions_df["mor"].reindex(regions_idx)).values
    fls.results.loc[(group, mode, "Relative Mortality", regions_idx), year] = (regions_df["rel_mor"].reindex(regions_idx)).values
    
    # Locate global results in results dataframe
    fls.results.loc[(group, mode, "Total Mortality", "World"), year] = regions_df["mor"].sum()
    fls.results.loc[(group, mode, "Relative Mortality", "World"), year] = (regions_df["mor"].sum() * 1e5 / regions_df["pop"].sum())



def AddMortalityAllAges(fls, sets):
    
    """
    Calculate total mortality and relative mortality for all-ages group 
    by summing the results of the three age groups and dividing by the total 
    population of the three age groups respectively. The funciton will also 
    calculate the global relative mortality for the all-age group by dividing 
    the global total mortality by the global population of all ages.
    """
    
    region_class = fls.region_class.set_index("hierid")
    
    # Prepare population dataframes and aggregate them
    population = {age: df.set_index("hierid") for age, df in fls.pop.items()}
    population_all = population["young"] + population["older"] + population["oldest"]
    
    population_all = (
        population_all
        .loc[:,[col for col in population_all.columns if any(y in col for y in sets.years)]]
        .merge(region_class, right_index=True, left_index=True)
        .groupby("IMAGE26")
        .sum()   # Sum population per IMAGE26 region
        .loc[:, lambda df: df.columns.isin([y for y in sets.years])]
        .rename(columns=int)  # Convert column names to integers 
    )
    
    # Calculate total mortality and relative mortality for all-ages groups 
    for mode in ["All", "Heat", "Cold"]:
        
        # Calculate total mortality for all age groups
        fls.results.loc[("all population", mode, "Total Mortality")] = (
            sum(fls.results.loc[(age, mode, "Total Mortality")] for age in sets.age_groups)
        ).values
        
        # Calculate relative mortality for all-age group        
        IMAGE26 = fls.results.loc[("all population", mode, "Relative Mortality")].index[:-1]
        
        fls.results.loc[("all population", mode, "Relative Mortality", IMAGE26)] = (
            fls.results.loc[("all population", mode, "Total Mortality", IMAGE26)]
            .mul(1e5)
            .div(population_all.where(population_all.reindex(IMAGE26) != 0))
        ).values
        
        # Calculate global relative mortality for all-age group
        fls.results.loc[("all population", mode, "Relative Mortality", "World")] = (
            fls.results.loc[("all population", mode, "Total Mortality", "World")]
            .mul(1e5)
            .div(population_all.sum(axis=0)) # Divide by global population of all ages
        )

    return fls.results


    values per region and year.
    """
    
    for group in sets.age_groups:
        for mode in ["All", "Heat", "Cold"]:
            output = results.loc[(group, mode, "Total Mortality")].reset_index()
            output.to_csv(sets.wdir + f"output/mortality_{group}_{mode}.OUT", index=False, header=False)

def PostprocessResults(sets, fls):
    
    """
    Postprocess final results and save to CSV file in output folder.
    1. Calculate total mortality and relative mortality for all-ages group by summing the results
    of the three age groups and dividing by the total population of the three age groups respectively.
    2. If IAM format is on, change the format of the results to match the IAM one.
    3. Save results in main working directory.
    """
    
    print("[3] Postprocessing and saving results...")
    
    # Calculate total mortality and relative mortality for all-ages group
    results = AddMortalityAllAges(fls, sets)
    
    # if sets.reporting_tool == True:
        
    #     # Store results in the format of the reporting tool (IMAGE26, Variable, year columns)
    #     pass
    
    # # Reset index and format results for IAMs if specified
    # if IAM_format==True:
    #     results = results.reset_index()
        
    #     # Asign mortality name according to units
    #     results.loc[results["units"] == "Relative Mortality", "var"] = "Relative Mortality"
    #     results.loc[results["units"] != "Relative Mortality", "var"] = "Mortality"

    #     # Rename all temperatures name
    #     results.loc[results["t_type"] == "All", "t_type"] = "All temperatures"

    #     # Rename 'age_group'
    #     results.loc[results["age_group"] == "all population", "age_group"] = "All ages"
    #     results.loc[results["age_group"] == "young", "age_group"] = "0-4 years"
    #     results.loc[results["age_group"] == "older", "age_group"] = "5-64 years"
    #     results.loc[results["age_group"] == "oldest", "age_group"] = "+65 years"

    #     # Create column 'Variable'
    #     results["Variable"] = (
    #         "Health|"
    #         + results["var"]
    #         + "|Non-optimal Temperatures|"
    #         + results["t_type"].str.capitalize()
    #         + "|"
    #         + results["age_group"].str.capitalize()
    #     )
    if sets.reporting_tool == True:
        
        # Export files in .OUT format
        ExportOUTFiles(results, sets)


    results = results.rename(columns={"IMAGE26": "region"})
    
    if sets.adaptation == True:
        adaptation = ""
    else:
        adaptation = "_noadapt"
    if sets.project is not None:
        project = f"{sets.project}"
    else:
        project = ""
        
    # Save results to CSV                
    results.to_csv(sets.wdir +
                   f"output/mortality_{project}_{sets.scenario}{adaptation}_{sets.years[0]}-{sets.years[-1]}.csv") 
    
    print("Scenario ran successfully!")