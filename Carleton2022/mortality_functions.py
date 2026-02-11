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


def CalculateMortality(wdir, years, temp_dir, gdp_dir, project, scenario, regions, adaptation, IAM_format):
    
    """
    Main function to calculate mortality projections for the given parameters.
    1. The function will first read important input data from the wdir/data/ folder 
    through the LoadMainFiles function.
    2. Calculate mortality per year form the years range, loafing first the
    daily temperature data and later using the function MortalityEffectsMinuend.
    3. Calculate the counterfactual factor through the funciton
    mortality_effects_substraend. This will o isolate the role of climate change from 
    changes in temperature-induced mortality that arise due to income growth.
    4. Substract the factors calculated in steps 2 and 3 and save the results
    
    Parameters:
    ----------
    wdir : str
        Path to main working directory. This folder must contain two folders:
        data (with all the data necessary for the model) and output (where results are stored)
    years : list
        Provide the range of years the model will run. The model can run every year or with 
        a longer frequency
    temp_dir: str
        Climate data path. If ERA5 data was chosen, available data is located in my folder:
        "X:/user/liprandicn/Data/ERA5/t2m_daily/". Otherwise, give the path to the climate data 
        from monthly statistics.
    scenario : str
        Possible scenarios:
        - SSP_carleton:
        Scenarios that use the socioeconomic data (population and GDP) from Carleton et al. 
        They can only be run from 2010 until 2100 as the GDP data provided by the authors only reaches that time.
        e.g. SSP2_carleton
        - ERA5 scenario:
        This scenario uses the default socioeconomic data from Carleton et al. (2022) but uses
        ERA5 temperature data from 2000 until 2025.
        e.g. SSP2_ERA5
        - IMAGE scenarios:
        These scenarios use population and GDP data from the IMAGE model. They can be run
        from 2000 until 2100.
    regions : str
        Region classification to use (e.g., "IMAGE26", "countries"). It can run with any region 
        classification from the file: wdir/data/regions
    adaptation: dic
        This dictionary will serve to turn adaptation on or off. It has the following structure:
        {"tmean": A, "loggdppc": B}. 
        o	If adaptation is on, A needs to be “temp_dir” (path to MS data); B can be “default” 
            (will use GDP projections from SSPs) or a path can be given to feed other GDP 
            projections (they need to be at the national level to perform the downscaling).
        o	If adaptation is off, set to None
    IAM_format : bool, optional
        If True, output will be formatted as IAMs' output (default is False)
        
    Returns:
    ----------
    None
    Saves the mortality results to CSV files in the output folder.
    """
    
    print(f"Starting mortality model calculations for scenario {scenario} and years {years}...")
    
    
    if temp_dir == gdp_dir:
        path = temp_dir + "/" + project + "/3_IMAGE_land/scen/" + scenario + "/netcdf/"
    else:
        path = temp_dir  
        
    if ("carleton" in scenario.lower()) and (years[0]<2010):
        print("Error: Carleton's socioeconomic data only reaches 2010. Changing years range...")
        years = [y for y in years if y >= 2010]
    
    # Load necessary files and define variables needed for calculations
    res = LoadMainFiles(
        wdir=wdir,  
        regions=regions, 
        project=project,
        scenario=scenario, 
        years=years,
        temp_dir=path, 
        gdp_dir=gdp_dir,
        adaptation=adaptation
        )  
        
    print("[2] Starting mortality calculations...")
        
    # Iterate over years
    for year in years:          
        
        #Calculate mortality per region and year 
        CalculateMortalityEffects(
            wdir=wdir, 
            year=year, 
            scenario=scenario, 
            temp_dir=path, 
            adaptation=adaptation, 
            regions=regions, 
            res=res
            )
        
    # Post process and save
    PostprocessResults(
        wdir=wdir, 
        years=years, 
        results=res.results, 
        project=project,
        scenario=scenario, 
        IAM_format=IAM_format, 
        adaptation=adaptation, 
        pop=res.pop, 
        region_class=res.region_class,
        age_groups=res.age_groups)
    
    
    
@dataclass
class LoadInputData:
    age_groups: list
    T : np.ndarray
    spatial_relation: gpd.GeoDataFrame
    ir: pd.DataFrame
    region_class: pd.DataFrame
    results: pd.DataFrame
    gammas: any
    pop: pd.DataFrame
    climtas_t0: np.ndarray
    loggdppc_t0 : np.ndarray
    erfs_t0: any
    tmin_t0: any
    gdppc_shares: any
    image_gdppc: any
    daily_temp_t0: pd.DataFrame
    base_years: list = range(2001,2011)

    
    
def LoadMainFiles(wdir, regions, project, scenario, years, temp_dir, gdp_dir, adaptation):
    
    """
    Read and load all main input files required for mortality calculations. The necessary data is 
    the necessary data is located in the wdir/data folder. This includes files for region classification, 
    population data, climate data, and socioeconomic scenarios. The function reads and processes these 
    files to prepare the input data required for mortality calculations.
    
    Parameters:
    ----------
    wdir : str
        Working directory
    temp_dir : str
        Path to climate data files. 
    regions : str
        Region classification 
    scenario : str
    years : list
    temp_dir : str
        Path to climate data files
    adaptation: dic
        This dictionary will serve to turn adaptation on or off. 
        
    Returns:
    LoadResults
    -------
    age_groups : list
        List of age groups
    T : np.ndarray
        Range of daily temperatures for ERF.
    spatial_relation : GeoDataFrame
        Spatial relationship between temperature grid cells and impact regions
    ir : GeoDataFrame
        GeoDataFrame with impact regions
    region_class : DataFrame
        DataFrame with region classification
    results_minuend : DataFrame
        DataFrame to store first part of results
    results_subtrahend : DataFrame
        DataFrame to store second part of results
    gammas : dict
        Dictionary with mortality response functions per age group
    pop : DataFrame
        Population data from selected SSP scenario
    climtas_t0 : np.ndarray
        Array with "present day" T_MEAN. Obtained from the paper.
    loggdppc_t0 : np.ndarray
        Array with "present day" GDPpc. Obtained from the paper.
    erfs_t0 : any
        Dictionary of numpy arrays that store the Exposure Response Functions (ERFs)
        OR None
    tmin_t0 : any
        Dictionary of numpy arrays with the location of the minimum temperature or None
    """
    
    print("[1] Loading main input files...")
    
    # Define age groups, keep order fixed
    age_groups = ["young", "older", "oldest"]
    
    # Define daily temperature range for Exposure Response Functions
    T = np.arange(-20, 40.1, 0.1).round(1)
    
    # Create relationship between temperature data and impact regions
    spatial_relation, ir = GridRelationship(wdir, project, scenario, temp_dir, gdp_dir, years)
    
    # Open file with region classification
    region_class = SelectRegions(wdir, regions)

    # Create results dataframe
    RESULTS = FinalDataframe(regions, region_class, age_groups, years)
    
    # Import gamma coefficients
    gammas = ImportGammaCoefficients(wdir)
    
    # Read population files
    pop = ImportPopulationData(wdir, scenario, years, age_groups, ir)    
    
    # Import present day covariates
    print("[1.4] Loading 'present day' covariates climtas and loggdppc...")
    
    CLIMTAS_T0, LOGGDPPC_T0 = ImportCovariates(
        wdir=wdir, 
        temp_dir=temp_dir, 
        scenario=scenario, 
        ir=ir, 
        year=None, 
        spatial_relation=spatial_relation, 
        adaptation=False, 
        gdppc_shares=None, 
        image_gdppc=None,
        counterfactual=None
        )
    
    # Generate a single time 'present day' ERFs (no adaptation)
    ERFS_T0, TMIN_T0 = GenerateERFAll(
        wdir=wdir,
        temp_dir=temp_dir, 
        scenario=scenario, 
        ir=ir, 
        year=None, 
        spatial_relation=spatial_relation, 
        age_groups=age_groups, 
        T=T, 
        gammas=gammas, 
        adaptation=False, 
        gdppc_shares=None, 
        image_gdppc=None, 
        erfs_t0=None,
        tmin_t0=None,
        counterfactual=None
        ) 
    
    print("[1.5] Loading present-day temperature data...")
    
    # Import present day temperatures
    DAILY_TEMP_T0 = ImportPresentDayTemperatures(
        wdir=wdir, 
        temp_dir=temp_dir, 
        scenario=scenario, 
        base_years=range(2001,2011), 
        ir=ir, 
        spatial_relation=spatial_relation
        )
    
    #  Read GDP shares for scenarios that do not use Carleton's socioeconomic data.
    
    if ("carleton" not in scenario.lower() and "era5" not in scenario.lower() and adaptation):
            
        print("[1.6] Loading GDP data from IMAGE...")
        # Generate GDPpc shares of regions within a country
        gdppc_shares = GenerateGDPpcShares(wdir=wdir, 
                                            ir=ir, 
                                            region_class=region_class)
        
        # Open TIMER gdp file and calculate regional GDP from IMAGE-regional shares
        image_gdppc = ReadOUTFiles(gdp_dir=gdp_dir,
                                   project=project, 
                                    scenario=scenario)
            
    # Set to None when using Carleton's socioeconomic data or when adaptation is off        
    else:  
    
        gdppc_shares = None
        image_gdppc = None
        
        
    return LoadInputData(
        age_groups=age_groups,
        T = T,
        spatial_relation=spatial_relation,
        ir=ir,
        region_class=region_class,
        results=RESULTS,
        gammas = gammas,
        pop = pop,
        climtas_t0 = CLIMTAS_T0,
        loggdppc_t0 = LOGGDPPC_T0,
        erfs_t0 = ERFS_T0,
        tmin_t0 = TMIN_T0,
        gdppc_shares = gdppc_shares,
        image_gdppc = image_gdppc,
        daily_temp_t0 = DAILY_TEMP_T0
    )


def GridRelationship(wdir, project, scenario, temp_dir, gdp_dir, years):
    
    """
    Create a geopandas.DataFrame with spatial relationship between temperature 
    data points from the nc files and impact regions. Therefore, for each grid cell,
    will assign each grid cell to the impact region it intersects with. If a grid cell 
    intersects multiple regions, it will be assigned to the region with the largest 
    overlapping area. The function can work with any resolution of temeprature data.
    
    Parameters:
    ----------
    wdir : str
        Working directory
    scenario : str
        Determines the type of climate data to use (ERA5 if "ERA5" is specified OR monthly statistics
        if any other scenario is given)
    temp_dir : str
        Path to climate data files
    years : list
        List of years to process
        
    Returns:
    ----------
    relationship : GeoDataFrame
        GeoDataFrame with spatial relationship between temperature grid cells and impact regions.
    ir : DataFrame
        DataFrame with impact regions. This file serves to align the regions of any new data.
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

    # Read climate data
    
    # ---------- If ERA5 data ----------
    if re.search(r"SSP[1-5]_ERA5", scenario):
        # Use function located in the utils_common folder to import ERA5 data in the right format
        grid,_ = tmp.DailyTemperatureERA5(
            era5_dir=temp_dir, 
            year=years[0], 
            temp_type="mean", 
            pop_ssp=None, 
            to_array=False
            )
    
    # --------- If Monthly Statistics (MS) data ----------  
    else:
        
        # --------- IMAGE scenario climate data ----------
        if temp_dir == gdp_dir:
            path = temp_dir + "/" + project + "/3_IMAGE_land/scen/" + scenario + "/netcdf/"
    
        # --------- Other MS scenario data ----------
        else:
            path = temp_dir
        
        # Use function to import monthly statistics (MS) of daily temperature data in the right format
        grid,_ = tmp.DailyFromMonthlyTemperature(
            temp_dir=path, 
            years=years[0], 
            temp_type="MEAN", 
            std_factor=1, 
            to_xarray=True
            )
    
    # Extract coordinates
    coord_names = grid.coords.keys()
    lon_vals = FindCoordinateName(["lon", "longitude", "x"], coord_names, grid)
    lat_vals = FindCoordinateName(["lat", "latitude", "y"], coord_names, grid)
    
    # Calculate grid spacing
    lon_size = np.abs(np.mean(np.diff(lon_vals)))
    lat_size = np.abs(np.mean(np.diff(lat_vals)))    
    
    # Create meshgrid 
    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)  
    
    # Create GeoDataFrame with points and their corresponding square polygons
    points_gdf = gpd.GeoDataFrame({
        "longitude": lon2d.ravel(),
        "latitude": lat2d.ravel(),
        "geometry": [
            CreateSquare(lon, lat, lon_size, lat_size)
            for lon, lat in zip(lon2d.ravel(), lat2d.ravel())
        ]
    })
    
    # Load .shp file with impact regions and set the same coordinate reference system (CRS)
    ir = gpd.read_file(wdir+"/data/carleton_sm/ir_shp/impact-region.shp")
    points_gdf = points_gdf.set_crs(ir.crs, allow_override=True)
    
    # Make spatial join
    relationship = gpd.sjoin(points_gdf, ir, how="inner", predicate="intersects")
    
    # Keep only necessary columns
    relationship = relationship[["geometry","index_right", "hierid"]]

    return relationship, ir["hierid"]



def FindCoordinateName(possible_names, coord_names, temperature):
    
    """
    Find the correct coordinate name in the dataset. This avoids having to
    set manually names such as "lon" or "longitude".
    
    Parameters:
    ----------
    possible_names : list
        List of possible coordinate names
    coord_names : list
        List of coordinate names in the dataset
    temperature : xarray DataArray
        Temperature data
        
    Returns:
    ----------
    np.ndarray
        Coordinate values
    """
    
    for name in possible_names:
        if name in coord_names:
            return temperature[name].values
    raise KeyError(f"No coordinate was found among: {possible_names}")



def SelectRegions(wdir, regions):
    
    """
    Select region classification file based on user input. The function will load a file
    created previously that contains the 24,378 impact regions' names and their corresponding
    IMAGE region, country, WHO region... The funciton will only import the impact regions column 
    and the selected calssification.
    
    Parameters:
    ----------
    wdir : str
        Main working directory
    regions : str
        Region classification name (eg. IMAGE26, countries...)
    
    Returns:
    ----------
    region_class:
        Pandas DataFrame with the "hierid" column (impact regions) and the selected region
        classification column.
    """
    
    print(f"[1.2] Loading region classification: {regions}...")
    
    # Load region classification file
    region_class = pd.read_csv(f"{wdir}/data/regions/region_classification.csv")
    
    if regions == "impact_regions":
        region_class = region_class[["hierid", "ISO3"]]
    if regions == "countries":
        region_class = region_class[["hierid", "ISO3", "gbd_level3"]]
    else:
        region_class = region_class[["hierid", "ISO3", regions]]
    
    return region_class



def FinalDataframe(regions, region_class, age_groups, years):
    
    """
    Create results dataframe with multiindex for age groups, temperature types, and regions.
    
    Parameters:
    ----------
    regions : str
        Region classification to use (e.g., "IMAGE26", "countries")
    region_class : DataFrame
        DataFrame with region classification
    age_groups : list
        List of age groups
    years : list
        List of years to process
        
    Returns:
    ----------
    results : DataFrame
        DataFrame to store results
    """
    
    # Extract dataframe with unique regions
    unique_regions = region_class[f"{regions}"].unique()
    unique_regions = unique_regions[~pd.isna(unique_regions)]
    # Append "World" to the list of regions
    unique_regions = np.append(unique_regions, "World")
    
    # Append all population to the list of regions
    age_groups = np.append(age_groups, "all")
    
    t_types = ["Heat", "Cold", "All"]
    results_units = ["Total deaths", "Deaths per 100,000"]
    
    # Create results multiindex dataframe
    results = pd.DataFrame(
        index=pd.MultiIndex.from_product([age_groups, t_types, results_units, unique_regions],
                                         names=["age_group", "t_type", "units", regions]), 
        columns=years
        )
    
    results.sort_index(inplace=True)
    
    return results 



def ImportGammaCoefficients(wdir):    
    
    """
    Import gamma coefficients from Carleton Suplementary Material and convert 
    them to the right format to be multiplied later on by the covariates
    (climtas y loggdppc).
    
    Parameters:
    ----------
    wdir : str
         Main working directory

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
    
    # Read csvv file with the gamma coefficients
    with open(wdir+"data/carleton_sm/Agespec_interaction_response.csvv") as f:
        
        # Extract relevant lines
        for i, line in enumerate(f, start=1):

            if i == 21:
                # Extract 1, climtas, loggdppc
                covar_names = [x for x in line.strip().split(", ")]
                # Convert to indices
                covar_map = {"1":0, "climtas":1, "loggdppc":2}
                # Create numpy array
                cov_idx = np.array([covar_map[str(x)] for x in covar_names])
                
            if i == 23:
                # Extract gamma coefficients
                gammas = np.array([float(x) for x in line.strip().split(", ")])
    
    # Reshape arrays to have the coefficients per age group
    gamma_g = gammas.reshape(3,12)
    cov_g = cov_idx.reshape(3,12)
                
    return gamma_g, cov_g



def ImportPopulationData(wdir, scenario, years, age_groups, ir):
    
    print(f"[1.3] Loading Population data for {scenario} scenario at the impact regions level...")
    
    # Extract SSP from scenario string
    match = re.search(r"(?i)\bssp\d+", scenario)
    # Extract corresponding SSP scenario
    ssp = match.group().upper()
    
    # Include ALWAYS population data from 2000 to 2010 (used in the subtrahend part)
    years = sorted(set(years).union(range(2000, 2011)))
        
    # Import population data based on scenario type
    if 'carleton' in scenario.lower() or re.search(r"SSP[1-5]_ERA5", scenario):
        # Import default population data from Carleton et al. (2022)
        pop = ImportDefaultPopulationData(wdir, ssp, years, age_groups, ir)
        
    else:
        # Import IMAGE population data nc4 file and calculate population per impact region
        pop = ImportIMAGEPopulationData(wdir, ssp, years, ir)
    
    return pop



def ImportDefaultPopulationData(wdir, ssp, years, age_groups, ir): 
    
    """
    Read dafault Carleton et al. (2022) population files for a given SSP scenario and age group.
    The files were created in the preprocessing step.
    
    Parameters:
    ----------
    wdir : str
        Working directory
    SSP : str
        Socioeconomic scenario (e.g., "SSP1", "SSP2")
    years : list
        List of years to process
    age_groups : list
        List of the three age groups
    ir : DataFrame
        DataFrame with the correct order impact regions' order. Use to align new DataFrames
        
    Returns:
    ----------
    POPULATION_GROUPS : dict
        Dictionary with population data per age group
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
            [[y for y in years if y >= 2023]] # Keep only years from 2023 onwards from SSPs
            .merge(POP_HISTORICAL, left_index=True, right_index=True)
            .pipe(lambda df: df.set_axis(df.columns.astype(str), axis=1))
            .reindex(ir.values) # Align to impact regions order
            .reset_index()
            .rename(columns={"hierid":"index"})
        )
        
        POPULATION_GROUPS[age_group] = POP_SSP
    
    return POPULATION_GROUPS



def ImportIMAGEPopulationData(wdir, ssp, years, ir):
    
    """
    Read population data from IMAGE nc4 files for a given SSP scenario and age group.
    
    Parameters:
    ----------
    wdir : str
        Working directory
    SSP : str
        Socioeconomic scenario (e.g., "SSP1", "SSP2")

    Returns:
    ----------
    POP_SSP : dict
        Dictionary with population data per age group
    """
    
    POP_SSP_YOUNG = pd.read_csv(f"{wdir}/data/population/pop_ssp/pop_{ssp.lower()}_young.csv")
    POP_SSP_YOUNG = POP_SSP_YOUNG[["hierid", "ISO3"] + [c for c in POP_SSP_YOUNG.columns if int(c) in years]]
    POP_SSP_OLDER = pd.read_csv(f"{wdir}/data/population/pop_ssp/POP_{ssp.lower()}_older.csv")
    POP_SSP_OLDER = POP_SSP_OLDER[["hierid", "ISO3"] + [c for c in POP_SSP_OLDER.columns if int(c) in years]]
    POP_SSP_OLDEST = pd.read_csv(f"{wdir}/data/population/pop_ssp/POP_{ssp.lower()}_oldest.csv")
    POP_SSP_OLDEST = POP_SSP_OLDEST[["hierid", "ISO3"] + [c for c in POP_SSP_OLDEST.columns if int(c) in years]]
    
    return {"young":POP_SSP_YOUNG,
            "older":POP_SSP_OLDER,
            "oldest":POP_SSP_OLDEST}




def GenerateERFAll(wdir, temp_dir, scenario, ir, year, spatial_relation, age_groups, T, gammas, adaptation, 
                   gdppc_shares, image_gdppc, erfs_t0, tmin_t0, counterfactual):
    
    """
    The code imports the gamma coefficients and the covariates (climtas and loggdppc) and feeds 
    this data into the function Generate ERF Group that will generate the Exposure Response Functions 
    (ERFs) and minimum temperature values per impact region and group. It will locate them in two
    dictionaries.
    
    Parameters:
    ----------
    wdir : str
        Main working directory
    temp_dir : str
        Path to where climate data is stored
    scenario : str
        Scenario
    ir : DataFrame
        DataFrame with impact regions. This file serves to align the regions of any new data.
    year : int
        Year used to import the climate (climtas, 30 year mean) and the income (13 year mean)
    spatial_relation : GeoDataFrame
        Geodataframe with the relationship between gridcells and impact regions. Used to align climate
        data to the impact region level.
    age_groups : list
        List of the three age groups.
    T : range
        Range of daily temperatures used to construct tthe Exposure Response Fucntions (ERFs).
    adaptation : dic
        Dictionary to determine whether adaptaiton is taken into account to generate the ERFs. Parameters
        of the dictionary serve to determine the type of adaptaiton used (with or without damages
        included in the income part).

    Returns:
    ----------
    mor_np: 
        Dictionary with the three 2-d arrays corresponding to each age group.
    tmin:
        Dictionary with the three 1-d arrays corresponding to the minimum temperature.
    """
    
    # Read coefficientes of covariates from Carleton SM
    gamma_g, cov_g = gammas[0], gammas[1]
    
    # Import covariates with or without adaptation
    climtas, loggdppc = ImportCovariates(
        wdir=wdir, temp_dir=temp_dir, 
        scenario=scenario, 
        ir=ir, 
        year=year, 
        spatial_relation=spatial_relation, 
        adaptation=adaptation, 
        gdppc_shares=gdppc_shares, 
        image_gdppc=image_gdppc,
        counterfactual=counterfactual
        )

    # Create covariates matrix
    covariates = np.column_stack([np.ones(len(climtas)), climtas, loggdppc])

    mor_np = {}; tmin = {}        

    # Generate arrays with erf and tmin per age group
    for i, group in enumerate(age_groups):
        
        if erfs_t0 is None or tmin_t0 is None:
            erfs_t0_group = None
            tmin_t0_group = None
            
        else:
            erfs_t0_group = erfs_t0[group]
            tmin_t0_group = tmin_t0[group]           
        
        mor_np[group], tmin[group] = GenerateERFGroup(i, covariates, gamma_g, cov_g, T, erfs_t0_group, tmin_t0_group)
        
    return mor_np, tmin



def ImportCovariates(wdir, temp_dir, scenario, ir, year, spatial_relation, adaptation, 
                     gdppc_shares, image_gdppc, counterfactual):
    
    """
    The main purpose of the function will be to give the covariates climtas and 
    loggdppc of the corresponding year as numpy arrays.
    
    The choice of this covariates will depend on whether adaptation is activated 
    or not and which GDP is fed into the function:
        - If adaptation is off (None), the function will import the covariates defined
        in the paper of Carleton et al as the "present day" covariates. The data is
        extracted directly from Carleton Supplementary Material
        - If adaptation is on the model can:
            - Climtas:
                The model will always import the data from the MS (monthly statistics)
            - GDP:
                A. Import the "default" GDP covariate from the paper (following the SSP
                projections from 2022)
                B. Import other GDP (2024 GDP projections, GDP with damages...)
                
    Parameters:
    ----------
    wdir : str
        Main working directory
    temp_dir : str
        Path to where climate data is stored
    scenario : str
        SSP scenario used for the GDP projections
    ir : DataFrame
        DataFrame with impact regions. This file serves to align the regions of any new data.
    year : int
        Year used to import the climate (climtas, 30 year mean) and the income (13 year mean).
    spatial_relation : GeoDataFrame
        GDF witht the relation between the grid cells of the climate data and the impact regions.
        Every grid cell has an impact region assigned
    adaptation : dic
        Dictionary to determine whether adaptaiton is taken into account to generate the ERFs. Parameters
        of the dictionary serve to determine the type of adaptaiton used (with or without damages
        included in the income part).

    Returns:
    ----------
    climtas : np.ndarray
        1D array with the 30 year climate mean per impact region, the order of the regions is given by ir.
    loggdppc : np.ndarray 
        1D array with the log of the 13 year gdppc mean per impact region, the order of the regions is given by ir.
    """
    
    # No adaptation -----------------------------------------
    # Use the "present day" covariates from Carleton et al. (2022) as the default covariates for the ERF generation.
    if adaptation==False:
        
        # Open covariates for "present day" (no adaptation)
        covariates_t0 = pd.read_csv(wdir+"data/carleton_sm/main_specification/mortality-allpreds.csv")
        
        # Rename regions column to reindex woth ir dataframe
        covariates_t0 = (
            covariates_t0
            .rename(columns={"region":"hierid"})
            .set_index("hierid")
            .reindex(ir.values)
        )
        
        # Extract only climtas and loggdppc as arrays
        climtas = covariates_t0["climtas"].values
        loggdppc = covariates_t0["loggdppc"].values
    
    # Adaptation ---------------------------------------------
    # Open covariates depending on the year and the scenario.
    else:
        
        # CLIMTAS ---------------------------
        if counterfactual==False:
            # Load climatology (30-year running mean, climtas) of the selected year
            climtas = ImportClimtas(temp_dir, year, spatial_relation, ir)
            
        else: 
            # Load climatology (30-year running mean, climtas) of the "present-day" period (2000-2010)
            climtas = ImportPresentDayClimtas(temp_dir, spatial_relation, ir)
        
        # GDP -------------------------------
        if 'carleton' in scenario.lower():
            # Load log(GDPpc) from Carleton et al. (2022) for the selected year and scenario
            loggdppc = ImportLogGDPpc(wdir, scenario, ir, year)
            
        elif gdppc_shares is None or image_gdppc is None:
            # Do not load any GDP data as these settings are used when loading present day covariates
            loggdppc = None
            
        else: 
            # Load log(GDPpc) at the impact region level using the GDPpc output from a TIMER run and 
            # the GDPpc shares to disaggregate it to the impact region level for a selected year
            loggdppc = ImportIMAGEloggdppc(year, image_gdppc, gdppc_shares)
        
    return climtas, loggdppc



def ImportLogGDPpc(wdir, scenario, ir, year):
    
    """
    Read GDP per capita files for a given SSP scenario.
    - If "default" follows the ssp string, the code will read the default SSP projections from 
    The files were created in the preprocessing step.
    
    Parameters:
    ----------
    wdir : str
        Working directory
    ssp : str
        Socioeconomic scenario (e.g., "SSP1", "SSP2")
    ir : DataFrame
        DataFrame with impact regions. This file serves to align the regions of any new data.
    year : int 
        Year used to get the loggdppc.
        
    Returns:
    ----------
    gdppc : np.ndarray
        DataFrame with GDP per capita data
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



def ImportIMAGEloggdppc(year, image_gdppc, gdppc_shares):
    
    """
    Calculate log(GDPpc) at the impact region level using the GDPpc output
    from a TIMER run.

    Returns:
        _type_: _description_
    """
    
    # Extract relevant year data (13 year rolling mean)
    image_gdppc = (
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
    gdppc = gdppc_shares.merge(image_gdppc, left_on="IMAGE26", right_on="region", how="left")
    
    # Calculate share of log(GDPpc) based on regional GDPpc
    gdppc["gdppc"] = gdppc["Value"] * gdppc["gdppc_share"] 
    gdppc["loggdppc"] = np.log(gdppc["gdppc"])
    
    return gdppc["loggdppc"].values

 

def GenerateGDPpcShares(wdir, ir, region_class):
    
    """
    Generate the corresponding GDPpc shares per impact region within an IMAGE region.
    The function will read the GDPpc data from Carleton et al. (2022) and calculate the share
    of each impact region within an IMAGE region. The final output will be a dataframe with the GDPpc
    shares per impact region.
    
    Parameters:
    ----------
    wdir : str
        Main working directory
    ir : DataFrame
        DataFrame with impact regions. This file serves to align the regions of any new data.
    region_class : DataFrame
        DataFrame with region classification
    
    Returns:
    ----------
    gdppc : DataFrame
        DataFrame with GDPpc shares per impact region.
    """
    
    # GDPPC_CETAL_DF = GDPPC_CETAL.to_dataframe().reset_index().merge(region_class, left_on='region', right_on="hierid")
    # GDPPC_CETAL_DF["gdppc_ir_shares"]=GDPPC_CETAL_DF.groupby(["model", "year", "ssp", "ISO3"])["gdppc"].transform(lambda x: x/x.sum())
    # GDPPC_CETAL_DF["gdppc_times_shares"] = GDPPC_CETAL_DF["gdppc"] * GDPPC_CETAL_DF["gdppc_ir_shares"]
    # GDPPC_CETAL_DF["gdppc_country"]=GDPPC_CETAL_DF.groupby(["model", "year", "ssp", "ISO3"])["gdppc_times_shares"].transform(lambda x: x.sum())
    # GDPPC_CETAL_DF["share_country_gdp"] = GDPPC_CETAL_DF["gdppc_times_shares"] / GDPPC_CETAL_DF["gdppc_country"]
    # GDPPC_CETAL_DF["ir_factor"] = GDPPC_CETAL_DF["share_country_gdp"]/GDPPC_CETAL_DF["gdppc_ir_shares"]
    
    # Open GDP data (can be any SSP), convert to dataframe, merge with region classification and filter
    GDPPC_CETAL_DF = (
        xr.open_dataset(f"{wdir}/data/carleton_sm/econ_vars/SSP2.nc4")
        .to_dataframe()
        .reset_index()
        .merge(region_class, left_on="region", right_on="hierid")
        .query("model == 'high' and year == 2010")
        .drop(["model", "year", "ssp"], axis=1)
    )
    
    # Calculate GDPpc shares per impact region within IMAGE region
    GDPPC_CETAL_DF["gdppc_ir_shares"] = GDPPC_CETAL_DF.groupby(["IMAGE26"])["gdppc"].transform(lambda x: x/x.sum())
    GDPPC_CETAL_DF["gdppc_times_shares"] = GDPPC_CETAL_DF["gdppc"] * GDPPC_CETAL_DF["gdppc_ir_shares"]
    GDPPC_CETAL_DF["gdppc_country"]=GDPPC_CETAL_DF.groupby(["IMAGE26"])["gdppc_times_shares"].transform(lambda x: x.sum())
    GDPPC_CETAL_DF["share_country_gdp"] = GDPPC_CETAL_DF["gdppc_times_shares"] / GDPPC_CETAL_DF["gdppc_country"]
    GDPPC_CETAL_DF["gdppc_share"] = GDPPC_CETAL_DF["share_country_gdp"]/GDPPC_CETAL_DF["gdppc_ir_shares"]

    # Reindex according to ir dataframe and select relevant columns
    GDPPC_CETAL_DF = (
        GDPPC_CETAL_DF
            .set_index("region")
            .reindex(ir.values)
            .reset_index()
            .loc[:, ["region", "IMAGE26", "gdppc_share"]]
    )
    
    return GDPPC_CETAL_DF



def ReadOUTFiles(gdp_dir, project, scenario):
    
    """
    Read GDPpc data from TIMER output files. The code will read the files for the selected scenario and project.
    
    Parameters:
    ----------
    gdp_dir : str
        Directory where the GDP data is stored.
    project : str
        Project name.
    scenario : str
        Scenario name.
        
    Returns:
    ----------
    xr_vars : xarray.Dataset
        Dataset containing the GDPpc data per IMAGEregion and year.
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
    # prism_regions = prism.Dimension('region', _DIM_IMAGE_REGIONS + ["Other"] + ["World"])
    prism_regions_world = prism.Dimension('region', _DIM_IMAGE_REGIONS + ["World"])
    
    listy = []

    # Read GDPpc data from TIMER output files. The code will read the files for the selected scenario and project.
    path_clim = gdp_dir + "/" + project + "/2_TIMER/outputlib/TIMER_3_4/" + project + "/"+ scenario + "/indicators/Economy/"
    
    # TODO: Change variable name as it is GDPpc with or without impacts depending on scenario
    VAR = "GDPpc_incl_impacts"
    
    # Create xarray dataset with the data from the OUT files. The code will read the files for the selected scenario and project.
    datafile = prism.TimeVariable(
            timeline=Timeline,
            dims=[prism_regions_world],
            file=path_clim+VAR+".OUT",
        )
    
    # The code will read the files for the selected scenario and project and create an xarray dataset with the data from the OUT files.
    listy.append(xr.merge([datafile[i]
                           .rename('Value')
                           .expand_dims({"Time": [i]}) for i in np.arange(_DIM_TIME['start'], 2101)])
                 .expand_dims({"Scenario": [scenario], "Variable": [VAR]}))
    
    xr_vars = xr.merge(listy)
 
    return xr_vars



def ImportClimtas(temp_dir, year, spatial_relation, ir):
    
    """
    Import climate data from the specified directory and return 'climtas', defined by Carleton as the
    30-year running mean temperature per impact region.
    1. The code will first calculate the annual mean from the mothly data to alter calculate the mean of 
    the past 30 years from the selected year. 
    2. The code will calculate the mean temperature per impact region using "spatial_relation" and will 
    return the data as a numpy array
    
    Parameters:
    ----------
    temp_dir : str
        Path where climate data is stored.
    year : int
        Year used to get the climtas.
    spatial_relation : GeoDataFrame
        GDF witht the relation between the grid cells of the climate data and the impact regions.
        Every grid cell has an impact region assigned.
    ir : DataFrame
        DataFrame with impact regions. This file serves to align the regions of any new data.
        
    Returns:         
    ----------
    climtas : np.ndarray
        1D array with the 30 year running mean of the anual temperature per impact region level.
        The order of each element of the array follows the order of the impact regions given by ir.
    """
    
    # Read monthly mean of daily mean temperature data
    MONTHLY_TEMPERATURE = xr.open_dataset(temp_dir+f"GTMP_MEAN_30MIN.nc")
    
    # Calculate annual mean temperature and climatology
    ANNUAL_TEMPERATURE = MONTHLY_TEMPERATURE["GTMP_MEAN_30MIN"].mean(dim="NM")
    
    # Calculate 30-year rolling mean temperature
    CLIMATOLOGY_TEMPERATURE = ANNUAL_TEMPERATURE.rolling(time=30, min_periods=1).mean().sel(time=f"{year}")
    
    # Assign pixels to every impact region
    TEMP_DICT = {}
    CLIMATOLOGY_T_VALUES = CLIMATOLOGY_TEMPERATURE.values.ravel()
    TEMP_DICT[year] = CLIMATOLOGY_T_VALUES[spatial_relation.index]

    # Calculate mean temperature per impact region and round
    CLIMTAS = pd.DataFrame(TEMP_DICT, index=spatial_relation["index_right"])
    CLIMTAS = CLIMTAS.groupby("index_right").mean()
    
    # Fill in nan with 20
    CLIMTAS = CLIMTAS.fillna(20)
    CLIMTAS.insert(0, "hierid", ir)
    CLIMTAS = CLIMTAS.rename(columns={year: "tmean", "hierid":"region"})
    
    return CLIMTAS["tmean"].values



def ImportPresentDayClimtas(temp_dir, spatial_relation, ir):
    
    """
    Import climate data from the specified directory and return 'climtas', defined by Carleton as the
    30-year running mean temperature per impact region.
    1. The code will first calculate the annual mean from the mothly data to alter calculate the mean of 
    the past 30 years from the selected year. 
    2. The code will calculate the mean temperature per impact region using "spatial_relation" and will 
    return the data as a numpy array
    
    Parameters:
    ----------
    temp_dir : str
        Path where climate data is stored.
    year : int
        Year used to get the climtas.
    spatial_relation : GeoDataFrame
        GDF witht the relation between the grid cells of the climate data and the impact regions.
        Every grid cell has an impact region assigned.
    ir : DataFrame
        DataFrame with impact regions. This file serves to align the regions of any new data.
        
    Returns:         
    ----------
    climtas : np.ndarray
        1D array with the 30 year running mean of the anual temperature per impact region level.
        The order of each element of the array follows the order of the impact regions given by ir.
    """
    
    # Read monthly mean of daily mean temperature data
    TEMP_MEAN = xr.open_dataset(temp_dir+f"GTMP_MEAN_30MIN.nc")
    
    # Calculate annual mean temperature and climatology
    TEMP_MEAN_YEAR = TEMP_MEAN["GTMP_MEAN_30MIN"].mean(dim="NM")
    
    # Calculate 30-year rolling mean temperature
    TEMP_MEAN_30YEAR = TEMP_MEAN_YEAR.rolling(time=30, min_periods=1).mean()
    
    START_YEAR = 2001
    END_YEAR = 2010
    
    TEMP_MEAN_30YEAR_PRESENT = (
        TEMP_MEAN_30YEAR
        .sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-01-01"))
        .mean(dim="time")
    )
    
    # Assign pixels to every impact region
    TEMP_MEAN_VALS = TEMP_MEAN_30YEAR_PRESENT.values.ravel()
    TEMP_MEAN_INDEXED = TEMP_MEAN_VALS[spatial_relation.index]

    # Calculate mean temperature per impact region and round
    TEMP_MEAN_DF = pd.DataFrame(TEMP_MEAN_INDEXED, index=spatial_relation["index_right"])
    TEMP_MEAN_PER_IR = TEMP_MEAN_DF.groupby("index_right").mean()
    
    # Fill in nan with 20 ensuring all regions have a value although mortality null
    TEMP_MEAN_PER_IR = TEMP_MEAN_PER_IR.fillna(20)
    TEMP_MEAN_PER_IR.insert(0, "hierid", ir)
    
    return TEMP_MEAN_PER_IR[0].values
    
    
    
def GenerateERFGroup(model_idx, X, gamma_g, cov_g, T, erfs_t0, tmin):
    
    """
    The code will receive the gamma coefficients, the covariates position and the covariates (X)
    and generate the Exposure Response Function per impact region for a given age group. It will also 
    apply the conditions imposed by Carleton 2022: match the tmin to null mortality and weak
    monotonicity.
    
    Parameters:
    ----------
    model_idx : int
        Index of the age group (0 --> young; 1 --> older; 2 --> oldest) used to extract the gamma 
        coefficients and the covariate values.
    X : np.ndarray
        2D array (matrix) of covariates. The array has dimension 24378x3, with the rows the impact regions, 
        the first column corresponds to the linear term (column of ones), the second column correspond to 
        the climatas values and the third column corresponds to the loggdppc.
    gamma_g : np.ndarray
        3D array with the gamma coefficients that multiply the covariates (climtas, loggdppc) and the 
        independent term (x1). The coefficients are shaped to have a layer per age group and the 12 gamma
        coefficients ordered in matrices of 4x3 (where the rows will multiply every degree of the
        polynomial and the columns will be multiplied by the covariates according to cov_g).
    cov_g : np.ndarray
        3D array with a number (from 0 to 2) that tells each gamma coefficient which covariate should
        multiply (0 --> independent term and multiply by 1, 1 --> multiply by climtas, 
        2 --> multiply by logggdppc)
    T : range
        Range of daily temperatures that the Exposure Response Functions will use as the dependent term.
        The resolution is 0.1 degreee Celsius and the current range goes from -40 to 60 degrees
    
    Returns:
    ----------
    erf_final : np.ndarray
        2D array with the rows representing the ERF of an impact region and the columns, the value the ERF
        would have were the ERF in the corresponding daily temperature.
        As the daily temperature goes from -40 to 60, the first column correspond to the values the ERFs 
        would have at -40.0 degrees, the second one corresponds to the values at -39.9 degrees, and so on...
    tmin : np.ndarray
        1D array with the rows being the daily temperature at which the ERF of an impact region are minimized.
    """
    
    # List of locations of gamma and covariates
    g = gamma_g[model_idx]
    c = cov_g[model_idx]

    # Multiply each covariate by its corresponding gamma
    base = X[:, c] * g
    
    # Compute the sum of the covariates to get polynomial coefficients
    tas = base[:, 0:3].sum(axis=1)  
    tas2 = base[:, 3:6].sum(axis=1)  
    tas3 = base[:, 6:9].sum(axis=1) 
    tas4 = base[:, 9:12].sum(axis=1)
    
    # Rise temperature to the power of 1,2,3,4
    Tpowers = np.vstack([T**1, T**2, T**3, T**4])  

    # Generate raw Exposure Response Function
    erf = (
        tas[:,None] * Tpowers[0] +
        tas2[:,None] * Tpowers[1] +
        tas3[:,None] * Tpowers[2] +
        tas4[:,None] * Tpowers[3]
    )
    
    # Impose zero mortality at tmin by vertically shifting erf
    erf_shifted, tmin_g = ShiftERFToTmin(erf, T, tas, tas2, tas3, tas4, tmin)
    
    #  # Ensure ERFs do not exceed no-adaptation ERFs 
    if erfs_t0 is not None:
        erf_shifted = np.minimum(erf_shifted, erfs_t0)
    
    # Impose weak monotonicity to the left and the right of the erf
    erf_final = MonotonicityERF(T, erf_shifted, tmin_g)

    return erf_final, tmin_g



def ShiftERFToTmin(raw, T, tas, tas2, tas3, tas4, tmin): 
    
    """   
    The code will apply the first constraint imposed by Carleton et al (see more in Appendix pp. A62).
    Following the paper, the minimum of the ERF at Present Day temperatures is located between 20 and 
    30 degrees. Later the function is shifted vertucally to ensure the minimum matches null mortality. 
    This procedure is done for all the ERF per age groups.
    The tmin remains fixed at future times.
     
    Parameters:
    ----------    
    raw : np.ndarray
        Raw ERFs array result of the foruth degree polynomial (see Appendix pp. A35)
    T : range
        Range of daily temperatures
    tas : np.ndarray
        coefficients of first degree of the polynomial (result of 1 + gamma_1,1*climtas + gamma1,2*loggdppc)
    tas2 : np.ndarray
        coefficients of second degree of the polynomial (result of 1 + gamma_2,1*climtas + gamma2,2*loggdppc)
    tas3 : np.ndarray
        coefficients of third degree of the polynomial (result of 1 + gamma_3,1*climtas + gamma3,2*loggdppc)
    tas4 : np.ndarray
        coefficients of fourth degree of the polynomial (result of 1 + gamma_3,1*climtas + gamma3,2*loggdppc)
        
    Returns:
    ----------
    erf : np.ndarray
        2D array with ERFs shifted vertically
    tmin_g : np.ndarray
        1D array with the rows being the daily temperature at which the ERF of an impact region are minimized.
    """
    
    if tmin is None:
        # Locate idx of T (temperature array) between 20 and 30 degrees C
        idx_min_start = np.where(np.isclose(T, 10.0, atol=0.05))[0][0]
        idx_min_end   = np.where(np.isclose(T, 30.0, atol=0.05))[0][0]
        segment = raw[:, idx_min_start:idx_min_end]
        
        # Find local minimum of erf between 20 and 30 degrees
        idx_local_min = np.argmin(segment, axis=1)
        tmin = T[idx_min_start + idx_local_min]
        
    # Calcualte mortality value at fixed tmin
    erf_at_tmin = tas*tmin + tas2*tmin**2 + tas3*tmin**3 + tas4*tmin**4
    
    # Shift vertical functions so tmin matches 0 deaths
    erf = raw - erf_at_tmin[:,None]
        
    return erf, tmin



def MonotonicityERF(T, erf, tmin_g):
    
    """
    The code applies the second constraint from Carleton et al (see Appendix pp. A65), weak 
    monotonicity. The code ensures that towards colder and hotter temperatures than the tmin, 
    the ERFs must be at least as harmful as temperatures closer to the minimum mortality temperature.
    
    Parameters:
    ----------
    T : range
        Range of daily temperatures
    erf : np.ndarray
        ERFs array result of the foruth degree polynomial (see Appendix pp. A35) and the vertical shift
    tmin_g :  np.ndarray
        1D array with the rows being the daily temperature at which the ERF of an impact region are minimized.
        
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
    
    
        
def DailyTemperature2IR(temp_dir, year, ir, spatial_relation, scenario):
    
    """
    Convert daily temperature data of one year to impact region level.
    All grid cells intersecting an impact region are considered.
    Return a dataframe with mean daily temperature per impact region for the given year.
    
    Parameters:
    ----------
    temp_dir : str
        Path where climate data is stored
    year : int
        Year of interest
    ir : GeoDataFrame
        GeoDataFrame with impact regions
    spatial_relation : GeoDataFrame
        Spatial relationship between temperature grid cells and impact regions
    scenario : str
        Determines the type of temperature to import (ERA5 or monthlfscenrioy statistics)
        
    Returns:
    ----------
    df_rounded : DataFrame
        DataFrame with daily mean temperature per impact region for the given year
    """
    
    print(f"[2.1] Loading daily temperature data for year {year}...")
    
    if "ERA5" in scenario:
        
        # Open daily temperature data from ERA5
        DAILY_TEMPERATURE = ERA5Temperature2IR(temp_dir, year, ir, spatial_relation)
        
    else:
                
        # Read daily temperature data generated from monthly statistics
        DAILY_TEMPERATURE,_ = tmp.DailyFromMonthlyTemperature(
            temp_dir=temp_dir, 
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
    DAILY_TEMPERATURE = DAILY_TEMPERATURE.iloc[:,1:].to_numpy()
    
    return DAILY_TEMPERATURE



def MSTemperature2IR(temp, year, ir, spatial_relation):
    
    """
    Import daily temperature data of one year from montlhy statistics and convert it
    to the impact region level.
    
    Parameters:
    ----------
    temp_dir : str
        Path where climate data is stored
    year : int
        Year of interest
    ir : GeoDataFrame
        GeoDataFrame with impact regions
    spatial_relation : GeoDataFrame
        Spatial relationship between temperature grid cells and impact regions
        
    Returns:
    ----------
    df_rounded : DataFrame
        DataFrame with daily mean temperature per impact region for the given year
    """
    
    # Create a list of dates for the specified year
    date_list = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D").astype(str)
    
    # Temporarily store daily temperatures in a dictionary
    TEMPERATURE_DIC = {}
    for i, day in enumerate(date_list):
        DAILY_TEMPERATURE = temp[...,i].ravel()
        TEMPERATURE_DIC[day] = DAILY_TEMPERATURE[spatial_relation.index]

    # Calculate mean temperature per impact region and round
    DAILY_TEMPERATURES_DF = pd.DataFrame(TEMPERATURE_DIC, index=spatial_relation["index_right"])
    
    # Calculate mean temperature per impact region, fill in nan with 20 and round to 1 decimal
    DAILY_TEMPERATURES_DF = (
        DAILY_TEMPERATURES_DF
        .groupby("index_right")
        .mean()
        .fillna(20)
        .round(1)
    )
    
    # Insert hierid column with impact region names
    DAILY_TEMPERATURES_DF.insert(0, "hierid", ir)
    
    return DAILY_TEMPERATURES_DF



def ERA5Temperature2IR(temp_dir, year, ir, spatial_relation):
    
    """
    Import daily temperature data of one year from ERA5 and convert it
    to the impact region level.
    
    Parameters:
    ----------
    temp_dir : str
        Path where climate data is stored
    year : int
        Year of interest
    ir : GeoDataFrame
        GeoDataFrame with impact regions
    spatial_relation : GeoDataFrame
        Spatial relationship between temperature grid cells and impact regions
        
    Returns:
    ----------
    df_rounded : DataFrame
        DataFrame with daily mean temperature per impact region for the given year
    """
    
    # Read ERA5 daily temperature data for a specific year
    DAILY_TEMPERATURE, _ = tmp.DailyTemperatureERA5(
        era5_dir=temp_dir,
        year=year, 
        temp_type="mean", 
        pop_ssp=None, 
        to_array=False)
    
    DAILY_TEMPERATURE = DAILY_TEMPERATURE.t2m
    
    # Select all available dates
    DATES = DAILY_TEMPERATURE["valid_time"].values
    
    # Create a list of dates for the specified year
    DATE_LIST =(
        DATES[np.isin(DAILY_TEMPERATURE["valid_time"]
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
    DAILY_TEMPERATURE_DF = pd.DataFrame(TEMPERATURE_DIC, index=spatial_relation["index_right"])
    
    DAILY_TEMPERATURE_DF = (
        DAILY_TEMPERATURE_DF
        .groupby("index_right")
        .mean()
        .round(1)
    )

    DAILY_TEMPERATURE_DF.insert(0, "hierid", ir)
    
    return DAILY_TEMPERATURE_DF



def CalculateMortalityEffects(wdir, year, scenario, temp_dir, adaptation, regions, res):
    
    """
    The code calculates equaiton 2a or 2c from the paper, depending whether adaptation is on or off.
    1. It first calculates the first part of the equation (called minuend here) and then the second part 
    (the counterfactual mortality called subtrahend here).
    2. The substraction is done per impact region, age group, and type of temperature (all, heat, cold).
    3. Finally, it will agregate the results spatially to the selected region classification (IMAGE26, ISO3...)
    
    Parameters:
    ----------
    wdir : str
        Main working directory
    year : int
        Year to calculate mortality
    scenario : str
        Scenario
    temp_dir : str
        Path where climate data is stored
    adaptation : dic
        Dictionary with the adaptation parameters
    regions : str
        Name of the region classification
    res : class
        Class where the input files are called
    Returns:
    ----------
    None
    The function will append the results in the DataFrame called results
    """
    
    # Read daily temperature data from specified source
    DAILY_TEMP_T = DailyTemperature2IR(temp_dir=temp_dir, 
                                       year=year, 
                                       ir=res.ir, 
                                       spatial_relation=res.spatial_relation, 
                                       scenario=scenario)
    
    print(f"[2.2] Calculating marginal mortality for year {year}...")
    
    # Calculate mortality per region and year (first term of equations 2' or 2a' from the paper)
    MOR_ALL_MIN, MOR_HEAT_MIN, MOR_COLD_MIN =  CalculateMarginalMortality(
        wdir=wdir, 
        temp_dir=temp_dir, 
        year=year, 
        scenario=scenario, 
        daily_temp=DAILY_TEMP_T, 
        adaptation=adaptation, 
        res=res,
        counterfactual=False
        )
    
    print(f"[2.3] Calculating counterfactual mortality for year {year}...")

    # Calculate mortality per region and year (second term of equations 2' or 2a' from the paper)
    MOR_ALL_SUB, MOR_HEAT_SUB, MOR_COLD_SUB = CalculateMarginalMortality(
        wdir=wdir, 
        temp_dir=temp_dir, 
        year=year,
        scenario=scenario, 
        daily_temp=res.daily_temp_t0, 
        adaptation=adaptation, 
        res=res,
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
            
            

def CalculateMarginalMortality(wdir, temp_dir, year, scenario, daily_temp, adaptation, res, counterfactual):
    
    """
    Calculate mortality effects from non optimal temperatures (first term of equation 2 from paper).
    Depending whether adaptation is on, the code will either import the ERFs with no adaptation or
    generate ERFs depending on the adaptation parameters.
    The daily temperature data will be converted to indices based on the range of T.
    Mortality per impact region will be calculated per age group and temperature type (all, heat and cold) 
    in the Mortality From Temperature Index function.
    
    Parameters:
    ----------
    wdir : str
        Main working directory
    year : int
        Year to calculate mortality
    scenario : str
        Scenario
    temp_dir : str
        Path where climate data is stored
    adaptation : dic
        Dictionary with the adaptation parameters
    daily_temp : np.ndarray
        Array with the daily temperatures per impact region for a given region
    regions : str
        Name of the region classification
    res : class
        Class where the input files are called
        
    Returns:
    ----------
    None
    The function will appedn the results in the DataFrame called results minuend
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
            wdir=wdir,
            temp_dir=temp_dir,
            scenario=scenario,
            ir=res.ir,
            year=year,
            spatial_relation=res.spatial_relation, 
            age_groups=res.age_groups, 
            T=res.T, 
            gammas=res.gammas, 
            adaptation=adaptation,
            gdppc_shares=res.gdppc_shares, 
            image_gdppc=res.image_gdppc, 
            erfs_t0=res.erfs_t0,
            tmin_t0=res.tmin_t0,
            counterfactual=counterfactual
            )
    
    # Use pre-calculated ERFs with no adaptation or income growth
    else: 
        ERFS_T = res.erfs_t0, 
        
    # ------------------- Calculate mortality ------------------
    
    MOR_ALL, MOR_HEAT, MOR_COLD = {}, {}, {}
    
    for group in res.age_groups:      
        MOR_ALL[group], MOR_HEAT[group], MOR_COLD[group] = MortalityFromTemperatureIndex(
            daily_temp=DAILY_TEMP, 
            rows=ROWS, 
            erfs=ERFS_T, 
            tmin=res.tmin_t0,
            min_temp=MIN_TEMP, 
            group=group)
            
    return MOR_ALL, MOR_HEAT, MOR_COLD



def ImportPresentDayTemperatures(wdir, temp_dir, scenario, base_years, ir, spatial_relation):
    
    """
    The function will import the daily temperatures from 2000 to 2010 calculated in the
    preprocessing step usign either ERA5 data or climate data from prescribed scenario.
    
    Parameters:
    ----------
    wdir : str
        Path to main working directory
    
    Results:
    ----------
    T_0 : dic
        Dictionary of numpy arrays with the daily temperature per impact region and year.
    """
    
    # ------------------ ERA5 ------------------
    # Load pre-calculated present day temperatures from ERA5
    # TODO: finish this part
    if re.search(r"SSP[1-5]_ERA5", scenario):
        
        T_0 = {}
        
        for year in base_years:
            
            # Read pre-calculated daily temperature at impact region level
            ERA5_T0 = pd.read_csv(wdir+f"data/climate_data/ERA5_T0_{year}.csv")
            
            # Store in dictionary as numpy arrays
            T_0[year] = ERA5_T0.iloc[:,2:].to_numpy()
            
        YEARS_NO_LEAP = []
        
        for year, arr in T_0.items():
            if arr.shape[1] == 366:
                arr = np.delete(arr, 59, axis=1)
            YEARS_NO_LEAP.append(year)
        T0_MEAN = np.mean(YEARS_NO_LEAP, axis=0)
            
            
    # -------------- Scenario data --------------
    # Load daily temperature data from prescribed scenario
    else: 
        
        DAILY_TEMPERATURE,_ = tmp.DailyFromMonthlyTemperature(
            temp_dir=temp_dir, 
            years=base_years,
            temp_type="MEAN",
            std_factor=1, 
            to_xarray=False
        )

        T0_MEAN = MSTemperature2IR(
            temp=DAILY_TEMPERATURE, 
            year=2000, 
            ir=ir, 
            spatial_relation=spatial_relation)
        
        # Convert "Present-day" temepratures dataframe to numpy array    
        T0_MEAN = T0_MEAN.iloc[:,1:].to_numpy()
        
    return T0_MEAN

    

def MortalityFromTemperatureIndex(daily_temp, rows, erfs, tmin, min_temp, group):
    
    """
    The code calculates annual relative mortality fusing the ERFs and the daily temperature
    data in a stylized way: 
    1. It takes the temperature indices calculated one function outside and are used as input to
    locate the corresponding mortality value from the ERF array. 
    2. It sums the daily mortality values to the annual level
    3. Separates the daily temperatures into temperatures above the tmin (Heat temperatures) and 
    below the tmin (cold temperature) and repeats steps 2 and 3 for these temperature types.

    Parameters:
    ----------
    daily_temp : np.ndarray
        Daily temperature data per impact region for a given year
    temp_idx : np.ndarray
        The corresponding index of the daily temperature data based on T. 
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
        
    Returns:
    ---------
    result_all : np.ndarray
        Annual relative mortality from all Non-Optimal temperatures (Heat and cold)
    result_heat : np.ndarray
        Annual relative mortality from heat non-optimal temperatures
    results_cold : np.ndarray
        Annual relative mortality from cold non-optimal temperatures
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
    Aggregate spatially the annual relative mortality from the impact region level
    to the region classification chosen and locate the results of mortality from heat, cold and
    all-type mortality in the final results dataframe.
    
    Parameters:
    ----------
    year : int
        year to process the mortality data
    group : str
        Age group
    mor : np.ndarray
        Array with annual mortality at the impact region level
    regions : str
        Region classification name
    mode : str
        Determine type of temperature (Heat, Cold or All)
    res : class
        
    Returns:
    ----------
    None
    Results are stored in the results DataFrame
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
    REGIONS_INDEX = res.results.loc[(group, mode, "Total deaths"), year].index[:-1]
    res.results.loc[(group, mode, "Total deaths", REGIONS_INDEX), year] = (REGIONS_DF["mor"].reindex(REGIONS_INDEX)).values
    res.results.loc[(group, mode, "Deaths per 100,000", REGIONS_INDEX), year] = (REGIONS_DF["rel_mor"].reindex(REGIONS_INDEX)).values
    
    # Locate global results in results dataframe
    res.results.loc[(group, mode, "Total deaths", "World"), year] = REGIONS_DF["mor"].sum()
    res.results.loc[(group, mode, "Deaths per 100,000", "World"), year] = (REGIONS_DF["mor"].sum() * 1e5 / REGIONS_DF["pop"].sum())



def AddMortalityAllAges(results, pop, region_class, years, age_groups):
    
    region_class = region_class.set_index("hierid")
    
    # Prepare population dataframes for aggregation
    pop = {
        age: df.set_index("hierid")
        for age, df in pop.items()
    }
    
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
    
    # Calculate global population by summing all regions
    pop_all_world = pop_all.sum(axis=0)
    
    # Calculate total mortality and relative mortality for all-ages groups 
    for mode in ["All", "Heat", "Cold"]:
        
        # Calculate total mortality for all age groups
        results.loc[("all", mode, "Total deaths")] = (
            sum(results.loc[(age, mode, "Total deaths")] for age in age_groups)
        ).values
        
        # Calculate relative mortality for all-age group        
        IMAGE26 = results.loc[("all", mode, "Deaths per 100,000")].index[:-1]
        
        results.loc[("all", mode, "Deaths per 100,000", IMAGE26)] = (
            results.loc[("all", mode, "Total deaths", IMAGE26)]
            .mul(1e5)
         .div(pop_all.where(pop_all.reindex(IMAGE26) != 0))
        ).values
        
        # Calculate global relative mortality for all-age group
        results.loc[("all", mode, "Deaths per 100,000", "World")] = (
            results.loc[("all", mode, "Total deaths", "World")]
            .mul(1e5)
            .div(pop_all_world.sum())
        )

    return results



def PostprocessResults(wdir, years, results, project, scenario, IAM_format, adaptation, pop, region_class, age_groups):
    
    """
    Postprocess final results and save to CSV file in output folder.
    1. Calculate total mortality and relative mortality for all-ages group by summing the results
    of the three age groups and dividing by the total population of the three age groups respectively.
    2. If IAM format is on, change the format of the results to match the IAM one.
    3. Save results in main working directory.
    """
    
    print("[3] Postprocessing and saving results...")
    
    # Calculate total mortality and relative mortality for all-ages group
    RESULTS = AddMortalityAllAges(results, pop, region_class, years, age_groups)
    
    # Reset index and format results for IAMs if specified
    if IAM_format==True:
        RESULTS = RESULTS.reset_index()
        
        # Asign mortality name according to units
        RESULTS.loc[RESULTS["units"] == "Deaths per 100,000", "var"] = "Relative Mortality"
        RESULTS.loc[RESULTS["units"] != "Deaths per 100,000", "var"] = "Mortality"

        # Rename all temperatures name
        RESULTS.loc[RESULTS["t_type"] == "all", "t_type"] = "All temperatures"

        # Rename 'age_group'
        RESULTS.loc[RESULTS["age_group"] == "all", "age_group"] = "All ages"
        RESULTS.loc[RESULTS["age_group"] == "young", "age_group"] = "0-4 years"
        RESULTS.loc[RESULTS["age_group"] == "older", "age_group"] = "5-64 years"
        RESULTS.loc[RESULTS["age_group"] == "oldest", "age_group"] = "+65 years"

        # Create column 'Variable'
        RESULTS["Variable"] = (
            "Health|"
            + RESULTS["var"]
            + "|Non-optimal Temperatures|"
            + RESULTS["t_type"].str.capitalize()
            + "|"
            + RESULTS["age_group"].str.capitalize()
            + "|"
            + RESULTS["units"]
        )
            
        RESULTS = RESULTS[["IMAGE26", "Variable"] + list(RESULTS.columns[4:-2])]
            
        RESULTS = RESULTS.rename(columns={"IMAGE26": "region"})
    
    if adaptation == True:
        adapt = ""
    else:
        adapt = "_noadapt"
        
    # Save results to CSV                
    RESULTS.to_csv(wdir +
                   f"output/mortality_carleton_{project}_{scenario}{adapt}_{years[0]}-{years[-1]}.csv", 
                   index=False) 
    
    print("Scenario ran successfully!")