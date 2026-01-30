import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from dataclasses import dataclass
from shapely.geometry import Polygon
from shapely.geometry import box
import re
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils_common import temperature as tmp
import prism
from rasterio.features import rasterize
import country_converter as coco


### ------------------------------------------------------------------------------


def CalculateMortality(wdir, years, temp_dir, scenario, regions, adaptation, IAM_format=False):
    
    """
    Main function to calculate mortality projections for the given parameters.
    1. The function will first read important input data from the wdir/data/ folder 
    through the LoadMainFiles function.
    2. Calculate mortality per year form the years range, loafing first the
    daily temperature data and later using the function MortalityEffectsMinuend.
    3. Calculate the conterfactual factor through the funciton
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
    
    # Load necessary files and define variables needed for calculations
    res = LoadMainFiles(wdir, temp_dir, regions, scenario, years, temp_dir, adaptation)  
        
    print("[2] Starting mortality calculations...")
        
    # Iterate over years
    for year in years:
        
        #Calculate mortality per region and year 
        CalculateMortalityEffects(wdir, year, scenario, temp_dir, adaptation, regions, res)
        
    # Post process and save
    PostprocessResults(wdir, years, res.results, scenario, IAM_format, adaptation, res.pop, res.region_class)
    
    
    
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

    
    
def LoadMainFiles(wdir, temp_dir, regions, scenario, years, climate_path, adaptation):
    
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
    climate_path : str
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
    spatial_relation, ir = GridRelationship(wdir, scenario, climate_path, years)
    
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
    climtas_t0, loggdppc_t0 = ImportCovariates(wdir, temp_dir, scenario, ir, None, spatial_relation, 
                                               None, None, None)
    
    # Generate a single time 'present day' ERFs (no adaptation)
    erfs_t0, tmin_t0, _, _ = GenerateERFAll(wdir, temp_dir, scenario, ir, None, spatial_relation, 
                                            age_groups, T, gammas, None, region_class, None, None) 
    
    print("[1.5] Loading present-day temperature data...")
    # Import present day temperatures
    BASE_YEARS = range(2001,2011)
    DAILY_TEMP_T0 = ImportPresentDayTemperatures(wdir, temp_dir, scenario, BASE_YEARS, ir, spatial_relation)
    
    # ------------------ If no adaptation
    if adaptation == None:
        
        # Set gdppc shares to None
        gdppc_shares = None
        image_gdppc = None
    
    # ------------------ If adaptation
    else:
        
        # If adaptation is prescribed by Carleton et al. (2022): default mode
        if adaptation.get("loggdppc") == "default":
            gdppc_shares = None
            image_gdppc = None
        
        # If adaptation with custom GDP data from TIMER
        else:
            print("[1.6] Loading GDP data from IMAGE...")
            # Generate GDPpc shares of regions within a country
            gdppc_shares = GenerateGDPpcShares(wdir, ir, region_class)
            
            # Open TIMER gdp file and calculate regional GDP from IMAGE-regional shares
            gdp_dir = adaptation.get("loggdppc")
            image_gdppc = ReadOUTFiles(gdp_dir, scenario)
        
        
    return LoadInputData(
        age_groups=age_groups,
        T = T,
        spatial_relation=spatial_relation,
        ir=ir,
        region_class=region_class,
        results=RESULTS,
        gammas = gammas,
        pop = pop,
        climtas_t0 = climtas_t0,
        loggdppc_t0 = loggdppc_t0,
        erfs_t0 = erfs_t0,
        tmin_t0 = tmin_t0,
        gdppc_shares = gdppc_shares,
        image_gdppc = image_gdppc,
        daily_temp_t0 = DAILY_TEMP_T0
    )


def GridRelationship(wdir, scenario, path, years):
    
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
    climate_path : str
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
            era5_dir=path, 
            year=years[0], 
            temp_type="mean", 
            pop_ssp=None, 
            to_array=False
            )
    
    # --------- If POP ----------
    elif scenario == "POP":
        grid = path
    
    # --------- If Monthly Statistics data ----------  
    else:
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
    results = pd.DataFrame(index=pd.MultiIndex.from_product([age_groups, t_types, results_units, unique_regions],
                                                            names=["age_group", "t_type", "units", regions]), 
                           columns=years)
    
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
    
    print(f"[1.3] Loading Population data for {scenario} scenario and aggregate it to impact regions...")
    
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
        pop = ImportIMAGEPopulationData(wdir, ssp, years)
    
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
        pop_present_day = pd.read_csv(f"{wdir}/data/population/historical_pop/POP_historical_{age_group}.csv")
        
        # Read population files projections per age group
        pop_projection = SSPxarrayToDataframe(wdir, ssp, age_name, ir)
        
        # Merge 'present-day' population with relevant years of scenario projection
        cols = ["region"] + [y for y in years if y >= 2023] 
        pop = pop_present_day.merge(pop_projection[cols], right_on="region", left_on="hierid", how="outer")
        
        # Change column name type and store in dictionary
        pop.columns = pop.columns.astype(str)
        POPULATION_GROUPS[age_group] = pop
    
    return POPULATION_GROUPS



def ImportIMAGEPopulationData(wdir, ssp, years):
    
    """
    The code imports the population data produced by IMAGE and converts it to population data
    for the three age groups at the impact region level.
    
    Parameters:
    ----------
    wdir : str
        Working directory
    ssp : str
        Socioeconomic scenario (e.g., "SSP1", "SSP2")
    years : list
        List of years to process
        
    Returns:
    ----------
    POPULATION_GROUPS : dict
        Dictionary with population data per age group
        
    Data sources:
    ----------
    1. Population data projections per age group from the SSP projections, available at:
        https://dataexplorer.wittgensteincentre.org/wcde-v3/
        The data was populaiton size (000's) at country level for all the 5-year age groups
        and SSP scenarios.
    """
    
    # Agregate raster IMAGE total population per impact region and year
    TOTAL_POPULATION_IR = IMAGEPopulation2ImpactRegion(wdir, ssp, years)
    
    # Load population data projections per age group to disagregate IMAGE data
    POPULATION_GROUPS = LoadAgeGroupPopulationData(wdir, ssp, years)

    # Pivot and merge function
    def pivot_and_merge(group_name):
        df = POPULATION_GROUPS[POPULATION_GROUPS["group"] == group_name].pivot(index=["Area", "ISO3"], columns="Year", values="share").reset_index()
        df = df.rename(columns={c: f"{c}_share" for c in df.columns if isinstance(c, int)})
        return TOTAL_POPULATION_IR.merge(df, on="ISO3", how="left")

    # Create population dataframes for each age group
    POP_YOUNG, POP_OLDER, POP_OLDEST = (pivot_and_merge(g) for g in ['young', 'older', 'oldest'])

    # Multiply shares by total population to get absolute numbers
    for pop in [POP_YOUNG, POP_OLDER, POP_OLDEST]:        
        for y in years:
            pop[str(y)] = pop[str(y)+"_share"] * pop[str(y)]
                    
    NON_SHARE_COLS = [c for c in pop.columns if "share" not in c]

    return {"young":POP_YOUNG[NON_SHARE_COLS],
            "older":POP_OLDER[NON_SHARE_COLS],
            "oldest":POP_OLDEST[NON_SHARE_COLS]}
    
    

def LoadAgeGroupPopulationData(wdir, ssp, years):
    
    # Load population data projections per 5-year age group
    POPULATION_5YEAR_AGE = pd.read_csv(wdir+"/data/population/IMAGE_pop/pop_shares/wcde_data.csv", 
                                       skiprows=8)
    
    POPULATION_5YEAR_AGE = CompletePopulationDataLustrum(POPULATION_5YEAR_AGE)
    
    # Select SSP and years
    POPULATION_5YEAR_AGE = POPULATION_5YEAR_AGE[(POPULATION_5YEAR_AGE['Scenario'] == ssp) & (POPULATION_5YEAR_AGE["Year"].isin(years))]
    
    # Define age groups to classify wcde ages
    AGE_GROUPS = {
        "young": ['0--4'],
        "older": [f'{i}--{i+4}' for i in range(5, 65, 5)],
        "oldest": [f'{i}--{i+4}' for i in range(65, 100, 5)] + ['100+']
    }
    
    # Assign age group to each age
    POPULATION_5YEAR_AGE.loc[:,'group'] = POPULATION_5YEAR_AGE['Age'].map(
        lambda x: next((grp for grp, ages in AGE_GROUPS.items() if x in ages), None)
    )
    
    # Aggregate population by group
    POPULATION_GROUPS = (
        POPULATION_5YEAR_AGE.dropna(subset=['group'])
           .groupby(['Area', 'Year', 'group'], as_index=False)['Population']
           .sum()
    )
    
    # Generate rows with missing years per area and group
    dfs = []

    # Interpolate for every Area and group combination
    for (area, group), group_df in POPULATION_GROUPS.groupby(['Area', 'group']):
        # Create year range
        years = range(group_df['Year'].min(), group_df['Year'].max() + 1)
        # Reindex to include all years
        group_df = group_df.set_index('Year').reindex(years)
        # Keep area and group columns
        group_df['Area'] = area
        group_df['group'] = group
        # Interpolate population values
        group_df['Population'] = group_df['Population'].interpolate(method='linear')
        # Reset index
        group_df = group_df.reset_index().rename(columns={'index': 'Year'})
        dfs.append(group_df)

    # Concateenate all dataframes
    POPULATION_GROUPS_ANNUAL =  pd.concat(dfs, ignore_index=True)

    # Total population per area and year
    POP_TOTAL = POPULATION_GROUPS_ANNUAL.groupby(["Area", "Year"])["Population"].sum().reset_index().rename(columns={"Population": "Population_total"})
    
    # Calculate share of each age group
    POPULATION_GROUPS_ANNUAL = POPULATION_GROUPS_ANNUAL.merge(POP_TOTAL, on=["Area", "Year"])
    POPULATION_GROUPS_ANNUAL["share"] = POPULATION_GROUPS_ANNUAL["Population"] / POPULATION_GROUPS_ANNUAL["Population_total"]
    
    # Convert locations to ISO3
    unique_locations = POPULATION_GROUPS_ANNUAL["Area"].unique()
    conversion_dict = {loc: coco.convert(names=loc, to='ISO3') for loc in unique_locations}
    POPULATION_GROUPS_ANNUAL['ISO3'] = POPULATION_GROUPS_ANNUAL["Area"].map(conversion_dict)
    
    return POPULATION_GROUPS_ANNUAL



def CompletePopulationDataLustrum(df):
    
    """
    Fill in missing years in the population data by forward-filling and backward-filling.
    This ensure that ckountries without data for certain years will have values filled in
    based on the nearest available data, keeping age group share consistent.
    """

    
    # Create a MultiIndex of all combinations of Area, Scenario, and Years
    UNIQUE_MLTIDX = pd.MultiIndex.from_product(
        [
            df["Area"].unique(),
            df["Scenario"].unique(),
            df["Year"].unique(),
            df["Age"].unique()
        ],
        names=["Area", "Scenario", "Year", "Age"]
    )
    
    # Reindex the DataFrame to include all combinations, filling missing values with NaN
    df_full = (
        df
        .set_index(["Area", "Scenario", "Year", "Age"])
        .reindex(UNIQUE_MLTIDX)
        .reset_index()
    )
    
    # Forward-fill and backward-fill missing values within each Population and Scenario group
    df_full["Population"] = (
        df_full
        .groupby(["Area", "Scenario", "Year", "Age"])["Population"]
        .transform("bfill")
        .transform("ffill")
    )
    
    return df_full

                

def IMAGEPopulation2ImpactRegion(wdir, ssp, years):
    
    '''
    Calculate total population per impact region for a specific year from IMAGE land
    population data files.

    This function assigns raster pixels from the IMAGE land population data to impact regions,
    sums the population values within each region, and adds the results as a new column to
    the input GeoDataFrame.

    Parameters
    ----------
    pop :  xarray
        IMAGE Land population nc4 file. Usually 5min resolution.
    impact_regions : geopandas.GeoDataFrame
        GeoDataFrame containing impact region polygons.
    year : int
        Year for which the population is being calculated (e.g., 2000, 2010).

    Returns
    -------
    impact_regions : pd.DataFrame
        oDataFrame with columns for the impact region, ISO3 and the corresponding population 
        per year.
    '''
    
    # Read in impact regions shapefile
    IMPACT_REGIONS = gpd.read_file(wdir+"data/carleton_sm/ir_shp/impact-region.shp")
    
    # Read IMAGE SSP population nc file
    POP_IMAGE = xr.open_dataset(wdir+f"data/population/IMAGE_pop/{ssp.lower()}/GPOP.nc")
    
    # Ensure CRS is set to EPSG:4326 and align with impact regions
    POP_IMAGE = POP_IMAGE.rio.write_crs("EPSG:4326", inplace=False)
    IMPACT_REGIONS = IMPACT_REGIONS.to_crs(POP_IMAGE.rio.crs)

    # Select relevant years including "present-day" years (2000-2010)
    POP_IMAGE = POP_IMAGE.sel(time=pd.to_datetime([f"{y}-01-01" for y in years]))
    
    MINLENGTH = len(IMPACT_REGIONS) + 1

    # Prepare tuples of (geometry, region_id) for rasterization
    SHAPES_AND_IDS = [(geom, idx) for idx, geom in enumerate(IMPACT_REGIONS.geometry, start=1)]
        
    # Rasterize region polygons once
    OUT_SHAPE = POP_IMAGE.isel(time=0).GPOP.shape

    # Get raster transform 
    RASTER_AFFINE = POP_IMAGE.rio.transform()    

    PIXEL_OWNER = rasterize(
        SHAPES_AND_IDS,
        out_shape=OUT_SHAPE,
        transform=RASTER_AFFINE,
        fill=0,          # 0 = without region
        all_touched=False,
        dtype='int32'
    )
    
    for i, year in enumerate(years):
        
        RASTER_DATA = POP_IMAGE.isel(time=i).GPOP.values
        
        # Mask valid data (NaN = nodata)
        VALID_POP_MASK = ~np.isnan(RASTER_DATA)

        # Sum population per region using np.bincount in pixels without NaN
        SUMS = np.bincount(PIXEL_OWNER[VALID_POP_MASK], 
                           weights=RASTER_DATA[VALID_POP_MASK], 
                           minlength=MINLENGTH)[1:]  

        # Add results to impact_regions GeoDataFrame
        IMPACT_REGIONS[f'{year}'] = SUMS
    
    # Add ISO3 column
    IMPACT_REGIONS["ISO3"] = IMPACT_REGIONS["hierid"].str[:3]

    # Only return regions names and population columns
    return IMPACT_REGIONS[["hierid", "ISO3"] + [c for c in IMPACT_REGIONS.columns if c.isdigit()]]



def SSPxarrayToDataframe(wdir, ssp, pop_group, ir):
    
    '''
    Convert Carleton et al. (2022) GDP or population NetCDF files into dataframe format.

    This function extracts regional projections for a given SSP scenario 
    and variable (GDP or population) from the Carleton et al. (2022) NetCDF datasets.
    The resulting dataset is flattened into a pandas DataFrame.

    Parameters
    ----------
    wdir : str
        Main working directory
    ssp : str
        SSP scenario identifier (e.g., 'SSP1', 'SSP3').
    pop_group : str or None
        Population variable to extract from the NetCDF file.
        If data_type == 'GDP', this parameter should typically be 'gdppc'.
    ir : DataFrame
        DataFrame with the correct order impact regions' order. Use to align new DataFrames

    Returns
    -------
    df : pandas.DataFrame
        DatFrame containing the extracted variable across years and regions. 
        Columns include:
        - 'region'
        - One column per year (e.g., 2010, 2015, …)
        - 'hierid'
    '''
      
    # Open xarray from Carleton data
    ssp = xr.open_dataset(wdir+f"data/carleton_sm/econ_vars/{ssp.upper()}.nc4") 
    
    # Select high or low GDP model
    ssp = ssp[pop_group].sel(model="low") 
    
    # Convert to dataframe
    df = ssp.to_dataframe() 
    
    # Unstack
    df = df.drop(['ssp', 'model'], axis=1).unstack('year') 
    df.columns = df.columns.get_level_values(1)
    
    # Reset index and nerge
    df = df.reset_index() 
    df['hierid'] = df['region']
    df = df.set_index("hierid").reindex(ir.values)
    
    return df



def GenerateERFAll(wdir, temp_dir, scenario, ir, year, spatial_relation, age_groups, T, gammas, adaptation, 
                   gdppc_shares, image_gdppc, erfs_t0):
    
    """
    The code imports the gamma coefficients and the covariates (climtas and loggdppc) and feeds 
    this data into the function GenerateERFGroup that will generate the Exposure Response Functions 
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
    climtas, loggdppc = ImportCovariates(wdir, temp_dir, scenario, ir, year, spatial_relation, adaptation, 
                                         gdppc_shares, image_gdppc)

    # Create covariates matrix
    covariates = np.column_stack([np.ones(len(climtas)), climtas, loggdppc])

    mor_np = {}; tmin = {}        

    # Generate arrays with erf and tmin per age group
    for i, group in enumerate(age_groups):
        
        if erfs_t0 is not None:
            erfs_t0_group = erfs_t0[group]
            
        else:
            erfs_t0_group = None
        
        mor_np[group], tmin[group] = GenerateERFGroup(i, covariates, gamma_g, cov_g, T, erfs_t0_group)
        
    return mor_np, tmin, climtas, loggdppc



def ImportCovariates(wdir, temp_dir, scenario, ir, year, spatial_relation, adaptation, 
                     gdppc_shares, image_gdppc):
    
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
    if adaptation==None:
        
        # Open covariates for "present day" (no adaptation)
        covariates_t0 = pd.read_csv(wdir+"data/carleton_sm/main_specification/mortality-allpreds.csv")
        
        # Rename regions column to reindex woth ir dataframe
        covariates_t0 = covariates_t0.rename(columns={"region":"hierid"})
        covariates_t0 = covariates_t0.set_index("hierid").reindex(ir.values)
        
        # Extract only climtas and loggdppc as arrays
        climtas = covariates_t0["climtas"].values
        loggdppc = covariates_t0["loggdppc"].values
    
    # Adaptation ---------------------------------------------
    else:
        # CLIMTAS ---------------------------
        if adaptation.get("climtas") == "default":
            # Climate data from Carleton not available
            raise ValueError("climtas cannot be 'default'. Provide a directory or set 'tmean_t0'.")
        
        elif adaptation.get("climtas") == "tmean_t0":
            # Open climate data provided 
            temp_dir = temp_dir
            climtas = ImportPresentDayClimtas(temp_dir, spatial_relation, ir)
            
        else:
            # Open climate data provided 
            temp_dir = adaptation.get("climtas")
            climtas = ImportClimtas(temp_dir, year, spatial_relation, ir)

        # GDP -------------------------------
        if adaptation.get("loggdppc") == "default":
            # Open GDPpc provided by Carleton et al, at impact region level per SSP
            loggdppc = ImportLogGDPpc(wdir, scenario, ir, year)
        else:
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
    
    scenario = re.search(r"(?i)\bssp\d+", scenario).group()
        
    # Read GDP per capita file
    ssp_xarray = xr.open_dataset(wdir+f"data/carleton_sm/econ_vars/{scenario.upper()}.nc4")   
    
    # Caclulate mean of economic models (high and low projections) and 13 yr rolling mean
    gdppc = ssp_xarray.gdppc.mean(dim='model').rolling(year=13, min_periods=1).mean().sel(year=year)
    
    # Convert to dataframe for reindexing
    gdppc = gdppc.to_dataframe().reset_index()
    gdppc = gdppc.drop(columns=["year", "ssp"])
    gdppc = gdppc.rename(columns={"region":"hierid"})
    
    # Calculate log(GDPpc)
    gdppc["loggdppc"] = np.log(gdppc["gdppc"])
    
    # Reindex according to hierid
    gdppc = gdppc.set_index("hierid").reindex(ir.values)
    
    # Return numpy array
    return gdppc["loggdppc"].values



def ImportIMAGEloggdppc(year, image_gdppc, gdppc_shares):
    
    """
    Calculate log(GDPpc) at the impact region level using the GDPpc output
    from a TIMER run.

    Returns:
        _type_: _description_
    """
    
    # Extract relevant year data (13 year rolling mean)
    image_gdppc = (image_gdppc.sel(Time=slice(year-13,year))
                   .mean(dim="Time").mean(dim="Scenario")
                   .mean(dim="Variable")
                   .pint.dequantify() # Remove pint units and warning
                   .to_dataframe()
                   .reset_index())
    
    # Merge dataframes
    gdppc = gdppc_shares.merge(image_gdppc, left_on="IMAGE26", right_on="region", how="left")
    
    # Calculate share of log(GDPpc) based on regional GDPpc
    gdppc["gdppc"] = gdppc["Value"] * gdppc["gdppc_share"] * 1000 # TODO: Check units of TIMER 
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
    
    # Open GDP data (can be any SSP)
    GDPPC = xr.open_dataset(wdir+"/data/carleton_sm/econ_vars/SSP1.nc4")

    # Create coordinate for countries 
    GDPPC = GDPPC.assign_coords(ISO3=("region", GDPPC.region.str.slice(0, 3).data))

    # Calculate GDP share per country
    GDPPC['gdppc_ir_share'] = GDPPC['gdppc'].groupby("ISO3").map(lambda g: g / g.sum(dim="region"))

    # Calculate mean GDPpc per country over models and years 2010-2015 
    # (Carleton calculates shares from 2008 to 2012) but data is only available from 2010
    GDPPC = GDPPC.mean(dim="model").sel(year=range(2010,2016)).mean(dim="year")
    GDPPC_COUNTRY = GDPPC.groupby("ISO3").sum()

    # Create coordinate for IMAGE26 regions
    region_class = region_class.groupby(["ISO3","IMAGE26"]).first().reset_index()
    mapping = region_class.set_index("ISO3")["IMAGE26"]
    GDPPC_COUNTRY = GDPPC_COUNTRY.assign_coords(IMAGE26=("ISO3", mapping.reindex(GDPPC_COUNTRY.ISO3.values).values))

    # Calculate GDP share per IMAGE26 region
    GDPPC_COUNTRY['gdppc_country_share'] = GDPPC_COUNTRY['gdppc'].groupby("IMAGE26").map(lambda g: g / g.sum(dim="ISO3"))

    # Calculate final GDPpc share per impact region
    GDPPC["gdppc_share"] = GDPPC["gdppc_ir_share"] * GDPPC_COUNTRY["gdppc_country_share"].sel(ISO3=GDPPC["ISO3"])

    # Convert to dataframe
    GDPPC_DF = GDPPC.gdppc_share.to_dataframe().reset_index()
    
    # Ensure region alignment
    GDPPC_DF = GDPPC_DF.set_index("region").reindex(ir.values).reset_index()
    
    # Keep relevant columns
    GDPPC_DF = GDPPC_DF[['region', 'IMAGE26', 'gdppc_share']]    
    
    return GDPPC_DF



def ReadOUTFiles(gdp_dir, scenario):
    
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
    prism_regions = prism.Dimension('region', _DIM_IMAGE_REGIONS + ["Other"] + ["World"])
    prism_regions_world = prism.Dimension('region', _DIM_IMAGE_REGIONS + ["World"])
    
    listy = []

    path_clim = gdp_dir+f"/2_TIMER/outputlib/TIMER_3_4/{re.split(r"[\\/]", gdp_dir)[-1]}/"+scenario+"/indicators/Economy/"
    
    VAR = "GDPpc_incl_impacts"
    
    datafile = prism.TimeVariable(
            timeline=Timeline,
            dims=[prism_regions_world],
            file=path_clim+VAR+".OUT",
        )
    
    listy.append(xr.merge([datafile[i].rename('Value').expand_dims({"Time": [i]}) for i in np.arange(_DIM_TIME['start'], 2101)]).expand_dims({"Scenario": [scenario], "Variable": [VAR]}))
    
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
    
    TEMP_MEAN_30YEAR_PRESENT = TEMP_MEAN_30YEAR.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-01-01")).mean(dim="time")
    
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
    
    
    
def GenerateERFGroup(model_idx, X, gamma_g, cov_g, T, erfs_t0):
    
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
    erf, tmin_g = ShiftERFToTmin(erf, T, tas, tas2, tas3, tas4)
    
    #  # Ensure ERFs do not exceed no-adaptation ERFs 
    if erfs_t0 is not None:
        erf = np.minimum(erf, erfs_t0)
    
    # Impose weak monotonicity to the left and the right of the erf
    erf_final = MonotonicityERF(T, erf, tmin_g)

    return erf_final, tmin_g



def ShiftERFToTmin(raw, T, tas, tas2, tas3, tas4): 
    
    """   
    The code will apply the first constraint imposed by Carleton et al (see more in Appendix pp. A62).
    Following the paper, the minimum of the ERF is located between 20 and 30 degrees. Later the function
    is shifted vertucally to ensure the minimum matches null mortality. This procedure is done for all
    the ERF per age groups.
     
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
    
    # Locate idx of T (temperature array) between 20 and 30 degrees C
    idx_min_start = np.where(np.isclose(T, 10.0, atol=0.05))[0][0]
    idx_min_end   = np.where(np.isclose(T, 30.0, atol=0.05))[0][0]
    segment = raw[:, idx_min_start:idx_min_end]
    
    # Find local minimum of erf between 20 and 30 degrees
    idx_local_min = np.argmin(segment, axis=1)
    tmin_g = T[idx_min_start + idx_local_min]
    
    # Calcualte mortality value at tmin
    erf_at_tmin = tas*tmin_g + tas2*tmin_g**2 + tas3*tmin_g**3 + tas4*tmin_g**4
    
    # Shift vertical functions so tmin matches 0 deaths
    erf = raw - erf_at_tmin[:,None]
    
    return erf, tmin_g



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
    
    return erf_final     
    
    
        
def DailyTemperatureToIR(climate_path, year, ir, spatial_relation, scenario):
    
    """
    Convert daily temperature data of one year to impact region level.
    All grid cells intersecting an impact region are considered.
    Return a dataframe with mean daily temperature per impact region for the given year.
    
    Parameters:
    ----------
    climate_path : str
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
        DAILY_TEMPERATURE = ERA5Temperature2IR(climate_path, year, ir, spatial_relation)
        
    else:
        # Read daily temperature data generated from monthly statistics
        DAILY_TEMPERATURE,_ = tmp.DailyFromMonthlyTemperature(
            temp_dir=climate_path, 
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
    climate_path : str
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



def ERA5Temperature2IR(climate_path, year, ir, spatial_relation):
    
    """
    Import daily temperature data of one year from ERA5 and convert it
    to the impact region level.
    
    Parameters:
    ----------
    climate_path : str
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
        era5_dir=climate_path,
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
    (the conterfactual mortality called subtrahend here).
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
    DAILY_TEMP_T = DailyTemperatureToIR(temp_dir, year, res.ir, res.spatial_relation, scenario)
    
    print(f"[2.2] Calculating marginal mortality for year {year}...")
    
    # Calculate mortality per region and year (first term of equations 2' or 2a' from the paper)
    MORTALITY_ALL_MIN, MORTALITY_HEAT_MIN, MORTALITY_COLD_MIN, climtas, loggdppc =  CalculateMarginalMortality(wdir, temp_dir, year, scenario, DAILY_TEMP_T, 
                                                                                                               adaptation, res)
    
    print(f"[2.3] Calculating conterfactual mortality for year {year}...")

    # Calculate mortality per region and year (second term of equations 2' or 2a' from the paper)
    MORTALITY_ALL_SUB, MORTALITY_HEAT_SUB, MORTALITY_COLD_SUB, climtas_sub, loggdppc_sub  = CalculateMarginalMortality(wdir, temp_dir, year, scenario, res.daily_temp_t0, 
                                                                                                                      {"climtas": "tmean_t0", "loggdppc": adaptation.get("loggdppc")}, res)
    
    print("[2.4] Aggregating results to", regions, "regions and storing in results dataframe...")
    
    # Calculate mortality difference per impact region 
    for group in res.age_groups: 
        
        MORTALITY_ALL = MORTALITY_ALL_MIN[group] - MORTALITY_ALL_SUB[group]
        MORTALITY_HEAT = MORTALITY_HEAT_MIN[group] - MORTALITY_HEAT_SUB[group]
        MORTALITY_COLD = MORTALITY_COLD_MIN[group] - MORTALITY_COLD_SUB[group]
        
        # Aggregate results to selected region classification and store in results dataframe
        for mode, mor in zip(["All", "Heat", "Cold"], [MORTALITY_ALL_MIN[group], MORTALITY_HEAT_MIN[group], MORTALITY_COLD_MIN[group]]):
            Mortality2Regions(year, group, mor, regions, mode, res)  
            
            

def CalculateMarginalMortality(wdir, temp_dir, year, scenario, daily_temp, adaptation, res):
    
    """
    Calculate mortality effects from non optimal temperatures (first term of equation 2 from paper).
    Depending whether adaptation is on, the code will either import the ERFs with no adaptation or
    generate ERFs depending on the adaptation parameters.
    The daily temperature data will be converted to indices based on the range of T.
    Mortality per impact region will be calculated per age group and temperature type (all, heat and cold) 
    in the MortalityFromTemperatureIndex function.
    
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
    
    if adaptation:    
        ERFS_T, TMIN_T, climtas, loggdppc = GenerateERFAll(wdir, temp_dir, scenario, res.ir, year, 
                                          res.spatial_relation, res.age_groups, res.T, res.gammas, adaptation,
                                          res.gdppc_shares, res.image_gdppc, res.erfs_t0)
        
    else: 
        ERFS_T, TMIN_T = res.ERFS_T0, res.tmin_t0
    
    MORTALITY_ALL, MORTALITY_HEAT, MORTALITY_COLD = {}, {}, {}
    
    for group in res.age_groups:      
        MORTALITY_ALL[group], MORTALITY_HEAT[group], MORTALITY_COLD[group] = MortalityFromTemperatureIndex(DAILY_TEMP, TEMP_INDEX, ROWS, 
                                                                ERFS_T, TMIN_T, MIN_TEMP, group)
            
    return MORTALITY_ALL, MORTALITY_HEAT, MORTALITY_COLD, climtas, loggdppc 



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
            T_0_df = pd.read_csv(wdir+f"data/climate_data/ERA5_T0_{year}.csv")
            
            # Store in dictionary as numpy arrays
            T_0[year] = T_0_df.iloc[:,2:].to_numpy()
            
            
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

        T_0 = MSTemperature2IR(
            temp=DAILY_TEMPERATURE, 
            year=2000, 
            ir=ir, 
            spatial_relation=spatial_relation)
        
    # Convert "Present-day" temepratures dataframe to numpy array    
    T_0 = T_0.iloc[:,1:].to_numpy()
        
    return T_0

    

def MortalityFromTemperatureIndex(daily_temp, temp_idx, rows, erfs, tmin, min_temp, group):
    
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
    
    # # Calculate relative mortality for all temperatures using the temperature indices
    # result_all = erfs[group][rows, temp_idx] 
    
    # # Sum relative mortality across all days
    # result_all = result_all.sum(axis=1)

    # Extract tmin values for the given age group
    TMIN = tmin[group][:, None]

    # Generate temperature indices for temepratures over tmin and calculate mortality
    MASK_HEAT = np.where(daily_temp > TMIN, daily_temp, TMIN)
    # Convert temperatures to indices where TMIN is idx 0
    TEMP_HEAT_IDX = np.round((MASK_HEAT - min_temp) * 10).astype(int)
    # Calculate mortality from heat temperatures
    MORTALITY_HEAT = erfs[group][rows, TEMP_HEAT_IDX]
    # Sum mortality from heat across all days
    ANNUAL_MORTALITY_HEAT = MORTALITY_HEAT.sum(axis=1)
    
    # Generate temperature indices for temepratures below tmin and calculate mortality
    MASK_COLD = np.where(daily_temp < TMIN, daily_temp, TMIN)
    TEMP_COLD_IDX = np.round((MASK_COLD - min_temp) * 10).astype(int)
    MORTALITY_COLD = erfs[group][rows, TEMP_COLD_IDX]
    ANNUAL_MORTALITY_COLD = MORTALITY_COLD.sum(axis=1)
    
    # Sum heat and cold mortality to get all mortality
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
    REGIONS_CLASS = res.region_class[["hierid", regions]]
    
    # Calculate total mortality difference per region
    REGIONS_CLASS["mor"] = (mor * res.pop[group][f"{year}"] /1e5)
    
    # Add population column to calculate relative mortality
    REGIONS_CLASS["pop"] = res.pop[group][f"{year}"]
    
    # Group total mortality per selected region definition
    REGIONS_CLASS = REGIONS_CLASS.drop(columns=["hierid"]).groupby(regions).sum()
    
    # Calculate relative mortality per 100,000 people
    REGIONS_CLASS["rel_mor"] = REGIONS_CLASS["mor"] * 1e5 / REGIONS_CLASS["pop"]
    
    # Locate results in dataframe
    REGIONS_INDEX = res.results.loc[(group, mode, "Total deaths"), year].index[:-1]
    
    # Locating regional results in results dataframe
    res.results.loc[(group, mode, "Total deaths", REGIONS_INDEX), year] = (REGIONS_CLASS["mor"].reindex(REGIONS_INDEX)).values
    res.results.loc[(group, mode, "Deaths per 100,000", REGIONS_INDEX), year] = (REGIONS_CLASS["rel_mor"].reindex(REGIONS_INDEX)).values
    
    # Locate global results in results dataframe
    res.results.loc[(group, mode, "Total deaths", "World"), year] = REGIONS_CLASS["mor"].sum()
    res.results.loc[(group, mode, "Deaths per 100,000", "World"), year] = (REGIONS_CLASS["mor"].sum() * 1e5 / REGIONS_CLASS["pop"].sum())



def AddMortalityAllAges(results, pop, regions_class, years):
    
    # Prepare population data per IMAGE26 region
    pop["young"] = pop["young"].set_index("hierid")
    pop["older"] = pop["older"].set_index("hierid")
    pop["oldest"] = pop["oldest"].set_index("hierid")
    
    pop_all = pop["young"] + pop["older"] + pop["oldest"]
    pop_all = pop_all.iloc[:,1:-1]
    
    regions_class = regions_class.set_index("hierid")
    
    pop_all = pop_all.merge(regions_class, right_index=True, left_index=True)
    pop_all = pop_all.groupby("IMAGE26").sum().iloc[:,:-1]
    
    years_str = [str(y) for y in years]
    
    pop_all = pop_all.loc[:, pop_all.columns.isin(years_str)]
    
    pop_all.columns = pop_all.columns.astype(int)
    
    pop_all_world = pop_all.sum(axis=0)
    

    for mode in ["All", "Heat", "Cold"]:
        
        # Calculate total mortality for all age groups
        results.loc[("all", mode, "Total deaths")] = (
            results.loc[("young", mode, "Total deaths")] + 
            results.loc[("older", mode, "Total deaths")] + 
            results.loc[("oldest", mode, "Total deaths")]
        ).values
        
        # Calculate relative mortality for all-age group        
        IMAGE26 = results.loc[("all", mode, "Deaths per 100,000")].index[:-1]
        
        results.loc[("all", mode, "Deaths per 100,000", IMAGE26)] = (
            results.loc[("all", mode, "Total deaths", IMAGE26)]
            .mul(1e5)
            .div(pop_all.where(pop_all != 0))
        ).values
        
        # Calculate global relative mortality for all-age group
        results.loc[("all", mode, "Deaths per 100,000", "World")] = (
            results.loc[("all", mode, "Total deaths", "World")]
            * 1e5
            / pop_all_world.sum()
        )

    return results


def PostprocessResults(wdir, years, results, scenario, IAM_format, adaptation, pop, region_class):
    
    """
    Postprocess final results and save to CSV file in output folder.
    1. Substract the subtrahend dataframe from the minuend one (following equations 2' and 2a')
    2. If IAM format is on, change the format of the results to match the IAM one
    3. Save results in main working directory
    """
    
    print("[3] Postprocessing and saving results...")
    
    # Calculate total mortality and relative mortality for all-ages group
    results = AddMortalityAllAges(results, pop, region_class, years)
    
    # Reset index and format results for IAMs if specified
    if IAM_format==True:
        results = results.reset_index()
        results["Variable"] = ("Mortality|Non-optimal Temperatures|"
                               + results["t_type"].str.capitalize() 
                               + " Temperatures" 
                               + "|" 
                               + results["age_group"].str.capitalize() 
                               + "population"
                               + "|"
                               + results["units"])
        results = results[["IMAGE26", "Variable"] + list(results.columns[4:-1])]
    results = results.rename(columns={"IMAGE26": "region"})
    
    if adaptation:
        adapt = ""
        gdp_dir = adaptation.get("loggdppc")
        project = re.split(r"[\\/]", gdp_dir)[-1]+"_"
    else:
        adapt = "_noadap"
        project = ""
        
    # Save results to CSV                
    results.to_csv(wdir+f"output/mortality_carleton_{project}{scenario}{adapt}_{years[0]}-{years[-1]}_MIN.csv", 
                   index=False) 
    
    print("Scenario ran successfully!")