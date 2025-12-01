import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from dataclasses import dataclass
from shapely.geometry import Polygon
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils_common import temperature as tmp



### ------------------------------------------------------------------------------



def calculate_mortality(wdir, years, temp_source, temp_dir, SSP, regions, adaptation, IAM_format=False):
    
    """
    Main function to calculate mortality projections for the given parameters
    
    Parameters:
    wdir : str
        Path to main working directory
    years : list
        List of years to process
    climate_type : str
        Type of climate data ("ERA5", "CMIP6", "AR6")
    climate_path : str
        Path to climate data files
    SSP : STR
        List of socioeconomic scenarios (e.g., "SSP1", "SSP2")
    regions : str
        Region classification to use (e.g., "IMAGE26", "ISO3")
    IAM_format : bool, optional
        If True, output will be formatted as IAMs" output (default is False)
        
    Returns:
    None
    Saves the mortality results to CSV files in the output folder per 
    climate model and scenario.
    """
    
    if regions == "countries":
        regions = "gbd_level3"
    
    res = load_main_files(wdir, temp_dir, regions, SSP, years, temp_source, temp_dir, adaptation)  
        
    print("[2] Starting mortality calculations - Minuend part")
        
    # Iterate over years
    for year in years:
        
        # Read daily temperature data
        daily_temp = daily_temperature_to_ir(wdir, temp_dir, year, res.ir, res.spatial_relation, 
                                             temp_source)
        
        # Calculate mortality per region and year (minuend term of equations 2' or 2a' from the paper)
        mortality_effects_minuend(wdir, year, SSP, temp_dir, adaptation, daily_temp, regions, res)
    
    # Calculate mortality per region and year (subtrahend term of equations 2' or 2a' from the paper)
    mortality_effects_subtrahend(wdir, year, SSP, temp_dir, adaptation, regions, res)
     
    # Post process and save
    postprocess_results(wdir, years, res.results_minuend, res.results_subtrahend, SSP, IAM_format, adaptation)
    
    
    
@dataclass
class LoadResults:
    age_groups: list
    T : any
    spatial_relation: any
    ir: any
    region_class: any
    results_minuend: any
    results_subtrahend: any
    gammas: any
    pop: any
    climtas_t0: any
    loggdppc_t0 : any
    erfs_t0: any
    tmin_t0: any

    
    
def load_main_files(wdir, temp_dir, regions, SSP, years, climate_type, climate_path, adaptation):
    
    """
    Read and load all main input files required for mortality calculations
    Parameters:
    wdir : str
        Working directory
    regions : str
        Region classification to use (e.g., "IMAGE26", "ISO3")
    SSP : STR
        Socioeconomic scenarios (e.g., "SSP1", "SSP2")
    years : list
        List of years to process
    climate_type : str
        Type of climate data ("ERA5", "AR6")
    climate_path : str
        Path to climate data files
    Returns:
    LoadResults
    -------
    age_groups : list
        List of age groups
    spatial_relation : GeoDataFrame
        Spatial relationship between temperature grid cells and impact regions
    ir : GeoDataFrame
        GeoDataFrame with impact regions
    region_class : DataFrame
        DataFrame with region classification
    results : DataFrame
        DataFrame to store final results
    mor_np : dict
        Dictionary with mortality response functions per age group
    t_min : DataFrame
        DataFrame with optimal temperatures per impact region and age group
    climate_models : list
        List of climate models to process
    """
    
    print("[1] Loading main input files...")
    
    # Define age groups, keep order fixed
    age_groups = ["young", "older", "oldest"]
    
    # Define daily temperature range for Exposure Response Functions
    T = np.arange(-40, 60.1, 0.1).round(1)
    
    # Create relationship between temperature data and impact regions
    spatial_relation, ir = grid_relationship(wdir, climate_type, climate_path, years)
    
    # Open file with region classification
    region_class = select_regions(wdir, regions)

    # Create results dataframe
    results_minuend = final_dataframe(regions, region_class, age_groups, SSP, years)
    results_subtrahend = final_dataframe(regions, region_class, age_groups, SSP, range(2000,2011))
    
    # Import gamma coefficients
    gammas = import_gamma_coefficients(wdir)
    
    # Read population files
    pop = read_population_csv(wdir, SSP, years, age_groups)    
    
    # Import present day covariates
    print("[1.5] Import 'present day' covariates climtas and loggdppc.")
    climtas_t0, loggdppc_t0 = import_covariates(wdir, temp_dir, SSP, ir, None, spatial_relation, None)
    
    # If no adaptation, import once the 'present day'
    if adaptation == None:
        erfs_t0, tmin_t0 = generate_erf_all(wdir, temp_dir, SSP, ir, None, spatial_relation, 
                                            age_groups, T, adaptation) 
    
    # If adaptation, fill with no data
    else:
        erfs_t0, tmin_t0 = None, None
        
        
    return LoadResults(
        age_groups=age_groups,
        T = T,
        spatial_relation=spatial_relation,
        ir=ir,
        region_class=region_class,
        results_minuend=results_minuend,
        results_subtrahend = results_subtrahend,
        gammas = gammas,
        pop = pop,
        climtas_t0 = climtas_t0,
        loggdppc_t0 = loggdppc_t0,
        erfs_t0 = erfs_t0,
        tmin_t0 = tmin_t0
    )



def grid_relationship(wdir, temp_source, climate_path, years):
    
    """
    Create a spatial relationship between temperature data points from the nc files and impact regions
    
    Parameters:
    wdir : str
        Working directory
    climate_type : str
        Type of climate data ("ERA5", "AR6")
    climate_path : str
        Path to climate data files
    years : list
        List of years to process
        
    Returns:
    relationship : GeoDataFrame
        GeoDataFrame with spatial relationship between temperature grid cells and impact regions
    ir : GeoDataFrame
        GeoDataFrame with impact regions
    """
    
    print("[1.1] Creating spatial relationship between temperature grid and impact regions...")
    
    def create_square(lon, lat, lon_size, lat_size): 
        """
        Return a square Polygon centered at (lon, lat).
        Function only works for climate data with squared grids
        """
        return Polygon([
            (lon, lat),
            (lon + lon_size, lat),
            (lon + lon_size, lat + lat_size),
            (lon, lat + lat_size)
        ])

    # Read climate data
    if temp_source == "ERA5":
        temperature, _ = tmp.daily_temp_era5(climate_path, years[0], "mean", pop_ssp=None, to_array=False)
    elif temp_source == "MS":
        temperature, _ = tmp.daily_from_monthly_temp(climate_path, years[0], "MEAN", to_xarray=True)
    else:
        raise ValueError(f"Unsupported climate type: {temp_source}")
    
    # Extract coordinates
    coord_names = temperature.coords.keys()
    lon_vals = find_coord_vals(["lon", "longitude", "x"], coord_names, temperature)
    lat_vals = find_coord_vals(["lat", "latitude", "y"], coord_names, temperature)
    
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
            create_square(lon, lat, lon_size, lat_size)
            for lon, lat in zip(lon2d.ravel(), lat2d.ravel())
        ]
    })
    
    # Load .shp file with impact regions and set the same coordinate reference system (CRS)
    ir = gpd.read_file(wdir+"/data/carleton_sm/ir_shp/impact-region.shp")
    points_gdf = points_gdf.set_crs(ir.crs, allow_override=True)
    
    # Make spatial join
    relationship = gpd.sjoin(points_gdf, ir, how="inner", predicate="intersects")
    
    # Keep only necessary columns
    relationship = relationship[["geometry","index_right"]]

    return relationship, ir["hierid"]



def find_coord_vals(possible_names, coord_names, temperature):
    
    """
    Find the correct coordinate name in the dataset
    
    Parameters:
    possible_names : list
        List of possible coordinate names
    coord_names : list
        List of coordinate names in the dataset
    temperature : xarray DataArray
        Temperature data
        
    Returns:
    np.ndarray
        Coordinate values
    """
    
    for name in possible_names:
        if name in coord_names:
            return temperature[name].values
    raise KeyError(f"No coordinate was found among: {possible_names}")



def select_regions(wdir, regions):
    
    """
    Select region classification file based on user input
    """
    
    print(f"[1.2] Loading region classification: {regions}...")
    
    # Load region classification file
    region_class = pd.read_csv(f"{wdir}/data/regions/region_classification.csv")
    
    if regions == "impact_regions":
        region_class = region_class[["hierid"]]
    if regions == "countries":
        region_class = region_class[["hierid", "gbd_level3"]]
    else:
        region_class = region_class[["hierid", regions]]
    
    return region_class



def final_dataframe(regions, region_class, age_groups, SSP, years):
    
    """
    Create final results dataframe with multiindex for age groups, temperature types, and regions
    
    Parameters:
    regions : str
        Region classification to use (e.g., "IMAGE26", "ISO3")
    region_class : DataFrame
        DataFrame with region classification
    age_groups : list
        List of age groups
    SSP : list
        List of socioeconomic scenarios (e.g., ["SSP1", "SSP2"])
    years : list
        List of years to process
        
    Returns:
    results : DataFrame
        DataFrame to store final results
    """
    
    print("[1.3] Creating final results dataframe...")
    
    unique_regions = region_class[f"{regions}"].unique()
    unique_regions = unique_regions[~pd.isna(unique_regions)]
    
    t_types = ["Hot", "Cold", "All"]
    
    # Create results multiindex dataframe
    results = pd.DataFrame(index=pd.MultiIndex.from_product([age_groups, t_types, unique_regions],
                                                            names=["age_group", "t_type", regions]), 
                           columns=years)
    
    results.sort_index(inplace=True)
    
    return results 



def import_gamma_coefficients(wdir):    
    
    with open(wdir+"data/carleton_sm/Agespec_interaction_response.csvv") as f:
        for i, line in enumerate(f, start=1):

            if i == 21:
                # Extract 1, climtas, loggdppc
                covar_names = [x for x in line.strip().split(", ")]
                # Convert to indices
                covar_map = {"1":0, "climtas":1, "loggdppc":2}
                cov_idx = np.array([covar_map[str(x)] for x in covar_names])
                
            if i == 23:
                gammas = np.array([float(x) for x in line.strip().split(", ")])
                
    gamma_g = gammas.reshape(3,12)
    cov_g = cov_idx.reshape(3,12)
                
    return gamma_g, cov_g



def read_population_csv(wdir, SSP, years, age_groups):
    
    """
    Read Carleton et al. (2022) population CSV files for a given SSP scenario and age group.
    The files were created in the preprocessing step.
    
    Parameters:
    wdir : str
        Working directory
    SSP : str
        Socioeconomic scenario (e.g., "SSP1", "SSP2")
    years : list
        List of years to process
        
    Returns:
    pop_groups : dict
        Dictionary with population data per age group
    """
    
    print(f"[1.4] Importing Population data for {SSP} scenario...")
    
    pop_groups = {}
    
    for age_group in age_groups:
        
        # Read 'present-day' population data
        pop_present_day = pd.read_csv(f"{wdir}/data/gdp_pop_csv/POP_historical_{age_group}.csv")
        
        # Read population files projections per age group
        pop_projection = pd.read_csv(f"{wdir}/data/gdp_pop_csv/POP_{SSP}_{age_group}.csv")  
        
        # Merge 'present-day' population with relevant years of scenario projection
        cols = ["region"] + [str(y) for y in years if y >= 2023] 
        pop = pop_present_day.merge(pop_projection[cols], right_on="region", left_on="hierid", how="outer")
        
        pop_groups[age_group] = pop
    
    return pop_groups



def generate_erf_all(wdir, temp_dir, SSP, ir, year, spatial_relation, age_groups, T, adaptation):
    
    # Read coefficientes of covariates from Carleton SM
    gamma_g, cov_g = import_gamma_coefficients(wdir)
    
    # Import covariates with or without adaptation
    climtas, loggdppc = import_covariates(wdir, temp_dir, SSP, ir, year, spatial_relation, adaptation)

    # Create covariates matrix
    covariates = np.column_stack([np.ones(len(climtas)), climtas, loggdppc])
    
    mor_np = {}; tmin = {}        
    
    # Generate arrays with erf and tmin per age group
    for i, group in enumerate(age_groups):
        
        mor_np[group], tmin[group] = generate_erf_group(i, covariates, gamma_g, cov_g, T)
        
    return mor_np, tmin



def import_covariates(wdir, temp_dir, SSP, ir, year, spatial_relation, adaptation):
    
    """_summary_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    # No adaptation
    if adaptation==None:
        
        # Open covariates for "present day" (no adaptation)
        covariates_t0 = pd.read_csv(wdir+"data/carleton_sm/main_specification/mortality-allpreds.csv")
        
        # Rename regions column to reindex woth ir dataframe
        covariates_t0 = covariates_t0.rename(columns={"region":"hierid"})
        covariates_t0 = covariates_t0.set_index("hierid").reindex(ir.values)
        
        # Extract only climtas and loggdppc as arrays
        climtas = covariates_t0["climtas"].values
        loggdppc = covariates_t0["loggdppc"].values
    
    # Adaptation case
    else:
        # CLIMTAS ---------------------------
        if adaptation.get("climtas") == "default":
            # Climate data from Carleton not available
            raise ValueError("climtas cannot be 'default'. Provide a directory.")
        if adaptation.get("climtas") == "tmean_t0":
            pass
        else:
            # Open climate data provided 
            climtas = import_climtas(temp_dir, year, spatial_relation, ir)

        # GDP -------------------------------
        if adaptation.get("loggdppc") == "default":
            # Open GDPpc provided by Carleton et al, at impact region level per SSP
            loggdppc = import_loggdppc(wdir, SSP, ir, year)
        else:
            gdp_dir = adaptation.get("loggdppc")
            loggdppc = import_loggdppc_w_damages(wdir, gdp_dir, ir, year)
        
    return climtas, loggdppc



def import_loggdppc(wdir, SSP, ir, year):
    
    """
    Read GDP per capita files for a given SSP scenario.
    The files were created in the preprocessing step.
    
    Parameters:
    wdir : str
        Working directory
    SSP : str
        Socioeconomic scenario (e.g., "SSP1", "SSP2")
        
    Returns:
    gdppc : DataFrame
        DataFrame with GDP per capita data
    """
        
    # Read GDP per capita file
    ssp = xr.open_dataset(wdir+f"data/carleton_sm/econ_vars/{SSP.upper()}.nc4")   
    
    # Caclulate mean of economic models (high and low projections) and 13 yr rolling mean
    gdppc = ssp.gdppc.mean(dim='model').rolling(year=13, min_periods=1).mean().sel(year=year)
    
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



def import_loggdppc_w_damages(wdir, gdp_dir, ir, year):
    
    ### TODO: Adapt code to Mark's output files
    
    gdppc_column = ""
    
    # Generate GDPpc shares of regions within a country
    gdppc_shares = generate_gdppc_shares(wdir, ir)
    
    # Open gdp file 
    # TODO: Set correct file name
    gdppc_damages = pd.csv(gdp_dir+f"file_name.csv") 
    
    # Merge dataframes
    gdppc = gdppc_shares.merge(gdppc_damages, left_on="iso3", right_on=gdppc_column, how="outer")
    
    # Calculate log(GDPpc) of the 13 running GDPpc mean
    gdppc["gdppc13"] = gdppc[gdppc_column].rolling(window=13, min_periods=1).mean()
    gdppc["loggdppc"] = np.log(gdppc["gdppc13"])
    
    # Select relevant year
    gdppc = gdppc[gdppc["year"] == f"{year}"]
    
    return gdppc["loggdppc"].values



def generate_gdppc_shares(wdir, ir):
    
    # Open GDP data (can be any SSP)
    ssp = xr.open_dataset(wdir+"/data/carleton_sm/econ_vars/SSP1.nc4")
    
    # Create coordinate for countries 
    ssp = ssp.assign_coords(iso3=("region", ssp.region.str.slice(0, 3).data))
    
    # Calculate GDP share per country
    ssp['gdppc_share'] = ssp['gdppc'].groupby("iso3").map(lambda g: g / g.sum(dim="region"))
    
    # Convert to dataframe
    ssp = ssp.isel(year=0, model=0).gdppc_share.to_dataframe().reset_index()
    
    # Ensure region alignment
    ssp = ssp.set_index("region").reindex(ir.values).reset_index()
    
    # Keep relevant columns
    ssp = ssp[['region', 'iso3', 'gdppc_share']]
    
    return ssp



def import_climtas(temp_dir, year, spatial_relation, ir):
    
    """
    Import climate data from the specified directory and return 'climtas', efined by Carleton as the
    30-year running mean temperature per impact region.
    """
    
    # Read monthly mean of daily mean temperature data
    temp_mean = xr.open_dataset(temp_dir+f"GTMP_MEAN_30MIN.nc")
    
    # Calculate annual mean temperature and climatology
    temp_mean_annual = temp_mean["GTMP_MEAN_30MIN"].mean(dim="NM")
    
    # Calculate 30-year rolling mean temperature
    tmean = temp_mean_annual.rolling(time=30, min_periods=1).mean()
    
    # Assign pixels to every impact region
    temp_dict = {}
    climate_temp = tmean.sel(time=f"{year}").values.ravel()
    temp_dict[year] = climate_temp[spatial_relation.index]

    # Calculate mean temperature per impact region and round
    climtas = pd.DataFrame(temp_dict, index=spatial_relation["index_right"])
    climtas = climtas.groupby("index_right").mean()
    
    # Fill in nan with 20
    climtas = climtas.fillna(20)
    climtas.insert(0, "hierid", ir)
    climtas = climtas.rename(columns={year: "tmean", "hierid":"region"})
    
    return climtas["tmean"].values
    
    
    
def generate_erf_group(model_id, X, gamma_g, cov_g, T):
    
    # List of locations of gamma and covariates
    g = gamma_g[model_id]
    c = cov_g[model_id]

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
    erf, tmin_g = shift_erf_to_tmin(erf, T, tas, tas2, tas3, tas4)
    
    # Impose weak monotonicity to the left and the right of the erf
    erf_final = monotonicity_erf(T, erf, tmin_g)

    return erf_final, tmin_g



def shift_erf_to_tmin(raw, T, tas, tas2, tas3, tas4): 
    
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



def monotonicity_erf(T, erf, tmin_g):
    
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
    
    
        
def daily_temperature_to_ir(wdir, climate_path, year, ir, spatial_relation, temp_source):
    
    """
    Convert daily temperature data of one year to impact region level.
    All grid cells intersecting an impact region are considered.
    Return a dataframe with mean daily temperature per impact region for the given year.
    Parameters:
    wdir : str
        Working directory
    climate_path : str
        Path to climate data
    year : int
        Year of interest
    ir : GeoDataFrame
        GeoDataFrame with impact regions
    spatial_relation : GeoDataFrame
        Spatial relationship between temperature grid cells and impact regions
    Returns:
    df_rounded : DataFrame
        DataFrame with mean daily temperature per impact region for the given year
    """
    
    print("[2.1] Importing daily temperature data for year", year)
    
    if temp_source == "ERA5":
        
        day_temp = era5_temp_to_ir(climate_path, year, ir, spatial_relation)
        
    if temp_source == "MS":
        day_temp = ms_temp_to_ir(climate_path, year, ir, spatial_relation)
    
    # Convert dataframe to numpy array    
    day_temp = day_temp.iloc[:,1:].to_numpy()
    
    return day_temp



def ms_temp_to_ir(climate_path, year, ir, spatial_relation):
    
    # Read daily temperature data generated from monthly statistics
    temp_t2m, _ = tmp.daily_from_monthly_temp(climate_path, year, "MEAN", to_xarray=True)
    
    # Create a list of dates for the specified year
    date_list = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D").astype(str)
    
    # Temporarily store daily temperatures in a dictionary
    temp_dict = {}
    for day in date_list:
        daily_temperatures = temp_t2m.sel(valid_time=day).values.ravel()
        temp_dict[day] = daily_temperatures[spatial_relation.index]

    # Calculate mean temperature per impact region and round
    day_temp_df = pd.DataFrame(temp_dict, index=spatial_relation["index_right"])
    day_temp_df = day_temp_df.groupby("index_right").mean()
    
    # Fill in nan with 20
    day_temp_df = day_temp_df.fillna(20)

    # # Alternative nearest neighbor filling approach
    # ir = gpd.read_file(wdir+"/data/carleton_sm/ir_shp/impact-region.shp")
    # ir = ir.join(day_temp_df, how="right")
    
    # ir_valid = ir[ir.notna().any(axis=1)].copy()
    # ir_nan = ir[ir.isna().any(axis=1)].copy()
    # ir_filled = ir_nan.sjoin_nearest(ir_valid, how="left", distance_col="dist")
    
    day_temp_df_rounded = day_temp_df.round(1)
    day_temp_df_rounded.insert(0, "hierid", ir)
    
    return day_temp_df_rounded



def era5_temp_to_ir(climate_path, year, ir, spatial_relation):
    
    # Read ERA5 daily temperature data for a specific year
    temp_t2m, _ = tmp.daily_temp_era5(climate_path, year, "mean", pop_ssp=None, to_array=False)
    temp_t2m = temp_t2m.t2m
    
    # Select all available dates
    dates = temp_t2m["valid_time"].values
    
    # Create a list of dates for the specified year
    date_list = dates[np.isin(temp_t2m["valid_time"].values.astype("datetime64[Y]"),
                            np.datetime64(f"{year}", "Y"))].astype("datetime64[D]").astype(str)
    
    # Temporarily store daily temperatures in a dictionary
    temp_dict = {}
    for day in date_list:
        daily_temperatures = temp_t2m.sel(valid_time=day).values.ravel()
        temp_dict[day] = daily_temperatures[spatial_relation.index]
            
    # Calculate mean temperature per impact region and round
    day_temp_df = pd.DataFrame(temp_dict, index=spatial_relation["index_right"])
    day_temp_df = day_temp_df.groupby("index_right").mean()
    day_temp_df_rounded = day_temp_df.round(1)
    day_temp_df_rounded.insert(0, "hierid", ir)
    
    return day_temp_df_rounded
    
    

def mortality_effects_minuend(wdir, year, SSP, temp_dir, adaptation, daily_temp, regions, res):
    
    # Clip daily temperatures to the range of the ERFs
    min_temp = res.T[0]
    max_temp = res.T[-1]
    daily_temp = np.clip(daily_temp, min_temp, max_temp)

    # Convert ALL daily temperatures to temperature indices
    temp_idx =  np.round(((daily_temp - min_temp) * 10)).astype(int)
    
    # Create rows array for indexing
    rows = np.arange(temp_idx.shape[0])[:, None]
    
    print("[2.2] Generating Exposure Response Functions...")
    
    if adaptation:    
        erfs_t, tmin_t = generate_erf_all(wdir, temp_dir, SSP, res.ir, year, 
                                          res.spatial_relation, res.age_groups, res.T, adaptation)
        
    else: 
        erfs_t, tmin_t = res.erfs_t0, res.tmin_t0
    
    print(f"[2.3] Calculating mortality for year {year}...")
    
    for group in res.age_groups:      
        mor_all, mor_hot, mor_cold = mortality_from_temp_idx(daily_temp, temp_idx, rows, erfs_t, 
                                                                    tmin_t, min_temp, group)
            
        for mode, mor in zip(["All", "Hot", "Cold"],
                                    [mor_all, mor_hot, mor_cold]):
            
            # Calculate mortality difference per region and store in results dataframe
            mortality_to_regions(year, group, mor, regions, mode, res, 'minuend')               



def import_present_day_temperatures(wdir):
    
    # Definition for present day calculation
    base_years = range(2000,2011)
    
    T_0 = {}
    
    for year in base_years:
        # Read pre-calculated daily temperature at impact region level
        T_0_df = pd.read_csv(wdir+f"data/climate_data/ERA5_T0_{year}.csv")
        # Store in dictionary as numpy arrays
        T_0[year] = T_0_df.iloc[:,2:].to_numpy()
        
    return T_0  



def mortality_effects_subtrahend(wdir, year, SSP, temp_dir, adaptation, regions, res):
    
    print ("[3] Mortality calculations - Subtrahend part...")
    
    years = range(2000,2011)
    
    print("[3.1] Loading present-day temperature data...")
    
    # Import present day temperatures
    daily_temp_t0 = import_present_day_temperatures(wdir)
    
    # Clip daily temperatures to the range of the ERFs
    min_temp = res.T[0]
    max_temp = res.T[-1]
    daily_temp_t0 = {key: np.clip(arr, min_temp, max_temp) 
               for key, arr in daily_temp_t0.items()}

    # Convert ALL daily temperatures to temperature indices
    temp_idx_t0 = {
        key: np.round(((arr - min_temp) * 10)).astype(int)
        for key, arr in daily_temp_t0.items()
        } 
       
    # Create rows array for indexing
    rows = np.arange(temp_idx_t0[2000].shape[0])[:, None]
    
    print("[3.2] Generating Exposure Response Functions - Subtrahend part...")
    
    if adaptation:    
        erfs_t, tmin_t = generate_erf_all(wdir, temp_dir, SSP, res.ir, year, 
                                          res.spatial_relation, res.age_groups, res.T, 
                                          {"tmean": "tmean_t0", "loggdppc": adaptation.get("loggdppc")} )
        
    else: 
        erfs_t, tmin_t = res.erfs_t0, res.tmin_t0
        
    print("[3.3] Calculating present-day mortality...")
    
    for year in years:
        
        daily_temp = daily_temp_t0[year]
        temp_idx = temp_idx_t0[year]
        
        for group in res.age_groups:      
            mor_all, mor_hot, mor_cold = mortality_from_temp_idx(daily_temp, temp_idx, rows, 
                                                                 erfs_t, tmin_t, min_temp, group)
            
            for mode, mor in zip(["All", "Hot", "Cold"],
                                    [mor_all, mor_hot, mor_cold]):
            
                # Calculate mortality difference per region and store in results dataframe
                mortality_to_regions(year, group, mor, regions, mode, res, 'subtrahend')  
    
    

def mortality_from_temp_idx(daily_temp, temp_idx, rows, erfs, tmin, min_temp, group):
    
    """
    _summary_

    Returns:
        _type_: _description_
    """
    
    # Calculate relative mortality for all temperatures using the temperature indices
    result_all = erfs[group][rows, temp_idx] 
    # Sum relative mortality across all days
    result_all = result_all.sum(axis=1)

    # Extract tmin values for the given age group
    tmin = tmin[group][:, None]

    # Generate temperature indices for temepratures over tmin and calculate mortality
    temp_heat_idx = heat_and_cold_temp_index(daily_temp, tmin, "hot", min_temp)
    result_heat = erfs[group][rows, temp_heat_idx]
    result_heat = result_heat.sum(axis=1)
    
    # Generate temperature indices for temepratures below tmin and calculate mortality
    temp_cold_idx = heat_and_cold_temp_index(daily_temp, tmin, "cold", min_temp) 
    result_cold = erfs[group][rows, temp_cold_idx]
    result_cold = result_cold.sum(axis=1)
    
    return result_all, result_heat, result_cold        



def heat_and_cold_temp_index(temp_matrix, threshold, condition, min_temperature):
    
    """
    Mask temperatures based on the tmin threshold, filling others with threshold value
    This ensures that temepratures that do not meet the condition will result in the minimum
    temperature and the corresponding 0 mortality from the ERF (exposure response function).
    """
    
    # Mas temperatures for heat and cold effects
    if condition == "hot":
        masked = np.where(temp_matrix > threshold, temp_matrix, threshold)
        
    elif condition == "cold":
        masked = np.where(temp_matrix < threshold, temp_matrix, threshold)
    
    # Convert masked temperatures to temperature indices
    idx = np.round((masked - min_temperature) * 10).astype(int)
    
    return idx



def mortality_to_regions(year, group, mor, regions, mode, res, substraction):
    
    """
    
    """
    
    if substraction == "minuend":
        results = res.results_minuend
    else:
        results = res.results_subtrahend
    
    # Create a copy of region classification dataframe
    regions_df = res.region_class.copy()
    
    # Calculate total mortality difference per region
    regions_df["mor"] = (mor * res.pop[group][f"{year}"] /1e5)
    
    # Group total mortality per selected region definition
    regions_df = regions_df.drop(columns=["hierid"]).groupby(regions).sum()
    
    # Locate results in dataframe
    regions_index = results.loc[(group, mode), year].index
    results.loc[(group, mode), year] = (regions_df["mor"].reindex(regions_index)).values
    


def postprocess_results(wdir, years, results_minuend, results_subtrahend, SSP, IAM_format, adaptation):
    
    """
    Postprocess final results and save to CSV file in output folder.
    """
    
    results_subtrahend = results_subtrahend.mean(axis=1)
    
    results = results_minuend.subtract(results_subtrahend, axis=0)
    
    print("[4] Postprocessing and saving results...")
    
    # Reset index and format results for IAMs if specified
    if IAM_format==True:
        results = results.reset_index()
        results["Variable"] = ("Mortality|Non-optimal Temperatures|"
                               + results["t_type"].str.capitalize() 
                               + " Temperatures" 
                               + "|" 
                               + results["age_group"].str.capitalize())
        results = results[["Scenario", "IMAGE26", "Variable"] + list(results.columns[4:-1])]
    results = results.rename(columns={"IMAGE26": "Region"})
    
    if adaptation:
        adapt = "_adaptation"
    else:
        adapt = ""
        
    # Save results to CSV              
    # results.to_csv(f"{wdir}/output/carleton_mortality_{regions}_{climate_type}_{years[0]}-{years[-1]}.csv")   
    results.to_csv(wdir+f"output/carleton_mortality_{SSP}{adapt}_{years[0]}-{years[-1]}.csv") 