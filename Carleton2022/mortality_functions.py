import numba as nb
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from dataclasses import dataclass
from shapely.geometry import Polygon
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common_utils import temperature as tmp
from common_utils import population as pop



### ------------------------------------------------------------------------------
### ----------------------------- PARAMETERS -------------------------------------


'''
Gamma coeficients of the covariants for the temperature response function of the 
three age groups (young, older, oldest) from Carleton et al. (2022).
'''

gamma_np = {
    'young': np.array([
        [-0.2643747697030857, -0.0012157807919976, 0.0285121426008164],
        [-0.0147654905557389, -0.0001292299812386, 0.0013467700198057],
        [0.0000555941144027,  0.000010228738298,  -0.0000128604018705],
        [0.0000188858412856, -2.48887855043e-07,  -1.50547526657e-06]
    ]),
    'older': np.array([
        [0.2478292444689566,  0.0022092761549115, -0.0258890110895998],
        [-0.0125437290633759, 0.0000123113770044, 0.0012019083245803],
        [-0.0002220037659301, -2.82565977452e-06, 0.0000227328454772],
        [0.0000129910024803,  1.82855114488e-08,  -1.21751952067e-06]
    ]),
    'oldest': np.array([
        [6.399027562773568,   0.0436967573579832, -0.6751842737945384],
        [-0.3221434191389331, 0.0013726982372035, 0.0295628065147365],
        [-0.0044299345528043, -0.0001067884304388, 0.00050851740502],
        [0.0002888631905257,  9.32783835571e-07,  -0.0000273410162051]
    ])
}



### ------------------------------------------------------------------------------
### ------------------------------ FUNCTIONS -------------------------------------



def postprocess_results(wdir, years, results, climate_type, climate_model, scenario_SSP, scenario_RCP, IAM_format, regions):
    
    '''
    Postprocess final results and save to CSV file in output folder.
    '''
    
    print('[4] Postprocessing and saving results...')
    
    # Reset index and format results for IAMs if specified
    if IAM_format==True:
        results = results.reset_index()
        results['Variable'] = ('Mortality|Non-optimal Temperatures|'
                               + results['t_type'].str.capitalize() 
                               + ' Temperatures' 
                               + '|' 
                               + results['age_group'].str.capitalize())
        results = results[['Scenario', 'IMAGE26', 'Variable'] + list(results.columns[4:-1])]
    results = results.rename(columns={'IMAGE26': 'Region'})

    # Save results to CSV              
    if climate_type == 'ERA5':
        results.to_csv(f'{wdir}/output/mortality_{climate_type}_{regions}_{years[0]}-{years[-1]}.csv')    
    
    

@nb.njit
def temp_to_column_location(temp):
    
    '''
    Get the column index in the mortality response function corresponding to a given temperature
    '''
    
    min_temperature = -50.0
    temp = np.round(temp, 1)
    return int(np.round(((temp - min_temperature) * 10)))



@nb.njit
def mortality_from_erf(temperature_data, mortality_data, mode, tmin_column):
    
    """
    Compute mortality based on temperature data and the mortality response function.
    
    Parameters
    ----------
    temperature_data : np.ndarray
        Array containing daily or regional temperature values.
    mortality_data : np.ndarray
        2D array representing the mortality response function 
        (rows = regions, columns = temperature bins).
    tmin_column : np.ndarray, optional
        Reference minimum (optimal) temperature for each region. 
        Used only for 'hot' and 'cold' modes.
    mode : str, optional
        Determines which temperature range to consider:
            'all'  -> use all temperatures
            'hot'  -> only temperatures above tmin_column
            'cold' -> only temperatures below tmin_column
    """
    
    n = len(temperature_data)
    result = np.empty(n, dtype=np.float64)
    
    if mode == 'all':
        for i in range(n):
            temp = temperature_data[i]
            mort_value = mortality_data[i, temp_to_column_location(temp)]        
            result[i] = mort_value
            
    if mode == 'hot':
        for i in range(n):
            temp = temperature_data[i]
            mort_value = mortality_data[i, temp_to_column_location(temp)]  
            result[i] = mort_value if temp > tmin_column[i] else np.nan
            
    if mode == 'cold':
        for i in range(n):
            temp = temperature_data[i]
            mort_value = mortality_data[i, temp_to_column_location(temp)]  
            result[i] = mort_value if temp < tmin_column[i] else np.nan

    return result



def calculate_mortality_year(daily_temperature, mortality_np, t_min, mode):
    
    '''
    Calculate yearly mortality for a given SSP scenario, mortality response function
    and temperature data.
    
    Parameters:
    daily_temperature : DataFrame
        DataFrame with daily temperature per impact region for the given year
    mortality_np : np.ndarray
        Mortality response function for the given age group
    t_min : np.ndarray
        Numpy array with optimal temperatures per impact region
    mode : str
        Determines which temperature range to consider:
            'all'  -> use all temperatures
            'hot'  -> only temperatures above tmin_column
            'cold' -> only temperatures below tmin_column
            
    Returns:
    mortality : DataFrame
        DataFrame with daily and annual mortality per impact region
    '''
    
    temperature_other_columns = 1  
    # Calculate mortality for each day and sum to get total mortality
    mortality = daily_temperature.iloc[:, :temperature_other_columns].copy()
    
    mortality = pd.concat([
        mortality,
        pd.concat([
            pd.Series(mortality_from_erf(temperature_series.to_numpy(), mortality_np, mode, t_min), 
                        name=day+"_mortality")
            for day, temperature_series in daily_temperature.iloc[:, temperature_other_columns:].items()
        ], axis=1)
    ], axis=1)

    # Sum daily mortalities to get total annual mortality per region
    mortality[f'annual_mortality_{mode}'] = mortality.filter(like='_mortality').sum(axis=1)
    
    return mortality
          
          
          
def annual_regional_mortality(results, daily_temperature, scenario_SSP, age_group, mor_np, pop_file, year, t_min, regions, region_class):
    
    '''
    Calculate yearly mortality for a given SSP scenario, age group, mortality response function, population file, and year
    
    Parameters:
    results : DataFrame
        DataFrame to store final results
    daily_temperature : DataFrame
        DataFrame with daily temperature per impact region for the given year
    scenario_SSP : str
        Socioeconomic scenario (e.g., 'SSP1', 'SSP2')
    age_group : str
        Age group (e.g., 'oldest', 'older', 'young')
    mor_np : np.ndarray
        Mortality response function for the given age group
    pop_file : DataFrame
        Population data for the given SSP scenario and age group
    year : int  
        Year of interest
    t_min : DataFrame
        DataFrame with optimal temperatures per impact region and age group
    regions : str
        Region classification to use (e.g., 'IMAGE26', 'ISO3')
    region_class : DataFrame    
        DataFrame with region classification
        
    Returns:
    None
    Saves the mortality results to the results DataFrame
    '''
    
    t_min_age = np.array(t_min[f'Tmin {age_group}']) 
     
    # Calculate relative mortality per region (deaths/100,000)
    mor_temp = calculate_mortality_year(daily_temperature, mor_np, t_min_age, mode='all')
    mor_temp_hot = calculate_mortality_year(daily_temperature, mor_np, t_min_age, mode='hot') 
    mor_temp_cold = calculate_mortality_year(daily_temperature, mor_np, t_min_age, mode='cold')
    
    # Concat mortality with population and region classification
    df = pd.concat([region_class, 
                    pop_file[age_group][f'{year}'], 
                    mor_temp['annual_mortality_all'],
                    mor_temp_hot['annual_mortality_hot'],  
                    mor_temp_cold['annual_mortality_cold']], axis=1)

    # Calculate total mortality
    df['mortality_all'] = df[f'{year}'] * df['annual_mortality_all'] /1e5
    df['mortality_cold'] = df[f'{year}'] * df['annual_mortality_cold'] /1e5
    df['mortality_hot'] = df[f'{year}'] * df['annual_mortality_hot'] /1e5
    
    # Group total mortality per selected region definition
    df = df.drop(columns=['hierid']).groupby(regions).sum()
    regions_index = results.loc[(scenario_SSP, age_group, 'All'), year].index
      
    # Locate results in dataframe
    results.loc[(scenario_SSP, age_group, 'All'), year] = (df['mortality_all'].reindex(regions_index)).values
    results.loc[(scenario_SSP, age_group, 'Hot'), year] = (df['mortality_hot'].reindex(regions_index)).values
    results.loc[(scenario_SSP, age_group, 'Cold'), year] = (df['mortality_cold'].reindex(regions_index)).values
    
    

def read_population_csv(wdir, SSP, years):
    
    '''
    Read Carleton et al. (2022) population CSV files for a given SSP scenario and age group.
    The files were created in the preprocessing step.
    
    Parameters:
    wdir : str
        Working directory
    SSP : str
        Socioeconomic scenario (e.g., 'SSP1', 'SSP2')
    years : list
        List of years to process
        
    Returns:
    pop_groups : dict
        Dictionary with population data per age group
    '''
    
    print(f'[2.1] Importing Population data for {SSP} scenario...')
    
    pop_groups = {}
    
    for age_group in ['oldest', 'older', 'young']:
    
        # For historical years, use historical population data
        if all(y in range(2000, 2022) for y in years):
            SSP = 'historical'
        
        # Read population files projections per age group
        POP = pd.read_csv(f'{wdir}/data/gdp_pop_csv/POP_{SSP}_{age_group}.csv')  

        # If years include both historical and future years, merge both files
        if years[0]<2010 and years[-1]>2022:
            
            # Read historical population file and merge with projections
            pop_historical = pd.read_csv(f'{wdir}/data/gdp_pop_csv/POP_historical_{age_group}.csv')
            cols_to_add = pop_historical.columns.difference(POP.columns)
            left = POP.iloc[:, :2] 
            right = POP.iloc[:, 2:]  
            POP = pd.concat([left, pop_historical[cols_to_add], right], axis=1)
            
        # Add population data to dictionary
        pop_groups[age_group] = POP
    
    return pop_groups
    
    
    
def era5_temperature_to_ir(climate_path, year, ir, spatial_relation):
    
    '''
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
    '''
    
    # Read ERA5 daily temperature data for a specific year
    era5_t2m, _ = tmp.daily_temp_era5(climate_path, year, 'mean', pop_ssp=None, to_array=False)
    era5_t2m = era5_t2m.t2m
    
    # Select all available dates
    dates = era5_t2m['valid_time'].values
    
    # Create a list of dates for the specified year
    date_list = dates[np.isin(era5_t2m['valid_time'].values.astype('datetime64[Y]'),
                              np.datetime64(f'{year}', 'Y'))].astype('datetime64[D]').astype(str)
    
    # Temporarily store daily temperatures in a dictionary
    temp_dict = {}
    for day in date_list:
        daily_temperatures = era5_t2m.sel(valid_time=day).values.ravel()
        temp_dict[day] = daily_temperatures[spatial_relation.index]
            
    # Calculate mean temperature per impact region and round
    day_temp_df = pd.DataFrame(temp_dict, index=spatial_relation['index_right'])
    day_temp_df = day_temp_df.groupby('index_right').mean()
    day_temp_df_rounded = day_temp_df.round(1)
    day_temp_df_rounded.insert(0, 'hierid', ir).index.values
    
    print(f'ERA5 daily temperature for {year} imported')
    
    return day_temp_df_rounded
    
    

def get_daily_temperatures(wdir, climate_type, climate_path, climate_model, scenario_RCP, year, spatial_relation, ir):
    
    print('[3.1] Importing daily temperature data for year', year)
    
    if climate_type == 'ERA5':
        # Read and convert temperature data to impact region level
        daily_temp = era5_temperature_to_ir(climate_path, year, ir, spatial_relation)
        
    return daily_temp



def read_mortality_response(wdir, group):
    
    '''
    Read mortality response function for a given age group
    '''
    
    # Read mortality response function csv file created in the preprocessing step
    mor = pd.read_csv(wdir + f'/data/exposure_response_functions/erf_no-adapt_{group}.csv')
    
    # Convert columns to float type and extract mortality values as numpy array
    columns = list(mor.columns)
    num_other_columns = 2
    mor.columns = columns[:num_other_columns] + list(np.array(columns[num_other_columns:], dtype="float"))
    mor_np = mor.iloc[:, num_other_columns:].round(2).to_numpy()
    
    return mor_np

    
    
def import_erfs(wdir):
    
    '''
    Read and import exposure response functions for the three age groups and Tmin.
    The csv files were created in the preprocessing step.
    
    Parameters:
    wdir : str
        Working directory
        
    Returns:
    mor_np : dict
        Dictionary with mortality response functions per age group
    t_min : DataFrame
        DataFrame with optimal temperatures per impact region and age group
    '''
    
    print('[1.4] Importing Exposure Response Functions for the three age groups...')
    
   # Read mortality response functions per age group
    mor_oldest = read_mortality_response(wdir, 'oldest')
    mor_older = read_mortality_response(wdir, 'older')
    mor_young = read_mortality_response(wdir, 'young')
    
    # Store mortality response functions in a dictionary
    mor_np = {'oldest': mor_oldest, 'older': mor_older, 'young': mor_young}
    
    # Open file with optimal temperature (Tmin) per impact region
    t_min = pd.read_csv(f'{wdir}/data/exposure_response_functions/T_min.csv')
    
    return mor_np, t_min


    
def final_dataframe(regions, region_class, age_groups, scenarios_SSP, years):
    
    '''
    Create final results dataframe with multiindex for scenarios, age groups, temperature types, and regions
    
    Parameters:
    regions : str
        Region classification to use (e.g., 'IMAGE26', 'ISO3')
    region_class : DataFrame
        DataFrame with region classification
    age_groups : list
        List of age groups
    scenarios_SSP : list
        List of socioeconomic scenarios (e.g., ['SSP1', 'SSP2'])
    years : list
        List of years to process
        
    Returns:
    results : DataFrame
        DataFrame to store final results
    '''
    
    print('[1.3] Creating final results dataframe...')
    
    unique_regions = region_class[f'{regions}'].unique()
    unique_regions = unique_regions[~pd.isna(unique_regions)]
    
    t_types = ['Hot', 'Cold', 'All']
    
    # Create results multiindex dataframe
    results = pd.DataFrame(index=pd.MultiIndex.from_product([scenarios_SSP, age_groups, t_types, unique_regions],
                                                            names=['Scenario', 'age_group', 't_type', regions]), 
                           columns=years)
    
    results.sort_index(inplace=True)
    
    return results 



def select_regions(wdir, regions):
    
    '''
    Select region classification file based on user input
    '''
    
    print(f'[1.2] Loading region classification: {regions}...')
    
    # Load region classification file
    region_class = pd.read_csv(f'{wdir}/data/regions/region_classification.csv')
    
    if regions == 'impact_regions':
        region_class = region_class[['hierid']]
    else:
        region_class = region_class[['hierid', regions]]
    
    return region_class



def find_coord_vals(possible_names, coord_names, temperature):
    
    '''
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
    '''
    
    for name in possible_names:
        if name in coord_names:
            return temperature[name].values
    raise KeyError(f"No coordinate was found among: {possible_names}")



def grid_relationship(wdir, climate_type, climate_path, years):
    
    '''
    Create a spatial relationship between temperature data points from the nc files and impact regions
    
    Parameters:
    wdir : str
        Working directory
    climate_type : str
        Type of climate data ('ERA5', 'AR6')
    climate_path : str
        Path to climate data files
    years : list
        List of years to process
        
    Returns:
    relationship : GeoDataFrame
        GeoDataFrame with spatial relationship between temperature grid cells and impact regions
    ir : GeoDataFrame
        GeoDataFrame with impact regions
    '''
    
    print('[1.1] Creating spatial relationship between temperature grid and impact regions...')
    
    def create_square(lon, lat, lon_size, lat_size): 
        '''
        Return a square Polygon centered at (lon, lat).
        Function only works for climate data with squared grids
        '''
        return Polygon([
            (lon, lat),
            (lon + lon_size, lat),
            (lon + lon_size, lat + lat_size),
            (lon, lat + lat_size)
        ])

    # Read climate data
    if climate_type == 'ERA5':
        temperature, _ = tmp.daily_temp_era5(climate_path, years[0], 'mean', pop_ssp=None, to_array=False)
    else:
        raise ValueError(f"Unsupported climate type: {climate_type}")
    
    # Extract coordinates
    coord_names = temperature.coords.keys()
    lon_vals = find_coord_vals(["lon", "longitude", "x"], coord_names, temperature)
    lat_vals = find_coord_vals(["lat", "latitude", "y"], coord_names, temperature)
    
    # Calculate grid spacing
    lon_size = np.abs(np.mean(np.diff(lon_vals)))
    lat_size = np.abs(np.mean(np.diff(lat_vals)))    

    # Converting lon to right -180 to 180 degrees if necessary
    lon_vals = np.where(lon_vals > 180, lon_vals - 360, lon_vals)  
    
    # Create meshgrid 
    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)  
    
    # Create GeoDataFrame with points and their corresponding square polygons
    points_gdf = gpd.GeoDataFrame({
        'longitude': lon2d.ravel(),
        'latitude': lat2d.ravel(),
        'geometry': [
            create_square(lon, lat, lon_size, lat_size)
            for lon, lat in zip(lon2d.ravel(), lat2d.ravel())
        ]
    })
    
    # Load .shp file with impact regions and set the same coordinate reference system (CRS)
    ir = gpd.read_file(wdir+'/data/carleton_sm/ir_shp/impact-region.shp')
    points_gdf = points_gdf.set_crs(ir.crs, allow_override=True)
    
    # Make spatial join
    relationship = gpd.sjoin(points_gdf, ir, how='inner', predicate='intersects')
    
    # Keep only necessary columns
    relationship = relationship[['geometry','index_right']]
    
    print('Spatial relationship (temperature grid to impact region level) created')

    return relationship, ir['hierid']



@dataclass
class LoadResults:
    age_groups: list
    spatial_relation: any
    ir: any
    region_class: any
    results: any
    mor_np: dict
    t_min: any
    climate_models: list
    scenarios_RCP: list
    

    
def load_main_files(wdir, regions, scenarios_SSP, years, climate_type, climate_path, scenarios_RCP):
    
    '''
    Read and load all main input files required for mortality calculations
    Parameters:
    wdir : str
        Working directory
    regions : str
        Region classification to use (e.g., 'IMAGE26', 'ISO3')
    scenarios_SSP : list
        List of socioeconomic scenarios (e.g., ['SSP1', 'SSP2'])
    years : list
        List of years to process
    climate_type : str
        Type of climate data ('ERA5', 'AR6')
    climate_path : str
        Path to climate data files
    scenarios_RCP : list
        List of climate scenarios (e.g., ['SSP126', 'SSP245'])
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
    scenarios_RCP : list
        List of climate scenarios to process
    '''
    
    print('[1] Loading main input files...')
    
    # Define age groups
    age_groups = ['oldest', 'older', 'young']
    
    # Create relationship between temperature data and impact regions
    spatial_relation, ir = grid_relationship(wdir, climate_type, climate_path, years)
    
    # Open file with region classification
    region_class = select_regions(wdir, regions)

    # Create results dataframe
    results = final_dataframe(regions, region_class, age_groups, scenarios_SSP, years)
    
    # Read erfs
    mor_np, t_min = import_erfs(wdir)
        
    if climate_type == 'ERA5':
        climate_models = ['']
        scenarios_RCP = ['']
        
    # elif climate_type == 'AR6':
    #     climate_models = climate_models_ar6.keys()
    #     scenarios_RCP = ['']

    return LoadResults(
        age_groups=age_groups,
        spatial_relation=spatial_relation,
        ir=ir,
        region_class=region_class,
        results=results,
        mor_np=mor_np,
        t_min=t_min,
        climate_models=climate_models,
        scenarios_RCP=scenarios_RCP
    )



def calculate_mortality(wdir, years, climate_type, climate_path, scenarios_SSP, scenarios_RCP, regions, IAM_format=False):
    
    '''
    Main function to calculate mortality projections for the given parameters
    
    Parameters:
    wdir : str
        Path to main working directory
    years : list
        List of years to process
    climate_type : str
        Type of climate data ('ERA5', 'CMIP6', 'AR6')
    climate_path : str
        Path to climate data files
    scenarios_SSP : list
        List of socioeconomic scenarios (e.g., ['SSP1', 'SSP2'])
    scenarios_RCP : list
        List of climate scenarios (e.g., ['SSP126', 'SSP245'])
    regions : str
        Region classification to use (e.g., 'IMAGE26', 'ISO3')
    IAM_format : bool, optional
        If True, output will be formatted as IAMs' output (default is False)
        
    Returns:
    None
    Saves the mortality results to CSV files in the output folder per 
    climate model and scenario.
    '''
    
    res = load_main_files(wdir, regions, scenarios_SSP, years, climate_type, climate_path, scenarios_RCP)    
        
    print('[2] Starting mortality calculations')
        
    # Iterate over climate models or use single ERA5/AR6 data
    for climate_model in res.climate_models:   
             
        # Iterate over population SSP scenarios
        for scenario_SSP in scenarios_SSP:
            
            # Read population files
            pop = read_population_csv(wdir, scenario_SSP, years)
            
            # Iterate over RCP scenarios
            for scenario_RCP in res.scenarios_RCP:     
                
                # Iterate over years
                for year in years:
                    
                    print(f'[3] Processing year {year}...')
                    
                    daily_temp = get_daily_temperatures(wdir, climate_type, climate_path, 
                                                    climate_model, scenario_RCP, year, 
                                                    res.spatial_relation, res.ir)
                    
                    print('[3.2] Calculating mortality per age group...')
                    
                    for age_group in res.age_groups:         

                        annual_regional_mortality(res.results, daily_temp, scenario_SSP, 
                                                age_group, res.mor_np[age_group], pop, year, 
                                                res.t_min, regions, res.region_class)
                    
            postprocess_results(wdir, years, res.results, climate_type, climate_model, 
                                scenario_SSP, scenario_RCP, IAM_format, regions)