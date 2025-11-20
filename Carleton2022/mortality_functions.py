import numba as nb
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from dataclasses import dataclass
from shapely.geometry import Polygon
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_common import temperature as tmp
from utils_common import population as pop



### ------------------------------------------------------------------------------


def postprocess_results(wdir, years, results, climate_type, SSP, IAM_format, regions):
    
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
    results.to_csv(f'{wdir}/output/carleton_mortality_{regions}_{climate_type}_{years[0]}-{years[-1]}.csv')    
    
    

@nb.njit
def temp_to_column_location(temp):
    
    '''
    Get the column index in the mortality response function corresponding to a given temperature
    '''
    
    min_temperature = -50.0
    temp = np.round(temp, 1)
    return int(np.round(((temp - min_temperature) * 10)))



@nb.njit
def mortality_from_erf(temperature_data, mortality_data, tmin_column, mode):
    
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
            pd.Series(mortality_from_erf(temperature_series.to_numpy(), mortality_np, t_min, mode), 
                        name=day+"_mortality")
            for day, temperature_series in daily_temperature.iloc[:, temperature_other_columns:].items()
        ], axis=1)
    ], axis=1)

    # Sum daily mortalities to get total annual mortality per region
    mortality[f'annual_mortality_{mode}'] = mortality.filter(like='_mortality').sum(axis=1)
    
    return mortality
          
          
          
def annual_regional_mortality(results, daily_temperature, SSP, age_group, mor_np, pop_file, year, t_min, regions, region_class):
    
    '''
    Calculate yearly mortality for a given SSP scenario, age group, mortality response function, population file, and year
    
    Parameters:
    results : DataFrame
        DataFrame to store final results
    daily_temperature : DataFrame
        DataFrame with daily temperature per impact region for the given year
    SSP : str
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
    
    if regions == 'countries':
        regions = 'gbd_level3' 
    
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
    regions_index = results.loc[(age_group, 'All'), year].index
      
    # Locate results in dataframe
    results.loc[(age_group, 'All'), year] = (df['mortality_all'].reindex(regions_index)).values
    results.loc[(age_group, 'Hot'), year] = (df['mortality_hot'].reindex(regions_index)).values
    results.loc[(age_group, 'Cold'), year] = (df['mortality_cold'].reindex(regions_index)).values
    
    

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
    
    print(f'[1.5] Importing Population data for {SSP} scenario...')
    
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



def ms_temp_to_ir(wdir, climate_path, year, ir, spatial_relation):
    
    # Read daily temperature data generated from monthly statistics
    temp_t2m, _ = tmp.daily_from_monthly_temp(climate_path, year, 'MEAN', to_xarray=True)
    
    # Create a list of dates for the specified year
    date_list = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D').astype(str)
    
    # Temporarily store daily temperatures in a dictionary
    temp_dict = {}
    for day in date_list:
        daily_temperatures = temp_t2m.sel(valid_time=day).values.ravel()
        temp_dict[day] = daily_temperatures[spatial_relation.index]

    # Calculate mean temperature per impact region and round
    day_temp_df = pd.DataFrame(temp_dict, index=spatial_relation['index_right'])
    day_temp_df = day_temp_df.groupby('index_right').mean()
    
    # Fill in nan with 20
    day_temp_df = day_temp_df.fillna(20)

    # # Alternative nearest neighbor filling approach
    # ir = gpd.read_file(wdir+'/data/carleton_sm/ir_shp/impact-region.shp')
    # ir = ir.join(day_temp_df, how='right')
    
    # ir_valid = ir[ir.notna().any(axis=1)].copy()
    # ir_nan = ir[ir.isna().any(axis=1)].copy()
    # ir_filled = ir_nan.sjoin_nearest(ir_valid, how="left", distance_col="dist")
    
    day_temp_df_rounded = day_temp_df.round(1)
    day_temp_df_rounded.insert(0, 'hierid', ir)
    
    return day_temp_df_rounded



def era5_temp_to_ir(climate_path, year, ir, spatial_relation):
    
    # Read ERA5 daily temperature data for a specific year
    temp_t2m, _ = tmp.daily_temp_era5(climate_path, year, 'mean', pop_ssp=None, to_array=False)
    temp_t2m = temp_t2m.t2m
    
    # Select all available dates
    dates = temp_t2m['valid_time'].values
    
    # Create a list of dates for the specified year
    date_list = dates[np.isin(temp_t2m['valid_time'].values.astype('datetime64[Y]'),
                            np.datetime64(f'{year}', 'Y'))].astype('datetime64[D]').astype(str)
    
    # Temporarily store daily temperatures in a dictionary
    temp_dict = {}
    for day in date_list:
        daily_temperatures = temp_t2m.sel(valid_time=day).values.ravel()
        temp_dict[day] = daily_temperatures[spatial_relation.index]
            
    # Calculate mean temperature per impact region and round
    day_temp_df = pd.DataFrame(temp_dict, index=spatial_relation['index_right'])
    day_temp_df = day_temp_df.groupby('index_right').mean()
    day_temp_df_rounded = day_temp_df.round(1)
    day_temp_df_rounded.insert(0, 'hierid', ir)
    
    return day_temp_df_rounded
    
    
    
def daily_temperature_to_ir(wdir, climate_path, year, ir, spatial_relation, temp_source):
    
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
    
    print('[3.1] Importing daily temperature data for year', year)
    
    if temp_source == 'ERA5':
        
        day_temp = era5_temp_to_ir(climate_path, year, ir, spatial_relation)
        
    if temp_source == 'MS':
        day_temp = ms_temp_to_ir(wdir, climate_path, year, ir, spatial_relation)
    
    return day_temp



def generate_erf_group(df, group, t):
    
    # Extract polynomial coefficients for the specified age group
    tas1 = df[f'coeff_1_{group}'].values[:, None]  
    tas2 = df[f'coeff_2_{group}'].values[:, None]
    tas3 = df[f'coeff_3_{group}'].values[:, None]
    tas4 = df[f'coeff_4_{group}'].values[:, None]
    
    # Broadcasting: generate polynomial values for all regions and temperatures
    poly_raw = (  tas1 * t + tas2 * t**2 + tas3 * t**3 + tas4 * t**4 ) 
    
    # Find indices for temperature range between 10 and 30 degrees Celsius
    idx_min_start = np.where(np.isclose(t, 10.0, atol=0.05))[0][0]
    idx_min_end   = np.where(np.isclose(t, 30.0, atol=0.05))[0][0]
    segment = poly_raw[:, idx_min_start:idx_min_end]
    
    # Determine Tmin (temperature at which mortality is minimized) and add to DataFrame
    df[f"Tmin {group}"] = t[idx_min_start + np.argmin(segment, axis=1)]
    
    # Shift polynomial values based on Tmin
    tmin = df[f"Tmin {group}"].values[:, None]
    poly_shift = (poly_raw - tas1*tmin - tas2*tmin**2 - tas3*tmin**3 - tas4*tmin**4)
    
    # Impose weak monotonicity: set negative values to zero
    idx_tmin = np.searchsorted(t, df[f"Tmin {group}"].values)
    
    for j in range(poly_shift.shape[1] - 2, -1, -1):
        mask = j < idx_tmin
        poly_shift[mask, j] = np.maximum(poly_shift[mask, j], poly_shift[mask, j + 1])
    
    for j in range(1, poly_shift.shape[1]):
        mask = j > idx_tmin
        poly_shift[mask, j] = np.maximum(poly_shift[mask, j], poly_shift[mask, j - 1])
    
    return df[['region', f'Tmin {group}']], poly_shift



def import_climate(temp_dir, year, spatial_relation, ir):
    
    '''
    Import climate data from the specified directory and return 30-year mean temperature 
    per impact region and year.
    '''
    
    # Read monthly mean of daily mean temperature data
    temp_mean = xr.open_dataset(temp_dir+f'GTMP_MEAN_30MIN.nc')
    
    # Calculate annual mean temperature and climatology
    temp_mean_annual = temp_mean['GTMP_MEAN_30MIN'].mean(dim='NM')
    
    # Calculate 30-year rolling mean temperature
    tmean = temp_mean_annual.rolling(time=30, min_periods=1).mean()
    
    temp_dict = {}
    climate_temp = tmean.sel(time=f'{year}').values.ravel()
    temp_dict[year] = climate_temp[spatial_relation.index]

    # Calculate mean temperature per impact region and round
    year_temp_df = pd.DataFrame(temp_dict, index=spatial_relation['index_right'])
    year_temp_df = year_temp_df.groupby('index_right').mean()
    
    # Fill in nan with 20
    year_temp_df = year_temp_df.fillna(20)
    year_temp_df.insert(0, 'hierid', ir)
    year_temp_df = year_temp_df.rename(columns={year: 'tmean', 'hierid':'region'})
    
    return year_temp_df
    
    
    
def import_gamma_coefficients(wdir):
    
    with open(wdir+'data/carleton_sm/Agespec_interaction_response.csvv') as f:
        for i, line in enumerate(f, start=1):
            if i == 23:
                
                # Read gamma coefficients from the specified line and convert to float
                nums = [float(x) for x in line.split(",")]
                
                # Split coefficients into age groups
                young_vals  = nums[0:12]
                older_vals  = nums[12:24]
                oldest_vals = nums[24:36]

                # Create reshaped numpy arrays for each age group    
                young_arr  = np.array(young_vals ).reshape(4,3)
                older_arr  = np.array(older_vals ).reshape(4,3)
                oldest_arr = np.array(oldest_vals).reshape(4,3)
                
                gamma_np = {
                    'young' : young_arr,
                    'older' : older_arr,
                    'oldest': oldest_arr
                }
                
    return gamma_np



def generate_erfs_adapt(wdir, year, gdppc, temp_dir, spatial_relation, ir):
    
    '''
    Generate the exposure response functions given an adaptation scenario.
    '''
    
    print('[3.1] Generating Exposure Response Functions...')
    
    # Import gamma coefficients from Carletom et al. (2022)
    gammas = import_gamma_coefficients(wdir)
        
    # Determine the resolution and range of daily temperatures
    t = np.arange(-50, 60.1, 0.1).round(1)
    
    # Import climate
    tmean =  import_climate(temp_dir, year, spatial_relation, ir)
    
    # Select GDP per capita for the given year
    gdppc = gdppc[gdppc['year']==year][['region', 'loggdppc']]
    
    # Merge GDP per capita with climate data
    climate_gdp = tmean.merge(gdppc, on='region', how='left')

    for group, coef_matrix in gammas.items():
        
        for degree, degree_row in enumerate(coef_matrix, start=1): 
            
            coeff_val = np.zeros(len(climate_gdp))
            
            const, c_tmean, c_gdppc = degree_row
            
            coeff_val = (const
                         + c_tmean * climate_gdp['tmean'].values
                         + c_gdppc * climate_gdp['loggdppc'].values) 
            
            climate_gdp[f'coeff_{degree}_{group}'] = coeff_val
            
    tmin_young, df_young  = generate_erf_group(climate_gdp, "young", t)
    tmin_older, df_older  = generate_erf_group(climate_gdp, "older", t)
    tmin_oldest, df_oldest  = generate_erf_group(climate_gdp, "oldest", t)
    
    tmin = tmin_young.merge(tmin_older, on='region').merge(tmin_oldest, on='region')
            
    return tmin, {'young': df_young, 'older': df_older, 'oldest': df_oldest}
        


def read_mortality_response(wdir, group):
    
    '''
    Read mortality response function for a given age group
    '''
    # Read mortality response function csv file created in the preprocessing step
    mor = pd.read_csv(wdir + f'/data/exposure_response_functions/erf_no-adapt_{group}.csv')
    num_other_columns = 2
        
    # Convert columns to float type and extract mortality values as numpy array
    columns = list(mor.columns)
    mor.columns = columns[:num_other_columns] + list(np.array(columns[num_other_columns:], dtype="float"))
    mor_np = mor.iloc[:, num_other_columns:].round(2).to_numpy()
    
    return mor_np

    
    
def import_erfs_noadapt(wdir):
    
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
    

    
def read_gdppc(wdir, SSP):
    
    '''
    Read GDP per capita files for a given SSP scenario.
    The files were created in the preprocessing step.
    
    Parameters:
    wdir : str
        Working directory
    SSP : str
        Socioeconomic scenario (e.g., 'SSP1', 'SSP2')
        
    Returns:
    gdppc : DataFrame
        DataFrame with GDP per capita data
    '''
    
    print(f'[1.4] Importing GDP per capita data for {SSP} scenario...')
    
    # Read GDP per capita file
    ssp = xr.open_dataset(wdir+f'data/carleton_sm/econ_vars/{SSP.upper()}.nc4')   
    
    gdppc = ssp['gdppc'].to_dataframe().reset_index()
    gdppc = gdppc.drop(columns=['model', 'ssp'])
    gdppc = gdppc.groupby(['region', 'year']).mean().reset_index()
    
    gdppc["gdppc13"] = (
        gdppc.groupby("region")["gdppc"]
        .rolling(window=13, min_periods=1)   
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    gdppc['loggdppc'] = np.log(gdppc['gdppc13'])
    
    return gdppc[['region', 'year', 'loggdppc']]


    
def final_dataframe(regions, region_class, age_groups, SSP, years):
    
    '''
    Create final results dataframe with multiindex for age groups, temperature types, and regions
    
    Parameters:
    regions : str
        Region classification to use (e.g., 'IMAGE26', 'ISO3')
    region_class : DataFrame
        DataFrame with region classification
    age_groups : list
        List of age groups
    SSP : list
        List of socioeconomic scenarios (e.g., ['SSP1', 'SSP2'])
    years : list
        List of years to process
        
    Returns:
    results : DataFrame
        DataFrame to store final results
    '''
    
    print('[1.3] Creating final results dataframe...')
    
    if regions == 'countries':
        regions = 'gbd_level3'
    
    unique_regions = region_class[f'{regions}'].unique()
    unique_regions = unique_regions[~pd.isna(unique_regions)]
    
    t_types = ['Hot', 'Cold', 'All']
    
    # Create results multiindex dataframe
    results = pd.DataFrame(index=pd.MultiIndex.from_product([age_groups, t_types, unique_regions],
                                                            names=['age_group', 't_type', regions]), 
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
    if regions == 'countries':
        region_class = region_class[['hierid', 'gbd_level3']]
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



def grid_relationship(wdir, temp_source, climate_path, years):
    
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
    if temp_source == 'ERA5':
        temperature, _ = tmp.daily_temp_era5(climate_path, years[0], 'mean', pop_ssp=None, to_array=False)
    elif temp_source == 'MS':
        temperature, _ = tmp.daily_from_monthly_temp(climate_path, years[0], 'MEAN', to_xarray=True)
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
    gammas: any
    pop: any
    gdppc: any
    

    
def load_main_files(wdir, regions, SSP, years, climate_type, climate_path, adaptation):
    
    '''
    Read and load all main input files required for mortality calculations
    Parameters:
    wdir : str
        Working directory
    regions : str
        Region classification to use (e.g., 'IMAGE26', 'ISO3')
    SSP : STR
        Socioeconomic scenarios (e.g., 'SSP1', 'SSP2')
    years : list
        List of years to process
    climate_type : str
        Type of climate data ('ERA5', 'AR6')
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
    '''
    
    print('[1] Loading main input files...')
    
    # Define age groups
    age_groups = ['oldest', 'older', 'young']
    
    # Create relationship between temperature data and impact regions
    spatial_relation, ir = grid_relationship(wdir, climate_type, climate_path, years)
    
    # Open file with region classification
    region_class = select_regions(wdir, regions)

    # Create results dataframe
    results = final_dataframe(regions, region_class, age_groups, SSP, years)
    
    # Read erfs
    mor_np, t_min = import_erfs_noadapt(wdir) if adaptation==False else (None, None)
    
    # Import gamma coefficients
    gammas = import_gamma_coefficients(wdir)
    
    # Read population files
    pop = read_population_csv(wdir, SSP, years)   
    
    # Read GDP per capita files if adaptation is on
    gdppc = read_gdppc(wdir, SSP) if adaptation else None
    
        
    return LoadResults(
        age_groups=age_groups,
        spatial_relation=spatial_relation,
        ir=ir,
        region_class=region_class,
        results=results,
        mor_np=mor_np,
        t_min=t_min,
        gammas = gammas,
        pop = pop,
        gdppc = gdppc
    )



def calculate_mortality(wdir, years, temp_source, temp_dir, SSP, regions, adaptation, IAM_format=False):
    
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
    SSP : STR
        List of socioeconomic scenarios (e.g., 'SSP1', 'SSP2')
    regions : str
        Region classification to use (e.g., 'IMAGE26', 'ISO3')
    IAM_format : bool, optional
        If True, output will be formatted as IAMs' output (default is False)
        
    Returns:
    None
    Saves the mortality results to CSV files in the output folder per 
    climate model and scenario.
    '''
    
    res = load_main_files(wdir, regions, SSP, years, temp_source, temp_dir, adaptation)    
        
    print('[2] Starting mortality calculations')
        
    # Iterate over years
    for year in years:
        
        print(f'[3] Processing year {year}...')
        
        if adaptation == True:
            t_min, erfs = generate_erfs_adapt(wdir, year, res.gdppc, temp_dir,
                                              res.spatial_relation, res.ir)
            
        else:
            erfs = res.mor_np
            t_min = res.t_min
        
        daily_temp = daily_temperature_to_ir(wdir, temp_dir, year, res.ir, res.spatial_relation, 
                                             temp_source)
        
        for age_group in res.age_groups:      

            annual_regional_mortality(res.results, daily_temp, SSP, age_group, erfs[age_group], 
                                      res.pop, year, t_min, regions, res.region_class)
        
        print('[3.2] Mortality per age group calculated.')
                
    postprocess_results(wdir, years, res.results, temp_source, SSP, IAM_format, regions)