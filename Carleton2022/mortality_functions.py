import numba as nb
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon



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



'''
CMIP6 climate models and their corresponding realization and grid label.
'''

climate_models_dic = {'AWI-CM-1-1-MR':['r1i1p1f1', 'gn'], 'BCC-CSM2-MR':['r1i1p1f1', 'gn'], 
                      'CAMS-CSM1-0':['r1i1p1f1', 'gn'], 'CESM2':['r4i1p1f1', 'gn'], 
                      'CESM2-WACCM':['r1i1p1f1', 'gn'], 'CMCC-CM2-SR5':['r1i1p1f1', 'gn'],
                      'CMCC-ESM2':['r1i1p1f1', 'gn'], 'CNRM-CM6-1-HR':['r1i1p1f2', 'gr'], 
                      'EC-Earth3':['r4i1p1f1', 'gr'], 'EC-Earth3-Veg':['r1i1p1f1', 'gr'], 
                      'GFDL-ESM4':['r1i1p1f1', 'gr1'], 'INM-CM4-8':['r1i1p1f1', 'gr1'],
                     'INM-CM5-0':['r1i1p1f1', 'gr1'], 'MPI-ESM1-2-HR':['r1i1p1f1', 'gn'], 
                     'MRI-ESM2-0':['r1i1p1f1', 'gn'], 'NorESM2-MM':['r1i1p1f1', 'gn'], 
                     'TaiESM1':['r1i1p1f1', 'gn']}




### ------------------------------------------------------------------------------
### ------------------------------ FUNCTIONS -------------------------------------



def postprocess_results(wdir, years, results, climate_type, climate_model, scenario_SSP, scenario_RCP, IAM_format, regions):
    
    '''
    Postprocess final results and save to CSV file in output folder.
    '''
    
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
    if climate_type == 'CMIP6':
        results.to_csv(f'{wdir}/output/mortality_{climate_type}_{climate_model}_{scenario_SSP}{scenario_RCP[-2:]}.csv')
    
    

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
    Calculate yearly mortality for a given SSP scenario and mortality response function
    '''
    
    # Use 2 when running climate variability files, otherwise use 1
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
    '''
    
    t_min_age = np.array(t_min[f'Tmin {age_group}']) 
     
    # Calculate relative mortality per region (deaths/100,000)
    mor_temp_hot = calculate_mortality_year(daily_temperature, mor_np, t_min_age, mode='hot') 
    mor_temp_cold = calculate_mortality_year(daily_temperature, mor_np, t_min_age, mode='cold')
    mor_temp = calculate_mortality_year(daily_temperature, mor_np, t_min_age, mode='all')
    
    # Calculate total mortality per region
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
    Read Carleton et al. (2022) population CSV files for a given SSP scenario and age group
    '''
    
    pop_groups = {}
    
    for age_group in ['oldest', 'older', 'young']:
    
        if all(y in range(2000, 2022) for y in years):
            SSP = 'historical'
        
        # Read population files
        POP = pd.read_csv(f'{wdir}/data/gdp_pop_csv/POP_{SSP}_{age_group}.csv')  

        if years[0]<2010 and years[-1]>2022:
            
            pop_historical = pd.read_csv(f'{wdir}/data/gdp_pop_csv/POP_historical_{age_group}.csv')
            cols_to_add = pop_historical.columns.difference(POP.columns)
            left = POP.iloc[:, :2] 
            right = POP.iloc[:, 2:]  
            POP = pd.concat([left, pop_historical[cols_to_add], right], axis=1)
            
        pop_groups[age_group] = POP
        
    print('Population for files imported')
    
    return pop_groups
    
    
    
def era5_temperature_to_ir(wdir, climate_path, year, ir, spatial_relation):
    
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
    era5_t2m = read_era5(wdir, climate_path, year)
    
    # Select all available dates
    dates = era5_t2m['time'].values
    
    # Create a list of dates for the specified year
    date_list = dates[np.isin(era5_t2m['time'].values.astype('datetime64[Y]'),
                              np.datetime64(f'{year}', 'Y'))].astype('datetime64[D]').astype(str)
    
    # Temporarily store daily temperatures in a dictionary
    temp_dict = {}
    for day in date_list:
        daily_temperatures = era5_t2m.sel(time=day).values.ravel()
        temp_dict[day] = daily_temperatures[spatial_relation.index]
            
    # Calculate mean temperature per impact region and round
    df = pd.DataFrame(temp_dict, index=spatial_relation['index_right'])
    df = df.groupby('index_right').mean()
    df_rounded = df.round(1)
    df_rounded.insert(0, 'hierid', ir['hierid'])
    
    print(f'ERA5 daily temperature for {year} imported')
    
    return df_rounded
    
    

def daily_temperatures(wdir, climate_type, climate_path, climate_model, scenario_RCP, year, spatial_relation, ir):
    
    if climate_type == 'ERA5':
        # Read and convert temperature data to impact region level
        daily_temp = era5_temperature_to_ir(wdir, climate_path, year, ir, spatial_relation)
        
    if climate_type == 'CMIP6':
        # Read temperature files (previously calculated to impact region level)
        daily_temp = pd.read_csv(f'{climate_path}/{climate_model}/{scenario_RCP}/BC_{climate_model}_{scenario_RCP}_{year}.csv') 
        
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
    
   # Read mortality response functions
    
    mor_oldest = read_mortality_response(wdir, 'oldest')
    mor_older = read_mortality_response(wdir, 'older')
    mor_young = read_mortality_response(wdir, 'young')
    
    mor_np = {'oldest': mor_oldest, 'older': mor_older, 'young': mor_young}
    
    # Open file with optimal temperature (Tmin) per impact region
    t_min = pd.read_csv(f'{wdir}/data/exposure_response_functions/T_min.csv')
    
    print('Exposure Response Functions imported')
    
    return mor_np, t_min


    
def final_dataframe(regions, region_class, age_groups, scenarios_SSP, years):
    
    '''
    Create final results dataframe with multiindex
    '''
    
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
    
    # Load region classification file
    region_class = pd.read_csv(f'{wdir}/data/region_classification.csv')
    
    if regions == 'impact_regions':
        region_class = region_class[['hierid']]
    else:
        region_class = region_class[['hierid', regions]]
    
    return region_class



def find_coord_vals(possible_names, coord_names, temperature):
    
    '''
    Find the correct coordinate name in the dataset
    '''
    
    for name in possible_names:
        if name in coord_names:
            return temperature[name].values
    raise KeyError(f"No coordinate was found among: {possible_names}")



def read_era5(wdir, climate_path, year):
    
    '''
    Read ERA5 daily temperature data
    '''
    
    era5_daily = xr.open_dataset(climate_path+f'/era5_t2m_mean_day_{year}.nc')
    
    # Shift longitudinal coordinates
    era5_daily = era5_daily.assign_coords(longitude=((era5_daily.coords['longitude'] + 180) % 360 - 180)).sortby("longitude")
    
    # Convert to Celsius 
    era5_daily -= 273.15
    
    # Rename time coordinate 
    era5_daily = era5_daily.rename({'valid_time': 'time'})

    return era5_daily.t2m



def grid_relationship(wdir, climate_type, climate_path, years):
    
    '''
    Create a spatial relationship between temperature data points from the nc files and impact regions
    '''
    
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
        temperature = read_era5(wdir, climate_path, years[0])
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
    ir = gpd.read_file(f'{wdir}'+'/data/carleton_sm/ir_shp/impact-region.shp')
    points_gdf = points_gdf.set_crs(ir.crs, allow_override=True)
    
    # Make spatial join
    relationship = gpd.sjoin(points_gdf, ir, how='inner', predicate='intersects')
    
    # Keep only necessary columns
    relationship = relationship[['geometry','index_right']]
    
    print('Spatial relationship (temperature grid to impact region level) created')

    return relationship, ir



def mortality_scenario(wdir, years, climate_type, climate_path, scenarios_SSP, scenarios_RCP, regions, IAM_format=False):
    
    '''
    Generate mortality projections for given parameters
    '''
    
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
    
    # Set climate models parameter based on climate type
    if climate_type == 'CMIP6':
        climate_models = climate_models_dic.keys()
        scenarios_RCP = scenarios_RCP
        
    elif climate_type == 'ERA5':
        climate_models = ['']
        scenarios_RCP = ['']
        
    elif climate_type == 'AR6':
        climate_models = climate_models_ar6.keys()
        scenarios_RCP = ['']
        
    print('Starting mortality calculations')
        
    # Iterate over climate models or use single ERA5/AR6 data
    for climate_model in climate_models:        
        # Iterate over population SSP scenarios
        for scenario_SSP in scenarios_SSP:
            # Read population files
            pop = read_population_csv(wdir, scenario_SSP, years)
            # Iterate over RCP scenarios
            for scenario_RCP in scenarios_RCP:                
                # Iterate over years
                for year in years:
                    daily_temp = daily_temperatures(wdir, climate_type, climate_path, 
                                                    climate_model, scenario_RCP, year, 
                                                    spatial_relation, ir)
                    # Iterate over age groups
                    for age_group in age_groups:                
                            # Calculate relative mortality and store in results dataframe
                            annual_regional_mortality(results, daily_temp, scenario_SSP, 
                                                    age_group, mor_np[age_group], pop, year, 
                                                    t_min, regions, region_class)
                    print('Mortality calculated for year', year)
                    
            postprocess_results(wdir, years, results, climate_type, climate_model, 
                                scenario_SSP, scenario_RCP, IAM_format, regions)