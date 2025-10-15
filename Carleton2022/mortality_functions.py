import numba as nb
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import params
from shapely.geometry import Point, Polygon



def generate_temp_per_day(temperature, day, relation):
    
    '''
    Convert temperature data for a specific day to impact region level
    '''
    
    # Get temperatures for the specified day 
    temperatures = temperature.sel(time=day).values.flatten() 
    # Associate the temperatures with the relationship precalculated
    relation['temperature'] = temperatures[relation.index] 
    # Calculate mean temperature per impact region
    result = relation.groupby('index_right')['temperature'].mean()

    return result



def read_era5(wdir, climate_path, year):
    
    '''
    Read ERA5 daily temperature data
    '''
    
    era5_daily = xr.open_dataset(climate_path+f'/era5_t2m_mean_day_{year}.nc')
    
    # Shift longitudinal coordinates
    era5_daily = era5_daily.assign_coords(longitude=((era5_daily.coords['longitude'] + 180) % 360 - 180)).sortby("longitude")
    
    # Convert to Celsius 
    era5_daily -= 273.15

    return era5_daily.t2m



def era5_emperature_to_ir(wdir, climate_path, year, ir, spatial_relation):
    
    '''
    Convert daily temperature data to impact region level
    '''
    
    # Read ERA5 daily temperature data for a specific year
    era5_t2m = read_era5(wdir, climate_path, year)
    
    # Create spatial relationship between temperature data points and impact regions
    fechas = era5_t2m['valid_time'].values
    
    # Create dataframe with temperature data at impact region level for all days in a year
    date_list = fechas[np.isin(fechas.astype('datetime64[Y]'), np.datetime64(f'{year}', 'Y'))]
    
    # Generate temperature per day at impact region level
    date_list = date_list.astype('datetime64[D]').astype(str)
    
    # Create dataframe with temperature data at impact region level for all days in a year
    df = pd.DataFrame(ir['hierid'])
    
    # Assign temperature data for each day
    for item in date_list:
        df[item] = generate_temp_per_day(era5_t2m, item, spatial_relation)
        
    # Round temperature values to one decimal place
    df_rounded = df.round(1)
    
    return df_rounded



def read_mortality_response(wdir, group):
    
    '''
    Read mortality response function for a given age group
    '''
    
    # Read mortality response function csv file created in the preprocessing step
    mor = pd.read_csv(wdir + f'exposure_response_functions/erf_no-adapt_{group}.csv')
    
    # Convert columns to float type and extract mortality values as numpy array
    columns = list(mor.columns)
    num_other_columns = 2
    mor.columns = columns[:num_other_columns] + list(np.array(columns[num_other_columns:], dtype="float"))
    mor_np = mor.iloc[:, num_other_columns:].round(2).to_numpy()
    
    return mor_np



def read_population_csv(wdir, SSP, age_group):
    
    '''
    Read Carleton et al. (2022) population CSV files for a given SSP scenario and age group
    '''
    
    # Read population files
    POP = pd.read_csv(f'{wdir}/data/gdp_pop_csv/POP_{SSP}_{age_group}.csv')  

    # Discard the regions of Antarctica and the Caspian Sea    
    POP = POP[~POP['hierid'].str.contains('ATA|CA-', regex=True)]   
    
    return POP   



def find_coord_vals(possible_names, coord_names, temperature):
    
    '''
    Find the correct coordinate name in the dataset
    '''
    
    for name in possible_names:
        if name in coord_names:
            return temperature[name].values
    raise KeyError(f"No coordinate was found among: {possible_names}")




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
    
    # Flatten the values
    lon_flatten = lon2d.flatten()    
    lat_flatten = lat2d.flatten()
    
    # Create dataframe with lat and lon combinations
    points_df = pd.DataFrame({'longitude': lon_flatten, 'latitude': lat_flatten})   
    
    #  Make the geometry with squares for the temperature values
    points_df['geometry'] = [create_square(lon, lat, lon_size, lat_size) 
                             for lon, lat in zip(points_df.longitude, points_df.latitude)] 
    
    # Convert to GeoDataFrame and reset index
    points_gdf = gpd.GeoDataFrame(points_df) 
    points_gdf = points_gdf.reset_index()
    
    # Load .shp file with impact regions
    ir = gpd.read_file(f'{wdir}'+'/data/ir_shp/impact-region.shp')
    
    # Set the same coordinate reference system (CRS)
    points_gdf = points_gdf.set_crs(ir.crs, allow_override=True)
    
    # Make spatial join
    relationship = gpd.sjoin(points_gdf, ir, how='inner', predicate='intersects')
    
    # Keep only necessary columns
    relationship = relationship[['geometry','index_right']]

    return relationship, ir



@nb.njit
def temp_to_column_location(temp):
    
    '''
    Get the column index in the mortality response function corresponding to a given temperature
    '''
    
    
    min_temperature = -50.0
    temp = np.round(temp, 1)
    return int(np.round(((temp - min_temperature) * 10)))



@nb.njit
def mortality_from_erf(temperature_data, mortality_data, tmin_column=None, mode='all'):
    
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
    
    for i in range(n):
        temp = temperature_data[i]
        mort_value = mortality_data[i, temp_to_column_location(temp)]
        
        if mode == 'all':
            result[i] = mort_value
        elif mode == 'hot':
            result[i] = mort_value if temp > tmin_column[i] else np.nan
        elif mode == 'cold':
            result[i] = mort_value if temp < tmin_column[i] else np.nan
        else:
            result[i] = np.nan  # unknown mode
        
    return result



def calculate_mortality_year(SSP, mortality_np):
    
    '''
    Calculate yearly mortality for a given SSP scenario and mortality response function
    '''
    
    # Use 2 when running climate variability files, otherwise use 1
    temperature_other_columns = 1 
    # Calculate mortality for each day and sum to get total mortality
    mortality_SSP = SSP.iloc[:, :temperature_other_columns].copy()
    
    mortality_SSP = pd.concat([
        mortality_SSP,
        pd.concat([
            pd.Series(mortality_from_erf(temperature_series.to_numpy(), mortality_np, tmin_column=None, mode='all'), 
                      name=day+"_mortality")
            for day, temperature_series in SSP.iloc[:, temperature_other_columns:].items()
        ], axis=1)
    ], axis=1)
    
    # Sum daily mortalities to get total mortality
    mortality_SSP['total mortality'] = mortality_SSP.filter(like='_mortality').sum(axis=1)
    
    return mortality_SSP
          
          
          
def relative_mortality(wdir, df, SSP, age_group, mor_np, pop_file, year):
    
    '''
    Calculate yearly mortality for a given SSP scenario, age group, mortality response function, population file, and year
    '''
    
    # Read temp files
    SSP = pd.read_csv(f'{wdir}/Climate data/Climate ensemble/{climate_model}/{scenario}/BC_{climate_model}_{scenario}_{year}.csv')  
    # Calculate relative mortality per region (deaths/100,000)
    mor_temp = calculate_mortality_year(SSP, mor_np) 
    # Discard the regions of Antarctica and the Caspian Sea
    mor_temp = mor_temp[~mor_temp['hierid'].str.contains('ATA|CA-', regex=True)]
    # Calculate mortality relative to SSP scenario
    total_mortality = np.sum(pop_file[f'{year}'].to_numpy() * mor_temp['total mortality'].to_numpy() / 1e5) 
    # Calculate relative mortality per 100,000
    relative_mortality = total_mortality * 1e5 /np.sum(pop_file[f'{year}'].to_numpy())
    # Locate results in dataframe
    df.at[(age_group, SSP), year] = relative_mortality
    # Print progress
    print(f'{climate_model} - {age_group} - {SSP} - {year}')
    
    
    
def mortality_scenario(wdir, years, climate_type, climate_path, scenarios_SSP, scenarios_RCP):
    
    '''
    Generate mortality projections for given parameters
    '''
    
    # Define age groups
    age_groups = ['oldest', 'older', 'young']
    
    # Create relationship between temperature data and impact regions
    spatial_relation, ir = grid_relationship(wdir, climate_type, climate_path, years)
    
    # Create results dataframe
    results = pd.DataFrame(index=pd.MultiIndex.from_product([age_groups, scenarios_SSP], 
                                                            names=['Age group', 'Scenario']),
                           columns=years)
    
    # Set climate models parameter based on climate type
    if climate_type == 'CMIP6':
        climate_models = params.climate_models_dic.keys()
        
    elif climate_type == 'ERA5':
        climate_models = ['ERA5']
        
    elif climate_type == 'AR6':
        climate_models = params.climate_models_ar6.keys()
        
    # for loop for climate models
    for climate_model in climate_models:
        for age_group in age_groups:
            
            # Read mortality response functions
            mor_np = read_mortality_response(wdir, age_group)
            
            for scenario_SSP in scenarios_SSP:

                # Read population files
                pop = read_population_csv(wdir, scenario_SSP, age_group)
                
                for scenario_RCP in scenarios_RCP:
                    for year in years:
                        temperature = era5_emperature_to_ir(wdir, climate_path, year, ir, spatial_relation)
                        print(temperature)
                        relative_mortality(wdir, results, scenario_RCP, age_group, mor_np, pop, year)
                        
        results.to_csv(f'{wdir}/output/mortality_no-adapt_{scenario_SSP}_{climate_model}.csv')