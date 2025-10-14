import numba as nb
import pandas as pd
import numpy as np
import geopandas as gpd
import params
from shapely.geometry import Point, Polygon


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



def relationship(ir, longitud, latitud, temperatura, extended=False):
    
    def create_square(lon, lat): ### function only works for climate data with squared grids
        return Polygon([
            (lon, lat),
            (lon + longitud, lat),
            (lon + longitud, lat + latitud),
            (lon, lat + latitud)
        ])

    if extended:
        new_lon = temperatura.longitude.values   # Obtaining the lon and lat values, format varies depending of the file
        new_lat = temperatura.latitude.values
    else:
        new_lon = temperatura.lon.values   # Obtaining the lon and lat values
        new_lat = temperatura.lat.values  

    new_lon = np.where(new_lon > 180, new_lon - 360, new_lon)   # Converting lon to right -180 to 180 degrees
    lon2d, lat2d = np.meshgrid(new_lon, new_lat)    # Create meshgrid
    lon_flatten = lon2d.flatten()    # Flatten the values
    lat_flatten = lat2d.flatten()
    points_df = pd.DataFrame({ 'longitude': lon_flatten, 'latitude': lat_flatten})    # Create dataframe with lat and lon
    points_df['geometry'] = [create_square(lon, lat) for lon, lat in zip(points_df.longitude, points_df.latitude)]  # Make the geometry with squares for the temperature values
    points_gdf = gpd.GeoDataFrame(points_df) # Convertir el DataFrame a un GeoDataFrame
    points_gdf = points_gdf.reset_index()
    points_gdf.crs = ir.crs  # Set same .crs
    relationship = gpd.sjoin(points_gdf, ir, how='inner', predicate='intersects') #Make spatial join only once
    relationship = relationship[['geometry','index_right']]
    
    #temperatures = temperatura.values.flatten() - 273.15
    #relationship['temperature'] = temperatures[relationship.index]
    #result = relationship.groupby('index_right')['temperature'].mean()

    return relationship



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
    
    
    
def mortality_projections(wdir, years, scenarios_SSP, scenarios_RCP, age_groups):
    
    '''
    Generate mortality projections for given parameters
    '''
    
    # Create results dataframe
    index = pd.MultiIndex.from_product([age_groups, scenario_SSP], names=['Age group', 'Scenario'])
    results = pd.DataFrame(index=index, columns=years)
    
    # for loop for climate models
    for climate_model in params.climate_models_dic.keys():
        for age_group in age_groups:
            
            # Read mortality response functions
            mor_np = read_mortality_response(wdir, age_group)
            
            for scenario_SSP in scenario_SSP:

                # Read population files
                pop = read_population_csv(wdir, scenario_SSP, age_group)
                
                for scenario_RCP in scenarios_RCP:
                    for year in years:
                        relative_mortality(wdir, results, scenario_RCP, age_group, mor_np, pop, year)
                        
        results.to_csv(f'{wdir}/output/mortality_no-adapt_{scenario_SSP}_{climate_model}.csv')