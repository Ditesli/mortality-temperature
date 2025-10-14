import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd   ### Not in imagepy312_base

### Set working directory
wdir = 'X:\\user\\liprandicn\\Health Impacts Model'


### -----------------------------------------------------------------------------------------
### Linear interpolation of population data to yearly data

def get_annual_pop(scenario, coarse=False, start_year=1970, end_year=2100):
    
    '''Read scenario-dependent population data and interpolate it to yearly data'''
    
    pop = xr.open_dataset(f'{wdir}\\Socioeconomic_Data\\Population\\GPOP\\GPOP_{scenario}.nc')
    
    if coarse:
        ### Reduce resolution to 15 min
        pop = pop.coarsen(latitude=3, longitude=3, boundary='pad').sum(skipna=False)
        
    # Linearly interpolate to yearly data
    yearly_data = pd.date_range(start=f'{start_year}/01/01', end=f'{end_year}/01/01', freq='YS')
    pop_yearly = pop.interp(time=yearly_data)
    
    print(f'{scenario} population data loaded')
    
    return pop_yearly


def get_all_population_data():
    
    pop_ssp_1 = get_annual_pop('SSP1_M')
    pop_ssp_2 = get_annual_pop('SSP2_CP')
    pop_ssp_3 = get_annual_pop('SSP3_H')
    pop_ssp_5 = get_annual_pop('SSP5_H')

    # Create xarray with all the SSPs scenarios
    pop_all_ssp = xr.concat([pop_ssp_1, pop_ssp_2, pop_ssp_3, pop_ssp_5], dim = 'ssp')
    pop_all_ssp['ssp'] = ['ssp1', 'ssp2', 'ssp3', 'ssp5']
    # Mask with positive pop values
    mask_positive_pop = (pop_all_ssp > 0).any(dim=['time', 'ssp'])
    
    return pop_all_ssp, mask_positive_pop


### -----------------------------------------------------------------------------------------
### Load climate files (AR6 & GCM data)

def read_climate_data():
    
    '''Import GCM data from IMAGE land and load AR6 mean pathways
    (this will be later replaced by the emulator) '''
    
    ### Import GCM data from IMAGE land 
    gcm_diff = xr.open_dataset(f'{wdir}\\ClimateData\\GCM\\ensemble_mean_ssp585_tas_15min_diff.nc')
    gcm_start = xr.open_dataset(f'{wdir}\\ClimateData\\GCM\\ensemble_mean_ssp585_tas_15min_start.nc')
    gcm_end = xr.open_dataset(f'{wdir}\\ClimateData\\GCM\\ensemble_mean_ssp585_tas_15min_end.nc')

    ### Import AR6 data 
    ar6 = pd.read_csv('X:\\user\\dekkerm\\Data\\AR6_snapshots\\AR6_snapshot_Nayeli_global.csv')
    cc_path_mean = ar6.iloc[:, 10:-1].groupby(ar6['Category']).mean()
    
    print('Climate data loaded')
    
    return gcm_diff, gcm_start, gcm_end, cc_path_mean


### -----------------------------------------------------------------------------------------
### Load IMAGE regions and temperature zone xarray files

def read_IMAGEregions_and_TempZone():
    
    ### Import ERA5 temperature zones
    era5_tz = xr.open_dataset(f'{wdir}\\ClimateData\\ERA5\\ERA5_mean_1980-2019_land_t2m_tz.nc')

    ### Read in IMAGE region data and interpolate to match files resolution
    greg = xr.open_dataset('X:\\user\\waaldl\\TOOLS\\PYTHON_SCRIPTS_GENERAL\\addtime\\GREG.nc')
    greg = greg.interp(longitude=era5_tz.longitude, latitude=era5_tz.latitude, method='nearest').mean(dim='time') 
    
    ### Convert files to numpy arrays
    greg = greg.GREG.values
    era5_tz = era5_tz.t2m.values
    
    print('IMAGE regions and Temperature zone data loaded')
    
    return greg, era5_tz


    
### ------------------------------------------------------------------------------------------

def read_erf_data(wdir, all_diseases=True):
    
    """
    Read the raw Exposure Response Functions of relevant diseases from the specified path.
    
    Returns:
    - erf: Dictionary containing the ERF dataframes. Relevant diseases only
    
    """
    if all_diseases:
        disease_list = ['ckd', 'cvd_cmp', 'cvd_htn', 'cvd_ihd', 'cvd_stroke', 'diabetes', 'inj_animal', 'inj_disaster', 'inj_drowning', 
                'inj_homicide', 'inj_mech', 'inj_othunintent', 'inj_suicide', 'inj_trans_other', 'inj_trans_road', 'resp_copd', 'lri']   
    else:
        disease_list = ['ckd', 'cvd_cmp', 'cvd_htn', 'cvd_ihd', 'cvd_stroke', 'diabetes', 'lri', 'resp_copd'] 
    
    erf_dict = {}
    
    for disease in disease_list:
        erf_disease = pd.read_csv(f'{wdir}\\GBD_Data\\Exposure_Response_Functions\\ERF\\{disease}_curve_samples.csv', index_col=[0,1])
        erf_dict[disease] = erf_disease
        
    return erf_dict, disease_list



def get_erf_dataframe(wdir, all_diseases=True, mean=True, random_draw=True, draw=None):
    
    '''Get a single erf draw according to the arguments of the function
    - Mean: Calculates the mean of all draws 
    - random_draw: Selects a random draw between the 1000 available
    - draw: Select a specific draw for all diseases (useful for propagation of uncertainty runs)
    
    The function:
    1. Selects a draw or calculates the mean per disease and puts them in a single dataframe
    2. Uses the np.exp function to convert original ln(RR) to RR
    3. Renames temperature zone columns
    4. Produces two dics corresponding to the max and min daily temperatures fin each 
    temperature zone available in the files. This serves to clip later on the daily T data
    and could potentially change in the future if the ERFs are extrapolated.
    5. Put datframe entries in float64 format
    6. Fills any NaN values
    
    Returns: 
    1. Dataframe with temperature_zone, daily_temperature as Multiindex, and 
    the diseases in the columns
    2. Dicionaries with max and min daily temperatures per temperature zone
    
    '''
    if all_diseases:
        erf_dict, disease_list = read_erf_data(wdir, all_diseases=True)
        
    else:
        erf_dict, disease_list = read_erf_data(wdir, all_diseases=False)
    
    erf = pd.DataFrame(index=erf_dict['ckd'].index)
    
    if random_draw:
        draw = random.randint(0,999)
        
    for disease in disease_list:
        if mean:
            erf[disease] = erf_dict[disease].mean(axis=1)

        else:
            erf[disease] = erf_dict[disease][f'draw_{draw}']  
    
    # Convert log(rr) to rr      
    erf = erf.apply(lambda x: np.exp(x))
    # Rename for posterior merging
    erf.rename_axis(index={'annual_temperature':'temperature_zone'}, inplace=True)  
    
    # Convert MultiIndex levels into columns
    erf_reset = erf.reset_index()
    
    # Perform groupby operation using the columns
    min_dict = erf_reset.groupby('temperature_zone')['daily_temperature'].min().to_dict()
    max_dict = erf_reset.groupby('temperature_zone')['daily_temperature'].max().to_dict()
        
    # Round daily temperature values and set float64 format to posterior merging
    erf.index = pd.MultiIndex.from_frame(erf.index.to_frame(index=False).assign(
        **{erf.index.names[1]: lambda x: np.round(x[erf.index.names[1]].astype(float), 1)}))
    
    # Fill dataframe columns to remove NaNs in diseases that have a smaller tmeperature range
    erf = erf.groupby('temperature_zone').bfill().ffill()

    # Keep only temperature zones as index
    erf = erf.reset_index().set_index('temperature_zone')
    
    print('ERF dataframe generated')
            
    return erf, disease_list, min_dict, max_dict



def get_tmrel_map(wdir, year, mean=True, random_draw=True, draw=None):
    
    '''Get a single TMREL draw according to the arguments of the function
    - Mean: Calculates the mean of all draws 
    - random_draw: Selects a random draw between the 100 available
    - draw: Select a specific draw for all diseases (useful for propagation of uncertainty runs)
    
    The function:
    1. Opens the .nc file with optimal temperatures for the selected year
    (Available: 1990, 2010, 2020). Default for future projections: 2020
    2. Selects a draw o calculates the mean
    
    Returns: 
    1. 2-D np.array with the optimal temperature per pixel
    
    '''
    # year = 2020
    
    tmrel = xr.open_dataset(f'{wdir}\\GBD_Data\\Exposure_Response_Functions\\TMRELs_{year}.nc')
    
    if random_draw:
        draw = random.randint(1,100)
        
    if mean:
            tmrel = tmrel.tmrel.values.mean(axis=2)

    else:
        tmrel = tmrel.sel(draw=draw).tmrel.values
        
    print('TMREL data loaded')
        
    return tmrel


def read_shp_countries():
    
    '''
    Open shapefile thet includes country-level geomtetries (GBD level3 locations),
    using thhe geopandas library
    
    Returns:
    - geopandas dataframe with the shapefile
    - dataframe with unique list of country codes and names
    '''
    
    # Open shapfile with country geometries
    gdf_countries = gpd.read_file(f'{wdir}\\GBD_Data\\GBD_locations\\Shapefile\\GBD_shapefile.shp').to_crs("EPSG:4326")
    
    # Create final dataframe
    df_climatology = gdf_countries[['loc_id', 'loc_name']]
    df_climatology = df_climatology.drop_duplicates()
    
    # Dissolve geometries by country
    gdf_dissolved = gdf_countries.dissolve(by='loc_name')
    gdf_dissolved = gdf_dissolved.reset_index()
    
    return gdf_dissolved, df_climatology