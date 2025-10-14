import xarray as xr
import pandas as pd
import numpy as np


### -----------------------------------------------------------------------------------------
### Linear interpolation of population data to yearly data

def map_ssp(ssp: str) -> str:
    mapping = {
        "SSP1": "SSP1_M",
        "SSP2": "SSP2_CP",
        "SSP3": "SSP3_H",
        "SSP5": "SSP5_H"
    }
    try:
        return mapping[ssp]
    except KeyError:
        raise ValueError(f"SSP '{ssp}' not valid. Use one of the following: {list(mapping.keys())}.")


def get_annual_pop(wdir, scenario, years=None):
    
    '''Read scenario-dependent population data and interpolate it to yearly data'''
    
    ssp = map_ssp(scenario)
    pop = xr.open_dataset(f'{wdir}\\data\\Socioeconomic_Data\\Population\\GPOP\\GPOP_{ssp}.nc')
    ### Reduce resolution to 15 min
    pop_coarse = pop.coarsen(latitude=3, longitude=3, boundary='pad').sum(skipna=False)
    # Linearly interpolate to yearly data
    yearly_data = pd.date_range(start='1970/01/01', end='2100/01/01', freq='YS')
    pop_yearly = pop_coarse.interp(time=yearly_data)
    
    if years:
        pop_yearly = pop_yearly.sel(time=slice(f'{years[0]}-01-01', f'{years[-1]}-01-01'))
    
    # if temperature_type:
    #     # Open random ERA5 file to get temperature data grid
    #     era5_xr = xr.open_dataset('X:\\user\\liprandicn\\Data\\ERA5\\t2m_daily\\era5_t2m_mean_day_2010.nc')
    #     era5_xr = era5_xr.assign_coords(longitude=((era5_xr.coords['longitude'] + 180) % 360 - 180)).sortby("longitude")
    #     # Interpolate population data to match ERA5 grid
    #     pop_yearly = pop_yearly.interp(latitude=era5_xr.latitude, longitude=era5_xr.longitude, method='nearest')
    
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

def read_climate_data(wdir):
    
    '''Import GCM data from IMAGE land and load AR6 mean pathways
    (this will be later replaced by the emulator) '''
    
    ### Import GCM data from IMAGE land 
    gcm_diff = xr.open_dataset(f'{wdir}\\data\\Climate_Data\\GCM\\ensemble_mean_ssp585_tas_15min_diff.nc')
    gcm_start = xr.open_dataset(f'{wdir}\\data\\Climate_Data\\GCM\\ensemble_mean_ssp585_tas_15min_start.nc')
    gcm_end = xr.open_dataset(f'{wdir}\\data\\Climate_Data\\GCM\\ensemble_mean_ssp585_tas_15min_end.nc')

    ### Import AR6 data 
    ar6 = pd.read_csv('X:\\user\\dekkerm\\Data\\AR6_snapshots\\AR6_snapshot_Nayeli_global.csv')
    cc_path_mean = ar6.iloc[:, 10:-1].groupby(ar6['Category']).mean()
    
    print('Climate data loaded')
    
    return gcm_diff, gcm_start, gcm_end, cc_path_mean


### -------------------------------------------------------------------------
### Read temperature zones

def read_temperature_zones(wdir):
    
    ### Import ERA5 temperature zones
    era5_tz = xr.open_dataset(f'{wdir}\\data\\Climate_Data\\ERA5\\ERA5_mean_1980-2019_land_t2m_tz.nc')

    ### Convert file to numpy array
    era5_tz = era5_tz.t2m.values
    
    print('Temperature zones data loaded')
    
    return era5_tz



### -----------------------------------------------------------------------------------------
### Load IMAGE regions and temperature zone xarray files

def read_region_classification(wdir, region_class):
    
    ### Import ERA5 temperature zones
    era5_tz = xr.open_dataset(f'{wdir}\\data\\Climate_Data\\ERA5\\ERA5_mean_1980-2019_land_t2m_tz.nc')
    
    if region_class == 'IMAGE26':
    
        ### Read in IMAGE region data and interpolate to match files resolution
        region_nc = xr.open_dataset(f'{wdir}\\data\\IMAGE_regions\\GREG_30MIN.nc')
        region_nc = region_nc.interp(longitude=era5_tz.longitude, latitude=era5_tz.latitude, method='nearest').mean(dim='time') 
        ### Convert files to numpy arrays
        region_nc = region_nc.GREG_30MIN.values
        
        # Get number of regions
        regions_range = range(1,27)
        
    if region_class == 'GBD_level3':
        
        # Read in GBD LEVEL 3 region DATA
        region_nc = xr.open_dataset(f'{wdir}\\data\\GBD_Data\\GBD_locations\\GBD_locations_level3.nc')
        region_nc = region_nc.interp(longitude=era5_tz.longitude, latitude=era5_tz.latitude, method='nearest')
        
        # Get number of regions
        regions_range = np.unique(region_nc.loc_id.values)[1:-1].astype(int)  # Exclude -1 and nan
        region_nc = region_nc.loc_id.values
    
    print(f'{region_class} regions loaded')
    
    return region_nc, regions_range


### -----------------------------------------------------------------------------------------
### Load ERF data for Method 1 (all diseases)

def open_all_diseases_data():
    
    ### List of all relevant diseases
    diseases = ['ckd', 'cvd_cmp', 'cvd_htn', 'cvd_ihd', 'cvd_stroke', 'diabetes', 'inj_animal', 'inj_disaster', 'inj_drowning', 
                'inj_homicide', 'inj_mech', 'inj_othunintent', 'inj_suicide', 'inj_trans_other', 'inj_trans_road', 'resp_copd', 'lri']

    erf = pd.read_csv(f'{wdir}\\ResponseFunctions\\erf_reformatted2\\erf_ckd.csv', index_col=0)
    erf = erf[['annual_temperature', 'daily_temperature']]
    erf = erf.set_index(['annual_temperature', 'daily_temperature'])

    for disease in diseases:
        df = pd.read_csv(f'{wdir}\\ResponseFunctions\\erf_reformatted2\\erf_{disease}.csv', index_col=[1,2])
        erf[disease] = df['rr']

    erf = erf.reset_index()
    erf.rename(columns={'daily_temperature':'dailyTemp', 'annual_temperature': 'meanTemp'}, inplace=True)

    min_dict = erf.groupby('meanTemp')['dailyTemp'].min().to_dict()
    max_dict = erf.groupby('meanTemp')['dailyTemp'].max().to_dict()
    
    print('ERF data imported')
    
    return diseases, erf, min_dict, max_dict


### -----------------------------------------------------------------------------------------
### Load ERF data for Method 2 

def open_all_diseases_data_method2():
    
    diseases = ['ckd', 'cvd_cmp', 'cvd_htn', 'cvd_ihd', 'cvd_stroke', 'diabetes', 'resp_copd', 'lri']
    
    erf = {}
    for disease in diseases:
        erf[disease] = pd.read_csv(f'{wdir}\\ResponseFunctions\\erf_reformatted\\erf_{disease}_mean.csv', header=None).to_numpy()

    return diseases, erf


    
### ------------------------------------------------------------------------------------------
### Method 1.1

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
        erf_disease = pd.read_csv(f'{wdir}\\data\\GBD_Data\\Exposure_Response_Functions\\ERF\\{disease}_curve_samples.csv', 
                                  index_col=[0,1])
        erf_disease.index = pd.MultiIndex.from_arrays([erf_disease.index.get_level_values(0),
                                                       erf_disease.index.get_level_values(1).round(1)])
        erf_dict[disease] = erf_disease
        
    return erf_dict, disease_list


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
    
    tmrel = xr.open_dataset(f'{wdir}\\data\\GBD_Data\\Exposure_Response_Functions\\TMRELs_{year}.nc')
    
    if random_draw:
        draw = random.randint(1,100)
        
    if mean:
            tmrel = tmrel.tmrel.values.mean(axis=2)

    else:
        tmrel = tmrel.sel(draw=draw).tmrel.values
        
    print('TMREL data loaded')
        
    return tmrel