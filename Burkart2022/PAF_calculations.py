'''
This script contains the functions to run the mortality module
'''

import pandas as pd
import numpy as np
import scipy as sp
import read_files as rf
from dataclasses import dataclass
import xarray as xr


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



###---------------------------------------------------------------------------------------------
### Generate daily temperature data for one year and one C-Category --- Interpolate within months and adding noise

def get_montlhy_nc(ccategory, year, month_start, month_end, cc_path_mean, gcm_diff, gcm_start, gcm_end):
    
    ### Use IMAGE land interpolation method
    nsat_month_cc = gcm_start + (cc_path_mean.at[ccategory, f'{year}'] / gcm_diff) * gcm_end
    
    ### Slice selected months and delete empty variable  "time"
    return nsat_month_cc.sel(NM=slice(month_start, month_end)).mean(dim='time', skipna=True)


def daily_temp_interp(ccategory, year, cc_path_mean, gcm_diff, gcm_start, gcm_end, noise, noise_leap):
    
    ### Apply get_montlhy_nc function to near years
    nsat_month_cc = get_montlhy_nc(ccategory, year, 1, 12, cc_path_mean, gcm_diff, gcm_start, gcm_end)
    nsat_month_cc_pre = get_montlhy_nc(ccategory, year-1, 12, 12, cc_path_mean, gcm_diff, gcm_start, gcm_end) if year > 2000 else nsat_month_cc.sel(NM=1)
    nsat_month_cc_pos = get_montlhy_nc(ccategory, year+1, 1, 1, cc_path_mean, gcm_diff, gcm_start, gcm_end) if year < 2100 else nsat_month_cc.sel(NM=12)

    ### concat data in one xarray
    dec_year_jan = xr.concat([nsat_month_cc_pre, nsat_month_cc, nsat_month_cc_pos], dim='NM')

    ### Generate monthly dates
    monthly_time = pd.date_range(start=f'15/12/{year-1}', end=f'15/2/{year+1}', freq='ME') - pd.DateOffset(days=15)
    
    ### Change NM data to monthly data and rename variable
    dec_year_jan = dec_year_jan.assign_coords(NM=monthly_time).rename({'NM': 'time'})
    
    ### Interpolation (slinear for now)
    dec_year_jan_daily = dec_year_jan.resample(time='1D').interpolate('slinear')
    
    ### Make Antarctica values NaN
    dec_year_jan_daily = dec_year_jan_daily.where(dec_year_jan_daily.latitude >= -60, np.nan)
    
    ### Keep only data within the selected year in numpy array format
    daily_1year = dec_year_jan_daily.sel(time=slice(f'{year}/1/1', f'{year}/12/31')).tas.values#.round(1)
    
    ### Add random noise
    
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        noise = noise_leap
        num_days = 366
    else:
        noise = noise
        num_days = 365
    
    ### Shuffle noise array
    np.random.shuffle(noise)
    ### Add to daily temperature
    daily_1year += noise
    
    print(f'{ccategory} - {year} daily temperatures generated')

    return daily_1year, num_days


def create_noise(std_value):
    
    np.random.seed(0)
    noise_leap = np.random.normal(scale=std_value, size=(720, 1440, 366))#.round(1) 
    ### Remove noise for 29th of February
    noise = np.delete(noise_leap, 59, axis=2)  
    
    print('Temperature noise generated')

    return noise, noise_leap


### ---------------------------------------------------------------------------------------------
### Read and process daily ERA5 Reanalysis data

def daily_temp_era5(year, pop_ssp, to_array=False):
    
    # Read file and shift longitude coordinates
    era5_daily = xr.open_dataset(f'X:\\user\\liprandicn\\Data\\ERA5\\t2m_daily\\era5_t2m_mean_day_{year}.nc')
    
    # Shift longitudinal coordinates
    era5_daily = era5_daily.assign_coords(longitude=((era5_daily.coords['longitude'] + 180) % 360 - 180)).sortby("longitude")
    
    # Convert to Celsius 
    era5_daily -= 273.15
    
    # Match grid with population data
    era5_daily = era5_daily.interp(longitude=pop_ssp.longitude, latitude=pop_ssp.latitude)
    
    # Swap axes to match required format
    if to_array:
        daily_temp = era5_daily.t2m.values.swapaxes(1,2).swapaxes(0,2)
    else: 
        daily_temp = era5_daily.drop_vars('number')
    
    # Define num_days for leap year/non-leap year
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        num_days = 366
    else:
        num_days = 365
        
    print(f'ERA5 {year} daily temperatures imported')
    
    return daily_temp, num_days



def rr_paf_rr(df, rr_year, diseases, year, region, temp_type):
    
    # Convert the RR to PAF
    df[[f'{col}' for col in diseases]] = np.where(df[diseases] < 1, 
                                                  df['population'].values[:, None] * (df[diseases] - 1),
                                                  df['population'].values[:, None] * (1 - 1 / df[diseases]))
    
    # Aggregate PAFs
    df_aggregated = df.sum(axis=0)
    
    # Convert aggregated PAF to RR and locate in annual RR dataframe
    rr_year.loc[region, (year, diseases, temp_type)] = [1/(1- df_aggregated[f'{d}']) for d in diseases]
    
    
    
    
    
    
def rr_to_paf(df, rr_year, diseases, year, region, temp_type):
    
    # Convert the RR to PAF
    df[[f'{col}' for col in diseases]] = np.where(df[diseases] < 1, 
                                                  df['population'].values[:, None] * (df[diseases] - 1),
                                                  df['population'].values[:, None] * (1 - 1 / df[diseases]))
    
    # Aggregate PAFs
    df_aggregated = df.sum(axis=0)
    
    # Locate aggregated PAF in annual dataframe
    rr_year.loc[region, (year, diseases, temp_type)] = [df_aggregated[f'{d}'] for d in diseases]
    
    

def get_temp_array_from_mask(daily_temp, valid_mask, pop_array, num_days):
    
    '''
    Creates a 1-D array from the daily temperature data by masking first cells with
    population and the flattening
    '''
    
    # Create an empty array to store the daily temperatures
    dayTemp_array = np.empty(len(pop_array), dtype=np.float32) 
    index = 0
    # Iterate over the number of days in the year
    for day in range(num_days):
        # Get the daily temperature
        dayTemp_np = daily_temp[:,:,day]
        # Mask the values to get only the ones with POP data
        dayTemp_values = dayTemp_np[valid_mask]
        # Append the values to the array
        dayTemp_array[index:index+len(dayTemp_values)] = dayTemp_values
        index += len(dayTemp_values) # or len(pop_array)
        
    return dayTemp_array
    
    
    
def get_array_from_mask(data, valid_mask, num_days):
    
    ''' 
    Converts GREG, yearly population and temperature zone xarrays into 1-D numpy arrays 
    by keeping only the entries where there is population data (data is more than 0)
    '''
    
    # Convert xarray to numpy and get the values for the valid mask
    data_masked = data[valid_mask]
    # Repeat the same values for the number of days in a year
    data_array = np.concatenate([data_masked] * num_days)
    
    return data_array
    
    
    
def get_data_masked_per_region(valid_mask, num_days, pop, era5_tz, daily_temp, tmrel): 
    
    '''
    Use the mask for the population data to mask the population temperature zone, tmrel map and 
    daily temperature data. The first three maps are repreated 365/366 times depending the 
    number of days in the specific year. This process creates 1-D arrays for the data representing
    the different combinations.
    '''
    
    # Get arrays for the data using the functions defined above
    pop_array = get_array_from_mask(pop, valid_mask, num_days)
    meanTemp_array = get_array_from_mask(era5_tz, valid_mask, num_days)
    dayTemp_array = get_temp_array_from_mask(daily_temp, valid_mask, pop_array, num_days)
    tmrel_array = get_array_from_mask(tmrel, valid_mask, num_days)
    
    # print(f'Data masked')
    
    return pop_array, meanTemp_array, dayTemp_array, tmrel_array
    
    

def create_population_df(mask, pop_ssp_year, temperature_zones, daily_temp, tmrel, num_days, min_dict, max_dict, single_erf=False):
    
    '''
    Create a dataframe that includes data on temperature zone, daily temperature, population and tmrel
    for all the grid cells with population data in a given IMAGE region.
    The dataframe is grouped per temperature_zone and daily_temperature, calculating the fraction of
    population per each combination.
    The daily temperature values are truncated according to the min and max values available in the ERF.
    This dataframe will be used to merge with the ERF and calculate the RR and PAF.    
    '''
    
    population_array, temperature_zones_array, daily_temperatures_array, tmrel_array = get_data_masked_per_region(mask, 
                                                                                                                  num_days, 
                                                                                                                  pop_ssp_year, 
                                                                                                                  temperature_zones, 
                                                                                                                  daily_temp, 
                                                                                                                  tmrel)
    
    #Change array type for posterior merging
    daily_temperatures_array = np.array(daily_temperatures_array, dtype=np.float64)
    
    # Create a dataframe that includes data on temperature zone, daily temperature, population and tmrel
    df_pop = pd.DataFrame({'temperature_zone': temperature_zones_array, 'daily_temperature': np.round(daily_temperatures_array,1),
                           'population': np.round(population_array,1), 'tmrel':np.round(tmrel_array,1)})

    # Truncate min and max Temperature values according to availability in ERF
    df_pop['daily_temperature'] = df_pop['daily_temperature'].clip(lower=df_pop['temperature_zone'].map(min_dict), 
                                                                   upper=df_pop['temperature_zone'].map(max_dict))
    
    if single_erf == False:
        # Group per temperature_zone, daily_temperature and tmrel, calculating the fraction of population per each combination
        df_pop = df_pop.groupby(['temperature_zone', 'daily_temperature', 'tmrel'], as_index=False).sum()
    
    else:
        # Group per daily_temperature and tmrel, calculating the fraction of population per each combination
        df_pop = df_pop.groupby(['daily_temperature', 'tmrel'], as_index=False).sum()
        
    df_pop['population'] /= df_pop['population'].sum()
    
    return df_pop

    
    
def get_regional_paf(pop_ssp_year, regions, region, year, num_days, temperature_zones, daily_temp, tmrel, 
                     df_erf_tmrel, rr_year, diseases, min_dict, max_dict, single_erf):
    
    '''
    Get PAF per region and year
    
    Parameters:
    - pop_ssp_year: population data for the specific year
    - regions: array with IMAGE region classification
    - region: specific IMAGE region to calculate the PAF
    - year: specific year to calculate the PAF
    - num_days: number of days in the specific year
    - temperature_zones: array with temperature zones classification
    - daily_temp: array with daily temperature data for the specific year
    - tmrel: array with TMREL data for the specific year
    - df_erf_tmrel: dataframe with the ERF data shifted by the TMREL
    - rr_year: dataframe to store the final RR values per region and year
    - diseases: list of diseases to calculate the RR
    - min_dict: dictionary with the minimum temperature values per temperature zone
    - max_dict: dictionary with the maximum temperature values per temperature zone
    '''
    
    # Get mask of 
    image_region_mask = (pop_ssp_year > 0.) & (regions == region)
    
    # Generate dataframe with population weighted factors per 
    df_pop = create_population_df(image_region_mask, pop_ssp_year, temperature_zones, 
                                daily_temp, tmrel, num_days, min_dict, max_dict)
    
    if single_erf == True:
        # Merge the ERF with the grouped data to assign rr, excluding the temperature_zone column
        df_all = pd.merge(df_pop, df_erf_tmrel,  on=['daily_temperature', 'tmrel'], how='left')
        
    else:
        # Merge the ERF with the grouped data to assign rr
        df_all = pd.merge(df_pop, df_erf_tmrel,  on=['temperature_zone', 'daily_temperature', 'tmrel'], how='left')

    # Make two new dataframes separating the cold and hot attributable deaths
    df_cold = df_all[df_all['daily_temperature'] < df_all['tmrel']].copy()
    df_hot = df_all[df_all['daily_temperature'] > df_all['tmrel']].copy()
        
    for df, temp_type in zip([df_hot, df_cold, df_all], ['hot', 'cold', 'all']):
        rr_to_paf(df, rr_year, diseases, year, region, temp_type)
        
        

def average_erf(df):
    
    '''
    Average all the columns of the Exposure Response Functions dataframe except 'daily_temperature'
    '''
    
    # Exclude the 'temperature_zone' column from averaging
    cols_to_average = df.columns.difference(['temperature_zone', 'tmrel'])
    
    # Calculate the mean for the selected columns, grouped by 'daily_temperature'
    df_mean = df[cols_to_average].groupby(df['daily_temperature']).transform('mean')
    
    # Combine the mean values with the 'temperature_zone' and 'daily_temperatures' columns
    df_mean['tmrel'] = df['tmrel']
    df_mean['daily_temperature'] = df['daily_temperature']
    
    df_mean = df_mean.drop_duplicates().reset_index(drop=True)
        
    
    return df_mean



def divide_by_tmrel(group, diseases):
    
    '''
    This function works per temperature zone groups. It locates the row whose daily temperature equals 
    the TMREL and divides this row for the rest of them
    '''    
    
    # Locate the rows whose daily temperature equals the TMREL
    fila_tmrel = group.loc[group['daily_temperature'] == group['tmrel'].iloc[0]]
    
    # Select first row
    reference = fila_tmrel.iloc[0][diseases]
    
    # Divide to shift the RR vertically
    group[diseases] = group[diseases] / reference
    
    return group



def shift_rr(df_erf, df_tz_tmrel, diseases):
    
    '''
    For every temperature zone, the merging assings all the possible TMREL. 
    This implies that we will have repeated rows for the daily temperature and relative risks 
    '''

    # Merge df_tz_tmrel with ERF data
    df_erf_tmrel = pd.merge(df_erf, df_tz_tmrel, on=['temperature_zone'], how='left') 
    
    # Set temperature_zone as index
    df_erf_tmrel = df_erf_tmrel.set_index('temperature_zone')
    
    # For each temperature zone, divide the RR by the TMREL
    df_erf_tmrel = df_erf_tmrel.groupby('temperature_zone', group_keys=False).apply(
        lambda group: divide_by_tmrel(group, diseases))

    # Reset index
    df_erf_tmrel = df_erf_tmrel.reset_index()

    return df_erf_tmrel



def tz_tmrel_combinations(pop_ssp, tmrel, temperature_zones):
    
    '''
    M1-1
    This will produce a dataframe with the unique combinations of temperature zones and TMREL
    '''
    
    # Get any cell with population data for the selected scenario
    mask_pop = (pop_ssp.GPOP > 0).any(dim='time')
    
    # Mask TMREL and temperature_zones arrays
    tmrel_valid_pop = tmrel[mask_pop.values]
    tz_valid_pop = temperature_zones[mask_pop.values]
    
    # Create dataframe with these data
    df_tz_tmrel = pd.DataFrame({'temperature_zone': tz_valid_pop, 'tmrel': np.round(tmrel_valid_pop,1)})
    
    # Remove duplicated rows
    df_tz_tmrel = df_tz_tmrel.drop_duplicates()
    
    return df_tz_tmrel



def linear_interp(xx, yy):
    
    '''
    Code to make linear interpolation over a DF column.
    Used to interpolate raw ln(RR) data 
    '''
    
    lin_interp = sp.interpolate.interp1d(xx, yy, kind='linear', fill_value='extrapolate')    
    return lambda zz: lin_interp(zz)  



def extrapolate_hot_cold(erf_tz, tz, erf_extrap, disease, zero_crossings, temp_lim, mode):
    
    if mode=='hot':
        
        # Choose index with last extreme
        index_peak = erf_tz.index[0] if len(zero_crossings) == 0 else erf_tz.index[zero_crossings[-1]]
        # Define interpolation with last range
        interp = linear_interp(erf_tz[index_peak:].index, erf_tz.loc[index_peak:].values)
        # Define temperature values to interpolate
        xx = np.round(
            np.linspace(erf_tz.index[-1]+0.1, temp_lim, int((temp_lim - erf_tz.index[-1])/0.1)+1), 1
        )
        
    if mode=='cold':
        
        # Choose index with first extreme
        index_peak = erf_tz.index[-1] if len(zero_crossings) == 0 else erf_tz.index[zero_crossings[0]]
        # Define interpolation with first range
        interp = linear_interp(erf_tz[:index_peak].index, erf_tz.loc[:index_peak].values)
        # Define temperature values to interpolate
        xx = np.round(
            np.linspace(temp_lim, erf_tz.index[0], int((erf_tz.index[0] - temp_lim)/0.1)+1), 1
        )
        
    yy = interp(xx)
    xx_multiindex = pd.MultiIndex.from_product([[tz], xx])
    erf_extrap.loc[xx_multiindex, disease] = yy



def extrapolate_erf(erf, temp_max, temp_min):
    
    # Round index level 1 to one decimal
    erf.index = pd.MultiIndex.from_arrays([erf.index.get_level_values(0), 
                                           erf.index.get_level_values(1).round(1)])

    # Define new dataframe to store original and extrapolated values
    erf_extrap = pd.DataFrame(index=pd.MultiIndex.from_product([range(6,29), np.round(np.arange(temp_min, temp_max+0.1, 0.1),1)],
                                                                names=["annual_temperature", "daily_temperature"]), 
                              columns=erf.columns)
    
    # Asign original values
    erf_extrap.loc[erf.index, erf.columns] = erf
    
    # Iterate over temperature zones and disease
    for tz in erf_extrap.index.levels[0]:
        for disease in erf.columns:
            
            # Select relevant column
            erf_tz = erf.loc[tz, disease].dropna()
            
            # Take derivative of selected series to find local extremes
            dy = np.gradient(erf_tz, erf_tz.index)
            zero_crossings = np.where(np.diff(np.sign(dy)) != 0)[0]
            
            if erf_tz.index[-1] < temp_max:
                extrapolate_hot_cold(erf_tz, tz, erf_extrap, disease, zero_crossings, temp_max, 'hot')
                
            if erf_tz.index[0] > temp_min:
                extrapolate_hot_cold(erf_tz, tz, erf_extrap, disease, zero_crossings, temp_min, 'cold')
            
    return erf_extrap  



def get_erf_dataframe(wdir, extrap_erf=False, temp_max=None, temp_min=None, all_diseases=True, 
                      mean=True, random_draw=True, draw=None):
    
    '''Get a single erf draw according to the arguments of the function, the function either:
    - Mean: Calculates the mean of all draws 
    - random_draw: Selects a random draw between the 1000 available
    - draw: Select a specific draw for all diseases (useful for propagation of uncertainty runs)
    
    The function:
    1. Selects a draw or calculates the mean per disease and puts them in a single dataframe
    2. Uses the np.exp function to convert original ln(RR) to RR
    3. Renames temperature zone columns
    4. Produces two dics corresponding to the max and min daily temperatures in each 
    temperature zone available in the files. This serves to clip later on the daily T data
    and could potentially change in the future if the ERFs are extrapolated.
    5. Put dataframe entries in float64 format
    6. Fills any NaN values
    
    Returns: 
    1. Dataframe with temperature_zone, daily_temperature as Multiindex, and 
    the diseases in the columns
    2. Dicionaries with max and min daily temperatures per temperature zone
    '''
        
    erf_dict, disease_list = rf.read_erf_data(wdir, all_diseases=all_diseases)
    
    # Choose disease with the largest daily temperature range to create the base dataframe
    
    # erf = pd.DataFrame(index=erf_dict['ckd'].index)
    erf = pd.DataFrame(index = pd.MultiIndex.from_arrays([erf_dict['ckd'].index.get_level_values(0), 
                                                          erf_dict['ckd'].index.get_level_values(1).round(1)]))
    
    # Select a random draw if none is provided...
    if random_draw:
        draw = random.randint(0,999)
        
    # ... or fill the dataframe with the selected draw or the mean of all draws
    for disease in disease_list:
        if mean:
            erf[disease] = erf_dict[disease].mean(axis=1)

        else:
            erf[disease] = erf_dict[disease][f'draw_{draw}']  
     
    # Extrapolate ERF 

    if extrap_erf == True:
        print('Extrapolating ERFs...')
        erf = extrapolate_erf(erf, temp_max, temp_min) 
         
    else:
        pass     
    
    # Convert log(rr) to rr   
    erf = erf.astype(float)   
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
    # This "flattens" the curves !!!
    erf = erf.groupby('temperature_zone').bfill().ffill()

    # Keep only temperature zones as index
    erf = erf.reset_index().set_index('temperature_zone')
    
    print('ERF dataframe generated')
            
    return erf, disease_list, min_dict, max_dict

    
    
@dataclass
class LoadResults:
    pop_ssp: any
    regions: any
    regions_range: any
    temperature_zones: any
    tmrel: any
    df_erf_tmrel: any
    diseases: any
    min_dict: dict
    max_dict: dict


def load_main_files(wdir, ssp, years, region_class, single_erf=False, extrap_erf=False, temp_max=None, temp_min=None,
                    all_diseases=False, mean=True, random_draw=False, draw=None) -> LoadResults:
    
    '''
    Load all the necessary files to run the main model, including:
    - Population data netcdf file for the selected SSP scenario
    - IMAGE regions netcdf file
    - Temperature zones netcdf file
    - TMREL netcdf file for the selected year (default 2020)
    - Exposure Response Function files for the relevant diseases, with the option to get the mean of draws, 
      a random draw or a specific draw (latest is useful for uncertainty analysis)
    '''
    
    # Load nc files that contain the temperature zones
    temperature_zones = rf.read_temperature_zones(wdir)
    
    # Load nc files that contain the region classification selected
    regions, regions_range = rf.read_region_classification(wdir, region_class)
    
    # Load population nc file of the selected scenario
    pop_ssp = rf.get_annual_pop(wdir, ssp, years)
    
    # Load Exposure Response Function files for the relevant diseases
    df_erf, diseases, min_dict, max_dict = get_erf_dataframe(
        wdir, extrap_erf, temp_max, temp_min, all_diseases, mean, random_draw, draw)
    
    # Load file with optimal temperatures for 2020 (default year)
    tmrel = rf.get_tmrel_map(wdir, 2020, mean, random_draw, draw)
    
    # Generate dataframe with RR shifted by the TMREL
    df_erf_tmrel = shift_rr(df_erf, tz_tmrel_combinations(pop_ssp, tmrel, temperature_zones), diseases)
    
    if single_erf == True:
        df_erf_tmrel = average_erf(df_erf_tmrel)
    
    return LoadResults(
        pop_ssp=pop_ssp,
        regions=regions,
        regions_range=regions_range,
        temperature_zones=temperature_zones,
        tmrel=tmrel,
        df_erf_tmrel=df_erf_tmrel,
        diseases=diseases,
        min_dict=min_dict,
        max_dict=max_dict
    )



def run_main_with_ccategory(wdir, ssp, years, region_class, ccategories, std_value, single_erf=False, 
                            extrap_erf=False, temp_max=None, temp_min=None,
                            all_diseases=True, mean=True, random_draw=False, draw=None):
    
    '''
    Run the main model using artificial daily temperature data generated with C-category 
    scenarios
    
    Parameters:
    - years: list of years to run the model
    - ssp: string with the name of the SSP scenario
    - single_erf: boolean, if True use a single mean ERF for all temperature zones
    - all_diseases: boolean, if True use all diseases, if False use only metabolic and cardiovascular diseases
    - mean: boolean, if True use the mean ERF, if False use a random draw or a specific draw
    - random_draw: boolean, if True use a random draw, if False use a specific draw
    - draw: integer, if not None use a specific draw (between 0 and 999)
    '''
    
    disease_part = "all-dis" if all_diseases else "8dis"
    erf_part = "1erf" if single_erf else ""
    extrap_part = "extrap" if extrap_erf else ""
    years_part = f"{years[0]}-{years[-1]}"

    # Load nc files that serve to generate artificial daily temperature
    gcm_diff, gcm_start, gcm_end, cc_path_mean = rf.read_climate_data(wdir)

    # Generate the daily temperature data variability
    noise, noise_leap = create_noise(std_value)
    
    #pop_ssp, image_regions, temperature_zones, tmrel, df_erf_tmrel, diseases, min_dict, max_dict 
    res = load_main_files(wdir, ssp, years, single_erf, extrap_erf, temp_max, temp_min, 
                          all_diseases, mean, random_draw, draw)
    
    for ccategory in ccategories:   
    
        # Create final dataframe
        rr_year = pd.DataFrame(index=res.regions_range, 
                               columns=pd.MultiIndex.from_product([years, res.diseases, ['cold', 'hot', 'all']]))  

        for year in years:
            
            daily_temp, num_days = daily_temp_interp(ccategory, year, cc_path_mean, gcm_diff, gcm_start, gcm_end, 
                                                         noise, noise_leap)

            # Select population for the corresponding year
            pop_ssp_year = res.pop_ssp.sel(time=f'{year}').mean('time').GPOP.values

            # Set a mask of pixels for each region
            for region in res.regions_range:
                
                get_regional_paf(pop_ssp_year, res.image_regions, region, year, num_days, res.temperature_zones, 
                                 daily_temp, res.tmrel, res.df_erf_tmrel, rr_year, res.diseases, res.min_dict, 
                                 res.max_dict, single_erf)
                    
            print(f'Year {year} done') 
            
    # Save the results and temperature statistics
    rr_year.to_csv(f'{wdir}\\output\\paf_ar6_{region_class}_{disease_part}_{erf_part}_{extrap_part}_{years_part}.csv')  

        
        
def run_main_with_era5(wdir, ssp, years, region_class, single_erf=False, extrap_erf=False, temp_max=None, 
                       temp_min=None, all_diseases=False, mean=True, random_draw=False, draw=None):
    
    '''
    Run the main model using ERA5 historical data
    
    Parameters:
    - years: list of years to run the model
    - ssp: string with the name of the SSP scenario
    - single_erf: boolean, if True use a single mean ERF for all temperature zones
    - all_diseases: boolean, if True use all diseases, if False use only metabolic and cardiovascular diseases
    - mean: boolean, if True use the mean ERF, if False use a random draw or a specific draw
    - random_draw: boolean, if True use a random draw, if False use a specific draw
    - draw: integer, if not None use a specific draw (between 0 and 999)
    '''
    
    #pop_ssp, image_regions, temperature_zones, tmrel, df_erf_tmrel, diseases, min_dict, max_dict 
    res = load_main_files(wdir, ssp, years, region_class, single_erf, extrap_erf, temp_max, temp_min, 
                          all_diseases, mean, random_draw, draw)

    default_years = range(1980, 2025)
    
    # Check if the years are within the default range, if not, use the default range
    if any(year not in default_years for year in years):
        years = default_years
        
    # Create final dataframe
    rr_year = pd.DataFrame(index=res.regions_range, 
                           columns=pd.MultiIndex.from_product([years, res.diseases, ['cold', 'hot', 'all']]))  

    for year in years:
        
        daily_temp, num_days = daily_temp_era5(year, res.pop_ssp, to_array=True)

        # Select population for the corresponding year
        pop_ssp_year = res.pop_ssp.sel(time=f'{year}').mean('time').GPOP.values

        # Set a mask of pixels for each region
        for region in res.regions_range:
            
            get_regional_paf(pop_ssp_year, res.regions, region, year, num_days, res.temperature_zones, 
                             daily_temp, res.tmrel, res.df_erf_tmrel, rr_year, res.diseases, res.min_dict, 
                             res.max_dict, single_erf)
                
        print(f'Year {year} done') 
        
    disease_part = "_all-dis" if all_diseases else "_8dis"
    erf_part = "_1erf" if single_erf else ""
    extrap_part = "_extrap" if extrap_erf else ""
    years_part = f"_{years[0]}-{years[-1]}"
            
    # Save the results and temperature statistics
    rr_year.to_csv(f'{wdir}\\output\\paf_era5_{region_class}{disease_part}{erf_part}{extrap_part}{years_part}.csv')  



def run_main(wdir, ssp, years, region_class, ccategories, std_value, single_erf=False, 
             extrap_erf=False, temp_max=None, temp_min=None, all_diseases=True, mean=True, 
             random_draw=False, draw=None):
    
    '''
    Run main function to calculate PAFs from non-optimal temperarure using either ERA5 data 
    or C-categories and climate variability
    '''
    
    if len(ccategories) == 0 and (std_value is None):
        
        print(f"Calculating PAFs with ERA5 data and {region_class} regions")   
        run_main_with_era5(wdir, ssp, years, region_class, single_erf, extrap_erf, temp_max, temp_min,
                           all_diseases, mean, random_draw, draw)
        
    elif len(ccategories) == 0 and (std_value is not None):
        print("Warning! You selected climate variability but no C-category. Please either select at least one C-category or leave it blank.")
        
    elif len(ccategories) > 0 and (std_value is None):
        print("Warning! You selected C-categories but no climate variability. Please either select at least one std value or set None.")
        
    else:
        print(f"Calculating PAFs with C-categories, climate variability and {region_class} regions")
        run_main_with_ccategory(wdir, ssp, years, region_class, ccategories, std_value, single_erf, 
                            extrap_erf, temp_max, temp_min, all_diseases, mean, random_draw, draw)
        
    print('PAFs calculation done')