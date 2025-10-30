import pandas as pd
import numpy as np
import scipy as sp
import xarray as xr
from dataclasses import dataclass



def map_ssp(ssp: str) -> str:
    mapping = {
        "SSP1": "SSP1_M",
        "SSP2": "SSP2_CP",
        "SSP3": "SSP3_H",
        "SSP5": "SSP5_H"
    }
    
    # Normalize input
    ssp_normalized = ssp.strip().upper()
    
    try:
        return mapping[ssp_normalized]
    except KeyError:
        raise ValueError(f"SSP '{ssp}' not valid. Use one of the following: {list(mapping.keys())}.")



def get_annual_pop(wdir, scenario, years=None):
    
    '''
    Read scenario-dependent population data and interpolate it to yearly data
    '''
    
    ssp = map_ssp(scenario)
    pop = xr.open_dataset(f'{wdir}\\data\\socioeconomic_Data\\population\\GPOP\\GPOP_{ssp}.nc')
    ### Reduce resolution to 15 min
    pop_coarse = pop.coarsen(latitude=3, longitude=3, boundary='pad').sum(skipna=False)
    # Linearly interpolate to yearly data
    yearly_data = pd.date_range(start='1970/01/01', end='2100/01/01', freq='YS')
    pop_yearly = pop_coarse.interp(time=yearly_data)
    
    if years:
        pop_yearly = pop_yearly.sel(time=slice(f'{years[0]}-01-01', f'{years[-1]}-01-01'))
    
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



def read_region_classification(wdir, region_class):
    
    ### Import ERA5 temperature zones
    era5_tz = xr.open_dataset(f'{wdir}\\data\\temperature_zones\\ERA5_mean_1980-2019_land_t2m_tz.nc')
    
    if region_class == 'IMAGE26':
    
        ### Read in IMAGE region data and interpolate to match files resolution
        region_nc = xr.open_dataset(f'{wdir}\\data\\region_classification\\IMAGE\\GREG_30MIN.nc')
        region_nc = region_nc.interp(longitude=era5_tz.longitude, latitude=era5_tz.latitude, method='nearest').mean(dim='time') 
        ### Convert files to numpy arrays
        region_nc = region_nc.GREG_30MIN.values
        
        # Get number of regions
        regions_range = range(1,27)
        
    if region_class == 'GBD_level3':
        
        # Read in GBD LEVEL 3 region DATA
        region_nc = xr.open_dataset(f'{wdir}\\data\\region_classification\\GBD\\GBD_locations_level3.nc')
        region_nc = region_nc.interp(longitude=era5_tz.longitude, latitude=era5_tz.latitude, method='nearest')
        
        # Get number of regions
        regions_range = np.unique(region_nc.loc_id.values)[1:-1].astype(int)  # Exclude -1 and nan
        region_nc = region_nc.loc_id.values
    
    print(f'{region_class} regions loaded')
    
    # Set negative values to 0
    region_nc = np.maximum(region_nc, 0)  
    
    return region_nc, regions_range



def read_erf_data(wdir, all_diseases=True):
    
    '''
    Read the raw Exposure Response Functions of relevant diseases from the specified path.
    
    Returns:
    - erf: Dictionary containing the ERF dataframes. Relevant diseases only
    '''
    
    if all_diseases:
        disease_list = ['ckd', 'cvd_cmp', 'cvd_htn', 'cvd_ihd', 'cvd_stroke', 'diabetes', 'inj_animal', 'inj_disaster', 'inj_drowning', 
                'inj_homicide', 'inj_mech', 'inj_othunintent', 'inj_suicide', 'inj_trans_other', 'inj_trans_road', 'resp_copd', 'lri']   
    else:
        disease_list = ['ckd', 'cvd_cmp', 'cvd_htn', 'cvd_ihd', 'cvd_stroke', 'diabetes', 'lri', 'resp_copd'] 
    
    erf_dict = {}

    for disease in disease_list:
        erf_disease = pd.read_csv(f'{wdir}\\data\\exposure_response_functions\\ERF\\{disease}_curve_samples.csv', 
                                index_col=[0,1])
        erf_disease.index = pd.MultiIndex.from_arrays([erf_disease.index.get_level_values(0),
                                                    erf_disease.index.get_level_values(1).round(1)])
        erf_dict[disease] = erf_disease
        
    return erf_dict, disease_list



def get_montlhy_nc(ccategory, year, month_start, month_end, cc_path_mean, gcm_diff, gcm_start, gcm_end):
    
    '''
    Generate daily temperature data for one year and one C-Category 
    Interpolate within months and adding noise
    '''
    
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



def clip_daily_temp(daily_temp, temperature_zones, min_dict, max_dict):
    
    # Clip temperatures to valid values
    valid_mask = ~np.isnan(temperature_zones)
    tz_int = np.zeros_like(temperature_zones, dtype=int)
    tz_int[valid_mask] = temperature_zones[valid_mask].astype(int)
    
    # Create min and max daily temperature values arrays per temperature zone
    min_values = np.array(list(min_dict.values()))
    max_values = np.array(list(max_dict.values()))
    
    # Create empty min and max maps
    min_map = np.full_like(temperature_zones, np.nan, dtype=float)
    max_map = np.full_like(temperature_zones, np.nan, dtype=float)
    
    # Assign min and max values to the maps based on temperature zones
    min_map[valid_mask] = min_values[tz_int[valid_mask] - 6]
    max_map[valid_mask] = max_values[tz_int[valid_mask] - 6]
    
    # Expand to third axis to match daily_temp shape
    min_map_3d = min_map[..., np.newaxis]
    max_map_3d = max_map[..., np.newaxis]
    
    # Clip daily temperatures
    daily_temp_clipped = np.clip(daily_temp, min_map_3d, max_map_3d)
    
    return daily_temp_clipped



def daily_temp_era5(climate_dir, year, pop_ssp, temperature_zones, min_dict, max_dict, to_array=False):
    
    '''
    Read daily ERA5 temperature data for a specific year, shift longitude coordinates,
    convert to Celsius, and match grid with population data.
    '''
    
    # Read file
    era5_daily = xr.open_dataset(climate_dir+f'\\era5_t2m_mean_day_{year}.nc')
    
    # Shift longitudinal coordinates
    era5_daily = era5_daily.assign_coords(longitude=((era5_daily.longitude + 180) % 360 - 180))
    era5_daily = era5_daily.sel(longitude=np.unique(era5_daily.longitude)).sortby("longitude")
    
    # Convert to Celsius 
    era5_daily -= 273.15
    
    # Match grid with population data. Nearest neighbor interpolation
    era5_daily = era5_daily.interp(latitude=np.clip(pop_ssp.latitude, 
                                                    era5_daily.latitude.min().item(), 
                                                    era5_daily.latitude.max().item()), 
                                method='nearest')
    
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
    
    # Clip temperatures to valid values
    daily_temp = clip_daily_temp(daily_temp, temperature_zones, min_dict, max_dict)
    
    # Convert to float64 with one decimal precision
    daily_temp = np.round(daily_temp*10,0).astype(np.float64)
    
    return daily_temp, num_days

  

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



def shift_relative_risk(df_erf, df_tz_tmrel, diseases):
    
    '''
    Shift the relative risk (RR) values so that the relative risk at the TMREL equals 1 for 
    each temperature zone. The final dataframe has the same structure as the input df_erf. plus
    the TMREL column, and for each combination of temperture zone and daily temperature, there 
    can be different RR values depending on the TMREL.
    '''

    # Merge df_tz_tmrel with ERF data
    df_erf_tmrel = pd.merge(df_erf, df_tz_tmrel, on=['temperature_zone'], how='left') 
    
    # Set temperature_zone as index
    df_erf_tmrel = df_erf_tmrel.set_index('temperature_zone')
    
    # For each temperature zone, divide the RR by the RR corresponding to the TMREL
    df_erf_tmrel = df_erf_tmrel.groupby('temperature_zone', group_keys=False).apply(
        lambda group: divide_by_tmrel(group, diseases))

    # Reset index
    df_erf_tmrel = df_erf_tmrel.reset_index()
    
    df_erf_tmrel['daily_temperature'] = (df_erf_tmrel['daily_temperature']*10).astype(np.float64)
    df_erf_tmrel['tmrel'] = (df_erf_tmrel['tmrel']*10).astype(np.float64)
    df_erf_tmrel['temperature_zone'] = df_erf_tmrel['temperature_zone'].astype(np.float64)

    return df_erf_tmrel



def tz_tmrel_combinations(pop_ssp, tmrel, temperature_zones):
    
    '''
    Produces a dataframe with the unique combinations of temperature zones and TMREL
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

    # Read TMREL data, default for future projections: 2020
    tmrel = xr.open_dataset(f'{wdir}\\data\\exposure_response_functions\\TMRELs_{year}.nc')
    
    # Select between a random draw, specific draw or mean
    if random_draw:
        draw = random.randint(1,100)
    if mean:
            tmrel = tmrel.tmrel.values.mean(axis=2)
    else:
        tmrel = tmrel.sel(draw=draw).tmrel.values
        
    print('TMREL data loaded')
        
    return tmrel



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



def get_erf_dataframe(wdir, extrap_erf=False, all_diseases=True, mean=True, random_draw=True, draw=None):
    
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
        
    erf_dict, disease_list = read_erf_data(wdir, all_diseases=all_diseases)
    
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
        temp_max = 50.
        temp_min = -25.
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



def read_temperature_zones(wdir):
    
    ### Import ERA5 temperature zones
    era5_tz = xr.open_dataset(f'{wdir}\\data\\temperature_zones\\ERA5_mean_1980-2019_land_t2m_tz.nc')

    ### Convert file to numpy array
    era5_tz = era5_tz.t2m.values.astype(np.float64)

    print('Temperature zones data loaded')
    
    return era5_tz

    
    
@dataclass
class LoadResults:
    pop_ssp: any
    regions: any
    regions_range: any
    temperature_zones: any
    tmrel: any
    erf_tmrel: any
    diseases: any
    min_dict: dict
    max_dict: dict



def load_main_files(wdir, ssp, years, region_class, single_erf=False, extrap_erf=False, 
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
    temperature_zones = read_temperature_zones(wdir)
    
    # Load nc files that contain the region classification selected
    regions, regions_range = read_region_classification(wdir, region_class)
    
    # Load population nc file of the selected scenario
    pop_ssp = get_annual_pop(wdir, ssp, years)
    
    # Load Exposure Response Function files for the relevant diseases
    df_erf, diseases, min_dict, max_dict = get_erf_dataframe(
        wdir, extrap_erf, all_diseases, mean, random_draw, draw)
    
    # Load file with optimal temperatures for 2020 (default year)
    tmrel = get_tmrel_map(wdir, 2020, mean, random_draw, draw)
    
    # Generate dataframe with RR shifted by the TMREL
    df_erf_tmrel = shift_relative_risk(df_erf, tz_tmrel_combinations(pop_ssp, tmrel, temperature_zones), diseases)
    
    if single_erf == True:
        df_erf_tmrel = average_erf(df_erf_tmrel)
        
    # Convert to 0.1 degree resolution and multiply by 10
    tmrel = np.round(tmrel*10,0).astype(np.float64)
    
    return LoadResults(
        pop_ssp=pop_ssp,
        regions=regions,
        regions_range=regions_range,
        temperature_zones=temperature_zones,
        tmrel=tmrel,
        erf_tmrel=df_erf_tmrel,
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
    gcm_diff, gcm_start, gcm_end, cc_path_mean = read_climate_data(wdir)

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

        
        
def run_main_with_era5(wdir, climate_dir, ssp, years, region_class, single_erf=False, extrap_erf=False, 
                    all_diseases=False, mean=True, random_draw=False, draw=None):
    
    '''
    Run the main model using ERA5 historical data (1980-2024)
    
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
    res = load_main_files(wdir, ssp, years, region_class, single_erf, extrap_erf,
                        all_diseases, mean, random_draw, draw)

    default_years = range(1980, 2025)
    
    # Check if the years are within the default range, if not, use the default range
    if any(year not in default_years for year in years):
        years = default_years
        
    # Create final dataframe
    rr_year = pd.DataFrame(index=res.regions_range, 
                        columns=pd.MultiIndex.from_product([years, res.diseases, ['cold', 'hot', 'all']]))  

    for year in years:
        
        daily_temp, num_days = daily_temp_era5(climate_dir, year, res.pop_ssp, res.temperature_zones, 
                                               res.min_dict, res.max_dict, to_array=True)

        # Select population for the corresponding year and convert to numpy array with non-negative values
        pop_ssp_year = np.clip(res.pop_ssp.sel(time=f'{year}').mean('time').GPOP.values, 0, None)
        
        # Calculate Population Attributable Factor per disease for the year
        calculate_paf(pop_ssp_year, res.regions, year, num_days, res.temperature_zones, 
                            daily_temp, res.tmrel, res.erf_tmrel, rr_year, res.diseases, res.regions_range)

        print(f'Year {year} done') 
        
    disease_part = "_all-dis" if all_diseases else "_8dis"
    erf_part = "_1erf" if single_erf else ""
    extrap_part = "_extrap" if extrap_erf else ""
    years_part = f"_{years[0]}-{years[-1]}"
            
    # Save the results and temperature statistics
    rr_year.to_csv(f'{wdir}\\output\\paf_era5_{region_class}{disease_part}{erf_part}{extrap_part}{years_part}.csv')  
    


def weight_avg_region(pafs, num_days, pop, regions, regions_range, mask, clip_baseline_temp):
    
    '''
    Calculate weighted average of PAFs per region
    '''
    
    # Apply mask to PAFs to select cold, hot or all temperatures
    if mask == 'all':
        pass
    
    elif mask == 'cold':
        pafs = np.where(clip_baseline_temp<0, pafs, 0)
        
    elif mask == 'hot':
        pafs = np.where(clip_baseline_temp>0, pafs, 0) 
    
    # Aggregate PAFs over days
    pafs = np.sum(pafs, axis=2) / num_days
    
    # Flatten arrays
    regions_flat = np.nan_to_num(regions.ravel()).astype(int)
    pafs_flat = np.nan_to_num(pafs.ravel())
    pop_flat = np.nan_to_num(pop.ravel())
    
    # Calculate weighted sum of PAFs per region
    weighted_sum = np.bincount(regions_flat, weights=pafs_flat * pop_flat)
    weight_pop_sum = np.bincount(regions_flat, weights=pop_flat)
    
    # Calculate weighted average for specified regions
    weighted_avg = weighted_sum[regions_range] / np.maximum(weight_pop_sum[regions_range], 1e-12)
    
    return weighted_avg



def map_paf_disease(lookup, disease, temperature_zones, tmrel, daily_temp, final_paf, year, num_days,
                    pop_ssp_year, regions, regions_range):

    # Get levels of the MultiIndex
    level_0 = lookup.index.levels[0]
    level_1 = np.insert(lookup.index.levels[1], 4, 178.)
    level_2 = lookup.index.levels[2]

    # Create empty 3D array
    array_3d = np.full((len(level_0), len(level_1), len(level_2)), np.nan)
    
    # Fill the 3D array with values from the lookup table for a given disease
    for (i0, i1, i2), value in lookup[disease].items():
        idx0 = level_0.get_loc(i0)
        idx1 = level_1.get_loc(i1)
        idx2 = level_2.get_loc(i2)
        array_3d[idx0, idx1, idx2] = value

    # Calculate indices for temperature zones, TMREL, and daily temperatures
    tz_idx = temperature_zones - np.nanmin(temperature_zones)
    tm_idx = tmrel - np.nanmin(tmrel)
    dt_idx = daily_temp - np.nanmin(daily_temp)

    # Crear máscara de los índices válidos
    valid_mask = ~np.isnan(tz_idx) & ~np.isnan(tm_idx) & ~np.isnan(dt_idx[...,0])

    # Convertir a enteros solo los válidos
    tz_idx_int = np.zeros_like(tz_idx, dtype=int)
    tm_idx_int = np.zeros_like(tm_idx, dtype=int)
    dt_idx_int = np.zeros_like(dt_idx, dtype=int)

    tz_idx_int[valid_mask] = tz_idx[valid_mask].astype(int)
    tm_idx_int[valid_mask] = tm_idx[valid_mask].astype(int)
    dt_idx_int[valid_mask] = dt_idx[valid_mask].astype(int)

    relative_risks = np.full(daily_temp.shape, np.nan)
    relative_risks[valid_mask] = array_3d[
        tz_idx_int[valid_mask][:, np.newaxis], 
        tm_idx_int[valid_mask][:, np.newaxis], 
        dt_idx_int[valid_mask]
    ]
    
    pafs = np.where(relative_risks < 1, 0, 1 - 1/relative_risks)
        
    for mode in ['all', 'hot', 'cold']:
            final_paf.loc[:,(year, disease, mode)] = weight_avg_region(pafs, num_days, pop_ssp_year, regions,
                                                                       regions_range, mode, daily_temp)
            
    print(f'{disease}')
            
            
    
def calculate_paf(pop_ssp_year, regions, year, num_days, temperature_zones, daily_temp, tmrel, erf_tmrel, 
                final_paf, diseases, regions_range):

    lookup = (
        erf_tmrel
        .set_index(['temperature_zone', 'tmrel', 'daily_temperature'])
        .sort_index()
        )
    
    for disease in diseases:
        map_paf_disease(lookup, disease, temperature_zones, tmrel, daily_temp, final_paf, year, num_days,
                        pop_ssp_year, regions, regions_range)


    
def run_main(wdir, climate_dir, ssp, years, region_class, ccategories, std_value, single_erf=False, 
            extrap_erf=False, all_diseases=True, mean=True, random_draw=False, draw=None):
    
    '''
    Run main function to calculate PAFs from non-optimal temperarure using either ERA5 data 
    or C-categories and climate variability
    '''
    
    if len(ccategories) == 0 and (std_value is None):
        
        print(f"Calculating PAFs with ERA5 data and {region_class} regions")   
        run_main_with_era5(wdir, climate_dir, ssp, years, region_class, single_erf, extrap_erf,
                        all_diseases, mean, random_draw, draw)
        
    elif len(ccategories) == 0 and (std_value is not None):
        print("Warning! You selected climate variability but no C-category. Please either select at least one C-category or leave it blank.")
        
    elif len(ccategories) > 0 and (std_value is None):
        print("Warning! You selected C-categories but no climate variability. Please either select at least one std value or set None.")
        
    else:
        print(f"Calculating PAFs with C-categories, climate variability and {region_class} regions")
        run_main_with_ccategory(wdir, ssp, years, region_class, ccategories, std_value, single_erf, 
                            extrap_erf, all_diseases, mean, random_draw, draw)
        
    print('PAFs calculation done')