'''
This script contains the functions to run the mortality module
'''

import pandas as pd
import numpy as np
import scipy as sp
import xarray as xr
from dataclasses import dataclass



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


    
def calculate_paf(daily_temp, opt_temp, min_val, max_val, risk_function, num_days, pop, regions, regions_range, final_paf, year):
    
    '''
    Calculate Population Attributable Fraction (PAF) 
    '''

    # Calculate baseline temperature
    baseline_temp = daily_temp - opt_temp[:,:,np.newaxis]
    
    # Clip baseline temperatures to min and max values
    clip_baseline_temp = np.round(np.clip(baseline_temp*10, min_val, max_val), 0).astype(int)

    # Prepare risk function lookup arrays
    temps = risk_function["baseline_temperature"].to_numpy()
    risks = risk_function["relative_risk"].to_numpy()
    
    # Create lookup array
    min_t = temps.min()
  
    # Get relative risks from risks lookup table
    relative_risks = risks[(clip_baseline_temp - min_t).astype(int)]
    
    # Calculate PAFs
    pafs = np.where(relative_risks < 1, 0, 1 - 1/relative_risks)
    
    for mode in ['all', 'hot', 'cold']:
        final_paf.loc[:,(year, mode)] = weight_avg_region(pafs, num_days, pop, regions, regions_range, mode, clip_baseline_temp)



def daily_temp_era5(era5_dir, year, pop_ssp, to_array=False):
    
    '''
    Read daily ERA5 temperature data for a specific year, shift longitude coordinates,
    convert to Celsius, and match grid with population data.
    '''
    
    # Read file and shift longitude coordinates
    era5_daily = xr.open_dataset(era5_dir+f'\\era5_t2m_mean_day_{year}.nc')
    
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
        
    print(f'[2.1] ERA5 {year} daily temperatures imported')
    
    return daily_temp, num_days



def load_optimal_temperatures(wdir):
    
    '''
    Load the optimal temperatures netcdf file calculated for a predefiend period (1980-2010) and 
    return as numpy array.
    '''
    
    # Load file with optimal temperatures for 1980-2010 period (default period)
    optimal_temps = xr.open_dataset(f'{wdir}\\data\\optimal_temperature\\era5_t2m_mean_1980-2010_p84.nc')
    
    # Convert to numpy array
    optimal_temps = optimal_temps.t2m_p84.values
    
    print('[1.4] Optimal temperatures loaded')
    
    return optimal_temps



def log_linear_interp(xx, yy):
    
    '''
    Code to make linear interpolation over a DataFrame column.
    Used to interpolate relative risk data 
    '''
    
    lin_interp = sp.interpolate.interp1d(xx, np.log(yy), kind='linear', fill_value='extrapolate')    
    
    return lambda zz: np.exp(lin_interp(zz))
    


def extrapolate_erf(erf, temp_max):
    
    # Round index to one decimal
    erf['baseline_temperature']= erf['baseline_temperature'].round(1)
    
    zero_index = erf.index[erf['baseline_temperature']==0.][0]
            
    # Define interpolation with last range
    interp = log_linear_interp(erf['baseline_temperature'].loc[zero_index:].values, 
                               erf['relative_risk'].loc[zero_index:].values)
    
    # Define temperature values to interpolate
    xx = np.round(
        np.linspace(erf['baseline_temperature'].iloc[-1]+0.1,
                    temp_max, 
                    int((temp_max - erf['baseline_temperature'].iloc[-1])/0.1)+1), 
        1
    )
    
    yy = interp(xx)
    
    erf_extrap = pd.DataFrame({
        'baseline_temperature': xx,
        'relative_risk': yy
        })
    
    erf_extrap = pd.concat([erf, erf_extrap], ignore_index=True)
            
    return erf_extrap 



def get_risk_function(wdir, extrap_erf=False, temp_max=None):
    
    ''' Load the single risk function from Honda et al. (2014)
    
    The function:
    1. Loads the file
    2. Uses the np.exp function to convert original ln(RR) to RR
    3. Defines the min and max temperature dictionaries
    5. Put dataframe entries in float64 format
    
    Returns: 
    1. Dataframe with daily_temperature and correspnoding RR values
    2. min and max temperature values
    '''
        
    risk_function = pd.read_csv(wdir+'\\data\\risk_function\\dummy\\interpolated_dataset.csv')
         
    # Convert log(rr) to rr   
    risk_function = risk_function.astype(float)   
    # risk_function = risk_function.apply(lambda x: np.exp(x))
    
    # Extrapolate risk_function
    if extrap_erf == True:
        print('Extrapolating ERFs...')
        risk_function = extrapolate_erf(risk_function, temp_max) 
         
    else:
        pass     
    
    # Prepare risk function for lookup, convert entries to int and multiply by 10
    risk_function['baseline_temperature'] = (risk_function['daily_temperature']*10).astype(int)
    
    # Perform groupby operation using the columns
    min_val = risk_function['baseline_temperature'].min()   
    max_val = risk_function['baseline_temperature'].max()
    
    print('[1.3] Risk function loaded')
            
    return risk_function, min_val, max_val
    
    
    
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
    Read scenario-dependent population data, interpolate it to yearly data, reduce resolution to 15 min,
    and select years range if provided.
    '''
    
    # Map scenario name and open selected scenario file 
    ssp = map_ssp(scenario)
    pop = xr.open_dataset(f'{wdir}\\data\\socioeconomic_Data\\population\\GPOP\\GPOP_{ssp}.nc')
    
    ### Reduce resolution to 15 min to match ERA5 data
    pop_coarse = pop.coarsen(latitude=3, longitude=3, boundary='pad').sum(skipna=False)
    
    # Linearly interpolate to yearly data
    yearly_data = pd.date_range(start='1970/01/01', end='2100/01/01', freq='YS')
    pop_yearly = pop_coarse.interp(time=yearly_data)
    
    # Select years if provided
    if years:
        pop_yearly = pop_yearly.sel(time=slice(f'{years[0]}-01-01', f'{years[-1]}-01-01'))
            
    print(f'[1.2] {scenario} population NetCDF file loaded')
    
    return pop_yearly



def read_region_classification(wdir, region_class):
    
    '''
    Load the region classification selected (IMAGE26 or GBD_level3)
    '''
    
    ### Import ERA5 temperature zones
    era5_tz = xr.open_dataset(f'{wdir}\\data\\optimal_temperature\\era5_t2m_mean_1980-2010_p84.nc')
    
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
    
    print(f'[1.1] {region_class} regions loaded')
    
    # Set negative values to 0
    region_nc = np.maximum(region_nc, 0)  
   
    return region_nc, regions_range



@dataclass
class LoadResults:
    pop_ssp: any
    regions: any
    regions_range: any
    opt_temp: any
    risk_function: any
    min_val: any
    max_val: any
    final_paf: any



def load_main_files(wdir, ssp, years, region_class, extrap_erf=False, temp_max=None
                    ) -> LoadResults:
    
    '''
    Load all the necessary files to run the main model, including:
    - Population data netcdf file for the selected SSP scenario
    - IMAGE regions netcdf file or GBD level 3 regions netcdf file
    - Optimal temperatures netcdf file
    - Risk function dataframe from Honda et al. (2014)
    '''
    
    print('[1] Loading main files...')
    
    # Load nc files that contain the region classification selected
    regions, regions_range = read_region_classification(wdir, region_class)
    
    # Load population nc file of the selected scenario
    pop_ssp = get_annual_pop(wdir, ssp, years)
    
    # Load Exposure Response Function files for the relevant diseases
    risk_function, min_val, max_val = get_risk_function(wdir, extrap_erf, temp_max)
    
    # Load file with optimal temperatures for 2020 (default year)
    optimal_temperatures = load_optimal_temperatures(wdir)
    
    # Create final dataframe
    final_paf = pd.DataFrame(index=regions_range, 
                           columns=pd.MultiIndex.from_product([years, ['cold', 'hot', 'all']]))  
    
    return LoadResults(
        pop_ssp=pop_ssp,
        regions=regions,
        regions_range=regions_range,
        opt_temp=optimal_temperatures,
        risk_function=risk_function,
        min_val=min_val,
        max_val=max_val,
        final_paf=final_paf
    )


 
def run_main(wdir, era5_dir, ssp, years, region_class, extrap_erf=False, temp_max=None):
    
    '''
    Run the main model using ERA5 historical data
    
    Parameters:
    - wdir: working directory
    - era5_dir: directory where ERA5 daily temperature data is stored
    - years: list of the period in which the model will be run
    - ssp: SSP scenario name
    - region_class: region classification to use ('IMAGE26' or 'GBD_level3')
    - extrap_erf: boolean, if True extrapolate risk functions, if False use original one
    - temp_max: maximum temperature to extrapolate to 
    - temp_min: minimum temperature to extrapolate to 
    '''
    
    #pop_ssp, image_regions, temperature_zones, tmrel, df_erf_tmrel, diseases, min_dict, max_dict 
    res = load_main_files(wdir, ssp, years, region_class, extrap_erf, temp_max)
    
    print('[2] Running main model...')

    for year in years:
        
        daily_temp, num_days = daily_temp_era5(era5_dir, year, res.pop_ssp, to_array=True)

        # Select population for the corresponding year and convert to numpy array with non-negative values
        pop_ssp_year = np.clip(res.pop_ssp.sel(time=f'{year}').mean('time').GPOP.values, 0, None)
        
        calculate_paf(daily_temp, res.opt_temp, res.min_val, res.max_val, res.risk_function, num_days, 
                      pop_ssp_year, res.regions, res.regions_range, res.final_paf, year)

        print(f'[2.2] Population Attributable Fraction calculated for {year}') 
        
    extrap_part = "_extrap" if extrap_erf else ""
    years_part = f"_{years[0]}-{years[-1]}"
            
    # Save the results and temperature statistics
    res.final_paf.to_csv(f'{wdir}\\output\\paf_era5_{region_class}{extrap_part}{years_part}.csv')  
    
    print('[3] Results saved. Process finished.')