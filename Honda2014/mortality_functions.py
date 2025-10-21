'''
This script contains the functions to run the mortality module
'''

import pandas as pd
import numpy as np
import scipy as sp
import xarray as xr
from dataclasses import dataclass


    
def rr_to_paf(df, rr_year, year, region, temp_type):
    
    '''
    Convert the relative risk value to the Population Attributable Fraction as in GBD methodology
    (Burkart et al., (2022)).
    Locate results in the final dataframe
    '''
    
    # Convert the RR to PAF
    df['pop_attrib_frac'] = np.where(df['relative_risk'] < 1, 
                                     df['population'] * (df['relative_risk'] - 1),
                                     df['population'] * (1 - 1 / df['relative_risk']))
    
    # Aggregate PAFs
    df_aggregated = df.sum(axis=0)
    
    # Locate aggregated PAF in annual dataframe
    rr_year.loc[region, (year, temp_type)] = df_aggregated[f'pop_attrib_frac']
    
    

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
    
    
    
def get_data_masked_per_region(valid_mask, num_days, pop, daily_temp, opt_temp): 
    
    '''
    Use the mask for the population data to mask the population temperature zone, tmrel map and 
    daily temperature data. The first three maps are repreated 365/366 times depending the 
    number of days in the specific year. This process creates 1-D arrays for the data representing
    the different combinations.
    '''
    
    # Get arrays for the data using the functions defined above
    pop_array = get_array_from_mask(pop, valid_mask, num_days)
    dayTemp_array = get_temp_array_from_mask(daily_temp, valid_mask, pop_array, num_days)
    opt_temp_array = get_array_from_mask(opt_temp, valid_mask, num_days)
    
    # print(f'Data masked')
    
    return pop_array, np.array(dayTemp_array, dtype=np.float64), opt_temp_array
    
    

def population_temperature_df(mask, pop_ssp_year, daily_temp, opt_temp, num_days):
    
    '''
    Create a dataframe that includes data on daily temperature, population and optimal temperature
    for all the grid cells with population data in a given IMAGE region.
    The dataframe output has the fraction of population per each combination.
    This dataframe will be used to merge with the ERF and calculate the RR and PAF.    
    '''
    
    population_array, daily_temps_array, opt_temp_array = get_data_masked_per_region(mask, 
                                                                                    num_days, 
                                                                                    pop_ssp_year, 
                                                                                    daily_temp, 
                                                                                    opt_temp)
    
    # Create a dataframe that includes data on temperature zone, daily temperature, population and tmrel
    df_pop = pd.DataFrame({'daily_temperature': np.round(daily_temps_array,1),
                           'population': np.round(population_array,1), 
                           'optimal_temperature':np.round(opt_temp_array,1)})

    # Group per daily_temperature and tmrel, calculating the fraction of population per each combination
    df_pop = df_pop.groupby(['daily_temperature', 'optimal_temperature'], as_index=False).sum()
    df_pop['population'] /= df_pop['population'].sum()
    
    return df_pop

    
    
def get_regional_paf(pop_ssp_year, regions, region, year, num_days, daily_temp, rr_year, 
                     opt_temp, risk_function, min_val, max_val):
    
    '''
    Get Population Attributable Fraction from cold-, hot- and overall non-optimal temperatures
    per region and year
    
    Parameters:
    - pop_ssp_year: population data for the specific year
    - regions: array with IMAGE region classification
    - region: specific IMAGE region to calculate the PAF
    - year: specific year to calculate the PAF
    - num_days: number of days in the specific year
    - daily_temp: array with daily temperature data for the specific year
    - rr_year: dataframe to store the final RR values per region and year
    - opt_temp: array with optimal temperature data
    - risk_function: dataframe with the ERF data
    - min_dict: dictionary with the minimum temperature values per temperature zone
    - max_dict: dictionary with the maximum temperature values per temperature zone
    '''
    
    # Get mask of 
    image_region_mask = (pop_ssp_year > 0.) & (regions == region)
    
    # Generate dataframe with population weighted factors per 
    temps_pop = population_temperature_df(image_region_mask, pop_ssp_year, daily_temp, opt_temp, num_days)
    
    # Shift ERF according to optimal temperature
    temps_pop['baseline_temperature'] = temps_pop['daily_temperature'] - temps_pop['optimal_temperature']
    
    # Clip baseline temperatures to min and max values
    temps_pop['baseline_temperature'] = temps_pop['baseline_temperature'].clip(lower=min_val, upper=max_val)
    
    # Round temperatures to 1 decimal for merging
    temps_pop['baseline_temperature'] = temps_pop['baseline_temperature'].round(1)
    risk_function['baseline_temperature'] = risk_function['baseline_temperature'].round(1)

    # Merge the ERF with the grouped data to assign rr, excluding the temperature_zone column
    temp_pop_rr = pd.merge(temps_pop[['population', 'baseline_temperature']], risk_function,  
                           on='baseline_temperature', how='left')

    # Make two new dataframes separating the cold and hot attributable deaths
    temp_pop_rr_cold = temp_pop_rr[temp_pop_rr['baseline_temperature'] < 0.].copy()
    temp_pop_rr_hot = temp_pop_rr[temp_pop_rr['baseline_temperature'] > 0.].copy()
        
    for df, temp_type in zip([temp_pop_rr_hot, temp_pop_rr_cold, temp_pop_rr], ['hot', 'cold', 'all']):
        rr_to_paf(df, rr_year, year, region, temp_type)



def opt_temp_combinations(pop_ssp, optimal_temperatures):
    
    '''
    This function creates a dataframe with all the unique values of optimal temperatures
    '''
    
    # Get any cell with population data for the selected scenario
    mask_pop = (pop_ssp.GPOP > 0).any(dim='time')
    
    opt_temp_valid_pop = optimal_temperatures[mask_pop.values]
    
    # Create dataframe with these data
    df_optimal = pd.DataFrame({'optimal_t': np.round(opt_temp_valid_pop,1)})
    
    # Remove duplicated rows
    df_optimal = df_optimal.drop_duplicates()
    
    return df_optimal



def daily_temp_era5(era5_dir, year, pop_ssp, to_array=False):
    
    '''
    Read daily ERA5 temperature data for a specific year, shift longitude coordinates,
    convert to Celsius, and match grid with population data.
    '''
    
    # Read file and shift longitude coordinates
    era5_daily = xr.open_dataset(era5_dir+f'\\era5_t2m_mean_day_{year}.nc')
    
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

        # Select population for the corresponding year
        pop_ssp_year = res.pop_ssp.sel(time=f'{year}').mean('time').GPOP.values

        # Set a mask of pixels for each region
        for region in res.regions_range:
            
            get_regional_paf(pop_ssp_year, res.regions, region, year, num_days, daily_temp, res.final_paf,
                             res.opt_temp, res.risk_function, res.min_val, res.max_val)
                
        print(f'[2.2] Population Attributable Fraction calculated for {year}') 
        
    extrap_part = "_extrap" if extrap_erf else ""
    years_part = f"_{years[0]}-{years[-1]}"
            
    # Save the results and temperature statistics
    res.final_paf.to_csv(f'{wdir}\\output\\excess_mortality_era5_{region_class}{extrap_part}{years_part}.csv')  
    
    print('[3] Results saved. Process finished.')