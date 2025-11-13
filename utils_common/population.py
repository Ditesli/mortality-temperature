import pandas as pd
import xarray as xr



def get_all_population_data(wdir, return_pop=False):
    
    '''
    Create a dataset with population data for all SSP scenarios 
    concatenated along a new 'ssp' dimension.
    '''
    
    ssp_scenarios = ['SSP1', 'SSP2', 'SSP3', 'SSP5']

    # Concatenate population data for all SSP scenarios
    pop_all_ssp = xr.concat(
        [get_annual_pop(wdir, ssp, years=range(1980, 2101)) for ssp in ssp_scenarios],
        dim='ssp'
    )

    # Assign ssp names to the new dimension
    pop_all_ssp['ssp'] = [s.lower() for s in ssp_scenarios]  # ['ssp1', 'ssp2', ...]
        
    # Mask with positive pop values
    mask_positive_pop = (pop_all_ssp > 0).any(dim=['time', 'ssp'])
    
    if return_pop:
        return pop_all_ssp, mask_positive_pop
    
    else:
        return mask_positive_pop



def get_annual_pop(wdir, scenario, temp_source, years=None):
    
    '''
    Read scenario-dependent population data, interpolate it to yearly data, reduce resolution to 15 min,
    and select years range if provided.
    '''
    
    # Map scenario name and open selected scenario file 
    ssp = map_ssp(scenario)
    pop = xr.open_dataset(f'{wdir}\\data\\socioeconomic_Data\\population\\GPOP\\GPOP_{ssp}.nc')
    
    if temp_source == 'ERA5':
        # Reduce resolution to 15 min to match ERA5 data
        pop_coarse = pop.coarsen(latitude=3, longitude=3, boundary='pad').sum(skipna=True)
        
    elif temp_source == 'MS':
        pop_coarse = pop.coarsen(latitude=6, longitude=6, boundary='pad').sum(skipna=True)
    
    # Adjust last year to be multiple of 5 for interpolation
    last_year = years[-1]
    if last_year % 5 != 0:
        last_year = last_year + (5 - last_year % 5)
        
    start_year = years[0]
    if start_year % 5 != 0:
        start_year = start_year - (start_year % 5)
    
    # Select years if provided
    if years:
        pop_coarse = pop_coarse.sel(time=slice(f'{start_year}-01-01', f'{last_year}-01-01'))
        
    # Linearly interpolate to yearly data
    yearly_data = pd.date_range(start=f'{years[0]}/01/01', end= f'{years[-1]}/01/01', freq='YS')
    pop_yearly = pop_coarse.interp(time=yearly_data)
    
    return pop_yearly



def map_ssp(ssp: str) -> str:
    
    '''
    Map input SSP names to the corresponding names used in the GPOP dataset.
    Parameters:
    ssp (str): Input SSP name (e.g., "SSP1", "SSP2", "SSP3", "SSP5").
    Returns:
    str: Corresponding GPOP SSP name.
    Raises:
    ValueError: If the input SSP name is not valid.
    '''
    
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