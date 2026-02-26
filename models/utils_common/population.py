import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import re



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



def get_annual_pop(wdir, scenario, years=None):
    
    '''
    Read scenario-dependent population data, interpolate it to yearly data, reduce resolution to 15 min,
    and select years range if provided.
    '''
    
    # Map scenario name and open selected scenario file 
    ssp = map_ssp(scenario)
    pop = xr.open_dataset(f'{wdir}\\data\\socioeconomic_Data\\population\\GPOP\\GPOP_{ssp}.nc')
    
    if re.search(r"SSP[1-5]_ERA5", scenario):
        # Reduce resolution to 15 min to match ERA5 data
        pop_coarse = pop.coarsen(latitude=3, longitude=3, boundary='pad').sum(skipna=True)
        
    else:
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
    
    
    
def read_region_classification(wdir, region_class, scenario):
    
    '''
    Load the region classification selected (IMAGE26 or country level) and return as numpy array 
    along with number of regions.
    Depending on the temperature data source and the region classification type, change the resolution 
    to either 0.5x0.5 or 0.25x0.25 degrees.
    In case the grid has negative values (e.g., IMAGE26), set them to 0.
    Parameters:
    - wdir: working directory
    - region_class: 'IMAGE26' or 'countries'
    - temp_source: 'ERA5' or 'MS'
    Returns:
    - region_nc: numpy array with region classification
    - regions_range: range or list with number of regions
    '''
    
    ### Import ERA5 temperature zones
    lats = np.arange(89.875, -90, -0.25)
    lons = np.arange(-179.875, 180, 0.25)

    ref_grid = xr.DataArray(
        np.full((lats.size, lons.size), np.nan),
        coords={"latitude": lats, "longitude": lons},
        dims=["latitude", "longitude"]
    )
        
    if region_class == 'IMAGE26':
    
        # Read in IMAGE region data and interpolate to match files resolution
        region_nc = xr.open_dataset(wdir+'data/regions_classification/IMAGE/GREG_30MIN.nc')
        
        # Interpolate to match temperature data resolution and average over time dimension for ERA5 data
        if re.search(r"SSP[1-5]_ERA5", scenario):
            region_nc = region_nc.interp(longitude=ref_grid.longitude, 
                                        latitude=ref_grid.latitude, 
                                        method='nearest').mean(dim='time') 
            
        # Convert files to numpy arrays
        region_nc = region_nc.GREG_30MIN.values
        
        # Get number of regions
        regions_range = range(1,27)
        
    if region_class == 'countries':
        
        # Read in GBD LEVEL 3 region data: countries and territories
        region_nc = xr.open_dataset(wdir+'data/regions_classification/GBD/GBD_locations_level3.nc')
        
        # Interpolate to match temperature data resolution
        region_nc = region_nc.interp(longitude=ref_grid.longitude, 
                                     latitude=ref_grid.latitude, 
                                     method='nearest')
        
        # Get number of regions
        regions_range = np.unique(region_nc.loc_id.values)[1:-1].astype(int)  # Exclude -1 and nan
        region_nc = region_nc.loc_id.values
        
        # Reduce resolution to 0.5x0.5 degrees
        if not re.search(r"SSP[1-5]_ERA5", scenario):
            
            # Reshape array to 4D blocks of 2x2
            arr_reshaped = region_nc.reshape(360, 2, 720, 2)

            # Calculate mode over the 2x2 blocks to reduce resolution
            region_nc = stats.mode(stats.mode(arr_reshaped, axis=3, keepdims=False).mode, axis=1, keepdims=False).mode
    
    # Set negative values to 0
    region_nc = np.maximum(region_nc, 0)  
        
    return region_nc, regions_range