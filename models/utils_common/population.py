import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import re, os
from . import temperature as tmp



def get_all_population_data(wdir, scenario, return_pop=False):
    
    '''
    Create a dataset with population data for all SSP scenarios 
    concatenated along a new 'ssp' dimension.
    '''
    
    SSPs = ['SSP1', 'SSP2', 'SSP3', 'SSP5']

    # Concatenate population data for all SSP scenarios
    pop_all_ssp = xr.concat(
        [LoadPopulationMap(wdir, scenario, ssp, years=range(2000, 2101)) for ssp in SSPs],
        dim='ssp'
    )

    # Assign ssp names to the new dimension
    pop_all_ssp['ssp'] = [s.lower() for s in SSPs]  
        
    # Mask with positive pop values
    mask_positive_pop = (pop_all_ssp > 0).any(dim=['time', 'ssp'])
    
    if return_pop:
        return pop_all_ssp, mask_positive_pop
    
    else:
        return mask_positive_pop



def LoadPopulationMap(wdir, scenario, ssp, years):
    
    '''
    Read scenario-dependent population data, interpolate it to yearly data, 
    reduce resolution to 15 min, and select years range if provided.
    '''
    
    # Open IMAGE SSP population data
    wdir_up = os.path.dirname(wdir)
    pop = xr.open_dataset(f'{wdir_up}/data/image_population/{ssp}/GPOP.nc')
    
    if re.search(r"ERA5", scenario):
        # Reduce resolution to 15 min to match ERA5 data
        pop_coarse = pop.coarsen(latitude=3, longitude=3, boundary='pad').sum(skipna=True)
    else:
        pop_coarse = pop.coarsen(latitude=6, longitude=6, boundary='pad').sum(skipna=True)
    
    # Select years if provided
    if years:
        pop_coarse = pop_coarse.sel(time=slice(f'{years[0]}-01-01', f'{years[-1]}-01-01'))
    
    return pop_coarse
    
    
    
def LoadRegionClassificationMap(wdir, temp_dir, region_class, scenario):
    
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

    if re.search(r"ERA5", scenario):
        temp_grid = tmp.DailyTemperatureERA5(temp_dir, 2000, "mean", pop_ssp=None, to_array=False)
    else:
        temp_grid,_ = tmp.OpenMontlhyTemperatures(temp_dir, "mean")
        
        
    if region_class == 'IMAGE26':
    
        # Read in IMAGE region data and interpolate to match files resolution
        regions = (
            xr.open_dataset(wdir+'/data/region_classification/GREG_30MIN.nc')
            .mean(dim="time") # Mean over time dimension
            .interp(longitude=temp_grid.longitude,  # Match temperature data resolution 
                    latitude=temp_grid.latitude, 
                    method='nearest')
            .GREG_30MIN.values
            )

        # Get number of regions excluding 27 and nan
        regions_range = np.unique(regions)[:-2].astype(int)
        
    if region_class == 'countries':
        
        # Read in GBD LEVEL 3 region data: countries and territories
        regions = (
            xr.open_dataset(wdir+'/data/GBD_locations/GBD_locations_level3.nc')
            .interp(longitude=temp_grid.longitude, 
                                     latitude=temp_grid.latitude, 
                                     method='nearest')
            .loc_id.values
        )
        
        # Get number of regions excluding -1 and nan
        regions_range = np.unique(regions)[1:-1].astype(int) 
        
    # Set negative values to 0
    regions = np.maximum(regions, 0)  
        
    return regions, regions_range