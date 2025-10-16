import pandas as pd
import xarray as xr
import numpy as np
import geopandas as gpd
import utils


wdir = 'X:\\user\\liprandicn\\mt-comparison\\Carleton2022\\data\\'

# Define scenarios
scenarios = ['SSP1', 'SSP2', 'SSP3', 'SSP5']
# Define groups
age_groups = ['total', 'young', 'older', 'oldest']
pop_groups = ['pop', 'pop0to4', 'pop5to64', 'pop65plus']
# Define years
years = np.arange(2015,2105,5)
# Define GDP model
gdp_models = ['low', 'high']



### ----------------------------------------------------------------------
''' 
Generate files per scenario and age group for all impact regions 
'''

# Generate population files
for scenario in scenarios:
    for pop_group, age_group in zip(pop_groups, age_groups):
        utils.var_nc_to_csv(wdir, 'POP', scenario, pop_group, age_group, 'low')


# Generate gdp files
for scenario in scenarios:
    for gdp_model in gdp_models:
        utils.var_nc_to_csv(wdir, 'GDP', scenario, 'gdppc', None, gdp_model)
        
    # Calculate mean gdp from high and low models
    SSP_GDP_high = pd.read_csv(f'{wdir}gdp_pop_csv/GDP_{scenario}_high.csv')
    SSP_GDP_low = pd.read_csv(f'{wdir}gdp_pop_csv/GDP_{scenario}_low.csv')
    SSP_GDP_mean = SSP_GDP_high.copy()
    for year in range(2010,2101):
        SSP_GDP_mean[f'{year}'] = (SSP_GDP_high[f'{year}'] + SSP_GDP_low[f'{year}']) / 2
    SSP_GDP_mean.to_csv(f'{wdir}gdp_pop_csv/GDP_{scenario}_mean.csv')
    
    
    
### ----------------------------------------------------------------------
''' 
Generate file that contains impact region codes, names and their corresponding 
IMAGE and GBD region
'''

# Path to IMAGE regions classification folder produced manually
image_regions = pd.read_excel('X:\\user\\liprandicn\\Health Impacts Model\\data\\IMAGE_regions\\regions_comparison.xlsx',
                              sheet_name='regions')

# Read impact regions shapefile and extract regions names
impact_regions = gpd.read_file(f'{wdir}'+'ir_shp/impact-region.shp')
impact_regions['ISO3'] = impact_regions['hierid'].str[:3]

# Merge with IMAGE regions to get IMAGE region codes
df = pd.merge(impact_regions[['hierid', 'ISO3']], image_regions, on='ISO3', how='left')
df.rename(columns={'hierid':'impact_region'}, inplace=True)

df.to_csv(f'{wdir}'+'region_classification.csv', index=False)