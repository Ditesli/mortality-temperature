import pandas as pd
import xarray as xr
import numpy as np
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