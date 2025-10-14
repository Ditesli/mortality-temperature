import pandas as pd
import xarray as xr
import numpy as np
import os

base_path = 'C:\\Users\\Nayeli\\Documents\\' # Base path
carleton_path = 'D:\\data\\2_projection\\2_econ_vars\\' #Load carleton path to population files


''' Generate files per scenario and age group for all impact regions '''

scenarios = ['SSP1', 'SSP2', 'SSP3', 'SSP5']
groups = ['pop', 'pop0to4', 'pop5to64', 'pop65plus']
age_groups = ['total', 'young', 'older', 'oldest']

for scenario in scenarios:
    ssp = xr.open_dataset(f'{carleton_path}/{scenario}.nc4')
    for group, age_group in zip(groups, age_groups):
        ssp_pop = ssp[group].sel(model='low')
        df = ssp_pop.to_dataframe()
        df = df.drop(['ssp', 'model'], axis=1).unstack('year')
        df.columns = df.columns.get_level_values(1)
        df = df.reset_index()
        df['hierid'] = df['region']
        df = df.merge(oldest['region'], on='region', how='right')
        folder_path = f'{base_path}/Main folder/GDP & POP/POP files' ### Optional: Save the file with ERA5 NSAT per impact region
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        df.to_csv(f'{folder_path}/POP_{scenario}_{age_group}.csv')


'''Generate files of global young, older, oldest population per year'''

age_groups = ['young', 'older', 'oldest']
groups = ['pop0to4', 'pop5to64', 'pop65plus']
years = np.arange(2015,2105,5)
index = pd.MultiIndex.from_product([age_groups, scenarios], names=['Age group', 'Scenario'])
df = pd.DataFrame(index=index, columns=years)

for scenario in scenarios:
    ssp_pop = xr.open_dataset(f'{carleton_path}/{scenario}.nc4')
    for age_group, group in zip(age_groups, groups): 
        pop_vals = ssp_pop[group].sel(model='high').sum(dim='region')[::5].values
        df.loc[(age_group, scenario),:] = pop_vals[1:]
        
df = df.round(1)
df.to_csv(f'{folder_path}/POP_global.csv')


''' Generate High, Low and Mean GDP projections '''

for scenario in scenarios:
    for model in ['low', 'high']:
        ssp = xr.open_dataset(f'{carleton_path}/{scenario}.nc4') # Open xarray from Carleton data
        ssp = ssp.gdppc.sel(model=f'{model}') # Select high or low model
        df = ssp.to_dataframe() # Convert to dataframe
        df = df.drop(['ssp', 'model'], axis=1).unstack('year') # Unstack
        df.columns = df.columns.get_level_values(1)
        df = df.reset_index() # Reset index
        df = df.merge(oldest['region'], on='region', how='right')
        folder_path = f'{base_path}/Main folder/GDP & POP/GDPpc files' ### Optional: Save the file with ERA5 NSAT per impact region
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        df.to_csv(f'{folder_path}/GDPpc_{scenario}_{model}.csv')

for SSP in scenarios:
    SSP_GDP_high = pd.read_csv(f'{folder_path}/GDPpc_{scenario}_high.csv')
    SSP_GDP_low = pd.read_csv(f'{folder_path}/GDPpc_{scenario}_low.csv')
    SSP_GDP_mean = SSP_GDP_high.copy()
    for year in range(2010,2101):
        SSP_GDP_mean[f'{year}'] = (SSP_GDP_high[f'{year}'] + SSP_GDP_low[f'{year}']) / 2
    #SSP_GDP_mean.rename(columns={'region': 'hierid'}, inplace=True)
    SSP_GDP_mean.to_csv(f'{folder_path}/GDPpc_{SSP}_mean.csv')