import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.join(os.getcwd(), '0_Params')) # Append the path
import calculate_mortality_noadap   # Import function to calculate mortality without adaptation
import MIMOSA_params

scenarios = ['SSP126', 'SSP245', 'SSP370', 'SSP585']
scenarios_SSP = ['SSP1', 'SSP2', 'SSP3', 'SSP5']
years = np.arange(2015,2105,5)
groups = ['oldest', 'older', 'young']
climate_models = ['AWI-CM-1-1-MR']
index = pd.MultiIndex.from_product([groups, scenarios], names=['Age group', 'Scenario'])
results = pd.DataFrame(index=index, columns=years)

base_path = 'C:/Users/Nayeli/Documents' ### Select main folder
RF_path = f'{base_path}/Main folder/Response functions/'


'''Monthly Running Mean'''
for group in groups: 
    mor_np = calculate_mortality_noadap.read_mortality_response(RF_path, group) # Read mortality response functions
    for scenario, scenario_SSP in zip(scenarios, scenarios_SSP):
        POP = pd.read_csv(base_path + f'GDP-POP/POP files/POP_{scenario_SSP}_{group}.csv') # Read population files
        POP['MIMOSA'] = POP['hierid'].apply(MIMOSA_params.get_region)
        POP_MIMOSA = POP[[f"{num}" for num in years] + ['MIMOSA']].groupby('MIMOSA').sum()
        for year in years:
            SSP = pd.read_csv(f'{base_path}/Main folder/Climate Variability/T_Monthly Mean/{scenario}/MonthlyMean_{scenario}_{year}.csv')
            mor_temp = calculate_mortality_noadap.calculate_mortality_year(SSP, mor_np) # Calculate relative mortality per impact region
            mor_temp['total'] = POP[f'{year}'] * mor_temp['total mortality'] / 1e5 # Calculate total mortality per impact region
            mor_temp['MIMOSA'] = mor_temp['hierid'].apply(MIMOSA_params.get_region) # Group total mortality per MIMOSA region
            mortemp_grouped = mor_temp[['total']+['MIMOSA']].groupby('MIMOSA').sum()
            relative_mortality = mortemp_grouped['total']*1e5 / POP_MIMOSA[f'{year}'] # Calculate mortality per 100,000
            results.loc[:, (group, scenario, year)] = relative_mortality
            print(f'{group} - {scenario} - {year}')
results.index = mortemp_grouped.index
results.to_csv(f'{base_path}/Main folder/Climate Variability/Mortality_ClimateVariability_MIMOSA_MontlhyMean.csv')


'''Seasonal Variability - Method 1'''
results = pd.DataFrame(columns=col_index)
for group in groups:
    mor_np = calculate_mortality_noadap.read_mortality_response(RF_path, group)  # Read mortality response functions 
    for scenario, scenario_SSP in zip(scenarios, scenarios_SSP):
        POP = pd.read_csv(base_path + f'GDP-POP/POP files/POP_{scenario_SSP}_{group}.csv') # Read population files
        POP['MIMOSA'] = POP['hierid'].apply(MIMOSA_params.get_region)
        POP_MIMOSA = POP[[f"{num}" for num in years] + ['MIMOSA']].groupby('MIMOSA').sum()
        for year in years:
            SSP = pd.read_csv(f'{base_path}/Main folder/Climate Variability/T_SeasonalVariability1/{scenario}/SeasonalVariability1_{scenario}_{year}.csv')
            mor_temp = calculate_mortality_noadap.calculate_mortality_year(SSP, mor_np) # Calculate relative mortality per impact region
            mor_temp['total'] = POP[f'{year}'] * mor_temp['total mortality'] / 1e5 # Calculate total mortality per impact region
            mor_temp['MIMOSA'] = mor_temp['hierid'].apply(MIMOSA_params.get_region) # Group total mortality per MIMOSA region
            mortemp_grouped = mor_temp[['total']+['MIMOSA']].groupby('MIMOSA').sum()
            relative_mortality = mortemp_grouped['total']*1e5 / POP_MIMOSA[f'{year}']  # Calculate mortality per 100,000
            results.loc[:, (group, scenario, year)] = relative_mortality
            print(f'{group} - {scenario} - {year}')
results.index = mortemp_grouped.index
results.to_csv(f'{base_path}/Main folder/Climate Variability/Mortality_ClimateVariability_MIMOSA_Seasonal1.csv')


'''Seasonal Variability - Method 2'''
results = pd.DataFrame(columns=col_index)
for group in groups:
    mor_np = calculate_mortality_noadap.read_mortality_response(RF_path, group)  # Read mortality response functions 
    for scenario, scenario_SSP in zip(scenarios, scenarios_SSP):
        POP = pd.read_csv(base_path + f'GDP-POP/POP files/POP_{scenario_SSP}_{group}.csv') # Read population files
        POP['MIMOSA'] = POP['hierid'].apply(MIMOSA_params.get_region)
        POP_MIMOSA = POP[[f"{num}" for num in years] + ['MIMOSA']].groupby('MIMOSA').sum()
        for year in years:
            SSP = pd.read_csv(f'{base_path}/Main folder/Climate Variability/T_SeasonalVariability2/{scenario}/SeasonalVariability2_{scenario}_{year}.csv')
            mor_temp = calculate_mortality_noadap.calculate_mortality_year(SSP, mor_np) # Calculate relative mortality per impact region
            mor_temp['total'] = POP[f'{year}'] * mor_temp['total mortality'] / 1e5 # Calculate total mortality per impact region
            mor_temp['MIMOSA'] = mor_temp['hierid'].apply(MIMOSA_params.get_region) # Group total mortality per MIMOSA region
            mortemp_grouped = mor_temp[['total']+['MIMOSA']].groupby('MIMOSA').sum()
            relative_mortality = mortemp_grouped['total']*1e5 / POP_MIMOSA[f'{year}']  # Calculate mortality per 100,000
            results.loc[:, (group, scenario, year)] = relative_mortality
            print(f'{group} - {scenario} - {year}')
results.index = mortemp_grouped.index
results.to_csv(f'{base_path}/Main folder/Climate Variability/Mortality_ClimateVariability_MIMOSA_Seasonal2.csv')