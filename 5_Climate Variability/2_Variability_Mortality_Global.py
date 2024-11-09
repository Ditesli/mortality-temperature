import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.join(os.getcwd(), '0_Params')) # Append the path
import calculate_mortality_noadap   # Import function to calculate mortality without adaptation

scenarios = ['SSP126', 'SSP245', 'SSP370', 'SSP585']
scenarios_SSP = ['SSP1', 'SSP2', 'SSP3', 'SSP5']
years = np.arange(2015,2105,5)
groups = ['oldest', 'older', 'young']
climate_models = ['AWI-CM-1-1-MR']
index = pd.MultiIndex.from_product([groups, scenarios], names=['Age group', 'Scenario'])
results_noadapt = pd.DataFrame(index=index, columns=years)

base_path = 'C:/Users/Nayeli/Documents' ### Select main folder
RF_path = f'{base_path}/Main folder/Response functions/'


'''Monthly Running Mean'''
for group in groups: 
    mor_np = calculate_mortality_noadap.read_mortality_response(RF_path, group) # Read mortality response functions
    for scenario, scenario_SSP in zip(scenarios, scenarios_SSP):
        POP = pd.read_csv(f'{base_path}/Main folder/GDP & POP/POP files/POP_{scenario_SSP}_{group}.csv') # Read population files
        POP = POP[~POP['hierid'].str.contains('ATA|CA-', regex=True)]
        for year in years:
            # Read temperature file
            SSP = pd.read_csv(f'{base_path}/Main folder/Climate Variability/T_Monthly Mean/{scenario}/MonthlyMean_{scenario}_{year}.csv')
            mor_temp = calculate_mortality_noadap.calculate_mortality_year(SSP, mor_np) # Calculate relative mortality per region (deaths/100,000)
            mor_temp = mor_temp[~mor_temp['hierid'].str.contains('ATA|CA-', regex=True)]
            total_mortality = np.sum(POP[f'{year}'].to_numpy() * mor_temp['total mortality'].to_numpy()/100000) # Calculate mortality relative to SSP scenario
            relative_mortality = total_mortality*100000/np.sum(POP[f'{year}'].to_numpy())
            results_noadapt.at[(group, scenario), year] = relative_mortality
            print(f'{group} - {scenario} - {year}')
results_noadapt.to_csv(f'{base_path}/Main folder/Climate Variability/Mortality_ClimateVariability_MontlhyMean.csv')


'''Seasonal Variability - Method 1'''
results_noadapt = pd.DataFrame(index=index, columns=years)

for group in groups: 
    mor_np = calculate_mortality_noadap.read_mortality_response(RF_path, group)
    for scenario, scenario_SSP in zip(scenarios, scenarios_SSP):
        POP = pd.read_csv(f'{base_path}/Main folder/GDP & POP/POP files/POP_{scenario_SSP}_{group}.csv')
        POP = POP[~POP['hierid'].str.contains('ATA|CA-', regex=True)]
        for year in years:
            SSP = pd.read_csv(f'{base_path}/Main folder/Climate Variability/T_SeasonalVariability1/{scenario}/SeasonalVariability1_{scenario}_{year}.csv')
            mor_temp = calculate_mortality_noadap.calculate_mortality_year(SSP, mor_np)
            mor_temp = mor_temp[~mor_temp['hierid'].str.contains('ATA|CA-', regex=True)]
            total_mortality = np.sum(POP[f'{year}'].to_numpy() * mor_temp['total mortality'].to_numpy()/100000)
            relative_mortality = total_mortality*100000/np.sum(POP[f'{year}'].to_numpy())
            results_noadapt.at[(group, scenario), year] = relative_mortality
            print(f'{group} - {scenario} - {year}')
results_noadapt.to_csv(f'{base_path}/Main folder/Climate Variability/Mortality_ClimateVariability_Seasonal1.csv')


'''Climate Variability - Method 2'''
results_noadapt = pd.DataFrame(index=index, columns=years)

for group in groups:
    mor_np = calculate_mortality_noadap.read_mortality_response(RF_path, group)
    for scenario, scenario_SSP in zip(scenarios, scenarios_SSP):
        POP = pd.read_csv(f'{base_path}/Main folder/GDP & POP/POP files/POP_{scenario_SSP}_{group}.csv')
        POP = POP[~POP['hierid'].str.contains('ATA|CA-', regex=True)]
        for year in years:
            SSP = pd.read_csv(f'{base_path}/Main folder/Climate Variability/T_SeasonalVariability2/{scenario}/SeasonalVariability2_{scenario}_{year}.csv')
            mor_temp = calculate_mortality_noadap.calculate_mortality_year(SSP, mor_np)
            mor_temp = mor_temp[~mor_temp['hierid'].str.contains('ATA|CA-', regex=True)]
            total_mortality = np.sum(POP[f'{year}'].to_numpy() * mor_temp['total mortality'].to_numpy()/100000)
            relative_mortality = total_mortality*100000/np.sum(POP[f'{year}'].to_numpy())
            results_noadapt.at[(group, scenario), year] = relative_mortality
            print(f'{group} - {scenario} - {year}')
results_noadapt.to_csv(f'{base_path}/Main folder/Climate Variability/Mortality_ClimateVariability_Seasonal2.csv')