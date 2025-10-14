import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
sys.path.append(os.path.join(os.getcwd(), '0_Params')) # Append the path
import calculate_mortality_adap   # Import function to calculate mortality without adaptation
import climate_models_info

base_path = 'C:/Users/Nayeli/Documents/' ### Select main path
folder_path = f'{base_path}/Main folder/Mortality/Adaptation' 
if not os.path.exists(folder_path):
        os.makedirs(folder_path)

scenarios = ['SSP126', 'SSP245', 'SSP370', 'SSP585']
scenarios_SSP = ['SSP1', 'SSP2', 'SSP3', 'SSP5']
years = np.arange(2020,2105,5)
groups = ['oldest', 'older', 'young']
index = pd.MultiIndex.from_product([groups, scenarios], names=['Age group', 'Scenario'])
results_adapt = pd.DataFrame(index=index, columns=years)


'''Calculate mortality with adaptation'''
covariate = 'GDP&T' ### This can be change to 'T' (only temperature evolves with time) or 'GDP' (only income increases with time)
RF_path = base_path + f'Main folder/Response Functions/RF Dataframes_{covariate}'
for climate_model in climate_models_info.climate_models_dic.keys():
    for group in groups:
        for scenario, scenario_SSP in zip(scenarios, scenarios_SSP):
            for year in years:
                #Read mortality files
                mor_np = calculate_mortality_adap.read_mortality_adap(RF_path,climate_model, scenario, group, year)
                # Read temperature file
                #SSP = pd.read_csv(f'D:\\Climate models - Bias corrected\\{climate_model}\\{scenario}\\BC_{climate_model}_{scenario}_{year}.csv')
                SSP = pd.read_csv(f'{base_path}/Climate data/Climate ensemble/{climate_model}/{scenario}/BC_{climate_model}_{scenario}_{year}.csv')  
                # Read population files
                POP = pd.read_csv(f'{base_path}/Main folder/GDP & POP/POP files/POP_{scenario_SSP}_{group}.csv') 
                population = POP.merge(SSP['hierid'], on='hierid', how='right')
                # Calculate relative mortality per region (deaths/100,000)
                mor_temp = calculate_mortality_adap.calculate_mortality_year(SSP, mor_np)
                # Calculate total mortality per region
                mor_temp['total'] = population[f'{year}']*mor_temp['total mortality']/100000
                # Calculate mortality relative to SSP scenario
                total_mortality = mor_temp['total'].sum()
                relative_mortality = total_mortality*100000/POP[f'{year}'].sum() 
                results_adapt.at[(group, scenario), year] = relative_mortality
                print(f'{climate_model} - {group} - {scenario} - {year}')
    results_adapt.to_csv(f'{folder_path}/TotalMortality_Adaptation_{covariate}_{climate_model}.csv')


