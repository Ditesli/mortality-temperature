import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.join(os.getcwd(), '0_Params')) # Append the path
import calculate_mortality_noadap   # Import function to calculate mortality without adaptation
import climate_models_info

base_path = 'C:/Users/Nayeli/Documents'
RF_path = f'{base_path}/Main folder/Response functions/'
folder_path = f'{base_path}/Main folder/Mortality/No Adaptation' 

''' Define important variables'''
scenarios = ['SSP126', 'SSP245', 'SSP370', 'SSP585']
scenarios_SSP = ['SSP1', 'SSP2', 'SSP3', 'SSP5']
years = np.arange(2015,2105,5)
groups = ['oldest', 'older', 'young']
index = pd.MultiIndex.from_product([groups, scenarios], names=['Age group', 'Scenario'])
results_cold = pd.DataFrame(index=index, columns=years)
results_hot = pd.DataFrame(index=index, columns=years)
tmin = pd.read_csv(f'{base_path}/Main folder/Response functions/T_min.csv'))

''' This loop iterates the climate models, scenarios, age groups and years to read temp and RF file'''
for climate_model in climate_models_info.climate_models_dic.keys():
    for group in groups:
        mor_np = calculate_mortality_noadap.read_mortality_response(RF_path, group) # Read mortality response functions and preprocess
        t_min = np.array(tmin[f'Tmin {group}']) # Open tmin file
        for scenario, scenario_SSP in zip(scenarios, scenarios_SSP):
            POP = pd.read_csv(f'{base_path}/Main folder/GDP & POP/POP files/POP_{scenario_SSP}_{group}.csv')  # Read population files
            POP = POP[~POP['hierid'].str.contains('ATA|CA-', regex=True)]
            for year in years: # for loop for the temporal period
                SSP = pd.read_csv(f'{base_path}/Climate data/Climate ensemble/{climate_model}/{scenario}/BC_{climate_model}_{scenario}_{year}.csv')  # Read temp files
                # Calculate relative mortality per region
                mor_temp = calculate_mortality_noadap.calculate_mortality_year_cold(SSP, mor_np, t_min)
                mor_temp = mor_temp[~mor_temp['hierid'].str.contains('ATA|CA-', regex=True)]
                # Calculate mortality relative to SSP scenario
                total_mortality = np.sum(POP[f'{year}'].to_numpy() * mor_temp['total_mortality'].to_numpy()/100000)
                relative_mortality = total_mortality*100000/np.sum(POP[f'{year}'].to_numpy())
                results_cold.at[(group, scenario), year] = relative_mortality
                print(f'{climate_model} - {group} - {scenario} - {year}')
    results_cold.to_csv(f'{folder_path}/ColdMortality_NoAdaptation_{climate_model}.csv')  ### Save files

''' This for loops is the same as above but modified for HotMortality files (Mortality due to hot temperatures)'''
for climate_model in climate_models_info.climate_models_dic.keys():
    for group in groups:
        mor_np = calculate_mortality_noadap.read_mortality_response(RF_path, group) # Read mortality response functions and preprocess
        t_min = np.array(tmin[f'Tmin {group}']) # Open tmin file
        for scenario, scenario_SSP in zip(scenarios, scenarios_SSP):
            POP = pd.read_csv(f'{base_path}/Main folder/GDP & POP/POP files/POP_{scenario_SSP}_{group}.csv')  # Read population files
            POP = POP[~POP['hierid'].str.contains('ATA|CA-', regex=True)]
            for year in years: # for loop for the temporal period
                SSP = pd.read_csv(f'{base_path}/Climate data/Climate ensemble/{climate_model}/{scenario}/BC_{climate_model}_{scenario}_{year}.csv')  # Read temp files
                # Calculate relative mortality per region
                mor_temp = calculate_mortality_noadap.calculate_mortality_year_hot(SSP, mor_np, t_min)
                mor_temp = mor_temp[~mor_temp['hierid'].str.contains('ATA|CA-', regex=True)]
                # Calculate mortality relative to SSP scenario
                total_mortality = np.sum(POP[f'{year}'].to_numpy() * mor_temp['total_mortality'].to_numpy()/100000)
                relative_mortality = total_mortality*100000/np.sum(POP[f'{year}'].to_numpy())
                results_hot.at[(group, scenario), year] = relative_mortality
                print(f'{climate_model} - {group} - {scenario} - {year}')
    results_hot.to_csv(f'{folder_path}/HotMortality_NoAdaptation_{climate_model}.csv')  ### Save files
