import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.join(os.getcwd(), '0_Params')) # Append the path
import calculate_mortality_noadap   # Import function to calculate mortality without adaptation
import climate_models_info
import MIMOSA_params

base_path = 'C:/Users/Nayeli/Documents' ### Select main folder
RF_path = f'{base_path}/Main folder/Response functions/'
folder_path = f'{base_path}/Main folder/MIMOSA/MIMOSA Mortality'
if not os.path.exists(folder_path):
        os.makedirs(folder_path)

'''Define important variables'''
scenarios = ['SSP126', 'SSP245', 'SSP370', 'SSP585']
scenario_SSP = 'SSP2' ### Results for MIMOSA assume a Middle-of-the-Road population growth
years = np.arange(2015,2105,5)
groups = ['oldest', 'older', 'young']
results = pd.DataFrame(columns = pd.MultiIndex.from_product([groups, scenarios, years], names=['Age group', 'Scenario', 'Year']))

'''Calculate regional mortality'''
for climate_model in climate_models_info.climate_models_dic.keys():
    for group in groups:
        mor_np = calculate_mortality_noadap.read_mortality_response(RF_path, group)

        POP = d.read_csv(f'{base_path}/Main folder/GDP & POP/POP files/POP_{scenario_SSP}_{group}.csv')
        POP['MIMOSA'] = POP['hierid'].apply(MIMOSA_params.get_region)
        POP_MIMOSA = POP[[f"{num}" for num in years] + ['MIMOSA']].groupby('MIMOSA').sum()

        for scenario in scenarios:          
            for year in years:
                SSP = pd.read_csv(f'{base_path}/Climate data/Climate ensemble/{climate_model}/{scenario}/BC_{climate_model}_{scenario}_{year}.csv')  # Read temp files
                mor_temp = calculate_mortality_noadap.calculate_mortality_year(SSP, mor_np) # Calculate relative mortality per region (deaths/100,000)
                mor_temp['total'] = POP[f'{year}'] * mor_temp['total mortality'] / 1e5 # Calculate total mortality per impact region
                mor_temp['MIMOSA'] = mor_temp['hierid'].apply(MIMOSA_params.get_region) # Group total mortality per MIMOSA region
                mortemp_grouped = mor_temp[['total']+['MIMOSA']].groupby('MIMOSA').sum()
                relative_mortality = mortemp_grouped['total']*1e5 / POP_MIMOSA[f'{year}'] # Calculate mortality per 100,000
                results.loc[:, (group, scenario, year)] = relative_mortality
                print(f'{climate_model} - {group} - {scenario} - {year}')
    results.index = mortemp_grouped.index
    results.to_csv(f'{folder_path}/MIMOSAMortality_NoAdaptation_{climate_model}.csv') ### Save files