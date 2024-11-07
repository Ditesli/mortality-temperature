import matplotlib.pyplot as plt 
import numpy as np
import xarray as xr
import pandas as pd
sys.path.append(os.path.join(os.getcwd(), '0_Params')) # Append the path
import climate_models_info

base_path = 'C:/Users/Nayeli/Documents' ### Select main folder
folder_ERA5 = 'D:ERA5'
folder_presentday = 'D:Climate Models - Present Day NSAT'
folder_historical = 'C:/Users/Nayeli/Documents/Thesis/CMIP/Climate Models - Historical NSAT'
folder_future = 'D:Climate Models - Future NSAT/'

def weighted_avg(data): #data must be #time, lat, lon#
    weights = np.cos(np.deg2rad(data.lat))
    return data.weighted(weights).mean(dim=('lat','lon')) -273.15


'''Bias Correction'''
bias_models = pd.DataFrame(index=climate_models_info.climate_models_dic.keys())
bias_models['Bias'] = ''

### Open ERA5 data ###
temp_ERA5 = xr.open_dataset(f'{folder_ERA5}/ERA5_Amon_tas_1990-2020.nc')
weights = np.cos(np.deg2rad(temp_ERA5.latitude))
temp_ERA5_weighted = temp_ERA5.weighted(weights).mean(dim=('latitude','longitude')) - 273.15
temp_ERA5_weighted_mean = temp_ERA5_weighted.mean('time').t2m
temperatura_ERA5 = temp_ERA5_weighted_mean.values

### Open climate models and calculate bias ###
def calculate_bias(climate_model, label, grid):
    temp = xr.open_dataset(f'{folder_presentday}/tas_Amon_{climate_model}_present_{label}_{grid}_1990-2020.nc')
    tas_weighted = weighted_avg(temp.tas)
    tas_mean = tas_weighted.mean(dim='time')
    tas_values = tas_mean.values
    bias =  temperatura_ERA5 - tas_values
    bias_models.loc[climate_model,'Bias'] = bias

for climate_model in climate_models_info.climate_models_dic.keys():
    calculate_bias(climate_model, climate_models_info.climate_models_dic[climate_model][0], climate_models_info.climate_models_dic[climate_model][1])


'''Calculate Historical GSAT'''
temperatures = pd.DataFrame(index=climate_models_info.climate_models_dic.keys())
temperatures['Historical'] = ''

def add_model(climate_model, variant_label, grid):
    tas_model = xr.open_dataset(f'{folder_historical}/tas_Amon_{climate_model}_historical_{variant_label}_{grid}.nc')
    tas_weighted = weighted_avg(tas_model.tas)
    tas_mean = tas_weighted.mean(dim='time')
    tas_values = tas_mean.values
    temperatures.loc[climate_model, 'Historical'] = tas_values + bias_models.loc[climate_model,'Bias']

for climate_model in climate_models_info.climate_models_dic.keys():
    add_model(climate_model, climate_models_info.climate_models_dic[climate_model][0], climate_models_info.climate_models_dic[climate_model][1])

temperatures = temperatures.round(2)
temperatures.to_csv(f'{base_path}/Main folder/Climate data/GSAT_Historical.csv')


'''Calculate future GSAT'''
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
years=np.arange(2015,2105,5)
col_index = pd.MultiIndex.from_product([climate_models_info.climate_models_dic.keys(), scenarios], names=['Climate Models', 'Scenarios'])
temperatures = pd.DataFrame(columns=col_index, index=years)

def add_model2(climate_model, variant_label, grid):
    for scenario in scenarios:
        tas_model = xr.open_dataset(f'{folder_future}/tas_Amon_{climate_model}_{scenario}_{variant_label}_{grid}_20150116-21001216.nc')
        tas_weighted = weighted_avg(tas_model.tas)
        tas_year = tas_weighted.groupby('time.year').mean(dim='time') + bias_models.loc[climate_model,'Bias']
        tas_df = tas_year.to_dataframe()
        temperatures.loc[:, (climate_model, scenario)] = tas_df['tas'].round(2)  

for climate_model in climate_models_info.climate_models_dic.keys():
    add_model2(climate_model, climate_models_info.climate_models_dic[climate_model][0], climate_models_info.climate_models_dic[climate_model][1])

temperatures = temperatures.round(2)
temperatures.to_csv(f'{base_path}/Main folder/Climate data/GSAT_Future.csv')