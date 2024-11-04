import numba as nb
import pandas as pd
import numpy as np

def read_mortality_adap(path, climate_model, scenario, group, year)
        mor = pd.read_csv(f'{path}/{climate_model}/{scenario}/{climate_model}_{scenario}_{group}_{year}.csv')
        columns = list(mor.columns)
        num_other_columns = 1
        mor.columns = columns[:num_other_columns] + list(np.array(columns[num_other_columns:], dtype="float"))
        #min_temperature = mor.columns[num_other_columns]
        mor_np = mor.iloc[:, num_other_columns:].round(2).to_numpy()
        return mor_np

# Use for calculating total mortality

@nb.njit
def temp_to_column_location(temp):
    min_temperature = -50.0
    temp = np.round(temp, 1)
    return int(np.round(((temp - min_temperature) * 2)))

@nb.njit
def get_mortality(temperature_data, mortality_data):
    return np.array([
        mortality_data[region_i, temp_to_column_location(temperature_data[region_i])]
        for region_i in range(len(temperature_data))
    ])

def calculate_mortality_year(SSP, mortality_np):
    temperature_other_columns = 1 # Use 2 when running climate variability files, otherwise use 1
    mortality_SSP = SSP.iloc[:, :temperature_other_columns].copy()
    mortality_SSP = pd.concat([
        mortality_SSP,
        pd.concat([
            pd.Series(get_mortality(temperature_series.to_numpy(), mortality_np), name=day+"_mortality")
            for day, temperature_series in SSP.iloc[:, temperature_other_columns:].items()
        ], axis=1)
    ], axis=1)
    mortality_SSP['total mortality'] = mortality_SSP.filter(like='_mortality').sum(axis=1)
    return mortality_SSP

### Use for calculating hot and cold temperatures

@nb.njit
def get_mortality_hot(temperature_data, mortality_data, tmin_column):
    return np.array([
        mortality_data[region_i, temp_to_column_location(temperature_data[region_i])] if temperature_data[region_i] >= tmin_column[region_i] 
        else np.nan
        for region_i in range(len(temperature_data))
    ])

@nb.njit
def get_mortality_cold(temperature_data, mortality_data, tmin_column):
    return np.array([
        mortality_data[region_i, temp_to_column_location(temperature_data[region_i])] if temperature_data[region_i] < tmin_column[region_i] 
        else np.nan
        for region_i in range(len(temperature_data))
    ])

def calculate_mortality_year_hot(SSP, mortality_np, tmin_column):
    temperature_other_columns = 1
    mortality_SSP = SSP.iloc[:, :temperature_other_columns].copy()
    mortality_SSP = pd.concat([
        mortality_SSP,
        pd.concat([
            pd.Series(get_mortality_hot(temperature_series.to_numpy(), mortality_np, tmin_column), name=day+"_mortality")
            for day, temperature_series in SSP.iloc[:, temperature_other_columns:].items()
        ], axis=1)
    ], axis=1)
    mortality_SSP['total mortality'] = mortality_SSP.filter(like='_mortality').sum(axis=1)
    return mortality_SSP

def calculate_mortality_year_cold(SSP, mortality_np, tmin_column):
    temperature_other_columns = 1
    mortality_SSP = SSP.iloc[:, :temperature_other_columns].copy()
    mortality_SSP = pd.concat([
        mortality_SSP,
        pd.concat([
            pd.Series(get_mortality_cold(temperature_series.to_numpy(), mortality_np, tmin_column), name=day+"_mortality")
            for day, temperature_series in SSP.iloc[:, temperature_other_columns:].items()
        ], axis=1)
    ], axis=1)
    mortality_SSP['total mortality'] = mortality_SSP.filter(like='_mortality').sum(axis=1)
    return mortality_SSP