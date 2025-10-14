import pandas as pd
import numpy as np
import os,sys
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from functools import reduce
from scipy.stats import linregress
import datetime
sys.path.append(os.path.join(os.getcwd(), '0_Params')) # Append the path
import climate_variability


base_path = 'C:/Users/Nayeli/Documents' ### Select main folder
monthlymean_path = f'{base_path}/Main folder/Climate variability/T_MonthlyMean' 
seasonal1_path = f'{base_path}/Main folder/Climate variability/T_SeasonalVariability1' 
seasonal2_path = f'{base_path}/Main folder/Climate variability/T_SeasonalVariability2' 
if not os.path.exists(monthlymean_path):
        os.makedirs(monthlymean_path)
        os.makedirs(seasonal1_path)
        os.makedirs(seasonal2_path)
model_path = 'D:\\Climate Models - Bias Corrected\\AWI-CM-1-1-MR' ### Select one climate model to generate artifical data
climate_model = 'AWI-CM-1-1-MR'



'''Daily variability - Monthly Running Mean'''

scenarios = ['SSP126', 'SSP245', 'SSP370', 'SSP585']
years = np.arange(2016,2100)
for scenario in scenarios:
    for year in years:
        df = climate_variability.running_mean(climate_model, model_path, scenario, year)
        df.to_csv(f'{monthlymean_path}/{scenario}/MonthlyMean_{scenario}_{year}.csv')
        print(f'{scenario} - {year}')

for scenario in scenarios: ### Generate data for the first and last year
    for year in [2015,2100]:
        df = climate_variability.running_mean2(climate_model, model_path, scenario, year)
        df.to_csv(f'{monthlymean_path}/{scenario}/MonthlyMean_{scenario}_{year}.csv')
        print(f'{scenario} - {year}')



'''Seasonal Variability - Method 1'''

### Compute anual mean ###
years = np.arange(2015,2101)
df = pd.read_csv('D:/Climate Models - Bias Corrected/AWI-CM-1-1-MR/SSP126/BC_AWI-CM-1-1-MR_SSP126_2015.csv')  # Read first year and scenario file
for scenario in scenarios:
    year_avg = pd.DataFrame()
    year_avg['hierid'] = df['hierid']
    for year in years:
        df = pd.read_csv(os.path.join(model_path, f'{scenario}\\BC_{climate_model}_{scenario}_{year}.csv'))
        df_temp = df[df.columns[1:]]
        year_avg[f'{year}'] = df_temp.mean(axis=1)
        print(f'{scenario} - {year}')
    year_avg = year_avg.round(4)
    year_avg.to_csv(f'{base_path}/Main folder/Climate variability/Annual Mean_{scenario}.csv')

### Add slopes and intercepts
for scenario in scenarios: 
    climate_variability.get_linear_reg(base_path, scenario)
    print(f'{scenario}')

### Generate seasonal trend
for scenario in scenarios:
    seasonal_trend = pd.DataFrame(columns=[f"{month:02d}-{day:02d}" for month in range(1, 13) for day in range(1, 32)])  
    year_mean = pd.read_csv(f'{base_path}/Main folder/Climate variability/Annual Mean_{scenario}.csv')
    for year in years:
        SSP = pd.read_csv(f'{model_path}/{scenario}/BC_{climate_model}_{scenario}_{year}.csv')
        temp = SSP.iloc[:,1:]
        result = temp.sub(year_mean[f'{year}'], axis=0)
        result.columns = pd.to_datetime(result.columns).strftime('%m-%d')
        seasonal_trend = reduce(lambda x, y: x.add(y, fill_value=0), [seasonal_trend, result])
        print(f'{scenario} - {year}')
    for column in seasonal_trend.columns:
        seasonal_trend[column] = seasonal_trend[column].apply(climate_variability.custom_divide, date=column)
    seasonal_trend = seasonal_trend.round(3)
    seasonal_trend.to_csv(f'{base_path}/Main folder/Climate variability/Seasonal_trend_{scenario}.csv')

### Generate files
for scenario in scenarios:
    seasonal_var = pd.read_csv(f'{base_path}/Main folder/Climate variability/Seasonal_trend_{scenario}.csv')
    annual_mean = pd.read_csv(f'{base_path}/Main folder/Climate variability/Annual Mean_{scenario}.csv')
    for year in years:
        SSP = pd.read_csv(f'{model_path}/{scenario}/BC_{climate_model}_{scenario}_{year}.csv')
        new_SSP = SSP.iloc[0:0]
        new_SSP['hierid'] = SSP['hierid']
        new_SSP.columns = [new_SSP.columns[0]] + pd.to_datetime(new_SSP.columns[1:]).strftime('%m-%d').tolist()
        for col in new_SSP.columns[1:]:
            new_SSP[col] = seasonal_var[col] + annual_mean['slope']*year + annual_mean['intercept']
        new_SSP = new_SSP.round(2)
        new_SSP.columns = [col if col.startswith('hierid') else pd.to_datetime(f"{year}-{col}").strftime('%d-%m-%Y') for col in new_SSP.columns]
        new_SSP.to_csv(f'{seasonal1_path}\{scenario}\\SeasonalVariability1_{scenario}_{year}.csv')
        print(f'{scenario} - {year}')



'''Seasonal Variability Method 2'''

### Calculat the trend of every day
months_number = np.arange(1,13)
days_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
for scenario in scenarios:
    for month_number, day_month in zip(months_number, days_month):
        climate_variability.daily_regression(model_path, base_path, climate_model, scenario, month_number, day_month)

#Calculate the slopes and intercepts for each day
dates = []
for month, days in enumerate(days_month, start=1):
    for day in range(1, days + 1):
        date_str = datetime(year=1, month=month, day=day).strftime('%m-%d')
        dates.append(date_str)

#Generate the slopes and intercepts
for scenario in scenarios:
    for day in dates:
        df = pd.read_csv(f'{base_path}/Main folder/Climate variability/{scenario}_inter/{scenario}_{day}.csv')
        df = df.iloc[:,1:-2]
        slopes = []
        intercepts = []
        for index, row in df.iterrows(): # Iterate over each row in the dataframe
            X = row.index.values.reshape(-1, 1).astype(float)  # Prepare the data for linear regression
            y = row.values.reshape(-1, 1)
            # Create and fit the model
            model = LinearRegression()
            model.fit(X, y)
            # Store the slope and intercept
            slopes.append(model.coef_[0][0])
            intercepts.append(model.intercept_[0])
        # Convert the lists to new columns in the dataframe
        df['slope'] = slopes
        df['intercept'] = intercepts
        print(f'{scenario} - {day}')
        df.to_csv(f'{base_path}/Main folder/Climate variability/{scenario}_inter/{scenario}_{day}.csv')

#Append slopes and intercepts in one file
coefficients = ['intercept', 'slope']
index = pd.MultiIndex.from_product([scenarios, dates, coefficients], names=['Scenario', 'Day', 'Coefficient'])
df = pd.DataFrame(index=range(24378), columns=index)
for scenario in scenarios:
    for day in dates:
        df_coeff = pd.read_csv(f'{base_path}/Main folder/Climate variability/{scenario}_inter/{scenario}_{day}.csv')
        df.loc[:, (scenario, day, 'intercept')] =  df_coeff['intercept']
        df.loc[:, (scenario, day, 'slope')] =  df_coeff['slope']
        print(f'{scenario} - {day}')
df.to_csv(f'{base_path}/Main folder/Climate variability/Coefficients.csv')  ### Store intermediate file

#Generate files
df = pd.read_csv(f'{base_path}/Main folder/Climate variability/Coefficients.csv', header=[0, 1, 2])
for scenario in scenarios:    
    for year in years:
        SSP = pd.read_csv(f'{model_path}/{scenario}/BC_{climate_model}_{scenario}_{year}.csv')
        new_SSP = SSP.iloc[0:0]
        new_SSP['hierid'] = SSP['hierid']
        new_SSP.columns = [new_SSP.columns[0]] + pd.to_datetime(new_SSP.columns[1:]).strftime('%m-%d').tolist()

        for col in new_SSP.columns[1:]:
            if col == '02-29':
                new_SSP[col] = (df.loc[:, (scenario, '02-28', 'intercept')] + df.loc[:, (scenario, '02-28', 'slope')]*year \
                               + df.loc[:, (scenario, '03-01', 'intercept')] + df.loc[:, (scenario, '03-01', 'slope')]*year) / 2
            else:
                new_SSP[col] = df.loc[:, (scenario, col, 'intercept')] + df.loc[:, (scenario, col, 'slope')]*year

        new_SSP.columns = [col if col.startswith('hierid') else pd.to_datetime(f"{year}-{col}").strftime('%d-%m-%Y') for col in new_SSP.columns]

        new_SSP.iloc[:, 1:] = new_SSP.iloc[:, 1:].map(lambda x: round(x, 2))
        new_SSP.to_csv(f'{seasonal2_path}\{scenario}\\SeasonalVariability2_{scenario}_{year}.csv')
        print(f'{scenario} - {year}')

