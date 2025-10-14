'''
1. Calculate the TMREL mean per each temperature zone for all locations available in the data provided
2. Calculate the mean and std of the Response functions from the 1000 draws 
2. Shift the data to match the TMREL (Theoretical Minimum Exposure Level) at 0 RR
4. Fill in lower/higher temperatures than the function to have a wider range of daily temperatures
(following Burkart's methodology)
5. Convert log(RR) to RR
6. Save in accessible format to optimize later calculations
'''

import numpy as np
import pandas as pd
import dask.dataframe as dd
import glob

### Set Temeprature Zones values [6,28]
tz_range = range(6,29)

### Filter only the summaries files to read the TMREL files across all locations
files = glob.glob("X:\\user\\liprandicn\\mortality_temperature\\temperature_erf\\TMRELs\\*summaries*.csv")

### Read files and append TMREL mean column
dfs_tmrel = dd.concat([  
    (df := dd.read_csv(file))['tmrelMean']  # Read TMREL column for each file
    for file in files  # Iterate over each file
], axis=1).compute()  # Append all columns 

### Add column mean to a dataframe with temperature zones in the rows and years in the columns
tmrel = pd.DataFrame({'tmrel': dfs_tmrel.mean(axis=1).values}, 
                     index=pd.MultiIndex.from_product([[1990,2010,2020], tz_range], names=['year', 'temp_zone'])
                     ).unstack(level=0).xs('tmrel', axis=1, level=0).round(1)

### Disease list
disease_list = ['ckd', 'cvd_cmp', 'cvd_htn', 'cvd_ihd', 'cvd_stroke', 'diabetes', 'inj_animal', 'inj_disaster', 'inj_drowning', 
                'inj_homicide', 'inj_mech', 'inj_othunintent', 'inj_suicide', 'inj_trans_other', 'inj_trans_road', 'lri', 'resp_copd']


for disease in disease_list:
    
    erf = pd.read_csv(f'X:\\user\\liprandicn\\Health Module\\ResponseFunctions\\erf\\{disease}_curve_samples.csv')
    erf['daily_temperature'] = erf['daily_temperature'].round(1)
    erf.set_index(['annual_temperature', 'daily_temperature'], inplace=True)
    erf = erf.apply(lambda x: np.exp(x))
    
    ### Shift RR to match 2010 mean TMREL of each temperature zone, reset dummy tz index
    erf_min = erf.groupby(level=0).apply(lambda x: x / erf.loc[(x.index.get_level_values(0)[0], tmrel.loc[x.index.get_level_values(0)[0], 2010])]).reset_index(level=0, drop=True)
    
    erf_min['rr'] = erf_min.mean(axis=1)
    erf_min['rr_std'] = erf_min.std(axis=1)
    
    erf_min = erf_min[['rr', 'rr_std']].round(4).reset_index()
    
    ### Save csv 
    erf_min.to_csv(f'X:\\user\\liprandicn\\Health Module\\ResponseFunctions\\erf_reformatted2\\erf_{disease}.csv')
            
### -------------------------------------------------------------------------------------------------------------
### Files for Method 2
                
### Iterate over all diseases
# for disease in disease_list:
    
#         erf = pd.read_csv(f'X:\\user\\liprandicn\\Health Module\\ResponseFunctions\\erf\\{disease}_curve_samples.csv')
#         erf['daily_temperature'] = erf['daily_temperature'].round(1)
#         erf.set_index(['annual_temperature', 'daily_temperature'], inplace=True)
#         erf = erf.astype(np.float64)
        
#         ### Shift RR to match 2010 mean TMREL of each temperature zone, reset dummy tz index
#         erf_min = erf.groupby(level=0).apply(lambda x: x - erf.loc[(x.index.get_level_values(0)[0], tmrel.loc[x.index.get_level_values(0)[0], 2010])]).reset_index(level=0, drop=True)
        
#         ### Add mean and std to new dataframes
#         tmin = -25
#         tmax = 35.1
#         erf_mean = pd.DataFrame({f'{disease}_mean' : erf_min.mean(axis=1)}, index=pd.MultiIndex.from_product([tz_range, np.arange(tmin,tmax,0.1).round(1)], names=['annual_temperature', 'daily_temperature']))
#         erf_std = pd.DataFrame({f'{disease}_std' : erf_min.std(axis=1)}, index=pd.MultiIndex.from_product([tz_range, np.arange(tmin,tmax,0.1).round(1)], names=['annual_temperature', 'daily_temperature']))
        
#         ### Fill NaN values with the first/last no NaN value 
#         erf_mean = erf_mean.groupby('annual_temperature').bfill().ffill()
#         erf_std = erf_std.groupby('annual_temperature').bfill().ffill()
        
#         ### Convert from ln(RR) to RR
#         erf_mean = erf_mean.apply(lambda x: np.exp(x)).unstack(level=0).round(4)
#         erf_std = erf_std.apply(lambda x: np.exp(x)).unstack(level=0).round(4)
        
#         ### Save csv without column/rows name to facilitate access in later computations
#         erf_mean.to_csv(f'X:\\user\\liprandicn\\Health Module\\ResponseFunctions\\erf_reformatted\\erf_{disease}_mean.csv', index=False, header=False)
#         erf_std.to_csv(f'X:\\user\\liprandicn\\Health Module\\ResponseFunctions\\erf_reformatted\\erf_{disease}_std.csv', index=False, header=False) ##### ???