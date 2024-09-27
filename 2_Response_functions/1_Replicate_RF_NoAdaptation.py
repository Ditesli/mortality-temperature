import pandas as pd
import numpy as np
import os

carleton_path = 'D:\\data\\2_projection\\3_impacts\\main_specification\\raw\\single\\rcp85\\CCSM4\\low\\SSP3\\' #Read covariates

t = np.arange(-50, 60.1, 0.1) ### This determines the resolution of the temperature range
t = t.round(1)

def response(tas1, tas2, tas3, tas4, t): ### This function calculates mortality using the covariates loggdppc and climtas, and the constraints imposed to the functions
    raw = tas1*t + tas2*t**2 + tas3*t**3 + tas4*t**4   
    tmin = t[np.argmin(raw[np.where(np.isclose(t, 10.0, atol=0.05))[0][0]:np.where(np.isclose(t, 30.0, atol=0.05))[0][0]]) + np.where(np.isclose(t, 10.0, atol=0.05))[0][0]]
    mortality = raw - tas1*tmin - tas2*tmin**2 - tas3*tmin**3 - tas4*tmin**4
    
    t_left = t[t < tmin]
    t_right = t[t > tmin]
    
    if len(t_left) > 0:
        for i in range(len(t_left) - 1, -1, -1):
            mortality[i] = max(mortality[i], mortality[i + 1])
    
    if len(t_right) > 0:
        for i in range(len(t_left)+1, len(mortality)):
            mortality[i] = max(mortality[i-1], mortality[i]) 
    
    return mortality


'''Select mortality per age group for 2015 (No Adaptation)'''
oldest = pd.read_csv((carleton_path + 'mortality-allcalcs-Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1-oldest.csv'), skiprows=13)
oldest = oldest.dropna()
oldest = oldest[oldest['year'] == 2015]
oldest = oldest.reset_index()

older = pd.read_csv((carleton_path + 'mortality-allcalcs-Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1-older.csv'), skiprows=13)
older = older.dropna()
older = older[older['year'] == 2015]
older = older.reset_index()

young = pd.read_csv((carleton_path + 'mortality-allcalcs-Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1-young.csv'), skiprows=13)
young = young.dropna()
young = young[young['year'] == 2015]
young = young.reset_index()


'''Generate dataframes'''
groups = ['oldest', 'older', 'young']
df_groups = [oldest, older, young]

for group, df_group in zip(groups, df_groups): # Iterate for each age group
    responses = []
    for i in range(len(df_group)): # Generate the response functions and append
        mortality = response(df_group['tas'][i], df_group['tas2'][i], df_group['tas3'][i], df_group['tas4'][i], t)
        responses.append(mortality.round(2))
    df = pd.DataFrame(responses, columns=[f"{temp:.1f}" for temp in t])  # Round column names
    df_merge = pd.concat([df_group['region'], df], axis=1, join='inner')  # Add the region columns
    df_merge.to_csv(os.path.join(os.getcwd(), '2_Response_functions', f'RF_NoAdaptation_{group}.csv'))




