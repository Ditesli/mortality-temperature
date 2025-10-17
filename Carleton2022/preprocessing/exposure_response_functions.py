import utils
import pandas as pd
import numpy as np

wdir = 'X:\\user\\liprandicn\\mt-comparison\\Carleton2022\\data\\'

### Determine the resolution of the temperature range
t = np.arange(-50, 60.1, 0.1).round(1)

### Age group names
age_groups = ['oldest', 'older', 'young']

### Open predictors dataframes

oldest = utils.open_predictors(wdir+'main_specification/mortality-allcalcs-Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1-older.csv')
older = utils.open_predictors(wdir+'main_specification/mortality-allcalcs-Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1-oldest.csv')
young = utils.open_predictors(wdir+'main_specification/mortality-allcalcs-Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1-young.csv')

df_groups = [oldest, older, young]


### ------------------------------------------------------------------------------

'''
Generate Exposure Response Functions without Adaptation using covaraiates from Carleton et al. (2022)
'''

# Iterate for each age group
for group, df_group in zip(age_groups, df_groups): 
    responses = []
    
    # Generate the response functions and append
    for i in range(len(df_group)):
        mortality = utils.exposure_response_function(df_group['tas'][i], df_group['tas2'][i], df_group['tas3'][i], df_group['tas4'][i], t)
        responses.append(mortality.round(2))
        
    # Round column names
    df = pd.DataFrame(responses, columns=[f"{temp:.1f}" for temp in t])  
    
    # Add the region columns
    df_merge = pd.concat([df_group['region'], df], axis=1, join='inner')  

    # Save csv file
    df_merge.to_csv(f'{wdir}/exposure_response_functions/erf_no-adapt_{group}.csv')


### ------------------------------------------------------------------------------

''' 
Generate Tmin (minimum of the response functions per impact region)
This step is useful for Responde Functions with Adaptation and Hot&Cold Mortality
'''

# Define dataframe to store Tmin values
df = pd.DataFrame(oldest['region'])

# Add empty columns for Tmin values of each age group
df['Tmin oldest'] = ''
df['Tmin older'] = ''
df['Tmin young'] = ''

# Get T_min for all age groups and IR
for i in range(len(oldest)):
    df.iloc[i,1] = utils.get_tmin(oldest['tas'][i], oldest['tas2'][i], oldest['tas3'][i], oldest['tas4'][i], t)
    df.iloc[i,2] = utils.get_tmin(older['tas'][i], older['tas2'][i], older['tas3'][i], older['tas4'][i], t)
    df.iloc[i,3] = utils.get_tmin(young['tas'][i], young['tas2'][i], young['tas3'][i], young['tas4'][i], t)

# Save csv file
df.to_csv(f'{wdir}/exposure_response_functions/T_min.csv') 