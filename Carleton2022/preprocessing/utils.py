import pandas as pd
import numpy as np


def exposure_response_function(tas1, tas2, tas3, tas4, t): 
    
    '''
    This function generates the mortality-temperature exposure response functions using 
    the covariates of loggdppc (logarithm of GDP per capita) and climtas (climate mean), 
    and the constraints imposed to the functions in Carleton et al. (2022).
    The function takes as input the coefficients tas1, tas2, tas3, and tas4,
    which are derived from the paper's regression analysis, and a temperature array t.
    It returns the mortality values corresponding to each temperature in the array t.
    '''
   
    # Generate raw fourth degree polynomial function
    raw = tas1*t + tas2*t**2 + tas3*t**3 + tas4*t**4   
    
    # Find the temperature at which the mortality is minimized within the range of 10 to 30 degrees Celsius
    tmin = t[np.argmin(raw[np.where(np.isclose(t, 10.0, atol=0.05))[0][0]:np.where(np.isclose(t, 30.0, atol=0.05))[0][0]]) + np.where(np.isclose(t, 10.0, atol=0.05))[0][0]]
    
    # Vertically shift the function to make Tmin be at zero
    mortality = raw - tas1*tmin - tas2*tmin**2 - tas3*tmin**3 - tas4*tmin**4  
    
    # Impose weak monotonicity (from the original paper)
    t_left = t[t < tmin]  
    t_right = t[t > tmin]
    
    # Apply weak monotonicity constraints
    if len(t_left) > 0:
        for i in range(len(t_left) - 1, -1, -1):
            mortality[i] = max(mortality[i], mortality[i + 1])
    
    if len(t_right) > 0:
        for i in range(len(t_left)+1, len(mortality)):
            mortality[i] = max(mortality[i-1], mortality[i]) 
    
    return mortality



def open_predictors(file_path):
    
    '''
    Select mortality per age group for 2015 (No Adaptation)
    '''
    
    df = pd.read_csv(file_path, skiprows=13)
    df = df[df['year'] == 2015]
    df = df.reset_index()
    
    return df



def get_tmin(tas1, tas2, tas3, tas4, t): 
    
    '''
    Define function to get the Tmin for each RF
    '''
    
    # Generate raw fourth degree polynomial function
    raw = tas1*t + tas2*t**2 + tas3*t**3 + tas4*t**4   
    
    # Find the temperature at which the mortality is minimized within the range of 10 to 30 degrees Celsius
    tmin = t[np.argmin(raw[np.where(np.isclose(t, 10.0, atol=0.05))[0][0]:np.where(np.isclose(t, 30.0, atol=0.05))[0][0]]) + np.where(np.isclose(t, 10.0, atol=0.05))[0][0]]    
    
    return tmin