import mortality_functions as mf

'''
Define scenario settings, ERF settings and file paths
'''

# Define path to main directory
wdir = 'X:\\user\\liprandicn\\mt-comparison\\Honda2014'
# ERA5 directory
era5_dir = 'X:\\user\\liprandicn\\Data\\ERA5\\t2m_daily'
# Define SSP scenario
ssp =  'SSP2' # SSP1, SSP2, SSP3, SSP5
# Define years range
years = range(2000,2020) 
# Define region classification
region_class = 'GBD_level3' # 'IMAGE26', 'GBD_level3',  for now...


'''
Set settings to extrapolate risk function
'''
# Choose between extrapolating risk function (True) or keeping it as in Honda 2014 (False)
extrap_erf = True
# Set extapolation range
temp_max = 40#int or None


'''
Run main model 
'''
mf.run_main(wdir,   # Working directory
            era5_dir, # ERA5 directory
            ssp,   # SSP scenario
            years,  #  Years range
            region_class,   # Region classification 
            extrap_erf,   # Extrapolate ERF(s)
            temp_max,   # Minimum temperature to extrapolate to
            )