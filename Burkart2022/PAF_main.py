import PAF_calculations as paf


'''
Define scenario settings, ERF settings and file paths
'''
# Path to main working directory
wdir = 'X:/user/liprandicn/mt-comparison/burkart2022/'
era5dir = 'X:/user/liprandicn/Data/ERA5/t2m_daily/'

# Define SSP scenario
ssp =  'SSP2' # SSP1, SSP2, SSP3, SSP5
# Define years range
years = range(2000,2020) 
# Define region classification
region_class = 'GBD_level3' # 'IMAGE26', 'GBD_level3',  for now...


'''
Set settings to select ERF
'''
# Choose between keeping temperature zones (False) or using single mean ERF for all zones (True)
single_erf = False
# Choose between extrapolating ERFs (True) or keeping them as in Burkart 2021 (False)
extrap_erf = False

# Choose between calculating mean RR (True), specific or random draw (False). 
# Only select one!!!
mean = True 
random_draw = False
draw = None # None or integer between 0 and 999


'''
Main model 
'''
paf.run_main(wdir,   # Working directory
             era5dir,
             ssp,   # SSP scenario
             years,  #  Years range
             region_class,   # Region classification 
             single_erf,   # Single ERF or use temperature zones
             extrap_erf,   # Extrapolate ERF(s)
             mean,   # Mean RR of the 1000 draws
             random_draw,   # Random draw
             draw   # Specific draw
             )