import paf_calculations as rrc

'''
Define scenario settings, ERF settings and file paths
'''

# Define path to main directory
wdir = 'X:\\user\\liprandicn\\Health Impacts Model'
# Define SSP scenario
ssp =  'SSP2' # SSP1, SSP2, SSP3, SSP5
# Define years range
years = range(2000,2020) 
# Define region classification
region_class = 'GBD_level3' # 'IMAGE26', 'GBD_level3',  for now...

'''
Optional settings for predefined years and C-categories
If empty, the model will run for ERA5 historical data
'''

# Define C-category warming scenario
# ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'] OR []
ccategories = []
# Define climate variability
# [1,5,10] OR None
std_value = None


'''
Set settings to select ERF
'''
# Choose between keeping temperature zones (False) or using single mean ERF for all zones (True)
single_erf = False
# Choose between extrapolating ERFs (True) or keeping them as in Burkart 2021 (False)
extrap_erf = True
# Set extapolation range
temp_max = 45 #int or None
temp_min = -22 # int or None
# Choose between calculating RR for all diseases (True) or only metabolic and cardiovascular ones (False)
all_diseases = True 

# Choose between calculating mean RR (True), specific or random draw (False). 
# Only select one!!!
mean = True 
random_draw = False
draw = None # None or integer between 0 and 999


'''
Run main model 
'''
rrc.run_main(wdir,   # Working directory
             ssp,   # SSP scenario
             years,  #  Years range
             region_class,   # Region classification 
             ccategories,   # C-categories if used AR6 temperature data
             std_value,   # Climate variability if used AR6 temperature data
             single_erf,   # Single ERF or use temperature zones
             extrap_erf,   # Extrapolate ERF(s)
             temp_max,   # Minimum temperature to extrapolate to
             temp_min,   #Maximum temperature to extrapolate 
             all_diseases,   # Burkart (2021) diseases or only metabolic and cardiovascular ones
             mean,   # Mean RR of the 1000 draws
             random_draw,   # Random draw
             draw   # Specific draw
             )