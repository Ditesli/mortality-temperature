import PAF_calculations as pafc

'''
File paths
'''
# Define path to main directory
wdir = 'X:\\user\\liprandicn\\mt-comparison\\Burkart2022'
# Define climate data directory
climate_dir = 'X:\\user\\liprandicn\\Data\\ERA5\\t2m_daily'

'''
Scenario settings
'''
# Define SSP scenario
ssp =  'SSP2' # SSP1, SSP2, SSP3, SSP5
# Define years range
years = range(2000,2020) 
# Define region classification
region_class = 'GBD_level3' # 'IMAGE26', 'GBD_level3',  for now...


'''
Optional settings C-categories
If empty, the model will run for ERA5 historical data
'''
# Define C-category from AR6
ccategories = [] # ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'] OR []
# Define climate variability
std_value = None # [1,5,10] OR None

'''
Set settings to select ERF
'''
# Choose between keeping temperature zones (False) or using single mean ERF for all zones (True)
single_erf = False
# Choose between extrapolating ERFs (True) or keeping them as in Burkart 2021 (False)
extrap_erf = False
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
pafc.run_main(wdir,   # Working directory
              climate_dir, # Climate data directory
              ssp,   # SSP scenario
              years,  #  Years range
              region_class,   # Region classification 
              ccategories,   # C-categories if used AR6 temperature data
              std_value,   # Climate variability if used AR6 temperature data
              single_erf,   # Single ERF or use temperature zones
              extrap_erf,   # Extrapolate ERF(s)
              all_diseases,   # Burkart (2021) diseases or only metabolic and cardiovascular ones
              mean,   # Mean RR of the 1000 draws
              random_draw,   # Random draw
              draw   # Specific draw
              )