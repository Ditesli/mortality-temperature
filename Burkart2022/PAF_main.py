import PAF_calculations as paf

'''
Define scenario settings, ERF settings and file paths
'''

# Path to main working directory
wdir = 'X:/user/liprandicn/mt-comparison/burkart2022/'

# Read file with IMAGE region names and corresponding countries
image_regions = pd.read_csv(f'{wdir}\\data\\IMAGE_regions\\IMAGE_regions.csv',  index_col=0, header=0)

# Read GBD file with mortality data
gbd_mortality = pd.read_csv(f'{wdir}\\data\\GBD_Data\\Mortality\\IHME-GBD_2021_DATA.csv')


# Define path to main directory
wdir = 'X:\\user\\liprandicn\\Health Impacts Model'
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
paf.run_main(wdir,   # Working directory
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