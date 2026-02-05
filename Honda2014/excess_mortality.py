import mortality_functions as mf

'''
Define scenario settings, ERF settings and file paths
'''

# Define path to main directory
wdir = 'X:/user/liprandicn/mt-comparison/honda2014/'
# Temperature data directory
temp_dir = 'X:/user/liprandicn/Data/ERA5/t2m_daily/'
# temp_dir = 'X:/user/scherrenbm/ModelDevelopment/IMAGE_Development/IMAGE_Daily_Indicators/SSP2/netcdf/'


# Define temperature data source
temp_source = 'ERA5' # MS (monthly statistics) or ERA5
# Define SSP scenario
ssp =  'SSP2' # SSP1, SSP2, SSP3, SSP5
# Define years range
years = range(2000,2025) 
# Define region classification
region_class = 'countries' # 'IMAGE26' or 'countries'


'''
Set settings to extrapolate risk function
'''
# Choose between extrapolating risk function (True) or keeping it as in Honda 2014 (False)
extrap_erf = False
# Set extapolation range, keep risk as cold temperatures as constant
temp_max = 40 #int or None
# Set optimal temperature range 
optimal_range = '1980-2010'  # 1980-2010 or 1990-2020


'''
Run main model 
'''
mf.run_main(wdir,   # Working directory
            temp_dir, # ERA5 directory
            temp_source,
            ssp,   # SSP scenario
            years,  #  Years range
            region_class,   # Region classification 
            optimal_range,
            extrap_erf,   # Extrapolate ERF(s)
            temp_max,   # Minimum temperature to extrapolate to
            )