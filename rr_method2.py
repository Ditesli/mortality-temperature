import numba as nb
import numpy as np
import xarray as xr
import pandas as pd
from generate_daily_temp import daily_temp_interp, create_noise, daily_temp_era5
from rr_calculation import get_yearly_rr,  get_spatial_mean
from read_files import get_annual_pop, read_climate_data, read_greg_and_TempZone, open_all_diseases_data_method2


### -----------------------------------------------------------------------------------------
### Load relevant files and define parameters

### Import Exposure Response Functions for one disease
diseases, erf = open_all_diseases_data_method2()
### Get climate, IMAGE regions, Temperature Zone and Population data
ssp = 'SSP2_CP'
gcm_diff, gcm_start, gcm_end, cc_path_mean = read_climate_data()
greg, temp_zone_data = read_greg_and_TempZone()
pop_ssp = get_annual_pop(ssp) 

### Define scenario parameters
years =  [2010,2019] # range(2010,2101)
std_values = [5] # [1,5,10]
ccategories = ['C1'] # ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
print(f'Years: {years[0]}-{years[-1]}, std_values: {std_values}, ccategories: {ccategories}')


### ------------------------------------------------------------------------------------------
### Run main function

print('Start of the loop')

for std_value in std_values:
    # Generate noise file
    noise, noise_leap = create_noise(std_value)
    
    for ccategory in ccategories:
        # Final dataframe
        final_rr = pd.DataFrame(index=range(1,27), columns=pd.MultiIndex.from_product([years, diseases]))      
        
        for year in years:
            print(f'{year} started')
            ### Generate annual daily temperature using dummy data (first line) or ERA5 reanalysis data (upon avaliability in folder)
            daily_temp, num_days = daily_temp_interp(ccategory, year, cc_path_mean, gcm_diff, gcm_start, gcm_end, noise, noise_leap)
            #daily_temp, num_days = daily_temp_era5(year, pop_ssp)
            ### Select yearly population
            pop_ssp_year = pop_ssp.sel(time=str(year)).mean('time').GPOP.values
            
            for disease in diseases:
                ### Calculate 3d-array with global and daily rr
                yearly_rr = get_yearly_rr(temp_zone_data, erf[disease], daily_temp)
                ### Calculate the temporal mean of the relative risk values
                rr_temporal_mean = np.nanmean(yearly_rr, axis=0) 
                print(f"{year}: Annual mean {disease} - calculated")
                ### Iterate over all IMAGE regions
                for region in range(1,27):
                    # Calculate the population weighted mean of the relative risk values for a given IMAGE region
                    final_rr.at[region, (year, disease)] = get_spatial_mean(rr_temporal_mean, pop_ssp_year, greg, region)
            print(f"{year}: RR generated")
        
        final_rr.to_csv(f'X:\\user\\liprandicn\\Health Module\\output\\rr_m2_all-diseases_{ccategory}_2010-2019_{std_value}std.csv')
        # final_rr.to_csv(f'X:\\user\\liprandicn\\Health Module\\output\\rr_m2_{disease}_{ccategory}_2010-2100_{std_value}std.csv')
        #final_rr.to_csv(f'X:\\user\\liprandicn\\Health Module\\output\\rr_m2_all-diseases_{year}_era5.csv')