'''
Generate country-level global climatologies, defined here as 30 year mean of the surface temperature and
generate country-level statistics of daily temperatures
'''

from temperature_functions import melt_and_save, get_data_era5, get_data_isimip, get_country_mask, compute_country_statistics, compute_temperature_statistics, compute_rh_statistics
from read_files import get_annual_pop, read_shp_countries
from generate_daily_temp import daily_temp_era5
import numpy as np
# import pandas as pd


wdir = 'X:\\user\\liprandicn\\Health Impacts Model'
calc_type = 'stats'  # Set data type: 'stats' for daily temperature data, 'clim' for climatology data
data_type = 'ERA5'
climate_variable = 'temperature'  # Choose between 'temperature' and 'relative_humidity'
years = range(1980,2020)


### -----------------------------------------------------------------------------------------------------
### Open files 
  
# Define the scenario and open population data. 
pop_ssp = get_annual_pop('SSP2_CP', coarse=True, start_year=years[0], end_year=years[-1])

# Open country-level geometries and rasterize as mask
gdf_countries, country_statistics = read_shp_countries()
country_mask = get_country_mask(pop_ssp, gdf_countries)

    
### ----------------------------------------------------------------------------------------------------
### Calculations

if data_type == 'ERA5':
    
    if climate_variable == 'temperature':
        
        for year in years:
            
            variable = 't2m'
            variable_type = 'mean'
    
            temperature = get_data_era5(calc_type, pop_ssp, year, daily_temp_era5, variable, variable_type)
            
            if calc_type == 'clim':
                # Create dataframe with country climatology
                intermediate_data = compute_country_statistics(temperature, pop_ssp, country_mask, year, country_statistics, f'{year}_climatology')

            if calc_type == 'stats':
                # Calculate statistics: mean, std, skewness, kurtosis, degree days
                country_statistics = compute_temperature_statistics(temperature, pop_ssp, country_mask, year, country_statistics, climate_variable,
                                                                 'valid_time', mean=True, std=True, skewness=False, kurtosis_=False, 
                                                                 image_degree_days=False, base_degree_days=False, bins=False)

        melt_and_save(country_statistics, calc_type, data_type, climate_variable, wdir, years, model=None, scenario=None) 
        
        
    if climate_variable == 'relative_humidity':
        
        for year in years:           
            
            country_statistics = compute_rh_statistics(calc_type, daily_temp_era5, pop_ssp, country_mask, year, country_statistics)
            
        melt_and_save(country_statistics, calc_type, data_type, climate_variable, wdir, years, model=None, scenario=None)
            
            
if data_type =='ISIMIP':
    
    for model in ['GFDL', 'IPSL', 'MPI', 'MRI', 'UKESM']:
        for scenario in ['ssp126', 'ssp370']:
            for year in years:
                print(f'Processing {model} - {scenario} - {year}')
                
                # Get temperature data for each model and scenario
                temperature,_ = get_data_isimip(calc_type, pop_ssp, model, scenario, year)
                
                if calc_type == 'clim':
                    country_statistics = compute_country_statistics(temperature, pop_ssp, country_mask, year, country_statistics, f'{year}_climatology')
                    
                if calc_type == 'stats':
                    country_statistics = compute_temperature_statistics(temperature, pop_ssp, country_mask, year, country_statistics, climate_variable,
                                                                 'time', mean=True, std=True, skewness=False, kurtosis_=False, 
                                                                 image_degree_days=False, base_degree_days=False, bins=False)
                    
            melt_and_save(country_statistics, calc_type, data_type, climate_variable, wdir, years, model=model, scenario=scenario)