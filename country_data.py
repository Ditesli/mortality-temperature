import pandas as pd
import numpy as np
import xarray as xr
from process_files import load_indicator, merge_xarrays

wdir = 'X:\\user\\liprandicn\\Health Impacts Model'


### -----------------------------------------------------------------------------------------------------
### Load and process csv files

temperature = load_indicator(f'{wdir}\\Climate_Data\\ERA5\\country_data\\country_statistics_1980-2019.csv',
                                iso3_col='location_name',
                                drop_cols=['location_id', 'location_name', 'CDD_18','CDD_22','CDD_24','CDD_26','HDD_18','HDD_22','HDD_24','HDD_26'],
                                rename_cols={'location_name': 'country'},
                                group_cols=['year', 'ISO3'],
                                attributes = [
                                    # Degree types
                                    *[(d, 'degree days', f'Heating or cooling degree days based on {d}', 'ERA5 reanalysis data') 
                                    for d in ['HDD_15', 'HDD_18_3', 'HDD_20', 'CDD_20', 'CDD_23_3', 'CDD_25']],
                                    # Bins
                                    *[(b, '5 degree bin', 'Bin containing the occurrence', 'ERA5 reanalysis data') 
                                    for b in ['bin_-20', 'bin_-20_-15', 'bin_-15_-10', 'bin_-10_-5', 'bin_-5_0', 'bin_0_5', 'bin_5_10',
                                                'bin_10_15', 'bin_15_20', 'bin_20_25', 'bin_25_30', 'bin_30_35', 'bin_35_40', 'bin_40']],
                                    # Other temperature variables
                                    ('temperature_mean', 'degree Celsius', 'Population-weighted annual mean temperature (t2m)', 'ERA5 reanalysis data'),
                                    ('temperature_std', 'degree Celsius', 'Population-weighted annual standard deviation of temperature (t2m)', 'ERA5 reanalysis data'),
                                    ('temperature_skewness', 'unitless', 'Skewness of temperature distribution (t2m)', 'ERA5 reanalysis data'),
                                    ('temperature_kurtosis', 'unitless', 'Kurtosis of temperature distribution (t2m)', 'ERA5 reanalysis data'),
                                    ('degree_days', 'degree days', 'Total degree days at 20 degrees setpoint', 'ERA5 reanalysis data'),
                                    ('HDD_20_squared', 'degree days', 'Squared Heating Degree Days at 20 degrees setpoint', 'ERA5 reanalysis data'),
                                    ('CDD_20_squared', 'degree days', 'Squared Cooling Degree Days at 20 degrees setpoint', 'ERA5 reanalysis data'),
                                    ])

climatology = load_indicator(f'{wdir}\\Climate_Data\\ERA5\\country_data\\ERA5_climatology_1970-2019.csv',
                               iso3_col='location_name',
                               cast_cols={'year': int},
                               filters={'year': range(1980, 2020)},
                               use_cols=['ISO3', 'year', 'climatology'],
                               group_cols=['year', 'ISO3'],
                               attributes=[('climatology', 'degree Celsius', '30 year running mean of temperature (t2m)', 'ERA5 reanalysis data')]
                               )

relative_humidity = load_indicator(f'{wdir}\\Climate_Data\\ERA5\\country_data\\ERA5_relative_humidity_statistics_1980-2019.csv',
                                    iso3_col='location_name',
                                    cast_cols={'year': int},
                                    use_cols=['ISO3', 'year', 'mean_relative_humidity', 'CDD_25_rh'],
                                    group_cols=['year', 'ISO3'],
                                    attributes=[('mean_relative_humidity', '%', 'Mean Relative Humidity', 'ERA5 reanalysis data subproduct'),
                                                ('CDD_25_rh', 'degree days * %', 'Cooling degree days weighted by RH', 'ERA5 reanalysis data subproduct')]
                                    )          
                               
mortality = load_indicator(f'{wdir}\\GBD_Data\\Mortality_Data\\IHME_GBD_2021_DATA_1980-2021_LEVEL_3-4.csv',
                            iso3_col='location_name',
                            filters={'year': range(1980, 2020)},
                            use_cols=['year', 'ISO3', 'age_name', 'cause_name', 'val'],
                            rename_cols={'age_name': 'age_group', 'val': 'total_mortality'},
                            rename_entries={'age_group': {' years': '', '<5': '0-4'}},
                            group_cols=['year', 'ISO3', 'age_group', 'cause_name'],
                            attributes=[('total_mortality', 'Total mortality', 'Total number of deaths for a specific cause', 'GBD 2021 data')]
                            )
          
gdp = load_indicator(f'{wdir}\\Socioeconomic_Data\\GDP\\GDP_cl.csv',
                        filters={'SSP': 'SSP1', 'Year': range(1980, 2020)},
                        use_cols=['ISO3', 'Year', 'value'],
                        rename_cols={'value': 'gdp', 'Year': 'year'},
                        group_cols=['ISO3', 'year'],
                        attributes=[('gdp', 'USD', 'GDP per capita in current USD', 'IMAGE Data')]
                        )

health_expenditure = load_indicator(f'{wdir}\\Socioeconomic_data\\Health\\WHO_CHE_USD.csv',
                                    filters={'Period': range(1980, 2020)},                                    
                                    use_cols=['SpatialDimValueCode', 'Period', 'Value'],
                                    rename_cols={'SpatialDimValueCode': 'ISO3', 'Period': 'year', 'Value': 'health_expenditure'},
                                    group_cols=['ISO3', 'year'],
                                    attributes=[('health_expenditure', 'USD', 'Current health expenditure per capita in current USD', 'WHO data')]
                                    )
                                    
medical_doctors = load_indicator(f'{wdir}\\Socioeconomic_data\\Health\\WHO_medical_doctors_pc.csv',
                                cast_cols={'Value': float},
                                filters={'Period': range(1980, 2020)},
                                use_cols=['SpatialDimValueCode', 'Period', 'Value'],
                                rename_cols={'SpatialDimValueCode': 'ISO3', 'Period': 'year', 'Value': 'medical_doctors'},
                                group_cols=['ISO3', 'year'],
                                attributes=[('medical_doctors', 'Doctors per 10000 people', 'Number of medical doctors per 1000 people', 'WHO data')]
                                )

hdi = load_indicator(f'{wdir}\\Socioeconomic_data\\Development\\hdr-data.xlsx',
                    filters={'year': range(1980, 2020)},
                    use_cols=['countryIsoCode', 'year', 'value'],
                    rename_cols={'countryIsoCode': 'ISO3', 'value': 'HDI'},
                    group_cols=['ISO3', 'year'],
                    attributes=[('HDI', 'Index', 'Human Development Index (HDI)', 'UNDP data')]
                    )

schooling_years = load_indicator(f'{wdir}\\Socioeconomic_data\\Development\\hdr-data_mean_years_schooling.xlsx',
                                filters={'year': range(1980, 2020)},
                                use_cols=['countryIsoCode', 'year', 'value'],
                                rename_cols={'countryIsoCode': 'ISO3', 'value': 'schooling_years'},
                                group_cols=['ISO3', 'year'],
                                attributes=[('schooling_years', 'Years', 'Mean years of schooling', 'UNDP data')]
                                )

gini = load_indicator(f'{wdir}\\Socioeconomic_data\\Development\\GINI\\API_SI.POV.GINI_DS2_en_csv_v2_2566.csv',
                        skiprows=3,
                        drop_cols= ['Country Name', 'Unnamed: 69', 'Indicator Name', 'Indicator Code'],
                        melt_df={'id_vars': ['Country Code'], 'var_name': 'year', 'value_name': 'GINI'},
                        cast_cols={'year': int},
                        filters={'year': range(1980, 2020)},
                        rename_cols={'Country Code': 'ISO3'},
                        group_cols=['ISO3', 'year'],
                        attributes=[('GINI', 'Index', 'Gini index of income inequality', 'World Bank data')]
                        )
                       
education_expenditure = load_indicator(f'{wdir}\\Socioeconomic_data\\Development\\Gov_exp_education\\API_SE.XPD.TOTL.GD.ZS_DS2_en_csv_v2_2627.csv',
                                        skiprows=3,
                                        drop_cols=['Country Name', 'Unnamed: 69', 'Indicator Name', 'Indicator Code'],
                                        melt_df={'id_vars': ['Country Code'], 'var_name': 'year', 'value_name': 'education_expenditure'},
                                        filters={'year': range(1980, 2020)},
                                        cast_cols={'year': int},
                                        rename_cols={'Country Code': 'ISO3'},
                                        group_cols=['ISO3', 'year'],
                                        attributes=[('education_expenditure', 'Percentage of GDP', 'Government expenditure on education as a percentage of GDP', 'World Bank data')]
                                        )     
                       
urban_share = load_indicator(f'{wdir}\\Socioeconomic_Data\\Population\\urban-population-of-total-population\\Urban population (% of total population).csv',
                            filters={'Disaggregation': 'total', 'Year': range(1980, 2020)},
                            use_cols=['Country Code', 'Year', 'Value'],   
                            rename_cols={'Country Code': 'ISO3', 'Year': 'year', 'Value': 'urban_share'},
                            group_cols=['ISO3', 'year'],
                            attributes=[('urban_share', '%', 'Rate of population living in urban areas', 'World Bank data')]
                            )

population = load_indicator(f'{wdir}\\Socioeconomic_Data\\Population\\unpopulation_dataportal.csv',
                            filters={'Time': range(1980, 2020)},
                            use_cols=['Iso3', 'Time', 'Age', 'Value'],
                            rename_cols={'Iso3': 'ISO3', 'Time': 'year', 'Value': 'population', 'Age': 'age_group'},
                            age_standar=True,
                            group_cols=['ISO3', 'year', 'age_group'],
                            attributes=[('population', 'Population', 'Population for a specific age group and year', 'UN population data')]
                           )

### ------------------------------------------------------------------------------------------------------
### Merge all dataframes and process

datasets = [
    v for k, v in globals().items()
    if isinstance(v, (xr.Dataset, xr.DataArray))
]

merge_xarrays(datasets)