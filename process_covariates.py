import pandas as pd
import numpy as np
import xarray as xr
import os
import country_converter as coco ###

wdir = 'X:\\user\\liprandicn\\Health Impacts Model'



def generate_panel_data(wdir):
    
    # Load covariates
    gdppc, education, fertility, gini, population, mortality, tfu25 = load_covariates(wdir)
    
    # Calculate 10-year lagged GDP per capita
    gdppc = gdppc.sort_values(["ISO3", "year"])
    gdppc['gdplag']=gdppc.groupby("ISO3", group_keys=False)["gdppc"].apply(lambda x: x.shift(1).rolling(window=10, min_periods=10).mean())
    gdppc = gdppc[gdppc['year']>=1980].reset_index(drop=True)

    # Merge datasets
    variables = mortality.merge(population, on=['ISO3', 'year', 'age', 'sex'], how='left')
    variables = variables.merge(education, on=['ISO3', 'year'], how='left')
    variables = variables.merge(fertility, on=['ISO3', 'year'], how='left')
    variables = variables.merge(gini, on=['ISO3', 'year'], how='left')
    variables = variables.merge(gdppc, on=['ISO3', 'year'], how='left')
    variables = variables.merge(tfu25, on=['ISO3', 'year'], how='left')

    low_pop_countries = variables[(variables['age'] == 'All ages') & (variables['population'] < 1e6)][['ISO3']].ISO3.unique()
    variables = variables[~variables['ISO3'].isin(low_pop_countries)]   

    age_map = {
        '0-4': ['0-4'],
        '5-14': ['5-9', '10-14'],
        '15-24': ['15-19', '20-24'],
        '25-34': ['25-29', '30-34'],
        '35-44': ['35-39', '40-44'],
        '45-54': ['45-49', '50-54'],
        '55-64': ['55-59', '60-64'],
        '65-74': ['65-69', '70-74'],
        '75-84': ['75-79', '80-84'],
        '85+': ['85+'],
        'All ages': ['All ages']
    }

    # Invert
    age_reverse_map = {
        original: broad for broad, group_list in age_map.items() for original in group_list
    }

    variables['age_group'] = variables['age'].map(age_reverse_map)

    variables = variables.groupby(['ISO3', 'year', 'age_group', 'sex', 'cause'], as_index=False).agg({
        'population': 'sum', 'total_mortality': 'sum',
        'gdppc': 'first', 'cause': 'first', 'sex': 'first', 'education': 'first', 
        'TFR': 'first', 'GINI': 'first', 'country': 'first', 'TFU25': 'first', 'gdplag': 'first'
    })

    variables['relative_mortality'] = variables['total_mortality']/variables['population'] * 1e5
    variables['logmortality'] = np.log(variables['total_mortality'] + 1)  # Adding 1 to avoid log(0)
    variables['loggdppc'] = np.log(variables['gdppc'])
    variables['gdppc_2'] = variables['gdppc'] ** 2
    variables['loggdppc_2'] = variables['loggdppc']**2
    variables['loggdplag'] = np.log(variables['gdplag'])
    variables['loggdplag_2'] = variables['loggdplag']**2
    
    # Create SDI
    variables = create_sdi(variables, 'loggdplag', 'education', 'TFU25')

    return variables



def load_covariates(wdir):
    
    gdppc = load_variable(f'{wdir}\\data\\Socioeconomic_data\\GDP\\API_NY.GDP.PCAP.KD_DS2_en_csv_v2_37434\\API_NY.GDP.PCAP.KD_DS2_en_csv_v2_37434.csv',
                        skiprows=3,
                        drop_cols= ['Unnamed: 69', 'Indicator Name', 'Indicator Code'],
                        melt_df={'id_vars': ['Country Code', 'Country Name'], 'var_name': 'year', 'value_name': 'gdppc'},
                        cast_cols={'year': int},
                        filters={'year': range(1970, 2020)},
                        rename_cols={'Country Code': 'ISO3', 'Country Name': 'country'},
                        )

    education = load_variable(f'{wdir}\\data\\Socioeconomic_data\\Education\\hdr-data_mean_years_schooling.xlsx',
                                filters={'year': range(1980, 2020)},
                                use_cols=['countryIsoCode', 'year', 'value'],
                                rename_cols={'countryIsoCode': 'ISO3', 'value': 'education'},
                                group_cols=['ISO3', 'year'],
                                )

    fertility = load_variable(f'{wdir}\\data\\Socioeconomic_Data\\Fertility\\API_SP.DYN.TFRT.IN_DS2_EN_csv_v2_37712\\API_SP.DYN.TFRT.IN_DS2_EN_csv_v2_37712.csv',
                                skiprows=3,
                                drop_cols= ['Country Name', 'Unnamed: 69', 'Indicator Name', 'Indicator Code'],
                                melt_df={'id_vars': ['Country Code'], 'var_name': 'year', 'value_name': 'TFR'},
                                cast_cols={'year': int},
                                filters={'year': range(1980, 2020)},
                                rename_cols={'Country Code': 'ISO3'}
                                )

    gini = load_variable(f'{wdir}\\data\\Socioeconomic_data\\GINI\\GINI_World_Bank\\API_SI.POV.GINI_DS2_en_csv_v2_2566.csv',
                            skiprows=3,
                            drop_cols= ['Country Name', 'Unnamed: 69', 'Indicator Name', 'Indicator Code'],
                            melt_df={'id_vars': ['Country Code'], 'var_name': 'year', 'value_name': 'GINI'},
                            cast_cols={'year': int},
                            filters={'year': range(1980, 2020)},
                            rename_cols={'Country Code': 'ISO3'},
                            )

    population = load_variable(f'{wdir}\\data\\Socioeconomic_Data\\Population\\unpopulation_dataportal_20250731162658.csv',
                                filters={'Time': range(1980, 2020)},
                                use_cols=['Iso3', 'Time', 'Age', 'Value', 'Sex'],
                                rename_cols={'Iso3': 'ISO3', 'Time': 'year', 'Value': 'population', 'Age': 'age', 'Sex':'sex'},
                                rename_entries={'sex':{'Both sexes':'Both'}},
                                age_standar=True,
                                )
    
    mortality = load_variable(f'{wdir}\\data\\GBD_Data\\Mortality\\GBD21_1980-2021_level2-3_sex3_age5.csv',
                            iso3_col='location',
                            filters={'year': range(1980, 2020), 'sex':['Male', 'Female']},
                            use_cols=['year', 'ISO3', 'age', 'cause', 'val', 'sex'],
                            rename_cols={'val': 'total_mortality'},
                            rename_entries={'age': {' years': '', '<5': '0-4'}},
                            )  
    
    tfu25 = load_variable(f'{wdir}\\data\\Socioeconomic_Data\\Fertility\\IHME_GBD_2019_FERTILITY_1950_2019_TFU25\\IHME_GBD_2019_FERTILITY_1950_2019_TFU25_Y2020M08D05.csv',
                      gbd_locations='location_id',
                      iso3_col='location_name',
                      drop_cols=['location_id', 'location_name', 'sex_id', 'sex', 'age_group_id', 'age_group_name', 'measure_id', 'measure_name', 
                                 'metric_id', 'metric_name', 'lower', 'upper'],
                      cast_cols={'year_id': int, 'val': float},
                      filters={'year_id': range(1980, 2020)},
                      rename_cols={'val': 'TFU25', 'year_id': 'year'},
                      )
    
    return gdppc, education, fertility, gini, population, mortality, tfu25



def create_iso3_column(df, column_name):
    
    '''
    Use the country_converter package to convert country names in a DataFrame column to ISO3 codes
    for facilitating data merging
    '''
    
    # Remove aggregated regions that are not countries
    region_names = ['Africa (R10)', 'Asia (R5)', 'China (R9)', 'China+ (R10)', 'India (R9)',  'India+ (R10)', 
                    'Europe (R10)', 'European Union (R9)', 'Latin America (R10)', 'Latin America (R5)', 
                    'Latin America (R9)', 'Middle East & Africa (R5)', 'Middle East & Africa (R9)',
                    'Middle East (R10)', 'North America (R10)', 'OECD & EU (R5)', 'Other Asia (R9)', 'Other OECD (R9)', 
                    'Pacific OECD (R10)', 'Reforming Economies (R10)', 'Reforming Economies (R5)', 
                    'Reforming Economies (R9)', 'Rest of Asia (R10)', 'USA (R9)', 'World', 'Channel Islands']
    
    df = df[~df[column_name].isin(region_names)].copy()
    
    # Convert country names to ISO3 codes
    unique_locations = df[column_name].unique()
    conversion_dict = {location: coco.convert(names=location, to='ISO3') for location in unique_locations}
    df['ISO3'] = df[column_name].map(conversion_dict)
    
    return df



def load_variable(file_path, sheet_name=None, skiprows=0, gbd_locations=None, iso3_col=None, filters=None, 
                  exclude_filters=False, drop_cols=None, use_cols=None, rename_cols=None, rename_entries=None, 
                  cast_cols=None, melt_df=None, age_standar=False, group_cols=None, attributes=None):
    
    '''
    Load csv files with country level covariates
    '''
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(file_path, skiprows=skiprows)
    elif file_ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path, skiprows=skiprows, sheet_name=sheet_name or 0)
    else:
        raise ValueError(f"Unsuported file format: {file_ext}")
    
    if gbd_locations:
        gbd_loc_hierarchy = pd.read_csv(f'{wdir}\\data\\GBD_Data\\GBD_locations\\GBD_2021_ALL_LOCATIONS_HIERARCHIES_Y2024M05D16.csv')   
        gbd_loc_hierarchy = gbd_loc_hierarchy[gbd_loc_hierarchy['Level'] == 3][['Location ID', 'Location Name']]
        df = df[df[gbd_locations].isin(gbd_loc_hierarchy['Location ID'])]

    if iso3_col:
        df = create_iso3_column(df, iso3_col) 
        
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, axis=1)
        
    if melt_df:
        df = pd.melt(
            df,
            id_vars=melt_df['id_vars'],
            var_name=melt_df.get('var_name', 'variable'),
            value_name=melt_df.get('value_name', 'value')
        ) 
        
    if cast_cols:
        for col, dtype in cast_cols.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        
    if filters:
        for col, vals in filters.items():
            if isinstance(vals, (list, range, set, tuple)):
                if exclude_filters:  
                    df = df[~df[col].isin(vals)]
                else:
                    df = df[df[col].isin(vals)]
            else:
                if exclude_filters:
                    df = df[df[col] != vals]
                else:
                    df = df[df[col] == vals]
    
    if use_cols:
        df = df[use_cols]  
        
    if rename_cols:
        df.rename(columns=rename_cols, inplace=True)
        
    if rename_entries:
        for col, replace_list in rename_entries.items():
            if col in df.columns:
                for old, new in replace_list.items():
                    df[col] = df[col].str.replace(old, new, regex=False)
    
    # Group specific age entries into 95+ and All ages
    if age_standar and all(col in df.columns for col in ['age', 'ISO3', 'year', 'population']):
        
        age85_filter = df['age'].isin(['85-89', '90-94', '95-99', '100+'])
        
        if age85_filter.any():
            df_85plus = df[age85_filter].groupby(['ISO3', 'year', 'sex'], as_index=False)['population'].sum()
            df_85plus['age'] = '85+'
            df = df[~age85_filter]
            df = pd.concat([df, df_85plus], ignore_index=True)

        df_allages = df.groupby(['ISO3', 'year', 'sex'], as_index=False)['population'].sum()
        df_allages['age'] = 'All ages'
        df = pd.concat([df, df_allages], ignore_index=True)
    
    print(f"{os.path.basename(file_path)} correctly imported")
    
    return df
        
        
        
def create_sdi(df, income, education, fertility):
    
    ''''
    Create Sociodemographic index (SDI) defined as the geometric mean of normalized
    lagged per capita income, average educational attainment over age 15 years, 
    and total fertility under 25 years
    More info at: 
    - https://www.thelancet.com/cms/10.1016/S0140-6736(18)31694-5/attachment/a2f9fe8f-1ee3-4acf-8b97-a74652292d30/mmc1.pdf
    - https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(17)32152-9/fulltext#supplementary-material
    
    -----------
    Parameters:
    income: pd.DataFrame
        DataFrame containing lagged per capita income with columns ['ISO3', 'year', 'gdplag']
    education: pd.DataFrame
        DataFrame containing average educational attainment over age 15 years with columns ['ISO3', 'year', 'education']
    fertility: pd.DataFrame
        DataFrame containing total fertility under 25 years with columns ['ISO3', 'year', 'TFR']
    --------
    Returns:
    pd.DataFrame
        DataFrame containing SDI values with columns ['ISO3', 'year', 'SDI']
    '''
    
    # Define best and worst values for normalization
    income_worst = np.log(250); income_best = np.log(60000)
    education_worst = 0; education_best = 17
    fertility_worst = 3; fertility_best = 0
    
    # Define min-max normalization function
    def normalize_covariate(value, worst, best):
        return (value - worst) / (best - worst)
    
    # Normalize covariates and calculate geometric mean
    df['sdi'] = ((normalize_covariate(df[income], income_worst, income_best))*
                    (normalize_covariate(df[education], education_worst, education_best))*
                    (normalize_covariate(df[fertility], fertility_worst, fertility_best)))**(1/3)
    
    return df

    
    
def ssp_sdi(wdirs):
    
    '''
    Create Sociodemographic index (SDI) for Shared Socioeconomic Pathways (SSPs)
    using lagged per capita income, average educational attainment over age 15 years
    and total fertility under 25 years
    
    -----------
    Parameters:
    wdirs: str
        Working directory
    --------
    Returns:
    DataFrame with the sdi values for each country, year and SSP scenario
    '''
    
    # Open income data to calculate lagged loggdppc
    income = load_variable(f'{wdir}\\data\\Socioeconomic_Data\\GDP\\ssp_basic_drivers_release_3.2.beta_full_reducedversion.xlsx', 
                    sheet_name='data',
                    iso3_col='Region',
                    drop_cols=['Model', 'Variable'],
                    melt_df={'id_vars':['Scenario', 'ISO3', 'Region', 'Unit'], 'var_name': 'year', 'value_name': 'gdppc'},
                    cast_cols={'year': int},
                    filters={'Unit':'USD_2015/yr'}, 
                    use_cols=['Scenario', 'ISO3', 'year', 'gdppc'])

    # Divide data into historical and ssp projections
    hist = income[income['Scenario'] == 'Historical Reference']
    hist = hist[hist['year'] < 2025]
    ssps = income[income['Scenario'] != 'Historical Reference']
    ssps = ssps[ssps['year'] >= 2025]

    # Repeat historical data for each SSP scenario
    hist_repeated = pd.concat([
        hist.assign(Scenario=scen) for scen in ssps['Scenario'].unique()
    ])

    # Merge historical and ssp data
    income_ext = pd.concat([hist_repeated, ssps]).reset_index(drop=True)

    # Calulate lagged gdppc (10-year moving average) 
    # -----> Include also the current year [2015 would use 2015,2010,2005] or exlcude it with shift(1) and only use 2010,2005???
    income_ext['gdplag'] = income_ext.groupby(['Scenario','ISO3'], group_keys=False)["gdppc"].apply(lambda x: x.rolling(window=3, min_periods=3).mean())

    income_ext['loggdplag'] = np.log(income_ext['gdplag'])
    # Keep only SSP projections
    income_ext = income_ext[income_ext['Scenario']!='Historical Reference']


    # Open fertility data
    fertility = pd.read_csv(f'{wdir}\\data\\Socioeconomic_Data\\Fertility\\wcde_data_age_specific_fertility_rate.csv', skiprows=8)
    # Calculate total fertility under 25 (TFU25)
    fertility['tfu25'] = fertility['Rate']*5
    fertility = (
        fertility
        .drop(columns='Age')          
        .groupby(['Scenario', 'Area', 'Period'])
        .sum() 
        .div(1000)
        .reset_index() 
    )
    # Change df format
    fertility = create_iso3_column(fertility, 'Area')
    fertility = fertility[['Scenario', 'ISO3', 'Period', 'tfu25']]
    fertility = fertility.rename(columns={'Period':'year'})
    fertility['year'] = fertility['year'].str[-4:].astype(int)
    fertility = fertility[fertility['year'] >= 2015]
    
    
    # Open covariates projections
    # gini = load_variable(f'{wdir}\\data\\Socioeconomic_Data\\GINI\\SSP-Extensions_Gini_Coefficient_v1.0.xlsx', 
    #                  sheet_name='data',
    #                  iso3_col='Region',
    #                  drop_cols=['Model', 'Variable', 'Unit'],
    #                  melt_df={'id_vars':['Scenario', 'ISO3', 'Region'], 'var_name': 'year', 'value_name': 'GINI'},
    #                  filters={'ISO3':['not found']}, exclude_filters=True,
    #                  use_cols=['Scenario', 'ISO3', 'year', 'GINI'])

    # Open education data
    education = load_variable(f'{wdir}\\data\\Socioeconomic_Data\\Education\\wcde_data_mean_schooling_years.csv', 
                        skiprows=8,
                        iso3_col='Area',
                        drop_cols=['Area', 'Age'],
                        rename_cols={'Years': 'education', 'Year':'year'})

    # Merge dataframes
    sdi = pd.merge(education, fertility, on=['ISO3', 'Scenario', 'year'], how='left')
    sdi = pd.merge(sdi, income_ext, on=['ISO3', 'Scenario', 'year'], how='left')

    # Create SDI
    sdi = create_sdi(sdi, 'loggdplag', 'education', 'tfu25')
    sdi = sdi[['Scenario', 'ISO3', 'year', 'Sex', 'education', 'tfu25', 'loggdplag', 'sdi']]
    
    return sdi