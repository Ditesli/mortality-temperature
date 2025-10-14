import pandas as pd
import numpy as np
import xarray as xr
import os
import country_converter as coco ###

wdir = 'X:\\user\\liprandicn\\Health Impacts Model'



def assign_all_attributes(ds: xr.Dataset, attributes: list):
    
    '''
    Assign all attributes to an xarray Dataset.
    '''
    
    for var, units, description, source in attributes:
        if var in ds:
            ds[var].attrs['units'] = units
            ds[var].attrs['description'] = description
            ds[var].attrs['source'] = source
            
    return ds



def create_iso3_column(df, column_name):
    
    '''
    Use the country_converter package to convert country names in a DataFrame column to ISO3 codes
    for facilitating data merging
    '''
    
    unique_locations = df[column_name].unique()
    conversion_dict = {location: coco.convert(names=location, to='ISO3') for location in unique_locations}
    df['ISO3'] = df[column_name].map(conversion_dict)
    
    return df



def load_indicator(file_path, sheet_name=None, skiprows=0, iso3_col=None, filters=None, drop_cols=None, use_cols=None, 
                     rename_cols=None, rename_entries=None, cast_cols=None, melt_df=None, age_standar=False,
                     group_cols=None, attributes=None):
    
    '''
    Load csv files with country level statistics
    '''
    
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.csv':
        df = pd.read_csv(file_path, skiprows=skiprows)
    elif file_ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path, skiprows=skiprows, sheet_name=sheet_name)
    else:
        raise ValueError(f"Unsuported file format: {file_ext}")

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
                df = df[df[col].isin(vals)]
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
    if age_standar and all(col in df.columns for col in ['age_group', 'ISO3', 'year', 'population']):
        age95_filter = df['age_group'].isin(['95-99', '100+'])
        if age95_filter.any():
            df_95plus = df[age95_filter].groupby(['ISO3', 'year'], as_index=False)['population'].sum()
            df_95plus['age_group'] = '95+'
            df = df[~age95_filter]
            df = pd.concat([df, df_95plus], ignore_index=True)

        df_allages = df.groupby(['ISO3', 'year'], as_index=False)['population'].sum()
        df_allages['age_group'] = 'All'
        df = pd.concat([df, df_allages], ignore_index=True)
    
    print(f"{os.path.basename(file_path)} correctly imported")
    
    df_to_xarray = file_to_xarray(df, group_cols)
    df_to_xarray = assign_all_attributes(df_to_xarray, attributes) if attributes else df_to_xarray
    
    return df_to_xarray
        


def file_to_xarray(df, group_cols):
    
    '''
    Convert a DataFrame with 'ISO3', 'year', and 'population' columns to an xarray Dataset.
    The DataFrame should have at least the columns 'ISO3' and 'year'.
    '''
    
    value_cols = [col for col in df.columns if col not in group_cols]
    ordered_df = df.groupby(group_cols)[value_cols].first().reset_index()
    data_xr = ordered_df.set_index(group_cols).to_xarray()
    
    return data_xr



def merge_xarrays(datasets, join='outer', compat='override'):
    """
    Une una lista de xarray.Dataset usando xr.merge de forma flexible.

    Parameters:
    - datasets (list): List of xarray.Dataset
    - join (str): 'outer', 'inner', 'left', 'right'
    - compat (str): How to manage conlficting variables ('override', 'identical', etc.)

    Returns:
    - xarray.Dataset merged and postprocessed
    """

    country_data = xr.merge(datasets, compat=compat, join=join)
    
    ### Calculate relative mortality and gdppc
    country_data['relative_mortality'] = (country_data['total_mortality'] / country_data['population']) * 1e5
    country_data['gdppc'] = country_data['gdp'] / country_data['population']
    # country_data = assign_all_attributes(country_data, [('relative_mortality', 'per 100,000', 'Relative mortality per 100,000 population', 'Calculated')])
    # country_data = assign_all_attributes(country_data, [('GDPpc', 'USD', 'GDP per capita in current USD', 'Calculated')])

    # Delete Greenland and countries with no temperature data (small islands)
    country_data = country_data.where(country_data['ISO3'] != 'GRL', drop=True)
    valid_countries = country_data['temperature_mean'].mean(dim='year', skipna=True).notnull()
    country_data = country_data.sel(ISO3=valid_countries)

    # Order age groups
    orden = list(country_data.age_group.values)
    orden.remove('5-9')
    orden.insert(1, '5-9')
    country_data = country_data.sel(age_group=orden)
    
    country_data = country_data.drop('Unnamed: 0')
    
    iso3_codes = country_data.coords['ISO3'].values
    country_names = coco.convert(names=iso3_codes, to='name_short', not_found=None)
    country_data = country_data.assign_coords(ISO3=country_names)
    country_data = country_data.rename({'ISO3':'country'})
    
    country_data.to_netcdf(f'{wdir}\\Analysis\\country_data_1980-2019.nc')
    
    make_age_groups(country_data)
    
    print('xarray dataset processed and saved to netCDF file')



def make_age_groups(country_data):

    ### Filter xarray to group of 3 age groups, calculate relative mortality and save processed file

    young_group = country_data.sel(age_group=['0-4']).sum(dim='age_group', keep_attrs=True)
    old_group = country_data.sel(age_group=['5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64']).sum(dim='age_group', keep_attrs=True)
    oldest_group = country_data.sel(age_group=['65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95+']).sum(dim='age_group', keep_attrs=True)
    total_group = country_data.sum(dim='age_group', keep_attrs=True)

    age_groups = xr.concat([young_group, old_group, oldest_group, total_group], pd.Index(['young', 'old', 'oldest', 'total'], name='age_group'))

    fixed_vars =list(age_groups.keys())
    fixed_vars.remove('relative_mortality')
    fixed_vars.remove('population')
    fixed_vars.remove('total_mortality')
    
    for var in fixed_vars:
        age_groups[var] = age_groups[var].isel(age_group=0).drop_vars('age_group')
        
    age_groups = age_groups.drop_vars(['relative_mortality'])

    age_groups['relative_mortality'] = (age_groups['total_mortality'] / age_groups['population']) * 1e5

    age_groups.to_netcdf(f'{wdir}\\Analysis\\grouped_country_data_1980-2019.nc', mode='w', format='NETCDF4')
    
    
    
def secondary_operations(xarray_data):
    
    # Filter country data to only include countries with a population greater than 1 million
    
    mask = xarray_data['population'].sel(age_group='total') > 1e6

    country_data = xr.Dataset()

    for var_name, da in xarray_data.data_vars.items():
        # Expand mas to match dimensions
        mask_expanded = mask
        for dim in da.dims:
            if dim not in mask_expanded.dims:
                mask_expanded = mask_expanded.expand_dims({dim: da.coords[dim]})
        # Apply mask
        country_data[var_name] = da.where(mask_expanded)

    # Compute additional variables

    country_data['loggdppc'] = np.log(country_data['gdppc'] + 1)
    country_data['log_kurtosis'] = np.log(country_data['temperature_kurtosis'] + 3)
    country_data['log_CDD_20'] = np.log(country_data['CDD_20'] + 1)
    country_data['log_HDD_20'] = np.log(country_data['HDD_20'] + 1)
    country_data['CDD_20_sq'] = country_data['CDD_20'] ** 2
    country_data['HDD_20_sq'] = country_data['HDD_20'] ** 2
    country_data['CDD_20_k'] = country_data['CDD_20']/365
    country_data['HDD_20_k'] = country_data['HDD_20']/365
    country_data['CDD_20_k_sq'] = country_data['CDD_20_sq']/365
    country_data['HDD_20_k_sq'] = country_data['HDD_20_sq']/365
    country_data['CDD_25_rh_k'] = country_data['CDD_25_rh']/365
    country_data['log_health_exp'] = np.log(country_data['health_expenditure'] + 1)
    country_data['log_rel_mor'] = np.log(country_data['relative_mortality']+1)
    country_data['log_mortality'] = np.log(country_data['total_mortality']+1)
    country_data['temperature_mean_sq'] = country_data['temperature_mean'] ** 2
    country_data['CDD_23_3_st'] = country_data['CDD_23_3']/365
    country_data["temperature_mean_cen"] = country_data["temperature_mean"] - country_data["temperature_mean"].mean()
    country_data["temperature_mean_sq_cen"] = country_data["temperature_mean_cen"] ** 2

    country_data['temperature_mean_3'] = country_data['temperature_mean'] ** 3
    country_data['temperature_mean_4'] = country_data['temperature_mean'] ** 4
    country_data['temperature_mean_loggdppc'] = country_data['temperature_mean'] * country_data['loggdppc']
    country_data['temperature_mean_sq_loggdppc'] = country_data['temperature_mean']**2 * country_data['loggdppc']
    country_data['temperature_kurtosis_sq'] = country_data['temperature_kurtosis']**2
    country_data['degree_days_loggdppc'] = country_data['degree_days'] * country_data['loggdppc']
    country_data['degree_days_sq'] = country_data['degree_days']**2
    country_data['degree_days_sq_loggdppc'] = country_data['degree_days_sq'] * country_data['loggdppc']
    country_data['log_population'] = np.log(country_data['population']+1)
    country_data['degree_days_std'] = country_data['degree_days']/365
    country_data['degree_days_std_sq'] = country_data['degree_days_std']**2
    country_data['degree_days_std_loggdppc'] = country_data['degree_days_std'] * country_data['loggdppc']
    country_data["health_expenditure_k"] = country_data["health_expenditure"] / 1000
    country_data['log_health_expenditure'] = np.log(country_data['health_expenditure']+1)
    country_data['CDD_20_k_loggdppc'] = country_data['CDD_20_k'] + country_data['loggdppc'] 
    country_data['HDD_20_k_loggdppc'] = country_data['HDD_20_k'] + country_data['loggdppc'] 

    country_data['bin_m20_m10'] = country_data['bin_-20_-15'] + country_data['bin_-15_-10']
    country_data['bin_m10_0'] = country_data['bin_-10_-5'] + country_data['bin_-5_0']
    country_data['bin_0_10'] = country_data['bin_0_5'] + country_data['bin_5_10']
    country_data['bin_10_20'] = country_data['bin_10_15'] + country_data['bin_15_20']
    country_data['bin_20_30'] = country_data['bin_20_25'] + country_data['bin_25_30']
    country_data['bin_30_40'] = country_data['bin_30_35'] + country_data['bin_35_40']
    country_data['bin_m15'] = country_data['bin_-20_-15'] + country_data['bin_-20']
    country_data['bin_m10'] = country_data['bin_-15_-10'] + country_data['bin_m15']
    country_data['bin_35'] = country_data['bin_35_40'] + country_data['bin_40']
    country_data['bin_30_40'] = country_data['bin_30_35'] + country_data['bin_35_40']
    country_data['bin_30'] = country_data['bin_30_35'] + country_data['bin_35']
    country_data['bin_25'] = country_data['bin_25_30'] + country_data['bin_30']
    country_data['bin35_loggdppc'] = country_data['bin_35'] * country_data['loggdppc']

    country_data = country_data.rename({'bin_-20':'bin_m20', 'bin_-20_-15': 'bin_m20_m15', 'bin_-15_-10': 'bin_m15_m10', 'bin_-10_-5': 'bin_m10_m5', 'bin_-5_0': 'bin_m5_0'})
    
    return country_data