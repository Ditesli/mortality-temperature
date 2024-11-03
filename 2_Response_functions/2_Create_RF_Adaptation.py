import pandas as pd
import numpy as np
import xarray as xr
from shapely.geometry import Point, Polygon, mapping
import geopandas as gpd
import os, sys
sys.path.append(os.path.join(os.getcwd(), '0_Params')) # Append the path
import climate_models_info, relationship_temp_to_ir, RF_info


base_path = 'C:/Users/Nayeli/Documents/' ### Select main path
carleton_path = 'D:\\data'  ### Set path to Carleton et al.'s folder
ir = gpd.read_file(f'{carleton_path}'+'\\2_projection\\1_regions\\ir_shp\\impact-region.shp')


''' Generate 30 year running mean (computationally demanding) '''
bias = pd.read_csv(base_path + 'Main folder/Climate data/Bias_Correction.csv')
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
years = np.arange(2020,2105,5)
index_col = pd.MultiIndex.from_product([climate_models_info.climate_models_dic.keys(), scenarios, years], names=['Climate Model', 'Scenario', 'Year'])
t_mean = pd.DataFrame(index=ir['hierid'], columns=index_col)

for climate_model in climate_models_info.climate_models_dic.keys():
    for scenario in scenarios:
        label =  climate_models_info.climate_models_dic[climate_model][0]
        grid = climate_models_info.climate_models_dic[climate_model][1]
        # Set the path for historial and future data
        xr1 = xr.open_dataset(f'D:\\Climate Models - Present Day NSAT\\tas_Amon_{climate_model}_historical_{label}_{grid}_1984-2014.nc')
        xr2 = xr.open_dataset(f'D:\\Climate Models - Future NSAT\\tas_Amon_{climate_model}_{scenario}_{label}_{grid}_2015-2100.nc')
        xr_concat = xr.concat([xr1, xr2], dim='time')
        for year in years:
            xr_30 = xr_concat.sel(time=slice(f'{year-30}', f'{year}')).mean(dim='time').tas
            latitud = ((xr_30.lat.values[1]-xr_30.lat.values[0]) + (xr_30.lat.values[2]-xr_30.lat.values[1]))/2
            longitud = ((xr_30.lon.values[1]-xr_30.lon.values[0]) + (xr_30.lon.values[2]-xr_30.lon.values[1]))/2
            relation = relationship_temp_to_ir.relationship(ir, longitud, latitud, xr_30, extended=False)
            temperatures = xr_30.values.flatten() - 273.15
            relation['temperature'] = temperatures[relation.index]
            result = relation.groupby('index_right')['temperature'].mean()
            t_mean.loc[:,(climate_model, scenario, year)] = result.round(2) + bias.loc[:,climate_model].round(2)
            print(f'{climate_model} - {scenario} - {year}')

t = t_mean.astype(float)
t = t.round(2)
t.to_csv(base_path + 'Main folder/Response functions/Covariate_Tmean.csv')


''' Generate 13 yr running mean of GDP '''
gdp = pd.DataFrame(index=t.index)
years = np.arange(2020,2105,5)
scenarios = ['SSP1', 'SSP2', 'SSP3', 'SSP5']
gdp_new = pd.DataFrame(index=t.index, columns=pd.MultiIndex.from_product([scenarios, years], names=['SSP', 'Year']))

for scenario in scenarios:
    df = pd.read_csv(base_path + 'Main folder/GPD & POP/GDPpc files/GDPpc_{scenario}_mean.csv'))
    df = df.set_index('hierid')
    df_aligned = df.reindex(gdp.index)
    df_years = df_aligned.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1)
    df_years.columns = pd.to_numeric(df_years.columns)
    df_years[2009] = df_years[2010]
    df_years[2008] = df_years[2010]
    df_years[2007] = df_years[2010]

    columnas = df_years.columns.tolist()
    # Reorganize the columns to have them at the  beginning
    new_columnas = [2007, 2008, 2009] + [col for col in columnas if col not in [2007, 2008, 2009]]
    df_years = df_years[new_columnas]
    
    for year in years:
        gdp_new.loc[:,(scenario,year)] = np.log(df_years.loc[:, np.arange(year-13, year)].mean(axis=1))

gdp_new.to_csv(base_path + 'Main folder/Response functions/Covariate_GDP.csv')



''' Open important dataframes '''

df_oldest = pd.read_csv(base_path + 'Main folder/Response functions/RF_NoAdaptation_oldest.csv')
df_oldest = df_oldest.drop(df_oldest.columns[[0, 1]], axis=1)
pattern = re.compile(r'^-?\d+(\.0|\.5)?$')
filtered_columns = filter(lambda col: pattern.match(col), df_oldest.columns)
selected_columns = list(filtered_columns)
df_oldest = df_oldest[selected_columns]

df_older = pd.read_csv(base_path + 'Main folder/Response functions/RF_NoAdaptation_older.csv')
df_older = df_older.drop(df_older.columns[[0, 1]], axis=1)
df_older = df_older[selected_columns]

df_young = pd.read_csv(base_path + 'Main folder/Response functions/RF_NoAdaptation_young.csv')
df_young = df_young.drop(df_young.columns[[0, 1]], axis=1)
df_young = df_young[selected_columns]

df_tmin = pd.read_csv(base_path + 'Main folder/Response functions/T_min.csv')


scenarios_gdp = ['SSP5', 'SSP3', 'SSP2', 'SSP1']
scenarios_tmean = ['SSP585', 'SSP370', 'SSP245', 'SSP126']
groups = ['oldest', 'older', 'young']
df_groups = [df_oldest, df_older, df_young]



''' Response functions with GDP and T changing '''

folder_path = base_path + 'Main folder/Response Functions/RF Dataframes_GDP&T'
if not os.path.exists(folder_path):
        os.makedirs(folder_path)

tmean = pd.read_csv(base_path + 'Main folder/Response functions/Covariate_Tmean.csv', index_col=[0], header=[0,1,2])
tmean.columns = tmean.columns.set_levels(['SSP126', 'SSP245', 'SSP370', 'SSP585'], level=1)

gdp = pd.read_csv(base_path + 'Main folder/Response functions/Covariate_GDP.csv', index_col=[0], header=[0,1])
gdp = gdp.round(3)

t = np.arange(-50, 60.5, 0.5)
t = t.round(1)

# For loop for climate models
for climate_model in climate_models_info.climate_models_dic.keys():
    # For loop between SSP scenarios for gdp and tmean
    for scenario_gdp, scenario_tmean in zip(scenarios_gdp, scenarios_tmean):
        # For loop between three age groups
        for group, df_group in zip(groups, df_groups):
            # For loop between time range 
            for year in np.arange(2020, 2105, 5):
                # Select the corresponding year column from gdp and tmean
                loggdppc = np.array(gdp.loc[:, (scenario_gdp, f'{year}')])
                climtas = np.array(tmean.loc[:, (climate_model, scenario_tmean, f'{year}')])
                # Create an empty list to append mortality calculations

                responses = np.empty((len(climtas), len(t)))
                for i in range(len(climtas)):
                    tas = get_tas(gamma_np[group], climtas[i], loggdppc[i])
                    mortality_df = response(df_group.iloc[i, :].to_numpy(), tas, df_tmin[f'Tmin {group}'][i], t)
                    responses[i, :] = mortality_df

                # Generate dataframe from responses and add region column   
                df_new = pd.DataFrame(responses, columns=[f"{temp:.1f}" for temp in t])
                df_rounded = df_new.round(1)

                #Merge region column and drop index column (duplicated)
                df_rounded = df_rounded.set_index(tmean.index)

                # Save as csv file
                df_rounded.to_csv({folder_path} + f'{climate_model}/{scenario_tmean}/{climate_model}_{scenario_tmean}_{group}_{year}.csv')
                print(f'{climate_model} - {scenario_tmean} - {group} - {year}')



''' Response functions with T only (GDP fixed) '''

folder_path2 = base_path + 'Main folder/Response Functions/RF Dataframes_T'
if not os.path.exists(folder_path2):
        os.makedirs(folder_path2)

tmean = pd.read_csv(base_path + 'Main folder/Response functions/Covariate_Tmean.csv', index_col=[0], header=[0,1,2])
tmean.columns = tmean.columns.set_levels(['SSP126', 'SSP245', 'SSP370', 'SSP585'], level=1)

gdp = pd.read_csv(carleton_path + '2_projection/3_impacts/main_specification/raw/single/rcp85/CCSM4/low/SSP3/mortality-allpreds.csv',
                  skiprows=14)
gdp_15 = gdp[gdp['year'] == 2015]
gdp = gdp_15.iloc[0:24378,:] 
gdp = gdp[['region', 'loggdppc']]
loggdppc_pre = gdp.set_index(['region'])
loggdppc = np.array(loggdppc_pre)

t = np.arange(-50, 60.5, 0.5)
t = t.round(1)

# For loop for climate models
for climate_model in climate_models_info.climate_models_dic.keys():
    # For loop between SSP scenarios for gdp and tmean
    for scenario_gdp, scenario_tmean in zip(scenarios_gdp, scenarios_tmean):
        # For loop between three age groups
        for group, df_group in zip(groups, df_groups):
            # For loop between time range 
            for year in np.arange(2020, 2105, 5):

                # Select the corresponding year column from tmean
                climtas = np.array(tmean.loc[:, (climate_model, scenario_tmean, f'{year}')])
                # Create an empty list to append mortality calculations
                responses = np.empty((len(climtas), len(t)))
                
                for i in range(len(climtas)):
                    tas = get_tas(gamma_np[group], climtas[i], loggdppc[i][0])
                    mortality_df = response(df_group.iloc[i, :].to_numpy(), tas, df_tmin[f'Tmin {group}'][i], t)
                    responses[i, :] = mortality_df

                # Generate dataframe from responses and add region column   
                df_new = pd.DataFrame(responses, columns=[f"{temp:.1f}" for temp in t])
                df_rounded = df_new.round(1)

                #Merge region column and drop index column (duplicated)
                df_rounded = df_rounded.set_index(tmean.index)

                # Save as csv file
                df_rounded.to_csv(folder_path2 + f'{climate_model}/{scenario_tmean}/{climate_model}_{scenario_tmean}_{group}_{year}.csv')
                print(f'{climate_model} - {scenario_tmean} - {group} - {year}')


''' Response functions with GDP only (T fixed) '''

folder_path3 = base_path + 'Main folder/Response Functions/RF Dataframes_GDP'
if not os.path.exists(folder_path3):
        os.makedirs(folder_path3)

gdp = pd.read_csv(base_path + 'Main folder/Response functions/Covariate_GDP.csv', index_col=[0], header=[0,1])

tmean = pd.read_csv(carleton_path + '/2_projection/3_impacts/main_specification/raw/single/rcp85/CCSM4/low/SSP3/mortality-allpreds.csv', skiprows=14)
tmean_15 = tmean[tmean['year'] == 2015]
tmean = tmean_15.iloc[0:24378,:] 
tmean = tmean[['region', 'climtas']]
tmean_pre = tmean.set_index(['region'])
climtas = np.array(tmean_pre)


t = np.arange(-50, 60.5, 0.5)
t = t.round(1)
# For loop for climate models
for climate_model in climate_models_info.climate_models_dic.keys():
    # For loop between SSP scenarios for gdp and tmean
    for scenario_gdp, scenario_tmean in zip(scenarios_gdp, scenarios_tmean):
        # For loop between three age groups
        for group, df_group in zip(groups, df_groups):
            # For loop between time range
            for year in np.arange(2020, 2105, 5):
                # Select the corresponding year column from gdp and tmean
                loggdppc = np.array(gdp.loc[:, (scenario_gdp, f'{year}')])
              #  climtas = np.array(tmean.loc[:, (climate_model, scenario_tmean, f'{year}')])
                # Create an empty list to append mortality calculations
                responses = np.empty((len(climtas), len(t)))
                for i in range(len(climtas)):
                    tas = get_tas(gamma_np[group], climtas[i][0], loggdppc[i])
                    mortality_df = response(df_group.iloc[i, :].to_numpy(), tas, df_tmin[f'Tmin {group}'][i], t)
                    responses[i, :] = mortality_df
                # Generate dataframe from responses and add region column
                df_new = pd.DataFrame(responses, columns=[f"{temp:.1f}" for temp in t])
                df_rounded = df_new.round(1)
                #Merge region column and drop index column (duplicated)
                df_rounded = df_rounded.set_index(tmean.index)
                # Save as csv file
                df_rounded.to_csv(folder_path3 + f'/{climate_model}/{scenario_tmean}/{climate_model}_{scenario_tmean}_{group}_{year}.csv')
                print(f'{climate_model} - {scenario_tmean} - {group} - {year}')
