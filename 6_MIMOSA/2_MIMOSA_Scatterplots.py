import pandas as pd
import numpy as np
import os, sys
import itertools
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.getcwd(), '0_Params')) # Append the path
import climate_models_info
import MIMOSA_params

base_path = 'C:/Users/Nayeli/Documents' ### Select main folder
folder_path = f'{base_path}/Main folder/MIMOSA'

scenarios = ['SSP126', 'SSP245', 'SSP370', 'SSP585']
scenarios_labels = ['SSP2-2.6', 'SSP2-4.5', 'SSP2-7.0', 'SSP2-8.5']
scenarios_t = ['ssp126', 'ssp245', 'ssp370', 'ssp585' ]
SSP = 'SSP2'
groups = ['oldest', 'older', 'young']
years = np.arange(2015,2105,5)
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'H', '+', 'x', 'h', 'd', '|', '_', '.']
colors = ['#2F5061', '#99c1b9', '#E57F84', '#f2d0a9']

marker_cycle = itertools.cycle(markers)
color_cycle = itertools.cycle(colors)
model_markers = {model: next(marker_cycle) for model in climate_models_info.climate_models_dic.keys()}
scenario_colors = {scenario: next(color_cycle) for scenario in scenarios}


'''Calculate DeltaT (how much has the temperature risen compared to preindustrial levels)'''
GSAT = pd.read_csv(f'{base_path}/Main folder/Climate data/GSAT_Future.csv',  header=[0, 1], index_col=[0])
Historical_GSAT = pd.read_csv(f'{base_path}/Main folder/Climate data/GSAT_Historical.csv', index_col=[0])

for climate_model in climate_models_info.climate_models_dic.keys():
    for scenario in scenarios_t:
        for year in GSAT.index:
            GSAT.loc[year, (climate_model, scenario)] = GSAT.loc[year, (climate_model, scenario)] - Historical_GSAT.loc[climate_model, 'Historical']
GSAT_2 = GSAT.round(2)
GSAT = GSAT_2.stack(level=0).swaplevel(0, 1, axis=0).unstack()


'''Open dataframes regional mortality'''
dataframes_SSP = {}
for climate_model in climate_models_info.climate_models_dic.keys():
    file_path = f'{folder_path}/MIMOSA Mortality/MIMOSA_Mortality_NoAdap_SSP2_{climate_model}.csv'
    dataframes_SSP[climate_model] = pd.read_csv(file_path, header=[0, 1, 2], index_col=[0])
    dataframes_SSP[climate_model] = dataframes_SSP[climate_model].drop(['Antarctica', 'Caspian Sea'])


'''Shift in the y-axis shift to put the “extra” mortality in the reference period 9(2015-2025) to zero'''
combined_df = pd.concat(dataframes_SSP, names=['climate_model', 'region'])
combined_df = combined_df.unstack(1).swaplevel(2,3,1).sort_index(level=2, axis=1)
combined_df2 = combined_df.copy()

for group in groups:
    for scenario in scenarios:
        for region in dataframes_SSP['CESM2'].index:
            #mean_mean = combined_df[group][scenario][region].loc[:,'2015':'2025'].mean().mean() #Present day mean of all climate models
            for climate_model in climate_models_info.climate_models_dic.keys():
                mean_cm = combined_df[group][scenario][region].loc[climate_model,'2015':'2025'].mean()
                #delta_mean = mean_mean - mean_cm
                for year in years:
                    combined_df2.loc[climate_model,(group, scenario, region, f'{year}')]  = combined_df[group][scenario][region].loc[climate_model, f'{year}'] - mean_cm

reverted_df = combined_df2.swaplevel(2, 3, axis=1).stack(level='region').sort_index()
dataframes_shift = {model: reverted_df.xs(model, level='climate_model') for model in reverted_df.index.get_level_values('climate_model').unique()}
reverted_df.to_csv(f'{folder_path}/Scatterplot_substract_mean.csv') ### Save intermediate file


'''(Upload) file and calculate the 5-95 percentile'''
reverted = pd.read_csv(f'{folder_path}/Scatterplot_substract_mean.csv', header=[0,1,2], index_col=[0,1])
dataframes = {model: reverted.xs(model, level='climate_model') for model in reverted.index.get_level_values('climate_model').unique()}

region_coeff = {}

for group in groups:
    region_coeff[group] = {}
    for region in dataframes['CESM2'].index:
        
        x = np.concatenate([GSAT.loc[model, :] for model in climate_models_info.climate_models_dic.keys()])
        y = np.concatenate([dataframes[model].loc[region, group] for model in climate_models_info.climate_models_dic.keys()])
        
        popt, pcov = MIMOSA_params.curve_fit(MIMOSA_params.quadratic, x, y)
        min_x = min(x)
        max_x = max(x)

        x, y = MIMOSA_params.order_data(x, y)
        y_fit = MIMOSA_params.quadratic(x, *popt)

        popt_5, popt_95 = MIMOSA_params.coeff_percentile(x,y, 0.05, 0.95)
        popt_10, popt_90 = MIMOSA_params.coeff_percentile(x,y, 0.1, 0.90)
        
        region_coeff[group][region] = {'popt': popt, 'min_x': min_x, 'max_x': max_x, 'popt_5':popt_5, 'popt_95':popt_95, 'popt_10':popt_10, 
                                       'popt_90':popt_90}


'''Make final plots'''

group='oldest'
fig, axs = plt.subplots(5, 6, figsize=(30, 22), dpi=300)
axs = axs.flatten()

for i, region in enumerate(dataframes['CESM2'].index[:28]):
    ax = axs[i]

    for climate_model in climate_models_info.climate_models_dic.keys():
        for scenario, scenario_t in zip(scenarios, scenarios_t):
            ax.scatter(GSAT.loc[climate_model,scenario_t], dataframes[climate_model].loc[region, (group, scenario)], label = climate_model,
                   marker=model_markers[climate_model], color=scenario_colors[scenario], alpha=0.5, zorder=2)

    x_fit = np.linspace(region_coeff[group][region]['min_x'], region_coeff[group][region]['max_x'], 100)  
    y_fit = MIMOSA_params.quadratic(x_fit, *region_coeff[group][region]['popt']) 
    
    median, = ax.plot(x_fit, y_fit, lw=2, color='k', zorder=2, label='Quantile regression')
    dashed, = ax.plot(x_fit, MIMOSA_params.quadratic(x_fit, *region_coeff[group][region]['popt_5']), lw=2, color='grey', linestyle='--', zorder=2, label='Uncertainty')
    ax.plot(x_fit, MIMOSA_params.quadratic(x_fit, *region_coeff[group][region]['popt_95']), lw=2, color='grey', linestyle='--', zorder=2)
    
    ax.grid(True, color='white', zorder=0)
    ax.set_xlabel('$\Delta$ GMST (°C)', fontsize=12)
    if i==0 or i==6 or i==12 or i==18 or i==24:
        ax.set_ylabel('Annual mortality / 100,000', fontsize =13)
    ax.set_title(f'{region}', weight='bold', fontsize=14)
    a, b, c = region_coeff[group][region]['popt']
    ax.text(0.5, 0.9, 'Damage curve', transform=ax.transAxes, ha='center', va='center', fontsize=12, color='k')
    ax.text(0.5, 0.82, f'{a.round(2)}x$^2$ + {b.round(2)}x + {c.round(2)}', transform=ax.transAxes, ha='center', va='center', fontsize=12, color='k')
    ax.set_facecolor('whitesmoke')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)

for j in range(26, 30):
    fig.delaxes(axs[j])

handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
color_handles = [plt.Line2D([0], [0], color=scenario_colors[scenario], lw=4) for scenario in scenarios]
color_labels = scenarios_labels
marker_handles = [plt.Line2D([0], [0], color='dimgrey', marker=model_markers[model], linestyle='None', markersize=10) for model in climate_models_info.climate_models_dic.keys()]
marker_labels = list(climate_models_info.climate_models_dic.keys())
marker_handles.append(median)
marker_handles.append(dashed)
marker_labels.append('Damage curve')
marker_labels.append('Uncertainty')

fig.legend(color_handles, color_labels, loc='lower right', bbox_to_anchor=(0.55, 0.12), shadow=False, ncol=2, fontsize=14, frameon=True, title='Scenarios', title_fontsize=15)
fig.legend(marker_handles, marker_labels, loc='lower right', bbox_to_anchor=(0.9, 0.12), shadow=False, ncol=4, fontsize=14, frameon=True, title='Climate Models', title_fontsize=15)
fig.text(0.44, 0.19, f'Age group: {group}', fontsize=22, weight='bold')
plt.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.95, hspace=0.4, wspace=0.2)
plt.savefig(f'{base_path}/Main folder/Figures/Fig_Scatterplots_{group}.png', dpi=300, bbox_inches='tight')
plt.show()