import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys

base_path = 'C:/Users/Nayeli/Documents' ### Select main folder
climate_model = 'AWI-CM-1-1-MR'
model_path = f'D:/Climate Models - Bias Corrected/{climate_model}'

'''Define params'''
scenarios = ['SSP126', 'SSP245', 'SSP370', 'SSP585']
legends = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']
decades = ['2015','2020','2025','2030','2035','2040','2045','2050','2055','2060','2065','2070','2075','2080', '2085','2090','2095','2100']
groups = ['oldest', 'older', 'young']
name_groups = ['Oldest (+65 years)', 'Older (5-64 years)', 'Young (0-4 years)']
colors_var = {'SSP126': '#99a83a', 'SSP245': '#16bdd5', 'SSP370': '#f1a539', 'SSP585': '#B52451'}
labels = ['2020', '2040', '2060', '2080', '2100']


'''Load files'''
df_var = pd.read_csv(f'{base_path}/Main folder/Mortality/No Adaptation/TotalMortality_NoAdaptation_{climate_model}.csv')
df_var.set_index(['Age group', 'Scenario'], inplace=True)

monthly_mean = pd.read_csv(f'{base_path}/Main folder/Climate Variability/Mortality_ClimateVariability_MontlhyMean.csv')
monthly_mean.set_index(['Age group', 'Scenario'], inplace=True)

seasonal1 = pd.read_csv(f'{base_path}/Main folder/Climate Variability/Mortality_ClimateVariability_SeasonalVariability1.csv')
seasonal1.set_index(['Age group', 'Scenario'], inplace=True)

seasonal2 = pd.read_csv(f'{base_path}/Main folder/Climate Variability/Mortality_ClimateVariability_SeasonalVariability2.csv')
seasonal2.set_index(['Age group', 'Scenario'], inplace=True)


'''Generate figure'''
dataframe = monthly_mean
percentage_dif = (dataframe-df_var)/df_var*100

all_values_series_oldest = pd.Series(percentage_dif.loc['oldest'].values.flatten())
percentile_oldest_day_5 = all_values_series_oldest.quantile(0.05); percentile_oldest_day_95 = all_values_series_oldest.quantile(0.95)
all_values_series_older = pd.Series(percentage_dif.loc['older'].values.flatten())
percentile_older_day_5 = all_values_series_older.quantile(0.05); percentile_older_day_95 = all_values_series_older.quantile(0.95)
all_values_series_young = pd.Series(percentage_dif.loc['young'].values.flatten())
percentile_young_day_5 = all_values_series_young.quantile(0.05); percentile_young_day_95 = all_values_series_young.quantile(0.95)

titles = ['Oldest (+65 years)', 'Older (5-64 years)', 'Young (0-4 years)'] 
panel_text_mean = [f'Mean: {all_values_series_oldest.mean().round(2)}', f'Mean: {all_values_series_older.mean().round(2)}', 
                    f'Mean: {all_values_series_young.mean().round(2)}']
panel_text = [f'5-95% range: ({percentile_oldest_day_5.round(2)}, {percentile_oldest_day_95.round(2)})',
              f'5-95% range: ({percentile_older_day_5.round(2)}, {percentile_older_day_95.round(2)})',
              f'5-95% range: ({percentile_young_day_5.round(2)}, {percentile_young_day_95.round(2)})',]

plt.figure(figsize=(15, 4.5), dpi=300)
plt.subplots_adjust(wspace=0.15)

for i, group in enumerate(groups, start=1):
    ax = plt.subplot(1, 3, i)
    group_data = percentage_dif.xs(group, level='Age group')
    
    for j,scenario in enumerate(scenarios):
        mean_line = group_data.loc[scenario].astype(float)
        ax.plot(decades, mean_line, label=legends[j], color=colors_var[scenario])
       
    ax.set_xlabel('Year', fontsize=11)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True)
    ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    ax.grid(True, color='white')
    ax.set_facecolor('whitesmoke')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-14.5,-6.5)
    ax.set_xlim('2015', '2100')
    if i == 1:
        ax.set_ylabel('Percentage difference (%)', fontsize=11)
        ax.legend(fontsize=10, loc='upper right', frameon=False)
    if i == 2:
        ax.text(0.51, 0.94, titles[i-1], transform=ax.transAxes, verticalalignment='top', fontsize=11, weight='bold')
        ax.text(0.69, 0.89, panel_text_mean[i-1], transform=ax.transAxes, verticalalignment='top', fontsize=11)
        ax.text(0.34, 0.84, panel_text[i-1], transform=ax.transAxes, verticalalignment='top', fontsize=11)
    else:
        ax.text(0.05, 0.19, titles[i-1], transform=ax.transAxes, verticalalignment='top', fontsize=11, weight='bold')
        ax.text(0.05, 0.14, panel_text_mean[i-1], transform=ax.transAxes, verticalalignment='top', fontsize=11)
        ax.text(0.05, 0.09, panel_text[i-1], transform=ax.transAxes, verticalalignment='top', fontsize=11)

plt.suptitle('Underestimation of global mortality by smoothing day-to-day variability', fontsize=12, y=0.95, weight='bold')
#plt.suptitle('Underestimation of global mortality by smoothing seasonal variability - Method 1', fontsize=12, y=0.95, weight='bold')
plt.suptitle('Underestimation of global mortality due to smoothing seasonal variability', fontsize=12, y=0.95, weight='bold')
plt.savefig(f'{base_path}/Main folder/Figures/Fig_PercentError_{dataframe}.png', bbox_inches='tight', dpi=300)
plt.show()