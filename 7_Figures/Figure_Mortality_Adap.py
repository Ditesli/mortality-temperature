'''You can uncomment some lines to generated the figures GDP&T, GDP, T'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
sys.path.append(os.path.join(os.getcwd(), '0_Params')) # Append the path
import climate_models_info

base_path = 'C:/Users/Nayeli/Documents' ### Select path to main folder
folder_path = f'{base_path}/Main folder/Figures '
if not os.path.exists(folder_path):
        os.makedirs(folder_path)

'''Define important variables'''
colors = {'SSP126': '#5ea7cd', 'SSP245': '#33416b', 'SSP370': '#dd8852', 'SSP585': '#c04242'}
labels = ['2020', '2030', '2040', '2050', '2060', '2070', '2080', '2090', '2100']
scenarios = ['SSP126', 'SSP245', 'SSP370', 'SSP585']
decades = ['2020','2025','2030','2035','2040','2045','2050','2055','2060','2065','2070','2075','2080', '2085','2090','2095','2100']
groups = ['Oldest (age > 64 years)', 'Older (5-64 years)', 'Young (age < 5 years)']
text_groups = ['Oldest', 'Older', 'Young']
text_groups2 = ['(age > 64 years)', '(5 - 64 years)', '(age < 5 years)']
legends = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']

dataframes = {}
covariate = 'GDP&T' ### or 'GDP' OR 'T'
for climate_model in climate_models_info.climate_models_dic.keys():
    file_path = f'{base_path}/Main folder/Mortality/Adaptation/TotalMortality_Adaptation_{covariate}_{climate_model}.csv'
    dataframes[climate_model] = pd.read_csv(file_path, index_col=[0,1])

index = pd.MultiIndex.from_product([groups, scenarios], names=['Age group', 'Scenario'])
average = pd.DataFrame(index=index, columns=decades)
stdev = pd.DataFrame(index=index, columns=decades)
for i in np.arange(0,12):
    for j in range(len(decades)):
        a = []
        for climate_model in climate_models_info.climate_models_dic.keys():
            a.append(dataframes[climate_model].iloc[i,j])
        average.iloc[i,j] = np.mean(a)
        stdev.iloc[i,j] = 2*np.std(a)


plt.figure(figsize=(15, 4.5), dpi=300)

for i, group in enumerate(groups, start=1):
    ax = plt.subplot(1, 3, i)
    group_data = average.xs(group, level='Age group')
    group_std = stdev.xs(group, level='Age group') 
    
    for j,scenario in enumerate(scenarios):
        mean_line = group_data.loc[scenario].astype(float)
        std_line = group_std.loc[scenario].astype(float)
        ax.plot(decades, mean_line, label=legends[j], color=colors[scenario], lw=1.5)
        if scenario == 'SSP126' or scenario == 'SSP585':
            ax.fill_between(decades, mean_line - std_line, mean_line + std_line, color=colors[scenario], alpha=0.2, zorder=2)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_xlim('2020', '2100')
    if i==1:
        ax.set_ylabel('Deaths / 100,000', fontsize=11)
        ax.legend(fontsize=11, loc='center left', bbox_to_anchor=(0.63, 0.7), frameon=False)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, color='white', zorder=0)
    ax.set_facecolor('whitesmoke')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    ax.text(0.6, 0.95, text_groups[i-1], transform=ax.transAxes, verticalalignment='top', fontsize=11, weight='bold')
    ax.text(0.6, 0.9, text_groups2[i-1], transform=ax.transAxes, verticalalignment='top', fontsize=11)

plt.subplots_adjust(wspace=0.12)
plt.suptitle('Mortality projections including income growth', fontsize=12, y=0.95, weight='bold')
plt.savefig(f'{folder_path}/Figures/Fig_Mortality_Global_Adap_{covariate}.png', bbox_inches='tight', dpi=300)
plt.show()
