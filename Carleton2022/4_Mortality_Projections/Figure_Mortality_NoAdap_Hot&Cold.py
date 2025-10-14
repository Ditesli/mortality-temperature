import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.getcwd(), '0_Params')) # Append the path
import climate_models_info

base_path = 'C:/Users/Nayeli/Documents' ### Select path to main folder
folder_path = f'{base_path}/Main folder/Figures'
if not os.path.exists(folder_path):
        os.makedirs(folder_path)

'''Define important variables'''
scenarios = ['SSP126', 'SSP245', 'SSP370', 'SSP585']
decades = ['2015','2020','2025','2030','2035','2040','2045','2050','2055','2060','2065','2070','2075','2080', '2085','2090','2095','2100']
groups = ['Oldest (+65 years)', 'Older (5-64 years)', 'Young (0-4 years)']
colors = {'SSP126': '#5ea7cd', 'SSP245': '#33416b', 'SSP370': '#dd8852', 'SSP585': '#c04242'}
colors_cold = {'SSP126': '#aed3f3', 'SSP245': '#7ebce8', 'SSP370': '#4a92c9', 'SSP585': '#0e233e'}
colors_hot = {'SSP126': '#ffb485', 'SSP245': '#ff934d', 'SSP370': '#dd8852', 'SSP585': '#c04242'}
labels = ['2020', '2040', '2060', '2080', '2100']
text_groups = ['Oldest', 'Older', 'Young']
text_groups2 = ['(age > 64 years)', '(5-64 years)', '(age < 5 years)']
legends = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']


'''Open dataframes'''
dataframes_hot = {}; dataframes_cold = {}
for climate_model in climate_models_info.climate_models_dic.keys():
    file_path = f'{base_path}/Main folder/Mortality/No Adaptation/'
    dataframes_hot[climate_model] = pd.read_csv(f'{file_path}/HotMortality_NoAdaptation_{climate_model}.csv', index_col=[0,1])
    #dataframes_cold[climate_model] = pd.read_csv(f'{file_path}/ColdMortality_NoAdaptation_{climate_model}.csv', index_col=[0,1])


'''Calculate mean and stdev'''
index = pd.MultiIndex.from_product([groups, scenarios], names=['Age group', 'Scenario'])

average_hot = pd.DataFrame(index=index, columns=decades)
stdev_hot = pd.DataFrame(index=index, columns=decades)
average_cold = pd.DataFrame(index=index, columns=decades)
stdev_cold = pd.DataFrame(index=index, columns=decades)

for i in np.arange(0,12):
    for j in range(len(decades)):
        a_cold = []; a_hot = []
        for climate_model in climate_models_info.climate_models_dic.keys():
            #a_cold.append(dataframes_cold[climate_model].iloc[i,j])
            a_hot.append(dataframes_hot[climate_model].iloc[i,j])
        #average_cold.iloc[i,j] = np.mean(a_cold)
        #stdev_cold.iloc[i,j] = 2*np.std(a_cold)
        average_hot.iloc[i,j] = np.mean(a_hot)
        stdev_hot.iloc[i,j] = 2*np.std(a_hot)


'''Generate plots'''
plt.figure(figsize=(15, 4.5), dpi=300)
plt.subplots_adjust(wspace=0.15)

for i, group in enumerate(groups, start=1):
    ax = plt.subplot(1, 3, i)
    group_data = average_hot.xs(group, level='Age group') #####group_data = average_cold.xs(group, level='Age group')
    group_std = stdev_hot.xs(group, level='Age group')  #####group_std = stdev_cold.xs(group, level='Age group') 
    
    for j,scenario in enumerate(scenarios):
        mean_line = group_data.loc[scenario].astype(float)
        std_line = group_std.loc[scenario].astype(float)
        
        ax.plot(decades, mean_line, label=legends[j], color=colors_hot[scenario], zorder=2, lw=2)
        #####ax.plot(decades, mean_line, label=legends[j], color=colors_cold[scenario], zorder=2, lw=2)
        if scenario=='SSP126' or scenario=='SSP585':
            ax.fill_between(decades, mean_line - std_line, mean_line + std_line, color=colors_hot[scenario], alpha=0.2, zorder=2)
            #####ax.fill_between(decades, mean_line - std_line, mean_line + std_line, color=colors_cold[scenario], alpha=0.2, zorder=2)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_xlim('2015', '2100')
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    if i==1:
        ax.set_ylabel('Deaths / 100,000', fontsize=11)
        ax.legend(fontsize=11, loc='center left', bbox_to_anchor=(0, 0.7), frameon=False)
    
    ax.set_xticks(labels)
    ax.set_xticklabels(labels)
    ax.grid(True, color='white', zorder=0)
    ax.set_facecolor('whitesmoke')
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    ax.text(0.05, 0.95, text_groups[i-1], transform=ax.transAxes, verticalalignment='top', fontsize=11, weight='bold')
    ax.text(0.05, 0.9, text_groups2[i-1], transform=ax.transAxes, verticalalignment='top', fontsize=11)

plt.suptitle('Hot temperatures - No Adaptation', fontsize=12, y=0.95, weight='bold')
#####plt.suptitle('Cold temperatures - No Adaptation', fontsize=12, y=0.95, weight='bold')
plt.savefig(f'{folder_path}/Fig_Mortality_Global_NoAdap_Hot.png', dpi=300, bbox_inches='tight')
#####plt.savefig(f'{folder_path}/Fig_Mortality_Global_NoAdap_Cold.png', dpi=300, bbox_inches='tight')
plt.show()
