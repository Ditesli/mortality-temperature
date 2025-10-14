import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

scenario='SSP245'; year=2024 # Example of any scenario and day
climate_model = 'CNRM-CM6-1-HR'
base_path = 'C:/Users/Nayeli/Documents' ### Select main folder

### Original temperature
SSP_var = pd.read_csv(f'{base_path}/Climate data/Climate ensemble/{climate_model}\\{scenario}\\BC_{climate_model}_{scenario}_{year}.csv')
SSP_var_1 = pd.read_csv(f'{base_path}/Climate data/Climate ensemble/{climate_model}\\{scenario}\\BC_{climate_model}_{scenario}_{year+1}.csv')
SSP_var = pd.concat([SSP_var, SSP_var_1.iloc[:,7:]], axis=1)
SSP_var_2 = pd.read_csv(f'{base_path}/Climate data/Climate ensemble/{climate_model}\\{scenario}\\BC_{climate_model}_{scenario}_{year+2}.csv')
SSP_var = pd.concat([SSP_var, SSP_var_2.iloc[:,7:]], axis=1)

### Daily variability smoothing
SSP_avg = pd.read_csv(f'{base_path}/Climate Variability/T_Monthly mean/{scenario}/MonthlyMean_{scenario}_{year}.csv')
SSP_avg_1 = pd.read_csv(f'{base_path}/Climate Variability/T_Monthly mean/{scenario}/MonthlyMean_{scenario}_{year+1}.csv')
SSP_avg = pd.concat([SSP_avg, SSP_avg_1.iloc[:,2:]], axis=1)
SSP_avg_2 = pd.read_csv(f'{base_path}/Climate Variability/T_Monthly mean/{scenario}/MonthlyMean_{scenario}_{year+2}.csv')
SSP_avg = pd.concat([SSP_avg, SSP_avg_2.iloc[:,2:]],axis=1)

### Method 1: Seasonal variability smoothing - Daily regression
SSP_avg_dayreg = pd.read_csv(f'{base_path}/Main folder/Climate variability/T_SeasonalVariability1/{scenario}/SeasonalVariability1_{scenario}_{year}.csv')
SSP_avg_dayreg1 = pd.read_csv(f'{base_path}/Main folder/Climate variability/T_SeasonalVariability1/{scenario}/SeasonalVariability1_{scenario}_{year+1}.csv')
SSP_avg_dayreg = pd.concat([SSP_avg_dayreg, SSP_avg_dayreg1.iloc[:,2:]], axis=1)
SSP_avg_dayreg2 = pd.read_csv(f'{base_path}/Main folder/Climate variability/T_SeasonalVariability1/{scenario}/SeasonalVariability1_{scenario}_{year+2}.csv')
SSP_avg_dayreg = pd.concat([SSP_avg_dayreg, SSP_avg_dayreg2.iloc[:,2:]],axis=1)

### Method 2: Seasonal variability smoothing - DICE
SSP_avg_dice = pd.read_csv(f'{base_path}/Main folder/Climate variability/T_SeasonalVariability2/{scenario}/SeasonalVariability2_{scenario}_{year}.csv')
SSP_avg_dice1 = pd.read_csv(f'{base_path}/Main folder/Climate variability/T_SeasonalVariability2/{scenario}/SeasonalVariability2_{scenario}_{year+1}.csv')
SSP_avg_dice = pd.concat([SSP_avg_dice, SSP_avg_dice1.iloc[:,2:]], axis=1)
SSP_avg_dice2 = pd.read_csv(f'{base_path}/Main folder/Climate variability/T_SeasonalVariability2/{scenario}/SeasonalVariability2_{scenario}_{year+2}.csv')
SSP_avg_dice = pd.concat([SSP_avg_dice, SSP_avg_dice2.iloc[:,2:]],axis=1)


'''Generate figures'''
titulos = ['Day-to-day variability smoothing', 'Seasonal variability smoothing\nMethod 1', 'Seasonal variability smoothing\nMethod 2']
labels = ['']

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
plt.subplots_adjust(wspace=0.15)

SSP_var.iloc[21787,1:].plot(ax=axs[0], label='Original', color='#f1a539')
SSP_avg.iloc[21787,1:].plot(ax=axs[0], label='Smoothed', color='#122B3E')
SSP_var.iloc[21787,1:].plot(ax=axs[2], label='Original', color='#f1a539')
SSP_avg_dayreg.iloc[21787,1:].plot(ax=axs[2], label='Smoothed', color='#122B3E')
SSP_var.iloc[21787,1:].plot(ax=axs[1], label='Original', color='#f1a539')
SSP_avg_dice.iloc[21787,1:].plot(ax=axs[1], label='Smoothed', color='#122B3E')

for i in np.arange(0,3):
    axs[i].set_title(f'{titulos[i]}', weight='bold', y=1.05, fontsize=12)
    axs[i].grid(True, color='white')
    axs[i].set_facecolor('whitesmoke')
    for spine in axs[i].spines.values():
        spine.set_visible(False)
    axs[i].tick_params(axis='x', length=0)
    axs[i].tick_params(axis='y', length=0)
    axs[i].legend(frameon=False, loc='upper left', fontsize=11)
    axs[i].set_ylim(-7, 33)
    axs[i].tick_params(axis='x', labelrotation=30)  
axs[0].set_ylabel('Temperature (°C)', fontsize=11)

plt.savefig(f'{base_path}/Main folder/Figures/Fig_ClimateVariability_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()