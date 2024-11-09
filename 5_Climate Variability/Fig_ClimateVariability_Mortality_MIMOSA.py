import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.getcwd(), '0_Params')) # Append the path
import calculate_mortality_noadap   # Import function to calculate mortality without adaptation
import MIMOSA_params

base_path = 'C:/Users/Nayeli/Documents' ### Select main folder
climate_model = 'AWI-CM-1-1-MR'
model_path = f'D:/Climate Models - Bias Corrected/{climate_model}'

tab20b = plt.cm.tab20b(range(20))
tab20c = plt.cm.tab20c(range(20))
Set3 =  plt.cm.Set3(range(10))
combined_colors = list(tab20b) + list(tab20c) + list(Set3)
custom_cmap = ListedColormap(combined_colors[:60])
name_groups = ['Oldest (+65 years)', 'Older (5-64 years)', 'Young (0-4 years)']


'''Open files'''
Original = pd.read_csv(f'{base_path}/MIMOSA/MIMOSA Mortality/MIMOSAMortality_allSSP_{climate_model}.csv', header=[0, 1, 2], index_col=[0])
Original = Original.drop(['Antarctica', 'Caspian Sea'])
#Monthly Running Mean
MonthlyMean = pd.read_csv(f'{base_path}/Main folder/Climate Variability/Mortality_ClimateVariability_MIMOSA_MontlhyMean.csv', header=[0, 1, 2], index_col=[0])
MonthlyMean = MonthlyMean.drop(['Antarctica', 'Caspian Sea'])
MonthlyMean_error = (MonthlyMean-Original)/Original*100
#Seasonal Variability Method 1
SV_Method1 = pd.read_csv(f'{base_path}/Main folder/Climate Variability/Mortality_ClimateVariability_MIMOSA_SeasonalVariability1.csv', header=[0, 1, 2], index_col=[0])
SV_Method1 = SV_Method1.drop(['Antarctica', 'Caspian Sea'])
SV_Method1_error = (SV_Method1-Original)/Original*100
#Seasonal Variability Method 2
SV_Method2 = pd.read_csv(f'{base_path}/Main folder/Climate Variability/Mortality_ClimateVariability_MIMOSA_SeasonalVariability2.csv', header=[0, 1, 2], index_col=[0])
SV_Method2 = SV_Method2.drop(['Antarctica', 'Caspian Sea'])
SV_Method2_error = (SV_Method2-Original)/Original*100

''''Pre calculations'''
groups = ['oldest', 'older', 'young']
quantil = [0.05, 0.95]
percentiles = ['5 percentile', '95 percentile']
col_index = pd.MultiIndex.from_product([groups, percentiles], names=['Age group', '5-95 range'])
Percentile_MonthlyMean = pd.DataFrame(index=Original.index, columns=col_index)
Percentile_Method1 = pd.DataFrame(index=Original.index, columns=col_index)
Percentile_Method2 = pd.DataFrame(index=Original.index, columns=col_index)

for group in groups:
    for region in MonthlyMean_error.index:
        for percentile, quant in zip(percentiles, quantil):
            percentile_MM = MonthlyMean_error.loc[region, group].quantile(quant)
            Percentile_MonthlyMean.loc[region, (group, percentile)] = round(percentile_MM,2)
        
            percentile_SV1 = SV_Method1_error.loc[region, group].quantile(quant)
            Percentile_Method1.loc[region, (group, percentile)] = round(percentile_SV1,2)
        
            percentile_SV2 = SV_Method2_error.loc[region, group].quantile(quant)
            Percentile_Method2.loc[region, (group, percentile)] = round(percentile_SV2,2)

'''Open Global files'''
df_var = pd.read_csv(f'{base_path}/Main folder/Mortality/No Adaptation/TotalMortality_NoAdaptation_{climate_model}.csv')
df_var.set_index(['Age group', 'Scenario'], inplace=True)

df_monthly_mean = pd.read_csv(f'{base_path}/Main folder/Climate Variability/Mortality_ClimateVariability_MontlhyMean.csv')
df_monthly_mean.set_index(['Age group', 'Scenario'], inplace=True)
percentage_dif = (df_monthly_mean-df_var)/df_var*100
percentage_dif_stacked = percentage_dif.stack(level=0)
percentage_dif_df = percentage_dif_stacked.to_frame()
percentage_dif_T = percentage_dif_df.transpose()
percentage_dif = percentage_dif_T.rename(index={0: 'WORLD'})
MonthlyMean_error.loc['WORLD', :] = percentage_dif.iloc[0,:]

df_method2 = pd.read_csv(f'{base_path}/Main folder/Climate Variability/Mortality_ClimateVariability_Seasonal1.csv')
df_method2.set_index(['Age group', 'Scenario'], inplace=True)
percentage_dif3 = (df_method2-df_var)/df_method2*100
percentage_dif3_stacked = percentage_dif3.stack(level=0)
percentage_dif3_df = percentage_dif3_stacked.to_frame()
percentage_dif3_T = percentage_dif3_df.transpose()
percentage_dif3 = percentage_dif3_T.rename(index={0: 'WORLD'})
SV_Method2_error.loc['WORLD', :] = percentage_dif3.iloc[0,:]

df_method1 = pd.read_csv(f'{base_path}/Main folder/Climate Variability/Mortality_ClimateVariability_Seasonal2.csv')
df_method1.set_index(['Age group', 'Scenario'], inplace=True)
percentage_dif2 = (df_method1-df_var)/df_method1*100
percentage_dif2_stacked = percentage_dif2.stack(level=0)
percentage_dif2_df = percentage_dif2_stacked.to_frame()
percentage_dif2_T = percentage_dif2_df.transpose()
percentage_dif2 = percentage_dif2_T.rename(index={0: 'WORLD'})
SV_Method1_error.loc['WORLD', :] = percentage_dif2.iloc[0,:]


'''Make Figure'''
dataframe = MonthlyMean_error ### Can be changed
mm = dataframe.transpose()
mm = mm.reset_index()
mm = mm.drop(['Scenario', 'Year'], axis=1)
mm = mm.set_index('Age group')

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 9), sharey=True, dpi=300, gridspec_kw={'hspace': 0, 'wspace':0.1})

for i, col in enumerate(groups):
    axes[i].grid(True, color='silver', linestyle='--', linewidth=0.5, zorder=0)
    axes[i].axvline(x = 0, linestyle='--', color='dimgrey', linewidth=0.75, zorder=2)
    axes[i].set_facecolor('whitesmoke')
    
    parts = axes[i].violinplot(dataset=mm.loc[col,::-1], vert=False, widths=1.2, showextrema=True, bw_method=1, 
                               quantiles=[[0.05, 0.95]]*27, showmeans=True)
    
    for pc, region in zip(parts['bodies'], mm.columns[::-1]):
        pc.set_facecolor(custom_cmap(MIMOSA_params.region_colors[region]))
        pc.set_edgecolor('black')
        pc.set_alpha(0.4)
        pc.set_zorder(3)
    
    for partname in ('cbars','cmins','cmaxes','cmedians','cquantiles'):
        vp = parts.get(partname)
        if vp is not None:
            vp.set_edgecolor('dimgray')
            vp.set_linewidth(1)
            vp.set_zorder(3)
            
    means = parts['cmeans']
    if means is not None:
        means.set_color('k')  
        
    axes[i].set_title(f'{name_groups[i]}', weight='bold', fontsize=12)
    axes[i].set_xlabel('Percentage error', fontsize=11)
    axes[i].tick_params(axis='x', length=0)
    axes[i].tick_params(axis='y', length=0)
    axes[i].set_xlim(-40, 10)
    axes[i].set_ylabel('')
    axes[i].set_yticks(range(1, len(mm.columns) + 1))
    axes[i].set_yticklabels(mm.columns[::-1], fontsize=10) 
    axes[i].set_xticklabels([f'{int(tick)}%' for tick in axes[i].get_xticks()], fontsize=10)
    for spine in axes[i].spines.values():
        spine.set_visible(False)

plt.subplots_adjust(hspace=0, wspace=0)
plt.suptitle('Underestimation of mortality due to smoothing day-to-day variability', fontsize=14, y=0.95)
#plt.suptitle('Underestimation of mortality due to smoothing seasonal variability', fontsize=14, y=0.95)
#plt.suptitle('Underestimation of mortality due to smoothing seasonal variability - Method 1', fontsize=14, y=0.95)
plt.savefig(f'{base_path}/Main folder/Figures/Fig_MIMOSA_PercentageError_{dataframe}.png', bbox_inches='tight', dpi=300)
plt.show()