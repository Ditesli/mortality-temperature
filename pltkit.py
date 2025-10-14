import pandas as pd
import numpy as np
import xarray as xr
import csv
import scipy as sp
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter



wdir = 'X:\\user\\liprandicn\\Health Impacts Model'

### ------------------------------------------------------------------------------------------
'''
Relevant parameters
'''
# C-categories names
ccategories = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'] 
# Standard deviation values for climate variability
std_values = ['1', '5', '10']
# Methods of RR aggregation
methods = ['m1', 'm2']

# Full list of diseases
diseases = {'ckd':'Chronic kidney disease', 'cvd_cmp':'Cardiomyopathy and myocarditis', 'cvd_htn':'Hypertensive heart disease', 
            'cvd_ihd':'Ischemic heart disease', 'cvd_stroke':'Stroke', 'diabetes':'Diabetes mellitus',
            'inj_animal':'Animal contact', 'inj_disaster':'Exposure to forces of nature', 'inj_drowning':'Drowning', 
            'inj_homicide':'Interpersonal violence', 'inj_mech':'Exposure to mechanical forces', 
            'inj_othunintent':'Other unintentional injuries', 'inj_suicide':'Self-harm', 'inj_trans_other':'Other transport injuries', 
            'inj_trans_road':'Road injuries', 'resp_copd':'Chronic obstructive pulmonary disease', 'lri':'Lower respiratory infections'}

# List of diseases
relevant_diseases = {'ckd':'Chronic kidney disease', 'cvd_cmp':'Cardiomyopathy and myocarditis', 'cvd_htn':'Hypertensive heart disease', 
                     'cvd_ihd':'Ischemic heart disease', 'cvd_stroke':'Stroke', 'diabetes':'Diabetes mellitus',
                     'resp_copd':'Chronic obstructive pulmonary disease', 'lri':'Lower respiratory infections'}

# List of GBD level 3 clusters of diseases
diseases_level3 = {'crd':'Chronic respiratory diseases', 
                   'cd':'Cardiovascular diseases',
                   'rit':'Respiratory infections and tuberculosis', 
                   'dkd':'Diabetes and kidney diseases',
                   'ei':'Enteric infections',
                   'ntdm':'Neglected tropical diseases and malaria'}

# List of original countries whose mortality data was used by Burkart et al.
original_countries = ['BRA', 'CHL', 'CHN', 'COL', 'GTM', 'MEX', 'NZL', 'ZAF', 'USA']

diseases_level2 = ['All causes',
                   'HIV/AIDS and sexually transmitted infections',
                   'Respiratory infections and tuberculosis',
                   'Enteric infections',
                   'Neglected tropical diseases and malaria',
                   'Other infectious diseases',
                   'Maternal and neonatal disorders',
                   'Nutritional deficiencies',
                   'Neoplasms',
                   'Cardiovascular diseases',
                   'Chronic respiratory diseases',
                   'Digestive diseases',
                   'Neurological disorders',
                   'Mental disorders',
                   'Substance use disorders',
                   'Diabetes and kidney diseases',
                   'Skin and subcutaneous diseases',
                   'Muskuloskeletal disorders',
                   'Other non-communicable diseases',
                   'Transport injuries',
                   'Unintentional injuries',
                   'Self-harm and interpersonal violence']

### ------------------------------------------------------------------------------------------
'''
Other plot parameters
'''
# When plotting scenario projections, use these years as ticks
ticks_labels_years = ['2010', '2030', '2050', '2070', '2090']
ticks_years = [0, 20, 40, 60, 80]


### -------------------------------------------------------------------------------------------
'''
Colormaps
'''

original_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

# Colormap for IMAGE regions
cmap_tab20b = plt.get_cmap('tab20b')
cmap_tab20c = plt.get_cmap('tab20c')

colors_tab20b = [cmap_tab20b(i) for i in range(cmap_tab20b.N)]
colors_tab20c = [cmap_tab20c(i) for i in range(cmap_tab20c.N)]

combined_cmap = ListedColormap(colors_tab20c + colors_tab20b)


### ---------------------------------------------------------------------------------------------
'''
Read files
'''

# Read file with IMAGE region names and corresponding countries
image_regions = pd.read_csv(f'{wdir}\\data\\IMAGE_regions\\IMAGE_regions.csv',  index_col=0, header=0)

# Read GBD file with mortality data
gbd_mortality = pd.read_csv(f'{wdir}\\data\\GBD_Data\\Mortality\\IHME-GBD_2021_DATA.csv')



### ---------------------------------------------------------------------------------------------
'''
Important functions
'''

### Log-linear extrapolating function 
def log_linear_interp(xx, yy, kind='linear'):
    logx = xx   
    logy = np.log10(yy)    
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind, fill_value='extrapolate')    
    log_interp = lambda zz: np.power(10.0, lin_interp(zz))
    return log_interp


### Function to extrapolate ERF
def interpolate_erf(start_graph, end_graph, erf, tz):
    
    interp_hot = log_linear_interp(erf.loc[tz][erf.loc[tz]['rr'].idxmin():].index, erf.loc[(tz,'rr')][erf.loc[tz]['rr'].idxmin():], kind='linear')  
    xnew_hot = np.arange(erf.loc[tz]['rr'].index[-1], end_graph, 0.1)
    ynew_hot = interp_hot(xnew_hot)
    interp_cold = log_linear_interp(erf.loc[tz][:erf.loc[tz]['rr'].idxmin()].index, erf.loc[(tz,'rr')][:erf.loc[tz]['rr'].idxmin()], kind='linear')  
    xnew_cold = np.arange(start_graph, erf.loc[tz]['rr'].index[0], 0.1)
    ynew_cold = interp_cold(xnew_cold)
    
    return xnew_hot, ynew_hot, xnew_cold, ynew_cold

#### Function to get counts and weights parameters for a Kernel Density Estimation
def get_counts_and_weights_for_kde(tmrel, era5_tz, pop_ssp_year, tz):
    
    # Boolean mask for grid cells in the current temperature zone
    mask = era5_tz.t2m.values == tz

    # Extract TMREL values at those grid points (across all draws)
    values = tmrel.tmrel.values[mask, :].reshape(-1)

    # Repeat population weights for each draw
    pop_weights = np.repeat(pop_ssp_year.GPOP.values[mask], tmrel.dims['draw'])

    # Filter out NaNs
    valid = ~np.isnan(values) & ~np.isnan(pop_weights)
    values = values[valid]
    pop_weights = pop_weights[valid]

    # Normalize population weights
    pop_weights = pop_weights / pop_weights.sum()

    counts = values
    weights = pop_weights
    
    return counts, weights

### Get bubbles for plot
def get_bubbles(counts, size):
    
    x_vals, y_vals, sizes = [], [], []

    for gdp, tz_array in counts.items():
        counter = Counter(tz_array)
        total = sum(counter.values())

        for tz, freq in counter.items():
            x_vals.append(gdp)
            y_vals.append(tz)
            sizes.append(freq / total * size)  
            
    return x_vals, y_vals, sizes

# Get population-weighted bubbles for plot
def get_bubbles_weighted_by_population(counts, weights, size_scale=1000):
    x_vals, y_vals, sizes = [], [], []

    for gdp_val in counts:
        tz_array = counts[gdp_val]
        pop_array = weights[gdp_val]

        total_weight = np.sum(pop_array)
        if total_weight == 0:
            continue

        for tz in np.unique(tz_array):
            tz_mask = tz_array == tz
            tz_weight = pop_array[tz_mask].sum()

            x_vals.append(gdp_val)
            y_vals.append(tz)
            sizes.append((tz_weight / total_weight) * size_scale)

    return x_vals, y_vals, sizes



def get_values_counts_weights_for_bubbles(x_xarray, y_array, pop_ssp_year, weighted=True):
    
    counts = {}
    weights = {}
    unique_values = np.unique(x_xarray.values)[:-1]

    for unique_value in unique_values:
        # Mask to get array with unique_value
        mask = x_xarray.values == unique_value
        
        # Mask temperature zone xarray and flatten
        if len(y_array.dims) == 2:
            values_var = y_array.values[mask].flatten()
        if len(y_array.dims) == 3: 
            values_var = y_array.tmrel.values[mask, :].reshape(-1) 
               
        
        if weighted==False:    
            # Remove cells with NaN values in the variable
            values_var = values_var[~np.isnan(values_var)]
            
            # Count unique tmrel values for this tz
            counts[unique_value] = (values_var)
    
        if weighted == True:
            
            # Mask temperature zone and population xarray and flatten
            if len(y_array.dims) == 2:
                values_pop = pop_ssp_year.values[mask].flatten()
            if len(y_array.dims) == 3: 
                values_pop = np.repeat(pop_ssp_year.values[mask], y_array.dims['draw'])

            # Remove cells with NaN values
            valid_cells = ~np.isnan(values_var) & ~np.isnan(values_pop)
            values_var = values_var[valid_cells]
            values_pop = values_pop[valid_cells]
            
            # Count unique values and weights
            counts[unique_value] = values_var
            weights[unique_value] = values_pop 
            
    return counts, unique_values, weights


### --------------------------------------------------------------------------------------------
'''
Stylize functions for plots
'''


def stylize_axes(ax, *, 
                 xscale=None,
                 yscale=None,
                 ylim=None,
                 xlim=None,
                 title=None,
                 xlabel=None,
                 ylabel=None,
                 xticks=None,
                 xtickslabels=False,
                 xtickslabels_kwargs=None,
                 yticks=None,
                 ytickslabels=False,
                 ytickslabels_kwargs=None,
                 facecolor=None,
                 grid=False,
                 grid_kwargs=None,
                 legend=False,
                 legend_kwargs=None):  
    
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xticks is not None:
        ax.set_xticks(xticks) 
    if xtickslabels is not False:
        ax.set_xticklabels(xtickslabels, **(xtickslabels_kwargs or {}))

    if yticks is not None:
        ax.set_yticks(yticks) 
    if ytickslabels is not False:
        ax.set_yticklabels(ytickslabels, **(ytickslabels_kwargs or {}))

    if facecolor is not None:
        ax.set_facecolor(facecolor)
    if grid:
        ax.grid(**(grid_kwargs or {}))
    if legend:
        ax.legend(**(legend_kwargs or {}))
    
    return ax



def stylize_plot(*, 
                xscale=None,
                yscale=None,
                ylim=None,
                xlim=None,
                title=None,
                xlabel=None,
                ylabel=None,
                legend=True, 
                legend_kwargs=None, 
                grid=False,
                grid_kwargs=None,
                suptitle=None, 
                suptitle_kwargs=None, 
                xticks_kwargs=None,
                yticks_kwargs=None,
                tight_layout=None,
                show=True):
    if xscale:
        plt.xscale(xscale)
    if yscale:
        plt.yscale(yscale)
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)    
    if legend:
        plt.legend(**(legend_kwargs or {}))
    if grid:
        plt.grid(**(grid_kwargs or {}))
    if suptitle:
        plt.suptitle(suptitle, **(suptitle_kwargs or {}))
    if xticks_kwargs:
        plt.xticks(**xticks_kwargs)
    if yticks_kwargs:
        plt.yticks(**yticks_kwargs)
    if tight_layout:
        plt.tight_layout()
    if show:
        plt.show()
        
        
        
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def rss(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum((y_true - y_pred) ** 2)


def mse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)