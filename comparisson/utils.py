import pandas as pd
import numpy as np


# Full list of causes
causes = {
    'ckd':'Chronic kidney disease', 
    'cvd_cmp':'Cardiomyopathy and myocarditis', 
    'cvd_htn':'Hypertensive heart disease', 
    'cvd_ihd':'Ischemic heart disease', 
    'cvd_stroke':'Stroke', 
    'diabetes':'Diabetes mellitus',
    'inj_animal':'Animal contact', 
    'inj_disaster':'Exposure to forces of nature', 
    'inj_drowning':'Drowning', 
    'inj_homicide':'Interpersonal violence', 
    'inj_mech':'Exposure to mechanical forces', 
    'inj_othunintent':'Other unintentional injuries', 
    'inj_suicide':'Self-harm', 
    'inj_trans_other':'Other transport injuries', 
    'inj_trans_road':'Road injuries', 
    'resp_copd':'Chronic obstructive pulmonary disease', 
    'lri':'Lower respiratory infections'
    }

relevant_causes = {
    'ckd':'Chronic kidney disease', 
    'cvd_cmp':'Cardiomyopathy and myocarditis', 
    'cvd_htn':'Hypertensive heart disease', 
    'cvd_ihd':'Ischemic heart disease', 
    'cvd_stroke':'Stroke',
    'diabetes':'Diabetes mellitus',
    'resp_copd':'Chronic obstructive pulmonary disease', 
    'lri':'Lower respiratory infections'
    }



def LoadMortality(wdir, filename, years, region, temp_type, unit, age_group, cause):
    
    """
    Load mortality from ANY calculation method as time series and constrained to 
    the parameters:
    - years: Years range
    - region: IMAGE region or "World"
    - temp_type: "Heat", "Cold", "All"
    - unit: "relative" (Relative mortality) or "total" (total mortality)
    - age_group: "young", "older", "oldest" (only valid for Carleton)
    - cause: cause of death (only valid for Burkart method)
    """
    
    df = pd.read_csv(wdir + filename + ".csv")

    # Initialize filter as True to not filter anything initially
    filter = pd.Series(True, index=df.index)

    # Apply age_group condition only if the column exists
    if "age_group" in df.columns:
        filter &= df["age_group"].str.lower().str.contains(age_group.lower())
        
    # Apply age_group condition only if the column exists
    if "cause" in df.columns:
        filter &= df["cause"].str.lower().str.contains(cause.lower())

    # Condition for t_type
    filter &= df["t_type"].str.lower() == temp_type.lower()

    # Condition for region
    filter &= df["region"] == region

    # Condition for units
    if "units" in df.columns:
        filter &= df["units"].str.lower().str.contains(unit.lower())

    # Apply filter and select columns starting from the 5th column
    df = df[filter][[str(y) for y in years if str(y) in df.columns]]
    
    # Convert column names to int
    df.columns = df.columns.astype(int)
    
    return df



def WeightsAndCountsForKDE(tmrel, era5_tz, pop_ssp_year, tz):
    
    """
    Function to get counts and weights parameters for a Kernel Density Estimation
    """
    
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



def StylizeAxes(ax, *, 
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




def load_burkart(wdir, regions_class, output_type, gbd_df, temp_type, age_group, continents):
    
    '''
    Load present day output data calculated using the method
    prescribed in Burkart et al., (2022)
    '''
    
    app = pd.read_csv(f'{wdir}\\burkart2022\\output\\paf_era5_GBD_level3_all-dis_{output_type}2000-2019.csv', header=[0,1,2], index_col=[0])
    
    gbd_ = gbd_df[
        (gbd_df['cause_name'].isin(diseases.values())) &
        (gbd_df['sex_name'] == 'Both') &
        (gbd_df['year'].isin(range(2000, 2020))) &
        (gbd_df['age_name'].isin(age_group)) &
        (gbd_df['location_name'] != 'Global')][['location_id', 'location_name', 'cause_name', 'year', 'val', 'upper', 'lower']]	
    gbd_['year']=gbd_['year'].astype(int)
        
    # Select mortality from different temperature types
    app = app.xs(temp_type, level=2, axis=1)
    
    # Stack dataframe to have location, year, cause_key as index and paf as column
    app = app.stack(level=[0,1], 
                    future_stack=True).reset_index().rename(columns={'level_0':'location_id', 'level_1':'year', 'level_2':'cause_key', 0:'paf'})
    
    # Map diseases names with keys
    app['cause_name'] = app['cause_key'].map({k:v for k,v in diseases.items()})
    app['year'] = app['year'].astype(int)
    
    # Merge with mortality data
    app = app.merge(gbd_, on=['location_id', 'cause_name', 'year'], how='left')
    
    # Calculate deaths attributable to temperature
    app['mean'] = app['val'] * app['paf'] 
    app['upper_est'] = app['upper'] * app['paf'] 
    app['lower_est'] = app['lower'] * app['paf'] 
    app = app[['year', 'cause_name', 'location_name', 'location_id', 'mean', 'upper_est', 'lower_est']]  
    
    if continents == True:
        # Merge with regions classification
        app = app.merge(regions_class[['gbd_location_id', 'continents']].drop_duplicates(), left_on='location_id', right_on='gbd_location_id', how='left')
        app = app[['continents', 'year', 'mean', 'upper_est', 'lower_est']].groupby(['year','continents']).sum().reset_index()
        app = app.groupby(['continents']).mean().reset_index()
        
    if continents == False:
        app = app[['location_name', 'location_id', 'year', 'mean', 'upper_est', 'lower_est']].groupby(['year','location_name', 'location_id']).sum().reset_index()
        app = app.groupby(['location_name', 'location_id']).mean().reset_index()
    
    return app



def load_carleton(wdir, regions_class, t_type, age_group=None):
    
    '''
    Load present day output data calculated using the method
    prescribed in Carleton et al., (2022)
    '''

    # Load output file
    carleton_df = pd.read_csv(f'{wdir}\\Carleton2022\\output\\mortality_ERA5_gbd-level3_2000-2019.csv', index_col=[0,1,2,3])
    # Select a scenario (SSP2 only for present day), and a temperature type
    
    # Select scenario, temperature type and, if specified, age group
    if age_group is None:
        carleton_df = carleton_df.xs('SSP2', level=0).xs(t_type, level=1).reset_index().groupby('gbd_level3').sum().iloc[:,1:]
        
    else:
        carleton_df = carleton_df.xs('SSP2', level=0).xs(age_group, level=0).xs(t_type, level=0).reset_index().groupby('gbd_level3').sum().iloc[:,1:]
    
    # Group countries by continent
    carleton_df = regions_class[['gbd_level3', 'continents']].set_index('gbd_level3').merge(carleton_df, left_index=True, right_index=True).reset_index()
    carleton_df = carleton_df.iloc[:,1:].groupby('continents').mean().mean(axis=1).reset_index()
    
    carleton_df.rename(columns={0:'mean'}, inplace=True)
    
    # Define upper an lower estimation as the mean for plotting
    carleton_df['upper_est'] = carleton_df['mean'] 
    carleton_df['lower_est'] = carleton_df['mean']
    
    return carleton_df



def load_honda(wdir, regions_class, gbd_df, ot_type, temp_type, period, merging_type):
    
    '''
    Load present day output data calculated using the method
    prescribed in Burkart et al., (2022)
    '''
    
    honda_df = pd.read_csv(f'{wdir}\\honda2014\\output\\paf_era5_GBD_level3_{period}_ot-{ot_type}.csv', index_col=0, header=[0,1])

    honda_df = honda_df.xs(temp_type, level=1, axis=1)
    
    gbd_ = gbd_df[
        (gbd_df['cause_name']=='All causes') &
        (gbd_df['sex_name'] == 'Both') &
        (gbd_df['year'].isin(range(2000, 2020))) &
        (gbd_df['age_name'].isin(['65-69 years', '70-74 years', '75-79 years', '85+ years'])) &
        (gbd_df['location_name'] != 'Global')][['location_id', 'location_name', 'cause_name', 'year', 'val', 'upper', 'lower']]	
    gbd_['year']=gbd_['year'].astype(int)
    gbd_ = gbd_.groupby(['location_id', 'location_name', 'cause_name', 'year']).sum().reset_index()

    
    # Stack dataframe to have location, year, cause_key as index and paf as column
    app = honda_df.stack(level=[0], 
                    future_stack=True).reset_index().rename(columns={'level_0':'location_id', 'level_1':'year', 0:'paf'})
    app['year'] = app['year'].astype(int)

    # Merge with mortality data
    app = app.merge(gbd_, on=['location_id', 'year'], how='left')

    # Calculate deaths attributable to temperature
    app['mean'] = app['val'] * app['paf'] * 0.88
    app['upper_est'] = app['upper'] * app['paf'] * 0.88
    app['lower_est'] = app['lower'] * app['paf']  * 0.88
    app = app[['year', 'location_name', 'mean', 'upper_est', 'lower_est']]  

    # Merge with regions classification
    app = app.merge(regions_class[['gbd_level3', merging_type]], left_on='location_name', right_on='gbd_level3', how='left').drop_duplicates()
    app = app.groupby(['year', merging_type]).sum().reset_index()[[merging_type, 'year', 'mean', 'upper_est', 'lower_est']]
    app = app.groupby([merging_type]).mean().reset_index()
       
    return app