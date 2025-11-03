import pandas as pd

# Full list of diseases
diseases = {'ckd':'Chronic kidney disease', 'cvd_cmp':'Cardiomyopathy and myocarditis', 'cvd_htn':'Hypertensive heart disease', 
            'cvd_ihd':'Ischemic heart disease', 'cvd_stroke':'Stroke', 'diabetes':'Diabetes mellitus',
            'inj_animal':'Animal contact', 'inj_disaster':'Exposure to forces of nature', 'inj_drowning':'Drowning', 
            'inj_homicide':'Interpersonal violence', 'inj_mech':'Exposure to mechanical forces', 
            'inj_othunintent':'Other unintentional injuries', 'inj_suicide':'Self-harm', 'inj_trans_other':'Other transport injuries', 
            'inj_trans_road':'Road injuries', 'resp_copd':'Chronic obstructive pulmonary disease', 'lri':'Lower respiratory infections'}



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



def load_honda(wdir, regions_class, gbd_df, temp_type, ):
    
    '''
    Load present day output data calculated using the method
    prescribed in Burkart et al., (2022)
    '''
    
    honda_df = pd.read_csv(f'{wdir}\\honda2014\\output\\paf_era5_GBD_level3_2000-2019.csv', index_col=0, header=[0,1])

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
    app['mean'] = app['val'] * app['paf'] 
    app['upper_est'] = app['upper'] * app['paf'] 
    app['lower_est'] = app['lower'] * app['paf'] 
    app = app[['year', 'location_name', 'mean', 'upper_est', 'lower_est']]  

    # Merge with regions classification
    app = app.merge(regions_class[['gbd_level3_mor', 'continents']], left_on='location_name', right_on='gbd_level3_mor', how='left').drop_duplicates()
    app = app.groupby(['year','continents']).sum().reset_index()[['continents', 'year', 'mean', 'upper_est', 'lower_est']]
    app = app.groupby(['continents']).mean().reset_index()
       
    return app