import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
from typing import Optional




def var_nc_to_csv(
    file_path: str,
    data_type: str,
    scenario: str,
    pop_group: Optional[str],
    age_group: Optional[str],
    gdp_model: str
    ) -> pd.DataFrame:
    
    '''
    Convert Carleton et al. (2022) GDP or population NetCDF files into CSV format.

    This function extracts regional projections for a given SSP scenario 
    and variable (GDP or population) from the Carleton et al. (2022) NetCDF datasets.
    The resulting dataset is flattened into a pandas DataFrame and saved as a CSV file 
    under the subdirectory `gdp_pop_csv/`.

    Parameters
    ----------
    file_path : str
        Base directory containing the Carleton et al. data structure. 
        Expected subdirectories:
        - `carleton_sm/econ_vars/` with NetCDF files (e.g., `"SSP1.nc4"`)
        - `exposure_response_functions/` with reference CSVs (e.g., `"erf_no-adapt_oldest.csv"`)
    data_type : {'GDP', 'POP'}
        Type of variable to extract:
        - 'GDP' for economic projections.
        - 'POP' for population projections.
    scenario : str
        SSP scenario identifier (e.g., 'SSP1', 'SSP3').
    pop_group : str or None
        Population variable to extract from the NetCDF file.
        If data_type == 'GDP', this parameter should typically be 'gdppc'.
    age_group : str or None
        Age group label to include in the output CSV filename. Ignored for GDP datasets.
    gdp_model : str
        GDP model variant (e.g., 'high', 'low', or 'mean').
        Used for file selection and naming.

    Returns
    -------
    df : pandas.DataFrame
        Flattened dataset containing the extracted variable across years and regions. 
        Columns include:
        - 'region'
        - One column per year (e.g., 2010, 2015, …)
        - 'hierid'

    '''
    
    # Read in oldest population data to get regions
    oldest = pd.read_csv(f'{file_path}/exposure_response_functions/erf_no-adapt_oldest.csv')
    
    # Open xarray from Carleton data
    ssp = xr.open_dataset(f'{file_path}/carleton_sm/econ_vars/{scenario}.nc4') 
    # Select high or low GDP model
    ssp = ssp[pop_group].sel(model=gdp_model) 
    # Convert to dataframe
    df = ssp.to_dataframe() 
    # Unstack
    df = df.drop(['ssp', 'model'], axis=1).unstack('year') 
    df.columns = df.columns.get_level_values(1)
    # Reset index
    df = df.reset_index() 
    df['hierid'] = df['region']
    df = df.merge(oldest['region'], on='region', how='right')
    # Save csv
    if data_type == 'GDP':
        df.to_csv(file_path+f'gdp_pop_csv/GDP_{scenario}_{gdp_model}.csv')
        
    elif data_type == 'POP':
        df.to_csv(file_path+f'gdp_pop_csv/POP_{scenario}_{age_group}.csv')
    
    return df



def gdp_pop_ssp_projections(
    wdir: str
    ) -> None:
    
    '''
    Generate GDP and population projection CSV files from Carleton et al. (2022) 
    Shared Socioeconomic Pathways (SSP) netCDF datasets.

    This function reads SSP population and GDP projections from NetCDF files 
    (via `var_nc_to_csv`) and exports them as CSV files for each scenario, 
    population group, and GDP model. It also computes the mean GDP projection 
    by averaging the "high" and "low" model variants for each scenario.

    Parameters
    ----------
    wdir : str
        Working directory path where NetCDF input files are located and 
        where CSV files will be saved. 

    Notes
    -----
    - Requires a helper function `var_nc_to_csv()` that converts a single NetCDF variable 
      to a CSV file given the provided parameters.
    - Generated CSV files are saved to the subdirectory `gdp_pop_csv/` within `wdir`.
    - GDP mean files are named using the pattern `"GDP_<scenario>_mean.csv"`.

    Outputs
    -------
    None
        The function writes CSV files to disk but does not return any Python object.
        
    '''
    
    # Define scenarios
    scenarios = ['SSP1', 'SSP2', 'SSP3', 'SSP5']
    # Define age groups
    age_groups = ['total', 'young', 'older', 'oldest']
    pop_groups = ['pop', 'pop0to4', 'pop5to64', 'pop65plus']
    # Define GDP model
    gdp_models = ['low', 'high']
    
    print('Generating GDP and Population projection csv files from Carleton et al. (2022)...')

    # Generate population files
    for scenario in scenarios:
        for pop_group, age_group in zip(pop_groups, age_groups):
            var_nc_to_csv(wdir, 'POP', scenario, pop_group, age_group, 'low')


    # Generate gdp files
    for scenario in scenarios:
        for gdp_model in gdp_models:
            var_nc_to_csv(wdir, 'GDP', scenario, 'gdppc', None, gdp_model)
            
        # Calculate mean gdp from high and low models
        SSP_GDP_high = pd.read_csv(f'{wdir}gdp_pop_csv/GDP_{scenario}_high.csv')
        SSP_GDP_low = pd.read_csv(f'{wdir}gdp_pop_csv/GDP_{scenario}_low.csv')
        SSP_GDP_mean = SSP_GDP_high.copy()
        for year in range(2010,2101):
            SSP_GDP_mean[f'{year}'] = (SSP_GDP_high[f'{year}'] + SSP_GDP_low[f'{year}']) / 2
        SSP_GDP_mean.to_csv(f'{wdir}gdp_pop_csv/GDP_{scenario}_mean.csv')
        
        

def region_classification_file(
    wdir: str,
    regions_file: str
    ) -> None:
    
    '''
    Generate a region classification CSV linking impact regions to IMAGE and GBD region codes.

    This function reads:
    - the Carleton et al. (2022) *impact regions* shapefile, and
    - an IMAGE region classification Excel file created manually,

    and combines them to produce a mapping between `hierid` codes, ISO3 country codes,
    and IMAGE/GBD regional classifications. The resulting file is exported as
    `region_classification.csv` in the specified working directory.

    Parameters
    ----------
    wdir : str
        Working directory path containing the Carleton model files.
        Must include a subdirectory `carleton_sm/ir_shp/` with `impact-region.shp`.
    regions_file : str
        Path to the IMAGE region classification Excel file.
        The Excel file must contain a sheet named `'regions'` with, a column `'ISO3'`
        and the corresponding regions information.

    Returns
    -------
    None
        The function writes the merged dataset to disk as a CSV file and does not return an object.

    Output
    ------
    A CSV file named:
        `region_classification.csv`
    located in the working directory (`wdir`).

    The resulting CSV typically contains columns such as:
        - 'hierid': Unique impact region identifier.
        - 'ISO3': ISO3 country code derived from the first 3 characters of `hierid`.
        - other region classification columns from the IMAGE regions file.
        
    '''
    
    print('Generating region classification file...')
    
    # Read IMAGE rtion excel file
    image_regions = pd.read_excel(regions_file, sheet_name='regions')

    # Read impact regions shapefile and extract regions names
    impact_regions = gpd.read_file(f'{wdir}'+'carleton_sm/ir_shp/impact-region.shp')
    impact_regions['ISO3'] = impact_regions['hierid'].str[:3]

    # Merge with IMAGE regions to get IMAGE region codes
    df = pd.merge(impact_regions[['hierid', 'ISO3']], image_regions, on='ISO3', how='left')
    df.to_csv(f'{wdir}'+'region_classification.csv', index=False)
    


def calculate_population_per_region(
    landscan_pop: rasterio.io.DatasetReader,
    impact_regions: gpd.GeoDataFrame,
    year: int
    ) -> gpd.GeoDataFrame:
    
    '''
    Calculate total population per impact region for a specific year using LandScan raster data.

    This function assigns raster pixels from the LandScan population dataset to impact regions,
    sums the population values within each region, and adds the results as a new column to
    the input GeoDataFrame.

    Parameters
    ----------
    landscan_pop : rasterio.io.DatasetReader
        Opened LandScan population raster (e.g., from `rasterio.open('landscan_pop.tif')`).
        One band containing population counts.
    impact_regions : geopandas.GeoDataFrame
        GeoDataFrame containing impact region polygons.
    year : int
        Year for which the population is being calculated (e.g., 2000, 2010).

    Returns
    -------
    impact_regions : geopandas.GeoDataFrame
        Updated GeoDataFrame with a new column named after `year` containing
        total population per impact region.

    '''

    # Read raster data and affine
    raster_data = landscan_pop.read(1)
    raster_affine = landscan_pop.transform

    # Mask no data values
    nodata_val = -2147483647
    valid_mask = raster_data != nodata_val

    # Create mask to assign pixels to impact regions
    shapes_and_ids = ((geom, idx) for idx, geom in enumerate(impact_regions.geometry, start=1))

    pixel_owner = rasterize(
        shapes_and_ids,
        out_shape=raster_data.shape,
        transform=raster_affine,
        fill=0,          # 0 = without region
        all_touched=False,
        dtype='int32'
    )

    # Calculate population sums per region
    masked_ids = pixel_owner[valid_mask]
    masked_values = raster_data[valid_mask]

    max_id = pixel_owner.max()
    sums = np.bincount(masked_ids, weights=masked_values, minlength=max_id + 1)[1:]

    # Add population sums to impact_regions GeoDataFrame
    impact_regions[year] = sums
    
    return impact_regions



def process_un_pop_data(
    wdir: str
    ) -> pd.DataFrame:
    
    '''
    Process UN population data to calculate the share of population per age group 
    for each country and year.

    This function reads a UN population CSV file, aggregates to three age groups, 
    and calculates the share of the population in the categories:
    'young' (0-4), 'older' (5-64), and 'oldest' (65+) for each country and year.

    Parameters
    ----------
    wdir : str
        Working directory containing the UN population CSV file.
        Expected file: 'unpopulation_dataportal.csv', with columns:
            - 'Iso3': ISO3 country code
            - 'Time': Year
            - 'Age': Age group (e.g., '0-4', '5-14', '15-64', '65+')
            - 'Value': Population count

    Returns
    -------
    share_pop : pandas.DataFrame
        DataFrame with population shares per age group and total population.
        Columns:
            - 'ISO': ISO3 country code
            - 'Time': Year
            - 'total': Total population for that country and year
            - 'share_young': Fraction of population aged 0-4
            - 'share_older': Fraction of population aged 5-64
            - 'share_oldest': Fraction of population aged 65+

    '''
    
    # Read UN population data file
    un_population = pd.read_csv(wdir+'unpopulation_dataportal.csv')

    # Keep relevant columns and aggregate age groups
    un_population = un_population[['Iso3', 'Time', 'Age', 'Value']]
    un_population.loc[un_population['Age'].isin(['5-14', '15-64']), 'Age'] = '5-64'

    # Calculate total population per country and year
    share_pop = un_population.groupby(['Iso3','Time']).sum().drop(columns='Age')

    # Aggregate population by age groups
    un_population = un_population.groupby(['Iso3', 'Time', 'Age']).sum()

    # Calculate share of young, older and oldest population
    share_pop['share_young'] = un_population.xs('0-4',level=2)['Value']/ share_pop['Value']
    share_pop['share_older'] = un_population.xs('5-64',level=2)['Value'] / share_pop['Value']
    share_pop['share_oldest'] =un_population.xs('65+',level=2)['Value']  / share_pop['Value']

    share_pop = share_pop.reset_index()

    share_pop = share_pop.rename(columns={'Iso3':'ISO', 'Value':'total'})
    
    return share_pop



def get_pop_share_file(
    wdir: str,
    impact_long: pd.DataFrame,
    impact_regions: pd.DataFrame,
    age_group: str
    ) -> None:
    
    '''
    Generate a CSV file with historical population per impact region for a specific age group.

    This function calculates the population of a given age group for each impact region
    by multiplying the total population (`Value`) by the age-specific population share
    (`share_<age_group>`). The resulting data is pivoted to a wide format (regions as rows,
    years as columns) and saved as a CSV file.

    Parameters
    ----------
    wdir : str
        Working directory where the output CSV file will be saved.
    impact_long : pandas.DataFrame
        Long-format DataFrame containing population and age group shares.
    impact_regions : pandas.DataFrame
        DataFrame containing impact region information. Must include 'hierid' for alignment.
    age_group : str
        Age group for which the population share is calculated (e.g., 'young', 'older', 'oldest').

    Returns
    -------
    None
        The function saves a CSV file named `'POP_historical_<age_group>.csv'` 
        in the subdirectory `'gdp_pop_csv/'` of `wdir`.

    '''
    
    # Calculate population share per age group
    impact_long[age_group] = impact_long['Value'] * impact_long[f'share_{age_group}']
    
    # Pivot to wide format and save
    impact_agegroup = (
        impact_long
        .pivot(index='hierid', columns='Time', values=age_group)
        .reset_index()
        .set_index('hierid')
        .reindex(impact_regions.set_index('hierid').index)
        )

    impact_agegroup.to_csv(wdir+'gdp_pop_csv/'+'POP_historical_'+age_group+'.csv')
    
    print(f'Population share file generated for age group: {age_group}')



def generate_historical_pop(
    wdir: str,
    landscan_file: str,
    impact_regions: str
    ) -> None:
    
    '''
    Generate historical population per impact region and age group from LandScan data.

    This function reads the LandScan global population raster and an impact region shapefile,
    calculates the total population per impact region for each year from 2000 to 2022,
    reshapes the data into long format, merges it with UN population share data per country 
    and age group, and generates population share CSV files for different age groups.

    Parameters
    ----------
    wdir : str
        Working directory where input and output files are stored.
    landscan_file : str
        Path to the LandScan population raster (`.tif` file).
    impact_regions : str
        Path to the shapefile containing impact regions (`impact-region.shp`).

    Returns
    -------
    None
        This function does not return a Python object. Instead, it generates CSV files per age group
        in the working directory.

    '''
    
    print('Generating historical population per impact region and age group...')
    
    # Open LandScan population raster and impact regions shapefile
    landscan_pop = rasterio.open(landscan_file)
    impact_regions = gpd.read_file(impact_regions)
    
    # Calculate population per impact region for each year from 2000 to 2022
    for year in range(2000,2023):
        impact_regions = calculate_population_per_region(landscan_pop, impact_regions, year)
        print(f'Population calculated for year: {year}')

    # Drop unnecessary columns
    impact_regions = impact_regions.drop(columns=['gadmid', 'color', 'AREA', 'PERIMETER', 'geometry'])

    # Reshape to long format
    impact_long = impact_regions.melt(id_vars=['hierid', 'ISO',], 
                                    var_name='Time', 
                                    value_name='Value')

    # Get UN population shares per country and year
    share_pop = process_un_pop_data(wdir)

    # Merge population shares with impact region population data
    impact_long = impact_long.merge(share_pop, on=['ISO', 'Time'], how='left')

    # Generate population share files per age group
    for age_group in ['young', 'older', 'oldest']:
        get_pop_share_file(wdir, impact_long, impact_regions, age_group)



def open_predictors(
    file_path: str
    ) -> pd.DataFrame:
    
    '''
    Load mortality data per age group for the year 2015 (No Adaptation scenario).

    This function reads a CSV file containing mortality predictors, skips the first
    13 rows (metadata or header info), filters the data for the year 2015,
    and resets the DataFrame index.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing mortality predictors. Expected columns include:
            - 'year': Year of the data
            - Other columns representing mortality rates per age group or region

    Returns
    -------
    df : pandas.DataFrame
        Filtered DataFrame containing only rows for the year 2015,
        with the index reset. Columns correspond to those in the input CSV.
    '''
    
    df = pd.read_csv(file_path, skiprows=13)
    df = df[df['year'] == 2015]
    df = df.reset_index()
    
    return df



def exposure_response_function(
    tas1: float,
    tas2: float,
    tas3: float,
    tas4: float,
    t: np.ndarray
) -> np.ndarray:
    
    '''
    Generate the mortality-temperature exposure response function based on Carleton et al. (2022).

    This function computes mortality values as a fourth-degree polynomial of temperature,
    applies a vertical shift so that the minimum mortality (Tmin) is zero, 
    and imposes weak monotonicity constraints outside the Tmin range, following 
    the specifications in Carleton et al. (2022).

    Parameters
    ----------
    tas1, tas2, tas3, tas4 : float
        Coefficients from the regression analysis in Carleton et al. (2022).
        They define the fourth-degree polynomial: 
            mortality_raw = tas1*t + tas2*t^2 + tas3*t^3 + tas4*t^4
    t : numpy.ndarray
        Array of temperature values (°C) at which to evaluate the mortality response.

    Returns
    -------
    mortality : numpy.ndarray
        Mortality values corresponding to each temperature in `t`, adjusted so that
        the minimum mortality occurs at zero and weak monotonicity is imposed.

    '''
   
    # Generate raw fourth degree polynomial function
    raw = tas1*t + tas2*t**2 + tas3*t**3 + tas4*t**4   
    
    # Find Tmin within 10–30°C
    idx_min_start = np.where(np.isclose(t, 10.0, atol=0.05))[0][0]
    idx_min_end = np.where(np.isclose(t, 30.0, atol=0.05))[0][0]
    tmin = t[idx_min_start + np.argmin(raw[idx_min_start:idx_min_end])]
    
    # Vertically shift the function to make Tmin be at zero
    mortality = raw - tas1*tmin - tas2*tmin**2 - tas3*tmin**3 - tas4*tmin**4  
    
    # Impose weak monotonicity (from the original paper)
    t_left = t[t < tmin]  
    t_right = t[t > tmin]
    
    # Apply weak monotonicity constraints
    if len(t_left) > 0:
        for i in range(len(t_left) - 1, -1, -1):
            mortality[i] = max(mortality[i], mortality[i + 1])
    
    if len(t_right) > 0:
        for i in range(len(t_left)+1, len(mortality)):
            mortality[i] = max(mortality[i-1], mortality[i]) 
    
    return mortality



def get_tmin(
    tas1: float,
    tas2: float,
    tas3: float,
    tas4: float,
    t: np.ndarray
    ) -> float:
    
    '''
    Compute the temperature at which the mortality response function is minimized (Tmin).

    This function evaluates a fourth-degree polynomial defined by the coefficients
    `tas1`, `tas2`, `tas3`, and `tas4` over an array of temperatures `t` and returns
    the temperature within 10–30°C at which the polynomial achieves its minimum value.

    Parameters
    ----------
    tas1, tas2, tas3, tas4 : float
        Coefficients of the fourth-degree polynomial:
            mortality_raw = tas1*t + tas2*t^2 + tas3*t^3 + tas4*t^4
    t : numpy.ndarray
        Array of temperature values (°C) over which to evaluate the polynomial.

    Returns
    -------
    tmin : float
        Temperature (°C) at which the mortality polynomial is minimized within the range 10–30°C.

    '''
    
    # Generate raw fourth degree polynomial function
    raw = tas1*t + tas2*t**2 + tas3*t**3 + tas4*t**4   
    
    # Find Tmin within 10–30°C
    idx_min_start = np.where(np.isclose(t, 10.0, atol=0.05))[0][0]
    idx_min_end = np.where(np.isclose(t, 30.0, atol=0.05))[0][0]
    tmin = t[idx_min_start + np.argmin(raw[idx_min_start:idx_min_end])]     
    
    return tmin



def generate_exposure_response_functions(
    wdir: str
    ) -> None:
    
    '''
    Generate Exposure Response Functions without Adaptation for different age groups.

    This function reads mortality predictor data for three age groups (oldest, older, young),
    computes the exposure response functions based on a fourth-degree polynomial,
    and saves the results as CSV files in the specified working directory.

    Parameters
    ----------
    wdir : str
        Working directory where input files are located and where output CSV files will be saved.

    Returns
    -------
    None
        The function writes three CSV files named:
            - `erf_no-adapt_oldest.csv`
            - `erf_no-adapt_older.csv`
            - `erf_no-adapt_young.csv`
        in the subdirectory `exposure_response_functions/` of `wdir`.

    '''
    
    print('Generating Exposure Response Functions without Adaptation for all age groups...')

    ### Determine the resolution of the temperature range
    t = np.arange(-50, 60.1, 0.1).round(1)

    ### Age group names
    age_groups = ['oldest', 'older', 'young']
    df_groups = [oldest, older, young]

    ### Open predictors dataframes
    oldest = open_predictors(wdir+'main_specification/mortality-allcalcs-Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1-older.csv')
    older = open_predictors(wdir+'main_specification/mortality-allcalcs-Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1-oldest.csv')
    young = open_predictors(wdir+'main_specification/mortality-allcalcs-Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1-young.csv')

    # Iterate for each age group
    for group, df_group in zip(age_groups, df_groups): 
        responses = []
        
        # Generate the response functions and append
        for i in range(len(df_group)):
            mortality = exposure_response_function(df_group['tas'][i], df_group['tas2'][i], df_group['tas3'][i], df_group['tas4'][i], t)
            responses.append(mortality.round(2))
            
        # Round column names
        df = pd.DataFrame(responses, columns=[f"{temp:.1f}" for temp in t])  
        
        # Add the region columns
        df_merge = pd.concat([df_group['region'], df], axis=1, join='inner')  

        # Save csv file
        df_merge.to_csv(f'{wdir}/exposure_response_functions/erf_no-adapt_{group}.csv')



def generate_tmin_file(
    wdir: str
    ) -> None:
    
    '''
    Generate a CSV file containing Tmin values for different age groups per impact region.

    This function reads mortality predictor data for three age groups (oldest, older, young),
    computes the temperature at which the mortality response function is minimized (Tmin)
    for each impact region and age group, and saves the results in a CSV file.

    Parameters
    ----------
    wdir : str
        Working directory where input files are located and where the output CSV will be saved.

    Returns
    -------
    None
        The function writes a CSV file named `T_min.csv` in the subdirectory 
        `exposure_response_functions/` of `wdir`.

    '''
    
    print('Generating Tmin file for all age groups...')
    
    # Define temperature range
    t = np.arange(-50, 60.1, 0.1).round(1)

    # Open predictors dataframes
    oldest = open_predictors(wdir+'main_specification/mortality-allcalcs-Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1-older.csv')
    older = open_predictors(wdir+'main_specification/mortality-allcalcs-Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1-oldest.csv')
    young = open_predictors(wdir+'main_specification/mortality-allcalcs-Agespec_interaction_GMFD_POLY-4_TINV_CYA_NW_w1-young.csv')    

    # Define dataframe to store Tmin values
    df = pd.DataFrame(oldest['region'])

    # Add empty columns for Tmin values of each age group
    df['Tmin oldest'] = ''
    df['Tmin older'] = ''
    df['Tmin young'] = ''

    # Get T_min for all age groups and IR
    for i in range(len(oldest)):
        df.iloc[i,1] = get_tmin(oldest['tas'][i], oldest['tas2'][i], oldest['tas3'][i], oldest['tas4'][i], t)
        df.iloc[i,2] = get_tmin(older['tas'][i], older['tas2'][i], older['tas3'][i], older['tas4'][i], t)
        df.iloc[i,3] = get_tmin(young['tas'][i], young['tas2'][i], young['tas3'][i], young['tas4'][i], t)

    # Save csv file
    df.to_csv(f'{wdir}/exposure_response_functions/T_min.csv') 