import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
from typing import Optional
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mortality_functions as mf



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
    
    # Read IMAGE csv file
    image_regions = pd.read_excel(regions_file, sheet_name='regions')

    # Read impact regions shapefile and extract regions names
    impact_regions = gpd.read_file(wdir+'data/carleton_sm/ir_shp/impact-region.shp')
    impact_regions['ISO3'] = impact_regions['hierid'].str[:3]

    # Merge with IMAGE regions to get IMAGE region codes
    df = pd.merge(impact_regions[['hierid', 'ISO3']], image_regions, on='ISO3', how='left')
    df.to_csv(wdir+'data/regions/region_classification.csv', index=False)
        


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
    un_population = pd.read_csv(wdir+'data/historical_pop/unpopulation_dataportal.csv')

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
    impact_long = impact_long['Value'] * impact_long[f'share_{age_group}']
    
    # Pivot to wide format and save
    impact_agegroup = (
        impact_long
        .pivot(index='hierid', columns='Time', values=age_group)
        .reset_index()
        .set_index('hierid')
        .reindex(impact_regions.set_index('hierid').index)
        )

    impact_agegroup.to_csv(wdir+'data/historical_pop/POP_historical_'+age_group+'.csv')
    
    print(f'Population share file generated for age group: {age_group}')
    
    

def import_present_day_temperature(wdir, era5_dir):
    
    years = range(2000,2011)
    
    spatial_relation, ir = mf.grid_relationship(wdir, "ERA5", era5_dir, years)
    
    for year in years:
        T_0 = mf.era5_temp_to_ir(era5_dir, year, ir, spatial_relation)
        T_0.to_csv(wdir+f'data/climate_data/ERA5_T0_{year}.csv')