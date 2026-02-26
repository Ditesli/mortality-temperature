import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from pathlib import Path
import rasterio
from rasterio.features import rasterize
import country_converter as coco
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mortality_functions as mf



def RegionClassificationFile(
    wdir: str,
    regions_class: str
    ) -> None:
    
    """
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
        The Excel file must contain a sheet named `"regions"` with, a column `"ISO3"`
        and the corresponding regions information.

    Returns
    -------
    None
        The function writes the merged dataset to disk as a CSV file.
    """
    
    print("Generating region classification file...")
    
    # Read IMAGE csv file
    image_regions = pd.read_excel(regions_class+"region_classification.csv", sheet_name="regions")

    # Read impact regions shapefile and extract regions names
    impact_regions = gpd.read_file(wdir+"data/carleton_sm/ir_shp/impact-region.shp")
    impact_regions["ISO3"] = impact_regions["hierid"].str[:3]

    # Merge with IMAGE regions to get IMAGE region codes
    df = pd.merge(impact_regions[["hierid", "ISO3"]], image_regions, on="ISO3", how="left")
    df.to_csv(wdir+"data/regions/region_classification.csv", index=False)
        


def PopulationHistorical(
    wdir: str,
    landscan_path: str,
    ) -> None:
    
    """
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

    """
    
    print("Generating historical population per impact region and age group...")
    
    # Open impact regions shapefile
    impact_regions = gpd.read_file(wdir+"/carleton_sm/ir_shp/impact-region.shp").to_crs("EPSG:4326")
    
    # Get UN population shares per country and year
    population_shares = ProcessUNPopulation5years(wdir)
    
    # Calculate population per impact region for each year from 2000 to 2022
    for year in range(2000,2023):
        # Open LandScan population raster
        landscan_pop = rasterio.open(landscan_path + f"landscan-global-{year}-assets/landscan-global-{year}.tif")
        impact_regions = Raster2ImpactRegionPopulation(landscan_pop, impact_regions, year)
        print(f"Population calculated for year: {year}")

    # Drop unnecessary columns, reshape to long format and mereg with UN population shares
    impact_regions_pop = (
        impact_regions
        .drop(columns=["gadmid", "color", "AREA", "PERIMETER", "geometry"])
        .melt(id_vars=["hierid", "ISO",], var_name="Time", value_name="Value")
        .merge(population_shares, on=["ISO", "Time"], how="left")
    )
    
    for age_group in ["young", "older", "oldest"]:
        
        # Calculate population share per age group
        impact_regions_pop[f"pop_{age_group}"] = impact_regions_pop["Value"] * impact_regions_pop[f"share_{age_group}"]

        # Pivot to wide format and save
        (
            impact_regions_pop
            [["hierid", "Time", f"pop_{age_group}"]]
            .pivot(index="hierid", columns="Time", values=f"pop_{age_group}")
            .reset_index()
            .set_index("hierid")
            .to_csv(wdir+"/population/pop_historical/pop_historical_"+age_group+".csv")
        )
        
        print(f"Population share file generated for age group: {age_group}")
        


def Raster2ImpactRegionPopulation(
    landscan_pop: rasterio.io.DatasetReader,
    impact_regions: gpd.GeoDataFrame,
    year: int
    ) -> gpd.GeoDataFrame:
    
    """
    Calculate total population per impact region for a specific year using LandScan raster data.

    This function assigns raster pixels from the LandScan population dataset to impact regions,
    sums the population values within each region, and adds the results as a new column to
    the input GeoDataFrame.

    Parameters
    ----------
    landscan_pop : rasterio.io.DatasetReader
        Opened LandScan population raster (e.g., from `rasterio.open("landscan_pop.tif")`).
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

    """

    # Read raster data and affine
    raster_data = landscan_pop.read(1)
    raster_data = landscan_pop.transform

    # Mask no data values
    nodata_val = -2147483647
    valid_mask = raster_data != nodata_val

    # Create mask to assign pixels to impact regions
    shapes_and_ids = ((geom, idx) for idx, geom in enumerate(impact_regions.geometry, start=1))

    pixel_owner = rasterize(
        shapes_and_ids,
        out_shape=raster_data.shape,
        transform=raster_data,
        fill=0,          # 0 = without region
        all_touched=False,
        dtype="int32"
    )

    # Calculate population sums per region
    masked_ids = pixel_owner[valid_mask]
    masked_values = raster_data[valid_mask]

    max_id = pixel_owner.max()
    sums = np.bincount(masked_ids, weights=masked_values, minlength=max_id + 1)[1:]

    # Add population sums to impact_regions GeoDataFrame
    impact_regions[year] = sums
    
    return impact_regions



def ProcessUNPopulation5years(
    wdir: str
    ) -> pd.DataFrame:
    
    """
    Process UN population data to calculate the share of population per age group 
    for each country and year.

    This function reads a UN population CSV file, aggregates to three age groups, 
    and calculates the share of the population in the categories:
    "young" (0-4), "older" (5-64), and "oldest" (65+) for each country and year.

    Parameters
    ----------
    wdir : str
        Working directory containing the UN population CSV file.
        Expected file: "unpopulation_dataportal.csv", with columns:
            - "Iso3": ISO3 country code
            - "Time": Year
            - "Age": Age group (e.g., "0-4", "5-14", "15-64", "65+")
            - "Value": Population count

    Returns
    -------
    share_pop : pandas.DataFrame
        DataFrame with population shares per age group and total population.
        Columns:
            - "ISO": ISO3 country code
            - "Time": Year
            - "total": Total population for that country and year
            - "share_young": Fraction of population aged 0-4
            - "share_older": Fraction of population aged 5-64
            - "share_oldest": Fraction of population aged 65+

    """
    
    # Read UN population data file
    un_population = (
        pd.read_csv(wdir+"population/unpopulation_dataportal.csv")
        [["Iso3", "Time", "Age", "Value"]] # Keep relevant columns
    )
    un_population.loc[un_population["Age"].isin(["5-14", "15-64"]), "Age"] = "5-64"

    # Calculate total population per country and year
    population_total = un_population.groupby(["Iso3","Time"]).sum().drop(columns="Age")

    # Aggregate population by age groups
    un_population = un_population.groupby(["Iso3", "Time", "Age"]).sum()

    # Calculate share of young, older and oldest population
    population_total["share_young"] = un_population.xs("0-4",level=2)["Value"]/ population_total["Value"]
    population_total["share_older"] = un_population.xs("5-64",level=2)["Value"] / population_total["Value"]
    population_total["share_oldest"] = un_population.xs("65+",level=2)["Value"]  / population_total["Value"]

    population_total = population_total.reset_index()

    population_total = population_total.rename(columns={"Iso3":"ISO", "Value":"total"})
    
    return population_total



def PopulationProjections(wdir, pop_dir):
     
    """
    The code imports the population data produced by an IMAGE run and converts it to population data
    for the three age groups and SSP scenarios at the impact region level.
    
    Parameters:
    ----------
    wdir : str
        Working directory
    pop_dir : str
        Directory where the IMAGE population nc files are stored. 
        
    Returns:
    ----------
    None. 
        The function saves CSV files with population projections per age group and SSP scenario at the 
        impact region level in the subdirectory `data/population/pop_ssp` of the working directory.
        If no folder exists, it will be created.
    """
    
    SSP = ["SSP1", "SSP2", "SSP3", "SSP5"]
    YEARS = range(2000, 2101)
    
    for ssp in SSP:
        
        # Agregate raster IMAGE total population per impact region and year
        total_population_ir = IMAGEPopulation2ImpactRegion(wdir=wdir, pop_dir=pop_dir, ssp=ssp, years=YEARS)
        
        # Load population data projections per age group to disagregate IMAGE data
        population_groups = LoadAgeGroupPopulationData(pop_dir=pop_dir, ssp=ssp, years=YEARS)
        
        # Pivot and merge function
        def pivot_and_merge(group_name):
            
            df = population_groups[population_groups["group"] == group_name].pivot(index=["Area", "ISO3"], columns="Year", values="share").reset_index()
            df = df.rename(columns={c: f"{c}_share" for c in df.columns if isinstance(c, int)})
            return total_population_ir.merge(df, on="ISO3", how="left")
        
        # Create population dataframes for each age group
        pop_young, pop_older, pop_oldest = (pivot_and_merge(g) for g in ["young", "older", "oldest"])
        
        # Create output directory if it doesn"t exist
        out_path = Path(wdir) / "data" / "population" / "pop_ssp"
        out_path.mkdir(parents=True, exist_ok=True)

        # Multiply shares by total population to get absolute numbers
        for pop, group in zip([pop_young, pop_older, pop_oldest],["young", "older", "oldest"]):        
            for y in YEARS:
                pop[str(y)] = pop[str(y)+"_share"] * pop[str(y)]
                        
            non_share_cols = [c for c in pop.columns if "share" not in c]
            
            # Save population projection files per age group and SSP scenario
            pop[non_share_cols].to_csv(out_path/ f"pop_{ssp.lower()}_{group}.csv", index=False)

        
        
def IMAGEPopulation2ImpactRegion(wdir, pop_dir, ssp, years):
    
    """
    Calculate total population per impact region for a specific year from IMAGE land
    population data files.

    This function assigns raster pixels from the IMAGE land population data to impact regions,
    sums the population values within each region, and adds the results as a new column to
    the input GeoDataFrame.

    Parameters
    ----------
    pop :  xarray
        IMAGE Land population nc4 file. Usually 5min resolution.
    impact_regions : geopandas.GeoDataFrame
        GeoDataFrame containing impact region polygons.
    year : int
        Year for which the population is being calculated (e.g., 2000, 2010).

    Returns
    -------
    impact_regions : pd.DataFrame
        oDataFrame with columns for the impact region, ISO3 and the corresponding population 
        per year.
    """
    
    print(f"Calculating population per impact region for SSP: {ssp}...")
    
    # Read in impact regions shapefile
    impact_regions = gpd.read_file(wdir+"data/carleton_sm/ir_shp/impact-region.shp")
    
    # Read IMAGE SSP population nc file
    pop_image = xr.open_dataset(pop_dir+f"/IMAGE_POP/{ssp.upper()}/GPOP.nc")
    
    # Ensure CRS is set to EPSG:4326 and align with impact regions
    pop_image = pop_image.rio.write_crs("EPSG:4326", inplace=False)
    impact_regions = impact_regions.to_crs(pop_image.rio.crs)

    # Select relevant years including "present-day" years (2000-2010)
    pop_image = pop_image.sel(time=pd.to_datetime([f"{y}-01-01" for y in years]))
    
    minlength = len(impact_regions) + 1

    # Prepare tuples of (geometry, region_id) for rasterization
    shapes_and_ids = [(geom, idx) for idx, geom in enumerate(impact_regions.geometry, start=1)]
        
    # Rasterize region polygons once
    out_shape = pop_image.isel(time=0).GPOP.shape

    # Get raster transform 
    raster_affine = pop_image.rio.transform()    

    pixel_owner = rasterize(
        shapes_and_ids,
        out_shape=out_shape,
        transform=raster_affine,
        fill=0,          # 0 = without region
        all_touched=False,
        dtype="int32"
    )
    
    year_data = {}
    
    for i, year in enumerate(years):
        
        raster_data = pop_image.isel(time=i).GPOP.values
        
        # Mask valid data (NaN = nodata)
        valid_pop_mask = ~np.isnan(raster_data)

        # Sum population per region using np.bincount in pixels without NaN
        sums = np.bincount(
            pixel_owner[valid_pop_mask], 
            weights=raster_data[valid_pop_mask], 
            minlength=minlength
        )[1:]  

        # Add results to impact_regions GeoDataFrame
        year_data[str(year)] = sums
        
    impact_regions = pd.concat(
        [impact_regions, pd.DataFrame(year_data, index=impact_regions.index)],
        axis=1
    )
    
    # Add ISO3 column
    impact_regions["ISO3"] = impact_regions["hierid"].str[:3]

    # Only return regions names and population columns
    return impact_regions[["hierid", "ISO3"] + [c for c in impact_regions.columns if str(c).isdigit()]]



def LoadAgeGroupPopulationData(pop_dir, ssp, years):
    
    """
    Load and process population data projections per age group from the SSP projections.
    This function reads the population projections from the SSP data, classifies ages into three groups
    (young, older, oldest), aggregates population by these groups, and calculates the share of each age group
    for each country and year. The resulting DataFrame includes the population share of each age group
    for each country and year, along with the corresponding ISO3 country codes.
    
    Parameters:
    ----------
    pop_dir : str
        Directory path where the population data projections are stored.
    ssp : str
        SSP scenario name (e.g., "SSP1", "SSP2", "SSP3", "SSP5").
    years : list or range
        List or range of years to include in the analysis (e.g., range(2000, 2101)).
        
    Returns:
    ----------
    pd.DataFrame
        A DataFrame containing the population share of each age group for each country and year, 
        along with ISO3 country codes. The DataFrame has columns:
            - "Area": Country name
            - "Year": Year
            - "group": Age group ("young", "older", "oldest")
            - "Population": Population count for that age group
            - "Population_total": Total population for that country and year
            - "share": Share of the population in that age group (Population / Population_total)
            - "ISO3": ISO3 country code
    """
    
    # Load population data projections per 5-year age group
    population_5year_age = (pd.read_csv(pop_dir+"/WCDE_POP_SSP/wcde_data.csv", 
                                       skiprows=8)
                            .query("Scenario == @ssp and Year in @years")
    )
    
    population_5year_age = CompletePopulationDataLustrum(population_5year_age)
    
    # Define age groups to classify wcde ages
    age_groups = {
        "young": ["0--4"],
        "older": [f"{i}--{i+4}" for i in range(5, 65, 5)],
        "oldest": [f"{i}--{i+4}" for i in range(65, 100, 5)] + ["100+"]
    }
    
    # Assign age group to each age
    population_5year_age.loc[:,"group"] = population_5year_age["Age"].map(
        lambda x: next((grp for grp, ages in age_groups.items() if x in ages), None)
    )
    
    # Aggregate population by group
    population_groups = (
        population_5year_age
        .dropna(subset=["group"]) # Drop rows with ages that don"t fit into defined groups ("All")
        .groupby(["Area", "Year", "group"], as_index=False)["Population"]
        .sum()
    )
    
    # Generate rows with missing years per area and group
    dfs = []

    # Interpolate for every Area and group combination
    for (area, group), group_df in population_groups.groupby(["Area", "group"]):
        # Create year range
        # years = range(group_df["Year"].min(), group_df["Year"].max() + 1)
        # Reindex to include all years
        group_df = group_df.set_index("Year").reindex(years)
        # Extend area and group columns
        group_df["Area"] = area
        group_df["group"] = group
        # Interpolate population values
        group_df["Population"] = group_df["Population"].interpolate(method="linear")
        # Reset index
        group_df = group_df.reset_index().rename(columns={"index": "Year"})
        dfs.append(group_df)

    # Concateenate all dataframes
    population_groups_annual =  pd.concat(dfs, ignore_index=True)
    
    # Calculate share of each age group
    population_groups_annual["Population_total"] = (
        population_groups_annual
        .groupby(["Area", "Year"])["Population"]
        .transform("sum")
    )
    population_groups_annual["share"] = (
        population_groups_annual["Population"] / 
        population_groups_annual["Population_total"]
    )
    
    # Convert locations to ISO3
    unique_locations = population_groups_annual["Area"].unique()
    conversion_dic = {loc: coco.convert(names=loc, to="ISO3") for loc in unique_locations}
    population_groups_annual["ISO3"] = population_groups_annual["Area"].map(conversion_dic)
    
    return population_groups_annual



def CompletePopulationDataLustrum(df):
    
    """
    Fill in missing years in the population data by forward-filling and backward-filling.
    This ensure that countries without data for certain years will have values filled in
    based on the nearest available data, keeping age group share consistent.
    """
    
    # Create a MultiIndex of all combinations of Area, Scenario, and Years
    unique_mltidx = pd.MultiIndex.from_product(
        [
            df["Area"].unique(),
            df["Scenario"].unique(),
            df["Year"].unique(),
            df["Age"].unique()
        ],
        names=["Area", "Scenario", "Year", "Age"]
    )
    
    # Reindex the DataFrame to include all combinations, filling missing values with NaN
    df_full = (
        df
        .set_index(["Area", "Scenario", "Year", "Age"])
        .reindex(unique_mltidx)
        .reset_index()
    )
    
    # Forward-fill and backward-fill missing values within each Population and Scenario group
    df_full["Population"] = (
        df_full
        .groupby(["Area", "Scenario", "Year", "Age"])["Population"]
        .transform("bfill")
        .transform("ffill")
    )
    
    return df_full



def DailyTemperaturesERA5PresentDay(wdir, era5_dir):
    
    # Present day years for which to calculate T_0 (2000-2010)
    YEARS = range(2000,2011)
    
    # Get spatial relationship between ERA5 grid and impact regions
    spatial_relation, ir = mf.GridRelationship(wdir, "ERA5", era5_dir, YEARS)
    
    # Create directory if it doesn't exist
    out_path = Path(wdir) / "data" / "climate_data"
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate T_0 for each year and save as CSV
    for year in YEARS:
        t_0 = mf.ERA5Temperature2IR(era5_dir, year, ir, spatial_relation)
        t_0.to_csv(wdir+f"data/climate_data/ERA5_T0_{year}.csv")
        
        

def ClimatologiesERA5(wdir, era5_dir, years):
    
    # Get spatial relationship between ERA5 grid and impact regions
    class Settings:
        def __init__(self, wdir, scenario, era5_dir, years):
            self.wdir = wdir
            self.scenario = scenario
            self.temp_path = era5_dir
            self.years = years
            
    sets = Settings(wdir=wdir, scenario="ERA5", era5_dir=era5_dir, years=years)

    spatial_relation, ir = mf.GridRelationship(sets)
    
    # Define period for climatology calculation (30-year running mean)
    period = 30

    # Calculate annual mean temperature for each year used in the analysis using ERA5 daily data
    annual_temperatures = {}
    
    for y in range(years[0]-period, years[-1]):
        print("Calculating annual mean temperature for year:", y)

        with xr.open_dataset(f"{era5_dir}era5_t2m_mean_day_{y}.nc") as ds:
            annual_temperatures[y] = ds["t2m"].mean(dim="valid_time")  - 273.15
            annual_temperatures[y] = (
                annual_temperatures[y]
                .assign_coords(longitude=((annual_temperatures[y].longitude + 180) % 360 - 180))
                .sortby("longitude")
            )
        
    # Calculate climatology as the 30-year running mean of annual temperatures in the analysis period
    climatologies_dic = { }
    
    for year in years:
        print("Calculating climatology for year:", year)

        years30 = range(year-period, year)

        running_sum = sum(annual_temperatures[y] for y in years30)
        climatology = running_sum / len(years30)
        
        climatology = climatology.values.ravel()
        climatologies_dic[year] = climatology[spatial_relation.index]
    
    # Apply spatial relationship to get climatology values at the impact region level 
    climatologies_df = pd.DataFrame(climatologies_dic, index=spatial_relation["index_right"])
    # Average them if multiple grid cells correspond to the same region
    climatologies_df = climatologies_df.groupby("index_right").mean()
    
    climatologies_df.insert(0, "hierid", ir)
    climatologies_df.to_csv(wdir+f"data/climate_data/era5/climatologies/ERA5_CLIMTAS_{years[0]}-{years[-1]}.csv",
                            index=False)
