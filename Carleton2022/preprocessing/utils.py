import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from pathlib import Path
import rasterio
from rasterio.features import rasterize
import numpy as np
import country_converter as coco
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mortality_functions as mf



def RegionClassificationFile(
    wdir: str,
    regions_file: str
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
        The function writes the merged dataset to disk as a CSV file and does not return an object.

    Output
    ------
    A CSV file named:
        `region_classification.csv`
    located in the working directory (`wdir`).

    The resulting CSV typically contains columns such as:
        - "hierid": Unique impact region identifier.
        - "ISO3": ISO3 country code derived from the first 3 characters of `hierid`.
        - other region classification columns from the IMAGE regions file.
        
    """
    
    print("Generating region classification file...")
    
    # Read IMAGE csv file
    image_regions = pd.read_excel(regions_file, sheet_name="regions")

    # Read impact regions shapefile and extract regions names
    impact_regions = gpd.read_file(wdir+"data/carleton_sm/ir_shp/impact-region.shp")
    impact_regions["ISO3"] = impact_regions["hierid"].str[:3]

    # Merge with IMAGE regions to get IMAGE region codes
    df = pd.merge(impact_regions[["hierid", "ISO3"]], image_regions, on="ISO3", how="left")
    df.to_csv(wdir+"data/regions/region_classification.csv", index=False)
        


def PopulationHistorical(
    wdir: str,
    landscan_path: str,
    impact_regions: str
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
    IMPACT_REGIONS = gpd.read_file(impact_regions)
    
    # Get UN population shares per country and year
    POPULATION_SHARES = ProcessUNPopulation5years(wdir)
    
    # Calculate population per impact region for each year from 2000 to 2022
    for year in range(2000,2023):
        # Open LandScan population raster
        LANDSCAN_POP = rasterio.open(landscan_path + f"landscan-global-{year}-assets/landscan-global-{year}.tif")
        IMPACT_REGIONS = Raster2ImpactRegionPopulation(LANDSCAN_POP, IMPACT_REGIONS, year)
        print(f"Population calculated for year: {year}")

    # Drop unnecessary columns, reshape to long format and mereg with UN population shares
    IMPACT_REGIONS_POP = (
        IMPACT_REGIONS
        .drop(columns=["gadmid", "color", "AREA", "PERIMETER", "geometry"])
        .melt(id_vars=["hierid", "ISO",], var_name="Time", value_name="Value")
        .merge(POPULATION_SHARES, on=["ISO", "Time"], how="left")
    )
    
    for age_group in ["young", "older", "oldest"]:
        
        # Calculate population share per age group
        IMPACT_REGIONS_POP[f"pop_{age_group}"] = IMPACT_REGIONS_POP["Value"] * IMPACT_REGIONS_POP[f"share_{age_group}"]

        # Pivot to wide format and save
        (
            IMPACT_REGIONS_POP
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
    UN_POPULATION = (
        pd.read_csv(wdir+"population/unpopulation_dataportal.csv")
        [["Iso3", "Time", "Age", "Value"]] # Keep relevant columns
    )
    UN_POPULATION.loc[UN_POPULATION["Age"].isin(["5-14", "15-64"]), "Age"] = "5-64"

    # Calculate total population per country and year
    POPULATION_TOTAL = UN_POPULATION.groupby(["Iso3","Time"]).sum().drop(columns="Age")

    # Aggregate population by age groups
    UN_POPULATION = UN_POPULATION.groupby(["Iso3", "Time", "Age"]).sum()

    # Calculate share of young, older and oldest population
    POPULATION_TOTAL["share_young"] = UN_POPULATION.xs("0-4",level=2)["Value"]/ POPULATION_TOTAL["Value"]
    POPULATION_TOTAL["share_older"] = UN_POPULATION.xs("5-64",level=2)["Value"] / POPULATION_TOTAL["Value"]
    POPULATION_TOTAL["share_oldest"] = UN_POPULATION.xs("65+",level=2)["Value"]  / POPULATION_TOTAL["Value"]

    POPULATION_TOTAL = POPULATION_TOTAL.reset_index()

    POPULATION_TOTAL = POPULATION_TOTAL.rename(columns={"Iso3":"ISO", "Value":"total"})
    
    return POPULATION_TOTAL



def PopulationProjections(wdir, pop_dir):
     
    """
    The code imports the population data produced by an IMAGE run and converts it to population data
    for the three age groups and SSP scenarios at the impact region level.
    
    Parameters:
    ----------
    wdir : str
        Working directory
        
    Returns:
    ----------
    None. 
        The function saves CSV files with population projections per age group and SSP scenario at the 
        impact region level in the subdirectory `data/population/pop_ssp` of the working directory.
        If no folder exists, it will be created.
        
    Data sources:
    ----------
    1. Population data projections per age group from the SSP projections, available at:
        https://dataexplorer.wittgensteincentre.org/wcde-v3/  (K.C. et al. (2024)).
        The data was populaiton size (000"s) at country level for all the 5-year age groups
        and SSP scenarios.
    """
    
    SSP = ["SSP1", "SSP2", "SSP3", "SSP5"]
    YEARS = range(2000, 2101)
    
    for ssp in SSP:
        
        # Agregate raster IMAGE total population per impact region and year
        TOTAL_POPULATION_IR = IMAGEPopulation2ImpactRegion(wdir=wdir, pop_dir=pop_dir, ssp=ssp, years=YEARS)
        
        # Load population data projections per age group to disagregate IMAGE data
        POPULATION_GROUPS = LoadAgeGroupPopulationData(pop_dir=pop_dir, ssp=ssp, years=YEARS)
        
        # Pivot and merge function
        def pivot_and_merge(group_name):
            
            df = POPULATION_GROUPS[POPULATION_GROUPS["group"] == group_name].pivot(index=["Area", "ISO3"], columns="Year", values="share").reset_index()
            df = df.rename(columns={c: f"{c}_share" for c in df.columns if isinstance(c, int)})
            return TOTAL_POPULATION_IR.merge(df, on="ISO3", how="left")
        
        # Create population dataframes for each age group
        POP_YOUNG, POP_OLDER, POP_OLDEST = (pivot_and_merge(g) for g in ["young", "older", "oldest"])
        
        # Create output directory if it doesn"t exist
        out_path = Path(wdir) / "data" / "population" / "pop_ssp"
        out_path.mkdir(parents=True, exist_ok=True)

        # Multiply shares by total population to get absolute numbers
        for pop, group in zip([POP_YOUNG, POP_OLDER, POP_OLDEST],["young", "older", "oldest"]):        
            for y in YEARS:
                pop[str(y)] = pop[str(y)+"_share"] * pop[str(y)]
                        
            NON_SHARE_COLS = [c for c in pop.columns if "share" not in c]
            
            # Save population projection files per age group and SSP scenario
            pop[NON_SHARE_COLS].to_csv(out_path/ f"pop_{ssp.lower()}_{group}.csv", index=False)

        
        
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
    IMPACT_REGIONS = gpd.read_file(wdir+"data/carleton_sm/ir_shp/impact-region.shp")
    
    # Read IMAGE SSP population nc file
    POP_IMAGE = xr.open_dataset(pop_dir+f"/IMAGE_POP/{ssp.upper()}/GPOP.nc")
    
    # Ensure CRS is set to EPSG:4326 and align with impact regions
    POP_IMAGE = POP_IMAGE.rio.write_crs("EPSG:4326", inplace=False)
    IMPACT_REGIONS = IMPACT_REGIONS.to_crs(POP_IMAGE.rio.crs)

    # Select relevant years including "present-day" years (2000-2010)
    POP_IMAGE = POP_IMAGE.sel(time=pd.to_datetime([f"{y}-01-01" for y in years]))
    
    MINLENGTH = len(IMPACT_REGIONS) + 1

    # Prepare tuples of (geometry, region_id) for rasterization
    SHAPES_AND_IDS = [(geom, idx) for idx, geom in enumerate(IMPACT_REGIONS.geometry, start=1)]
        
    # Rasterize region polygons once
    OUT_SHAPE = POP_IMAGE.isel(time=0).GPOP.shape

    # Get raster transform 
    RASTER_AFFINE = POP_IMAGE.rio.transform()    

    PIXEL_OWNER = rasterize(
        SHAPES_AND_IDS,
        out_shape=OUT_SHAPE,
        transform=RASTER_AFFINE,
        fill=0,          # 0 = without region
        all_touched=False,
        dtype="int32"
    )
    
    year_data = {}
    
    for i, year in enumerate(years):
        
        RASTER_DATA = POP_IMAGE.isel(time=i).GPOP.values
        
        # Mask valid data (NaN = nodata)
        VALID_POP_MASK = ~np.isnan(RASTER_DATA)

        # Sum population per region using np.bincount in pixels without NaN
        SUMS = np.bincount(
            PIXEL_OWNER[VALID_POP_MASK], 
            weights=RASTER_DATA[VALID_POP_MASK], 
            minlength=MINLENGTH
        )[1:]  

        # Add results to impact_regions GeoDataFrame
        year_data[str(year)] = SUMS
        
    IMPACT_REGIONS = pd.concat(
        [IMPACT_REGIONS, pd.DataFrame(year_data, index=IMPACT_REGIONS.index)],
        axis=1
    )
    
    # Add ISO3 column
    IMPACT_REGIONS["ISO3"] = IMPACT_REGIONS["hierid"].str[:3]

    # Only return regions names and population columns
    return IMPACT_REGIONS[["hierid", "ISO3"] + [c for c in IMPACT_REGIONS.columns if str(c).isdigit()]]



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
    POPULATION_5YEAR_AGE = (pd.read_csv(pop_dir+"/WCDE_POP_SSP/wcde_data.csv", 
                                       skiprows=8)
                            .query("Scenario == @ssp and Year in @years")
    )
    
    POPULATION_5YEAR_AGE = CompletePopulationDataLustrum(POPULATION_5YEAR_AGE)
    
    # Define age groups to classify wcde ages
    AGE_GROUPS = {
        "young": ["0--4"],
        "older": [f"{i}--{i+4}" for i in range(5, 65, 5)],
        "oldest": [f"{i}--{i+4}" for i in range(65, 100, 5)] + ["100+"]
    }
    
    # Assign age group to each age
    POPULATION_5YEAR_AGE.loc[:,"group"] = POPULATION_5YEAR_AGE["Age"].map(
        lambda x: next((grp for grp, ages in AGE_GROUPS.items() if x in ages), None)
    )
    
    # Aggregate population by group
    POPULATION_GROUPS = (
        POPULATION_5YEAR_AGE
        .dropna(subset=["group"]) # Drop rows with ages that don"t fit into defined groups ("All")
        .groupby(["Area", "Year", "group"], as_index=False)["Population"]
        .sum()
    )
    
    # Generate rows with missing years per area and group
    dfs = []

    # Interpolate for every Area and group combination
    for (area, group), group_df in POPULATION_GROUPS.groupby(["Area", "group"]):
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
    POPULATION_GROUPS_ANNUAL =  pd.concat(dfs, ignore_index=True)
    
    # Calculate share of each age group
    POPULATION_GROUPS_ANNUAL["Population_total"] = (
        POPULATION_GROUPS_ANNUAL
        .groupby(["Area", "Year"])["Population"]
        .transform("sum")
    )
    POPULATION_GROUPS_ANNUAL["share"] = (
        POPULATION_GROUPS_ANNUAL["Population"] / 
        POPULATION_GROUPS_ANNUAL["Population_total"]
    )
    
    # Convert locations to ISO3
    unique_locations = POPULATION_GROUPS_ANNUAL["Area"].unique()
    conversion_dict = {loc: coco.convert(names=loc, to="ISO3") for loc in unique_locations}
    POPULATION_GROUPS_ANNUAL["ISO3"] = POPULATION_GROUPS_ANNUAL["Area"].map(conversion_dict)
    
    return POPULATION_GROUPS_ANNUAL



def CompletePopulationDataLustrum(df):
    
    """
    Fill in missing years in the population data by forward-filling and backward-filling.
    This ensure that countries without data for certain years will have values filled in
    based on the nearest available data, keeping age group share consistent.
    """
    
    # Create a MultiIndex of all combinations of Area, Scenario, and Years
    UNIQUE_MLTIDX = pd.MultiIndex.from_product(
        [
            df["Area"].unique(),
            df["Scenario"].unique(),
            df["Year"].unique(),
            df["Age"].unique()
        ],
        names=["Area", "Scenario", "Year", "Age"]
    )
    
    # Reindex the DataFrame to include all combinations, filling missing values with NaN
    DF_FULL = (
        df
        .set_index(["Area", "Scenario", "Year", "Age"])
        .reindex(UNIQUE_MLTIDX)
        .reset_index()
    )
    
    # Forward-fill and backward-fill missing values within each Population and Scenario group
    DF_FULL["Population"] = (
        DF_FULL
        .groupby(["Area", "Scenario", "Year", "Age"])["Population"]
        .transform("bfill")
        .transform("ffill")
    )
    
    return DF_FULL



def DailyTemperaturesERA5PresentDay(wdir, era5_dir):
    
    years = range(2000,2011)
    
    spatial_relation, ir = mf.GridRelationship(wdir, "ERA5", era5_dir, years)
    
    for year in years:
        T_0 = mf.ERA5Temperature2IR(era5_dir, year, ir, spatial_relation)
        T_0.to_csv(wdir+f"data/climate_data/ERA5_T0_{year}.csv")
        
        

def ClimatologiesERA5(wdir, era5_dir, years):
    
    # Get spatial relationship between ERA5 grid and impact regions
    spatial_relation, ir = mf.GridRelationship(wdir, None,"ERA5", era5_dir, None, years)
    
    # Define period for climatology calculation (30-year running mean)
    PERIOD = 30

    # Calculate annual mean temperature for each year used in the analysis using ERA5 daily data
    ANNUAL_TEMPERATURES = {}
    
    for y in range(years[0]-PERIOD, years[-1]):
        print("Calculating annual mean temperature for year:", y)

        with xr.open_dataset(f"{era5_dir}era5_t2m_mean_day_{y}.nc") as ds:
            ANNUAL_TEMPERATURES[y] = ds["t2m"].mean(dim="valid_time")  - 273.15
        
    # Calculate climatology as the 30-year running mean of annual temperatures in the analysis period
    CLIMATOLOGIES_DIC = { }
    
    for year in years:
        print("Calculating climatology for year:", year)

        YEARS30 = [year-PERIOD, year-1]

        RUNNING_SUM = sum(ANNUAL_TEMPERATURES[y] for y in YEARS30)
        CLIMATOLOGY = RUNNING_SUM / len(YEARS30)
        
        CLIMATOLOGY = CLIMATOLOGY.values.ravel()
        CLIMATOLOGIES_DIC[year] = CLIMATOLOGY[spatial_relation.index]
        
    # Apply spatial relationship to get climatology values at the impact region level 
    CLIMATOLOGIES_DF = pd.DataFrame(CLIMATOLOGIES_DIC, index=spatial_relation["index_right"])
    # Average them if multiple grid cells correspond to the same region
    CLIMATOLOGIES_DF = (CLIMATOLOGIES_DF.groupby("index_right").mean().round(1))
    CLIMATOLOGIES_DF.insert(0, "hierid", ir)
    
    CLIMATOLOGIES_DF.to_csv(wdir+f"data/climate_data/era5/climatologies/ERA5_CLIMTAS_{years[0]}-{years[-1]}.csv",
                            index=False)
