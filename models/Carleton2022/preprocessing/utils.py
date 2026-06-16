import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
from pathlib import Path
import rasterio
from rasterio.features import rasterize
import country_converter as coco
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Carleton2022.model import mortality_functions as mf

        


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
    """
    
    print("Generating historical population per impact region and age group...")
    
    # Open impact regions shapefile
    impact_regions = gpd.read_file(
        wdir+"/data/CarletonSM/ir_shp/impact-region.shp"
        ).to_crs("EPSG:4326")
    
    # Get UN population shares per country and year
    population_shares = ProcessUNPopulation5years(wdir)
    
    # Calculate population per impact region for each year from 2000 to 2022
    for year in range(2000,2023):
        # Open LandScan population raster
        landscan_pop = rasterio.open(
            landscan_path 
            + f"/landscan-global-{year}-assets/landscan-global-{year}.tif"
            )
        impact_regions = Raster2ImpactRegionPopulation(landscan_pop, impact_regions, year)
        print(f"Population calculated for year: {year}")

    # Drop unnecessary columns, reshape to long format and mereg with UN population shares
    impact_regions_pop = (
        impact_regions
        .drop(columns=["gadmid", "color", "AREA", "PERIMETER", "geometry"])
        .melt(id_vars=["hierid", "ISO",], var_name="Time", value_name="Value")
        .rename(columns={"Time":"Year", "ISO":"ISO3"})
        .merge(population_shares, on=["ISO3", "Year"], how="left")
    )
    
    # Calculate population share per age group
    impact_regions_pop[f"pop"] = impact_regions_pop["Value"] * impact_regions_pop["share"]
    
    for age_group in ["young", "older", "oldest"]:
        
        # Pivot to wide format and save
        (
            impact_regions_pop
            [impact_regions_pop["group"] == age_group]
            [["hierid", "Year", f"pop"]]
            .pivot(index="hierid", columns="Year", values=f"pop")
            .reset_index()
            .set_index("hierid")
            .to_csv(
                wdir+"/data/Population/PopulationHistorical/pop_historical_"+age_group+".csv",
                float_format='%.2f'
                )
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
    """

    # Read raster data
    raster_data = landscan_pop.read(1)

    # Mask no data values
    nodata_val = -2147483647
    valid_mask = raster_data != nodata_val

    # Create mask to assign pixels to impact regions
    shapes_and_ids = [(geom, idx) for idx, geom in enumerate(impact_regions.geometry, start=1)]

    pixel_owner = rasterize(
        shapes_and_ids,
        out_shape=raster_data.shape,
        transform=landscan_pop.transform,
        fill=0,          # 0 = without region
        all_touched=True, # Consider pixels touched by the geometry as part of the region
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
    for each country and year. File contains 5-year age groups population data from
    1950 to 2025 at country level.

    This function reads a UN population CSV file, aggregates to three age groups, 
    and calculates the share of the population in the categories:
    "young" (0-4), "older" (5-64), and "oldest" (65+) for each country and year.
    """
    
    # Read UN population data file
    un_population = (
        pd.read_csv(
            os.path.dirname(wdir)+
            "/data/un_population/unpopulation_dataportal.csv")
        [["Iso3", "Time", "Age", "Value"]] # Keep relevant columns
        .rename(columns={"Time":"Year", "Value":"Population", "Iso3":"ISO3"}) # Rename columns for consistency
    )
      
    # Define age groups to classify wcde ages
    age_groups = {
        "young": ["0-4"],
        "older": [f"{i}-{i+4}" for i in range(5, 65, 5)],
        "oldest": [f"{i}-{i+4}" for i in range(65, 100, 5)] + ["100+"]
    }
    
    # Assign age group to each age
    un_population.loc[:,"group"] = un_population["Age"].map(
        lambda x: next((grp for grp, ages in age_groups.items() if x in ages), None)
    )
    
    #Aggregate population per group
    population_groups = (
        un_population
        .dropna(subset=["group"]) # Drop rows with ages that don"t fit into defined groups ("All")
        .groupby(["ISO3", "Year", "group"])["Population"]
        .sum()
    )
    
    # Calculate total population per country and year
    population_total = (
        population_groups
        .reset_index()
        .groupby(["ISO3","Year"])
        ["Population"]
        .sum()
    )

    # Calculate share of young, older and oldest population
    population_shares = []
    
    for group in ["young", "older", "oldest"]:
        share = population_groups.xs(group, level=2) / population_total
        population_shares.append(share.rename(f"{group}"))
    
    # Combine shares into a single DataFrame
    population_total = (
        pd.concat(population_shares, axis=1)
        .reset_index()
        .melt(
        id_vars=["ISO3", "Year"],
        value_vars=["young", "older", "oldest"],
        var_name="group")
        .rename(columns={"value":"share"})
    )
    
    return population_total



def PopulationIMAGE(wdir):
     
    """
    The code imports the population data produced by an IMAGE run and converts it to population data
    for the three age groups and SSP scenarios at the impact region level.

    The function saves CSV files with population projections per age group and SSP scenario at the 
    impact region level in the subdirectory `data/population/pop_ssp` of the working directory.
    If no folder exists, it will be created.
    """
    
    SSP = ["SSP1", "SSP2", "SSP3", "SSP5"]
    YEARS = range(1980, 2101)
    
    for ssp in SSP:
        
        # Agregate raster IMAGE total population per impact region and year
        total_population_ir = IMAGEPopulation2ImpactRegion(
            wdir=wdir, 
            ssp=ssp, 
            years=YEARS
            )
        
        # Load population data projections per age group to disagregate IMAGE data
        population_groups = LoadAgeGroupPopulationData(
            wdir=wdir,
            ssp=ssp,
            years=YEARS
            )
        
        # Pivot and merge function
        def pivot_and_merge(group_name):
            
            df = (
                population_groups[population_groups["group"] == group_name]
                .pivot(index="ISO3", columns="Year", values="share")
                .reset_index()
            )
            df = df.rename(columns={c: f"{c}_share" for c in df.columns if isinstance(c, int)})
            return total_population_ir.merge(df, on="ISO3", how="left")
        
        # Create population dataframes for each age group
        pop_young, pop_older, pop_oldest = (pivot_and_merge(g) for g in ["young", "older", "oldest"])
        
        # Create output directory if it doesn"t exist
        out_path = Path(wdir) / "data" / "Population" / "Population_IMAGE"
        out_path.mkdir(parents=True, exist_ok=True)

        # Multiply shares by total population to get absolute numbers
        for pop, group in zip([pop_young, pop_older, pop_oldest],["young", "older", "oldest"]):        
            for y in YEARS:
                pop[str(y)] = pop[str(y)+"_share"] * pop[str(y)]
                        
            non_share_cols = [c for c in pop.columns if "share" not in c]
            
            # Save population projection files per age group and SSP scenario
            pop[non_share_cols].to_csv(out_path/ f"pop_{ssp.lower()}_{group}.csv", float_format='%.2f', index=False)

        
        
def IMAGEPopulation2ImpactRegion(wdir, ssp, years):
    
    """
    Calculate total population per impact region for a specific year from IMAGE land
    population data files.

    This function assigns raster pixels from the IMAGE land population data to impact regions,
    sums the population values within each region, and adds the results as a new column to
    the input GeoDataFrame.
    """
    
    print(f"Calculating population per impact region for SSP: {ssp}...")
    
    # Read in impact regions shapefile
    impact_regions = gpd.read_file(wdir+"/data/CarletonSM/ir_shp/impact-region.shp")
    
    # Read IMAGE SSP population nc file
    pop_image = xr.open_dataset(
        os.path.dirname(wdir) +
        f"/data/IMAGE/IMAGE_Population/{ssp.upper()}/GPOP.nc"
        )
    
    # Ensure CRS is set to EPSG:4326 and align with impact regions
    pop_image = pop_image.rio.write_crs("EPSG:4326", inplace=False)
    impact_regions = impact_regions.to_crs(pop_image.rio.crs)

    # Select relevant years
    pop_image = pop_image.sel(time=pd.to_datetime([f"{y}-01-01" for y in years]))
    
    # Set minlength for np.bincount to ensure it can accommodate all region IDs + 0 (no region)
    minlength = len(impact_regions) + 1

    # Prepare tuples of (geometry, region_id) for rasterization
    shapes_and_ids = [(geom, idx) for idx, geom in enumerate(impact_regions.geometry, start=1)]

    # Rasterize region polygons to create a pixel-to-region mapping
    pixel_owner = rasterize(
        shapes_and_ids,
        out_shape=pop_image.isel(time=0).GPOP.shape, # Rasterize region polygons once
        transform=pop_image.rio.transform(), # Get raster transform 
        fill=0, # 0 = without region
        all_touched=True, # Consider pixels touched by the geometry as part of the region
        dtype="int32"
    )
    
    # Initialize dictionary to store population sums per year
    year_data = {}
    
    # Iterate over years and calculate population sums per region
    for i, year in enumerate(years):
        
        # Read raster data for the current year
        raster_data = pop_image.isel(time=i).GPOP.values
        
        # Mask valid data (NaN = nodata)
        valid_pop_mask = ~np.isnan(raster_data)

        # Sum population per region using np.bincount in pixels without NaN
        sums = np.bincount(
            pixel_owner[valid_pop_mask], # Region IDs for valid pixels
            weights=raster_data[valid_pop_mask], # Population values for valid pixels
            minlength=minlength # Ensure it can accommodate all region IDs + 0 (no region)
        )[1:]  # Remove pixel values that do not belong to any region (0)

        # Add results to impact_regions GeoDataFrame
        year_data[str(year)] = sums
    
    # Concatenate population sums for all years into the impact_regions DataFrame
    impact_regions = pd.concat(
        [impact_regions, pd.DataFrame(year_data, index=impact_regions.index)],
        axis=1
    )
    
    # Add ISO3 column
    impact_regions["ISO3"] = impact_regions["hierid"].str[:3]

    # Only return regions names and population columns
    return impact_regions[
        ["hierid", "ISO3"] + 
        [c for c in impact_regions.columns if str(c).isdigit()]
        ]



def LoadAgeGroupPopulationData(wdir, ssp, years):
    
    """
    Load and process population data projections per age group from the SSP projections.
    This function reads the population projections from the SSP data, classifies ages into three groups
    (young, older, oldest), aggregates population by these groups, and calculates the share of each age group
    for each country and year. The resulting DataFrame includes the population share of each age group
    for each country and year, along with the corresponding ISO3 country codes.
    """
    
    # Historical population shares ---------------------------------------------
    
    # Load UN population data file and filter for relevant years
    un_population = ProcessUNPopulation5years(wdir)
    un_population = un_population[un_population["Year"].isin(years)]
    
    
    # Projected population shares -----------------------------------------------
    
    
    # Load population data projections per 5-year age group (latest SSP projections)
    ssp_population = (
        pd.read_csv(
            os.path.dirname(wdir) +
            "/DATA/WCDE_POP_SSP/wcde_data.csv", 
            skiprows=8
            ).query("Scenario == @ssp and Year in @years")
    )
    
    # Fill in missing years in the population data by forward-filling and backward-filling
    ssp_population = CompletePopulationDataLustrum(ssp_population)
    
    # Define age groups to classify wcde ages
    age_groups = {
        "young": ["0--4"],
        "older": [f"{i}--{i+4}" for i in range(5, 65, 5)],
        "oldest": [f"{i}--{i+4}" for i in range(65, 100, 5)] + ["100+"]
    }
    
    # Assign age group to each age and set All ages to None
    ssp_population.loc[:,"group"] = ssp_population["Age"].map(
        lambda x: next((grp for grp, ages in age_groups.items() if x in ages), None)
    )
    
    # Aggregate population by group
    population_groups = (
        ssp_population
        .dropna(subset=["group"]) # Drop rows with ages that don"t fit into defined groups ("All")
        .groupby(["Area", "Year", "group"], as_index=False)["Population"]
        .sum()
    )
    
    # Generate rows with missing years per area and group
    dfs = []

    # Interpolate for every Area and group combination to get yearly population values
    for (area, group), group_df in population_groups.groupby(["Area", "group"]):

        # Reindex to include all years
        group_df = group_df.set_index("Year").reindex(range(2000,2101))
        # Extend area and group columns
        group_df["Area"] = area
        group_df["group"] = group
        # Interpolate population values
        group_df["Population"] = group_df["Population"].interpolate(method="linear")
        # Reset index
        group_df = group_df.reset_index().rename(columns={"index": "Year"})
        dfs.append(group_df)

    # Concateenate all dataframes
    population_groups =  pd.concat(dfs, ignore_index=True)
    
    # Calculate share of each age group
    population_groups["Population_total"] = (
        population_groups
        .groupby(["Area", "Year"])["Population"]
        .transform("sum")
    )
    population_groups["value"] = (
        population_groups["Population"] / 
        population_groups["Population_total"]
    )
    
    # Convert locations to ISO3
    unique_locations = population_groups["Area"].unique()
    conversion_dic = {loc: coco.convert(names=loc, to="ISO3") for loc in unique_locations}
    population_groups["ISO3"] = population_groups["Area"].map(conversion_dic)
    
    # Merge historical population and SSPs projections
    population = (
        pd.concat(
            [un_population,
            population_groups[
                population_groups["Year"]
                .isin(range(2026,years[-1]+1))
                ]
            .rename(columns={"value":"share"})
            [["ISO3", "Year", "group", "share"]]],
            axis=0
            )
        .set_index(["ISO3", "Year", "group"]) # Set multi-index for sorting and filtering
        .sort_index()
        .drop('not found', level=0) # Remove "World" from SSPs
        .reset_index()
    )
    
    return population



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



def DailyTemperaturesERA5PresentDay(wdir, era5_dir, years):
    
    sets = mf.ModelSettings(
        temp_dir=era5_dir,
        gdp_dir=None,
        wdir=wdir,
        project=None,
        scenario="ERA5_SSP2",
        years=years,
        adaptation=False,
        counterfactual=False,
        reporting_tool=False,
        draw="mean",
        emulator=False
    )
    
    # Get spatial relationship between ERA5 grid and impact regions
    spatial_relation, _ = mf.GridRelationship(sets)
    
    # Create directory if it doesn't exist
    out_path = Path(wdir) / "data" / "ClimateData" / "BaselineTemperatures"
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate T_0 for each year and save as CSV
    for year in years:
        
        print(f"Calculating T_0 for year: {year}...")
        
        t_0 = mf.ERA5Temperature2IR(era5_dir, year, spatial_relation)
        
        # Convert to xarray and save as netCDF
        t0_xarray = (
            t_0
            .stack()
            .to_xarray()
            .rename(level_1="date")
        )
        t0_xarray.name = "tmean0"
        
        # Save and compress file
        t0_xarray.to_netcdf(
                f"{out_path}/ERA5_tmean0_{year}.nc",
                encoding={
                    t0_xarray.name:{
                        "dtype": "float32",
                        'zlib': True,
                        'complevel': 6
                        }
                    }
            )
        
        

def ClimatologiesERA5(wdir, era5_dir, years):
    
    # Get spatial relationship between ERA5 grid and impact regions
    class Settings:
        def __init__(self, wdir, scenario, era5_dir, years, emulator=False):
            self.wdir = wdir
            self.scenario = scenario
            self.temp_dir = era5_dir
            self.years = years
            self.emulator = emulator
            
    sets = Settings(wdir=wdir, scenario="ERA5", era5_dir=era5_dir, years=years, emulator=False)

    spatial_relation, ir = mf.GridRelationship(sets)
    
    # Define period for climatology calculation (30-year running mean)
    period = 30

    # Calculate annual mean temperature for each year used in the analysis using ERA5 daily data
    annual_temperatures = {}
    
    for y in range(years[0]-period, years[-1]):
        print("Calculating annual mean temperature for year:", y)

        with xr.open_dataset(f"{era5_dir}/era5_t2m_mean_day_{y}.nc") as ds:
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
    
    out_path = Path(wdir) / "data" / "ClimateData" / "Climatologies"
    out_path.mkdir(parents=True, exist_ok=True)
    
    climatologies_df.to_csv(out_path+f"/ERA5_climtas_{years[0]}-{years[-1]}.csv",
                            index=False)
