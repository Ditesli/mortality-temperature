import pandas as pd
import numpy as np
import scipy as sp
import xarray as xr
import pyreadr
import re, sys, os, random
from pathlib import Path
from dataclasses import dataclass, field
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils_common import temperature as tmp
from utils_common import population as pop



def CalculatePAF(
    wdir: str,
    temp_dir: str,
    project: str,
    scenario: str,
    years: list,
    ):

    sets = ModelSettings(
        wdir=wdir,
        temp_dir=temp_dir,
        project=project,
        scenario=scenario,
        years=years
    )

    model = PAFModel(sets=sets)

    model.run()
    
    
    
@dataclass
class ModelSettings:
    wdir: str
    temp_dir: str
    project: str
    scenario: str
    years: list
    causes: dict = field(
        default_factory=lambda: {
        "all":"All causes",
        "cvd":"Cardiovascular diseases", 
        "ncrc":"Non-cardiorespiratory diseases",
        "rsp":"Respiratory diseases"         
    })
    
    def __post_init__(self):
        
        # Include last year 
        if isinstance(self.years, range):
            self.years = range(self.years.start, self.years.stop + 1)
        
        # Reduce range years if working with ERA5 data
        ERA5_START_YEAR = 2000
        ERA5_END_YEAR = 2025

        if re.search(r"ERA5", self.scenario):
            self.years = [
                y for y in self.years
                if ERA5_START_YEAR <= y <= ERA5_END_YEAR
            ]
            
            
            
@dataclass
class PAFModel:
    sets: ModelSettings
    
    """
    The model calculates the Population Attributable Factor from mortality to heat and cold
    for respiratory, cardiovascular, non-cardiorespiratory diseases and all causes of death, 
    according to Scovronick et al., (2024). The model uses the Exposure Response Functions 
    from the paper as provided by the authors.
    The model:
    1. Validates the input years depending on the scenario.
    2. Reads in the needed files to perform all calculations in load_inputs. All needed data
    should be located in the folder wdir/data.
    3. Calculates the PAF for the years range given by assigning first the Relative Risk 
    corresponding to the selected TMRELs, temperature zone, and country at a grid cell level 
    paired with the fraction of the population in that grid cell and merge this data to get 
    the share of people within a country exposed to a relative risk.
    4. If data on cause-specific mortality is available the model will also calculate total
    attributable mortality and (relative attributable mortality per region).
    
    Parameters:
    sets : ModelSettings
        Paths to main working directory, climate data and income data. This folder must contain 
        two folders: data (used for calculations) and output (where results are stored).
    years : list
        Provide the range of years the model will run.
    project : str
        Name of the project, used to locate the data in the right folder. It can be any of the projects
        included in wdir/data/IMAGE_land/scen/ AND/OR to use in the output file name.
    scenario : str
        - SSP#_ERA5:
        This scenario uses ERA5 temperature data records and historical mortality records retreived
        from the GBD database. Scenario runs from 2000 to 2025.
        - IMAGE scenarios:
        Future projections are not available as cause-specific mortality projections have not been
        calculated.
        
    Returns:
    ----------
    None
    Saves the mortality results to CSV files in the output folder.
    """
    
    def load_inputs(self):
        self.fls = LoadInputData.from_files(sets=self.sets)
        
    def run(self):
        print(f"Running model for project {self.sets.project} and scenario {self.sets.scenario}...")
        self.load_inputs()
        
        print("[2] Starting PAF calculations...")
        
        for year in self.sets.years:
            CalculatePAFYear(
                sets=self.sets,
                fls=self.fls,
                year=year
                )
            
        self.postprocess()
        
    def postprocess(self):
        PostprocessResults(sets=self.sets, fls=self.fls)
            


@dataclass
class LoadInputData:
    
    """
    Container for all input data required to run the model.

    Attributes
    ----------
    regions: np.ndarray
        2D array with region locations per grid cell aligned with daily temperature resolution.
    regions_range: np.ndarray
        1D array with location indices.
    pop_map: np.ndarray
        2D array with population per grid cell aligned with daily temperature resolution.
    erf: pd.DataFrame
        Dataframe with Exposure Response Functions from the original paper.
    paf: pd.DataFrame
        Dataframe to store Population Attributable Fraction results.
    """
    
    regions: np.ndarray
    regions_range: np.ndarray
    pop_map: np.ndarray
    erf: pd.DataFrame
    tmin: np.ndarray
    pmap: xr.Dataset
    region_dict: dict
    image_dict: dict
    paf: pd.DataFrame


    @classmethod
    def from_files(cls, sets):
        
        """
        Load all input files required for PAF calculations. 
        Data is located in the wdir/data folder.  
        """
        
        print("[1] Loading input files...")
        
        print("[1.1] Loading SSP population data...")
        ssp = re.search(r"SSP\d", sets.scenario).group()
        pop_map = pop.LoadPopulationMap(
            wdir=sets.wdir,
            scenario=sets.scenario, 
            ssp=ssp, 
            years=sets.years
            ) 
        
        print(f"[1.2] Loading region classification...")
        regions, regions_range = pop.LoadRegionClassificationMap(
            wdir=sets.wdir,
            temp_dir=sets.temp_dir, 
            region_class="countries",
            scenario=sets.scenario,
            pop_map=pop_map
            )
        
        # Load Exposure Response Functions (ERF; 1 draw or mean) and set dicts of min and max temp.
        erf, tmin, pmap = LoadExposureResponseFunctionsAndPercentiles(sets, range(1980,2011))
        
        region_dict, image_dict = LoadRegionClassificationDicts(sets.wdir)
            
        print("[1.5] Creating final dataframe to store results...")
        paf = pd.DataFrame(
            index=pd.MultiIndex.from_product([
                regions_range, 
                ["all", "cold", "heat"], 
                sets.causes.values(),
                ["40_rr", "40_rr_low", "40_rr_high",
                 "55_rr", "55_rr_low", "55_rr_high",
                 "70_rr", "70_rr_low", "70_rr_high",
                 "85_rr", "85_rr_low", "85_rr_high"]
                ]),
            columns=sets.years
            )  
        
        
        return cls(
            regions=regions,
            regions_range=regions_range,
            pop_map=pop_map,
            erf=erf,
            tmin=tmin,
            pmap=pmap,
            region_dict=region_dict,
            image_dict=image_dict,
            paf=paf
        )
   


def LoadExposureResponseFunctionsAndPercentiles(sets, years):
    
    """
    Load ERF as provided by the authors and the percentiles map previously calculated
    in the preprocessing part.

    Returns:
    - erf: 
        3D array with the ERF values for each cause, age group and high, medium and low 
        estimation of the curves.
    - tmin: 
        Array with the indices of the minimum ERF value for each cause and region, 
        used to separate between hot and cold temperatures.
    - percentiles_map: 
        xarray dataset with the percentiles map for the selected years range.
    """
    
    # Read in ERF functions from Scovromick et al., (2024)
    erf = pyreadr.read_r(
        sets.wdir +
        "/data/Scovronick_SM/Fig2_20Nov2025.Rdata"
        )
    
    # Convert to numpy array for optimized calculations
    erf = np.stack(
        [erf[cause].values[:,1:] for cause in sets.causes.keys()], 
        axis=0
        ).swapaxes(1,2)
    
    # Use the range between 25th and 99th percentile to fin the minimum
    erf_range = erf[:, :, 34:109]
    # Get the index with the minimum value to separate hot and cold temperatures
    tmin = np.argmin(erf_range, axis=2) + 34
    
    # Load percentiles map previously calculatedin preprocessing
    percentiles_map = xr.open_dataset(
        sets.wdir + 
        f"/data/Percentiles_Maps/ERA5_Tmean_Percentiles_{years[0]}-{years[-1]}.nc"
        )
    
    return erf, tmin, percentiles_map



def LoadRegionClassificationDicts(wdir):
    
    print("[1.4] Loading region classification dictionaries...")
    
    # Move one level up to access all-model data and region classification file
    wdir_up = os.path.dirname(wdir)
    
    # Create dictionaries to map location ids to ISO3 codes
    region_names = (
        pd.read_csv(f"{wdir_up}/data/region_classification.csv")
        [["gbd_location_id", "ISO3"]]
        .drop_duplicates()
        .dropna()
    )
    
    region_dict = dict(zip(
        region_names["gbd_location_id"].astype(int), 
        region_names["ISO3"]))
    
    # Dictionary to map location ids to IMAGE region names
    region_names = (
        pd.read_csv(f"{wdir_up}/data/region_classification.csv")
        [["IMAGE26", "ISO3"]]
        .drop_duplicates()
        .dropna()
    )
    
    image_dict = dict(zip(  
        region_names["ISO3"],
        region_names["IMAGE26"]))
    
    return region_dict, image_dict



def CalculatePAFYear(sets, fls, year):
    
    print(f"[2.1] Loading daily temperature data for year {year}...") 
    daily_temp, _ = tmp.LoadDailyTemperatures(temp_dir=sets.temp_dir,
                                                     scenario=sets.scenario,
                                                     temp_type="mean",
                                                     year=year, 
                                                     pop_map=fls.pop_map,
                                                     std_factor=1)

    # Select population for the corresponding year
    pop_year = fls.pop_map.sel(time=f"{year}").mean("time").GPOP.values

    print(f"[2.2] Calculating Population Attributable Fractions for year {year}...")
    
    for region in fls.regions_range:
        CalculateRegionalPAF(fls, pop_year, region, year, daily_temp)
        
        
    
def CalculateRegionalPAF(fls, pop_year, region, year, daily_temp):
    
    """
    Get the Population Atributable Fraction per region and year by:
    1. Getting the corresponding indices of the temperature percentiles of every grid cell 
    and every day using the percentiles_map previously calculated in the preprocessing part.
    2. Using the indices to get the corresponding Relative Risk from the ERF functions 
    provided by the authors.
    3. Separating the dataframe into cold and heat RR.
    4. Calculating the PAF per temperature type and storing it in the final dataframe.
    """
    
    # Get region mask 
    region_mask = (pop_year > 0.) & (fls.regions == region)
    # Get the population per region by masking the selected region map
    pop_region = fls.pop_map.mean(dim="time").GPOP.values[region_mask]
    
    # Mask the percentiles map and daily temperature data to the selected region
    pmap = fls.pmap.t2m.values[:, region_mask]
    temp = daily_temp[region_mask]
    
    # Map the corresponding indices of the pmap to the temperature data to assign the corresponding percentile    
    p_indices = np.argmin(
        np.abs(pmap[:, :, np.newaxis] - temp[np.newaxis, :, :]), 
        axis=0
    )
    
    # Get the corresponding relative risks from the indices
    rr = fls.erf[:,:,p_indices] - 1 # Substract 1 following Zhao et al., (2021)
    
    # Mask to separate between rr from heat and rr from cold temperatures
    mask_cold = p_indices[np.newaxis, np.newaxis, :, :] < fls.tmin[:, :, np.newaxis, np.newaxis]
    
    # Calculate the average RR for heat and cold temperatures by averaging across all days of the year (axis=3)
    sum_cold = np.sum(np.where(mask_cold, rr, 0), axis=3)
    sum_hot = np.sum(np.where(~mask_cold, rr, 0), axis=3)
    
    count_cold = np.sum(mask_cold.astype(int), axis=3)
    count_hot = np.sum((~mask_cold).astype(int), axis=3)
    
    rr_cold = np.divide(sum_cold, count_cold, out=np.zeros_like(sum_cold), where=count_cold!=0)
    rr_hot = np.divide(sum_hot, count_hot, out=np.zeros_like(sum_hot), where=count_hot!=0)
    
    # Get PAF per region by calculating the population-weighted average of the RR across all cells in the region
    if pop_region.size > 0 and np.sum(pop_region) != 0:

        paf_cold = np.average(rr_cold, axis=2, weights=pop_region)
        paf_hot = np.average(rr_hot, axis=2, weights=pop_region)
        paf_all = paf_hot + paf_cold
        
    else:
        paf_cold = np.full(fls.tmin.shape, np.nan)
        paf_hot = np.full(fls.tmin.shape, np.nan)
        paf_all = np.full(fls.tmin.shape, np.nan)
    
    # Append to final dataframe
    fls.paf.loc[(region, "cold"), year] = paf_cold.flatten()
    fls.paf.loc[(region, "heat"), year] = paf_hot.flatten()
    fls.paf.loc[(region, "all"), year] = paf_all.flatten()
    
    
    
def PostprocessResults(sets, fls):
        
    print("[3] Model run complete. Saving results...")
    
    paf = fls.paf
    paf.index.names=["region", "t_type", "cause", "age_group"]
    
    # Substract counterfactual mortality
    # paf = paf.sub(paf[list(range(2001,2011))].mean(axis=1), axis=0)
    
    # Create project folder if it doesn"t exist
    out_path = Path(sets.wdir) / "output" / f"{sets.project}" 
    out_path.mkdir(parents=True, exist_ok=True)
    
    years_part = f"_{sets.years[0]}-{sets.years[-1]}"
    
    # paf.to_csv(
    #     out_path /
    #     f"PAF_{sets.project}_{sets.scenario}_ISO3{years_part}.csv"
    #     ) 
    
    print("[3.1] Calculating attributable mortality and saving results...")
    
    PAF2Mortality(sets, fls, paf, out_path, years_part, ages="oldest")
    PAF2Mortality(sets, fls, paf, out_path, years_part, ages="All")
    
    print("Model ran succesfully!")
    
    
    
def PAF2Mortality(sets, fls, paf, out_path, years_part, ages):
    
    # Set age part of the file name depending on the age groups included in the analysis
    age_part = "" if ages == "All" else "_oldest" if ages == "oldest" else age_part

    # Set causes of death to grab from GBD data
    gbd_causes = [
        "All causes", 
        "Cardiovascular diseases", 
        "Chronic respiratory diseases", 
        "Respiratory infections and tuberculosis"
        ]

    # Import GBD mortality data and population and convert paf to xarrays
    gbd_mor = LoadGBDmortality(sets, fls, gbd_causes, "Scovronick", ages)
    paf = ReformatPAF(fls)
    pop = LoadUNpopulationData(sets, ages)
    
    # Merge the three xarrays to have all data in the same format and coordinates
    paf_mor_pop = xr.merge([pop, gbd_mor, paf], join="outer") 
    
    if ages == "oldest":
        paf_mor_pop = AggCoordElementsXarray(paf_mor_pop, "age_group", ["70", "85"], "oldest", exclude=False)

    # Calculate total mortality and relative mortality
    paf_mor_pop["mor"] = paf_mor_pop['paf'] * paf_mor_pop['val']
    paf_mor_pop["rel_mor"] = paf_mor_pop["mor"] * 1e5 / paf_mor_pop["pop"]

    # Convert xarray to dataframe to save as csv files
    ProcessXarray2csv(sets, paf_mor_pop, "ISO3", out_path, years_part, age_part)

    # Map location ids to IMAGE region names
    paf_mor_pop['ISO3'] = xr.DataArray(
        [fls.image_dict[id] for id in paf_mor_pop['ISO3'].values], 
        coords=paf_mor_pop['ISO3'].coords, 
        dims=paf_mor_pop['ISO3'].dims
        )

    # Aggregate mortality and population data by IMAGE region
    mor_image = paf_mor_pop.groupby("ISO3").sum().drop_vars(["paf", "rel_mor"])

    # Calculate global mortality and population
    mor_image = xr.concat([
        mor_image,
        mor_image.sum(dim='ISO3').assign_coords(ISO3="World")],
        dim='ISO3')

    # Calcualte relative mortality and PAF for IMAGE regions
    mor_image["rel_mor"] = mor_image["mor"] * 1e5 / mor_image["pop"]
    mor_image["paf"] = mor_image["mor"] / mor_image["val"]

    # Convert xarray to dataframe to save as csv files
    ProcessXarray2csv(sets, mor_image, "IMAGE", out_path, years_part, age_part)
    
    

def LoadGBDmortality(sets, fls, causes, model, ages):
    
    """
    Load GBD mortality data for the selected causes and age groups. 
    The data is filtered to include only the years included in the model run 
    and to exclude global mortality data. The data is then converted to
    an xarray for optimized calculations of relative and total mortality.
    """
    
    # Load GBD mortality records
    gbd_mor = pd.read_csv(f"{os.path.dirname(sets.wdir)}/data/GBD_mortality/IHME-GBD_2022_DATA.csv")

    mask = (
            gbd_mor["cause_name"].isin(causes) & # Only selected causes of death
            (gbd_mor["sex_name"] == "Both") & # Both sexes
            gbd_mor["year"].isin(sets.years) & # Only years assessed
            (gbd_mor["location_name"] != "Global") # Exclude global mortality
        )

    # Mask and convert to xarray
    gbd_mor = (
        gbd_mor.loc[mask, ["location_id","cause_name","age_name","year","val"]]
        .rename(columns={"age_name": "age_group", "location_id":"ISO3", "cause_name":"cause"})
        .set_index(["ISO3","cause","age_group","year"])
        .to_xarray()
    )

    if model == "Scovronick":
        
        # Merge respiratory causes of death into a single category
        rsp_causes = ["Chronic respiratory diseases", "Respiratory infections and tuberculosis"]
        rr_40_group = [f'{year}-{year+4} years' for year in range(30,45,5)]
        rr_55_group = [f'{year}-{year+4} years' for year in range(45,60,5)]
        rr_70_group = [f'{year}-{year+4} years' for year in range(65,75,5)] if ages == "oldest" else [f'{year}-{year+4} years' for year in range(60,75,5)]
        rr_85_group = [f"{year}-{year+4} years" for year in range(75,85,5)] + ["85+ years"]

        gbd_mor = AggCoordElementsXarray(gbd_mor, "cause", rsp_causes, "Respiratory diseases", exclude=True)
        gbd_mor = AggCoordElementsXarray(gbd_mor, "age_group", rr_40_group, "40", exclude=True)
        gbd_mor = AggCoordElementsXarray(gbd_mor, "age_group", rr_55_group, "55", exclude=True)
        gbd_mor = AggCoordElementsXarray(gbd_mor, "age_group", rr_70_group, "70", exclude=True)
        gbd_mor = AggCoordElementsXarray(gbd_mor, "age_group", rr_85_group, "85", exclude=True)
        
        # Remove other age groups that are not included in the analysis
        gbd_mor = gbd_mor.where(~gbd_mor.coords["age_group"].isin([c for c in gbd_mor.age_group.values if " " in c]), drop=True)

        # Calculate Non-cardiorespiratory causes
        gbd_ncrc = (
            gbd_mor.sel(cause="All causes") 
            - gbd_mor.sel(cause="Cardiovascular diseases")
            - gbd_mor.sel(cause="Respiratory diseases")
        ).assign_coords(cause="Non-cardiorespiratory diseases")

        gbd_mor = xr.concat([gbd_mor, gbd_ncrc], dim='cause')

    # Map location ids to ISO3 codes
    gbd_mor["ISO3"] = xr.DataArray(
        [fls.region_dict[id] for id in gbd_mor['ISO3'].values], 
        coords=gbd_mor['ISO3'].coords, 
        dims=gbd_mor['ISO3'].dims
    ).astype(object)
    
    # Sort by age group, ISO3 and cause
    gbd_mor = gbd_mor.sortby("age_group").sortby("ISO3").sortby("cause")
    
    return gbd_mor



def ReformatPAF(fls):
    
    """
    Convert the PAF dataframe to an xarray with the same coordinates 
    as the GBD mortality and UN population data, to be able to merge the 
    three datasets and calculate attributable mortality.
    """
    
    paf = fls.paf

    # Split the age group into age and certainty level (low, medium, high) 
    age_group_split = paf.index.get_level_values('age_group').str.split('_', expand=True)

    # Assign the age group and certainty level to separate columns in the dataframe
    paf["age"] = age_group_split.get_level_values(0)
    paf['certainty'] = np.where(
        age_group_split.get_level_values(2).isna(), 
        'medium', 
        age_group_split.get_level_values(2)
        )

    # Reformat dataframe to include age and certainty columns and convert to xarray
    paf = (
        paf
        .reset_index()
        .drop(columns=["age_group"])
        .rename(columns={"age":"age_group", "region":"ISO3"})
        .melt(id_vars=["ISO3", "t_type", "cause", 'age_group', 'certainty'], var_name='year', value_name='paf')
        .assign(paf=lambda df: df['paf'].astype(float)) 
        .set_index(["ISO3", "t_type", "cause", "age_group", "certainty", "year"])
        .to_xarray()
    )

    # Map location ids to ISO3 codes
    paf["ISO3"] = xr.DataArray(
        [fls.region_dict[id] for id in paf['ISO3'].values], 
        coords=paf['ISO3'].coords, 
        dims=paf['ISO3'].dims
        ).astype(object)

    paf = paf.sortby("age_group").sortby("ISO3").sortby("cause")
    
    return paf

    
    
def LoadUNpopulationData(sets, ages):
    
    """
    Load UN population data for the selected years range and age groups.
    The data is filtered to include only the years included in the model run. 
    The data is then converted to an xarray for optimized calculations of 
    relative and total mortality. The 5-year age groups are aggregated into the 
    same age groups as the GBD mortality data and the PAF data to be able to merge
    the three datasets and calculate attributable mortality.
    """

    # Load UN population data
    un_pop = (
        pd.read_csv(os.path.dirname(sets.wdir)+"/data/un_population/unpopulation_dataportal.csv")
        [["Iso3", "Time", "Age", "Value"]] # Keep relevant columns
        .rename(columns={"Value": "pop", "Time": "year", "Age": "age_group", "Iso3": "ISO3"})
        .set_index(["ISO3", "year", "age_group"]) 
        .to_xarray()
        .sel(year=slice(sets.years[0], sets.years[-1]))
    )

    # Aggregate 5-year age groups into the same age groups as the other xarrays
    rr_40_group = [f'{year}-{year+4}' for year in range(30,45,5)]
    rr_55_group = [f'{year}-{year+4}' for year in range(45,60,5)]
    rr_70_group = [f'{year}-{year+4}' for year in range(65,75,5)] if ages == "oldest" else [f'{year}-{year+4}' for year in range(60, 75, 5)]
    rr_85_group = [f"{year}-{year+4}" for year in range(75,100,5)] + ["100+"]

    un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=rr_85_group, new_elem="85", exclude=True)
    un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=rr_70_group, new_elem="70", exclude=True)
    un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=rr_55_group, new_elem="55", exclude=True)
    un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=rr_40_group, new_elem="40", exclude=True)

    # Drop age groups that are not included in the analysis (e.g. 0-4 years, 5-9 years, etc.)
    un_pop = un_pop.where(~un_pop.coords["age_group"].isin([c for c in un_pop.age_group.values if "-" in c]), drop=True)

    un_pop['pop'] = un_pop['pop'].where(un_pop['pop'] != 0)

    un_pop = un_pop.sortby("age_group").sortby("ISO3")

    return un_pop
    


def ProcessXarray2csv(sets, data_array, region_type, out_path, years_part, age_part):
    
    """
    Convert the xarray with mortality data to a dataframe and save it as a csv file. 
    The xarray is pivoted to have the years as columns and the other coordinates as rows.
    The unit of the mortality data is added as a column. The resulting dataframe 
    is saved as a csv file in the output folder.
    """
    
    def ProcessMortalityData(data_array, unit_name):
    
        df = (
            data_array.to_dataframe()
            .reset_index()
            .pivot_table(
                index=['ISO3', 't_type', 'cause', 'age_group'],
                columns='year', 
                values=data_array.name)  # Uses the array name as the value column
            .reset_index()
            )
        df['units'] = unit_name
        
        return df
    
    mor = ProcessMortalityData(data_array["mor"], 'Total Mortality')
    rel_mor = ProcessMortalityData(data_array["rel_mor"], 'Relative Mortality')
    
    # Concatenate the results and save
    mor_rel_mor = pd.concat([
        mor, rel_mor], axis=0)[
        ['ISO3', 't_type', 'cause', 'age_group', 'units'] 
        + list(mor.columns[4:-1])
    ].rename(columns={"ISO3": "region"})
        
    mor_rel_mor.to_csv(
            out_path /
            f"Mortality_{sets.project}_{sets.scenario}_{region_type}{years_part}{age_part}.csv"
            ) 
    
    
       
def AggCoordElementsXarray(array, coord, old_elems, new_elem, exclude):

    """
    Aggregate elements of a coordinate in an xarray by summing the 
    values across the specified dimension.
    """

    agg_xarray = (
        array
        .where(array[coord].isin(old_elems), drop=True)
        .sum(dim=coord)
        .assign_coords({coord: new_elem})
    )
    
    if exclude==True:
        xarray = array.where(~array[coord].isin(old_elems), drop=True)
    else:
        xarray = array
                             
    new_xarray = xr.concat(
        [xarray, 
        agg_xarray], 
        dim=coord
    )
    
    return new_xarray