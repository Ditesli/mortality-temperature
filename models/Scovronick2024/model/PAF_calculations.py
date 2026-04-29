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
from utils_common import paf2mortality as p2m



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
        
        print("[1.4] Loading region classification dictionaries...")
        region_dict, image_dict = p2m.LoadRegionClassificationDicts(sets.wdir)
            
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
    paf = paf.sub(paf[list(range(2001,2011))].mean(axis=1), axis=0)
    
    # Create class to store output path and file name format
    class ScenarioNaming:
        def __init__(self, sets):
            self.years_part = f"_{sets.years[0]}-{sets.years[-1]}"
            self.out_path = Path(sets.wdir) / "output" / f"{sets.project}" 
            
    sn = ScenarioNaming(sets)
    
    # Create project folder if it doesn't exist
    sn.out_path.mkdir(parents=True, exist_ok=True)
    
    # Save PAF results as csv file
    paf.to_csv(
        sn.out_path /
        f"PAF_{sets.project}_{sets.scenario}_ISO3{sn.years_part}.csv"
        ) 
    
    print("[3.1] Calculating attributable mortality and saving results...")
    
    PAF2Mortality(sets, fls, paf, sn)
    
    print("Model ran succesfully!")
    
    
    
def PAF2Mortality(sets, fls, paf, sn):

    # Set causes of death to grab from GBD data
    gbd_causes = [
        "All causes", 
        "Cardiovascular diseases", 
        "Chronic respiratory diseases", 
        "Respiratory infections and tuberculosis"
        ]

    # Import GBD mortality data and population and convert paf to xarrays
    gbd_mor = p2m.LoadGBDmortality(sets, fls, gbd_causes, "Scovronick")
    paf = ReformatPAF(fls)
    pop = p2m.LoadUNpopulationData(sets, "Scovronick")
    
    # Merge the three xarrays to have all data in the same format and coordinates
    paf_mor_pop = xr.merge([pop, gbd_mor, paf], join="outer") 
    
    # Create "oldest" age group for model comparison
    paf_mor_pop = p2m.AggCoordElementsXarray(paf_mor_pop, "age_group", ["65", "85"], "oldest", exclude=False)

    # Calculate total mortality and relative mortality
    paf_mor_pop["mor"] = paf_mor_pop['paf'] * paf_mor_pop['val']
    paf_mor_pop["rel_mor"] = paf_mor_pop["mor"] * 1e5 / paf_mor_pop["pop"]

    # Convert xarray to dataframe to save as csv files
    p2m.ProcessXarray2csv(sets, paf_mor_pop, "Scovronick", "ISO3", sn)

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
    p2m.ProcessXarray2csv(sets, mor_image, "Scovronick", "IMAGE", sn)



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
    
    paf_65 = paf.sel(age_group="70").assign_coords(age_group="65")
    paf = xr.concat([paf, paf_65], dim="age_group")

    paf = paf.sortby("age_group").sortby("ISO3").sortby("cause")
    
    return paf