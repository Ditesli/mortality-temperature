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
            
    #     self.postprocess()
        
    # def postprocess(self):
    #     PostprocessResults(sets=self.sets, fls=self.fls)
            


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
    pop_region: pd.DataFrame
    erf: pd.DataFrame
    tmin: np.ndarray
    pmap: xr.Dataset
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
        
        print(f"[1.2] Calculating population per impact region for SSP: {ssp}...")
        pop_region = pop.IMAGEPopulation2Regions(
            shp_dir=sets.wdir+"/data/GBD_locations/gbd_shapefiles/", 
            shp_name="GBD_shapefile",
            pop_dir=os.path.dirname(sets.wdir)+"/data", 
            ssp="SSP2",
            years=sets.years)   
        
        print(f"[1.3] Loading region classification...")
        regions, regions_range = pop.LoadRegionClassificationMap(
            wdir=sets.wdir,
            temp_dir=sets.temp_dir, 
            region_class="countries",
            scenario=sets.scenario,
            pop_map=pop_map
            )
        
        # Load Exposure Response Functions (ERF; 1 draw or mean) and set dicts of min and max temp.
        erf, tmin = LoadExposureResponseFunctions(sets)
        
        # Load percentiles map
        percentiles_map = LoadPercentilesMap(sets, range(1980,2011))
            
        print("[1.6] Creating final dataframe to store results...")
        paf = pd.DataFrame(
            index=pd.MultiIndex.from_product([
                regions_range, 
                ["all", "cold", "heat"], 
                sets.causes,
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
            pop_region=pop_region,
            erf=erf,
            tmin=tmin,
            pmap=percentiles_map,
            paf=paf
        )
   


def LoadExposureResponseFunctions(sets):
    
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
    
    return erf, tmin



def LoadPercentilesMap(sets, years):
    
    percentiles_map = xr.open_dataset(
        sets.wdir + 
        f"/data/Percentiles_Maps/ERA5_Tmean_Percentiles_{years[0]}-{years[-1]}.nc"
        )
    
    return percentiles_map



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
        CalculateRegionalPAF(sets, fls, pop_year, region, year, daily_temp)
        
        
    
def CalculateRegionalPAF(sets, fls, pop_year, region, year, daily_temp):
    
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
    paf_cold = np.average(rr_cold, axis=2, weights=pop_region)
    paf_hot = np.average(rr_hot, axis=2, weights=pop_region)
    paf_all = paf_hot + paf_cold
    
    # Append to final dataframe
    fls.paf.loc[(region, "cold"), year] = paf_cold.flatten()
    fls.paf.loc[(region, "heat"), year] = paf_hot.flatten()
    fls.paf.loc[(region, "all"), year] = paf_all.flatten()

    print(region_mask)

