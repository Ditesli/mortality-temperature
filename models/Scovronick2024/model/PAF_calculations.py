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
        "cvd":"Cardiovascular diseases", 
        "rsp":"Respiratory diseases", 
        "ncrc":"Non-cardiorespiratory diseases", 
        "all":"All causes", 
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
    paf: pd.DataFrame


    @classmethod
    def from_files(cls, sets):
        
        """
        Load all input files required for PAF calculations. 
        Data is located in the wdir/data folder.  
        """
        
        print("[1] Loading input files...")
        
        print("[1.2] Loading SSP population data...")
        ssp = re.search(r"SSP\d", sets.scenario).group()
        pop_map = pop.LoadPopulationMap(
            wdir=sets.wdir,
            scenario=sets.scenario, 
            ssp=ssp, 
            years=sets.years
            )
        
        print(f"[1.3] Calculating population per impact region for SSP: {ssp}...")
        pop_region = pop.IMAGEPopulation2Regions(
            shp_dir=sets.wdir+"/data/GBD_locations/gbd_shapefiles/", 
            shp_name="GBD_shapefile",
            pop_dir=os.path.dirname(sets.wdir)+"/data", 
            ssp="SSP2",
            years=sets.years)   
        
        print(f"[1.4] Loading region classification...")
        regions, regions_range = pop.LoadRegionClassificationMap(
            wdir=sets.wdir,
            temp_dir=sets.temp_dir, 
            region_class="countries",
            scenario=sets.scenario,
            pop_map=pop_map
            )
        
        # Load Exposure Response Functions (ERF; 1 draw or mean) and set dicts of min and max temp.
        erf = LoadExposureResponseFunctions(sets)
            
        print("[1.8] Creating final dataframe to store results...")
        paf = pd.DataFrame(
            index=regions_range, 
            columns=pd.MultiIndex.from_product([sets.years, sets.causes, ["cold", "heat", "all"]])
            )  
        
        
        return cls(
            regions=regions,
            regions_range=regions_range,
            pop_map=pop_map,
            pop_region=pop_region,
            erf=erf,
            paf=paf
        )
   


def LoadExposureResponseFunctions(sets):
    
    erf = pyreadr.read_r(sets.wdir + "/data/Fig2_20Nov2025.Rdata")
    
    return erf



def CalculatePAFYear(sets, fls, year):
    
    print(f"[2.1] Loading daily temperature data for year {year}...") 
    daily_temp, num_days = tmp.LoadDailyTemperatures(temp_dir=sets.temp_dir,
                                                     scenario=sets.scenario,
                                                     temp_type="mean",
                                                     year=year, 
                                                     pop_map=fls.pop_map,
                                                     std_factor=1)

    # Select population for the corresponding year
    pop_year = fls.pop_map.sel(time=f"{year}").mean("time").GPOP.values

    print(f"[2.2] Calculating Population Attributable Fractions for year {year}...")
    for region in fls.regions_range:
        CalculateRegionalPAF(sets, fls, pop_year, region, year, num_days, daily_temp)
        
        
        
    
def CalculateRegionalPAF(sets, fls, pop_year, region, year, num_days, daily_temp):
    
    """
    Get the Population Atributable Fraction per region and year by:
    1. Creating a dataframe with population weighted factors per temperature zones and daily temperatures
    2. Merging the dataframe with the ERF shifted by the TMREL to assign RR values
    3. Separating the dataframe into cold and heat attributable deaths
    4. Calculating the PAF per temperature type and storing it in the final dataframe
    """
    
    # Get population mask within selected region
    region_mask = (pop_year > 0.) & (fls.regions == region)
