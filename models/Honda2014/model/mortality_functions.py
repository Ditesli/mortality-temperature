import pandas as pd
import numpy as np
import scipy as sp
import xarray as xr
from dataclasses import dataclass
from pathlib import Path
import os, sys, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..")))
from utils import temperature as tmp
from utils import population as pop
from utils import paf2mortality as p2m



def CalculatePAF(
    wdir: str,
    temp_dir: str,
    project: str,
    scenario: str,
    years: list,
    optimal_range: str,
    extrap_erf: bool,
    temp_max: any
    ):

    sets = ModelSettings(
        wdir=wdir,
        temp_dir=temp_dir,
        project=project,
        scenario=scenario,
        years=years,
        optimal_range=optimal_range,
        extrap_erf=extrap_erf,
        temp_max=temp_max
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
    optimal_range: str
    extrap_erf: bool
    temp_max: any
    
    def __post_init__(self):
        
        # Include last year 
        if isinstance(self.years, range):
            self.years = range(self.years.start, self.years.stop + 1)
        
        # Reduce range years if working with ERA5 data
        ERA5_END_YEAR = 2025

        if re.search(r"ERA5", self.scenario):
            self.years = [
                y for y in self.years
                if y <= ERA5_END_YEAR
            ]
            
            
            
@dataclass
class PAFModel:
    sets: ModelSettings
    
    """
    Run the main model from Honda et al., 2014. The model calculates the 
    Population Attributable Fraction (PAF), and total and relative mortality 
    due to non-optimal temperatures for a given scenario, and time period.
    The model has the option to read in either the original Relative Risk
    functions from Honda et al., 2014, or to use the adapted version used
    in the Lancet Countdown reports from 2002-2024 (Romanello et al., 2024).
    The model also has the option to extrapolate the ERF curves to a defined 
    temperature range.
    As with the other models coded, the calculations are given per year, in each
    timestep, daily temperature data is loaded, the PAF is calculated for each 
    grid cell and then aggregated to the annual and regional level. 
    The results are saved in a csv file, then the attributable (total and relative) 
    mortality is calculated and saved in a separate csv file.
    """
    
    def load_inputs(self):
        self.fls = LoadInputData.from_files(sets=self.sets)
        
    def run(self):
        print("----------------------------------------------------------------")
        print(f"Running Mortality-Temperature model (Honda et al., 2014 version)")
        print(f"for project: {self.sets.project}, scenario: {self.sets.scenario}, and years: {self.sets.years[0]}-{self.sets.years[-1]}...")
        print("----------------------------------------------------------------")
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
    pop_ssp: np.ndarray
        2D array with population per grid cell aligned with daily temperature resolution.
    regions: np.ndarray
        2D array with region locations per grid cell aligned with daily temperature resolution.
    regions_range: np.ndarray
        1D array with location indices.
    opt_temp: np.ndarray
        2D array with optimal temperatures as defined by Honda et al.
    erf: pd.DataFrame
        DataFrame with ERFs
    temp_min:
    temp_max:
    paf: pd.DataFrame
        Dataframe to store Population Attributable Fraction results.
    """
    
    pop_ssp: xr.DataArray
    regions: np.ndarray
    regions_range: np.ndarray
    opt_temp: np.ndarray
    risks: np.ndarray
    temp_min: dict
    temp_max: dict
    region_dict: dict
    image_dict: dict
    paf: pd.DataFrame
    paf_counter: pd.DataFrame
    base_years: list=range(1980,1990)
    
    @classmethod
    def from_files(cls, sets):
        
        """
        Load all input files required for PAF calculations. 
        Data is located in the wdir/data folder.
        """
        
        print("[1] Loading main files...")
        
        print("[1.1] Loading SSP population data...")
        ssp = re.search(r"SSP\d", sets.scenario).group()
        years = sorted(set(sets.years).union(cls.base_years))
        
        pop_ssp = pop.LoadPopulationMap(
            wdir=sets.wdir, 
            scenario=sets.scenario, 
            ssp=ssp, 
            years=years.sort()
            )
        
        print(f"[1.2] Loading region classification...")
        regions, regions_range = pop.LoadRegionClassificationMap(
            sets.wdir, 
            temp_dir=sets.temp_dir,
            region_class="countries", 
            scenario=sets.scenario,
            pop_map=pop_ssp
            )

        # Load Exposure Response Function files for the relevant diseases
        risks, temp_min, temp_max = LoadERF(
            wdir=sets.wdir,
            project=sets.project, 
            extrap_erf=sets.extrap_erf, 
            temp_max=sets.temp_max
            )
        
        # Load file with optimal temperatures for 1980-2010
        optimal_temperatures = LoadOptimalTemperatures(
            wdir=sets.wdir,
            optimal_range=sets.optimal_range,
            scenario=sets.scenario
            )
        
        print("[1.5] Loading region classification dictionaries...")
        region_dict, image_dict = p2m.LoadRegionClassificationDicts(sets.wdir)
        
        print("[1.6] Creating final dataframe to store results...")
        paf = pd.DataFrame(
            index=pd.MultiIndex.from_product([["cold", "heat", "all"], ["mean", "upper", "lower"], regions_range]), 
            columns=sets.years
            ).sort_index()
        paf_counter = pd.DataFrame(
            index=pd.MultiIndex.from_product([["cold", "heat", "all"], ["mean", "upper", "lower"], regions_range]), 
            columns=cls.base_years
            ).sort_index()
        
        return cls(
            pop_ssp=pop_ssp,
            regions=regions,
            regions_range=regions_range,
            opt_temp=optimal_temperatures,
            risks=risks,
            temp_min=temp_min,
            temp_max=temp_max,
            region_dict=region_dict,
            image_dict=image_dict,
            paf=paf,
            paf_counter=paf_counter
        ) 
        
        
        
def LoadERF(wdir, project, extrap_erf=False, temp_max=None):
    
    """ 
    Load the single risk function from Honda et al. (2014).
    The function also outputs the min and max temperature dictionaries.
    """
    
    print("[1.3] Loading Exposure Response Function...")
    
    if re.search(r"honda", project.lower()):
        function = "Honda"
    elif re.search(r"romanello", project.lower()):
        function = "Romanello2024"
        
    risk_function = (pd.read_csv(wdir+f"/data/risk_function/RiskFunction_{function}.csv")
        .astype(float))
    
    # Extrapolate risk_function
    if extrap_erf == True:
        print("Extrapolating ERFs...")
        risk_function = ExtrapolateERF(risk_function, temp_max) 

    # Prepare risk function for lookup, convert entries to int and multiply by 10
    risk_function["index_temperature"] = (risk_function["daily_temperature"]*10).astype(int)
    
    # Perform groupby operation using the columns
    temp_min = risk_function["index_temperature"].min()   
    temp_max = risk_function["index_temperature"].max()
    
    # Prepare risk function lookup arrays
    risks = risk_function[["relative_risk_mean", "relative_risk_upper", "relative_risk_lower"]].to_numpy()
    
            
    return risks, temp_min, temp_max



def ExtrapolateERF(erf, temp_max):
    
    """
    If extrapolation is indicated, this function extrapolates the ERF curves to a 
    defined range using log-linear interpolation based on the last segment of the curves.
    It identifies local extremes to determine the segments for extrapolation.
    Returns a new dataframe with original and extrapolated values.
    """
    
    
    def log_linear_interp(xx, yy):
        # Linear interpolation of raw ln(RR) data over a df column  
        lin_interp = sp.interpolate.interp1d(xx, np.log(yy), kind="linear", fill_value="extrapolate")    
        return lambda zz: np.exp(lin_interp(zz))
    
    
    # Round index to one decimal
    erf["daily_temperature"]= erf["daily_temperature"].round(1)
    
    zero_index = erf.index[erf["daily_temperature"]==0.][0]
            
    # Define interpolation with last range
    interp={}
    
    for mode in ["mean", "upper", "lower"]:
        interp[mode] = log_linear_interp(
            erf["daily_temperature"].loc[zero_index:].values, 
            erf[f"relative_risk_{mode}"].loc[zero_index:].values
            )
    
    # Define temperature values to interpolate
    xx = np.round(
        np.linspace(erf["daily_temperature"].iloc[-1]+0.1, temp_max, 
        int((temp_max - erf["daily_temperature"].iloc[-1])/0.1)+1),
        1
        )

    erf_extrap = pd.DataFrame({
        "daily_temperature": xx,
        "relative_risk_mean": interp["mean"](xx),
        "relative_risk_upper": interp["upper"](xx),
        "relative_risk_lower": interp["lower"](xx)
        })
    
    erf_extrap = pd.concat([erf, erf_extrap], ignore_index=True)
            
    return erf_extrap 



def LoadOptimalTemperatures(wdir, optimal_range, scenario):
    
    """
    Load the optimal temperatures netcdf file calculated for a predefiend 
    period (1980-2010) and return as numpy array.
    """
    
    print("[1.4] Loading optimal temperatures...")
    
    # Load file with optimal temperatures for 1980-2010 period (default period)
    optimal_temps = xr.open_dataset(
        wdir +
        f"/data/optimal_temperatures/era5_t2m_{optimal_range}.nc"
        )
    
    if not re.search(r"ERA5", scenario):
        # Reduce resolution to 0.5x0.5 degrees
        optimal_temps = optimal_temps.coarsen(latitude=2, longitude=2, boundary="pad").mean(skipna=True)
    
    return optimal_temps[f"t2m_{optimal_range[-3:]}"].values



def CalculatePAFYear(sets, fls, year):
    
    """
    Calculate PAF for a given year, and subtract counterfactual PAFs if indicated
    """
    
    print(f"[2.1] Loading {year} daily temperatures...")
    daily_temp, num_days = tmp.LoadDailyTemperatures(
        temp_dir=sets.temp_dir,
        scenario=sets.scenario,
        temp_type="max",
        year=year, 
        pop_map=fls.pop_ssp,
        std_factor=1
        )

    print(f"[2.2] Calculating Population Attributable Fraction for {year}...") 
    
    # Calcuate PAFs per region and corresponding year
    AnnualPAFperRegion(
        fls=fls,
        year=year,
        num_days=num_days,
        daily_temp=daily_temp,
        counter=False
        )
    
    if year in fls.base_years:
        
        print(f"[2.3] Calculating counterfactual PAFs for year {year}...")
    
        AnnualPAFperRegion(
                    fls=fls,
                    year=year,
                    num_days=num_days,
                    daily_temp=daily_temp,
                    counter=True
                    )

    

def AnnualPAFperRegion(fls, year, num_days, daily_temp, counter):
    
    """
    Calculate weighted average of PAFs per region an year, using population as weights. 
    The function first calculates the PAFs per grid cell, then aggregates to the annual 
    level and finally calculates the weighted average per region.
    """
    
    # Select population for the corresponding year and convert to numpy array with non-negative values
    pop_year = np.clip(fls.pop_ssp.sel(time=f"{year}").mean("time").GPOP.values, 0, None)

    # Calculate baseline temperature (t - OT)
    baseline_temp = daily_temp - fls.opt_temp[...,np.newaxis]
    
    # Clip baseline temperatures to min and max values
    clip_baseline_temp = np.round(np.clip(baseline_temp*10, fls.temp_min, fls.temp_max), 0).astype(int)

    # Create mask for valid baseline temperatures
    mask = ~np.isnan(baseline_temp)

    # Get indices for lookup
    indices = (clip_baseline_temp[mask] - fls.temp_min).astype(int)
    
    for i,var in enumerate(["mean", "upper", "lower"]):
    
        # Initialize relative risks array
        relative_risks = np.full_like(baseline_temp, np.nan, dtype=float)

        # Get relative risks from risks lookup table    
        relative_risks[mask] = fls.risks[:,i][indices]
        
        # Flatten arrays
        regions_flat = np.nan_to_num(fls.regions.ravel()).astype(int)
        pop_flat = np.nan_to_num(pop_year.ravel())
        
        for mode in ["cold", "heat", "all"]:
            
            paf = np.where(relative_risks < 1, 0, 1 - 1/relative_risks)
            
            if mode == "cold":
                paf = np.where(clip_baseline_temp<0, paf, 0)
            if mode == "heat":
                paf = np.where(clip_baseline_temp>0, paf, 0)
            
            # Aggregate PAFs over days
            paf = np.sum(paf, axis=2) / num_days        
            paf_flat = np.nan_to_num(paf.ravel())
            
            # Calculate weighted sum of PAFs per region
            weighted_sum = np.bincount(regions_flat, weights=paf_flat * pop_flat)
            weight_pop_sum = np.bincount(regions_flat, weights=pop_flat)
            
            # Calculate weighted average for specified regions
            weighted_paf = weighted_sum[fls.regions_range] / np.maximum(weight_pop_sum[fls.regions_range], 1e-12)
        
            # Locate results in paf dataframe
            if counter == False:
                fls.paf.loc[(mode, var), year] = weighted_paf
            else:
                fls.paf_counter.loc[(mode, var), year] = weighted_paf    
        
    

def PostprocessResults(sets, fls):
    
    print("[3] Model run complete. Postprocessing...")
    
    paf = fls.paf
    paf.index.names = ["t_type", "var_erf", "region"]
    
    paf_counter = fls.paf_counter.mean(axis=1)
    paf.index.names = ["t_type", "var_erf", "region"]
    
    # Substract counterfactual PAFs
    paf_counterfactual = paf.sub(paf_counter, axis=0)
    
    print("[3.1] Saving PAF results...")
    
    class ScenarioNaming:
        def __init__(self, sets):
            self.extrap_part = "_extrap" if sets.extrap_erf else ""
            self.years_part = f"_{sets.years[0]}-{sets.years[-1]}"
            self.out_path = Path(sets.wdir) / "output" / f"{sets.project}"
            self.model = "Honda"
    sn = ScenarioNaming(sets)
    
    # Create project folder if it doesn' exist
    sn.out_path.mkdir(parents=True, exist_ok=True)
    
    print("3.2 Calculating attributable mortality and saving results...")
    
    causes = ["All causes"]  
    
    paf = ReformatPAF(fls, paf)
    paf_counterfactual = ReformatPAF(fls, paf_counterfactual)
    
    # Calculate mortality from PAF
    sn.counter = ""
    p2m.PAF2Mortality(sets, fls, paf, causes, sn)
    
    # Calculate mortality from PAF in the counterfactual scenario
    sn.counter = "_counterfactual"
    p2m.PAF2Mortality(sets, fls, paf_counterfactual, causes, sn)

    print("Model ran succesfully!")


    
def ReformatPAF(fls, paf):
    
    """
    Convert the PAF dataframe to an xarray with the same coordinates 
    as the GBD mortality and UN population data, to be able to merge the 
    three datasets and calculate attributable mortality.
    """
    
    paf = (
        paf
        .stack(future_stack=True)
        .reset_index()
        .rename(columns={"level_3": "year", 0: "paf", "region": "ISO3"})
        .assign(cause="All causes")
        .set_index(["ISO3", "t_type", "var_erf", "cause", "year"])
        .assign(paf=lambda df: df["paf"].astype(float)) 
        .to_xarray()
    )
            
    # Map location ids to ISO3 codes
    paf["ISO3"] = xr.DataArray(
        [fls.region_dict[id] for id in paf["ISO3"].values], 
        coords=paf["ISO3"].coords, 
        dims=paf["ISO3"].dims
    ).astype(object)

    paf = paf.sortby("ISO3").sortby("cause").sortby("t_type").sortby("var_erf").sortby("year")
    
    return paf