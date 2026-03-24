import pandas as pd
import numpy as np
import scipy as sp
import xarray as xr
from dataclasses import dataclass
from pathlib import Path
import os, sys, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_common import temperature as tmp
from utils_common import population as pop



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
    Run the main model 
    ADD DESCRIPTION
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
    min_val:
    max_val:
    paf: pd.DataFrame
        Dataframe to store Population Attributable Fraction results.
    """
    
    pop_ssp: xr.DataArray
    regions: np.ndarray
    regions_range: np.ndarray
    opt_temp: np.ndarray
    erf: pd.DataFrame
    min_val: dict
    max_val: dict
    paf: pd.DataFrame
    
    @classmethod
    def from_files(cls, sets):
        
        '''
        Load all input files required for PAF calculations. 
        Data is located in the wdir/data folder.
        '''
        
        print('[1] Loading main files...')
        
        print("[1.1] Loading SSP population data...")
        ssp = re.search(r"SSP\d", sets.scenario).group()
        pop_ssp = pop.LoadPopulationMap(sets.wdir, sets.scenario, ssp, sets.years)
        
        print(f"[1.2] Loading region classification...")
        regions, regions_range = pop.LoadRegionClassificationMap(
            sets.wdir, 
            temp_dir=sets.temp_dir,
            region_class="countries", 
            scenario=sets.scenario,
            pop_ssp=pop_ssp)

        # Load Exposure Response Function file
        erf, min_val, max_val = LoadERF(sets.wdir, sets.extrap_erf, sets.temp_max)
        
        # Load file with optimal temperatures for 1980-2010
        optimal_temperatures = LoadOptimalTemperatures(sets.wdir, sets.optimal_range, sets.scenario)
        
        # Create final dataframe
        paf = pd.DataFrame(index=pd.MultiIndex.from_product([['Cold', 'Heat', 'All'], regions_range]), 
                            columns=sets.years)  
        
        return cls(
            pop_ssp=pop_ssp,
            regions=regions,
            regions_range=regions_range,
            opt_temp=optimal_temperatures,
            erf=erf,
            min_val=min_val,
            max_val=max_val,
            paf=paf
        ) 
        
        
        
def LoadERF(wdir, extrap_erf=False, temp_max=None):
    
    ''' 
    Load the single risk function from Honda et al. (2014).
    The function also outputs the min and max temperature dictionaries.
    '''
    
    print('[1.3] Loading Exposure Response Function...')
        
    risk_function = (pd.read_csv(wdir+'/data/risk_function/interpolated_dataset.csv')
        .astype(float))
    
    # Extrapolate risk_function
    if extrap_erf == True:
        print('Extrapolating ERFs...')
        risk_function = ExtrapolateERF(risk_function, temp_max) 

    # Prepare risk function for lookup, convert entries to int and multiply by 10
    risk_function['index_temperature'] = (risk_function['daily_temperature']*10).astype(int)
    
    # Perform groupby operation using the columns
    min_val = risk_function['index_temperature'].min()   
    max_val = risk_function['index_temperature'].max()
            
    return risk_function, min_val, max_val



def ExtrapolateERF(erf, temp_max):
    
    """
    If extrapolation is indicated, this function extrapolates the ERF curves to a 
    defined range using log-linear interpolation based on the last segment of the curves.
    It identifies local extremes to determine the segments for extrapolation.
    Returns a new dataframe with original and extrapolated values.
    """
    
    
    def log_linear_interp(xx, yy):
        # Linear interpolation of raw ln(RR) data over a df column  
        lin_interp = sp.interpolate.interp1d(xx, np.log(yy), kind='linear', fill_value='extrapolate')    
        return lambda zz: np.exp(lin_interp(zz))
    
    
    # Round index to one decimal
    erf['daily_temperature']= erf['daily_temperature'].round(1)
    
    zero_index = erf.index[erf['daily_temperature']==0.][0]
            
    # Define interpolation with last range
    interp = log_linear_interp(erf['daily_temperature'].loc[zero_index:].values, 
                               erf['relative_risk'].loc[zero_index:].values)
    
    # Define temperature values to interpolate
    xx = np.round(np.linspace(erf['daily_temperature'].iloc[-1]+0.1, temp_max, 
                    int((temp_max - erf['daily_temperature'].iloc[-1])/0.1)+1), 1)

    erf_extrap = pd.DataFrame({
        'daily_temperature': xx,
        'relative_risk': interp(xx)
        })
    
    erf_extrap = pd.concat([erf, erf_extrap], ignore_index=True)
            
    return erf_extrap 



def LoadOptimalTemperatures(wdir, optimal_range, scenario):
    
    '''
    Load the optimal temperatures netcdf file calculated for a predefiend 
    period (1980-2010) and return as numpy array.
    '''
    
    print('[1.4] Loading optimal temperatures...')
    
    # Load file with optimal temperatures for 1980-2010 period (default period)
    optimal_temps = xr.open_dataset(wdir+f'/data/optimal_temperatures/era5_t2m_max_{optimal_range}_p84.nc')
    
    if not re.search(r"ERA5", scenario):
        # Reduce resolution to 0.5x0.5 degrees
        optimal_temps = optimal_temps.coarsen(latitude=2, longitude=2, boundary='pad').mean(skipna=True)
    
    return optimal_temps.t2m_p84.values



def CalculatePAFYear(sets, fls, year):
    
    '''
    Run the main model using ERA5 historical data
    
    Parameters:
    - wdir: working directory
    - era5_dir: directory where ERA5 daily temperature data is stored
    - years: list of the period in which the model will be run
    - ssp: SSP scenario name
    - region_class: region classification to use ('IMAGE26' or 'GBD_level3')
    - extrap_erf: boolean, if True extrapolate risk functions, if False use original one
    - temp_max: maximum temperature to extrapolate to 
    - temp_min: minimum temperature to extrapolate to 
    '''
    
    print(f'[2.1] Loading {year} daily maximum temperatures...')
    daily_temp, num_days = tmp.LoadDailyTemperatures(
        temp_dir=sets.temp_dir,
        scenario=sets.scenario,
        temp_type="max",
        year=year, 
        pop_ssp=fls.pop_ssp,
        std_factor=1
        )

    # Select population for the corresponding year and convert to numpy array with non-negative values
    pop_year = np.clip(fls.pop_ssp.sel(time=f'{year}').mean('time').GPOP.values, 0, None)
    
    print(f'[2.2] Calculating Population Attributable Fraction for {year}...') 

    # Calculate baseline temperature
    baseline_temp = daily_temp - fls.opt_temp[...,np.newaxis]
    
    # Clip baseline temperatures to min and max values
    clip_baseline_temp = np.round(np.clip(baseline_temp*10, fls.min_val, fls.max_val), 0).astype(int)

    # Prepare risk function lookup arrays
    temps = fls.erf["index_temperature"].to_numpy()
    risks = fls.erf["relative_risk"].to_numpy()

    # Initialize relative risks array
    relative_risks = np.full_like(baseline_temp, np.nan, dtype=float)

    # Create mask for valid baseline temperatures (used with IMAGE data)
    mask = ~np.isnan(baseline_temp)

    # Get indices for lookup
    indices = (clip_baseline_temp[mask] - temps.min()).astype(int)

    # Get relative risks from risks lookup table    
    relative_risks[mask] = risks[indices]

    # Calculate PAFs
    pafs = np.where(relative_risks < 1, 0, 1 - 1/relative_risks)
    
    # Calculate regional PAFs
    for mode in ['All', 'Heat', 'Cold']:
        fls.paf.loc[mode,year] = WeightedAvgOfPAFperRegion(fls, pafs, num_days, pop_year, mode, clip_baseline_temp)

    

def WeightedAvgOfPAFperRegion(fls, pafs, num_days, pop_year, mode, clip_base_temp):
    
    '''
    Calculate weighted average of PAFs per region
    '''
    
    # Apply mask to PAFs to select cold, hot or all temperatures
    if mode == 'Cold':
        pafs = np.where(clip_base_temp<0, pafs, 0)
    if mode == 'Heat':
        pafs = np.where(clip_base_temp>0, pafs, 0) 
    
    # Aggregate PAFs over days
    pafs = np.sum(pafs, axis=2) / num_days
    
    # Flatten arrays
    regions_flat = np.nan_to_num(fls.regions.ravel()).astype(int)
    pafs_flat = np.nan_to_num(pafs.ravel())
    pop_flat = np.nan_to_num(pop_year.ravel())
    
    # Calculate weighted sum of PAFs per region
    weighted_sum = np.bincount(regions_flat, weights=pafs_flat * pop_flat)
    weight_pop_sum = np.bincount(regions_flat, weights=pop_flat)
    
    # Calculate weighted average for specified regions
    weighted_avg = weighted_sum[fls.regions_range] / np.maximum(weight_pop_sum[fls.regions_range], 1e-12)
    
    return weighted_avg
    
    

def PostprocessResults(sets, fls):
    
    print("[3] Model run complete. Postprocessing...")
    
    # Substracting counterfactual mortality
    paf = fls.paf.sub(fls.paf[list(range(2001, 2011))].mean(axis=1), axis=0)
    paf.index.names = ["t_type", "region"]
    
    print("[3.1] Saving PAF results...")

    extrap_part = "_extrap" if sets.extrap_erf else ""
    years_part = f"_{sets.years[0]}-{sets.years[-1]}"

    # Create project folder if it doesn"t exist
    out_path = Path(sets.wdir) / "output" / f"{sets.project.upper()}" 
    out_path.mkdir(parents=True, exist_ok=True)
            
    # Save the results and temperature statistics
    paf.to_csv(out_path / 
                    f'PAF_{sets.project}_{sets.scenario}_ISO3{years_part}{extrap_part}_ot-{sets.optimal_range[-4:]}.csv')  

    print(["3.2 Calculating attributable mortality and saving results"])

    PAF2Mortality(sets, paf, out_path, years_part, extrap_part)

    print("Model ran succesfully!")
    
    

def PAF2Mortality(sets, paf, out_path, years_part, extrap_part):
    
    print("3.2 Calculating attributable mortality and saving results...")
    
    if re.search(r"ERA5", sets.scenario):
        
        # Load GBD mortality records
        wdir_up = os.path.dirname(sets.wdir)
        gbd_mor = pd.read_csv(f'{wdir_up}/data/GBD_mortality/IHME-GBD_2022_DATA.csv')
        
        mask = (
            (gbd_mor['cause_name'] == 'All causes') &
            (gbd_mor['sex_name'] == 'Both') &
            (gbd_mor['year'].isin(sets.years)) &
            (gbd_mor['age_name'].isin(['65-69 years', '70-74 years', '75-79 years', "80-84 years", '85+ years'])) &
            (gbd_mor['location_name'] != 'Global')
        )
        
        # Reformat mortality
        gbd_mor = (
            gbd_mor.loc[mask, ['location_id','location_name','year','val']]
            .groupby(['location_id', 'location_name', 'year'], as_index=False)
            .sum() # Sum all age groups 
            .pivot(index=['location_id', 'location_name'], # Move years to columns
                columns='year',
                values='val')
            .reset_index()
            .rename(columns={"location_id":"region"})
            .set_index("region")
        )
        
        
        # ----------- Mortality for ISO3 countries ------------
        
        t_types = ["Heat", "Cold", "All"]
        
        # Multiply per gbd mortality
        result = pd.concat(
            [paf.loc[tt] * gbd_mor.drop(columns="location_name") for tt in t_types],
            keys=t_types,
            names=["t_type"]
        )
        
        result = (
            result
            .reset_index() # Reset
            .merge(
                gbd_mor
                .groupby(["location_name", "region"])
                .first().reset_index()
                [["location_name", "region"]], # Merge with GBD mortality data
                left_on="region", 
                right_on="region", 
                how="left"
                )
            .drop(columns={"region"}) # Discard region number from GBD
            .rename(columns={"location_name":"region"}) # Leave only country name
            )
        
        result.to_csv(out_path / 
                   f'Mortality_{sets.project}_{sets.scenario}_ISO3{years_part}{extrap_part}_ot-{sets.optimal_range[-4:]}.csv')  
        
    
        # --------------- Mortality for IMAGE regions --------------
        
        region_class = pd.read_csv(
            wdir_up + 
            "/data/region_classification.csv").drop_duplicates(subset="gbd_location_id", keep='first')

        image_results = (
            result
            .merge(region_class[["gbd_level3", "IMAGE26"]], left_on="region", right_on="gbd_level3", how="left")
            .drop(columns=["region", "gbd_level3"])
            .groupby(["t_type", "IMAGE26"])
            .sum()
            .reset_index()
            .rename(columns={"IMAGE26":"region"})
        )
        
        world_results = image_results.groupby("t_type").sum()
        world_results["region"] = "World"
        image_world_results = pd.concat([image_results, world_results.reset_index()], ignore_index=True)
        
        image_world_results.to_csv(out_path / 
                   f'Mortality_{sets.project}_{sets.scenario}_IMAGE26{years_part}{extrap_part}_ot-{sets.optimal_range[-4:]}.csv')  
    
        
        # -------------- PAF for IMAGE regions --------------
        
        gbd_mor_image = (
            gbd_mor
            .reset_index()
            .merge(region_class[["gbd_level3", "IMAGE26"]], left_on="location_name", right_on="gbd_level3", how="left")
            .drop(columns=["region", "location_name", "gbd_level3"])
            .rename(columns={"IMAGE26": "region"})
            .groupby(["region"])
            .sum()
        )
        
        image_results = image_results.set_index(["t_type", "region"])
        
        image_paf = pd.concat(
            [image_results.loc[tt] / gbd_mor_image.replace(0, np.nan) for tt in t_types],
            keys=t_types,
            names=["t_type"]
        )
        
        image_paf.to_csv(out_path / 
                   f'PAF_{sets.project}_{sets.scenario}_IMAGE26{years_part}{extrap_part}_ot-{sets.optimal_range[-4:]}.csv')  
    
    
    else:
        print["Cause-specific mortality projections not available yet :("]