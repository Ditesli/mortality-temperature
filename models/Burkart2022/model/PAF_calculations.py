import pandas as pd
import numpy as np
import scipy as sp
import xarray as xr
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
    draw: any,
    single_erf: bool,
    extrap_erf: bool
    ):

    sets = ModelSettings(
        wdir=wdir,
        temp_dir=temp_dir,
        project=project,
        scenario=scenario,
        years=years,
        draw=draw,
        single_erf=single_erf,
        extrap_erf=extrap_erf,
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
    draw: any
    single_erf: bool
    extrap_erf: bool
    causes: dict = field(
        default_factory=lambda: {
        "ckd":"Chronic kidney disease", 
        "cvd_cmp":"Cardiomyopathy and myocarditis", 
        "cvd_htn":"Hypertensive heart disease", 
        "cvd_ihd":"Ischemic heart disease", 
        "cvd_stroke":"Stroke", 
        "diabetes":"Diabetes mellitus",
        "inj_animal":"Animal contact", 
        "inj_disaster":"Exposure to forces of nature", 
        "inj_drowning":"Drowning", 
        "inj_homicide":"Interpersonal violence", 
        "inj_mech":"Exposure to mechanical forces", 
        "inj_othunintent":"Other unintentional injuries", 
        "inj_suicide":"Self-harm", 
        "inj_trans_other":"Other transport injuries", 
        "inj_trans_road":"Road injuries", 
        "resp_copd":"Chronic obstructive pulmonary disease", 
        "lri":"Lower respiratory infections"
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
    Model that calculates the Population Attributable Factor from mortality to heat and cold
    according to Burkart., et al (2022). The model uses the Exposure Response Functions and
    TMRELs derived in their study and as provided by the authors.
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
        Provide the range of years and step the model will run.
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
    draw : any
        Provide either:
        - An integer to select a specific draw of the ERFs and TMRELs
        - "mean" to get the mean of all draws
        - "random" to get a random draw
    single_erf : bool
        True or False depending on whether a single function for all causes will be used.
    extrap_erf : bool
        True or False dependeding on whether the log-linear extrapolation will be applied.
        
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
    temperature_zones: np.ndarray
        2D array with temperature zones per grid cell aligned with daily temperature resolution.
    regions: np.ndarray
        2D array with region locations per grid cell aligned with daily temperature resolution.
    regions_range: np.ndarray
        1D array with location indices.
    pop_map: np.ndarray
        2D array with population per grid cell aligned with daily temperature resolution.
    min_dict: dict
        Minimum daily temperature of the ERF per cause of death.
    max_dict: dict
        Maximum daily temperature of the ERF per cause of death.
    tmrel: np.ndarray
        2D array with TMRELs per grid cell aligned with daily temperature resolution.
    df_erf_tmrel: pd.DataFrame
        Dataframe with unique combinations of temperature zones and TMRELs per location.
    paf: pd.DataFrame
        Dataframe to store Population Attributable Fraction results.
    """
    
    temperature_zones: np.ndarray
    regions: np.ndarray
    regions_range: np.ndarray
    pop_map: np.ndarray
    pop_region: pd.DataFrame
    min_dict: dict
    max_dict: dict
    tmrel: np.ndarray
    df_erf_tmrel: pd.DataFrame
    paf: pd.DataFrame


    @classmethod
    def from_files(cls, sets):
        
        """
        Load all input files required for PAF calculations. 
        Data is located in the wdir/data folder.  
        """
        
        print("[1] Loading input files...")
        
        temperature_zones = LoadTemperatureZones(sets)
        
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
        erf, min_dict, max_dict = LoadExposureResponseFunctions(sets)
        
        # Load Theoretical Minimum Response Exposure Level
        tmrel = LoadTMRELsMap(sets, 2010) # Default year: 2010
        
        # Shift Relative Risks of ERF according to the TMREL
        df_erf_tmrel = ShiftRRfromTMREL(
            erf=erf, 
            pop_map=pop_map,
            tmrel=tmrel, 
            temperature_zones=temperature_zones,
            causes=sets.causes
            )
        
        # Average to one ERF to discard cuase of death
        if sets.single_erf == True:
            df_erf_tmrel = AverageToSingleERF(df_erf_tmrel)
            
        print("[1.8] Creating final dataframe to store results...")
        paf = pd.DataFrame(
            index=regions_range, 
            columns=pd.MultiIndex.from_product([sets.years, sets.causes, ["cold", "heat", "all"]])
            )  
        
        
        return cls(
            temperature_zones=temperature_zones,
            regions=regions,
            regions_range=regions_range,
            pop_map=pop_map,
            pop_region=pop_region,
            min_dict=min_dict,
            max_dict=max_dict,           
            tmrel=tmrel,
            df_erf_tmrel=df_erf_tmrel,
            paf=paf
        )
    


def LoadTemperatureZones(sets):
    
    """
    Import ERA5 temperature zones nc file and convert to numpy array
    """
    
    print("[1.1] Loading temperature zones as numpy array...")
    
    # Import ERA5 temperature zones
    era5_tz = (
        xr.open_dataset(f"{sets.wdir}/data/temperature_zones/ERA5_mean_1980-2019_land_t2m_tz.nc")
        .t2m.values
    )
    
    if not re.search(r"SSP[1-5]_ERA5", sets.scenario):
        # Reshape array to 4D blocks of 2x2
        arr_reshaped = era5_tz.reshape(360, 2, 720, 2)
        # Calculate mode over the 2x2 blocks to reduce resolution
        era5_tz = sp.stats.mode(sp.stats.mode(arr_reshaped, axis=3, keepdims=False).mode, axis=1, keepdims=False).mode
    
    return era5_tz



def LoadExposureResponseFunctions(sets):
    
    """
    Get a single erf dataframe with all the Relative Risks of all causes. Depending on the
    draw the dataframe can contain:
    - the mean of all draws 
    - random draw between the 1000 available
    - select a specific draw (useful for propagation of uncertainty runs)
    Later the original ln(RR) are converted to RR and fill in the nans of the dataframe to 
    complete all daily temperature ranges with flattened curves.
    The function also produces two dics corresponding to the max and min daily temperatures 
    in each temperature zone available in the files.
    """
    
    print("[1.5] Loading Exposure Response Functions (ERFs)...")
    
    #  Read the raw Exposure Response Functions from the specified path.
    erf_dict = {}
    for cause in list(sets.causes.keys()):
        # Open file for selected cause of death
        erf_cause = pd.read_csv(
            f"{sets.wdir}/data/burkart_sm/ERF/{cause}_curve_samples.csv", index_col=[0,1]
            )
        erf_cause.index = pd.MultiIndex.from_arrays(
            [erf_cause.index.get_level_values(0), erf_cause.index.get_level_values(1).round(1)]
            )
        erf_dict[cause] = erf_cause

    # Choose the cause with the largest daily temperature range to create the base dataframe
    erf = pd.DataFrame(
        index=pd.MultiIndex.from_arrays([erf_dict["ckd"].index.get_level_values(0), 
                                         erf_dict["ckd"].index.get_level_values(1).round(1)])
        )
    

    # Fill the dataframe with the selected draw or the mean of all draws
    for cause in sets.causes.keys():
        if sets.draw == "mean":
            erf[cause] = erf_dict[cause].mean(axis=1)
        elif sets.draw == "random":
            draw = random.randint(0,999)
            erf[cause] = erf_dict[cause][f"draw_{draw}"]  
        elif isinstance(sets.draw_type, int):
            erf[cause] = erf_dict[cause][f"draw_{sets.draw}"]
        
     
    # Extrapolate ERF 
    if sets.extrap_erf == True:
        print("[1.4.1] Extrapolating ERFs...")
        erf = ExtrapolateERF(erf)        
      
      
    erf = (
        erf
        .astype(float).apply(lambda x: np.exp(x)) # Convert log(rr) to rr   
        .rename_axis(index={"annual_temperature":"temperature_zone"})
        .reset_index() # Convert MultiIndex levels into columns
        .set_index("temperature_zone") # Keep only one index
        .groupby("temperature_zone", group_keys=False)
        .apply(lambda g: g.bfill().ffill()) # Flatten the curves 
    )
    
    # Get min and max temperature values per cause
    min_dict = erf.groupby("temperature_zone")["daily_temperature"].min().to_dict()
    max_dict = erf.groupby("temperature_zone")["daily_temperature"].max().to_dict()
        
    return erf, min_dict, max_dict



def ExtrapolateERF(erf):
    
    """
    If extrapolation is indicated, this function extrapolates the ERF curves to a 
    defined range using log-linear interpolation based on the last segment of the curves.
    It identifies local extremes to determine the segments for extrapolation.
    Returns a new dataframe with original and extrapolated values.
    """
    
    # Set extapolation range
    temp_max = 50 
    temp_min = -22
    
    # Round index level 1 to one decimal
    erf.index = pd.MultiIndex.from_arrays(
        [erf.index.get_level_values(0), 
        erf.index.get_level_values(1).round(1)]
        )

    # Define new dataframe to store original values
    erf_extrap = pd.DataFrame(
        index=pd.MultiIndex.from_product([
            range(6,29), 
            np.round(np.arange(temp_min, temp_max+0.1, 0.1),1)],
            names=["annual_temperature", "daily_temperature"]
            ), 
        columns=erf.columns
        )
    erf_extrap.loc[erf.index, erf.columns] = erf
    
    # Iterate over temperature zones and disease
    for tz in erf_extrap.index.levels[0]:
        for disease in erf.columns:
            
            # Select relevant column
            erf_tz = erf.loc[tz, disease].dropna()
            
            # Take derivative of selected series to find local extremes
            dy = np.gradient(erf_tz, erf_tz.index)
            zero_crossings = np.where(np.diff(np.sign(dy)) != 0)[0]
            
            # Extrapolate extremes and locate in dataframe
            if erf_tz.index[-1] < temp_max:
                ExtrapolateToHeatAndCold(erf_tz, tz, erf_extrap, disease, zero_crossings, temp_max, "heat")
            if erf_tz.index[0] > temp_min:
                ExtrapolateToHeatAndCold(erf_tz, tz, erf_extrap, disease, zero_crossings, temp_min, "cold")
            
    return erf_extrap 



def ExtrapolateToHeatAndCold(erf, tz, erf_extrap, disease, zero_cross, t_lim, mode):
        
    """
    Extrapolate hot and cold tails of the ERF curves by applying a log-linear
    extrapolation in the last values of the functions.
    """
    
    def LogLinearExtrap(xx, yy):
        # Linear interpolation of raw ln(RR) data over a df column.        
        lin_interp = sp.interpolate.interp1d(xx, yy, kind="linear", fill_value="extrapolate")    
        return lambda zz: lin_interp(zz)  
    
    # Extapolate towards hot or cold temperatures
    if mode=="heat":
        
        # Choose index with last extreme
        index_peak = erf.index[0] if len(zero_cross) == 0 else erf.index[zero_cross[-1]]
        # Define interpolation with last values
        interp = LogLinearExtrap(erf[index_peak:].index, erf.loc[index_peak:].values)
        # Define temperature values to interpolate
        xx = np.round(np.linspace(erf.index[-1]+0.1, t_lim, int((t_lim - erf.index[-1])/0.1)+1), 1)
        
    if mode=="cold":
        
        index_peak = erf.index[-1] if len(zero_cross) == 0 else erf.index[zero_cross[0]]
        interp = LogLinearExtrap(erf[:index_peak].index, erf.loc[:index_peak].values)
        xx = np.round(np.linspace(t_lim, erf.index[0], int((erf.index[0] - t_lim)/0.1)+1), 1)
    
    # Locate extrapolated values
    yy = interp(xx)
    xx_multiindex = pd.MultiIndex.from_product([[tz], xx])
    erf_extrap.loc[xx_multiindex, disease] = yy
    
    
    
def LoadTMRELsMap(sets, year):
    
    """
    The function gets a single TMREL draw according to draw argument:
    - mean of all draws 
    - random draw between the 100 available
    - specific draw (useful for propagation of uncertainty runs)
    The function loads the .nc file with optimal temperatures for the selected year
    (Available: 1990, 2010, 2020). Default: 2010.
    """
            
    print("[1.6] Loading Theoretical Minimum Risk Exposure Levels (TMRELs)...")
    
    tmrel = xr.open_dataset(f"{sets.wdir}/data/TMRELs_nc/TMRELs_{year}.nc")
    
    if not re.search(r"SSP[1-5]_ERA5", sets.scenario):
        # Reduce resolution to 0.5x0.5 degrees
        tmrel = tmrel.coarsen(latitude=2, longitude=2, boundary="pad").mean(skipna=True)
    
    if sets.draw == "mean":
        tmrel = tmrel.tmrel.values.mean(axis=2)
    elif sets.draw == "random":
        tmrel = tmrel.sel(draw=random.randint(1,100)).tmrel.values
    elif isinstance(sets.draw, int):
        draw = sets.draw % 100 if sets.draw > 100 else sets.draw
        tmrel = tmrel.sel(draw=draw).tmrel.values
        
    return tmrel



def ShiftRRfromTMREL(erf, pop_map, tmrel, temperature_zones, causes):
    
    """
    Every temperature zone gets assings all the possible TMREL (with positive POP). 
    The dataframe is later merged with the ERF to divide the RR values by the RR at 
    TMREL for each temperature zone, shifting the curves so that RR=1 at TMREL.
    """
            
    print("[1.7] Shifting Relative Risks (RRs) relative to TMRELs...")
    
    
    def divide_by_tmrel(group, causes):
        
        """
        Locates per temperature zone the row whose daily temperature equals 
        the TMREL to divide the rest of rows and vertically shift the ERFs.
        """    
        
        # Locate the rows whose daily temperature equals the TMREL
        row_tmrel = group.loc[group["daily_temperature"] == group["tmrel"].iloc[0]]
        # Divide by first row to shift the RR vertically
        group[causes] = group[causes] / row_tmrel.iloc[0][causes]
        
        return group

    
    # Mask TMREL and temperature_zones arrays with valid POP
    mask_pop = (pop_map.GPOP > 0).any(dim="time")
    tmrel_valid_pop = tmrel[mask_pop.values]
    tz_valid_pop = temperature_zones[mask_pop.values]

    
    df_erf_tmrel = (
        pd.DataFrame({"temperature_zone": tz_valid_pop, "tmrel": np.round(tmrel_valid_pop,1)})
        .drop_duplicates()
        .merge(erf, on=["temperature_zone"], how="right") # Merge with ERF data
        .set_index("temperature_zone")  # Set temperature_zone as index
        .groupby("temperature_zone", group_keys=False).apply( # For each tz, divide RR by TMREL
            lambda group: divide_by_tmrel(group, list(causes.keys())))
        .reset_index()
    )

    return df_erf_tmrel

        

def AverageToSingleERF(df):
    
    """
    Average all the columns of the Exposure Response Functions dataframe except 
    "daily_temperature" to get a single ERF for all temperature zones.
    """
    
    print("[1.4.1] Generating single ERF per disease...")
    
    # Exclude the "temperature_zone" and "tmrel" columns from averaging
    cols_to_average = df.columns.difference(["temperature_zone", "tmrel"])
    
    # Calculate the mean for the selected columns, grouped by "daily_temperature"
    df_mean = df[cols_to_average].groupby(df["daily_temperature"]).transform("mean")
    
    # Combine the mean values with the "temperature_zone" and "daily_temperatures" columns
    df_mean["tmrel"] = df["tmrel"]
    df_mean["daily_temperature"] = df["daily_temperature"]
    
    df_mean = df_mean.drop_duplicates().reset_index(drop=True)
        
    return df_mean

     
        
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
    
    # Generate dataframe with population weighted factors per region and rest of variables
    df_region = CreatePopulationDF(sets, fls, region_mask, pop_year, daily_temp, num_days)
    
     # Merge the ERF with the grouped data to assign Relative Risk values,
    if sets.single_erf == True: # Excluding temperature_zones
        df_all = df_region.merge(fls.df_erf_tmrel, on=["daily_temperature", "tmrel"], how="left")  
    else:
        df_all = df_region.merge(fls.df_erf_tmrel, on=["temperature_zone", "daily_temperature", "tmrel"], how="left")

    # Make two new dataframes separating the attributable deaths from heat and cold
    df_cold = df_all[df_all["daily_temperature"] < df_all["tmrel"]].copy()
    df_heat = df_all[df_all["daily_temperature"] > df_all["tmrel"]].copy()
    
    causes = list(sets.causes.keys())
    
    # Calculating PAFs per temperature types
    for df, temp_type in zip([df_heat, df_cold, df_all], ["heat", "cold", "all"]):
        
        # Convert the RR to PAF following GBD method
        paf_region = pd.DataFrame(
            np.where(df[causes] < 1, 
                    df["population"].values[:, None] * (df[causes] - 1),
                    df["population"].values[:, None] * (1 - 1 / df[causes])
            ),
            columns=[f"{col}" for col in sets.causes],
            index=df.index
        ).sum(axis=0)

        # Locate aggregated PAF in annual dataframe
        fls.paf.loc[region, (year, causes, temp_type)] = [paf_region[f"{d}"] for d in causes]
            
        

def CreatePopulationDF(sets, fls, mask, pop_year, daily_temp, num_days):
    
    """
    Create a dataframe that includes data on temperature zone, daily temperature, population and tmrel
    for all the grid cells with population data in a given IMAGE region.
    The dataframe is grouped per temperature_zone and daily_temperature, calculating the fraction of
    population per each combination.
    The daily temperature values are truncated according to the min and max values available in the ERF.
    This dataframe will be used to merge with the ERF and calculate the RR and PAF.    
    """
    
    # Get 1D arrays where POP>0 using the regional mask and matching daily temp dimension (num_days)
    pop_array = np.concatenate([pop_year[mask]]*num_days)
    tz_array = np.concatenate([fls.temperature_zones[mask]]*num_days)
    tmrel_array = np.concatenate([fls.tmrel[mask]]*num_days)
    
    # Get 1D array of daily temperature using regional mask
    dayTemp_array = MaskTemperatureDataRegionally(daily_temp, mask, pop_array, num_days)

    # Create a dataframe with data on temperature zone, daily temperature, population and tmrel
    df_region = pd.DataFrame(
        {"temperature_zone": tz_array, 
        "daily_temperature": np.round(dayTemp_array,1),
        "population": np.round(pop_array,1), 
        "tmrel": np.round(tmrel_array,1)}
        )

    # Truncate min and max Temperature values according to availability in ERF
    df_region["daily_temperature"] = (
        df_region["daily_temperature"]
        .clip(lower=df_region["temperature_zone"].map(fls.min_dict), 
              upper=df_region["temperature_zone"].map(fls.max_dict))
    )
    
    # Calculate the fraction of population per (temperature_zone), daily_temperature and tmrel
    if sets.single_erf == False:
        df_region = df_region.groupby(["temperature_zone", "daily_temperature", "tmrel"], as_index=False).sum()
    else:
        df_region = df_region.groupby(["daily_temperature", "tmrel"], as_index=False).sum()
        
    df_region["population"] /= df_region["population"].sum()
    
    return df_region

    

def MaskTemperatureDataRegionally(daily_temp, valid_mask, pop_array, num_days):
    
    """
    Creates a 1-D array from the daily temperature data by masking first cells with
    population in a region.
    """
    
    # Create an empty array to store the daily temperatures
    dayTemp_array = np.empty(len(pop_array), dtype=np.float32) 
    index = 0
    
    # Iterate over the number of days in the year
    for day in range(num_days):        
        # Get the daily temperature with POP data
        dayTemp_np = daily_temp[:,:,day][valid_mask]
        # Append the values to the array
        dayTemp_array[index:index+len(dayTemp_np)] = dayTemp_np
        index += len(dayTemp_np) 
        
    return np.array(dayTemp_array, dtype=np.float64)
    


def PostprocessResults(sets, fls):
        
    print("[3] Model run complete. Saving results...")
    
    # Stack dataframe to leave only years in columns
    paf = fls.paf.stack([1,2], future_stack=True).reorder_levels([1,2,0]).sort_index()
    paf.index.names = ["disease", "t_type", "region"]
    
    # Substract counterfactual mortality
    # paf = paf.sub(paf[list(range(2001,2011))].mean(axis=1), axis=0)
    
    erf_part = "_1erf" if sets.single_erf else ""
    extrap_part = "_extrap" if sets.extrap_erf else ""
    years_part = f"_{sets.years[0]}-{sets.years[-1]}"
    
    # Create project folder if it doesn"t exist
    out_path = Path(sets.wdir) / "output" / f"{sets.project}" 
    out_path.mkdir(parents=True, exist_ok=True)
            
    # Save the results and temperature statistics
    paf.to_csv(out_path /
                   f"PAF_{sets.project}_{sets.scenario}_ISO3{years_part}{extrap_part}{erf_part}.csv")  
    
    PAF2Mortality(sets, fls, paf, out_path, erf_part, extrap_part, years_part, ages="All")
    PAF2Mortality(sets, fls, paf, out_path, erf_part, extrap_part, years_part, ages="oldest")
    
    print("Model ran succesfully!")
      


def PAF2Mortality(sets, fls, paf, out_path, erf_part, extrap_part, years_part, ages):
    
    print("[3.1] Calculating attributable mortality and saving results...")
    
    if re.search(r"ERA5", sets.scenario):
        
        if ages == "All":
            age_groups = [
                "<5 years", "5-9 years", "10-14 years", "15-19 years", "20-24 years", 
                "25-29 years", "30-34 years", "35-39 years", "40-44 years", 
                "45-49 years", "50-54 years", "55-59 years", "60-64 years", 
                "65-69 years", "70-74 years", "75-79 years", "80-84 years",
                "80-85 years", "85+ years"
                ]
            age_part = ""
        elif ages == "oldest":
            age_groups = ["65-69 years","70-74 years","75-79 years", "80-84 years", "85+ years"]
            age_part = "_oldest"

        # Load GBD mortality records
        wdir_up = os.path.dirname(sets.wdir)
        gbd_mor = pd.read_csv(f"{wdir_up}/data/GBD_mortality/IHME-GBD_2022_DATA.csv")

        mask = (
            gbd_mor["cause_name"].isin(sets.causes.values()) & 
            (gbd_mor["sex_name"] == "Both") & # Both sexes
            gbd_mor["year"].isin(sets.years) & # Only years assessed
            gbd_mor["age_name"].isin(age_groups) & # Select 5-year age groups
            (gbd_mor["location_name"] != "Global") # Exclude global mortality
        )

        gbd_mor = (
            gbd_mor.loc[mask, ["location_id","location_name","cause_name","year","val"]]
            .groupby(["location_id", "location_name", "cause_name", "year"], as_index=False)
            .sum() # Sum age groups per location, cause and year
            .pivot(index=["location_id", "location_name", "cause_name"],
                    columns="year",
                    values="val") # Pivot years to columns
            .reset_index()
            .set_index(["cause_name", "location_id"]) # Reindex for later merging
            .rename(index={v: k for k, v in sets.causes.items()}, level=0) # Invert dict to map
            .rename_axis(index={"cause_name": "disease", "location_id": "region"})
        )

        # --------------- Mortality for ISO3 countries --------------

        # Reorder levels
        paf_reordered = paf.reorder_levels(order=[1,0,2])
        t_types = ["heat", "cold", "all"]

        # Multiply per gbd mortality
        result = pd.concat(
            [paf_reordered.loc[tt] * gbd_mor.drop(columns="location_name") for tt in t_types],
            keys=t_types,
            names=["t_type"]
        )

        # Change region id to region name
        result= (
            result
            .reset_index() # Reset
            .merge(gbd_mor.groupby(["location_name", "region"]).first().reset_index()[["location_name", "region"]], # Merge with GBD mortality data
                    left_on="region", 
                    right_on="region", 
                    how="left")
            .drop(columns={"region"}) # Discard region number from GBD
            .rename(columns={"location_name":"region", "disease":"cause"}) # Leave only country name
            )

        result_allcauses = result.groupby(["t_type", "region"]).sum()
        result_allcauses["cause"] = "All causes"
        # Concat cause specific and all causes dataframes
        result_allcauses = pd.concat([result_allcauses.reset_index(), result], ignore_index=True)
        
        # -------------- Relative mortality ---------------------
        
        res = result_allcauses.merge(fls.pop_region, left_on="region", right_on="loc_name", how="left")
        # 
        res = (
            res
            .assign(**{
                f"{col.replace('_pop', '')}_relmor":
                res[int(col.replace('_pop', ''))] * 1e5 / res[col].replace(0, np.nan)
                for col in res.columns if str(col).endswith('_pop')
            })
            .set_index(["t_type", "region", "cause"])
            .filter(like="_relmor")
            .rename(columns=lambda col: int(col.replace('_relmor', '')))
            .reset_index()
        )
        
        result_allcauses.insert(0, "unit", "Total Mortality")
        res.insert(0, "unit", "Relative Mortality")
        
        result_allcauses = pd.concat([result_allcauses, res], ignore_index=True)
        
        result_allcauses.to_csv(
            out_path /
            f"Mortality_{sets.project}_{sets.scenario}_ISO3{years_part}{age_part}{extrap_part}{erf_part}.csv"
            ) 
        

        # -------------- Mortality for IMAGE regions -------------

        region_class = pd.read_csv(
            wdir_up + 
            "/data/region_classification.csv"
            ).drop_duplicates(subset="gbd_location_id", keep="first")

        image_results = (
            result
            .merge(region_class[["gbd_level3", "IMAGE26"]], left_on="region", right_on="gbd_level3", how="left")
            .drop(columns=["region", "gbd_level3"])
            .groupby(["t_type", "cause", "IMAGE26"])
            .sum()
            .reset_index()
            .rename(columns={"IMAGE26":"region"})
        )

        # Add global mortality result
        world = image_results.groupby(["cause", "t_type"]).sum()
        world["region"] = "World"
        image_world_results = pd.concat([image_results, world.reset_index()], ignore_index=True)
        image_allcauses = image_world_results.groupby(["t_type", "region"]).sum()
        image_allcauses["cause"] = "All causes"
        image_allcauses = pd.concat([image_allcauses.reset_index(), image_world_results], ignore_index=True)
        
        
        # -------- Relative mortality for IMAGE26 ------------
        
        image_pop = (
            fls.pop_region
            .merge(region_class[["gbd_location_id", "IMAGE26"]], left_on=["loc_id"], right_on="gbd_location_id", how="left")
            .groupby("IMAGE26")
            .sum()
            .drop(columns=["loc_name", "loc_id", "gbd_location_id"])
        )
        image_pop.loc["World",:] = image_pop.sum(axis=0)
        
        res = image_allcauses.merge(image_pop.reset_index(), left_on="region", right_on="IMAGE26", how="left")
        
        res = (
            res
            .assign(**{
                f"{col.replace('_pop', '')}_relmor":
                res[int(col.replace('_pop', ''))] * 1e5 / res[col].replace(0, np.nan)
                for col in res.columns if str(col).endswith('_pop')
            })
            .set_index(["t_type", "region", "cause"])
            .filter(like="_relmor")
            .rename(columns=lambda col: int(col.replace('_relmor', '')))
            .reset_index()
        )
        
        image_allcauses_copy = image_allcauses.copy()
        image_allcauses_copy.insert(0, "unit", "Total Mortality")
        res.insert(0, "unit", "Relative Mortality")
        
        image_allcauses_copy = pd.concat([image_allcauses_copy, res], ignore_index=True)        

        image_allcauses_copy.to_csv(
            out_path /
            f"mortality_{sets.project}_{sets.scenario}_IMAGE{years_part}{age_part}{extrap_part}{erf_part}.csv"
            ) 

        # ----------- PAF for IMAGE regions ----------------

        gbd_mor_image = (
            gbd_mor
            .reset_index()
            .merge(region_class[["gbd_level3", "IMAGE26"]], left_on="location_name", right_on="gbd_level3", how="left")
            .drop(columns=["region", "location_name", "gbd_level3"])
            .rename(columns={"disease":"cause", "IMAGE26": "region"})
            .groupby(["cause", "region"])
            .sum()
        )

        image_results = image_results.set_index(["t_type", "cause", "region"])
        image_paf = pd.concat(
            [image_results.loc[tt] / gbd_mor_image for tt in t_types],
            keys=t_types,
            names=["t_type"]
        )
        image_paf.to_csv(out_path /
                    f"PAF_{sets.project}_{sets.scenario}_IMAGE{years_part}{extrap_part}{erf_part}.csv") 

    else:
        print("Cause-specific mortality projections not available yet :(")