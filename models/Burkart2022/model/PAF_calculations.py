import pandas as pd
import numpy as np
import scipy as sp
import xarray as xr
import re, sys, os, random
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
    regions: str,
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
        regions=regions,
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
    regions: str 
    draw: any
    single_erf: bool
    extrap_erf: bool
    diseases: dict = field(default_factory=lambda: {
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
        if isinstance(self.years, range):
            self.years = range(self.years.start, self.years.stop + 1)



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
                fls=self.sets,
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
        Array with temperature zones per grid cell and aligned with daily temperature resolution.
    regions: any

    """
    
    temperature_zones: np.ndarray
    paf_final: any
    pop_ssp: any
    regions: any
    regions_range: any
    tmrel: any
    df_erf_tmrel: any
    min_dict: dict
    max_dict: dict


    @classmethod
    def from_files(cls, sets):
        
        """
        Read and load all input files required for PAF calculations. Data is located in 
        the wdir/data folder.  
        """
        
        print("[1] Loading files for calculations...")
        
        temperature_zones = LoadTemperatureZones(sets)
        
        print(f"[1.2] Loading region classification ({sets.regions}) map...")
        regions, regions_range = pop.LoadRegionClassificationMap(sets.wdir, sets.temp_dir, sets.regions, sets.scenario)
        
        print("[1.3] Loading SSP population data...")
        ssp = re.search(r"SSP\d", sets.scenario).group()
        pop_ssp = pop.LoadPopulationMap(sets.wdir, sets.scenario, ssp, sets.years)
        
        df_erf, min_dict, max_dict = LoadExposureResponseFunctions(sets)
        tmrel = LoadTMRELsMap(sets, 2010) # Default years: 2010
        df_erf_tmrel = ShiftRRfromTMREL(df_erf, tz_tmrel_combinations(pop_ssp, tmrel, temperature_zones), sets.diseases)
        
        if sets.single_erf == True:
            print("[1.4.1] Generating single ERF per disease...")
            df_erf_tmrel = average_erf(df_erf_tmrel)
            
        print("[1.5] Creating final dataframe to store results...")
        paf_final = pd.DataFrame(index=regions_range, 
                            columns=pd.MultiIndex.from_product([sets.years, sets.diseases, ["cold", "heat", "all"]]))  
        
        
        return cls(
            paf_final=paf_final,
            pop_ssp=pop_ssp,
            regions=regions,
            regions_range=regions_range,
            temperature_zones=temperature_zones,
            tmrel=tmrel,
            min_dict=min_dict,
            max_dict=max_dict
        )
    


def LoadTemperatureZones(sets):
    
    """
    Import ERA5 temperature zones and convert to numpy array
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
    Get a single erf dataframe with all the Relative Risks of all diseases. Depending on the
    draw the dataframe can contain:
    - the mean of all draws 
    - random draw between the 1000 available
    - select a specific draw (useful for propagation of uncertainty runs)
    Later the original ln(RR) are converted to RR and fill in the nans of the dataframe to 
    complete all daily temperature ranges with flattened curves.
    The function also produces two dics corresponding to the max and min daily temperatures 
    in each temperature zone available in the files.
    """
    
    print("[1.4] Loading Exposure Response Functions (ERFs)...")
    
    #  Read the raw Exposure Response Functions from the specified path.
    erf_dict = {}
    for disease in list(sets.diseases.keys()):
        erf_disease = pd.read_csv(f"{sets.wdir}/data/burkart_sm/ERF/{disease}_curve_samples.csv", 
                                    index_col=[0,1])
        erf_disease.index = pd.MultiIndex.from_arrays([erf_disease.index.get_level_values(0),
                                                        erf_disease.index.get_level_values(1).round(1)])
        erf_dict[disease] = erf_disease

    # Choose disease with the largest daily temperature range to create the base dataframe
    erf = pd.DataFrame(index=pd.MultiIndex.from_arrays([erf_dict["ckd"].index.get_level_values(0), 
                                                        erf_dict["ckd"].index.get_level_values(1).round(1)]))
    

    # Fill the dataframe with the selected draw or the mean of all draws
    for disease in sets.diseases.keys():
        if sets.draw == "mean":
            erf[disease] = erf_dict[disease].mean(axis=1)
        elif sets.draw == "random":
            draw = random.randint(0,999)
            erf[disease] = erf_dict[disease][f"draw_{draw}"]  
        elif isinstance(sets.draw_type, int):
            erf[disease] = erf_dict[disease][f"draw_{sets.draw}"]
        
     
    # Extrapolate ERF 
    if sets.extrap_erf == True:
        print("[1.3.1] Extrapolating ERFs...")
        erf = ExtrapolateERF(erf)     
    else:
        pass     
    
    
    erf = (
        erf
        .astype(float).apply(lambda x: np.exp(x)) # Convert log(rr) to rr   
        .rename_axis(index={"annual_temperature":"temperature_zone"})
        .reset_index() # Convert MultiIndex levels into columns
        .set_index("temperature_zone") # Keep only one index
        .groupby("temperature_zone", group_keys=False)
        .apply(lambda g: g.bfill().ffill()) # Flatten the curves 
    )
    
    # Get min and max temperature values per disease
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
    erf.index = pd.MultiIndex.from_arrays([erf.index.get_level_values(0), 
                                           erf.index.get_level_values(1).round(1)])

    # Define new dataframe to store original values
    erf_extrap = pd.DataFrame(index=pd.MultiIndex.from_product([range(6,29), np.round(np.arange(temp_min, temp_max+0.1, 0.1),1)],
                                                                names=["annual_temperature", "daily_temperature"]), 
                              columns=erf.columns)
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
    
    def linear_interp(xx, yy):
        # Linear interpolation of raw ln(RR) data over a df column.        
        lin_interp = sp.interpolate.interp1d(xx, yy, kind="linear", fill_value="extrapolate")    
        return lambda zz: lin_interp(zz)  
    
    # Extapolate towards hot or cold temperatures
    if mode=="heat":
        
        # Choose index with last extreme
        index_peak = erf.index[0] if len(zero_cross) == 0 else erf.index[zero_cross[-1]]
        # Define interpolation with last values
        interp = linear_interp(erf[index_peak:].index, erf.loc[index_peak:].values)
        # Define temperature values to interpolate
        xx = np.round(np.linspace(erf.index[-1]+0.1, t_lim, int((t_lim - erf.index[-1])/0.1)+1), 1)
        
    if mode=="cold":
        
        index_peak = erf.index[-1] if len(zero_cross) == 0 else erf.index[zero_cross[0]]
        interp = linear_interp(erf[:index_peak].index, erf.loc[:index_peak].values)
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
            
    print("[1.5] Loading Theoretical Minimum Risk Exposure Levels (TMRELs)...")
    
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

     
        
def CalculatePAFYear(sets, fls, year):
    
    print(f"Calculating Population Attributable Fraction for year {year}...") 
    
    daily_temp, num_days = tmp.load_temperature_type(sets.temp_dir, sets.scenario, "mean", year, fls.pop_ssp)

    # Select population for the corresponding year
    pop_ssp_year = fls.pop_ssp.sel(time=f"{year}").mean("time").GPOP.values

    # Set a mask of pixels for each region
    for region in sets.regions_range:
        
        get_regional_paf(pop_ssp_year, sets.regions, region, year, num_days, fls.temperature_zones, 
                            daily_temp, fls.tmrel, fls.df_erf_tmrel, fls.paf_final, sets.diseases, fls.min_dict, 
                            fls.max_dict, sets.single_erf)



def rr_to_paf(df, rr_year, diseases, year, region, temp_type):
    
    """
    Convert the Relative Risk into the Population Atributable Fraction following
    Burkart et al.
    """
    
    # Convert the RR to PAF
    df[[f"{col}" for col in diseases]] = np.where(df[diseases] < 1, 
                                                  df["population"].values[:, None] * (df[diseases] - 1),
                                                  df["population"].values[:, None] * (1 - 1 / df[diseases]))
    
    # Aggregate PAFs
    df_aggregated = df.sum(axis=0)
    
    # Locate aggregated PAF in annual dataframe
    rr_year.loc[region, (year, diseases, temp_type)] = [df_aggregated[f"{d}"] for d in diseases]
    
    

def get_temp_array_from_mask(daily_temp, valid_mask, pop_array, num_days):
    
    """
    Creates a 1-D array from the daily temperature data by masking first cells with
    population and the flattening
    """
    
    # Create an empty array to store the daily temperatures
    dayTemp_array = np.empty(len(pop_array), dtype=np.float32) 
    index = 0
    
    # Iterate over the number of days in the year
    for day in range(num_days):
        
        # Get the daily temperature
        dayTemp_np = daily_temp[:,:,day]
        
        # Mask the values to get only the ones with POP data
        dayTemp_values = dayTemp_np[valid_mask]
        
        # Append the values to the array
        dayTemp_array[index:index+len(dayTemp_values)] = dayTemp_values
        index += len(dayTemp_values) # or len(pop_array)
        
    return dayTemp_array
    
    
    
def get_array_from_mask(data, valid_mask, num_days):
    
    """ 
    Converts GREG, yearly population and temperature zone xarrays into 1-D numpy arrays 
    by keeping only the entries where there is population data (data is more than 0)
    """
    
    # Convert xarray to numpy and get the values for the valid mask
    data_masked = data[valid_mask]
    # Repeat the same values for the number of days in a year
    data_array = np.concatenate([data_masked] * num_days)
    
    return data_array
    
    
    
def get_data_masked_per_region(valid_mask, num_days, pop, era5_tz, daily_temp, tmrel): 
    
    """
    Use the mask for the population data to mask the population temperature zone, tmrel map and 
    daily temperature data. The first three maps are repreated 365/366 times depending the 
    number of days in the specific year. This process creates 1-D arrays for the data representing
    the different combinations.
    """
    
    # Get arrays for the data using the functions defined above
    pop_array = get_array_from_mask(pop, valid_mask, num_days)
    meanTemp_array = get_array_from_mask(era5_tz, valid_mask, num_days)
    dayTemp_array = get_temp_array_from_mask(daily_temp, valid_mask, pop_array, num_days)
    tmrel_array = get_array_from_mask(tmrel, valid_mask, num_days)
    
    # print(f"Data masked")
    
    return pop_array, meanTemp_array, dayTemp_array, tmrel_array
    
    

def create_population_df(mask, pop_ssp_year, temperature_zones, daily_temp, tmrel, num_days, min_dict, max_dict, single_erf=False):
    
    """
    Create a dataframe that includes data on temperature zone, daily temperature, population and tmrel
    for all the grid cells with population data in a given IMAGE region.
    The dataframe is grouped per temperature_zone and daily_temperature, calculating the fraction of
    population per each combination.
    The daily temperature values are truncated according to the min and max values available in the ERF.
    This dataframe will be used to merge with the ERF and calculate the RR and PAF.    
    """
    
    pop_array, t_zones_array, daily_t_array, tmrel_array = get_data_masked_per_region(mask, 
                                                                                    num_days, 
                                                                                    pop_ssp_year, 
                                                                                    temperature_zones, 
                                                                                    daily_temp, 
                                                                                    tmrel)

    #Change array type for posterior merging
    daily_temperatures_array = np.array(daily_t_array, dtype=np.float64)
    
    # Create a dataframe that includes data on temperature zone, daily temperature, population and tmrel
    df_pop = pd.DataFrame({"temperature_zone": t_zones_array, "daily_temperature": np.round(daily_temperatures_array,1),
                           "population": np.round(pop_array,1), "tmrel":np.round(tmrel_array,1)})

    # Truncate min and max Temperature values according to availability in ERF
    df_pop["daily_temperature"] = df_pop["daily_temperature"].clip(lower=df_pop["temperature_zone"].map(min_dict), 
                                                                   upper=df_pop["temperature_zone"].map(max_dict))
    
    if single_erf == False:
        # Group per temperature_zone, daily_temperature and tmrel, calculating the fraction of population per each combination
        df_pop = df_pop.groupby(["temperature_zone", "daily_temperature", "tmrel"], as_index=False).sum()
    
    else:
        # Group per daily_temperature and tmrel, calculating the fraction of population per each combination
        df_pop = df_pop.groupby(["daily_temperature", "tmrel"], as_index=False).sum()
        
    df_pop["population"] /= df_pop["population"].sum()
    
    return df_pop

    
    
def get_regional_paf(pop_ssp_year, regions, region, year, num_days, temperature_zones, daily_temp, tmrel, 
                     df_erf_tmrel, rr_year, diseases, min_dict, max_dict, single_erf):
    
    """
    Get the Population Atributable Fraction per region and year by:
    1. Creating a dataframe with population weighted factors per temperature zone and daily temperature
    2. Merging the dataframe with the ERF shifted by the TMREL to assign RR values
    3. Separating the dataframe into cold and heat attributable deaths
    4. Calculating the PAF and storing it in the final dataframe
    
    Parameters:
    - pop_ssp_year: population data for the specific year
    - regions: array with IMAGE region classification
    - region: specific IMAGE region to calculate the PAF
    - year: specific year to calculate the PAF
    - num_days: number of days in the specific year
    - temperature_zones: array with temperature zones classification
    - daily_temp: array with daily temperature data for the specific year
    - tmrel: array with TMREL data for the specific year
    - df_erf_tmrel: dataframe with the ERF data shifted by the TMREL
    - rr_year: dataframe to store the final RR values per region and year
    - diseases: list of diseases to calculate the RR
    - min_dict: dictionary with the minimum temperature values per temperature zone
    - max_dict: dictionary with the maximum temperature values per temperature zone
    """
    
    # Get mask of 
    image_region_mask = (pop_ssp_year > 0.) & (regions == region)
    
    # Generate dataframe with population weighted factors per 
    df_pop = create_population_df(image_region_mask, pop_ssp_year, temperature_zones, 
                                daily_temp, tmrel, num_days, min_dict, max_dict)
    
    if single_erf == True:
        # Merge the ERF with the grouped data to assign rr, excluding the temperature_zone column
        df_all = pd.merge(df_pop, df_erf_tmrel,  on=["daily_temperature", "tmrel"], how="left")
        
    else:
        # Merge the ERF with the grouped data to assign rr
        df_all = pd.merge(df_pop, df_erf_tmrel,  on=["temperature_zone", "daily_temperature", "tmrel"], how="left")

    # Make two new dataframes separating the cold and heat attributable deaths
    df_cold = df_all[df_all["daily_temperature"] < df_all["tmrel"]].copy()
    df_heat = df_all[df_all["daily_temperature"] > df_all["tmrel"]].copy()
        
    for df, temp_type in zip([df_heat, df_cold, df_all], ["heat", "cold", "all"]):
        rr_to_paf(df, rr_year, diseases, year, region, temp_type)
        
        

def average_erf(df):
    
    """
    Average all the columns of the Exposure Response Functions dataframe except "daily_temperature".
    This is useful when using a single ERF for all temperature zones.
    Parameters:
    - df: dataframe with the ERF data
    Returns:
    - df_mean: dataframe with the averaged ERF data
    """
    
    # Exclude the "temperature_zone" column from averaging
    cols_to_average = df.columns.difference(["temperature_zone", "tmrel"])
    
    # Calculate the mean for the selected columns, grouped by "daily_temperature"
    df_mean = df[cols_to_average].groupby(df["daily_temperature"]).transform("mean")
    
    # Combine the mean values with the "temperature_zone" and "daily_temperatures" columns
    df_mean["tmrel"] = df["tmrel"]
    df_mean["daily_temperature"] = df["daily_temperature"]
    
    df_mean = df_mean.drop_duplicates().reset_index(drop=True)
        
    
    return df_mean



def divide_by_tmrel(group, diseases):
    
    """
    This function works per temperature zone groups. It locates the row whose daily temperature equals 
    the TMREL and divides this row for the rest of them
    """    
    
    # Locate the rows whose daily temperature equals the TMREL
    fila_tmrel = group.loc[group["daily_temperature"] == group["tmrel"].iloc[0]]
    
    # Select first row
    reference = fila_tmrel.iloc[0][diseases]
    
    # Divide to shift the RR vertically
    group[diseases] = group[diseases] / reference
    
    return group



def ShiftRRfromTMREL(df_erf, df_tz_tmrel, diseases):
    
    """
    For every temperature zone, the merging assings all the possible TMREL. 
    This implies that we will have repeated rows for the daily temperature and relative risks.
    This function divides the RR values by the RR at TMREL for each temperature zone,
    effectively shifting the curves so that RR=1 at TMREL.
    Parameters:
    - df_erf: dataframe with the ERF data
    - df_tz_tmrel: dataframe with unique combinations of temperature zones and TMREL
    - diseases: list of diseases to calculate the RR
    Returns:
    - df_erf_tmrel: dataframe with the shifted RR values
    """
            
    print("[1.4] Shifting Relative Risks (RRs) relative to TMRELs...")

    # Merge df_tz_tmrel with ERF data
    df_erf_tmrel = pd.merge(df_erf, df_tz_tmrel, on=["temperature_zone"], how="left") 
    
    # Set temperature_zone as index
    df_erf_tmrel = df_erf_tmrel.set_index("temperature_zone")
    
    # For each temperature zone, divide the RR by the TMREL
    df_erf_tmrel = df_erf_tmrel.groupby("temperature_zone", group_keys=False).apply(
        lambda group: divide_by_tmrel(group, diseases))

    # Reset index
    df_erf_tmrel = df_erf_tmrel.reset_index()

    return df_erf_tmrel



def tz_tmrel_combinations(pop_ssp, tmrel, temperature_zones):
    
    """
    This will produce a dataframe with the unique combinations of temperature zones and TMREL.
    This is necessary to later merge with the ERF dataframe to shift the RR curves
    Parameters:
    - pop_ssp: xarray with population data for the selected scenario
    - tmrel: 2-D np.array with the optimal temperature per pixel
    - temperature_zones: 2-D np.array with the temperature zones per pixel
    Returns:
    - df_tz_tmrel: dataframe with unique combinations of temperature zones and TMREL
    """
    
    # Get any cell with population data for the selected scenario
    mask_pop = (pop_ssp.GPOP > 0).any(dim="time")
    
    # Mask TMREL and temperature_zones arrays
    tmrel_valid_pop = tmrel[mask_pop.values]
    tz_valid_pop = temperature_zones[mask_pop.values]
    
    # Create dataframe with these data
    df_tz_tmrel = pd.DataFrame({"temperature_zone": tz_valid_pop, "tmrel": np.round(tmrel_valid_pop,1)})
    
    # Remove duplicated rows
    df_tz_tmrel = df_tz_tmrel.drop_duplicates()
    
    return df_tz_tmrel



def PostprocessResults(sets, fls):
        
    print("[3] Model run complete. Saving results...")
    
    erf_part = "_1erf" if sets.single_erf else ""
    extrap_part = "_extrap" if sets.extrap_erf else ""
    years_part = f"_{sets.years[0]}-{sets.years[-1]}"
            
    # Save the results and temperature statistics
    fls.paf_final.to_csv(f"{sets.wdir}\\output\\PAF_{sets.project}_{sets.scenario}_{sets.regions}_{years_part}{extrap_part}{erf_part}.csv")  
    
    print("Model ran succesfully!")