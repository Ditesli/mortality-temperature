import numpy as np
import xarray as xr
import netCDF4 # netCDF4 needs to be imported to solve incompatibility issues with tensorflow
import tensorflow as tf
from datetime import datetime
from keras.losses import MeanSquaredError
from keras.losses import MeanAbsoluteError
import yaml, pym, prism
from dataclasses import dataclass, field


# Disable the logging from keras
tf.keras.config.disable_interactive_logging()

# Model specifications
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
TEMPERATURE_C_TO_K = 273.15
is_coupled = False



class Pattern_Scaling:
    sets: any
    
    #%% === INITIALIZATION ===
    def __init__(self, sets):
        

        # ----------- Load monthly historical CRU data -----------
        self.temperature_historic = (
            xr.open_dataset(sets.emulator["temperature_historical"] + "GTMP_30MIN.nc")
            .rename({"NM":"month"})
            .assign_coords(month = MONTH_NAMES) # Align month coordinates
            )["GTMP_30MIN"]
        
        
        # ----------- Load pattern scalling ΔT data map -----------
        self.temperature_scalling = (
            xr.open_dataset(
            sets.emulator["pattern_scalling"]
            + "GTMP_"
            + sets.emulator["pattern_scalling_scenario"]
            + ".nc"
            )
            .rename({"category_1":"month"})
            .assign_coords(month = MONTH_NAMES) # Align month coordinates
            .isel({"time":0})  # Select unique year
        )["GTMP_"+ sets.emulator["pattern_scalling_scenario"]]
        

        # ----------- Load temperature variability map -----------
        self.temperature_variability = (
            xr.open_dataset(sets.emulator["climate_variability"])
            .rename({"lon":"longitude", "lat":"latitude"}) # Rename coords
            .set_index(grid=["latitude", "longitude"])
            .unstack("grid")
            .assign_coords( # Align the coordinates
                longitude=lambda ds: ds.longitude - 180 - 0.25,
                latitude=lambda ds: ds.latitude + 0.25
            )
            .sortby('latitude', ascending=False) # Reverse the order of latitude vals
            .groupby("time.year") # Group by year, to apply the variability per month
        )


        #  ----------- Load global IMAGE temperature change -----------
        
        final_year =  sets.years[-1]+1 if sets.years[-1] < 2100 else 2100
        
        timeline = prism.Timeline(
                start=prism.Q_(1980,"year"), # Can be changed
                end=prism.Q_(final_year,"year"),
                stepsize=prism.Q_(1, "year")
                )

        self.global_temperature = prism.TimeVariable(
            dims = (),
            unit = "K",
            file = sets.emulator["temperature_projection"],
            timeline = timeline
        )

        # Load ΔT between beginning and end of the century for selected RCP
        with open(
            sets.emulator["pattern_scalling"] 
            + sets.emulator["pattern_scalling_scenario"] 
            + ".dat","r"
            ) as fid:

            self.delta_global_temperature = prism.Q_(np.float64(fid.readlines()),"K")


    ### =========== APPLY PATTERN SCALING ===========
    
    def ApplyPatternScalling(self, time):
        
        FIRST_SCENARIO_YEAR = 2020

        temperature_extended = []

        # Ensure the year is not higher that 2100
        years_to_loop = [time - 1, time, time] if time == 2100 else [time - 1, time, time + 1]

        for i, year in enumerate(years_to_loop):

            # Convert time to datetime (time historic ends in scenario year)
            time_historic = np.minimum(year, FIRST_SCENARIO_YEAR)

            # Historical data: Apply the (running-mean) historic data
            temperature = self.temperature_historic.interp(
                dict(time = np.datetime64(datetime(time_historic,1,1))))

            # Future: Apply scenario-historical data + pattern scaled with historical
            if year > FIRST_SCENARIO_YEAR:
                temperature += (
                    self.temperature_scalling
                    * prism.M_(
                        (
                            self.global_temperature[prism.Q_(year,"year")] 
                            - self.global_temperature[prism.Q_(FIRST_SCENARIO_YEAR,"year")]
                        )
                        / self.delta_global_temperature
                    )
                )

            # Add the variability (per month)
            for m in range(0,12):
                temperature.loc[
                                dict(
                                    longitude=self.temperature_variability[year]["longitude"],
                                    latitude=self.temperature_variability[year]["latitude"],
                                    month = MONTH_NAMES[m]
                                )
                        ] += (self.temperature_variability[year]["tas_interp"].isel(time=m))


            if i==0:
                # Add December of the previous year
                temperature_extended.append(temperature.sel(month="Dec"))

            if i==1:
                # Separate single year to apply the emulator to the correct year
                temperature_2_emulator = temperature
                # Add to extended dataset 
                temperature_extended.append(temperature)
                
            if i==2:
                # Add January of the next year
                temperature_extended.append(temperature.sel(month="Jan"))

        # Concatenate the three years to create the extended dataset
        temperature_extended = xr.concat(temperature_extended, dim="month", coords="different", compat="equals")

        return temperature_2_emulator, temperature_extended
    
    
    def ClimatologyFromPatternScalling(self, time):
        
        FIRST_SCENARIO_YEAR = 2020

        years = time if type(time)==range else [time]

        # Years to create the climatology
        years_to_loop = range(years[0]-30, years[-1])

        temperature_data = []

        for year_clim in years_to_loop:

            # Convert time to datetime (time historic ends in scenario year)
            time_historic = np.minimum(year_clim, FIRST_SCENARIO_YEAR)

            # Historical data: Apply the (running-mean) historic data
            temperature = self.temperature_historic.interp(
                dict(time = np.datetime64(datetime(time_historic,1,1))))

            # Future: Apply scenario-historical data + pattern scaled with historical
            if year_clim > FIRST_SCENARIO_YEAR:
                temperature += (
                    self.temperature_scalling
                    * prism.M_(
                        (
                            self.global_temperature[prism.Q_(year_clim,"year")] 
                            - self.global_temperature[prism.Q_(FIRST_SCENARIO_YEAR,"year")]
                        )
                        / self.delta_global_temperature
                    )
                )
            
            # Append annual temperature data to the list
            temperature_data.append(temperature.mean(dim="month"))

        # Calculate the climatology by averaging across the years
        climatology = (
            xr.concat(temperature_data, dim="time", coords="different", compat="equals")
            .mean(dim="time")
        )

        return climatology
        


@dataclass
class Run_Emulator:
    sets: any
    
    # INITIALIZATION ===============================================================================

    def __init__(self, sets):
        
        # Set the variables that we want to predict and output
        self.predicted_variables = ['cdd','wdd','tsd','tskew','tkurt','tmin','tmax']
        self.output_variables = ["tsd"]

        
        # Initialise pattern-scaling
        self.IMAGE_pattern_scaling = Pattern_Scaling(sets)

        # === Initialize grids === # 

        # Determine the lon/lat coordiantes
        self.lon_IMAGE = np.arange(-180,180,0.5)
        self.lat_IMAGE = np.arange(-90,90,0.5)
        
        bounding_box = {
            "lon": [-180, 180],
            "lat": [-60.5,  84.75]
            }

        # Get the coordinates of the region
        self.lon_emulator = np.arange(
            bounding_box["lon"][0],
            bounding_box["lon"][1],
            0.5 # Emulator resolution
        )

        self.lat_emulator = np.arange(
            bounding_box["lat"][0],
            bounding_box["lat"][1],
            0.5 # Emulator resolution
        )

        # === Initialize datasets === # 

        # Initialize the data that we need for normalization
        self.normalization_data = self.InitialiseNormalizationDataset(sets, self.predicted_variables)

        # === Initialize emulator === #

        # Import the emulator
        self.emulator = tf.keras.models.load_model(
            sets.emulator["emulator_data"] + sets.emulator["emulator_name"],
            custom_objects = {
                "masked_mse" : MeanSquaredError(),
                "masked_mae" : MeanAbsoluteError()
            }
        )
        

    def InitialiseNormalizationDataset(self, sets, predicted_variables):
        
        """
        Imports and initializes the data that is requiered to normalize the emulator.
        
        Normalization, so the mean is around "0" and the standard deivation is around "1" is
        important to obtain good results from the emulator. We will apply gaussian normalization,
        which requiers mean and standard deviation.
        """

        # Load the normalization in memory  
        normalization_data = xr.Dataset()    

        # Loop across all the predicted variables
        for var_name in ["tas", *predicted_variables]:

            # Loop across the data used for normalization (e.g. mean, std)
            for var_type in ["mean","std"]:

                # Load the data
                raw_normalization_data = (
                    xr.open_dataset(
                    sets.emulator['emulator_data']
                    + sets.emulator[f'standarization_{var_type}'].replace("$$$",var_name),
                    decode_times = False
                    )
                    .rename({var_name:var_name + '_' + var_type}) # Rename "mean" or "std"
                    .drop_vars(["time_bnds","time"]) # Drop the time bounds dimension
                )

                # Put everything into one dataarray
                normalization_data = xr.merge(
                    [normalization_data,raw_normalization_data],
                    compat="override",
                    join="outer"
                    )
                
                
        normalization_data = (
            normalization_data
            .rename({"lon":"longitude", "lat":"latitude"}) # Rename coords
            .assign_coords(longitude=lambda ds: (ds.longitude + 180) % 360 - 180)
            .sortby("longitude")
            .interp(
                dict(longitude = self.lon_IMAGE, latitude = self.lat_IMAGE)
                )
            .isel({"time":0}) # Select the unique time step
            .drop_vars("height") # Drop the height dimension
        )

        return normalization_data
    
        
    # ==============================================================================================
    # RUN SCRIPTS 
    # ==============================================================================================
    
    def run_emulator(self, year):
        
        # Initialize data-arrays ----------------------------------------------------------------
        
        # Create data-arrays that will contain the temperature data and climate indicators
        self.temperature_IMAGE = xr.DataArray(
            name="temperature",
            dims=["longitude","latitude","month"],
            coords=dict(
                longitude=("longitude", self.lon_IMAGE.tolist()),
                latitude=("latitude", self.lat_IMAGE.tolist()),
                month=("month", MONTH_NAMES),
            ),
        )        
        
        # Initialize final dataframe with emulator statistics
        self.climate_indicators = xr.DataArray(
            name="climate_indicator",
            dims=["longitude","latitude","indicator","month"],
            coords=dict(
                longitude=("longitude", self.lon_IMAGE.tolist()),
                latitude=("latitude", self.lat_IMAGE.tolist()),
                indicator=("indicator", self.predicted_variables),
                month=("month", MONTH_NAMES)
            ),
        ).astype("float32") # Convert to float 32 to reduce the amount of data
        
        
        # Receive Climate -------------------------------------------------------------------------

        # Calculate the IMAGE pattern scaling for one year
        temperature_IMAGE, temperature_extended = self.IMAGE_pattern_scaling.ApplyPatternScalling(year)

        # Pre-process the data 
        self.temperature_IMAGE[:] = (
            temperature_IMAGE
            .interp(dict(
            longitude = self.lon_IMAGE, latitude = self.lat_IMAGE))
            .transpose("latitude","longitude","month")
        )
        
        # Run emulator -------------------------------------------------------------------------

        # Call the emulator
        self.RunEmulatorOneYear()

        # Return mean temperature and STD
        return temperature_extended, self.climate_indicators

            

    def RunEmulatorOneYear(self):
        
        """ 
        Run the emulator for a given year
        """

        # Make sure temperature is in "Kelvin"
        self.temperature_IMAGE[:] += TEMPERATURE_C_TO_K

        # Normalize the data from IMAGE
        temperature_IMAGE =  (
            (self.temperature_IMAGE - self.normalization_data["tas_mean"])
            / self.normalization_data["tas_std"]
        )

        # Loop across the months
        for month in MONTH_NAMES:

            # Create the grid that we will use as input
            temperature = np.zeros([1,len(self.lon_IMAGE),len(self.lat_IMAGE),1])
            temperature[0,:,:,0] = temperature_IMAGE.sel(dict(month = month)).to_numpy()

            # Make 100% sure that no NaN have ended up in the data (including the ocean). Any
            # "0" values in the ocean do not affect the emulation.
            temperature[np.isnan(temperature)] = 0

            # Predict the emulator for current emulator region
            raw_climate_indicator = self.emulator.predict(temperature.transpose([0,2,1,3])) 

            # Add the predicted climate to the climate indicator
            self.climate_indicators.loc[
                dict(longitude = self.lon_emulator, latitude = self.lat_emulator, month = month)] = (
                    raw_climate_indicator[0,:,:,:].transpose([1,0,2]))

        # Denormalize the data and keep only STD
        var_name = "tsd"
        
        self.climate_indicators = (
            self.climate_indicators
            .sel(indicator=var_name) # Keep only STD
            .sortby('latitude', ascending=False) # Reverse latitude order to go from 90 to -90
            .transpose("latitude", "longitude", "month") # Transpose to use later as numpy array
        )
        
        self.climate_indicators += (
            (
                self.climate_indicators *
                self.normalization_data[var_name+'_std']
            )
            + self.normalization_data[var_name+'_mean']
        )