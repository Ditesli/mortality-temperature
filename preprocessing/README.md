# Preprocessing folder description

This folder contains the scripts to generate data that will be later used in the main model. In particular:

1. Code to create GCM data 
2. Code to create temperature zones
3. Code to give right format to the Exposure Response Functions

## Generate_GCM_data
This file consists on reformatting temperature data to save memory and make later calculation more optimal. NOTE: this code will be discarded once real temperature data is available.

### Input
The code requires temperature output from 5 Global Circulation Models. This data encompasses the 30-year mean of the beginning and end of the century. 

### Process
1. The code calculates the difference between the end and the beginning of the century per model.
2. The ensemble mean is calculated for the beginning and end of the century, and difference.
3. Interpolates data to 15 min.
4. Set sea cells to NaN.
5. Regrid the files to match the coordinates of population data.

### Output

The code results in three files $\it{ensemble\_mean\_start}$, $\it{ensemble\_mean\_end}$ and $\it{ensemble\_mean\_diff}$. These files serve later on to make the usual spatial interpolation from $GMST$ data to global data.


## Generate_ERA5_tz

The code generate the temperature zones used in the paper from [Burkart et al., 2021](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736%2821%2901700-1/fulltext). Following their methodology, a temperature zone is defined as the temperature mean from 1980-2019.

### Input

ERA5 reanalysis montlhy data. In particular 2 meter temperature, $\it{t2m }$, on single level.

### Process
1. Generate mean.
2. Convert to Celsius.
3. Set sea cells to NaN.
4. Regrid data to match population data.
5. Set Antarctica values to NaN.
6. Round temperature data to integer.
7. Limit the temperature values to the 6-28 degrees range.

### Output 
NetCDF file with temperature zone per pixel.

## Clean_format_ERF
This code converts the raw ERF draws to single funcitons per disease. The code has two approaches, one for each RR calculation method. NOTE: One will later on be discarded, when we choose a method.

### Input
Files provided of ERF and TMRELs.

### Process
1. Calculate the mean TMREL of all locations available in the folder for 1990, 2010, and 2019, and for each temperature zone.
2. Calculate the mean and STD of each Exposure Responsu Function.
3. Reescale the ERF with respect to the TMREL per temperature zone for the year 2010.
4. Convert the log(RR) to RR.
5. Give the files the format needed depending on the method (M1 or M2).

### Output
ERF in the right format.