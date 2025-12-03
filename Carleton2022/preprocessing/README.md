The preprocessing part serves to do some calculations prior to running the main model:

1. Generate the region classification file to include the impact region classification (24,378 regions) and their corresponding ISO3 code, IMAGE region and WHO region classification.
2. Generate historical population files (2000-2010) per age group following the paper. This one prescribes to use Landscan population data and UN population data to generate the files.
3. Generate “present day” temperature, defined in the paper as T_0 and as the daily temperature between 2000 and 2010. This code uses ERA5 grid cell data and aggregates it spatially to generate data at the impact region level.

This part of the calculations can be skipped if the Carleton2022/data folder is copied entirely, as output data is already located there.
