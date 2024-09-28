This folder contains the necesary code to correct the bias of the model ensemble and generate the daily temperature at the impact region level. Some intermediate, small output files are included in the folder.


### 1_Impact region temperature.py
This script requires importing the data from the Carleton et al.'s article, available at the [Harvard Economic Datasets website](https://hu.sharepoint.com/sites/HarvardEconomicsDatasets/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FHarvardEconomicsDatasets%2FShared%20Documents%2FCarleton%20et%20al%20%282022%29&p=true&ga=1).

The script requires to upload the climate data files (daily NSAT) in .nc format to calculate the mean temperature in each Carleton's impact region. The resulting data is saved in a .csv file. The code can be modified to process more than one file or more than one climate model. The climate model data from CMIP6 can be easily retrieved from the websites (among others): 
- [Copernicus CMIP6 Climate Projections](https://cds.climate.copernicus.eu/datasets/projections-cmip6?tab=overview)
- [Earth System Grid Federation ESGF Metagrid Node](https://aims2.llnl.gov/search)
- [Earth System Grid Federation Deutscher Wetterdienst Node](https://esgf.dwd.de/search/cmip6-dwd/)

### 2_Bias correction.py
This script requires to use NSAT of "Present Day" from reanalysis data. This can be retrieved, for example, from the [ERA5 website](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview). 
Present day data, spanning here from 1990 to 2020 is corrected in every model using the following formula:

$$
T_{\text{corr}_{r,t,i}} = T_{r,t,i} + \Delta T \\
\text{with} \hspace{2mm} \Delta T = T_{ERA5} - T_{i} 
$$

And $r$ the grid cell, $t$ the year and $i$ the climate model. This means that each spatial and temporal unit of every climate model is added the $\Delta T$ to correct the bias that temperatures in the models can be consistently too low or too high. Please note that $\Delta T$ is a single value per model, meaning that it represents Global Surface Air Temperature GSAT for the mean 1990-2020.

Please note that the final function of this script requires a large capacity of storage.
