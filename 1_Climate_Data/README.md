This folder contains the necesary code to correct the bias of the model ensemble and generate the daily temperature at the impact region level.


### 1_Impact region temperature
This script requires importing the data from the Carleton et al.'s article, available at the [Harvard Economic Datasets website](https://hu.sharepoint.com/sites/HarvardEconomicsDatasets/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FHarvardEconomicsDatasets%2FShared%20Documents%2FCarleton%20et%20al%20%282022%29&p=true&ga=1).

The script requires to upload the climate data files (daily NSAT) in .nc format to calculate the mean temperature in each Carleton's impact region. The resulting data is saved in a .csv file. The code can be modified to process more than one file or more than one climate model. The climate model data from CMIP6 can be easily retrieved from the websites (among others): 
- [Copernicus CMIP6 Climate Projections](https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip6?tab=form)
- [Earth System Grid Federation ESGF Metagrid Node](https://aims2.llnl.gov/search)
- [Earth System Grid Federation Deutscher Wetterdienst Node](https://esgf.dwd.de/search/cmip6-dwd/)

### 2_Bias correction