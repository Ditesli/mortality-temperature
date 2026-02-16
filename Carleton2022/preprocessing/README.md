The preprocessing part serves to generate only once data used in the main model. This saves computation time. The preprocessing part can be skipped if the Carleton2022/data folder is copied entirely, as necessary input data is already located there.

## Region Classification File
Generate the region classification file to include the impact region classification (24,378 regions) and their corresponding ISO3 code, IMAGE region and WHO region classification.

## Population Data
### Historical Data
Generate files with historical population for the years 2000 to 2022. The approach follows the one described in the original paper in [Appendix B.3.3](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/qje/137/4/10.1093_qje_qjac020/2/qjac020_online_appendix.pdf?Expires=1771852600&Signature=zks28d0U0f3punXjTfhiv7PzaVGiJ6xJU4p-6WjnjMhard~KiFvaTeNblPy8~P8xzu3pPSnJ-NMC1cJ2vdye4MdN5uSrHbCjCmEA4SMq4-7Ehg-2Nh4SARPOUXE3iEp1dmo86xI0qcd3SmTHr1nYLCIpSHHTlr9ng5g8nYp0QG5w4Nib531cR2ei2OCu~ThyXSp1Uk2TX1EisGGOXvYCFiWJFkjfySyNFWr54bxLS9pXV4yBOFPMYQJpwrr~4Z55k0SgD2pg6lD~hFmcei3kT8YgOdaLMuy3t5cgk~Qll8wCGW~6ELx8skVCkgdvVowzha1us8yHmIVeQsPUg-iUCA__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA).
To run the function, the code needs:

- Path to the main working directory where all output data from the preprocessing will be stored. This folder must contain a subfolder called <strong>carleton_sm</strong> with the following files from Carleton et al., (2022) suplemmentary materials (see their [Zenodo repository](https://zenodo.org/records/6416119)).

  - <em> carleton_sm\1_estimation\2_csvv\Agespec_interaction_response.csvv </em>
  - <em> data\2_projection\3_impacts\main_specification\raw\single\rcp85\CCSM4\low\SSP3\mortality-allpreds.csv </em>
     (Only present day (2015) covariates were saved to use less storage.)
  - <em> data\2_projection\2_econ_vars </em> (folder).
  - <em> data\2_projection\1_regions\ir_shp </em> (folder).

- Population data from [World Population Prospects 2024](https://population.un.org/wpp/) for the age groups (0-4, 5-14, 15-64, 65+) for the years 2000-2022. 
- Landscan population data available at [LandScan Global, 30 Arc-second Annual Global Gridded Population Datasets from 2000 to 2022](https://doi.org/10.1038/s41597-025-04817-z).

### Future projections
The code imports the population data produced by an IMAGE run disagregates it to the three age groups used in the paper and merge the data spatially to the impact region level. This is done for the relevant SSP scenarios. 

To disaggregate to age groups, the code uses the population data projections per age group from the SSP projections, available at [(K.C. et al. (2024))](https://dataexplorer.wittgensteincentre.org/wcde-v3/).


## ERA5 temperature data

### "Present-day" temperatures
“Present day” temperature are defined in the paper as T_0 and as the daily temperatures of the period 2001-2010. This code uses [ERA5](https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=overview) daily data and aggregates it spatially to generate data at the impact region level.

### Climatologies
Generate files with ERA5 climatology (defined here as the 30-year running mean) At the impact region level. This is used to generate Exposure Response Functions (ERFs) with adaptation.


