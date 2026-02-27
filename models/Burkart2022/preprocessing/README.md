The preprocessing part serves to generate only once data used in the main model. This saves computation time. The preprocessing part can be skipped if the Burkart2022/data folder is copied entirely, as necessary input data is already located there.


## Temperature zones
Generate temperature zones at the grid cell level defined by Burkart et al. as the mean temperature from 1980-2019. Following their method, monthly mean temperatures from [ERA5 data](cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means) were used. A [sea-land mask from ERA5](confluence.ecmwf.int/pages/viewpage.action?pageId=140385202#ERA5Land:datadocumentation-parameterlistingParameterlistings) and a population mask that includes the cells with population data were used to keep only the cells with either population or land cells.
    

## GBD locations
The script generates raster files for the GBD locations at level 3 using the shapefile available at the [GBD Publication Figures Repository](https://github.com/matejmisik/gbd_publication_figures/tree/main/Figures/Global_map) (retrieved on 04/2025).


## TMRELs
TMRELs (Theoretical Minimum Risk Exposure Levels) as raster files.
