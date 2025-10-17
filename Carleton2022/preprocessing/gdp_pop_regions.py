import utils


# Path to working directory where all data is stored
wdir = 'X:\\user\\liprandicn\\mt-comparison\\Carleton2022\\data\\'
# Path to IMAGE regions classification folder produced manually
regions_file = 'X:\\user\\liprandicn\\Health Impacts Model\\data\\IMAGE_regions\\regions_comparison.xlsx'
# Open LandScan population raster 
landscan_file = f'{wdir}'+'LandScan_Global/landscan-global-2000-assets/landscan-global-2000.tif'
# Open impact regions shapefile
impact_regions = f'{wdir}'+'carleton_sm/ir_shp/impact-region.shp'
    


### ----------------------------------------------------------------------
''' 
Generate files per scenario and age group for all impact regions 
'''

utils.gdp_pop_ssp_projections(wdir)
    
    
    
### ----------------------------------------------------------------------
''' 
Generate file that contains impact region codes, names and their corresponding 
IMAGE and GBD region
'''

utils.region_classification_file(wdir, 
                                 regions_file)


    
### ----------------------------------------------------------------------
''' 
Generate files that contains historical GDP and Population per impact region
'''

utils.generate_historical_pop(wdir, 
                              landscan_file, 
                              impact_regions)




