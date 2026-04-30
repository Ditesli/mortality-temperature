import xarray as xr
import pandas as pd
import os



def LoadRegionClassificationDicts(wdir):
    
    """
    Load regions name from GBD and its corresponding ISO3 country code
    and IMAGE26 region name as dictionaries to map locations later on.
    """
        
    # Move one level up to access all-model data and region classification file
    wdir_up = os.path.dirname(wdir)
    
    # Create dictionaries to map location ids to ISO3 codes
    region_names = (
        pd.read_csv(f"{wdir_up}/data/region_classification.csv")
        [["gbd_location_id", "ISO3"]]
        .drop_duplicates()
        .dropna()
    )
    
    region_dict = dict(zip(
        region_names["gbd_location_id"].astype(int), 
        region_names["ISO3"]))
    
    # Dictionary to map location ids to IMAGE region names
    region_names = (
        pd.read_csv(f"{wdir_up}/data/region_classification.csv")
        [["IMAGE26", "ISO3"]]
        .drop_duplicates()
        .dropna()
    )
    
    image_dict = dict(zip(  
        region_names["ISO3"],
        region_names["IMAGE26"]))
    
    return region_dict, image_dict



def AggCoordElementsXarray(array, coord, old_elems, new_elem, exclude):

    """
    Aggregate elements of a coordinate in an xarray by summing the 
    values across the specified dimension.
    """

    agg_xarray = (
        array
        .where(array[coord].isin(old_elems), drop=True)
        .sum(dim=coord)
        .assign_coords({coord: new_elem})
    )
    
    if exclude==True:
        xarray = array.where(~array[coord].isin(old_elems), drop=True)
    else:
        xarray = array
                             
    new_xarray = xr.concat(
        [xarray, 
        agg_xarray], 
        dim=coord
    )
    
    return new_xarray



def LoadGBDmortality(sets, fls, causes, model):
    
    """
    Load GBD mortality data for the selected causes and age groups. 
    The data is filtered to include only the years included in the model run 
    and to exclude global mortality data. The data is then converted to
    an xarray for optimized calculations of relative and total mortality.
    """
    
    # Load GBD mortality records
    gbd_mor = pd.read_csv(f"{os.path.dirname(sets.wdir)}/data/GBD_mortality/IHME-GBD_2022_DATA.csv")

    mask = (
            gbd_mor["cause_name"].isin(causes) & # Only selected causes of death
            (gbd_mor["sex_name"] == "Both") & # Both sexes
            gbd_mor["year"].isin(sets.years) & # Only years assessed
            (gbd_mor["location_name"] != "Global") # Exclude global mortality
        )

    # Mask and convert to xarray
    gbd_mor = (
        gbd_mor.loc[mask, ["location_id","cause_name","age_name","year","val"]]
        .rename(columns={"age_name": "age_group", "location_id":"ISO3", "cause_name":"cause"})
        .set_index(["ISO3","cause","age_group","year"])
        .to_xarray()
    )

    if model == "Burkart":
        
        # Create "oldest" age group composed by people over 65 years
        oldest_group = [f"{year}-{year+4} years" for year in range(65,85,5)] + ["85+ years"]
        gbd_mor = AggCoordElementsXarray(gbd_mor, "age_group", oldest_group, "oldest", exclude=True)
        
        # Remove other age groups that are not included in the analysis
        gbd_mor = gbd_mor.where(~gbd_mor.coords["age_group"].isin([c for c in gbd_mor.age_group.values if "years" in c]), drop=True)
    
    if model == "Honda":
        
        oldest_group = [f"{year}-{year+4} years" for year in range(65,85,5)] + ["85+ years"]
        gbd_mor = AggCoordElementsXarray(gbd_mor, "age_group", oldest_group, "oldest", exclude=True)
        
        # Remove other age groups that are not included in the analysis
        gbd_mor = gbd_mor.where(~gbd_mor.coords["age_group"].isin([c for c in gbd_mor.age_group.values if " " in c]), drop=True)

    
    if model == "Scovronick":
        
        # Merge respiratory causes of death and age_groups into a single category
        rsp_causes = ["Chronic respiratory diseases", "Respiratory infections and tuberculosis"]
        oldest_65_group = [f'{year}-{year+4} years' for year in range(65,75,5)]
        rr_40_group = [f'{year}-{year+4} years' for year in range(30,45,5)]
        rr_55_group = [f'{year}-{year+4} years' for year in range(45,60,5)]
        rr_70_group =  [f'{year}-{year+4} years' for year in range(60,75,5)]
        rr_85_group = [f"{year}-{year+4} years" for year in range(75,85,5)] + ["85+ years"]

        gbd_mor = AggCoordElementsXarray(gbd_mor, "cause", rsp_causes, "Respiratory diseases", exclude=True)
        gbd_mor = AggCoordElementsXarray(gbd_mor, "age_group", oldest_65_group, "65", exclude=False)
        gbd_mor = AggCoordElementsXarray(gbd_mor, "age_group", rr_40_group, "40", exclude=True)
        gbd_mor = AggCoordElementsXarray(gbd_mor, "age_group", rr_55_group, "55", exclude=True)
        gbd_mor = AggCoordElementsXarray(gbd_mor, "age_group", rr_70_group, "70", exclude=True)
        gbd_mor = AggCoordElementsXarray(gbd_mor, "age_group", rr_85_group, "85", exclude=True)
        
        # Remove other age groups that are not included in the analysis
        gbd_mor = gbd_mor.where(~gbd_mor.coords["age_group"].isin([c for c in gbd_mor.age_group.values if " " in c]), drop=True)

        # Calculate Non-cardiorespiratory causes
        gbd_ncrc = (
            gbd_mor.sel(cause="All causes") 
            - gbd_mor.sel(cause="Cardiovascular diseases")
            - gbd_mor.sel(cause="Respiratory diseases")
        ).assign_coords(cause="Non-cardiorespiratory diseases")

        gbd_mor = xr.concat([gbd_mor, gbd_ncrc], dim='cause')

    # Map location ids to ISO3 codes
    gbd_mor["ISO3"] = xr.DataArray(
        [fls.region_dict[id] for id in gbd_mor['ISO3'].values], 
        coords=gbd_mor['ISO3'].coords, 
        dims=gbd_mor['ISO3'].dims
    ).astype(object)
    
    # Sort by age group, ISO3 and cause
    gbd_mor = gbd_mor.sortby("age_group").sortby("ISO3").sortby("cause").sortby("year")
    
    return gbd_mor



def LoadUNpopulationData(sets, model):
    
    """
    Load UN population data for the selected years range and age groups.
    The data is filtered to include only the years included in the model run. 
    The data is then converted to an xarray for optimized calculations of 
    relative and total mortality. The 5-year age groups are aggregated into the 
    same age groups as the GBD mortality data and the PAF data to be able to merge
    the three datasets and calculate attributable mortality.
    """

    # Load UN population data
    un_pop = (
        pd.read_csv(os.path.dirname(sets.wdir)+"/data/un_population/unpopulation_dataportal.csv")
        [["Iso3", "Time", "Age", "Value"]] # Keep relevant columns
        .rename(columns={"Value": "pop", "Time": "year", "Age": "age_group", "Iso3": "ISO3"})
        .set_index(["ISO3", "year", "age_group"]) 
        .to_xarray()
        .sel(year=slice(sets.years[0], sets.years[-1]))
    )

    if model == "Burkart":
        
        # Aggregate 5-year age groups into the same age groups as the other xarrays
        oldest_group = [f'{year}-{year+4}' for year in range(65,100,5)] + ["100+"]
        all_ages_group = [f'{year}-{year+4}' for year in range(0,100,5)] + ["100+"]
        
        un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=oldest_group, new_elem="oldest", exclude=False)
        un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=all_ages_group, new_elem="All ages", exclude=True)

        un_pop = un_pop.sortby("age_group").sortby("ISO3").sortby("year")
        
    if model == "Honda":

        # Aggregate 5-year age groups into the same age groups as the other xarrays
        oldest_group = [f'{year}-{year+4}' for year in range(65,100,5)] + ["100+"]
        un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=oldest_group, new_elem="oldest", exclude=True)
        
        # Drop age groups that are not included in the analysis (e.g. 0-4 years, 5-9 years, etc.)
        un_pop = un_pop.where(~un_pop.coords["age_group"].isin([c for c in un_pop.age_group.values if "-" in c]), drop=True)
        
    if model == "Scovronick":
        
        # Aggregate 5-year age groups into the same age groups as the other xarrays
        rr_40_group = [f'{year}-{year+4}' for year in range(30,45,5)]
        rr_55_group = [f'{year}-{year+4}' for year in range(45,60,5)]
        rr_65_group = [f'{year}-{year+4}' for year in range(65,75,5)]
        rr_70_group = [f'{year}-{year+4}' for year in range(60, 75, 5)]
        rr_85_group = [f"{year}-{year+4}" for year in range(75,100,5)] + ["100+"]

        un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=rr_85_group, new_elem="85", exclude=True)
        un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=rr_65_group, new_elem="65", exclude=False)
        un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=rr_70_group, new_elem="70", exclude=True)
        un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=rr_55_group, new_elem="55", exclude=True)
        un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=rr_40_group, new_elem="40", exclude=True)

        # Drop age groups that are not included in the analysis (e.g. 0-4 years, 5-9 years, etc.)
        un_pop = un_pop.where(~un_pop.coords["age_group"].isin([c for c in un_pop.age_group.values if "-" in c]), drop=True)

        un_pop = un_pop.sortby("age_group").sortby("ISO3")

    un_pop['pop'] = un_pop['pop'].where(un_pop['pop'] != 0)

    return un_pop



def PAF2Mortality(sets, fls, paf, causes, sn):
    
    # Load GBD mortality records
    gbd_mor = LoadGBDmortality(sets, fls, causes, sn.model)
    pop = LoadUNpopulationData(sets, sn.model)
    
    # Merge the three xarrays to have all data in the same format and coordinates
    paf_mor_pop = xr.merge([pop, gbd_mor, paf], join="outer") 
    
    if sn.model == "Scovronick":
         # Create "oldest" age group for model comparison
        paf_mor_pop = AggCoordElementsXarray(paf_mor_pop, "age_group", ["65", "85"],
                                             "oldest", exclude=False)

    
    ### ----------------------- ISO3 -------------------------
    
    # Calculate total mortality and relative mortality
    paf_mor_pop["mor"] = paf_mor_pop['paf'] * paf_mor_pop['val']
    
    if sn.model == "Burkart":
        paf_mor_pop = xr.concat([
            paf_mor_pop,
            paf_mor_pop.sum(dim="cause").assign_coords(cause="All causes")
            ],
            dim="cause")
    
    paf_mor_pop["rel_mor"] = paf_mor_pop["mor"] * 1e5 / paf_mor_pop["pop"]

    # Convert xarray to dataframe to save as csv files
    ProcessXarray2csv(sets, paf_mor_pop, "ISO3", sn)
    
     ### ----------------------- IMAGE -------------------------
     
    # Map location ids to IMAGE region names
    paf_mor_pop['ISO3'] = xr.DataArray(
        [fls.image_dict[id] for id in paf_mor_pop['ISO3'].values], 
        coords=paf_mor_pop['ISO3'].coords, 
        dims=paf_mor_pop['ISO3'].dims
        )

    # Aggregate mortality and population data by IMAGE region
    mor_image = paf_mor_pop.groupby("ISO3").sum().drop_vars(["paf", "rel_mor"])

    # Calculate global mortality and population
    mor_image = xr.concat([
        mor_image,
        mor_image.sum(dim='ISO3').assign_coords(ISO3="World")],
        dim='ISO3')

    # Calcualte relative mortality and PAF for IMAGE regions
    mor_image["rel_mor"] = mor_image["mor"] * 1e5 / mor_image["pop"]
    mor_image["paf"] = mor_image["mor"] / mor_image["val"]

    # Convert xarray to dataframe to save as csv files
    ProcessXarray2csv(sets, mor_image, "IMAGE", sn)



def ProcessXarray2csv(sets, data_array, regions, sn):
    
    """
    Convert the xarray with mortality data to a dataframe and save it as a csv file. 
    The xarray is pivoted to have the years as columns and the other coordinates as rows.
    The unit of the mortality data is added as a column. The resulting dataframe 
    is saved as a csv file in the output folder.
    """
    
    def ProcessMortalityData(data_array, unit_name):
    
        df = (
            data_array.to_dataframe()
            .reset_index()
            .pivot_table(
                index=['ISO3', 't_type', 'cause', 'age_group'],
                columns='year', 
                values=data_array.name)  # Uses the array name as the value column
            .reset_index()
            )
        df['units'] = unit_name
        
        return df
    
    mor = ProcessMortalityData(data_array["mor"], 'Total Mortality')
    rel_mor = ProcessMortalityData(data_array["rel_mor"], 'Relative Mortality')
    
    # Concatenate the results and save
    mor_rel_mor = pd.concat([
        mor, rel_mor], axis=0)[
        ['ISO3', 't_type', 'cause', 'age_group', 'units'] 
        + list(mor.columns[4:-1])
    ].rename(columns={"ISO3": "region"})
    
    if sn.model == "Scovronick":
        file_name = sn.years_part
    if sn.model == "Burkart":
        file_name = f"{sn.years_part}{sn.extrap_part}{sn.erf_part}"
    if sn.model == "Honda":
        file_name = f"{sn.years_part}{sn.extrap_part}_OT-{sets.optimal_range[-4:]}"
        
    mor_rel_mor.to_csv(f"{sn.out_path}/mortality_{sets.project}_{sets.scenario}_{regions}{file_name}.csv", index=False) 