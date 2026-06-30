import xarray as xr
import pandas as pd
import os



def LoadRegionClassificationDicts(wdir):
    
    """
    Load regions name from GBD and its corresponding ISO3 country code
    and IMAGE26 region name as dictionaries to map locations later on.
    """
    
    # Create dictionaries to map location ids to ISO3 codes
    region_names = (
        pd.read_csv(
            os.path.dirname(wdir) +
            f"/data/RegionClassification/region_classification.csv")
        [["gbd_location_id", "ISO3"]]
        .drop_duplicates()
        .dropna()
    )
    
    region_dict = dict(zip(
        region_names["gbd_location_id"].astype(int), 
        region_names["ISO3"]))
    
    # Dictionary to map location ids to IMAGE region names
    region_names = (
        pd.read_csv(os.path.dirname(wdir) +
            f"/data/RegionClassification/region_classification.csv")
        [["IMAGE26", "ISO3"]]
        .drop_duplicates()
        .dropna()
    )
    
    image_dict = dict(
        zip(  
            region_names["ISO3"],
            region_names["IMAGE26"]
            )
        )
    
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
    gbd_mor = pd.read_csv(f"{os.path.dirname(sets.wdir)}/data/GBD/Mortality/IHME-GBD_2023_DATA.csv")

    mask = (
        gbd_mor["cause_name"].isin(causes) & # Only selected causes of death
        (gbd_mor["sex_name"] == "Both") & # Both sexes
        gbd_mor["year"].isin(sets.years) & # Only years assessed
        (gbd_mor["location_name"] != "Global") # Exclude global mortality
    )

    # Mask and convert to xarray
    gbd_mor = (
        gbd_mor
        .loc[mask, ["location_id","cause_name","age_name","year","val","upper","lower"]]
        .rename(columns={"age_name": "age_group", "location_id":"ISO3", "cause_name":"cause"})
        .set_index(["ISO3","cause","age_group","year"])
        .to_xarray()
    )


    if model == "Burkart":
        
        # Create "oldest" age group composed by people over 65 years
        oldest_group = [f"{year}-{year+4} years" for year in range(65,85,5)] + ["85+ years"]
        gbd_mor = AggCoordElementsXarray(
            array=gbd_mor,
            coord="age_group",
            old_elems=oldest_group,
            new_elem="oldest",
            exclude=True
            )

        # Remove other age groups that are not included in the analysis
        gbd_mor = gbd_mor.where(
            ~gbd_mor.coords["age_group"]
            .isin([c for c in gbd_mor.age_group.values if "years" in c]), 
            drop=True
            )
    
    if model == "Honda":
        
        oldest_group = [f"{year}-{year+4} years" for year in range(65,85,5)] + ["85+ years"]
        gbd_mor = AggCoordElementsXarray(
            array=gbd_mor, 
            coord="age_group",
            old_elems=oldest_group,
            new_elem="oldest",
            exclude=True
            )
        
        # Remove other age groups that are not included in the analysis
        gbd_mor = gbd_mor.where(
            ~gbd_mor.coords["age_group"]
            .isin([c for c in gbd_mor.age_group.values if " " in c]),
            drop=True
            )

    
    if model == "Scovronick":
        
        # Merge respiratory causes of death and age_groups into a single category
        rsp_causes = [
            "Chronic respiratory diseases", 
            "Lower respiratory infections",
            "Upper respiratory infections"
            ]
        oldest_65_group = [f'{year}-{year+4} years' for year in range(65,75,5)]
        rr_40_group = [f'{year}-{year+4} years' for year in range(30,45,5)]
        rr_55_group = [f'{year}-{year+4} years' for year in range(45,60,5)]
        rr_70_group =  [f'{year}-{year+4} years' for year in range(60,75,5)]
        rr_85_group = [f"{year}-{year+4} years" for year in range(75,85,5)] + ["85+ years"]

        gbd_mor = AggCoordElementsXarray(
            array=gbd_mor,
            coord="cause",
            old_elems=rsp_causes,
            new_elem="Respiratory diseases",
            exclude=True
            )
        gbd_mor = AggCoordElementsXarray(
            array=gbd_mor,
            coord="age_group",
            old_elems=oldest_65_group,
            new_elem="65",
            exclude=False
            )

        for age, group in zip(["40", "55", "70", "85"], [rr_40_group, rr_55_group, rr_70_group, rr_85_group]):
            gbd_mor = AggCoordElementsXarray(
                array=gbd_mor,
                coord="age_group",
                old_elems=group,
                new_elem=age,
                exclude=True
                )
        
        # Remove other age groups that are not included in the analysis
        gbd_mor = gbd_mor.where(
            ~gbd_mor.coords["age_group"]
            .isin([c for c in gbd_mor.age_group.values if " " in c]),
            drop=True
            )

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
    
    # Rename mean mortality estimation
    gbd_mor = gbd_mor.rename({"val":"mean"})
    
    # Convert mortality uncertainty into coordinate
    gbd_mor = gbd_mor.to_array(dim="var_mor", name="gbd_mor")
    
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
        
        for old_elem, new_elem, boolean in zip([oldest_group, all_ages_group], ["oldest", "All ages"], [False, True]):
            un_pop = AggCoordElementsXarray(
                array=un_pop, 
                coord="age_group", 
                old_elems=old_elem, 
                new_elem=new_elem, 
                exclude=boolean
            )

        un_pop = un_pop.sortby("age_group").sortby("ISO3").sortby("year")
        
    if model == "Honda":

        # Aggregate 5-year age groups into the same age groups as the other xarrays
        oldest_group = [f'{year}-{year+4}' for year in range(65,100,5)] + ["100+"]
        un_pop = AggCoordElementsXarray(
            array=un_pop,
            coord="age_group",
            old_elems=oldest_group,
            new_elem="oldest",
            exclude=True
            )
        
        # Drop age groups that are not included in the analysis (e.g. 0-4 years, 5-9 years, etc.)
        un_pop = un_pop.where(
            ~un_pop.coords["age_group"]
            .isin([c for c in un_pop.age_group.values if "-" in c]),
            drop=True
            )
        
    if model == "Scovronick":
        
        # Aggregate 5-year age groups into the same age groups as the other xarrays
        rr_40_group = [f'{year}-{year+4}' for year in range(30,45,5)]
        rr_55_group = [f'{year}-{year+4}' for year in range(45,60,5)]
        rr_65_group = [f'{year}-{year+4}' for year in range(65,75,5)]
        rr_70_group = [f'{year}-{year+4}' for year in range(60,75,5)]
        rr_85_group = [f"{year}-{year+4}" for year in range(75,100,5)] + ["100+"]

        un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=rr_85_group, new_elem="85", exclude=True)
        un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=rr_65_group, new_elem="65", exclude=False)
        un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=rr_70_group, new_elem="70", exclude=True)
        un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=rr_55_group, new_elem="55", exclude=True)
        un_pop = AggCoordElementsXarray(array=un_pop, coord="age_group", old_elems=rr_40_group, new_elem="40", exclude=True)

        # Drop age groups that are not included in the analysis (e.g. 0-4 years, 5-9 years, etc.)
        un_pop = un_pop.where(
            ~un_pop.coords["age_group"]
            .isin([c for c in un_pop.age_group.values if "-" in c]),
            drop=True
            )

        un_pop = un_pop.sortby("age_group").sortby("ISO3")

    un_pop['pop'] = un_pop['pop'].where(un_pop['pop'] != 0)

    return un_pop



def PAF2Mortality(sets, fls, paf, causes, sn):
    
    ### ------------------- Merge files ----------------------------
    
    # Load GBD mortality records
    gbd_mor = LoadGBDmortality(sets, fls, causes, sn.model)
    pop = LoadUNpopulationData(sets, sn.model)
    
    # Merge the three xarrays to have all data in the same format and coordinates
    paf_mor_pop = xr.merge([pop, gbd_mor, paf], join="outer") 
    
    if sn.model == "Scovronick":
        # Create "oldest" age group for model comparison
        paf_mor_pop = AggCoordElementsXarray(
            paf_mor_pop, 
            "age_group", 
            ["65", "85"],
            "oldest",
            exclude=False
            )

    
    ### ----------------------- ISO3 -------------------------
    
    
    # Calculate total mortality and relative mortality
    paf_mor_pop["mor"] = paf_mor_pop['paf'] * paf_mor_pop['gbd_mor']
    
    if sn.model == "Burkart":
        # Calculate total mortality and relative mortality for "All causes" category
        total_causes = paf_mor_pop.sum(dim="cause")
        total_causes = total_causes.expand_dims(cause=["All causes"])
        paf_mor_pop = xr.concat([paf_mor_pop, total_causes], dim="cause")
    
    # Calculate relative mortality per 100,000 people
    paf_mor_pop["rel_mor"] = paf_mor_pop["mor"] * 1e5 / paf_mor_pop["pop"]
    
    image_results = paf_mor_pop.copy()

    # Drop pop and gbd mortality and prepare dataset for merging with multiindex
    paf_mor_pop = (
        paf_mor_pop
        .drop_vars(["pop", "gbd_mor"])
        .assign_coords(region_type=("ISO3", ["ISO3"] * len(paf_mor_pop.ISO3))) 
        .rename({"ISO3": "region"})
    )

    
    ### ----------------------- IMAGE -------------------------
    
     
    # Map location ids to IMAGE region names
    image_results['ISO3'] = xr.DataArray(
        [fls.image_dict[id] for id in image_results['ISO3'].values], 
        coords=image_results['ISO3'].coords, 
        dims=image_results['ISO3'].dims
        )

    # Aggregate mortality and population data by IMAGE region
    image_results = image_results.groupby("ISO3").sum().drop_vars(["paf", "rel_mor"])

    # Calculate global mortality and population
    image_results = xr.concat([
        image_results,
        image_results.sum(dim='ISO3').assign_coords(ISO3="World")],
        dim='ISO3'
        )

    # Calcualte relative mortality and PAF for IMAGE regions
    image_results["rel_mor"] = image_results["mor"] * 1e5 / image_results["pop"]
    image_results["paf"] = image_results["mor"] / image_results["gbd_mor"]

    image_results = (
        image_results
        .drop_vars(["pop", "gbd_mor"])
        .assign_coords(region_type=("ISO3", ["IMAGE"] * len(image_results.ISO3))) 
        .rename({"ISO3": "region"})
    )
    
    ### ---------------------- Save Final Dataframe ----------------------------
    
    final_results = (
        xr.concat([paf_mor_pop, image_results], dim="region", join="outer")
        .set_index(geo=["region", "region_type"])
        .rename({"mor":"mortality", "rel_mor":"relative_mortality"})
    )
    
    # Create file name based on model and scenario characteristics
    if sn.model == "Scovronick":
        file_name = f"{sn.years_part}{sn.counter}"
    if sn.model == "Burkart":
        file_name = f"{sn.years_part}{sn.extrap_part}{sn.erf_part}{sn.counter}{sn.draw}"
    if sn.model == "Honda":
        file_name = f"{sn.years_part}{sn.extrap_part}{sn.counter}"   
        
    compresion_config= {
                "dtype": "float32",    # Save as float32 for las memory usage
                "zlib": True,
                "complevel": 6,
    } 

    encoding_total = {
        var: compresion_config for var in final_results.data_vars
    }
        
    final_results.reset_index("geo").to_netcdf(
        f"{sn.out_path}/mortality_{sets.project}_{sets.scenario}{file_name}.nc",
        encoding=encoding_total
    )