import pandas as pd
import numpy as np
import glob, re
import xarray as xr
import matplotlib.pyplot as plt


IMAGE_REGIONS = {
    # America
    "CAN":["Canada", 1],
    "USA":["USA", 2],
    "MEX":["Mexico", 3],
    "RCAM":["Central America", 4],
    "BRA":["Brazil", 5],
    "RSAM":["Rest of South America", 6],
    # Africa
    "NAF":["Northern Africa", 7],
    "WAF":["Western Africa", 8],
    "EAF":["Eastern Africa", 9],
    "SAF":["South Africa", 10], 
    # Europe
    "WEU":["Western Europe", 11], 
    "CEU":["Central Europe", 12], 
    "TUR":["Turkey", 13], 
    "UKR":["Ukraine region", 14],
    # Asia
    "STAN":["Central Asia", 15],
    "RUS":["Russia region", 16], 
    "ME":["Middle East", 17],
    "INDIA":["India", 18], 
    "KOR":["Korea region", 19], 
    "CHN":["China region", 20], 
    "SEAS":["Southeastern Asia", 21], 
    "INDO":["Indonesia region", 22], 
    "JAP":["Japan", 23],
    # Oceania + other
    "OCE": ["Oceania", 24],
    "RSAS": ["Rest of South Asia", 25],
    "RSAF": ["Rest of Southern Africa", 26]
    } 



# Full list of causes
causes = {
    'ckd':'Chronic kidney disease', 
    'cvd_cmp':'Cardiomyopathy and myocarditis', 
    'cvd_htn':'Hypertensive heart disease', 
    'cvd_ihd':'Ischemic heart disease', 
    'cvd_stroke':'Stroke', 
    'diabetes':'Diabetes mellitus',
    'inj_animal':'Animal contact', 
    'inj_disaster':'Exposure to forces of nature', 
    'inj_drowning':'Drowning', 
    'inj_homicide':'Interpersonal violence', 
    'inj_mech':'Exposure to mechanical forces', 
    'inj_othunintent':'Other unintentional injuries', 
    'inj_suicide':'Self-harm', 
    'inj_trans_other':'Other transport injuries', 
    'inj_trans_road':'Road injuries', 
    'resp_copd':'Chronic obstructive pulmonary disease', 
    'lri':'Lower respiratory infections'
    }



def LoadMortalityold(wdir, filename, years, region, temp_type, unit, age_group, cause, val_mor, val_erf):
    
    """
    Load mortality from ANY calculation method as time series and constrained to 
    the parameters:
    - years: Years range
    - region: IMAGE region or "World"
    - temp_type: "Heat", "Cold", "All"
    - unit: "relative" (Relative mortality) or "total" (total mortality)
    - age_group: "young", "older", "oldest" (only valid for Carleton)
    - cause: cause of death (only valid for Burkart method)
    """
    
    df = pd.read_csv(wdir + filename + ".csv")

    # Initialize filter as True to not filter anything initially
    filter = pd.Series(True, index=df.index)

    # Apply age_group condition only if the column exists
    if "age_group" in df.columns:
        filter &= df["age_group"].str.lower().str.contains(age_group.lower())
        
    # Apply age_group condition only if the column exists
    if "cause" in df.columns:
        filter &= df["cause"].str.lower().str.contains(cause.lower())

    # Condition for t_type
    filter &= df["t_type"].str.lower() == temp_type.lower()

    # Condition for region
    filter &= df["region"] == region
    
    # Condition for value
    if "val_mor" in df.columns:
        filter &= df["val_mor"].str.lower() == val_mor.lower()
        
    if "val_erf" in df.columns:
        filter &= df["val_erf"].str.lower() == val_erf.lower()

    # Condition for units
    if "units" in df.columns:
        filter &= df["units"].str.lower().str.contains(unit.lower())

    # Apply filter and select columns starting from the 5th column
    df = df[filter][[str(y) for y in years if str(y) in df.columns]]

    # Convert column names to int
    df.columns = df.columns.astype(int)
    
    return df



def LoadScatter(wdir, filename, years, temp_type, unit, age_group, cause):
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(wdir + filename + ".csv")

    # Initialize filter as True to not filter anything initially
    filter = pd.Series(True, index=df.index)

    # Apply age_group condition only if the column exists
    if "age_group" in df.columns:
        filter &= df["age_group"].str.lower().str.contains(age_group.lower())
        
    # Apply age_group condition only if the column exists
    if "cause" in df.columns:
        filter &= df["cause"].str.lower().str.contains(cause.lower())

    # Condition for t_type
    filter &= df["t_type"].str.lower() == temp_type.lower()

    # Condition for units
    if "units" in df.columns:
        filter &= df["units"].str.lower().str.contains(unit.lower())
        
    # Load region classification    
    region_class = (
        pd.read_csv(wdir+"data/region_classification.csv")
        [["ISO3", "continents"]]
        .drop_duplicates()
        )

    # Apply filter and select columns starting from the 5th column
    df = df[filter][["region"]+[str(y) for y in years if str(y) in df.columns]]

    # Merge with region classification and group by continents, summing the values
    df = df.merge(region_class, left_on="region", right_on="ISO3", how="left").groupby("continents").sum().drop(columns=["ISO3", "region"])
    
    # Add a row for the world total by summing all the continents
    df.loc["World"] = df.sum(numeric_only=True)
    
    # Convert column names to int
    df.columns = df.columns.astype(int)
    
    final_df = pd.DataFrame(index=df.index)
    final_df["mean"] = df.mean(axis=1)
    final_df["p95"] = df.quantile(0.95, axis=1)
    final_df["p5"] = df.quantile(0.05, axis=1)
    
    final_df.sort_index(inplace=True)
    
    return final_df



def LoadMortality(wdir, filename, region_type, region, t_type, cause, age_group, variable): 
    
    files = wdir + "/" + filename + ".nc"
    file_list = sorted(glob.glob(files))

    if file_list:
        ds = xr.open_mfdataset(file_list, combine="nested", concat_dim="draw")
    else:
        print("Files not found.")
    
    filters = {
    "region_type":region_type,
    "region":region,
    "t_type":t_type,
    "age_group":age_group
    }
    
    if "cause" in ds.variables:
        filters["cause"] = cause


    da_selected = ds.set_index(geo=["region_type", "region"]).sel(**filters)[variable]

    if "carleton" in filename.lower():
        dims = ["draw"]
    elif "burkart" in filename.lower() or "romanello" in filename.lower():
        dims = ["draw", "var_mor"]
    else:
        dims = ["draw", "var_mor", "var_erf"]

    da_mean = da_selected.mean(dim=dims)
    da_p025 = da_selected.quantile(0.025, dim=dims)
    da_p975 = da_selected.quantile(0.975, dim=dims)
    
    return da_mean, da_p025, da_p975



def LoadBallesterScatter(wdir):
    
    # Open ballester results
    ballester = pd.read_excel(wdir + "ballester2025/ballester2025_lancet_tab8.xlsx")
    
    # Opern region classification
    regions = pd.read_csv(wdir+"data/region_classification.csv")[["gbd_location_id", "UN_M49_level1"]].drop_duplicates().dropna()
    regions["gbd_location_id"] = regions["gbd_location_id"].astype(int)
    
    years = range(2012,2022)
    
    # Load GBD mortality records
    gbd_mor = pd.read_csv(wdir+f"/data/GBD_mortality/IHME-GBD_2022_DATA.csv")

    mask = (
            (gbd_mor["cause_name"] == "All causes") & # Only selected causes of death
            (gbd_mor["sex_name"] == "Both") & # Both sexes
            gbd_mor["year"].isin(years) & # Only years assessed
            (gbd_mor["location_name"] != "Global") & # Exclude global mortality
            (gbd_mor["age_name"] == "All ages") # All ages
        )

    # Mask and convert to xarray
    gbd_mor = (
        gbd_mor
        .loc[mask, ["location_id", "year","val", "upper", "lower"]]
        .groupby(["location_id"])
        .mean()
        .reset_index()
        .merge(regions, left_on="location_id", right_on="gbd_location_id", how="left")
        .groupby("UN_M49_level1")
        .sum()
        .reset_index()
        .merge(ballester[["region", "2012-2021"]], left_on="UN_M49_level1", right_on="region", how="inner")
    )
    
    # Calculate mortality for each region and confidence intervals
    gbd_mor["mean"] = gbd_mor["val"] * gbd_mor["2012-2021"] / 100
    gbd_mor["p5"] = gbd_mor["lower"] * gbd_mor["2012-2021"] / 100
    gbd_mor["p95"] = gbd_mor["upper"] * gbd_mor["2012-2021"] / 100
    
    gbd_mor = gbd_mor[["region", "mean", "p5", "p95"]].set_index("region")
    
    # Merge America regions into one
    gbd_mor.loc["America"] = gbd_mor.loc["Northern America"] + gbd_mor.loc["Latin America and the Caribbean"]
    gbd_mor.drop(["Northern America", "Latin America and the Caribbean"], inplace = True)
    
    # Get world total by summing all regions
    gbd_mor.loc["World"] = gbd_mor.sum(numeric_only=True)
   
    return gbd_mor.sort_index()



def StylizeAxes(ax, *, 
                xscale=None,
                yscale=None,
                ylim=None,
                xlim=None,
                title=None,
                title_kwargs=None,
                xlabel=None,
                ylabel=None,
                xticks=None,
                xtickslabels=False,
                xtickslabels_kwargs=None,
                yticks=None,
                ytickslabels=False,
                ytickslabels_kwargs=None,
                facecolor=None,
                grid=False,
                grid_kwargs=None,
                legend=False,
                legend_kwargs=None,
                spines=None,
                **kwargs):  
    
    if xscale is not None: ax.set_xscale(xscale)
    if yscale is not None: ax.set_yscale(yscale)
    
    if ylim is not None: ax.set_ylim(ylim)
    if xlim is not None: ax.set_xlim(xlim)
    
    if title is not None: ax.set_title(title, **(title_kwargs or {}))
    
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    
    if xticks is not None: ax.set_xticks(xticks) 
    if xtickslabels is not False: ax.set_xticklabels(xtickslabels, **(xtickslabels_kwargs or {}))
    
    if yticks is not None: ax.set_yticks(yticks) 
    if ytickslabels is not False: ax.set_yticklabels(ytickslabels, **(ytickslabels_kwargs or {}))
    
    if facecolor is not None: ax.set_facecolor(facecolor)
    if grid: ax.grid(**(grid_kwargs or {}))
    
    if legend: ax.legend(**(legend_kwargs or {}))
    
    if spines is not None:
        for spine, is_visible in spines.items():
            ax.spines[spine].set_visible(is_visible)
    
    return ax



def StylizePlot(*, 
                xscale=None,
                yscale=None,
                ylim=None,
                xlim=None,
                title=None,
                xlabel=None,
                ylabel=None,
                legend=True, 
                legend_kwargs=None, 
                grid=False,
                grid_kwargs=None,
                suptitle=None, 
                suptitle_kwargs=None, 
                xticks_kwargs=None,
                yticks_kwargs=None,
                tight_layout=None,
                facecolor=None,
                despine=None):
    if xscale:
        plt.xscale(xscale)
    if yscale:
        plt.yscale(yscale)
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)    
    if legend:
        plt.legend(**(legend_kwargs or {}))
    if grid:
        plt.grid(**(grid_kwargs or {}))
    if suptitle:
        plt.suptitle(suptitle, **(suptitle_kwargs or {}))
    if xticks_kwargs:
        plt.xticks(**xticks_kwargs)
    if yticks_kwargs:
        plt.yticks(**yticks_kwargs)
    if tight_layout:
        plt.tight_layout()
    if facecolor:
        plt.gca().set_facecolor(facecolor)
    if despine:
        ax = plt.gca()
        for spine_edge in despine:
            if spine_edge in ax.spines:
                ax.spines[spine_edge].set_visible(False)