import numpy as np
import pandas as pd
import xarray as xr
import glob, re, os, prism
from pathlib import Path
import statsmodels.api as sm
from patsy import dmatrices
import itertools, math
import matplotlib.pyplot as plt



def LoadMortality(wdir, filename, region_type, region, t_type, cause, age_group, variable): 
    
    files = wdir + "/" + filename + ".nc"

    ds = xr.open_mfdataset(files)
    
    filters = {
    "region_type":region_type,
    "region":region,
    "t_type":t_type,
    "age_group":age_group
    }
    
    if "cause" in ds.variables:
        filters["cause"] = cause

    da_selected = ds.set_index(geo=["region_type", "region"]).sel(**filters)[variable]
    
    return da_selected



def ImportGDPpc(rel_path, scenarios, region):
    
    """
    Read GDPpc data from TIMER output files from the selected scenario and project.
    """
    
    # Define dimensions and timeline for the xarray dataset
    _DIM_TIME = dict(start=1971, end=2100, stepsize=1)
    
    # Define timeline
    Timeline = prism.Timeline(
        start=_DIM_TIME['start'],
        end=_DIM_TIME['end'],
        stepsize=_DIM_TIME['stepsize']
        )
    
    # Define order of TIMER regions
    _DIM_IMAGE_REGIONS = [
        "CAN", "USA", "MEX", "RCAM", "BRA",
        "RSAM", "NAF", "WAF", "EAF", "SAF",
        "WEU", "CEU", "TUR", "UKR", "STAN",
        "RUS", "ME", "INDIA", "KOR", "CHN",
        "SEAS", "INDO", "JAP", "OCE", "RSAS",
        "RSAF"
        ]



    gdp = {}

    for scenario in scenarios:
        
        gdp_dir = rel_path.format(scenario=scenario)

        # Add extra regions depending on the file extension (always check if order is right with new files)
        extra_regions = ["dummy", "World"] if gdp_dir[-3:] in ["scn", "dat"] else ["World"]
        prism_regions_world = prism.Dimension('region', _DIM_IMAGE_REGIONS + extra_regions)
        
        listy = []
        
        VAR = "GDPpc"
        
        # Create xarray dataset with the data from the OUT files. 
        datafile = prism.TimeVariable(
                timeline=Timeline,
                dims=[prism_regions_world],
                file=gdp_dir,
            )
        
        listy.append(
            xr.merge(
                [
                    datafile[i]
                    .rename('Value')
                    .expand_dims({"Time": [i]}) for i in np.arange(_DIM_TIME['start'], 2101)
                    ]
                )
            .expand_dims({"Scenario": [scenario], "Variable": [VAR]})
            )
        
        xr_vars = xr.merge(listy)
        
        gdp[scenario] = (
            xr_vars
            .sel(Time=slice(2000,2100))
            .sel(region=region)
            .to_dataframe()
            .reset_index()
            .rename(columns={"Time":"year", "Value":"gdppc", "Scenario":"scenario"})
            .drop(columns=["Variable", "region"])
        )
 
    return gdp


def ImportPopulation(wdir, region, age_group):
    
    years = range(2000,2101)
    
    region_class = pd.read_csv(
        str(Path(wdir).parents[3]) +
        f"/data/RegionClassification/region_classification.csv"
        )[["hierid", "ISO3", "IMAGE26"]].iloc[:24378].rename(columns={"IMAGE26":"IMAGE"})
    
    pop = {}
    
    for ssp in ["ssp1", "ssp2", "ssp3", "ssp5"]:
        
        pop_ssp_all = None
        age_groups = ["oldest", "older", "young"] if age_group == 'All ages' else [age_group]

        for group in age_groups:
            pop_ssp_group = (
                pd.read_csv(
                    str(Path(wdir).parents[1]) +
                    f"/data/Population/PopulationIMAGE/pop_{ssp.lower()}_{group}.csv")
                .pipe(
                    lambda df: df.filter(
                    ["hierid"] +
                    [c for c in df.columns if c.isdigit() and int(c) in years]
                ))
                .set_index("hierid")
                .pipe(lambda df: df.set_axis(df.columns.astype(int), axis=1))
            )
            
            if pop_ssp_all is None:
                pop_ssp_all = pop_ssp_group
            else:
                pop_ssp_all = pop_ssp_all.add(pop_ssp_group, fill_value=0)
                    
        if region == "World":
            pop_ssp_all = pop_ssp_all.sum(axis=0).to_frame().reset_index().rename(columns={0:"population", "index":"year"})
            
        pop[ssp] = pop_ssp_all
        
    return pop



def GenerateDataframe4Shapley(wdir, gdp_dir, scenarios, region, age_group, variable):
    
    """
    Read in the uncertainty runs from the temperature-mortality model and concat all scenarios
    in a single dataframe, including the predictors from the Shapley-Owen decompostion. Save
    the dataframe as parquet file and return it in the function.
    """

    # Import annual GDPpc for all scenarios from TIMER
    gdp = ImportGDPpc(gdp_dir, scenarios, region)
    # Import annual population for all scenarios from IMAGE-Land
    pop = ImportPopulation(wdir, region, age_group)

    # All output files from the temperature-mortality model
    file_list = sorted(glob.glob(wdir+"*.nc"))

    # List to append mortality and predictors data
    final_list = []

    for i in range(len(file_list)):
        
        print(i)
        
        # Complete filename

        filename = re.search(r'([^\\/]+)\.nc$', file_list[i]).group(1)
        # Complete scenario name
        scenario = re.search(r'uncertainty_(.*?)_ssp', filename).group(1)
        # Climate variability filename
        variability = re.search(rf'{scenario}_(.*?)_2000', filename).group(1)
        # ERF draw name
        draw = re.search(r'2000-2100_TRAP_(.*)', filename).group(1)
        # SSP of the scenario
        ssp = scenario[:4]
        # Climate target of the scenario
        climate = scenario[5:]
        
        # Load mortality file i given a region, temperature type and age group
        cause=None
        t_type="heat"
        region_type="IMAGE"
        ds = LoadMortality(wdir, filename, region_type, region, t_type, cause, age_group, variable)
        
        # Conver to dataframe and drop useless columns
        ds = ds.to_dataframe().reset_index().drop(columns=["t_type", "geo", "region_type", "region"])
        
        # Merge with gdp data (no row should be excluded)
        ds = ds.merge(gdp[scenario], on="year", how="left")
        # Assign variability label
        ds["variability"] = variability
        # Assign ERF draw label
        # ds["erf_draw"] = draw
        # Assign SSP label
        ds["ssp"] = ssp
        # Assign climate target label
        ds["climate"] = climate
        # Merge with pop data (no row should be excluded)
        ds = ds.merge(pop[ssp.lower()], on="year", how="left")        
        
        intensity_erf = ([114.37473976, 143.82825596, 162.47447805, 176.7048847, 189.55993789, 202.39755928, 215.05416884, 229.89455102, 248.84048401, 281.24544739])
        ds["erf_draw"] = intensity_erf[int(draw)]
        
        # Append in list
        final_list.append(ds)
    
    # Concat all dataframes thorugh axis 0 generating a single dataframe with all scenarios and predictors
    all_results = pd.concat(final_list, axis=0, ignore_index=True)

    # Save cleaned dataframe as parquet
    all_results.to_parquet(wdir + f"data4shapley_{region}_{age_group}.parquet", index=False)
    
    return all_results


        
def ComputeRegressionsAndStats(data, year, target_variable, predictors):
    
    """
    Compute the OLS regressions and statistics of year
    """

    # Filter data for selected year
    df_year = data.loc[data['year'] == year]

    # Descriptive Stats -------------------------------------------------------
    
    stats = {}
    
    # Generate the basic descriptive stats for the dependent variable for each year
    stats["avg"]=np.average(df_year[target_variable])
    stats["median"]=np.median(df_year[target_variable])
    stats["StDev"]=np.std(df_year[target_variable])
    
    # Generate the 75th and 90th percentile for the dependent variable for each year
    stats["percentile_90"]=np.percentile(df_year[target_variable], 90)
    stats["percentile_75"]=np.percentile(df_year[target_variable], 75)
    
    
    # Regressions ------------------------------------------------------------
    
    # Dict to store R2
    r2_results = {(): 0.0} 
    
    n = len(predictors)
    
    # For loops go trhough all variable combinations
    for r in range(1, n + 1):
        for combo in itertools.combinations(predictors, r):
            
            flattened_combo = []
            for item in combo:
                if isinstance(item, tuple):
                    flattened_combo.extend(item)
                else:
                    flattened_combo.append(item)
            
            # Sort alphabetically the predictors names
            sorted_combo = tuple(sorted(list(flattened_combo)))
            
            # print(f"R2 for {sorted_combo}")

            # Build Patsy formula joining predictors using +
            patsy_formula = f"{target_variable} ~ {" + ".join(sorted_combo)}"
            
            try:
                # Generate matrices from Patsy and do Ordinary Least Square regresion
                y, X = dmatrices(patsy_formula, data=df_year, return_type='dataframe')
                model = sm.OLS(y, X)
                results = model.fit()
                
                # Save R2 in dictionary
                r2_results[sorted_combo] = results.rsquared
                
            except Exception as e:
                print(f"Error for combo {sorted_combo}: {e}")
                r2_results[sorted_combo] = None
                
    return stats, r2_results
        
        
        
def CalculateShapleyDecomposition(r2_results, predictors):
    
    
    def flatten_tuple(combo):
        flattened = []
        for item in combo:
            if isinstance(item, tuple):
                flattened.extend(item)
            else:
                flattened.append(item)
        return tuple(sorted(flattened))
    
    n = len(predictors)
    shapley_values = {}
    
    # Iterate for all predictors
    for v in predictors:
        
        # Get the rest of the variables
        other_vars = [x for x in predictors if x != v]
        # Initialize the shapley value with 0
        phi_v = 0.0
        
        # Iterate over the possible subsets
        for s_size in range(0,n):
            
            # Iterate over all combinations
            for comb in itertools.combinations(other_vars, s_size):
                
                # print(f"Shapley for {v} - subset size {s_size} - combination {comb}")
                
                # Sort subsets
                subset_with = flatten_tuple(comb + (v,))
                subset_without = flatten_tuple(comb)
                
                # Calculate R2
                r2_with = r2_results.get(subset_with, 0.0)
                r2_without = r2_results.get(subset_without, 0.0)
                
                # Calculate marginal contribution
                marginal_contribution = r2_with - r2_without
                
                # Calculate the regression contribution weighted by the number of permutations
                weight = (math.factorial(s_size) * math.factorial(n - 1 - s_size)) / math.factorial(n)
                
                # Aggregate to the Shaley-Owen value
                phi_v += marginal_contribution * weight
                
        # Save in dictionary the final Shapley value of variable v
        v_name = "_".join(v) if isinstance(v, tuple) else v
        shapley_values[f"R2_{v_name}_Shapley"] = phi_v

    # Get R2 of the whole model using all preictors together
    r2_full_model = r2_results.get(flatten_tuple(predictors), 0.0)
    
    # Calculate the residual of the Shapley values (what's not explained by them)
    shapley_values["R2_Residual"] = 1.0 - r2_full_model
    
    total_shapley_sum = r2_results.get(flatten_tuple(predictors), 0.0)
    
    # Shapley test indicates sum of all Shapley values must EQUAL the model R2
    test_validation = r2_full_model - total_shapley_sum

    shapley_values["Total_R2_Shapley"] = total_shapley_sum
    shapley_values["R2_Full_Model"] = r2_full_model
    shapley_values["Test_R2_Shapley_vs_R2"] = test_validation
    
    return shapley_values
        
        

def BuildShapleyRow(variable_name, year, stats, r2_results, shapley_values, predictors):
    
    """
    Build needed rows for plotting
    """
    
    # 1. Información Básica y Estadísticas Descriptivas (Bloque A)
    row = {
        'Variable': variable_name,
        'Year': year,
        'Average': stats.get('avg'),
        'Median': stats.get('median'),
        'StDev': stats.get('StDev'),
        'percentile_90': stats.get('percentile_90'),
        'percentile_75': stats.get('percentile_75'),
        'Avg-SD': stats.get('avg') - stats.get('StDev')
    }
    
    # Add all R2 combinations
    for combo, r2_val in r2_results.items():
        if combo == ():
            continue
        combo_name = "_".join(combo)
        row[f"R2_{combo_name}"] = r2_val
        
    # Add R2 of the whole model, residal and validations
    row['R2_Residual'] = shapley_values.get('R2_Residual')
    
    # Add total contributions of Shapley values
    for v in predictors:
        v_name = "_".join(v) if isinstance(v, tuple) else v
        row[f"R2_{v_name}_Shapley"] = shapley_values[f"R2_{v_name}_Shapley"]
    
    row['R2_Full_Model'] = shapley_values.get('R2_Full_Model')
    row['R2_Shapley_Sum'] = shapley_values.get('Total_R2_Shapley')
    row['Test_R2_Shapley_vs_R2'] = int(round(shapley_values.get('Test_R2_Shapley_vs_R2', 0)))
    
    row['Plot_Avg_SD'] = row["Avg-SD"] if row["Avg-SD"]>0 else 0
    row['Plot_Avg_SD_Surrogate'] = row["Avg-SD"] if row["Avg-SD"]<0 else 0
    row['Plot_Resid'] = (2 * row["StDev"] * row['R2_Residual'])
    
    # Caclulate Avg vs Midpoint
    Low_end = row['Plot_Avg_SD'] + row['Plot_Avg_SD_Surrogate']
    
    SD2=0
    for v in predictors:
        v_name = "_".join(v) if isinstance(v, tuple) else v
        row[f"Plot_{v_name}"] = 2 * row["StDev"] * shapley_values.get(f'R2_{v_name}_Shapley', 0)
        SD2+=row[f"Plot_{v_name}"]
    High_end = SD2 + row['Plot_Avg_SD'] + row['Plot_Avg_SD_Surrogate'] + row["Plot_Resid"]
    Length = High_end + Low_end
    Mid_point = Length / 2
    Avg_vs_Mid_point = row["Average"] - Mid_point # which must equal to zero in the final table
    
    row['Avg_vs_Mid_point'] = int(round(Avg_vs_Mid_point))
    
    row["R2_Full_Model_sum"] = shapley_values.get('R2_Full_Model')
    
    for v in predictors:
        v_name = "_".join(v) if isinstance(v, tuple) else v
        row[f"R2_{v}_Shapley_norm"] = shapley_values[f"R2_{v_name}_Shapley"] / row["R2_Full_Model_sum"]

    return row



def ComputeShapleyOwen(wdir, gdp_dir, scenarios, region, age_group, variable, predictors):
    
    
    # Generate a clean dataframe of the mortality and its predictors
    data = GenerateDataframe4Shapley(wdir, gdp_dir, scenarios, region, age_group, variable)
    
    # data = replicateK(variable=variable)
    
    df = []
    
    for year in range(2010,2101):
        
        print(year)
        
        # Compute statistics of selected year and R2 values from the OLS regressions built using all predictors combinations
        stats, r2_variables = ComputeRegressionsAndStats(data, year, variable, predictors)
        
        # Compute Shapley values for each variable
        shapley_values = CalculateShapleyDecomposition(r2_variables, predictors)
        
        row = BuildShapleyRow(variable, year, stats, r2_variables, shapley_values, predictors)
        
        df.append(row)

    df = pd.DataFrame(df)
    
    colours = {
    'Plot_Avg_SD':'#FF000000', # note that the colour here is transparent on purpose
    'Plot_Avg_SD_Surrogate':'C0', # note that the colour here is identical to Plot_Amb below
    'Plot_gdppc':'C0', 
    'Plot_population':'C1', 
    'Plot_climate':'C2', 
    'Plot_variability':'C3',
    "Plot_erf_draw": "C4",
    "Plot_Resid":"C5" 
    }

    fig, ax = plt.subplots()
    ax = df[['Year', 'Plot_Avg_SD', 'Plot_Avg_SD_Surrogate'] + 
        [f"Plot_{p}" for p in predictors] + 
        ['Plot_Resid']].\
        set_index('Year').\
        plot(
        kind='bar',
        stacked=True, 
        color=colours,
        edgecolor = "none",
        width=0.45,
        title='Shapley-Owen Decomposition', 
        ax=ax
        )
    ax.legend(
        title="Predictors", 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left'
    )
    df[['Average']].\
        plot(
            kind='line', 
            marker = 'd', 
            color='black', 
            ax=ax,
            label=None
            )

    # The line plot for the Median
    df[['Median']].\
        plot(
            kind='line', 
            linestyle='--', 
            color='r', 
            ax=ax,
            label=None
        )
        
    fig.tight_layout()
    plt.show() 




def replicateK(variable):
    
    wdir = "C:/Users/liprandicn/Downloads/"
    ssps_df = pd.read_csv(wdir+"df_ssps_cleaned.csv")
    
    
    # List of dependent variable that are of interest for this analysis
    variables = [
       # Primary Energy      
       'Primary_Energy', 
       'Primary_Energy_Wind',
       'Primary_Energy_Solar', 
       'Primary_Energy_Biomass',
       'Primary_Energy_Biomass_w_CCS', 
       'Primary_Energy_Hydro',
       'Primary_Energy_Nuclear', 
       'Primary_Energy_Geothermal', 
       'Primary_Energy_Non_Biomass_Renewables',
       'Primary_Energy_Fossil',
       "Primary_Energy_Fossil_w_CCS",
       "Primary_Energy_Fossil_wo_CCS",
       'Primary_Energy_Coal',
       'Primary_Energy_Coal_w_CCS',
       'Primary_Energy_Coal_wo_CCS',
       'Primary_Energy_Oil',
       'Primary_Energy_Oil_w_CCS',
       'Primary_Energy_Oil_wo_CCS',
       'Primary_Energy_Gas',
       "Primary_Energy_Gas_w_CCS",
       "Primary_Energy_Gas_wo_CCS",

       # Final Energy
       'Final_Energy', 
       'Final_Energy_Electricity', 
       'Final_Energy_Gases',
       'Final_Energy_Heat', 
       'Final_Energy_Liquids', 
       'Final_Energy_Solids',
       'Final_Energy_Hydrogen', 
       'Final_Energy_Solar', 
       'Final_Energy_Industry',
       'Final_Energy_Residential_and_Commercial',
       'Final_Energy_Transportation',
       'Final_Energy_Electrification',
       
       # Emissions
       'Emissions_CO2',
       'Emissions_CH4',
       'Emissions_N2O', 
       'Emissions_Kyoto_Gases',
       'Emissions_CO2_Carbon_Capture_and_Storage',
       'Emissions_CO2_Carbon_Capture_and_Storage_Biomass',
       'Emissions_CO2_Fossil_Fuels_and_Industry', 
       'Emissions_CO2_Land_Use',

       # Economic variables 
       'Price_Carbon',
       'GDP_PPP',
        
       ]
    
    # Reorder SSPs to start with SSP2 such that the combination of AIM/CGE (model) + SSP2 (scenario) 
    # is used as the constant in the regressions, as per the analysis of this paper
    ssps_df['scenario'] = pd.Categorical(ssps_df['scenario'], ['SSP2', 'SSP1', 'SSP3', 'SSP4', 'SSP5'])
    
    data = ssps_df.dropna(subset=[variable])
    
    return ssps_df



        
scenarios =  [
    "SSP2_M_CP_ERA_NoImpacts",
    "SSP2_M_CP_ERA_AllImpacts",
    "SSP2_M_CP_ERA_NoEcon",
    "SSP2_M_CP_Default_NoEcon",
    "SSP1_M_CP_ERA_AllImpacts",
    "SSP1_M_CP_ERA_NoImpacts",
    "SSP1_M_CP_ERA_NoEcon",
    "SSP1_ML_ERA_NoImpacts",
    "SSP2_ML_ERA_NoImpacts",
    "SSP1_ML_ERA_AllImpacts",
    "SSP2_ML_ERA_AllImpacts",
    "SSP1_ML_ERA_NoEcon",
    "SSP2_ML_ERA_NoEcon",
    "SSP1_VLLO_ERA_NoImpacts",
    "SSP2_VLLO_ERA_NoImpacts",
    "SSP1_VLLO_ERA_AllImpacts",
    "SSP2_VLLO_ERA_AllImpacts"
]
predictors = ["variability", "gdppc", "population", "climate", "erf_draw"] #[("cumulative_emissions", "squared_cumulative_emissions"), "model", "scenario"]
variable="mortality"#"Primary_Energy"
age_group = "All ages"
region = "World"
gdp_dir =  "X:/user/dekkerm/IMAGE_environments/IMPACTS/2_TIMER/outputlib/TIMER_3_5/IMPACTS/{scenario}/indicators/Economy/GDPpc_incl_impacts.out"
wdir = "X:\\user\\liprandicn/Projects\\mt-comparison\\models/Carleton2022/output/SPARCCLE_uncertainty\\"

ComputeShapleyOwen(wdir, gdp_dir, scenarios, region, age_group, variable, predictors)