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
        draw = re.search(r'2000-2100_(.*)', filename).group(1)
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
        ds["erf_draw"] = draw
        # Assign SSP label
        ds["ssp"] = ssp
        # Assign climate target label
        ds["climate"] = climate
        # Merge with pop data (no row should be excluded)
        ds = ds.merge(pop[ssp.lower()], on="year", how="left")
        
        # Append in list
        final_list.append(ds)
    
    # Concat all dataframes thorugh axis 0 generating a single dataframe with all scenarios and predictors
    all_results = pd.concat(final_list, axis=0, ignore_index=True)

    # Save cleaned dataframe as parquet
    all_results.to_parquet(wdir + f"data4shapley_{region}_{age_group}.parquet", index=False)
    
    return all_results
        
        
        
def CalculateShapleyDecomposition(r2_results, predictors):
    
    n = len(predictors)
    shapley_values = {}
    
    # Iterate for all predictors variables
    for v in predictors:
        
        # Get the rest of the variables
        other_vars = [x for x in predictors if x != v]
        phi_v = 0.0
        
        # Iterate over the possible subsets
        for s_size in range(1,n):
            
            # Iterate over all combinations
            for comb in itertools.combinations(other_vars, s_size):
                
                # Sort subsets
                subset_without = tuple(sorted(comb))
                subset_with = tuple(sorted(comb + (v,)))
                
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
        shapley_values[f"R2_{v}_Shapley"] = phi_v

    # Get R2 of the whole model using all preictors together
    full_model_key = tuple(sorted(predictors))
    r2_full_model = r2_results.get(full_model_key, 0.0)
    
    # Calculate the residual
    shapley_values["R2_Residual"] = 1.0 - r2_full_model
    
    # Shapley test indicates sum of all Shapley values must EQUAL the model R2
    total_shapley_sum = sum(shapley_values[f"R2_{v}_Shapley"] for v in predictors)
    test_validation = r2_full_model - total_shapley_sum

    shapley_values["Total_R2_Shapley"] = total_shapley_sum
    shapley_values["R2_Full_Model"] = r2_full_model
    shapley_values["Test_R2_Shapley_vs_R2"] = test_validation
    
    return shapley_values
        
        
        
def ComputeRegressionsAndStats(data, year, target_variable, predictors):
    
    """
    Compute the OLS regressions and statistics of year
    """

    # Filter data for selected year
    df_year = data.loc[data['year'] == year]

    # Descriptive Stats -------------------------------------------------------
    
    stats = {}
    
    # Generate the basic descriptive stats for the dependent variable for each year
    stats["avg"]=np.average(df_year[variable])
    stats["median"]=np.median(df_year[variable])
    stats["StDev"]=np.std(df_year[variable])
    
    # Generate the 75th and 90th percentile for the dependent variable for each year
    stats["percentile_90"]=np.percentile(df_year[variable], 90)
    stats["percentile_75"]=np.percentile(df_year[variable], 75)
    
    
    # Regressions ------------------------------------------------------------
    
    # Dict to store R2
    r2_results = {(): 0.0} 
    
    n = len(predictors)
    
    # For loops go trhough all variable combinations
    for r in range(1, n + 1):
        for combo in itertools.combinations(predictors, r):
            
            # Sort alphabetically the predictors names
            sorted_combo = tuple(sorted(list(combo)))

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



def prepare_shapley_plot_columns(shapley_values, avg, stdev, predictors_list):
    """
    Generaliza los cálculos de desviaciones estándar ponderadas (SD2) y las 
    métricas de centrado para gráficos de barras apiladas de Shapley con N variables.
    
    Argumentos:
    - shapley_values: Diccionario devuelto por la función anterior (calculate_shapley_decomposition).
    - avg: Media de la variable dependiente para ese año/periodo (float).
    - stdev: Desviación estándar de la variable dependiente para ese año/periodo (float).
    - predictors_list: Lista con los nombres de las variables (ej. ['variability', 'gdppc', ...]).
    
    Retorna:
    - Un diccionario con todas las métricas de graficación preparadas.
    """
    plot_metrics = {}
    
    # 1. Calcular el límite base (Avg_SD)
    avg_sd = avg - stdev
    plot_metrics["Avg_SD"] = avg_sd
    
    # 2. Lógica de control para valores negativos (Surrogate)
    if avg_sd >= 0:
        plot_metrics["Plot_Avg_SD"] = avg_sd
        plot_metrics["Plot_Avg_SD_Surrogate"] = 0.0
    else:
        plot_metrics["Plot_Avg_SD"] = 0.0
        plot_metrics["Plot_Avg_SD_Surrogate"] = avg_sd
        
    # 3. Calcular dinámicamente las alturas de los efectos (2 * SD * R2_Shapley)
    # Procesa automáticamente tus 5 variables independientes
    total_sd2_predictors = 0.0
    for v in predictors_list:
        r2_shapley_key = f"R2_{v}_Shapley"
        r2_value = shapley_values.get(r2_shapley_key, 0.0)
        
        sd2_value = 2 * stdev * r2_value
        plot_metrics[f"SD2_{v}"] = sd2_value
        total_sd2_predictors += sd2_value
        
    # 4. Calcular la altura para el Residuo
    r2_residual = shapley_values.get("R2_Residual", 0.0)
    sd2_resid = 2 * stdev * r2_residual
    plot_metrics["SD2_Resid"] = sd2_resid
    
    # 5. Cálculos de centrado matemático sobre la media
    # Sumamos las alturas de todas las variables + el residuo + las bases de control
    total_stacks_height = total_sd2_predictors + sd2_resid
    
    low_end = plot_metrics["Plot_Avg_SD"] + plot_metrics["Plot_Avg_SD_Surrogate"]
    high_end = total_stacks_height + low_end
    
    length = high_end + low_end
    mid_point = length / 2
    
    # Guardar métricas de control
    plot_metrics["Low_end"] = low_end
    plot_metrics["High_end"] = high_end
    plot_metrics["Length"] = length
    plot_metrics["Mid_point"] = mid_point
    plot_metrics["Avg_vs_Mid_point"] = avg - mid_point  # Debe ser muy cercano a 0
    
    return plot_metrics



def BuildShapleyRow(variable_name, year, stats, r2_results, shapley_values, predictors):
    """
    Construye dinámicamente una fila estructurada con todas las métricas de Shapley
    y graficación para N variables, adaptándose de forma automática.
    
    Argumentos:
    - variable_name: Nombre de la variable dependiente Y (str).
    - year: Año analizado (int).
    - stats: Diccionario con estadísticas descriptivas (Average, Median, StDev, etc.).
    - r2_results: Diccionario con los R2 de todas las regresiones del bloque B.
    - shapley_values: Diccionario con los Shapley del bloque C.
    - plot_metrics: Diccionario con las métricas de centrado del bloque D.
    - predictors_list: Lista con los nombres de tus variables (las 5 variables actuales).
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
        # 'Avg-SD': plot_metrics.get('Avg_SD')
    }
    
    # Add all R2 combinations
    for combo, r2_val in r2_results.items():
        if combo == ():
            continue
        combo_name = "_".join(combo)
        row[f"R2_{combo_name}"] = r2_val
        
    # Add R2 of the whole model, residal and validations
    row['R2_Residual'] = shapley_values.get('R2_Residual')
    row['R2_Full_Model'] = shapley_values.get('R2_Full_Model')
    row['R2_Shapley_Sum'] = shapley_values.get('Total_R2_Shapley')
    row['Test_R2_Shapley_vs_R2'] = int(round(shapley_values.get('Test_R2_Shapley_vs_R2', 0)))
    
    row["Avg-SD"] = row["Average"] - row["StDev"]
    row['Plot_Avg_SD'] = row["Avg-SD"] if row["Avg-SD"]>0 else 0
    row['Plot_Avg_SD_Surrogate'] = row["Avg-SD"] if row["Avg-SD"]<0 else 0
    row['Plot_Resid'] = (2 * row["StDev"] * row['R2_Residual'])
    
    # Caclulate Avg vs Midpoint
    Low_end = row['Plot_Avg_SD'] + row['Plot_Avg_SD_Surrogate']
    High_end = sum([2 * row["StDev"] * shapley_values.get(f'R2_{predictor}_Shapley', 0) for predictor in predictors]) + row['Plot_Avg_SD'] + row['Plot_Avg_SD_Surrogate']
    Length = High_end + Low_end
    Mid_point = Length / 2
    Avg_vs_Mid_point = row["Average"] - Mid_point # which must equal to zero in the final table
    
    row['Avg_vs_Mid_point'] = int(round(Avg_vs_Mid_point))
    
    # Process independent variables
    total_shapley = shapley_values.get('Total_R2_Shapley', 1.0) # Avoid division by 0
    
    for v in predictors:
        sh_val = shapley_values.get(f"R2_{v}_Shapley", 0.0)
        sd2_val = 2 * row["StDev"] * shapley_values.get(f'R2_{v}_Shapley', 0)
        
        # Dynamic columns per variable
        row[f"R2_{v}_Shapley"] = sh_val
        row[f"Plot_{v}"] = sd2_val
        row[f"R2_{v}_Shapley_norm"] = sh_val / total_shapley if total_shapley != 0 else 0.0

    return row



def ComputeShapleyOwen(wdir, gdp_dir, scenarios, region, age_group, variable, predictors):
    
    
    # Generate a clean dataframe of the mortality and its predictors
    # data = GenerateDataframe4Shapley(wdir, gdp_dir, scenarios, region, age_group, variable)
    
    data = replicateK(variable=variable)
    
    df = []
    
    for year in range(2000,2110,10):
        
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
    "Plot_Resid":"C4" 
    }

    fig, ax = plt.subplots()
    ax = df[['Year','Plot_Avg_SD','Plot_Avg_SD_Surrogate',
    'Plot_gdppc','Plot_population','Plot_climate', "Plot_variability",'Plot_Resid']].\
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
predictors = ["variability", "gdppc", "population", "climate", "erf_draw"]
variable="mortality"
age_group = "All ages"
region = "World"
gdp_dir =  "X:/user/dekkerm/IMAGE_environments/IMPACTS/2_TIMER/outputlib/TIMER_3_5/IMPACTS/{scenario}/indicators/Economy/GDPpc_incl_impacts.out"
wdir = "X:\\user\\liprandicn/Projects\\mt-comparison\\models/Carleton2022/output/SPARCCLE_uncertainty\\"


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



ComputeShapleyOwen(wdir, gdp_dir, scenarios, region, age_group, variable, predictors)