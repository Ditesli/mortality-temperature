def running_mean(base_path, scenario, year):
    
    df_previous = pd.read_csv(os.path.join(base_path, f'{scenario}\\BC_{climate_model}_{scenario}_{year-1}.csv'))
    df_current = pd.read_csv(os.path.join(base_path, f'{scenario}\\BC_{climate_model}_{scenario}_{year}.csv'))
    df_next = pd.read_csv(os.path.join(base_path, f'{scenario}\\BC_{climate_model}_{scenario}_{year+1}.csv'))
    
    # Las columnas de fecha comienzan desde la 7ª columna
    temp_previous = df_previous.columns[1:]
    temp_current = df_current.columns[1:]
    temp_next = df_next.columns[1:]
    
    # Concatenar solo las columnas de fecha
    df_concat = pd.concat([
        df_previous[temp_previous],
        df_current[temp_current],
        df_next[temp_next]
    ], axis=1)
    
    df_running_mean = df_concat.rolling(window=31, center=True, axis=1).mean()
    df_result = df_running_mean[temp_current]
    
    df_rounded = df_result.round(2)
    df_final = pd.concat([df_current['hierid'], df_rounded], axis=1)
    
    return df_final


def running_mean2(base_path, scenario, year): # Run for every scenario for the years 2015 and 2100
    df = pd.read_csv(os.path.join(base_path, f'{scenario}\\BC_{climate_model}_{scenario}_{year}.csv'))
    temp = df.columns[1:]
    
    df_result = df[temp]
    for i, fecha in enumerate(temp):
        df_result.loc[:, fecha] = df[temp].iloc[:, max(0, i-15):min(len(temp), i+16)].mean(axis=1)
    
    df_rounded = df_result.round(2)
    df_final = pd.concat([df['hierid'], df_rounded], axis=1)
    df_final.to_csv(f'D:\\Climate Model - Climate Variability\\T - Monthly Mean\\{scenario}\\Monthly Mean_{scenario}_{year}.csv')
    
    return df_final


def get_linear_reg(base_path, scenario):
    
    year_mean = pd.read_csv(f'{base_path}/Main folder/Climate variability/Annual Mean_{scenario}.csv')
    df = year_mean.iloc[:,2:]

    # Initialize lists to hold the slopes and intercepts
    slopes = []
    intercepts = []

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        # Prepare the data for linear regression
        X = row.index.values.reshape(-1, 1).astype(float)  # Years as float
        y = row.values.reshape(-1, 1)

        # Create and fit the model
        model = LinearRegression()
        model.fit(X, y)

        # Store the slope and intercept
        slopes.append(model.coef_[0][0])
        intercepts.append(model.intercept_[0])

    # Convert the lists to new columns in the dataframe
    year_mean['slope'] = slopes
    year_mean['intercept'] = intercepts

    year_mean.to_csv(f'{base_path}/Main folder/Climate variability/Annual Mean_{scenario}.csv')


def custom_divide(value, date):
    if date == '02-29':
        return value / 21
    else:
        return value / 86
    

def daily_regression(model_path, base_path, climate_model, scenario, month, days_month):

    years = np.arange(2015,2101)

    # Genera la lista de fechas del año (considerando año no bisiesto)
    start_date = datetime.date(2015, month, 1)  # Año no bisiesto para simplificar
    end_date = datetime.date(2015, month, days_month)
    delta = datetime.timedelta(days=1)

    column_names = [str(year) for year in range(2015, 2101)]
    dfs = {}

    current_date = start_date
    while current_date <= end_date:
        day = current_date.strftime(f'%m-%d')
        df_name = current_date.strftime(f'{scenario}-%m-%d')
        dfs[df_name] = pd.DataFrame(columns=column_names)
        current_date += delta

        for year in years:
            SSP = pd.read_csv(f'{model_path}/{scenario}/BC_{climate_model}_{scenario}_{year}.csv'))
            SSP = SSP.iloc[:,1:]
            SSP.columns = pd.to_datetime(SSP.columns).strftime('%m-%d').tolist()
            dfs[df_name][f'{year}'] = SSP[day]
            print(f'{day}-{year}')

        dfs[df_name] = dfs[df_name].round(2)
        inter_path = f'{base_path}/Main folder/Climate variability/{scenario}_inter'
        if not os.path.exists(inter_path):
            os.makedirs(inter_path)
        dfs[df_name].to_csv(f'{inter_path}/{scenario}_{day}.csv')