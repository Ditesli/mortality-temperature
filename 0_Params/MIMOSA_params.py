from scipy.optimize import curve_fit
import numpy as np

MIMOSA = {
    'Canada': ['CAN'],
    'USA': ['SPM', 'USA', 'UMI'],
    'Mexico': ['MEX'],
    'Central America': ['AIA', 'ABW', 'BHS', 'BRB', 'BLZ', 'BMU', 'CYM', 'CRI', 'DMA', 'DOM', 'SLV', 'GRD', 'GLP', 'GTM', 'HTI', 'HND', 'JAM', 'MTQ',
                        'MSR', 'ANT', 'NIC', 'PAN', 'PRI', 'KNA', 'LCA', 'VCT', 'TTO', 'TCA', 'VGB', 'VIR', 'CUB', 'BLM', 'ATG', 'MAF', 'CUW', 'SMX', 
                        'BES', 'CL-'],
    'Brazil': ['BRA'],
    'Rest of South America': ['ARG', 'BOL', 'CHL', 'COL', 'ECU', 'FLK', 'GUF', 'GUY', 'PRY', 'PER', 'SUR', 'URY', 'VEN', 'SGS'],
    'Northern Africa': ['DZA', 'EGY', 'LBY', 'MAR', 'TUN', 'ESH'],
    'Western Africa': ['BEN', 'BFA', 'CMR', 'CPV', 'CAF', 'TCD', 'COD', 'COG', 'CIV', 'GNQ', 'GAB', 'GMB', 'GHA', 'GIN', 'GNB', 'LBR', 'MLI', 'MRT',
                       'NER', 'NGA', 'STP', 'SEN', 'SLE', 'SHN', 'TGO'],
    'Eastern Africa': ['BDI', 'COM', 'DJI', 'ERI', 'ETH', 'KEN', 'MDG', 'MUS', 'REU', 'RWA', 'SYC', 'SOM', 'SDN', 'UGA', 'SSD', 'MYT'],
    'South Africa': ['ZAF'],
    'Western Europe': ['AND', 'AUT', 'BEL', 'DNK', 'FRO', 'FIN', 'FRA', 'DEU', 'GIB', 'GRC', 'ISL', 'IRL', 'ITA', 'LIE', 'LUX', 'MLT', 'MCO', 'NLD', 
                       'NOR', 'PRT', 'SMR', 'ESP', 'SWE', 'CHE', 'GBR', 'VAT', 'SJM', 'IMN', 'JEY', 'ALA', 'GGY', 'GRL'],
    'Central Europe': ['ALB','BIH','BGR','HRV', 'CYP', 'CZE', 'EST', 'HUN', 'LVA', 'LTU', 'MKD', 'POL', 'ROU', 'SRB', 'SVK', 'SVN', 'MNE','KO-'],
    'Turkey': ['TUR'],
    'Ukraine region': ['BLR', 'MDA', 'UKR'],
    'Central Asia': ['KAZ', 'KGZ', 'TJK', 'TKM', 'UZB'],
    'Russia region': ['ARM', 'AZE', 'GEO', 'RUS'],
    'Middle East': ['BHR', 'IRN', 'IRQ', 'ISR', 'JOR', 'KWT', 'LBN', 'OMN', 'QAT', 'SAU', 'SYR', 'ARE', 'YEM', 'PSE'],
    'India': ['IND'],
    'Korea region': ['PRK', 'KOR'],
    'China region': ['CHN', 'HKG', 'MAC', 'MNG', 'TWN'],
    'Southeastern Asia': ['BRN', 'KHM', 'LAO', 'MYS', 'MMR', 'PHL', 'SGP', 'THA', 'VNM', 'SP-'],
    'Indonesia region': ['TLS', 'IDN', 'PNG', 'GUM', 'CXR'],
    'Japan': ['JPN'],
    'Oceania': ['ASM', 'AUS', 'COK', 'FJI', 'PYF', 'KIR', 'MHL', 'FSM', 'NRU', 'NCL', 'NZL', 'NIU', 'MNP', 'PLW', 'PCN', 'WSM', 'SLB', 'TKL', 'TON',
                'TUV', 'VUT', 'VUT', 'WLF', 'HMD', 'CCK', 'NFK', 'ATF'],
    'Rest of South Asia': ['AFG', 'BGD', 'BTN', 'MDV', 'NPL', 'PAK', 'LKA', 'IOT'],
    'Rest of Southern Africa': ['AGO', 'BWA', 'LSO', 'MWI', 'MOZ', 'NAM', 'SWZ', 'TZA', 'ZMB', 'ZWE', 'BVT'], 
    #'Greenland': ['GRL'],
    'Antarctica': ['ATA'],
    'Caspian Sea': ['CA-']
}


def get_region(hierid):
    iso_code = hierid.split('.')[0]
    iso_to_region = {iso: region for region, isos in MIMOSA.items() for iso in isos}
    return iso_to_region.get(iso_code, 'Unknown')


region_colors = {'Brazil':0, 'Canada':1, 'Central America':3, 'Central Asia': 4, 'Central Europe': 5, 'China region': 6, 'Eastern Africa': 7, 
                 'India': 9, 'Indonesia region': 11, 'Japan': 10, 'Korea region': 12, 'Mexico': 13, 'Middle East': 15, 'Northern Africa': 16, 
                 'Oceania': 17, 'Rest of South America': 18, 'Rest of South Asia': 19, 'Rest of Southern Africa': 20, 'Russia region': 21, 
                 'South Africa': 22, 'Southeastern Asia': 23, 'Turkey': 24, 'USA': 25, 'Ukraine region': 26, 'Western Africa': 27, 'Western Europe': 14,
                 'WORLD':36}



def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def count_points_above(y_fit, y, alpha):
            new_y_fit = alpha * y_fit
            return np.sum(y > new_y_fit)

def find_alpha(y_fit, y, target_points, tol=1, max_iter=200):
    mid = 1.0; high = 20; low = -20; iteration = 0

    while iteration < max_iter:
        iteration += 1
        points_above = count_points_above(y_fit, y, mid)
    
        if abs(points_above - target_points) < tol:
            break

        if points_above > target_points:
            low = mid
        else:
            high = mid

        mid = (high + low) / 2.0

    return mid

def order_data(x, y):
    sorted_indices = np.argsort(x)
    return x[sorted_indices], y[sorted_indices]

# Función para encontrar los índices de los valores más cercanos
def find_indices_within_interval(x, start, end):
    indices = np.where((x >= start) & (x < end))[0]
    return indices

def coeff_percentile(x, y, lower_bound, upper_bound):

    x_ordered, y_ordered = order_data(x,y)
    
    quantile_10_x = []; quantile_10_y = []; quantile_90_x = []; quantile_90_y = []
    
    # Recorrer los intervalos y encontrar los valores correspondientes
    intervals = [(i, i + 1) for i in range(7)]
    for start, end in intervals:
        indices = find_indices_within_interval(x_ordered, start, end)
        if indices.size > 0:
            y_values = y_ordered[indices]
            x_values = x_ordered[indices]
            
            q10 = np.quantile(y_values, lower_bound)
            q90 = np.quantile(y_values, upper_bound)
            
            # Encontrar los índices de los quantiles
            q10_index = np.argmin(np.abs(y_values - q10))
            q90_index = np.argmin(np.abs(y_values - q90))
            
            # Almacenar los valores correspondientes
            quantile_10_x.append(x_values[q10_index])
            quantile_10_y.append(y_values[q10_index])
            quantile_90_x.append(x_values[q90_index])
            quantile_90_y.append(y_values[q90_index])
    
    # Convertir las listas a arrays
    quantile_10_x = np.array(quantile_10_x); quantile_10_y = np.array(quantile_10_y)
    
    quantile_90_x = np.array(quantile_90_x); quantile_90_y = np.array(quantile_90_y)
    
    popt_10, _ = curve_fit(quadratic, quantile_10_x, quantile_10_y)
    popt_90, _ = curve_fit(quadratic, quantile_90_x, quantile_90_y)
    return popt_10, popt_90