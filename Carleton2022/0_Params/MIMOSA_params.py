from scipy.optimize import curve_fit
import numpy as np


def get_region(hierid):
    iso_code = hierid.split('.')[0]
    iso_to_region = {iso: region for region, isos in MIMOSA.items() for iso in isos}
    return iso_to_region.get(iso_code, 'Unknown')


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