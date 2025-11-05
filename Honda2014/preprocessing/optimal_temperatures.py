
import utils

'''
This code calculates the historical 83.6th percentile of daily maximum temperature (T2M)
for the historical period using ERA5 daily data. This percentile is defined by 
Honda et al., (2014) as the optimal temperature.
The code processes the data in latitude bands of 30 degrees to optimize memory usage.
'''


# Set path to ERA5 data
data_path = 'X:/user/liprandicn/Data/ERA5/t2m_daily/'

# Set path to save results
final_path = 'X:/user/liprandicn/mt-comparison/Honda2014/data/optimal_temperatures/'


# Define the years range and step size for processing
years = range(1990,2021)
# Define the step size for latitude bands
step = 40

# Calculate historical percentiles
utils.calculate_optimal_temperature(data_path,
                                    final_path,
                                    years, 
                                    step)


