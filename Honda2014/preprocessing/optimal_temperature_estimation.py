'''
This code calculates the historical 83.6th percentile of daily maximum temperature (T2M)
for the historical period using ERA5 daily data. 
The code processes the data in latitude bands of 30 degrees to optimize memory usage.
'''


import utils


# Set path to ERA5 data
data_path = 'X:\\user\\liprandicn\\Data\\ERA5\\t2m_daily\\'

# Set path to save results
final_path = 'X:\\user\\liprandicn\\mt-comparison\\Honda2014\\data\\optimal_temeprature\\'

# Define the years range and step size for processing
years = range(1980,2011)
# Define the step size for latitude bands
step = 30

# Calculate historical percentiles
utils.calculate_optimal_temperature(data_path,
                                    final_path,
                                    years, 
                                    step)