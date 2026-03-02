The preprocessing part serves to generate only once data used in the main model. This saves computation time. The preprocessing part can be skipped if the Honda2014/data folder is copied entirely, as necessary input data is already located there.

## Optimal temperatures

The code calculates the historical 83.6th percentile of daily maximum temperature for the historical period using ERA5 daily data. This percentile is defined by Honda et al., (2014) as the optimal temperature. The code processes the data in latitude bands of certain degrees to optimize memory usage.

