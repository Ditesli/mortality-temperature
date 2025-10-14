import mortality_functions as mf

wdir = 'X:\user\liprandicn\mt-comparison\Carleton2022'

''' Define important variables'''
scenarios_RCP = ['SSP126', 'SSP245', 'SSP370', 'SSP585']
scenarios_SSP = ['SSP1', 'SSP2', 'SSP3', 'SSP5']
years = range(2015,2105,5)
groups = ['oldest', 'older', 'young']
index = pd.MultiIndex.from_product([groups, scenarios], names=['Age group', 'Scenario'])
results_noadapt = pd.DataFrame(index=index, columns=years)