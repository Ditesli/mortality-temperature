import numpy as np


### ------------------------------------------------------------------------------
'''
Gamma coeficients of the covariants for the temperature response function of the 
three age groups (young, older, oldest) from Carleton et al. (2022).
'''

gamma_np = {
    'young': np.array([
        [-0.2643747697030857, -0.0012157807919976, 0.0285121426008164],
        [-0.0147654905557389, -0.0001292299812386, 0.0013467700198057],
        [0.0000555941144027,  0.000010228738298,  -0.0000128604018705],
        [0.0000188858412856, -2.48887855043e-07,  -1.50547526657e-06]
    ]),
    'older': np.array([
        [0.2478292444689566,  0.0022092761549115, -0.0258890110895998],
        [-0.0125437290633759, 0.0000123113770044, 0.0012019083245803],
        [-0.0002220037659301, -2.82565977452e-06, 0.0000227328454772],
        [0.0000129910024803,  1.82855114488e-08,  -1.21751952067e-06]
    ]),
    'oldest': np.array([
        [6.399027562773568,   0.0436967573579832, -0.6751842737945384],
        [-0.3221434191389331, 0.0013726982372035, 0.0295628065147365],
        [-0.0044299345528043, -0.0001067884304388, 0.00050851740502],
        [0.0002888631905257,  9.32783835571e-07,  -0.0000273410162051]
    ])
}



### ------------------------------------------------------------------------------
'''
IMAGE regions and their corresponding ISO country codes.
'''

IMAGE = {
    'Canada': ['CAN'],
    'USA': ['SPM', 'USA', 'UMI'],
    'Mexico': ['MEX'],
    'Central America': ['AIA', 'ABW', 'BHS', 'BRB', 'BLZ', 'BMU', 'CYM', 'CRI', 'DMA', 
                        'DOM', 'SLV', 'GRD', 'GLP', 'GTM', 'HTI', 'HND', 'JAM', 'MTQ',
                        'MSR', 'ANT', 'NIC', 'PAN', 'PRI', 'KNA', 'LCA', 'VCT', 'TTO', 
                        'TCA', 'VGB', 'VIR', 'CUB', 'BLM', 'ATG', 'MAF', 'CUW', 'SMX', 
                        'BES', 'CL-'],
    'Brazil': ['BRA'],
    'Rest of South America': ['ARG', 'BOL', 'CHL', 'COL', 'ECU', 'FLK', 'GUF', 'GUY', 
                              'PRY', 'PER', 'SUR', 'URY', 'VEN', 'SGS'],
    'Northern Africa': ['DZA', 'EGY', 'LBY', 'MAR', 'TUN', 'ESH'],
    'Western Africa': ['BEN', 'BFA', 'CMR', 'CPV', 'CAF', 'TCD', 'COD', 'COG', 'CIV', 
                       'GNQ', 'GAB', 'GMB', 'GHA', 'GIN', 'GNB', 'LBR', 'MLI', 'MRT',
                       'NER', 'NGA', 'STP', 'SEN', 'SLE', 'SHN', 'TGO'],
    'Eastern Africa': ['BDI', 'COM', 'DJI', 'ERI', 'ETH', 'KEN', 'MDG', 'MUS', 'REU', 
                       'RWA', 'SYC', 'SOM', 'SDN', 'UGA', 'SSD', 'MYT'],
    'South Africa': ['ZAF'],
    'Western Europe': ['AND', 'AUT', 'BEL', 'DNK', 'FRO', 'FIN', 'FRA', 'DEU', 'GIB', 
                       'GRC', 'ISL', 'IRL', 'ITA', 'LIE', 'LUX', 'MLT', 'MCO', 'NLD', 
                       'NOR', 'PRT', 'SMR', 'ESP', 'SWE', 'CHE', 'GBR', 'VAT', 'SJM', 
                       'IMN', 'JEY', 'ALA', 'GGY', 'GRL'],
    'Central Europe': ['ALB','BIH','BGR','HRV', 'CYP', 'CZE', 'EST', 'HUN', 'LVA', 
                       'LTU', 'MKD', 'POL', 'ROU', 'SRB', 'SVK', 'SVN', 'MNE','KO-'],
    'Turkey': ['TUR'],
    'Ukraine region': ['BLR', 'MDA', 'UKR'],
    'Central Asia': ['KAZ', 'KGZ', 'TJK', 'TKM', 'UZB'],
    'Russia region': ['ARM', 'AZE', 'GEO', 'RUS'],
    'Middle East': ['BHR', 'IRN', 'IRQ', 'ISR', 'JOR', 'KWT', 'LBN', 'OMN', 'QAT', 
                    'SAU', 'SYR', 'ARE', 'YEM', 'PSE'],
    'India': ['IND'],
    'Korea region': ['PRK', 'KOR'],
    'China region': ['CHN', 'HKG', 'MAC', 'MNG', 'TWN'],
    'Southeastern Asia': ['BRN', 'KHM', 'LAO', 'MYS', 'MMR', 'PHL', 'SGP', 'THA', 'VNM', 'SP-'],
    'Indonesia region': ['TLS', 'IDN', 'PNG', 'GUM', 'CXR'],
    'Japan': ['JPN'],
    'Oceania': ['ASM', 'AUS', 'COK', 'FJI', 'PYF', 'KIR', 'MHL', 'FSM', 'NRU', 'NCL', 
                'NZL', 'NIU', 'MNP', 'PLW', 'PCN', 'WSM', 'SLB', 'TKL', 'TON', 'TUV',
                'VUT', 'VUT', 'WLF', 'HMD', 'CCK', 'NFK', 'ATF'],
    'Rest of South Asia': ['AFG', 'BGD', 'BTN', 'MDV', 'NPL', 'PAK', 'LKA', 'IOT'],
    'Rest of Southern Africa': ['AGO', 'BWA', 'LSO', 'MWI', 'MOZ', 'NAM', 'SWZ', 'TZA', 
                                'ZMB', 'ZWE', 'BVT'], 
    #'Greenland': ['GRL'],
    'Antarctica': ['ATA'],
    'Caspian Sea': ['CA-']
}



### ------------------------------------------------------------------------------
'''
CMIP6 climate models and their corresponding realization and grid label.
'''

climate_models_dic = {'AWI-CM-1-1-MR':['r1i1p1f1', 'gn'], 'BCC-CSM2-MR':['r1i1p1f1', 'gn'], 
                      'CAMS-CSM1-0':['r1i1p1f1', 'gn'], 'CESM2':['r4i1p1f1', 'gn'], 
                      'CESM2-WACCM':['r1i1p1f1', 'gn'], 'CMCC-CM2-SR5':['r1i1p1f1', 'gn'],
                      'CMCC-ESM2':['r1i1p1f1', 'gn'], 'CNRM-CM6-1-HR':['r1i1p1f2', 'gr'], 
                      'EC-Earth3':['r4i1p1f1', 'gr'], 'EC-Earth3-Veg':['r1i1p1f1', 'gr'], 
                      'GFDL-ESM4':['r1i1p1f1', 'gr1'], 'INM-CM4-8':['r1i1p1f1', 'gr1'],
                     'INM-CM5-0':['r1i1p1f1', 'gr1'], 'MPI-ESM1-2-HR':['r1i1p1f1', 'gn'], 
                     'MRI-ESM2-0':['r1i1p1f1', 'gn'], 'NorESM2-MM':['r1i1p1f1', 'gn'], 
                     'TaiESM1':['r1i1p1f1', 'gn']}



### ------------------------------------------------------------------------------
'''
Colors for the different regions in the plots.
'''

region_colors = {'Brazil': 0, 'Canada': 1, 'Central America': 3, 'Central Asia': 4, 'Central Europe': 5, 
                 'China region': 6, 'Eastern Africa': 7, 'India': 9, 'Indonesia region': 11, 
                 'Japan': 10, 'Korea region': 12, 'Mexico': 13, 'Middle East': 15, 'Northern Africa': 16, 
                 'Oceania': 17, 'Rest of South America': 18, 'Rest of South Asia': 19, 
                 'Rest of Southern Africa': 20, 'Russia region': 21, 'South Africa': 22, 
                 'Southeastern Asia': 23, 'Turkey': 24, 'USA': 25, 'Ukraine region': 26, 
                 'Western Africa': 27, 'Western Europe': 14, 'WORLD': 36}
