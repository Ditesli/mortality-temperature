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