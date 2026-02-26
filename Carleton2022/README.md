## Mortality-Temperature Model - Carleton et al. Version

The model originates from the work by [(Carleton et al., 2022)](https://doi.org/10.1093/qje/qjac020). It is an econometric model that relates daily temperature with the relative mortality (deaths per 100k people), through functions where mortality increases towards hot and cold non-optimal temperatures. These functions, developed for a total of 24,378 administrative regions and three age groups, are defined as fourth degree polynomials, 

$$
M_{a,i,t} = \sum_{k=1}^4 \left(\gamma_{0,a,k} + \gamma_{1,a,k} T_{MEAN_{i,t}} + \gamma_{2,a,k} \text{log}(GDPpc)_{i,t} \right) \mathbb{T}_{i,t}^k
$$

where $a$ is the age group, $i$ is the administrative region (or impact region) and $t$ the year. $T_{MEAN}$ is defined as the 30 year running mean of the mean temperature (climatology), $GDPpc$ is the 13 year running mean of annual GDP per capita, and the coefficients $γ$ are the result of the regression analysis performed by the authors. Therefore, for every age group, impact region and year, we have a funciton called a Exposure Response Function that calculates the daily mortality $M$ associated to a daily temperature $T$.

Some of the advantages of employing this methodology in an Integrated Assessment Model include: the disagregation of mortality between three age groups: young (0-4 years), older (5-64 years), and oldest (+65 years). This disaggregation allows to analyze the larger impact on the elderly; as the income per capita and climatology are included as covariates of the equation; letting the coefficients to evolve in time allows to have adaptation in which individuals are assumed to take individual actions, constrained to their income, to have a lesser mortality risk.
