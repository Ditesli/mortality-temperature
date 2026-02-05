## Mortality-Temperature Model - Carleton Version

The model is based on (Carleton et al., 2022). It is an econometric model that relates daily temperature with the relative mortality (deaths per 100k people). This model is defined as fourth degree polynomial where $T_MEAN$ is defined as the 30 year running mean of the mean temperature, $GDPpc$ is the 13 year running mean of the annual GDP per capita, the coefficients γ are the result of the regression analysis performed by the authors. Therefore, for every age group a, impact region i (out of 24,378) and year t, we have a so called Exposure Response Function that calculates mortality M and is dependent of the daily temperature $T$.


$$
M_{a,i,t} = \sum_{k=1}^4 \left(\gamma_{0,a,k} + \gamma_{1,a,k} T_{MEAN_{i,t}} + \gamma_{2,a,k} \text{log}(GDPpc)_{i,t} \right) \mathbb{T}_{i,t}^k
$$


Among the advantages of this methodology is the separation of mortality between three age groups: young (0-4 years), older (5-64 years), and oldest (+65 years). Also, since the income per capita, expressed as log(GDPpc), is included in the coefficients of the equation; it is possible to include a form of adaptation in which individuals take some choices such as installation of air conditioning, constrained to their incomes. 