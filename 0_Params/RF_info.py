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

@nb.njit
def get_tas(tas_coeffs, climtas, loggdppc):
    tas = np.zeros(4)
    for i in range(4):
        # tas_coeffs[i] access tas1, tas2, tas3, tas4
        tas[i] = tas_coeffs[i, 0] + tas_coeffs[i, 1] * climtas + tas_coeffs[i, 2] * loggdppc
    return tas


@nb.njit
def response(df_group, tas, tmin, t):
    
    raw = tas[0]*t + tas[1]*t**2 + tas[2]*t**3 + tas[3]*t**4

    mortality = raw - tas[0]* tmin - tas[1]*tmin**2 - tas[2]*tmin**3 - tas[3]*tmin**4
    
    mortality = np.minimum(mortality, df_group)

    t_left = t[t < tmin]
    t_right = t[t > tmin]

    if len(t_left) > 0:
        for i in range(len(t_left) - 1, -1, -1):
            mortality[i] = max(mortality[i], mortality[i + 1])
    
    if len(t_right) > 0:
        for i in range(len(t_left) + 1, len(mortality)):
            mortality[i] = max(mortality[i - 1], mortality[i])
            
    mortality = np.maximum(mortality, 0)
    
    return mortality