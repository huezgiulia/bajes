from __future__ import division, unicode_literals, absolute_import
import numpy as np



# funzione che mi chiama calc_magnitudes di mkn e modifica output finale per renderlo compatibile a bajes!
def xkn(params_xkn, mkn):
    
    #print('glob param dentro xkn def model', mkn.glob_params)
    mag = mkn.calc_magnitudes(params_xkn)
    
    #print('mag inside xkn function', mag)
    mag_compatibile = {}
    # deve restituire solo dizionario con keys=bande e item=array di magnitudini! (FORSE DA FARE MEGLIO LA TRASFORMAZIONE DEL DIZIONARIO!)
    for key in mag.keys():
        mag_compatibile[key] = mag[key]['mag']

    #print('mag_compatibile inside xkn function', mag_compatibile)
    return mag_compatibile

def xkn_wrapper(time, params):

    #print('param passati a wrapper xkn', params)
    #  param passati a wrapper xkn {'mej_isotropic': 0.04740583369906999, 'vel_isotropic': 0.003539260122094731, 'opac_isotropic': 0.6929262194369953,
    #  'distance': 51.01561176215043, 'time_shift': 24.099454664148883, 'cos_iota': -0.609651415778415, 'log_sigma_mag_B': -6.8905296702290615, 
    # 'log_sigma_mag_g': -7.526032131831177, 'log_sigma_mag_I': 0.678184379020788, 'log_sigma_mag_R': -9.51899990229635, 'log_sigma_mag_K': -7.039455721796221,
    #  'log_sigma_mag_z': 4.468354330877071, 'eps0': 1e+18, 'eps_alpha': 1.3, 'eps_time': 1.3, 'eps_sigma': 0.11, 't_gps': 1187008857.0, 'iota': 2.22641708353551, 
    # 'photometric-lambdas': {'B': 4.45e-07, 'g': 4.75e-07, 'I': 7.75e-07, 'R': 6.58e-07, 'K': 2.19e-06, 'z': 8.5e-07},
    #  'xkn_config': <bajes.obs.kn.approx.xkn.mkn.MKN object at 0x7f5164ca0220>, 'variabili_xkn': <bajes.obs.kn.approx.xkn.config.MKNConfig object at 0x7f26aecadfd0> } 

    
    # variabili su cui voglio fare inferenza!
    input_xkn = {
                'distance':             params['distance'],
                'm_ej_dynamics':        params['mej_isotropic'],
                'vel_dynamics':         params['vel_isotropic'],
                'op_dynamics':          params['opac_isotropic'],
                'view_angle':           params['iota']
                }
    
    variabili_xkn = params['mkn_config'].get_vars(input_xkn)
    #print('variabili_xkn', variabili_xkn)

    return xkn(variabili_xkn, params['xkn_config'])


###   --->  DA FARE: AGGIUNGERE ALTRI WRAPPER A SECONDA DELLE COMPONENTI DEL MODELLO  <---