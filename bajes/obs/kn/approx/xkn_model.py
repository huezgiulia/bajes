from __future__ import division, unicode_literals, absolute_import
import numpy as np



# funzione che mi chiama calc_magnitudes di mkn e modifica output finale per renderlo compatibile con bajes!
def xkn(params_xkn, mkn):
    
    #print('times in mkn', mkn.times)
    #print('glob param dentro xkn def model', mkn.glob_params)
    mag = mkn.calc_magnitudes(params_xkn)
    #print('mkn time array in xkn', mkn.times)
    #print('mkn time array in xkn len', len(mkn.times))
    
    #print('mag inside xkn function', mag)
    #mag_compatibile = {}
    # deve restituire solo dizionario con keys=bande e item=array di magnitudini! (FORSE DA FARE MEGLIO LA TRASFORMAZIONE DEL DIZIONARIO!)
    #for key in mag.keys():
    #    #print('TIME SU CUI XKN COMPUTE MAGNITUDES',mag[key]['time'])
    #    mag_compatibile[key] = mag[key]['mag']

    #print('mag_compatibile inside xkn function', mag_compatibile)
    return mag #mag_compatibile

def xkn_wrapper_1comp(time, params):
    '''
    Wrapper for one component model. Fixed name of the component: "dynamics"
    '''
    input_xkn = params
    '''
    input_xkn = {}
    input_xkn['distance'] = params['distance']
    input_xkn['view_angle'] = params['iota']

    for i,ci in enumerate(['dynamics']):
        input_xkn[f'm_ej_{ci:s}'] = params[f'mej_{i+1}'] 
        input_xkn[f'vel_{ci:s}'] = params[f'vel_{i+1}']

        if f"opac_high_{i+1}" in params:
            input_xkn[f'high_lat_op_{ci:s}'] = params[f'opac_high_{i+1}']
            input_xkn[f'low_lat_op_{ci:s}'] = params[f'opac_{i+1}']
        else:
            input_xkn[f'op_{ci:s}'] = params[f'opac_{i+1}']

        if f"step_angle_op_{i+1}" in params:
            input_xkn[f'step_angle_op_{ci:s}'] = params[f'step_angle_op_{i+1}']
    '''
    # inference variables
    variabili_xkn = params['mkn_config'].get_vars(input_xkn)

    return xkn(variabili_xkn, params['xkn_config'])



### nome parametri deve essere in accordo con congig.ini file!
def xkn_wrapper_2comp(time, params):
    '''
    Wrapper for two component model. Fixed names of the components: "dynamics" "secular"
    '''
    #print('param passati a wrapper xkn 2c', params)
    input_xkn = params
    '''
    input_xkn = {}
    input_xkn['distance'] = params['distance']
    input_xkn['view_angle'] = params['iota']

    for i,ci in enumerate(['dynamics', 'secular']):
        input_xkn[f'm_ej_{ci:s}'] = params[f'mej_{i+1}'] 
        input_xkn[f'vel_{ci:s}'] = params[f'vel_{i+1}']

        if f"opac_high_{i+1}" in params:
            input_xkn[f'high_lat_op_{ci:s}'] = params[f'opac_high_{i+1}']
            input_xkn[f'low_lat_op_{ci:s}'] = params[f'opac_{i+1}']
        else:
            input_xkn[f'op_{ci:s}'] = params[f'opac_{i+1}']

        if f"step_angle_op_{i+1}" in params:
            input_xkn[f'step_angle_op_{ci:s}'] = params[f'step_angle_op_{i+1}']
    '''
    # inference variables
    variabili_xkn = params['mkn_config'].get_vars(input_xkn)

    return xkn(variabili_xkn, params['xkn_config'])


### nome parametri deve essere in accordo con congig.ini file!
def xkn_wrapper_3comp(time, params):
    '''
    Wrapper for three components model. Fixed names of the components: "dynamics" "secular" "winnd
    '''
    #print('param passati a wrapper xkn 3c', params)
    input_xkn = params
    '''
    input_xkn = {}
    input_xkn['distance'] = params['distance']
    input_xkn['view_angle'] = params['iota']

    for i,ci in enumerate(['dynamics', 'secular', 'wind']):
        input_xkn[f'm_ej_{ci:s}'] = params[f'mej_{i+1}'] 
        input_xkn[f'vel_{ci:s}'] = params[f'vel_{i+1}']

        if f"opac_high_{i+1}" in params:
            input_xkn[f'high_lat_op_{ci:s}'] = params[f'opac_high_{i+1}']
            input_xkn[f'low_lat_op_{ci:s}'] = params[f'opac_{i+1}']
        else:
            input_xkn[f'op_{ci:s}'] = params[f'opac_{i+1}']

        if f"step_angle_op_{i+1}" in params:
            input_xkn[f'step_angle_op_{ci:s}'] = params[f'step_angle_op_{i+1}']
    '''
    # inference variables
    variabili_xkn = params['mkn_config'].get_vars(input_xkn)

    return xkn(variabili_xkn, params['xkn_config'])
