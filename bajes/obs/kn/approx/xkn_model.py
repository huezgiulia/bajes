from __future__ import division, unicode_literals, absolute_import
import numpy as np


def xkn(params_xkn, mkn):

    # compute magnitudes
    mag = mkn.calc_magnitudes(params_xkn)
    return mag


def xkn_wrapper_1comp(time, params):
    ''' Wrapper for one component model. Fixed name of the component: "dynamics". '''

    input_xkn = params
    # inference variables
    variabiles_xkn = params['mkn_config'].get_vars(input_xkn)
    return xkn(variabiles_xkn, params['xkn_config'])


def xkn_wrapper_2comp(time, params):
    ''' Wrapper for two component model. Fixed names of the components: "dynamics", "secular". '''

    input_xkn = params
    # inference variables
    variabiles_xkn = params['mkn_config'].get_vars(input_xkn)
    return xkn(variabiles_xkn, params['xkn_config'])


def xkn_wrapper_3comp(time, params):
    ''' Wrapper for three components model. Fixed names of the components: "dynamics", "secular", "wind". '''

    input_xkn = params
    # inference variables
    variabiles_xkn = params['mkn_config'].get_vars(input_xkn)
    return xkn(variabiles_xkn, params['xkn_config'])
