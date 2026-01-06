from __future__ import division, unicode_literals, absolute_import
import numpy as np
import afterglowpy as grb
from bajes import MPC_2_CM

def afterglowpy(t, nu, grb_params):
    # compute fluxes
    Fnu = grb.fluxDensity(t, nu, **grb_params)
    return Fnu


def afterglow_wrapper(t, nu, params):
    ''' Wrapper for grb model from afterglowpy.'''

    params['thetaObs']  = np.pi / 2 - np.abs(np.arccos(params['cos_iota']) - np.pi / 2)
    afterglowpy_params  = ['thetaObs', 'thetaCore', 'E0', 'n0', 'p', 'epsilon_e', 'epsilon_B', 'thetaWing', 'jetType', 'xi_N', 'd_L', 'z']
    grb_params          = {k: v for k, v in params.items() if k in afterglowpy_params}
    if 'distance' in params:
        params['d_L']   = params['distance']

    grb_params['E0']        = 10**params['E0']
    grb_params['n0']        = 10**params['n0']
    grb_params['epsilon_e'] = 10**params['epsilon_e']
    grb_params['epsilon_B'] = 10**params['epsilon_B']
    grb_params['d_L']       = params['d_L'] * MPC_2_CM
    
    return afterglowpy(t, nu, grb_params)