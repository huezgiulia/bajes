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

    grb_params = params.copy()
    grb_params['E0']        = 10**params['E0']
    grb_params['n0']        = 10**params['n0']
    grb_params['epsilon_e'] = 10**params['epsilon_e']
    grb_params['epsilon_B'] = 10**params['epsilon_B']
    grb_params['d_L']       = params['d_L'] * MPC_2_CM
    grb_params['thetaObs']  = np.pi / 2 - np.abs(np.arccos(params['thetaObs']) - np.pi / 2)
    
    return afterglowpy(t, nu, grb_params)