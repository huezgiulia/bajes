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

    grb_params = {}

    grb_params['jetType']   = params['jetType']
    grb_params['thetaCore'] = params['thetaCore']
    grb_params['d_L']       = params['distance'] * MPC_2_CM
    grb_params['thetaObs']  = np.pi / 2 - np.abs(np.arccos(params['cos_iota']) - np.pi / 2)
    grb_params['E0']        = 10**params['E0']
    grb_params['n0']        = 10**params['n0']
    grb_params['epsilon_e'] = 10**params['epsilon_e']
    grb_params['epsilon_B'] = 10**params['epsilon_B']
    grb_params['p']         = params['p']
    grb_params['xi_N']      = params['xi_N']
    grb_params['z']         = params['z']

    if 'thetaWing' in params:
        grb_params['thetaWing'] = params['thetaWing']
    if 'b' in params:
        grb_params['b'] = params['b']
    if 'specType' in params:
        grb_params['specType'] = params['specType']
    
    return afterglowpy(t, nu, grb_params)