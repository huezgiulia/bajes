from __future__ import division, unicode_literals, absolute_import
import numpy as np
import afterglowpy as grb
from bajes import MPC_2_CM

def compute_centroid_afterglow(time, nu, params):
    """
    Compute centroid position using afterglowpy
    as in the example in plotCentroidAndSize.py
    """
    moment = np.empty(time.shape, dtype=int)

    # compute moment 0 (flux)
    moment[:] = grb.jet.MOM_0
    Fnu = grb.fluxDensity(time, nu, **params, moment=moment)

    # compute moment 1 (integral of x_obs * intensity)
    moment[:] = grb.jet.MOM_X
    FnuX = grb.fluxDensity(time, nu, **params, moment=moment)

    # compute moment 2 (integral of x_obs^2 * intensity)
    moment[:] = grb.jet.MOM_XX
    FnuXX = grb.fluxDensity(time, nu, **params, moment=moment)

    # compute moment 2 (integral of y_obs^2 * intensity)
    moment[:] = grb.jet.MOM_YY
    FnuYY = grb.fluxDensity(time, nu, **params, moment=moment)

    # Get the intensity-weighted distance measures
    X_cm = FnuX / Fnu     # in cm
    dA = params['d_L'] / (1 + params['z'])**2
    X_rad = X_cm / dA
    rad2mas = 1000 * 3600 * 180 / np.pi
    x = X_rad * rad2mas

    return Fnu, x

def compute_centroid_position(params, xc, flux):
    """
    Compute theoretical centroid position as in Ryan et al. 2023
    """
    xra     = xc*np.sin(params['PA'])+params['RA']
    xdec    = xc*np.cos(params['PA'])+params['DEC']

    return np.array([flux, xra, xdec])

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

    try:
        flux, xc        = compute_centroid_afterglow(t, nu, grb_params)
    except Exception as e:
        print(f"Afterglowpy error: {e}")
        return -np.inf
    centroid_pos    = compute_centroid_position(params, xc, flux) 

    return centroid_pos
