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
    xra     = xc*np.sin(params['pa'])+params['ra']
    xdec    = xc*np.cos(params['pa'])+params['dec']

    return np.array([flux, xra, xdec])

def afterglow_wrapper(t, nu, params):
    ''' Wrapper for grb model from afterglowpy.'''

    params['thetaObs']  = np.pi / 2 - np.abs(np.arccos(params['cos_iota']) - np.pi / 2)
    afterglowpy_params  = ['thetaObs', 'thetaCore', 'E0', 'n0', 'p', 'epsilon_e', 'epsilon_B', 'thetaWing', 'jetType', 'xi_N', 'd_L', 'z']
    params_grb          = {k: v for k, v in params.items() if k in afterglowpy_params}

    params_grb['E0']        = 10**params_grb['E0']
    params_grb['n0']        = 10**params_grb['n0']
    params_grb['epsilon_e'] = 10**params_grb['epsilon_e']
    params_grb['epsilon_B'] = 10**params_grb['epsilon_B']
    params_grb['d_L']       = params_grb['d_L'] * MPC_2_CM

    try:
        flux, xc        = compute_centroid_afterglow(t, nu, params_grb)
    except Exception as e:
        print(f"Afterglowpy error: {e}")
        return -np.inf
    centroid_pos    = compute_centroid_position(params, xc, flux) 

    return centroid_pos