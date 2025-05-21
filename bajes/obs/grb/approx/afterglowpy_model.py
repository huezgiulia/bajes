from __future__ import division, unicode_literals, absolute_import
import numpy as np
import afterglowpy as grb


def afterglowpy(t, nu, grb_params):
    # compute fluxes
    Fnu = grb.fluxDensity(t, nu, **grb_params)
    return Fnu


def afterglow_wrapper(t, nu, params):
    ''' Wrapper for grb model from afterglowpy.'''

    grb_params = params
    return afterglowpy(t, nu, grb_params)