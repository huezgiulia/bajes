#!/usr/bin/env python
from __future__ import division, unicode_literals, absolute_import
import numpy as np

# logger
import logging
logger = logging.getLogger(__name__)

from .utils import *

def initialize_amlikelihood_kwargs(opts):

    from ..obs.am.filter import Filter

    # initialize wavelength dictionary for photometric bands
    nus = {}
    if len(opts.am_nus) == 0:

        # if nus are not given use the standard ones
        from ..obs.am import __photometric_bands__ as ph_bands
        for bi in opts.am_bands:
            if bi in list(ph_bands.keys()):
                nus[bi] = ph_bands[bi]
            else:
                logger.error("Unknown photometric band {}. Please use the wave-length option (lambda) to select the band.".format(bi))
                raise ValueError("Unknown photometric band {}. Please use the wave-length option (lambda) to select the band.".format(bi))

    else:
        # check bands
        if len(opts.am_bands) != len(opts.am_nus):
            logger.error("Number of band names does not match the number of wave-length. Please give in input the same number of arguments in the respective order.")
            raise ValueError("Number of band names does not match the number of wave-length. Please give in input the same number of arguments in the respective order.")

        for bi,li in zip(opts.am_bands, opts.am_nus):
            nus[bi] = li

    # initialize grb keyword arguments
    l_kwargs = {}
    l_kwargs['approx']              = opts.am_approx
    l_kwargs['filters']             = Filter(opts.mag_folder_am, nus)

    # set intrinsic parameters bounds
    ra_bounds   = [opts.ra_min, opts.ra_max]
    dec_bounds  = [opts.dec_min, opts.dec_max]
    pa_bounds   = [opts.pa_min, opts.pa_max]

            
    # define priors
    priors = initialize_amprior(approx=opts.am_approx, bands=opts.am_bands,
                                ra_bounds=ra_bounds,dec_bounds=dec_bounds,pa_bounds=pa_bounds,
                                t_gps=opts.t_gps,
                                time_shift_bounds=[opts.time_shift_min, opts.time_shift_max],
                                fixed_names=opts.fixed_names, fixed_values=opts.fixed_values,
                                )
    
    # save observations in pickle
    cont_kwargs = {'filters': l_kwargs['filters']}
    save_container(opts.outdir+'/am_obs.pkl', cont_kwargs)
    return l_kwargs, priors

def initialize_amprior(approx,
                       bands,
                       ra_bounds,    
                       dec_bounds,
                       pa_bounds,
                       t_gps,
                       time_shift_bounds    = None,
                       fixed_names          = [],
                       fixed_values         = [],
                    ):

    from ..inf.prior import Prior, Parameter, Variable, Constant

    # initializing dictionary for wrap up all information
    dict = {}

    # setting parameters
    if ra_bounds[0] == None and ra_bounds[1] == None:
        dict['ra']   = Parameter(name='ra',
                                    min=0,
                                    max=2*np.pi)
        logger.warning("Requested bounds for ra parameter is empty. Setting standard bound [0, 2pi]")
    else:
        dict['ra']        = Parameter(name='ra',
                                    min=ra_bounds[0], 
                                    max=ra_bounds[1])
        
    if dec_bounds[0] == None and dec_bounds[1] == None:
        dict['dec']   = Parameter(name='dec',
                                    min=0,
                                    max=2*np.pi)
        logger.warning("Requested bounds for dec parameter is empty. Setting standard bound [0, 2pi]")
    else:
        dict['dec']        = Parameter(name='dec',
                                    min=dec_bounds[0], 
                                    max=dec_bounds[1])
    
    if pa_bounds[0] == None and pa_bounds[1] == None:
        dict['pa']   = Parameter(name='pa',
                                    min=-10,
                                    max=10)
        logger.warning("Requested bounds for position angle parameter is empty. Setting standard bound [-10, 10]mas")
    else:
        dict['pa']        = Parameter(name='pa',
                                    min=pa_bounds[0], 
                                    max=pa_bounds[1])

    # set fixed parameters
    if len(fixed_names) != 0 :
        assert len(fixed_names) == len(fixed_values)
        for ni,vi in zip(fixed_names,fixed_values) :
            if ni not in list(dict.keys()):
                logger.warning("Requested fixed parameter ({}={}) is not in the list of all parameters. The command will be ignored.".format(ni,vi))
            else:
                dict[ni] = Constant(ni, vi)

    params, variab, const = fill_params_from_dict(dict)

    logger.info("Setting parameters for sampling ...")
    for pi in params:
        logger.info(" - {} in range [{:.2f},{:.2f}]".format(pi.name , pi.bound[0], pi.bound[1]))

    # logger.info("Setting variable properties ...")

    logger.info("Setting constant properties ...")
    for ci in const:
        logger.info(" - {} fixed to {}".format(ci.name , ci.value))

    logger.info("Initializing prior ...")

    return Prior(parameters=params, variables=variab, constants=const)
