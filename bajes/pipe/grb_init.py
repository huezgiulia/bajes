#!/usr/bin/env python
from __future__ import division, unicode_literals, absolute_import
import numpy as np

# logger
import logging
logger = logging.getLogger(__name__)

from .utils import *

def initialize_grblikelihood_kwargs(opts):

    from ..obs.grb.filter import Filter

    # initialize wavelength dictionary for photometric bands
    nus = {}
    if len(opts.grb_nus) == 0:

        # if nus are not given use the standard ones
        from ..obs.grb import __photometric_bands__ as ph_bands
        for bi in opts.grb_bands:
            if bi in list(ph_bands.keys()):
                nus[bi] = ph_bands[bi]
            else:
                logger.error("Unknown photometric band {}. Please use the wave-length option (lambda) to select the band.".format(bi))
                raise ValueError("Unknown photometric band {}. Please use the wave-length option (lambda) to select the band.".format(bi))

    else:
        # check bands
        if len(opts.grb_bands) != len(opts.grb_nus):
            logger.error("Number of band names does not match the number of wave-length. Please give in input the same number of arguments in the respective order.")
            raise ValueError("Number of band names does not match the number of wave-length. Please give in input the same number of arguments in the respective order.")

        for bi,li in zip(opts.grb_bands, opts.grb_nus):
            nus[bi] = li

    # initialize grb keyword arguments
    l_kwargs = {}
    l_kwargs['approx']              = opts.grb_approx
    l_kwargs['filters']             = Filter(opts.mag_folder_grb, nus)

    # set intrinsic parameters bounds
    theta_obs_bounds    = [opts.theta_obs_min, opts.theta_obs_max]
    theta_core_bounds   = [opts.theta_core_min, opts.theta_core_max]
    e0_bounds           = [opts.e0_min, opts.e0_max]
    n0_bounds           = [opts.n0_min, opts.n0_max]
    p_bounds            = [opts.p_min, opts.p_max]
    epsilone_bounds     = [opts.epsilone_min, opts.epsilone_max]
    epsilonB_bounds     = [opts.epsilonB_min, opts.epsilonB_max]
    theta_wing_bounds   = [opts.theta_wing_min, opts.theta_wing_max]
    xin_bounds          = [opts.xin_min, opts.xin_max]
    b_bounds            = [opts.b_min, opts.b_max]
            
    # define priors
    priors = initialize_grbprior(approx=opts.grb_approx, bands=opts.grb_bands, model=opts.grb_model,
                                theta_obs_bounds=theta_obs_bounds, theta_core_bounds=theta_core_bounds,
                                e0_bounds=e0_bounds, n0_bounds=n0_bounds, p_bounds=p_bounds,
                                epsilone_bounds=epsilone_bounds, epsilonB_bounds=epsilonB_bounds,
                                theta_wing_bounds=theta_wing_bounds, xin_bounds=xin_bounds,b_bounds=b_bounds,
                                t_gps=opts.t_gps,
                                dist_max=opts.dist_max, dist_min=opts.dist_min,
                                dist_flag=opts.dist_flag,
                                time_shift_bounds=[opts.time_shift_min, opts.time_shift_max],
                                fixed_names=opts.fixed_names, fixed_values=opts.fixed_values,
                                )
    
    # save observations in pickle
    cont_kwargs = {'filters': l_kwargs['filters']}
    save_container(opts.outdir+'/grb_obs.pkl', cont_kwargs)
    return l_kwargs, priors

def initialize_grbprior(approx,
                       bands,
                       model,
                       theta_obs_bounds,
                       theta_core_bounds,
                       e0_bounds,
                       n0_bounds,
                       p_bounds ,
                       epsilone_bounds,
                       epsilonB_bounds,
                       theta_wing_bounds,
                       xin_bounds,
                       b_bounds,
                       t_gps,
                       dist_max             = None,
                       dist_min             = None,                    
                       dist_flag            = False,
                       time_shift_bounds    = None,
                       fixed_names          = [],
                       fixed_values         = [],
                    ):

    from ..inf.prior import Prior, Parameter, Variable, Constant

    # initializing dictionary for wrap up all information
    dict = {}

    # setting jetType
    if model == 'TopHat':
        dict['jetType'] = Constant('jetType', -1)
    elif model == 'Gaussian':
        dict['jetType'] = Constant('jetType', 0)
    elif model == 'PowerLaw':
        dict['jetType'] = Constant('jetType', 4)
    else:
        logger.error("Unknown jet type. Please use 'TopHat', 'PowerLaw' or 'Gaussian'.")
        raise RuntimeError("Unknown jet type. Please use 'TopHat', 'PowerLaw' or 'Gaussian'.")

    # setting parameters
    if theta_obs_bounds[0] == None and theta_obs_bounds[1] == None:
        dict['thetaObs']   = Parameter(name='thetaObs',
                                    min=0,
                                    max=1.57)
        logger.warning("Requested bounds for theta-obs parameter is empty. Setting standard bound [0, pi]")
    else:
        dict['thetaObs']        = Parameter(name='thetaObs',
                                    min=theta_obs_bounds[0], 
                                    max=theta_obs_bounds[1])

    if theta_core_bounds[0] == None and theta_core_bounds[1] == None:
        dict['thetaCore']   = Parameter(name='thetaCore',
                                    min=0, 
                                    max=1.57)
        logger.warning("Requested bounds for theta-core parameter is empty. Setting standard bound [0, pi]")
    else:
        dict['thetaCore']   = Parameter(name='thetaCore',
                                    min=theta_core_bounds[0], 
                                    max=theta_core_bounds[1])

    if e0_bounds[0] == None and e0_bounds[1] == None:
        dict['E0']          = Parameter(name='E0',
                                    min=45,
                                    max=57)
        logger.warning("Requested bounds for log E0 parameter is empty. Setting standard bound [45, 57]")
    else:   
        dict['E0']          = Parameter(name='E0',
                                    min=e0_bounds[0], 
                                    max=e0_bounds[1])

    if n0_bounds[0] == None and n0_bounds[1] == None:
        dict['n0']          = Parameter(name='n0',
                                    min=-10,
                                    max=10)
        logger.warning("Requested bounds for log n0 parameter is empty. Setting standard bound [-10,10]")
    else:
        dict['n0']          = Parameter(name='n0',
                                    min=n0_bounds[0], 
                                    max=n0_bounds[1])   

    if p_bounds[0] == None and p_bounds[1] == None:
        dict['p']           = Parameter(name='p',
                                    min=2,
                                    max=3)
        logger.warning("Requested bounds for p parameter is empty. Setting standard bound [2,3]")
    else:
        dict['p']           = Parameter(name='p',
                                    min=p_bounds[0], 
                                    max=p_bounds[1])

    if epsilone_bounds[0] == None and epsilone_bounds[1] == None:
        dict['epsilon_e']   = Parameter(name='epsilon_e',
                                    min=-10,
                                    max=0)
        logger.warning("Requested bounds for log epsilon_e parameter is empty. Setting standard bound [-10,0]")
    else:
        dict['epsilon_e']   = Parameter(name='epsilon_e',
                                    min=epsilone_bounds[0], 
                                    max=epsilone_bounds[1])
        
    if epsilonB_bounds[0] == None and epsilonB_bounds[1] == None:
        dict['epsilon_B']          = Parameter(name='epsilon_B',
                                    min=-10,
                                    max=0)
        logger.warning("Requested bounds for log epsilon_B parameter is empty. Setting standard bound [-10,0]")
    else:
        dict['epsilon_B']   = Parameter(name='epsilon_B',
                                    min=epsilonB_bounds[0], 
                                    max=epsilonB_bounds[1])  
 
    if xin_bounds[0] == None and xin_bounds[1] == None:
        dict['xi_N']        = Parameter(name='xi_N',
                                    min=0,
                                    max=1)
        logger.warning("Requested bounds for xi_N parameter is empty. Setting standard bound [0,1]")
    else: 
        dict['xi_N']        = Parameter(name='xi_N',
                                    min=xin_bounds[0], 
                                    max=xin_bounds[1])

    if model == 'PowerLaw':
        if b_bounds[0] == None and b_bounds[1] == None:
            dict['b']           = Parameter(name='b',
                                        min=0, 
                                        max=10)
            logger.warning("Requested bounds for b parameter is empty. Setting standard bound [0,10]")
        else:
            dict['b']           = Parameter(name='b',
                                        min=b_bounds[0],
                                    max=b_bounds[1])

    if model == 'PowerLaw' or model == 'Gaussian':
        if theta_wing_bounds[0] == None and theta_wing_bounds[1] == None:
            dict['thetaWing']   = Parameter(name='thetaWing',
                                        min=0,
                                        max=1.57)
        else:
            dict['thetaWing']   = Parameter(name='thetaWing',
                                        min=theta_wing_bounds[0], 
                                        max=theta_wing_bounds[1])
            logger.warning("Requested bounds for theta-wing parameter is empty. Setting standard bound [0, pi]")

    dict['z']               = Parameter(name='z',
                                    min=0.001,
                                    max=10)

    # setting distance
    if dist_min == None and dist_max == None:
        logger.warning("Requested bounds for distance parameter is empty. Setting standard bound [10,1000] Mpc")
        dist_min = 10.
        dist_max = 1000.
    elif dist_min == None:
        logger.warning("Requested lower bounds for distance parameter is empty. Setting standard bound 10 Mpc")
        dist_min = 10.
    elif dist_max == None:
        logger.warning("Requested bounds for distance parameter is empty. Setting standard bound 1 Gpc")
        dist_max = 1000.

    if dist_flag=='log':
        dict['d_L']   = Parameter(name='d_L',
                                       min=dist_min,
                                       max=dist_max,
                                       prior='log-uniform')
    elif dist_flag=='vol':
        dict['d_L']   = Parameter(name='d_L',
                                       min=dist_min,
                                       max=dist_max,
                                       prior='quadratic')
    else:
        logger.error("Invalid distance flag for Prior initialization. Please use 'vol', 'com' or 'log'.")
        raise RuntimeError("Invalid distance flag for Prior initialization. Please use 'vol', 'com' or 'log'.")

    # setting time_shift
    if time_shift_bounds == None:
        logger.warning("Requested bounds for time_shift parameter is empty. Setting standard bound [-1.0,+1.0] day")
        time_shift_bounds  = [-86400.,+86400.]

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
