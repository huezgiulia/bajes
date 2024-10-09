#!/usr/bin/env python
from __future__ import division, unicode_literals, absolute_import
import numpy as np

# logger
import logging
logger = logging.getLogger(__name__)

from .utils import *

def initialize_knlikelihood_kwargs(opts):

    from ..obs.kn.filter import Filter

    # initial check
    # NOTE: this works only with GrossmanKBP models
    ncomps = int(opts.kn_approx.split('-')[1])

    if (ncomps != len(opts.mej_min)) or (ncomps != len(opts.mej_max)):
        logger.error("Number of components does not match the number of ejected mass bounds. Please give in input the same number of arguments in the respective order.")
        raise ValueError("Number of components does not match the number of ejected mass bounds. Please give in input the same number of arguments in the respective order.")
    if (ncomps != len(opts.vel_min)) or (ncomps != len(opts.vel_max)):
        logger.error("Number of components does not match the number of velocity bounds. Please give in input the same number of arguments in the respective order.")
        raise ValueError("Number of components does not match the number of velocity bounds. Please give in input the same number of arguments in the respective order.")
    if (ncomps != len(opts.opac_min)) or (ncomps != len(opts.opac_max)):
        logger.error("Number of components does not match the number of opacity bounds. Please give in input the same number of arguments in the respective order.")
        raise ValueError("Number of components does not match the number of opacity bounds. Please give in input the same number of arguments in the respective order.")  
    if (len(opts.step_angle_mej_min) != len(opts.step_angle_mej_max)):
        logger.error("Number of lower bounds does not match the number upper bounds for step_angle_mej. Please give in input the same number of arguments in the respective order.")
        raise ValueError("Number of lower bounds does not match the number oupper bounds. Please give in input the same number of arguments in the respective order.")     
    if (len(opts.vel_high_min) != len(opts.vel_high_max)):
        logger.error("Number of lower bounds does not match the number upper bounds for vel_high. Please give in input the same number of arguments in the respective order.")
        raise ValueError("Number of lower bounds does not match the number oupper bounds. Please give in input the same number of arguments in the respective order.")       
    if (len(opts.step_angle_vel_min) != len(opts.step_angle_vel_max)):
        logger.error("Number of lower bounds does not match the number upper bounds for step_angle_vel. Please give in input the same number of arguments in the respective order.")
        raise ValueError("Number of lower bounds does not match the number oupper bounds. Please give in input the same number of arguments in the respective order.")       
    if (len(opts.opac_high_min) != len(opts.opac_high_max)):
        logger.error("Number of lower bounds does not match the number upper bounds for opac_high. Please give in input the same number of arguments in the respective order.")
        raise ValueError("Number of lower bounds does not match the number oupper bounds. Please give in input the same number of arguments in the respective order.")       
    if (len(opts.step_angle_op_min) != len(opts.step_angle_op_max)):
        logger.error("Number of lower bounds does not match the number upper bounds for step_angle_op. Please give in input the same number of arguments in the respective order.")
        raise ValueError("Number of lower bounds does not match the number oupper bounds. Please give in input the same number of arguments in the respective order.")       
    


    if opts.time_shift_max == None:
        logger.warning("Upper bound for time shift is not provided. Setting default to 1 hr.")
        opts.time_shift_max = 3600

    if opts.time_shift_min == None:
        opts.time_shift_min = -opts.time_shift_max

    # initialize wavelength dictionary for photometric bands
    lambdas = {}
    if len(opts.lambdas) == 0:

        # if lambdas are not given use the standard ones
        from ..obs.kn import __photometric_bands__ as ph_bands
        for bi in opts.bands:
            if bi in list(ph_bands.keys()):
                lambdas[bi] = ph_bands[bi]
            else:
                logger.error("Unknown photometric band {}. Please use the wave-length option (lambda) to select the band.".format(bi))
                raise ValueError("Unknown photometric band {}. Please use the wave-length option (lambda) to select the band.".format(bi))

    else:
        # check bands
        if len(opts.bands) != len(opts.lambdas):
            logger.error("Number of band names does not match the number of wave-length. Please give in input the same number of arguments in the respective order.")
            raise ValueError("Number of band names does not match the number of wave-length. Please give in input the same number of arguments in the respective order.")

        for bi,li in zip(opts.bands, opts.lambdas):
            lambdas[bi] = li

    # initialize likelihood keyword arguments
    l_kwargs = {}
    l_kwargs['approx']              = opts.kn_approx
    l_kwargs['filters']             = Filter(opts.mag_folder, lambdas, dered=opts.dered)
    l_kwargs['v_min']               = opts.vgrid_min
    l_kwargs['n_v']                 = opts.n_v
    l_kwargs['n_time']              = opts.n_t
    l_kwargs['t_start']             = opts.init_t
    l_kwargs['t_scale']             = opts.t_scale
    l_kwargs['use_calib_sigma_lc']  = opts.use_calib_sigma_lc
    
    # inizialize MKN class
    from xkn import MKN, MKNConfig
    if not opts.xkn:
            logger.info("No config file was passed for MKN inizialization.")
            l_kwargs['mkn_config']          = None
            l_kwargs['xkn_config']          = None

    else:
        config_path  = os.path.abspath(opts.xkn)
        tag             = config_path.split('.')[-1]

        if tag == 'ini':
            logger.info("Initialize MKNConfig object from the config file")
            mkn_config = MKNConfig(config_path)
            l_kwargs['mkn_config']          = mkn_config

            logger.info("Initialize MKN object from the read config-parameters")
            mkn = MKN(*mkn_config.get_params(), log_level="WARNING")
            l_kwargs['xkn_config']          = mkn
        
        else: 
            logger.error("Impossible to initialize MKN object from {} file. Use ini.".format(tag))
            l_kwargs['mkn_config']          = None
            l_kwargs['xkn_config']          = None



    # set intrinsic parameters bounds
    mej_bounds  = [[mmin, mmax] for mmin, mmax in zip(opts.mej_min, opts.mej_max)]
    vel_bounds  = [[vmin, vmax] for vmin, vmax in zip(opts.vel_min, opts.vel_max)]
    opac_bounds = [[omin, omax] for omin, omax in zip(opts.opac_min, opts.opac_max)] 

    step_angle_mej_bounds = [None] * ncomps
    if opts.step_angle_mej_min != [] and opts.step_angle_mej_max != []:
        for min, max in zip(opts.step_angle_mej_min, opts.step_angle_mej_max):
            where_min = int(min.split('-')[1])
            where_max = int(max.split('-')[1])
            if where_min == where_max:
                step_angle_mej_bounds[where_max-1] = [float(min.split('-')[0]), float(max.split('-')[0])]
            else:
                logger.error("Different component assigned for step_angle_mej prior bounds.")
                raise ValueError("Different component assigned for step_angle_mej prior bounds.")

    # set additive high latitude parameters bounds for velocity and relative step angle
    vel_high_bounds = [None] * ncomps
    if opts.vel_high_min != [] and opts.vel_high_max != []:
        for vmin, vmax in zip(opts.vel_high_min, opts.vel_high_max):
            where_vmin = int(vmin.split('-')[1])
            where_vmax = int(vmax.split('-')[1])
            if where_vmin == where_vmax:
                vel_high_bounds[where_vmax-1] = [float(vmin.split('-')[0]), float(vmax.split('-')[0])]
            else:
                logger.error("Different component assigned for vel_high prior bounds.")
                raise ValueError("Different component assigned for vel_high prior bounds.")

    step_angle_vel_bounds = [None] * ncomps
    if opts.step_angle_vel_min != [] and opts.step_angle_vel_max != []:
        for avmin, avmax in zip(opts.step_angle_vel_min, opts.step_angle_vel_max):
            where_avmin = int(avmin.split('-')[1])
            where_avmax = int(avmax.split('-')[1])
            if where_avmin == where_avmax:
                step_angle_vel_bounds[where_avmax-1] = [float(avmin.split('-')[0]), float(avmax.split('-')[0])]
            else:
                logger.error("Different component assigned for step_angle_vel prior bounds.")
                raise ValueError("Different component assigned for step_angle_vel prior bounds.")

    # set additive high latitude parameters bounds for opacity and relative step angle
    opac_high_bounds = [None] * ncomps
    if opts.opac_high_min != [] and opts.opac_high_max != []:
        for omin, omax in zip(opts.opac_high_min, opts.opac_high_max):
            where_omin = int(omin.split('-')[1])
            where_omax = int(omax.split('-')[1])
            if where_omin == where_omax:
                opac_high_bounds[where_omax-1] = [float(omin.split('-')[0]), float(omax.split('-')[0])]
            else:
                logger.error("Different component assigned for opac_high prior bounds.")
                raise ValueError("Different component assigned for opac_high prior bounds.")
            
    step_angle_op_bounds = [None] * ncomps
    if opts.step_angle_op_min != [] and opts.step_angle_op_max != []:
        for aomin, aomax in zip(opts.step_angle_op_min, opts.step_angle_op_max):
            where_aomin = int(aomin.split('-')[1])
            where_aomax = int(aomax.split('-')[1])
            if where_aomin == where_aomax:
                step_angle_op_bounds[where_aomax-1] = [float(aomin.split('-')[0]), float(aomax.split('-')[0])]
            else:
                logger.error("Different component assigned for step_angle_op prior bounds.")
                raise ValueError("Different component assigned for step_angle_op prior bounds.")
            
    # define priors
    priors = initialize_knprior(approx=opts.kn_approx, bands=opts.bands,
                                mej_bounds=mej_bounds, step_angle_mej_bounds=step_angle_mej_bounds,
                                vel_bounds=vel_bounds,
                                vel_high_bounds=vel_high_bounds, step_angle_vel_bounds=step_angle_vel_bounds,
                                opac_bounds=opac_bounds,
                                opac_high_bounds=opac_high_bounds, step_angle_op_bounds=step_angle_op_bounds,
                                t_gps=opts.t_gps,
                                dist_max=opts.dist_max, dist_min=opts.dist_min,
                                eps0_max=opts.eps_max,  eps0_min=opts.eps_min,
                                dist_flag=opts.dist_flag, log_eps0_flag=opts.log_eps_flag,
                                heating_sampling=opts.heat_sampling, heating_alpha=opts.heating_alpha,
                                heating_time=opts.heating_time,heating_sigma=opts.heating_sigma,
                                time_shift_bounds=[opts.time_shift_min, opts.time_shift_max],
                                fixed_names=opts.fixed_names, fixed_values=opts.fixed_values,
                                prior_grid=opts.priorgrid, kind='linear',
                                use_calib_sigma=opts.use_calib_sigma_lc,
                                sigma_max=opts.sigma_max, sigma_min=opts.sigma_min)
    

    # save observations in pickle
    cont_kwargs = {'filters': l_kwargs['filters']}
    save_container(opts.outdir+'/kn_obs.pkl', cont_kwargs)
    return l_kwargs, priors

def initialize_knprior(approx,
                       bands,
                       mej_bounds,
                       step_angle_mej_bounds,
                       vel_bounds,
                       vel_high_bounds,
                       step_angle_vel_bounds,
                       opac_bounds,
                       opac_high_bounds,
                       step_angle_op_bounds,
                       t_gps,
                       dist_max             = None,
                       dist_min             = None,
                       eps0_max             = None,
                       eps0_min             = None,
                       dist_flag            = False,
                       log_eps0_flag        = False,
                       heating_sampling     = False,
                       heating_alpha        = 1.3,
                       heating_time         = 1.3,
                       heating_sigma        = 0.11,
                       time_shift_bounds    = None,
                       fixed_names          = [],
                       fixed_values         = [],
                       prior_grid           = 2000,
                       kind                 = 'linear',
                       use_calib_sigma      = True,
                       sigma_max            = None,
                       sigma_min            = None,):

    from ..inf.prior import Prior, Parameter, Variable, Constant

    # get names of component from approximant
    if   approx=='GrossmanKBP-1-isotropic':     comps = ['isotropic']
    elif approx=='GrossmanKBP-1-equatorial':    comps = ['equatorial']
    elif approx=='GrossmanKBP-1-polar':         comps = ['polar']
    elif approx=='GrossmanKBP-2-isotropic':     comps = ['isotropic1', 'isotropic2']
    elif approx=='GrossmanKBP-2-equatorial':    comps = ['isotropic', 'equatorial']
    elif approx=='GrossmanKBP-2-polar':         comps = ['isotropic', 'polar']
    elif approx=='GrossmanKBP-2-eq+pol':        comps = ['equatorial', 'polar']
    elif 'GrossmanKBP-2-NRfits' in approx:      comps = ['dyn', 'wind']
    elif approx=='GrossmanKBP-3-isotropic':     comps = ['isotropic1', 'isotropic2', 'isotropic3']
    elif approx=='GrossmanKBP-3-anisotropic':   comps = ['isotropic', 'equatorial', 'polar']
    elif approx=='Xkn-1':                       comps = ['dynamics']
    elif approx=='Xkn-2':                       comps = ['dynamics', 'secular']
    elif approx=='Xkn-3':                       comps = ['dynamics', 'secular', 'wind']

    # initializing disctionary for wrap up all information
    dict = {}

    # setting ejecta properties for every component
    for i,ci in enumerate(comps):
        dict[f'mej_{ci:s}']  = Parameter(name='mej_{}'.format(ci), min = mej_bounds[i][0], max = mej_bounds[i][1])
        if step_angle_mej_bounds[i] != None:
            dict['step_angle_mej_{}'.format(ci)] = Parameter(name='step_angle_mej_{}'.format(ci), min = step_angle_mej_bounds[i][0], max = step_angle_mej_bounds[i][1])
        dict['vel_{}'.format(ci)]  = Parameter(name='vel_{}'.format(ci), min = vel_bounds[i][0], max = vel_bounds[i][1])
        if vel_high_bounds[i] != None and step_angle_vel_bounds[i] != None:
            dict['vel_high_{}'.format(ci)] = Parameter(name='vel_high_{}'.format(ci), min = vel_high_bounds[i][0], max = vel_high_bounds[i][1])
            dict['step_angle_vel_{}'.format(ci)] = Parameter(name='step_angle_vel_{}'.format(ci), min = step_angle_vel_bounds[i][0], max = step_angle_vel_bounds[i][1])
        dict['opac_{}'.format(ci)] =  Parameter(name='opac_{}'.format(ci), min = opac_bounds[i][0], max = opac_bounds[i][1])
        if opac_high_bounds[i] != None and step_angle_op_bounds[i] != None:
            dict['opac_high_{}'.format(ci)] = Parameter(name='opac_high_{}'.format(ci), min = opac_high_bounds[i][0], max = opac_high_bounds[i][1])
            dict['step_angle_op_{}'.format(ci)] = Parameter(name='step_angle_op_{}'.format(ci), min = step_angle_op_bounds[i][0], max = step_angle_op_bounds[i][1]) 

    # setting eps0
    if eps0_min == None and eps0_max == None:
        logger.warning("Requested bounds for heating parameter eps0 is empty. Setting standard bound [1e17,1e19].")
        eps0_min = 1.e17
        eps0_max = 5.e19
    elif eps0_min == None and eps0_max != None:
        eps0_min = 1.e17
        eps0_max = eps0_max

    if log_eps0_flag:
        dict['eps0']   = Parameter(name='eps0', min = eps0_min, max = eps0_max, prior = 'log-uniform')
    else:
        dict['eps0']   = Parameter(name='eps0', min = eps0_min, max = eps0_max)

    # set heating coefficients
    if heating_sampling:
        logger.warning("Including extra heating coefficiets in sampling using default bounds with uniform prior.")
        dict['eps_alpha']   = Parameter(name='eps_alpha',    min=1., max=10.)
        dict['eps_time']    = Parameter(name='eps_time',     min=0., max=25.)
        dict['eps_sigma']   = Parameter(name='eps_sigma',    min=1.e-5, max=50.)
    else:
        dict['eps_alpha']   = Constant('eps_alpha', heating_alpha)
        dict['eps_time']    = Constant('eps_time',  heating_time)
        dict['eps_sigma']   = Constant('eps_sigma', heating_sigma)

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
        dict['distance']   = Parameter(name='distance',
                                       min=dist_min,
                                       max=dist_max,
                                       prior='log-uniform')
    elif dist_flag=='vol':
        dict['distance']   = Parameter(name='distance',
                                       min=dist_min,
                                       max=dist_max,
                                       prior='quadratic')
    elif dist_flag=='com':
        from ..obs.utils.cosmo import Cosmology
        from .utils import _get_astropy_version
        _av = _get_astropy_version()
        if int(_av[0])>=5:
            cosmo = Cosmology(cosmo='Planck18')
        else:
            cosmo = Cosmology(cosmo='Planck18_arXiv_v2')
        dict['distance']   = Parameter(name='distance',
                                       min=dist_min,
                                       max=dist_max,
                                       func=log_prior_comoving_volume,
                                       func_kwarg={'cosmo': cosmo},
                                       interp_kwarg=interp_kwarg)
    elif dist_flag=='src':
        from ..obs.utils.cosmo import Cosmology
        from .utils import _get_astropy_version
        _av = _get_astropy_version()
        if int(_av[0])>=5:
            cosmo = Cosmology(cosmo='Planck18')
        else:
            cosmo = Cosmology(cosmo='Planck18_arXiv_v2')
        dict['distance']   = Parameter(name='distance',
                                       min=dist_min,
                                       max=dist_max,
                                       func=log_prior_sourceframe_volume,
                                       func_kwarg={'cosmo': cosmo},
                                       interp_kwarg=interp_kwarg)
    else:
        logger.error("Invalid distance flag for Prior initialization. Please use 'vol', 'com' or 'log'.")
        raise RuntimeError("Invalid distance flag for Prior initialization. Please use 'vol', 'com' or 'log'.")

    # setting time_shift
    if time_shift_bounds == None:
        logger.warning("Requested bounds for time_shift parameter is empty. Setting standard bound [-1.0,+1.0] day")
        time_shift_bounds  = [-86400.,+86400.]

    dict['time_shift']  = Parameter(name='time_shift', min=time_shift_bounds[0], max=time_shift_bounds[1])

    # setting inclination
    dict['cos_iota']   =  Parameter(name='cos_iota', min=-1., max=+1.)

    

    # use NR fits for dynamical ejecta and baryonic wind
    if 'GrossmanKBP-2-NRfits' in approx:

        logger.warning("Activating NR fits for ejecta properties. This option works only with joint KN+GW model. Please be sure you are using the correct framework.")
        # NOTE: the NR fits work only if the prior already includes the BNS parameters, i.e. mchirp, q, lambda1, lambda2.
        # These parameters are used to determined the predictions of the fits and they are automatically included by the
        # GW initialization routine. So the NR ejecta fits work only with GW+KN framework.

        dyn_tag     = comps[0]
        wind_tag    = comps[1]

        from ..obs.kn.utils import NRfit_recal_mass_dyn, NRfit_recal_vel_dyn, NRfit_recal_mass_wind

        # include calibrations and disk fracion
        dict['disk_frac']         = Parameter(name='disk_frac',         min = 0.,   max = 1.,   prior='uniform')
        dict['NR_fit_recal_mdyn'] = Parameter(name='NR_fit_recal_mdyn', min = -1.,  max = 1.,   prior='normal', mu=0., sigma=0.136)
        dict['NR_fit_recal_vdyn'] = Parameter(name='NR_fit_recal_vdyn', min = -1.,  max = 1.,   prior='normal', mu=0., sigma=0.21)

        # fix (m-dyn, v-dyn, m-wind) with NR fits
        dict['mej_{}'.format(dyn_tag)]  = Variable(name='mej_{}'.format(dyn_tag),   func=NRfit_recal_mass_dyn)
        dict['vel_{}'.format(dyn_tag)]  = Variable(name='vel_{}'.format(dyn_tag),   func=NRfit_recal_vel_dyn)
        dict['mej_{}'.format(wind_tag)] = Variable(name='mej_{}'.format(wind_tag),  func=NRfit_recal_mass_wind)

    # include theoretical error
    if use_calib_sigma:
        if sigma_min == None and sigma_max == None:
            logger.warning("Requested bounds for systematic deviation parameter is empty. Setting standard bound [-10,5]")
            sigma_min = -10.
            sigma_max = 5.
        elif sigma_min == None:
            logger.warning("Requested lower bounds for systematic deviation parameter is empty. Setting standard bound -10")
            sigma_min = -10.
        elif sigma_max == None:
            logger.warning("Requested bounds for systematic deviation parameter is empty. Setting standard bound 5")
            sigma_max = 5.
        for bi in bands:
            # use uniform prior in log_calib_sigma since it is easier for the sampler
            # sigma ~ 1/sigma (log-uniform), then log(sigma) ~ 1
            dict['log_sigma_mag_{}'.format(bi)]   = Parameter(name='log_sigma_mag_{}'.format(bi),
                                                              min = sigma_min, max = sigma_max, prior='uniform')

    # set fixed parameters
    if len(fixed_names) != 0 :
        assert len(fixed_names) == len(fixed_values)
        for ni,vi in zip(fixed_names,fixed_values) :
            if ni not in list(dict.keys()):
                logger.warning("Requested fixed parameter ({}={}) is not in the list of all parameters. The command will be ignored.".format(ni,vi))
            else:
                dict[ni] = Constant(ni, vi)

    dict['t_gps']  = Constant('t_gps', t_gps)

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
