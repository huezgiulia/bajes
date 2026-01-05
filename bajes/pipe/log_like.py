from __future__ import division, unicode_literals, absolute_import
import numpy as np

import logging
logger = logging.getLogger(__name__)

from scipy.special import i0e

from . import erase_init_wrapper
from ..inf.likelihood import Likelihood

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

from bajes import MPC_2_CM

def upper_limit(sigma, data, model = None):
    '''
    sigma >  0 corresponds to magnitude data below threshold: the sigma are good and not touched
    sigma <= 0 corresponds to magnitude data above threshold:
          with mag_diff = mag_model - mag_data, thus
          large_sigma_mask: mag_diff >= 0 corresponds to the model being above threshold as well,
                            sigma cannot discriminate the model and a large sigma value is returned
          abs_sigma_mask:   mag_diff <  0 corresponds to the model being below threshold,
                            sigma can discriminate the model and the absolute value of sigma is returned
    '''
    sigma = np.array(sigma).copy()
    for i in range(0,len(sigma)):
        if sigma[i] <= 0:
            if np.any(model == None):
                mag_diff = - 1
            else: mag_diff =  model[i] - data[i]
            if mag_diff >= 0:
                sigma[i] = 1e6
            else:
                if sigma[i] < 0:
                    sigma[i] = np.abs(sigma[i])

                else:
                    flux = 10**(- 0.4 * (data[i] + 48.6))
                    flux_low = flux - 0.1 * flux
                    flux_high = flux + 0.1 * flux
                    mag_low = -2.5 * np.log10(flux_high) - 48.6
                    mag_high = -2.5 * np.log10(flux_low) - 48.6
                    sigma[i] = mag_high - mag_low
    return sigma

# GRAVITATIONAL-WAVE LIKELIHOOD
# Gaussian Likelihood function:
# -0.5 (d-h|d-h) = Re(d|h) - 0.5 (d|d) - 0.5 (h|h)

class GWLikelihood(Likelihood):
    """
        Log-likelihood object,
        it assumes that the data are evaluated on the same frequency axis (given as input)
    """

    def __init__(self, ifos, datas, dets, noises,
                 freqs, srate, seglen, approx,
                 nspcal=0, spcal_freqs=None,
                 nweights=0, len_weights=None,
                 marg_phi_ref=False, marg_time_shift=False,
                 roq=None,
                 **kwargs):

        # run standard initialization
        super(GWLikelihood, self).__init__()

        # set data properties
        self.ifos   = ifos
        self.dets   = dets

        # Load ROQ object containing both frequency axes and weights for all detectors.
        self.roq = roq

        # store information
        self.nspcal = nspcal
        self.nweights = nweights

        # set marginalization flags
        self.marg_phi_ref       = marg_phi_ref
        self.marg_time_shift    = marg_time_shift

        # Compatibility check.
        if((self.roq is not None) and (self.marg_time_shift)):
            logger.error("Time-shift marginalization has not been implemented with the ROQ approximation.")
            raise AttributeError("Time-shift marginalization has not been implemented with the ROQ approximation.")

        n_freqs     = None
        f_min_check = None
        f_max_check = None

        # iterate over detectors
        for ifo in self.ifos:

            self.dets[ifo].store_measurement(datas[ifo], noises[ifo],
                                             nspcal, spcal_freqs,
                                             nweights, len_weights)

            if f_min_check == None:
                f_min_check = datas[ifo].f_min
            else:
                if datas[ifo].f_min != f_min_check:
                    logger.error("Input f_min of data and model do not match in detector {}.".format(ifo))
                    raise ValueError("Input f_min of data and model do not match in detector {}.".format(ifo))

            if f_max_check == None:
                f_max_check = datas[ifo].f_max
            else:
                if datas[ifo].f_max != f_max_check:
                    logger.error("Input f_max of data and model do not match in detector {}.".format(ifo))
                    raise ValueError("Input f_max of data and model do not match in detector {}.".format(ifo))

            if n_freqs == None:
                n_freqs = len(datas[ifo].freqs)
            else:
                if len(datas[ifo].freqs) != n_freqs:
                    logger.error("Number of data samples does not match in detector {}.".format(ifo))
                    raise ValueError("Number of data samples does not match in detector {}.".format(ifo))

            if datas[ifo].seglen != seglen:
                logger.error("Input seglen of data and model do not match in detector {}.".format(ifo))
                raise ValueError("Input seglen of data and model do not match in detector {}.".format(ifo))

            self.logZ_noise += -0.5 * self.dets[ifo]._dd
            self.Nfr        = n_freqs
            mask            = datas[ifo].mask

        # initialize waveform generator
        from ..obs.gw.waveform import Waveform
        if self.roq is not None: self.wave = erase_init_wrapper(Waveform(self.roq['freqs_join'], srate, seglen, approx))
        else:                    self.wave = erase_init_wrapper(Waveform(freqs[mask],            srate, seglen, approx))

        if self.roq is not None and self.wave.domain == 'time':
            logger.error("ROQ is available only with frequency-domain waveforms.")
            raise ValueError("ROQ is available only with frequency-domain waveforms.")

    def log_like(self, params):
        """
            log-likelihood function
        """

        # compute waveform
        logger.debug("Generating waveform for {}".format(params))
        wave    = self.wave.compute_hphc(params, roq=self.roq)
        logger.debug("Waveform generated".format(params))

        # if hp, hc == [None], [None]
        # the requested parameters are unphysical
        # Then, return -inf
        if not any(wave.plus):
            logger.warning("Likelihood method returned NaN for the set of parameters: {}.".format(params))
            return -np.inf

        if(np.any(np.isnan(wave.plus)) or np.any(np.isnan(wave.cross))):
            logger.warning('Nans in the waveform, with the configuration: {}. Returning -inf in the likelihood.'.format(params))
            return -np.inf
        if(np.any(np.isinf(wave.plus)) or np.any(np.isinf(wave.cross))):
            logger.warning('Infinities in the waveform, with the configuration: {}. Returning -inf in the likelihood.'.format(params))
            return -np.inf

        hh = 0.
        dd = 0.
        _psd_fact = 0.

        if self.marg_time_shift:

            dh_arr = np.zeros(self.Nfr, dtype=complex)

            # compute inner products
            for ifo in self.ifos:
                logger.debug("Projecting over {}".format(ifo))
                dh_arr_thisifo, hh_thisifo, dd_thisifo, _psdf = self.dets[ifo].compute_inner_products(wave, params, self.wave.domain, psd_weight_factor=True)
                dh_arr = dh_arr + np.fft.fft(dh_arr_thisifo)
                hh += np.real(hh_thisifo)
                dd += np.real(dd_thisifo)
                _psd_fact += _psdf

            # evaluate logL
            logger.debug("Estimating likelihood")
            if self.marg_phi_ref:
                abs_dh  = np.abs(dh_arr)
                I0_dh   = np.log(i0e(abs_dh)) + abs_dh
                R       = logsumexp(I0_dh-np.log(self.Nfr))
            else:
                re_dh   = np.real(dh_arr)
                R       = logsumexp(re_dh-np.log(self.Nfr))

        else:

            dh = 0.+0.j

            # compute inner products
            for ifo in self.ifos:
                logger.debug("Projecting over {}".format(ifo))
                dh_arr_thisifo, hh_thisifo, dd_thisifo, _psdf = self.dets[ifo].compute_inner_products(wave, params, self.wave.domain, psd_weight_factor=True, roq=self.roq)
                # In the ROQ case, the sum was already taken when computing the scalar product with the weights.
                if self.roq is not None: dh += (dh_arr_thisifo)
                else:                    dh += (dh_arr_thisifo).sum()
                hh += np.real(hh_thisifo)
                dd += np.real(dd_thisifo)
                _psd_fact += _psdf

            # evaluate logL
            logger.debug("Estimating likelihood")
            if self.marg_phi_ref:
                abs_dh = np.abs(dh)
                R      = np.log(i0e(abs_dh)) + abs_dh
            else:
                R      = np.real(dh)

        logL = - 0.5*(hh + dd) + R - self.logZ_noise - 0.5*_psd_fact
        if np.isnan(logL):
            logL = -np.inf
        return logL

# KILO-NOVA LIKELIHOOD
# Gaussian Likelihood function:
# -0.5 (|d-L|/s)**2
class KNLikelihood(Likelihood):

    def __init__(self, filters, approx, priors,
                 prior_grid=900, kind='linear',
                 v_min=1.e-7, n_v=400,
                 n_time=400, t_start=1., t_scale='linear',
                 use_calib_sigma_lc=False,
                 **kwargs):

        # run standard initialization
        super(KNLikelihood, self).__init__()

        # set data properties
        self.filters = filters

        # compute data normalization
        self.logZ_noise = -0.5*sum([np.power(self.filters.magnitudes[bi]/self.filters.mag_stdev[bi],2.).sum() for bi in self.filters.bands])
        self.logNorm    = -0.5*sum([np.log(2*np.pi*self.filters.mag_stdev[bi]**2).sum() for bi in self.filters.bands])

        # initilize time axis for lightcurve model
        if t_start > 86400:
            logger.warning("Initial time for lightcurve evaluation is larger than a day (86400 s). Setting t_start to 1h")
            t_start = 3600

        # the time axis passed to the lightcurve goes from t_start (~0) to the size of the measurement times
        # subsequently (line 489) the time axis is rescaled such that t=0 goes to t_gps
        t_size = np.max(filters.all_times)- np.min(filters.all_times)
        if 'time_shift' in priors.names:
            ip = priors.names.index('time_shift')
            t_size += priors.bounds[ip][1]-priors.bounds[ip][0]

        if t_scale=='linear':
            t_axis  = np.linspace(t_start, t_size+t_start, n_time)
        elif t_scale=='geom':
            t_axis  = np.geomspace(t_start, t_size+t_start, n_time)
        elif t_scale=='log':
            t_axis  = np.logspace(np.log10(t_start), np.log10(t_size+t_start), num=n_time)
        elif t_scale=='mixed':
            t1      = np.logspace(np.log10(t_start), np.log10(t_size+t_start), num=n_time//2)
            dt      = t_size/(2+n_time/2)
            t2      = np.linspace(t_start+dt, t_size+t_start-dt, n_time//2)
            t_axis  = np.sort(np.concatenate(t1,t2))
        else:
            raise ValueError("Unknown property {} for t_scale variable during KNLikelihood initialization.".format(t_scale))

        # initialize lightcurve model
        from ..obs.kn.lightcurve import Lightcurve
        light_kwargs    = {'v_min': v_min, 'n_v': n_v, 't_start': t_start , 'xkn_config' : kwargs['xkn_config'], 'mkn_config' : kwargs['mkn_config']}
        self.light      = Lightcurve(times=t_axis, lambdas=filters.lambdas, approx=approx, **light_kwargs)
        self.approx = approx

        # calib_sigma flag
        self.use_calib_sigma = use_calib_sigma_lc

    def log_like(self, params):
        
        # check the dynamical velocity  parameters value (for the case with NR fit)
        # if params['vel_dynamics'] > 0.333 or params['vel_dynamics'] < 1e-4:
        #     logL = -np.inf
        #     return logL
        
        if '3-NRfits' in self.approx:
            # check the two disk_frac parameters values (for the case with NR fit)
            if params['disk_frac_sec'] + params['disk_frac_wind'] > 0.6:
                logL = -np.inf
                return logL
            
        if 'sum' in self.approx:
            # check that the sum of the component masses is the one from the fit
            from ..obs.kn.utils import NRfit_recal_mass_dyn, NRfit_recal_mass_wind
            Mtot_fit = NRfit_recal_mass_dyn(1.1975, 1.4, 254, 639, 0) + NRfit_recal_mass_wind(1.1975, 1.4, 254, 639, 0.4)
            if (np.abs(params['mej_dynamics'] + params['mej_wind'] - Mtot_fit)) > 1e-4:
                logL = -np.inf
                return logL
            
        if 'constr' in self.approx:
            from ..obs.kn.utils import NRfit_recal_mass_dyn_new, NRfit_recal_mass_sec_new, NRfit_recal_vel_dyn_new
            sum_component = NRfit_recal_mass_dyn_new(params['mchirp'], params['q'], params['lambda1'], params['lambda2'], 0) + NRfit_recal_mass_sec_new(params['mchirp'], params['q'], params['lambda1'], params['lambda2'], 0.7)
            if (params['mej_isotropic1'] + params['mej_isotropic2']) > sum_component:
                logL = -np.inf
                return logL
            vel_component = NRfit_recal_vel_dyn_new(params['mchirp'], params['q'], params['lambda1'], params['lambda2'], 0)
            if np.sqrt((params['vel_isotropic1']**2 + params['vel_isotropic2']**2)/2) > vel_component:
                logL = -np.inf
                return logL

        # compute lightcurve

        # If the used model is one inside bajes, 'mags' is a magnitudes dictionary
        # If the used model is one inside xkn, 'mags' is a magnitudes AND times dictionary
        mags    = self.light.compute_mag(params)
        logL    = 0.

        if self.use_calib_sigma:
            for bi in self.filters.bands:

                if params['xkn_config'] == None:  # bajes model
                    lambda_bi = bi
                    interp_mag  = np.interp(self.filters.times[bi], self.light.times+params['t_gps'], mags[lambda_bi])
                
                else: # xkn model
                    # tranform keys from band names into lambdas[nm] (ONLY FOR XKN MODELS)
                    lambda_bi = int(self.filters.lambdas[bi]*1e9)
                    interp_mag  = np.interp(self.filters.times[bi], mags[lambda_bi]['time']+params['t_gps'], mags[lambda_bi]['mag'])

                sigma2      = self.filters.mag_stdev[bi]**2. + np.exp(params['log_sigma_mag_{}'.format(bi)])**2.
                residuals   = (((self.filters.magnitudes[bi]-interp_mag))**2.)/sigma2
                logL       += -0.5*(residuals + np.log(2*np.pi*sigma2)).sum()

        else:
            for bi in self.filters.bands:

                if params['xkn_config'] == None:  # bajes model
                    lambda_bi = bi
                    interp_mag  = np.interp(self.filters.times[bi], self.light.times+params['t_gps'], mags[lambda_bi])
                
                else: # xkn model
                    # tranform keys from band names into lambdas[nm] (ONLY FOR XKN MODELS)
                    lambda_bi = int(self.filters.lambdas[bi]*1e9)
                    interp_mag  = np.interp(self.filters.times[bi], mags[lambda_bi]['time']+params['t_gps'], mags[lambda_bi]['mag'])

                residuals   = ((self.filters.magnitudes[bi]-interp_mag)/self.filters.mag_stdev[bi])**2.
                logL       += -0.5*residuals.sum() 
            logL += self.logNorm
        if np.isnan(logL):
            logL = -np.inf
        return logL


# GRB LIKELIHOOD
# Gaussian Likelihood function:
# -0.5 (|d-L|/s)**2
class GRBLikelihood(Likelihood):

    def __init__(self, filters, approx,
                 **kwargs):

        # run standard initialization
        super(GRBLikelihood, self).__init__()

        # set data properties
        self.filters = filters

        # self.logZ_noise = -0.5*sum([np.power(self.filters.magnitudes[bi]/self.filters.mag_stdev[bi],2.).sum() for bi in self.filters.nu])
        self.logNorm    = -0.5*sum([np.log(2*np.pi*upper_limit(self.filters.mag_stdev[bi],self.filters.magnitudes[bi])**2).sum() for bi in self.filters.nu])

        # initialize lightcurve model
        from ..obs.grb.lightcurve import GRB_Lightcurve
        self.light      = GRB_Lightcurve(times=filters.all_times, nu=filters.nu, approx=approx, **kwargs)

    def log_like(self, params):

        # compute lightcurve
        afterglowpy_params  = ['thetaObs', 'thetaCore', 'E0', 'n0', 'p', 'epsilon_e', 'epsilon_B', 'thetaWing', 'jetType', 'xi_N', 'd_L', 'z']
        params_grb          = {k: v for k, v in params.items() if k in afterglowpy_params}
        if 'distance' in params:
            params_grb['d_L'] = params['distance'] * MPC_2_CM
        if 'cos_iota' in params:
           params_grb['thetaObs'] = np.pi - np.arccos(params['cos_iota'])
        try: 
            mags    = self.light.compute_mag(params_grb)
            logL    = 0.

            for bi in self.filters.nu:
                lambda_bi = bi
                interp_mag  = np.interp(self.filters.times[bi], self.light.times, mags[lambda_bi])
                residuals = ((self.filters.magnitudes[bi]- interp_mag)/upper_limit(self.filters.mag_stdev[bi], self.filters.magnitudes[bi], interp_mag))**2.
                logL       += -0.5*residuals.sum() 
            logL += self.logNorm

            return logL
        except Exception as e:
           logger.error(f"Afterglowpy error: {e}")
           return -np.inf
