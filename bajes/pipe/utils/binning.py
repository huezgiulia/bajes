"""
    Functions and methods inspired by GWBinning,
    https://bitbucket.org/dailiang8/gwbinning,
    arXiv:1806.08792v2 [astro-ph.IM] (2018),
    B. Zackay et al.
    and by bilby relative binning, 
    https://git.ligo.org/lscsoft/bilby,
    arXiv:2312.06009 [gr-qc] (2023),
    K. Krishna et al.
"""

import numpy as np

import logging
logger = logging.getLogger(__name__)

from scipy.special import i0e

from ...inf.likelihood import Likelihood
from .. import erase_init_wrapper
from ...obs.gw.detector import compute_spcalenvs

from bajes.obs.gw.waveform import PolarizationTuple
from bajes.obs.gw import Waveform

# construct frequency bins for relative binning
def setup_bins(f_full, f_lo, f_hi, eps, chi=1.):
    """
        construct frequency binning
        f_full: full frequency grid
        [f_lo, f_hi] is the frequency range one would like to use for matched filtering
        chi, eps are tunable parameters [see Barak, Dai & Venumadhav 2018]
        return the number of bins, a list of bin-edge frequencies, and their positions in the full frequency grid
    """

    f = f_full[(f_full>=f_lo)&(f_full<=f_hi)]

    # f^ga power law index
    ga = np.array([-5.0/3.0, -2.0/3.0, 1.0, 5.0/3.0, 7.0/3.0])
    dalp = chi*2.0*np.pi/np.absolute((f_lo**ga)*(np.heaviside(-ga,1)) - (f_hi**ga)*(np.heaviside(ga,1)))
    dphi = np.sum(np.array([ np.sign(ga[i])*dalp[i]*f**ga[i] for i in range(len(ga)) ]), axis=0)
    Dphi = dphi - dphi[0]
    # now construct frequency bins
    Nbin = int(Dphi[-1]//eps)

    last_index = -1
    bin_inds = []
    bin_freqs = []
    for i in range(Nbin + 1):
        bin_index = np.where(Dphi >= ((i / Nbin) * Dphi[-1]))[0][0]
        if bin_index == last_index:
            continue
        bin_freq = f[bin_index]
        last_index = bin_index
        bin_index = np.where(f_full >= bin_freq)[0][0]
        bin_inds.append(bin_index)
        bin_freqs.append(bin_freq)

    fbin = np.array(bin_freqs)
    fbin_ind = np.array(bin_inds)

    return (Nbin, fbin, fbin_ind)

def noise_weighted_inner_product(aa, bb, power_spectral_density, duration):
    """
    Adapted from bilby
    Calculate the noise weighted inner product between two arrays.

    Parameters
    ==========
    aa: array_like
        Array to be complex conjugated
    bb: array_like
        Array not to be complex conjugated
    power_spectral_density: array_like
        Power spectral density of the noise
    duration: float
        duration of the data

    Returns
    =======
    Noise-weighted inner product.
    """

    integrand = np.conj(aa) * bb / power_spectral_density
    return 4 / duration * np.sum(integrand)

# compute summary data given a bin partition and fiducial waveforms
def compute_sdat(f, fbin, seglen, ndtct, psd_full, sFT_full, h0_full, i_wav):
    """
        Compute summary data
        Need to compute for each detector
        Parameters:
        f is the frequency grid on i_wav indices
        fbin is the bin edges
        fbin_ind gives the positions of bin edges in the full grid
        ndtct is the number of detectors
        psd is a list of PSDs
        sFT is a list of frequency-domain strain data
        h0  is a list of fiducial waveforms
        i_wav is an array of index based on f_max and f_min
        Note that sFT and h0 need to be provided with the full frequency resolution
        """

    summary_data = dict()
    Nbin = len(fbin) - 1
    # total duration of time-domain sequence
    duration = seglen

    # loop over detectors
    for k in range(ndtct):
        psd_masked = psd_full[k][i_wav]
        h0_masked = h0_full[k][i_wav]
        sFT_masked = sFT_full[k][i_wav]
        masked_bin_inds = []
        for edge in fbin:
            index = np.where(f == edge)[0][0]
            masked_bin_inds.append(index)
        a0, b0, a1, b1 = np.zeros((4,Nbin), dtype=complex)   

        # total number of frequency bins
        for i in range(Nbin):
                    start_idx = masked_bin_inds[i]
                    end_idx = masked_bin_inds[i + 1]
                    start = f[start_idx]
                    stop = f[end_idx]
                    idxs = slice(start_idx, end_idx)

                    strain = sFT_masked[idxs]
                    h0 = h0_masked[idxs]
                    psd = psd_masked[idxs]

                    frequencies = f[idxs]                    
                    central_frequency = (start + stop) / 2
                    delta_frequency = frequencies - central_frequency

                    a0[i] = noise_weighted_inner_product(h0, strain, psd, duration)
                    b0[i] = noise_weighted_inner_product(h0, h0, psd, duration)
                    a1[i] = noise_weighted_inner_product(h0, strain * delta_frequency, psd, duration)
                    b1[i] = noise_weighted_inner_product(h0, h0 * delta_frequency, psd, duration)

        summary_data[k] = (a0, a1, b0, b1)
    return summary_data    

# Gaussian Likelihood function -0.5 * (s-h|s-h) with Frequency binning
class GWBinningLikelihood(Likelihood):
    """
        Log-likelihood object,
        it assumes that the data are evaluated on the same frequency axis (given as input)
    """

    def __init__(self,
                 ifos, datas, dets, noises, fiducial_params,
                 freqs, srate, seglen, approx, eps, f_mrg,
                 nspcal=0, spcal_freqs=None,
                 nweights=0, len_weights=None,
                 marg_phi_ref=False, marg_time_shift=False,
                 **kwargs):

        # run standard initialization
        super(GWBinningLikelihood, self).__init__()

        # f_mrg is used to cut the frequency before and after the "merger" 
        # to use relative binning only in the inspiral part
        self.f_mrg = f_mrg

        if self.f_mrg != 0:
            self.i_rb       = np.where((freqs<=self.f_mrg))
            self.i_pm       = np.where((freqs>self.f_mrg))
            self.freqs_pm   = freqs[self.i_pm]
            freqs           = freqs[self.i_rb] 

        # set dictionaries of bajes objects
        self.noises = noises
        self.datas  = datas
        self.ifos   = ifos
        self.dets   = {ifo : erase_init_wrapper(dets[ifo]) for ifo in self.ifos}

        self.srate  = srate
        self.seglen = seglen
        self.freqs  = freqs

        # frequency binning needs f=0, but this value is value is unphysical.
        # some fiducial template might give an error,
        # then we add a little bit to it
        if freqs[0] == 0.:
            freqs[0] += freqs[1]/100.

        # generate h0
        i_wav       = np.where((freqs>=fiducial_params['f_min']))
        wave0       = Waveform(freqs[i_wav], self.srate , self.seglen, approx)
        h0p, h0c    = wave0.compute_hphc(fiducial_params)

        # fill below f_min
        l_low   = len(freqs) - len(np.concatenate(i_wav))
        h0p     = np.append(np.zeros(l_low), h0p)
        h0c     = np.append(np.zeros(l_low), h0c)
        h0      = PolarizationTuple(plus=h0p, cross=h0c)

        psds   = []
        sFTs   = []
        h0s    = []

        self.dd_nopsdweights = {}
        self.dd = 0.

        f_min_check = None
        f_max_check = None

        for i,ifo in enumerate(self.ifos):
            self.dets[ifo].store_measurement(datas[ifo], noises[ifo], nspcal=nspcal, spcal_freqs=spcal_freqs)

            # check if frequency ranges are consistent between data and model for every IFO
            if f_min_check == None:
                f_min_check = datas[ifo].f_min
            else:
                if datas[ifo].f_min != f_min_check:
                    logger.error("Input f_min of data and model do not match in detector {}.".format(ifo))
                    raise ValueError("Input parameter (f_min) of data and model do not match in detector {}.".format(ifo))
            
            if f_max_check == None:
                f_max_check = datas[ifo].f_max
            else:
                if datas[ifo].f_max != f_max_check:
                    logger.error("Input f_Nyq of data and model do not match in detector {}.".format(ifo))
                    raise ValueError("Input parameter (f_Nyq) of data and model do not match in detector {}.".format(ifo))

            if datas[ifo].seglen != self.seglen:
                logger.error("Input seglen of data and model do not match in detector {}.".format(ifo))
                raise ValueError("Input parameter (seglen) of data and model do not match in detector {}.".format(ifo))

            # compute PSDs and quantities for logL evaluation
            psds.append(noises[ifo].interp_psd_pad(freqs)*datas[ifo].window_factor)
            sFTs.append(datas[ifo].freq_series)
            h0s.append(self.dets[ifo].project_fdwave(h0, fiducial_params, wave0.domain, freqs=freqs))
            self.dd_nopsdweights[ifo] = datas[ifo].inner_product(datas[ifo],noises[ifo],[datas[ifo].f_min,datas[ifo].f_max])
            self.dd += np.real(self.dd_nopsdweights[ifo])

        f_min = f_min_check
        f_max = f_max_check

        maximum_nonzero_index = np.where(h0p != 0j)[0][-1]
        if f_max > freqs[maximum_nonzero_index]:
            f_max = freqs[maximum_nonzero_index]

        # store relative binning data
        self.Nbin, self.fbin, self.fbin_ind = setup_bins(freqs, f_min, f_max, eps)
        self.sdat   = compute_sdat(freqs[i_wav], self.fbin, self.seglen, len(self.ifos), psds, sFTs, h0s, i_wav)
        self.h0_bin = np.array([h0s[i][self.fbin_ind] for i,ifo in enumerate(self.ifos)])

        # initialize waveform generator
        self.wave   = erase_init_wrapper(Waveform(self.fbin, self.srate, self.seglen, approx))
        self.wave0  = erase_init_wrapper(Waveform(self.freqs[i_wav], self.srate, self.seglen, approx))
        if self.f_mrg !=0:
            self.wave_pm  = erase_init_wrapper(Waveform(self.freqs_pm, self.srate, self.seglen, approx))

        for ifo in self.ifos:
            self.dets[ifo].freqs    = self.fbin
            self.dets[ifo].srate    = self.srate
            self.dets[ifo].seglen   = self.seglen

        # set calibration envelopes
        self.nspcal     = nspcal
        if self.nspcal > 0.:
            self.spcal_freqs = spcal_freqs

        # set marginalization flags
        self.marg_phi_ref = marg_phi_ref

    def inner_products_singleifo(self, i, ifo, params, hphc):
        wav = self.dets[ifo].project_fdwave(hphc, params, self.wave.domain, freqs=self.fbin)

        if self.nspcal > 0:
            cal = compute_spcalenvs(ifo, self.nspcal, params)
            cal = np.interp(self.fbin, self.spcal_freqs, cal)
            wav = wav*cal

        rf      = self.compute_rf(wav, i)
        dh, hh  = self.prods_sdat(i, rf)
        return dh, hh
    
    # compute (d|h) and (h|h)
    def prods_sdat(self, k, rdata):
        """
            Compute products (h|h), (d|h) for the k-th detector using summary data,
            for logL evalutation with marginalized phi_ref
        """
        r0, r1 = rdata

        # compute logL components
        hh      = self.sdat[k][2]*np.absolute(r0)**2 + self.sdat[k][3]*2.0*(r0*np.conjugate(r1)).real
        dh      = self.sdat[k][0]*np.conjugate(r0) + self.sdat[k][1]*np.conjugate(r1)

        return np.sum(dh), np.sum(hh)

    # compute relative waveform r(f) = h(f)/h0(f)
    def compute_rf(self, h, i):

        """
            compute the ratio r(f) = h(f)/h0(f) where h0(f) is some fiducial waveform and h(f) correspond to parameter combinations par
            h : current waveform, already sampled at fbin and projected on the detector
            h0: fiducial waveform (it is important to pass one that is NOT shifted to the right merger time)
            fbin: frequency bin edges
            par is list of parameters: [Mc, eta, chieff, chia, Lam, dtc]
            tc: best-fit time
            """

        f       = self.fbin
        h0_bin  = self.h0_bin[i]

        # waveform ratio
        r   = h / h0_bin
        r0  = 0.5*(r[:-1] + r[1:])
        r1  = (r[1:] - r[:-1])/(f[1:] - f[:-1])

        return np.array([r0, r1], dtype=np.complex128)
    
    def log_like(self, params):
        """
            log-likelihood function,
            params : current sample as dictionary (filled from Prior)
        """

        #generate waveform
        hphc    = np.array(self.wave.compute_hphc(params, freqs=self.fbin))
        hphcPT = PolarizationTuple(plus=hphc[0], cross=hphc[1])

        # dh , hh
        inner_prods = np.transpose([list(self.inner_products_singleifo(i, ifo, params, hphcPT)) for i,ifo in enumerate(self.ifos)])
        dh = np.sum(inner_prods[0])
        hh = np.real(np.sum(inner_prods[1]))

        if self.marg_phi_ref:
            dh  = np.abs(dh)
            R   = dh + np.log(i0e(dh))
        else:
            dh  = np.real(dh)
            R   = dh

        lnl = R - 0.5*hh

        if self.f_mrg != 0:
            lnl += self.log_like_pm(params)

        return np.real(lnl)
    
    def log_like_pm(self, params):
        """
            log-likelihood function
        """
    
        freqs = self.freqs_pm
        wave    = self.wave_pm.compute_hphc(params,freqs=freqs)   

        for ifo in self.ifos:
            self.dets[ifo].psd = self.noises[ifo].interp_psd_pad(freqs)
            self.dets[ifo].data = self.datas[ifo].freq_series[self.i_pm]
            dh_arr_thisifo, hh_thisifo, dd_thisifo, _psdf = self.dets[ifo].compute_inner_products(wave, params, self.wave.domain, psd_weight_factor=True, freqs=freqs)
            dh = (dh_arr_thisifo).sum()
            hh = np.real(hh_thisifo)
            dd = np.real(dd_thisifo)
            _psd_fact = _psdf

        if self.marg_phi_ref:
            abs_dh = np.abs(dh)
            R      = np.log(i0e(abs_dh)) + abs_dh
        else:
            R      = np.real(dh)

        logL  =  R - 0.5*hh

        return logL
