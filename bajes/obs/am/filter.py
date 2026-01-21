from __future__ import division, unicode_literals
import numpy as np

class Filter(object):

    def __init__(self, folder, nu):

        self.nu         = nu
        self.fluxes     = {}
        self.flux_stdev = {}
        self.times      = {}
        self.ras        = {}
        self.decs       = {}
        self.ra_stdev   = {}
        self.dec_stdev  = {}
    
        all_times  = []

        for k in list(self.nu.keys()):

            # read data from given folder
            # obs. the data file names have to be identical to the nu
            # obs. the data files should contain three columns, time, magnitudes and standard deviations
            try:
                t,m,sm,ra,sra,dec,sdec = np.genfromtxt(folder + '/{}.txt'.format(k), usecols=[0,1,2,3,4,5,6], unpack=True)
            except Exception as exc:
                raise RuntimeError("Error occured while loading {}".format(folder + '/{}.txt.'.format(k)))

            try:
                assert len(m) == len(sm)
                assert len(m) == len(t)
                assert len(ra) == len(sra)
                assert len(dec) == len(sdec)
                assert len(ra) == len(t)
            except Exception as exc:
                raise RuntimeError("Unconsistent data length detected in magnitude file {}".format(folder + '/{}.txt.'.format(k)))

            self.fluxes[k]      = m
            self.flux_stdev[k]  = sm
            self.times[k]       = t
            self.ras[k]         = ra
            self.decs[k]        = dec
            self.ra_stdev[k]    = sra
            self.dec_stdev[k]   = sdec

            all_times      = np.concatenate([all_times, t])

        self.all_times = np.sort(list(set(all_times)))

    @property
    def bands(self):
        return list(self.nu.keys())

    @property
    def wavelengths(self):
        return [self.nu[bi] for bi in self.bands]
