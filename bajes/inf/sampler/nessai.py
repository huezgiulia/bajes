from __future__ import division, unicode_literals, absolute_import
import numpy as np
import h5py

import logging
logger = logging.getLogger(__name__)

import nessai
from nessai.model import Model
from nessai.flowsampler import FlowSampler
from nessai.utils.logging import configure_logger

from . import SamplerBody

import os


class NessaiModel(Model):

    def __init__(self, posteriors):
        
        self.posterior = posteriors
        self.priors = posteriors.prior
        self.names = [p.name for p in self.priors.parameters]
        self.bounds = {p.name: p.bound for p in self.priors.parameters}

    def scalar(self, x):
        return float(np.asarray(x).squeeze())        
    
    def log_prior(self, x):
        # x_vec = [x[name] for name in self.names]
        if np.ndim(x[self.names[0]]) > 0 and len(np.asarray(x[self.names[0]])) > 1:
            return np.array([
                self.priors.log_prior([
                    x[name][i] for name in self.names
                ])
                for i in range(len(x[self.names[0]]))
            ])

        # Scalar case
        else:

            x_vec = [float(np.asarray(x[name])) for name in self.names]

            return self.priors.log_prior(x_vec)
        # x_vec = [self.scalar(x[name]) for name in self.names]
        # return self.priors.log_prior(x_vec)

    def log_likelihood(self, x):
        # x_vec = [x[name] for name in self.names]
        x_vec = [self.scalar(x[name]) for name in self.names]
        return self.posterior.log_like(x_vec)

class SamplerNessai(SamplerBody):

    def __initialize__(self, posterior, pool=None,
                       **kwargs):        
        configure_logger(output=self.outdir,label='bajes')
        self.sampler = FlowSampler(NessaiModel(posterior), output=self.outdir, nlive=kwargs['nlive'], 
                                   stopping=kwargs['tolerance'], pool=pool,)
    
    def __run__(self):
        self.sampler.run()       

        logZ = self.sampler.ns.log_evidence
        logZerr = self.sampler.ns.log_evidence_error
        with open(f"{self.outdir}/evidence.dat", "w") as f:
            f.write('#\tlogX\tlogZ\tlogZerr\n')
            f.write(f'0\t{logZ:.6f}\t{logZerr:.6f}\n')

    def get_posterior(self):
        with h5py.File(self.outdir + '/result.hdf5', "r") as f:
            samples = f["posterior_samples"]
            names = list(samples.dtype.names)
            data = np.column_stack([samples[name] for name in names])
        header = " ".join(names)
        np.savetxt(self.outdir + '/posterior.dat', data, header=header)

