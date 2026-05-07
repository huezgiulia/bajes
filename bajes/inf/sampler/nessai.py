from __future__ import division, unicode_literals, absolute_import
import numpy as np

import logging
logger = logging.getLogger(__name__)

import nessai
from nessai.model import Model
from nessai.flowsampler import FlowSampler

from . import SamplerBody

import os

class NessaiModel(Model):

    def __init__(self, posteriors):
        
        self.posterior = posteriors
        self.priors = posteriors.prior
        self.names = [p.name for p in self.priors.parameters]
        self.bounds = {p.name: p.bound for p in self.priors.parameters}
        
    def log_prior(self, x):
        x_vec = [x[name] for name in self.names]
        return self.priors.log_prior(x_vec)

    def log_likelihood(self, x):
        x_vec = [x[name] for name in self.names]
        return self.posterior.log_like(x_vec)

class SamplerNessai(SamplerBody):

    def __initialize__(self, posterior, pool=None,
                       **kwargs):
        self.sampler = FlowSampler(NessaiModel(posterior), output=self.outdir, nlive=kwargs['nlive'], stopping=kwargs['tolerance'])
    
    def __run__(self):
        self.sampler.run()