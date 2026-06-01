from __future__ import division, unicode_literals, absolute_import
import numpy as np
import h5py

import logging
logger = logging.getLogger(__name__)

import nessai
from nessai.model import Model
from nessai.flowsampler import FlowSampler
# from nessai.gw.proposal import GWFlowProposal
from nessai.proposal import FlowProposal
from nessai.gw.reparameterisations import get_gw_reparameterisation
from nessai.utils.logging import configure_logger

from . import SamplerBody

import os

class GWFlowProposalBajes(FlowProposal):
    """Wrapper for FlowProposal that has defaults for CBC-PE
        From nessai code, adapted for bajes MMA"""

    aliases = {
        "chirp_mass": ("mass", None),
        "mass_ratio": ("mass_ratio", None),
        # "ra": ("sky-ra-dec", ["dec", "Dec"]),
        # "dec": ("sky-ra-dec", ["ra"]),
        "azimuth": ("sky-az-zen", ["zenith", "zen", "Zen", "Zenith"]),
        "zenith": ("sky-az-zen", ["azimuth", "az", "Az", "Azimuth"]),
        "theta_1": ("angle-sine", None),
        "theta_2": ("angle-sine", None),
        "tilt_1": ("angle-sine", None),
        "tilt_2": ("angle-sine", None),
        "theta_jn": ("angle-sine", None),
        "iota": ("angle-sine", None),
        "phi_jl": ("angle-2pi", None),
        "phi_12": ("angle-2pi", None),
        "phase": ("angle-2pi", None),
        "psi": ("angle-pi", None),
        "geocent_time": ("time", None),
        "time_jitter": ("periodic", None),
        "a_1": ("default", None),
        "a_2": ("default", None),
        "chi_1": ("default", None),
        "chi_2": ("default", None),
        "luminosity_distance": ("distance", None),
    }
    """
    Dictionary of aliases used to determine the default reparameterisations
    for common gravitational-wave parameters.
    """
    use_default_reparameterisations = True
    """
    GW specific reparameterisations will be used by default. This is different
    to the parent class where they are disabled by default.
    """

    def get_reparameterisation(self, reparameterisation):
        """Function to get reparameterisations that checks GW defaults and
        aliases
        """
        return get_gw_reparameterisation(reparameterisation)

    def add_default_reparameterisations(self):
        """
        Add default reparameterisations for parameters that have not been
        specified.
        """
        parameters = [
            n
            for n in self.model.names
            if n not in self._reparameterisation.parameters
        ]
        logger.info(f"Adding default reparameterisations for {parameters}")

        for p in parameters:
            logger.debug(f"Trying to add reparameterisation for {p}")
            if p in self._reparameterisation.parameters:
                logger.debug(f"Parameter {p} is already included")
                continue
            name, extra_params = self.aliases.get(p.lower(), (None, None))
            if name is None:
                logger.debug(f"{p} is not a known GW parameter")
                continue
            if extra_params is not None:
                p = [p] + [ep for ep in extra_params if ep in self.model.names]
            else:
                p = [p]
            prior_bounds = {k: self.model.bounds[k] for k in p}
            reparam, kwargs = get_gw_reparameterisation(name)
            logger.info(
                f"Adding reparameterisation {reparam.__name__} for {p} "
                f"with config: {kwargs}"
            )
            self._reparameterisation.add_reparameterisation(
                reparam(parameters=p, prior_bounds=prior_bounds, **kwargs)
            )



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
                                   stopping=kwargs['tolerance'], n_pool=pool._processes,
                                   flow_class=GWFlowProposalBajes, analytic_priors=True,
                                   flow_config=dict(n_blocks=6,n_neurons=40),
                                   )

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

