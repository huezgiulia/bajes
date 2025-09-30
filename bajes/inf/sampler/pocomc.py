from __future__ import division, unicode_literals, absolute_import
import numpy as np

import logging
logger = logging.getLogger(__name__)

from itertools import repeat

import pocomc as pc

from . import SamplerBody
from .proposal import _init_proposal_methods

def initialize_proposals(like, priors):

    prop_kwargs  = {}

    return BajesPocoMCProposal(priors, **prop_kwargs)

class PriorPocoMC():
    """Wrapper for prior class to make it compatible with pocomc.

    Parameters
    ----------
    bilby_priors: bilby.core.prior.PriorDict
        Bilby prior dictionary.
    """

    logpdf = None
    """Log-prior probability density function.
    """

    rvs = None
    """Function for drawing random samples from the prior.
    """

    def __init__(
        self,
        priors,
    ):
        self.priors = priors
        self.sampling_parameters = []
        for p in priors.parameters:
            self.sampling_parameters.append(p.name)

        self.logpdf = self._logpdf_with_constraints
        self.rvs = self._rvs_with_constraints


    def to_dict(self, x):
        return {k: x[..., i] for i, k in enumerate(self.sampling_parameters)}

    def from_dict(self, x, keys=None):
        if keys is None:
            keys = self.sampling_parameters
        return np.array([x[v] for v in keys]).T

    def _logpdf_with_constraints(self, x):
        x_dict = self.to_dict(x)
        # The priors already include the constraints
        return self.bilby_priors.ln_prob(x_dict, axis=0)

    def _rvs_with_constraints(self, size=1):
        return self.from_dict(
            self.bilby_priors.sample_subset_constrained(
                keys=list(self.bilby_priors.keys()), size=size
            ),
            self.sampling_parameters,
        )

    @property
    def bounds(self):
        return self.priors.bounds

    @property
    def dim(self):
        return len(self.sampling_parameters)


class BajesPocoMCProposal(object):

    def __init__(self, priors, props=None,
                    nsplits=2, randomize_split=True,
                    **kwargs):

        self.names                  = priors.names
        self.bounds                 = priors.bounds
        self.ndim                   = len(self.names)
        self.period_reflect_list    = priors.periodics

        self._proposals, self._weights = _init_proposal_methods(priors, props=props, **kwargs)

        # run proposal init
        super(BajesPocoMCProposal, self).__init__(nsplits=nsplits, randomize_split=randomize_split)

    def get_proposal(self, s, c, p, model):
        _p = model.random.choice(self._proposals, p=self._weights)
        return _p.get_proposal(s, c, p, model)


class SamplerPocoMC(SamplerBody):

    def __initialize__(self, posterior, nlive, 
                       proposals=None, 
                       pool=None, n_active=512, 
                       flow='nsf6', precondition=True,
                       **kwargs):
        
        n_steps = self.ndim
        n_max_steps = 10*self.ndim
        self.ncheckpoints = kwargs.get('ncheckpoints')

        prior = PriorPocoMC(posterior.prior)

        from scipy.stats import uniform, norm
        prior = pc.Prior(10*[norm(0.0, 3.0)])

        # # initialize proposals
        # if proposals == None:
        #     logger.info("Initializing proposal methods ...")
        #     proposals = initialize_proposals(posterior.like, prior)
        
        # log_like = posterior.like.log_like(params)

        def log_like(x):
            return -np.sum(10.0 * (x[:, ::2] ** 2.0 - x[:, 1::2]) ** 2.0 + (x[:, ::2] - 1.0) ** 2.0, axis=1)
        
        # initialize sampler
        logger.info("Initializing sampler ...")
        self.sampler = pc.Sampler(prior=prior,
                                    likelihood=log_like,
                                    vectorize=True,
                                    random_state=0,
                                    n_effective=nlive,
                                    n_steps=n_steps,
                                    n_max_steps=n_max_steps,
                                    pool=pool,
                                    )

        # extract prior samples for initial state
        logger.info("Extracting prior samples ...")

        self.stop   = False

    def __restore__(self, pool, **kwargs):

        # re-initialize pool
        self.sampler.pool   = pool


    def __run__(self):

        while not self.stop:

            it = 5
            path = 'states/pmc_' + str(it) + '.state'
            print(self.ncheckpoints)
            self.sampler.run(save_every = self.ncheckpoints) #,resume_state_path=path) 

            # update sampler status
            # self.update()

            # compute stopping condition
            # self._stop_sampler()

        # final store inference
        self.store()

    def get_posterior(self):

        samples, weights, logl, logP = self.sampler.posterior()
        self.posterior_samples  = samples

        self.real_nout = self.posterior_samples.shape[0]
        logger.info("  - number of posterior samples : {}".format(self.real_nout))

        post_file = open(self.outdir + '/posterior.dat', 'w')

        post_file.write('#')
        for n in range(self.ndim):
            post_file.write('{}\t'.format(self.names[n]))
        post_file.write('logP\n')

        for i in range(self.real_nout):
            for j in range(self.ndim):
                post_file.write('{}\t'.format(self.posterior_samples[i][j]))
            post_file.write('{}\n'.format(logP[i]))

        post_file.close()

    def make_plots(self):

        try:
            import matplotlib.pyplot as plt
            import corner
        except Exception:
            logger.warning("Impossible to produce standard plots. Cannot import matplotlib.")

        try:
            samples, weights, logl, logP = self.sampler.posterior()

            fig = plt.figure()
            corner.corner(samples[:,:4], weights=weights, color="C0")

            fig.savefig(self.outdir+'/posterior.png', dpi=200)

            plt.close()

        except Exception:
            pass
        
