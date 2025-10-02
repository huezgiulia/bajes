from __future__ import division, unicode_literals, absolute_import
import numpy as np

import logging
logger = logging.getLogger(__name__)

import pocomc as pc

from . import SamplerBody
from .proposal import _init_proposal_methods


def initialize_proposals(like, priors, use_slice=False, use_gw=False):

    prop_kwargs  = {}

    prop_kwargs['like'] = like
    prop_kwargs['dets'] = like.dets

    return BajesPocoMCProposal(priors, **prop_kwargs)

class PriorPocoMC():

    def __init__(
        self,
        priors,
    ):
        self.priors = priors
        self.sampling_parameters = []
        for p in priors.parameters:
            self.sampling_parameters.append(p.name)

    def logpdf(self, x):
        N, D = x.shape
        logp = np.zeros(N)
        for i in range(N):
            logp[i] = self.priors.log_prior(x[i])
        return logp
    
    def rvs(self, size=1):
        samples = np.zeros((size, self.dim))
        for i in range(size):
            samples[i] = self.priors.sample()
        return samples

    @property
    def bounds(self):
        return np.array(self.priors.bounds)

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

        # # initialize proposals
        # if proposals == None:
        #     logger.info("Initializing proposal methods ...")
        #     proposals = initialize_proposals(posterior.like, prior)
               
        # initialize sampler
        logger.info("Initializing sampler ...")
        self.sampler = pc.Sampler(prior=prior,
                                    likelihood=posterior.log_like,
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

            it = 5 # use (number of iterations // ncheckpoints) * ncheckpoints
            path = 'states/pmc_' + str(it) + '.state'
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
        
