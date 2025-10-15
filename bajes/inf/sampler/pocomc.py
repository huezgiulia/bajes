from __future__ import division, unicode_literals, absolute_import
import numpy as np

import logging
logger = logging.getLogger(__name__)

import pocomc as pc

from . import SamplerBody

import os


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
            samples[i] = self.priors.sample
        return samples

    @property
    def bounds(self):
        return np.array(self.priors.bounds)

    @property
    def dim(self):
        return len(self.sampling_parameters)


class SamplerPocoMC(SamplerBody):

    def __initialize__(self, posterior, nlive, pool=None,
                       proposals=None, 
                       n_steps = 50, n_max_steps = 500,
                       # BNS - slow https://arxiv.org/pdf/2506.18977 
                       n_tot = 9192, n_active=1024, n_effective=2048,
                       flow='nsf6', precondition=True,
                       **kwargs):

        n_steps = self.ndim
        n_max_steps = 10*self.ndim
        self.nsave = kwargs['nsave']

        # periodic parameters
        index_periodic = []
        for p,i in zip(posterior.prior.parameters, range(len(posterior.prior.parameters))):
            if p.periodic: 
                index_periodic.append(i)

        prior = PriorPocoMC(posterior.prior)                      
        # initialize sampler
        logger.info("Initializing sampler ...")
        self.sampler = pc.Sampler(prior=prior,
                                    likelihood=posterior.log_like,
                                    random_state=0,
                                    # n_effective=nlive,
                                    # n_steps=n_steps,
                                    # n_max_steps=n_max_steps,
                                    n_steps = 50, n_max_steps = 500,
                                    n_active=1024, n_effective=2048,
                                    pool=pool,periodic=index_periodic,
                                    )

        # extract prior samples for initial state
        logger.info("Extracting prior samples ...")

        self.stop   = False

    def __restore__(self, pool, **kwargs):

        # re-initialize pool
        self.sampler.pool   = pool

    def __run__(self):
        while not self.stop:
            
            # find resume state
            if not os.path.exists('states'):
                path = None
            else:
                files = os.listdir('states')
                if 'pmc_final.state' in files:
                        path = 'states/pmc_final.state'
                        self.stop = True
                else:
                    # look for the file with the highest iteration number
                    iterations = [int(f.split('_')[1].split('.')[0]) for f in files if f.startswith('pmc_')]
                    if len(iterations) > 0:
                        max_iter = max(iterations)
                        path = 'states/pmc_' + str(max_iter) + '.state'

            self.sampler.run(save_every = self.nsave, resume_state_path=path, n_total = 9192,)

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
            import corner
        except Exception:
            logger.warning("Impossible to produce standard plots. Cannot import corner.")

        try:
            samples, weights, logl, logP = self.sampler.posterior()

            fig = corner.corner(samples[:,:self.ndim], weights=weights, show_titles=True, labels=self.names)
            fig.savefig(self.outdir+'/corner.png', dpi=200)

        except Exception:
            pass
        
