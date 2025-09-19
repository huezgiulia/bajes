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


class SamplerMCMC(SamplerBody):

    def __initialize__(self, posterior,nwalk,
                       proposals=None,
                       pool=None, **kwargs):

        # initialize proposals
        if proposals == None:
            logger.info("Initializing proposal methods ...")
            proposals = initialize_proposals(posterior.like, posterior.prior)

        # initialize sampler
        logger.info("Initializing sampler ...")
        self.sampler = pc.sampler(prior=posterior.prior,
                                    likelihood=posterior.like,
                                    vectorize=True,
                                    random_state=0)

        # extract prior samples for initial state
        logger.info("Extracting prior samples ...")
        self._previous_state = posterior.prior.get_prior_samples(nwalk)

        self.stop   = False

    def __restore__(self, pool, **kwargs):

        # re-initialize pool
        if pool == None:
            self.sampler.pool   = pool
        else:
            self.sampler.pool   = pool

    def __update__(self):

        (worst, ustar, vstar, loglstar, logvol, logwt, logz, logzvar, h, nc, worst_it, boundidx, bounditer, eff, delta_logz, blob) = self._last_iter

        args = {'it' :      self.sampler.it,
                'eff' :     '{:.2f}%'.format(eff),
                'nc' :      nc,
                #'logL' :    '{:.3f}'.format(loglstar),
                'logLmax' : '{:.3f}'.format(np.max(self.sampler.live_logl)),
                'logZ' :    '{:.3f}'.format(logz),
                'H' :       '{:.2f}'.format(h),
                'dZ' :      '{:.3f}'.format(delta_logz)}

        return args
    
    # def _stop_sampler(self):


    def __run__(self):

        while not self.stop:

            # make steps
            for results in self.sampler.sample(self._previous_state, iterations=self.ncheckpoint, tune=True):
                pass

            # update previous state
            self._previous_state  = results

            # update sampler status
            self.update()

            # compute stopping condition
            self._stop_sampler()

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

            plt.savefig(self.outdir+'/posterior.png', dpi=200)

            plt.close()

        except Exception:
            pass
