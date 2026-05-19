from __future__ import division, unicode_literals
import numpy as np

import logging
logger = logging.getLogger(__name__)

# # Dictionary of known approximants
# # Each key corresponds to the name of the approximant
# # Each value has to be a dictionary
# # that include the following keys:
# #   * 'path':   string to method to be imported, e.g. bajes.obs.gw.approx.taylorf2.taylorf2_35pn_wrapper
# #   * 'type':   define if the passed func is a function or a class, options: ['fnc', 'cls']
# #   * 'domain': define if the method returns a frequency- or time-domain waveform, options: ['time', 'freq']

__approx_dict__ = { ### TIME-DOMAIN
                    # funcs
                    'grb_afterglow':                {'path': 'bajes.obs.grb.approx.afterglowpy_model.afterglow_wrapper',
                                                             'type': 'fnc'},
                  }

def __get_lightcurve_generator__(approx, times, nu, **kwargs):

    # get approximant list
    __known_approxs__ = list(__approx_dict__.keys())

    # unknown approx
    if (approx not in __known_approxs__):
        logger.error("Unable to read approximant string. Please use a valid string: {}.".format(__known_approxs__))
        raise AttributeError("Unable to read approximant string. Please use a valid string: {}.".format(__known_approxs__))

    this_light = __approx_dict__[approx]

    # set module string and import
    from importlib import import_module
    path_to_method  = this_light['path'].split('.')
    light_module    = import_module('.'.join(path_to_method[:-1]))

    # this condition never occurs if the code is properly written
    if path_to_method[-1] not in dir(light_module):
        raise AttributeError("Unable to import {} method from {}".format(path_to_method[-1], light_module))

    # get waveform generator and domain string
    if this_light['type'] == 'fnc':
        light_func = getattr(light_module, path_to_method[-1])
    else:
        # this condition never occurs if the __approx_dict__ is properly written
        raise AttributeError("Unable to define method for grb lightcurve generator. Check bajes.obs.grb.lightcurve.__approx_dict__")

    return light_func


class GRB_Lightcurve(object):
    """
        GRB Lightcurve object
    """

    def __init__(self, times, nu, approx, **kwargs):
        """
            Initialize the Lightcurve with a frequency axis and the name of the approximant
        """
        
        self.times      = times
        self.nu         = nu
        self.approx     = approx
        logger.info("Setting {} lightcurve ...".format(self.approx))

        # get waveform generator from string
        self.light_func = __get_lightcurve_generator__(self.approx, self.times, self.nu, **kwargs)

    def compute_mag(self, params): 
        # include band information in params
        dict_mag = {}
        for n in self.nu.keys():
            flux        = self.light_func(self.times, self.nu[n], params)
            mag         = -2.5 * np.log10(flux/1e26) - 48.6
            dict_mag[n] = mag
        return dict_mag
