#!/usr/bin/env python
from __future__ import unicode_literals, absolute_import
__import__("pkg_resources").declare_namespace(__name__)

from .lightcurve import GRB_Lightcurve, __approx_dict__
from ... import CLIGHT_SI

__known_approxs__   = list(__approx_dict__.keys())

__photometric_bands__       = { 'Xray'      : 1.2e18,
                                'Xray2'     : 1.0e18,
                                'Xray3'     : 2.4e18,
                               
                                # optical bands
                                'r'         : 4.816e14,
                                'u'         : 8.652e14,
                                'i'         : 3.958e14,
                                'g'         : 6.379e14,
                                'H'         : 1.834e14,
                                'J'         : 2.398e14,
                                'Y'         : 2.939e14,
                                'Z'         : 3.417e14,
                                'F606W'     : 5.040e14,
                                'R'         : 4.680e14,
                                # 'z'         : 3.31e14,
                                # 'l'         : 8.57e13,
                                # 'B'         : 6.734e14,
                                # 'K'         : 1.37e14,

                                # radio bands
                                'L1.45'     : 1.45e9,
                                'L1.77'     : 1.77e9,
                                'S2.7'      : 2.68e9,
                                'S3.5'      : 3.2e9,
                                'C4.8'      : 4.8e9,
                                'C5'        : 5.0e9,
                                'C5.5'      : 5.5e9,
                                'C6.1'      : 6.1e9,
                                'C6.5'      : 6.5e9,
                                'C7.1'      : 7.1e9,
                                'C7.4'      : 7.4e9,
                                'C7.5'      : 7.5e9,
                                'X8.5'      : 8.5e9,
                                'X9'        : 9.0e9,
                                'X11'       : 11.0e9,
                                'Ku13.4'    : 13.5e9,
                                'Ku15.9'    : 16.0e9,
                                'K19'       : 19.0e9,
                                'K19.2'     : 19.2e9,
                                'K22'       : 22.0e9,
                                'K24.5'     : 24.5e9,
                                'Q39'       : 39.0e9,

                                # 'R'         : 6.0e9,
                                'L'         : 1.39e9,
                                'S'         : 3.5e9,
                                'Ku'        : 17e9,
                                'C'         : 7e9,
                                'Ka'        : 30e9,
                                'ALMA'      : 105.5e9,
                                'X'         : 11e9,
                                'O'         : 5.1e14,
                                'radio-3GHz': 3e9,
                                'radio-6GHz': 6e9,
                                'X-ray-1keV': 2.42e17,
                                'bessellv'  : 5.08e14,
                            }
