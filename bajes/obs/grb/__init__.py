#!/usr/bin/env python
from __future__ import unicode_literals, absolute_import
__import__("pkg_resources").declare_namespace(__name__)

from .lightcurve import GRB_Lightcurve, __approx_dict__
from ... import CLIGHT_SI

__known_approxs__   = list(__approx_dict__.keys())

__photometric_bands__       = { 'Xray'      : 1.2e18,
                                'Xray2'     : 1.0e18,
                               
                                # optical bands
                                'r'         : 4.82e14,
                                'u'         : 8.45e14,
                                'i'         : 3.93e14,
                                'g'         : 6.32e14,
                                'z'         : 3.31e14,
                                'l'         : 8.57e13,
                                # 'B'         : 6.734e14,
                                # 'K'         : 1.37e14,

                                # radio bands
                                'R'         : 6.0e9,
                                'L'         : 1.39e9,
                                'S'         : 3.5e9,
                                'Ku'        : 17e9,
                                'C'         : 7e9,
                                'Ka'        : 30e9,
                                'ALMA'      : 105.5e9,
                                'X'         : 11e9,
                                'O'         : 5.1e14,
                            }