#!/usr/bin/env python
from __future__ import unicode_literals, absolute_import

from .lightcurve import __approx_dict__

__known_approxs__   = list(__approx_dict__.keys())

__photometric_bands__       = { # X-ray bands
                                'X-ray-1keV' : 2.42e17,
                                'X-ray-10keV': 2.4e18,
                               
                                # optical bands   
                                'F444W'     : 0.69e14,
                                'F277W'     : 1.10e14,
                                'k'         : 1.4e14,
                                'h'         : 1.8e14,
                                'j'         : 2.43e14,
                                'z'         : 3.36e14,
                                'Ic'        : 3.73e14,
                                'i'         : 3.91e14,
                                'Rc'        : 4.68e14,
                                'r'         : 4.83e14,
                                'v'         : 5.48e14,
                                'g'         : 6.38e14,
                                'b'         : 6.86e14,
                                'u'         : 8.54e14,
                                'uvw1'      : 11.2e14,
                                'uvm'       : 13.4e14,

                                # radio bands
                                'L'         : 1.4e9,
                                'S'         : 3.0e9,
                                'C'         : 6.0e9,
                                'X'         : 10.0e9,
                                'K'         : 12.0e9,
                                'Ku'        : 15.0e9,
                                'Ka'        : 30.0e9,
                                'ALMA'      : 105.5e9,
                            }
