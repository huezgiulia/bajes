#!/usr/bin/env python
from __future__ import unicode_literals, absolute_import

from .lightcurve import AM_Lightcurve, __approx_dict__
from ... import CLIGHT_SI

__known_approxs__   = list(__approx_dict__.keys())

__photometric_bands__       = { # radio bands
                                'VLBI_4.5GHz':   4.5e9,
                                'VLBI_5GHz':     5.0e9,
                            }
