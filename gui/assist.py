# This file is part of MyPaint.
# Copyright (C) 2008-2013 by Martin Renold <martinxyz@gmx.ch>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""Freehand drawing modes"""

## Imports

import math
from numpy import array
from numpy import isfinite
from lib.helpers import clamp
import logging
from collections import deque
logger = logging.getLogger(__name__)

import gtk2compat
from gettext import gettext as _
import gobject
import gtk
from gtk import gdk
from gtk import keysyms

## Module settings

# Which workarounds to allow for motion event compression
EVCOMPRESSION_WORKAROUND_ALLOW_DISABLE_VIA_API = True
EVCOMPRESSION_WORKAROUND_ALLOW_EVHACK_FILTER = True

# Consts for the style of workaround in use
EVCOMPRESSION_WORKAROUND_DISABLE_VIA_API = 1
EVCOMPRESSION_WORKAROUND_EVHACK_FILTER = 2
EVCOMPRESSION_WORKAROUND_NONE = 999


## Class defs

class Assistbase(object):

    # Stablizer ring buffer
    _sampling_max = 32
    _samples = [None] * _sampling_max 
    _sample_index = 0
    _sample_count = 0

    def __init__(self):
        self.reset()

    def reset(self):
        Assistbase._sample_index = 0
        Assistbase._sample_count = 0

    def get_current(self, x, y):
        """only stub"""
        return (x, y)

    def fetch(self, x, y):
        """Fetch samples"""
        Assistbase._samples[Assistbase._sample_index] = (x, y)
        Assistbase._sample_index+=1
        Assistbase._sample_index%=Assistbase._sampling_max
        Assistbase._sample_count+=1

class Stabilizer(Assistbase):
    """ Stabilizer class, which fetches 
    gtk.event x/y position as a sample,and return 
    the avearage of recent samples.
    """

    def __init__(self):
        super(Stabilizer, self).__init__()

    def get_current(self):
        if self._sample_count < self._sampling_max:
            return None

        ox = 0
        oy = 0
        idx = 0
        while idx < self._sampling_max:
            cx, cy = self._get_stabilize_element(idx)
            ox += cx
            oy += cy
            idx += 1

        return (ox / self._sampling_max, oy / self._sampling_max)

    def _get_stabilize_element(self, idx):
        return self._samples[(self._sample_index + idx) % self._sampling_max]
    
        

