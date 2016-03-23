# This file is part of MyPaint.
# Copyright (C) 2016 dothiko<dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or


## Imports

import math
import array
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

import gui.drawutils

## Module settings



## Class defs

class Assistbase(object):

    # Stablizer ring buffer
    _sampling_max = 32
    _samples_x = array.array('f')
    _samples_y = array.array('f')
    _samples_p = array.array('f')
    _sample_index = 0
    _sample_count = 0

    def __init__(self):
        if len(Assistbase._samples_x) < Assistbase._sampling_max:
            for i in range(Assistbase._sampling_max):
                Assistbase._samples_x.append(0.0) 
                Assistbase._samples_y.append(0.0) 
                Assistbase._samples_p.append(0.0) 
        self.reset()

    def reset(self):
        Assistbase._sample_index = 0
        Assistbase._sample_count = 0

    def get_current(self, button, time):
        """only stub"""
        pass

    def fetch(self, x, y, pressure):
        """Fetch samples"""
        Assistbase._samples_x[Assistbase._sample_index] = x 
        Assistbase._samples_y[Assistbase._sample_index] = y 
        Assistbase._samples_p[Assistbase._sample_index] = pressure 
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

    def get_current(self, button, time):
        if self._sample_count < self._sampling_max:
            return None


        rx = 0
        ry = 0
        rp = 0
        idx = 0
        while idx < self._sampling_max:
            rx += self._get_stabilized_x(idx) 
            ry += self._get_stabilized_y(idx) 
            rp += self._get_stabilized_pressure(idx)
            idx += 1


        rx /= self._sampling_max 
        ry /= self._sampling_max
        rp /= self._sampling_max

        if button == None:
            if (self._prev_button != None):
                if (self._prev_rx and 
                    math.hypot(rx - self._prev_rx, ry - self._prev_ry) > 2): 
                    # From my experience, 2 is good value to make trail.
                    self._prev_rx = rx
                    self._prev_ry = ry
                    return (rx, ry, rp)
            rp = 0.0 

        self._prev_button = button
        self._prev_rx = rx
        self._prev_ry = ry


        return (rx, ry, rp)

    def reset(self):
        super(Stabilizer, self).reset()
        self._prev_rx = None
        self._prev_ry = None
        self._prev_button = None

    def _get_stabilized_x(self, idx):
        return self._samples_x[(self._sample_index + idx) % self._sampling_max]
    
    def _get_stabilized_y(self, idx):
        return self._samples_y[(self._sample_index + idx) % self._sampling_max]

    def _get_stabilized_pressure(self, idx):
        return self._samples_p[(self._sample_index + idx) % self._sampling_max]
        

