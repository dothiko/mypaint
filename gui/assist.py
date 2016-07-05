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

from gi.repository import Gtk, Gdk

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
    _current_index = 0

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

    def enum_current(self, button, time):
        """
        Only stub.
        This method should return a value with yield.
        """
        pass

    def fetch(self, x, y, pressure, time):
        """Fetch samples"""
        Assistbase._samples_x[Assistbase._sample_index] = x 
        Assistbase._samples_y[Assistbase._sample_index] = y 
        Assistbase._samples_p[Assistbase._sample_index] = pressure 
        Assistbase._current_index = Assistbase._sample_index
        Assistbase._sample_index+=1
        Assistbase._sample_index%=Assistbase._sampling_max
        Assistbase._sample_count+=1

    def get_current_index(self, offset):
        return (self._current_index + offset) % self._sampling_max

    def set_active_cb(self, flag):
        """ activated from Gtk.Action """
        pass
            

class Stabilizer(Assistbase):
    """ Stabilizer class, which fetches 
    gtk.event x/y position as a sample,and return 
    the avearage of recent samples.
    """
    STABILIZE_START_MAX = 24

    def __init__(self, app):
        super(Stabilizer, self).__init__()
        self._stabilize_cnt = None
        self.app = app

    def enum_current(self, button, time):
        if self._sample_count < self._sampling_max:
            raise StopIteration

        rx = 0.0
        ry = 0.0
        rp = self._latest_pressure
        idx = 0

        while idx < self._sampling_max:
            rx += self._get_stabilized_x(idx)
            ry += self._get_stabilized_y(idx)
            idx += 1

        rx /= self._sampling_max 
        ry /= self._sampling_max

        _prev_button = self._prev_button
        self._prev_button = button
        self._prev_rx = rx
        self._prev_ry = ry
        self._prev_rp = rp

        # Heading / Trailing glitch workaround
        if button == 1:
            if (_prev_button == None):
                self._stabilize_cnt = 0
                yield (rx, ry, 0.0)
                yield (rx, ry, self._get_initial_pressure(rp))
                raise StopIteration
            elif self._stabilize_cnt < self.STABILIZE_START_MAX:
                rp = self._get_initial_pressure(rp)
        elif button == None:
            if (_prev_button != None):
                rp = self._get_stabilized_pressure(idx)
                if rp > 0.0:
                    self._prev_button = 1
                    yield (rx, ry, rp)
                    raise StopIteration
                else:
                    yield (rx, ry, 0.0)
                    raise StopIteration

            rp = 0.0

        yield (rx, ry, rp)
        raise StopIteration

    def _get_initial_pressure(self, rp):
        self._stabilize_cnt += 1
        return rp * float(self._stabilize_cnt) / self.STABILIZE_START_MAX

    def reset(self):
        super(Stabilizer, self).reset()
        self._prev_rx = None
        self._prev_ry = None
        self._prev_button = None
        self._prev_time = None
        self._prev_rp = None
        self._release_time = None
        self._latest_pressure = None
        

    def _get_stabilized_x(self, idx):
        return self._samples_x[self.get_current_index(idx)]
    
    def _get_stabilized_y(self, idx):
        return self._samples_y[self.get_current_index(idx)]

    def _get_stabilized_pressure(self, idx):
        return self._samples_p[self.get_current_index(idx)]

    def fetch(self, x, y, pressure, time):
        """Fetch samples"""
        self._latest_pressure = pressure
        
        # To reject extreamly slow and near samples
        if self._prev_time == None or time - self._prev_time > 8:
            px = self._get_stabilized_x(0)
            py = self._get_stabilized_x(0)
            if math.hypot(x - px, y - py) > 4:
                super(self.__class__, self).fetch(x, y, pressure, time)
            self._prev_time = time

