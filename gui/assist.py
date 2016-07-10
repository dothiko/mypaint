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
        """ Enum current assisted position, from fetched samples.
        This is the most important method of assistant class.
        This method should return a value with yield.
        """
        pass # By the way, this is melely stub.

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

    ## Overlay drawing related

    def queue_draw_area(self, tdw):
        """ Queue draw area for overlay """
        pass

    def draw_overlay(self, cr):
        """ Drawing overlay """
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


class Stabilizer_Krita(Assistbase):
    """ Stabilizer class, like Krita's one. 

    This stablizer actually average angle.
    """
    STABILIZE_RADIUS = 40 # Stabilizer radius, in DISPLAY pixel.

    def __init__(self, app):
        super(Stabilizer_Krita, self).__init__()
        self._stabilize_cnt = None
        self.app = app

    def enum_current(self, button, time):

        px = self._prev_rx
        py = self._prev_ry

        if px == None or py == None:
            self._prev_rx = self._cx
            self._prev_ry = self._cy
            yield (self._prev_rx , self._prev_ry , 0)
            raise StopIteration

        dx = self._cx - px
        dy = self._cy - py
        cur_length = math.hypot(dx, dy)

        if cur_length <= self.STABILIZE_RADIUS:
            raise StopIteration

        move_length = cur_length - self.STABILIZE_RADIUS
        mx = (dx / cur_length) * move_length
        my = (dy / cur_length) * move_length
        self._prev_rx += mx
        self._prev_ry += my
        
        yield (self._prev_rx , self._prev_ry , self._latest_pressure)
        raise StopIteration


    def _get_initial_pressure(self, rp):
        self._stabilize_cnt += 1
        return rp * float(self._stabilize_cnt) / self.STABILIZE_START_MAX

    def reset(self):
        super(Stabilizer_Krita, self).reset()
        self._prev_rx = None
        self._prev_ry = None
        self._cx = None
        self._cy = None
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
        self._cx = x
        self._cy = y

        

    ## Overlay drawing related

    def _get_guide_center(self):
        if self._latest_pressure != 0.0:
            return (self._prev_rx, self._prev_ry)
        else:
            return (self._cx, self._cy)

    def queue_draw_area(self, tdw):
        """ Queue draw area for overlay """
        if self._prev_rx != None:
            full_rad = (self.STABILIZE_RADIUS + 2) * 2
            half_rad = full_rad / 2
            cx, cy = self._get_guide_center()
            tdw.queue_draw_area(cx - half_rad, cy - half_rad,
                    full_rad, full_rad)

    def draw_overlay(self, cr):
        """ Drawing overlay """
        if self._prev_rx != None:
            cr.save()
            cr.set_line_width(1)
            cx, cy = self._get_guide_center()
            cr.arc( cx, cy,
                    self.STABILIZE_RADIUS,
                    0.0,
                    2*math.pi)
            cr.stroke_preserve()
            cr.set_dash( (10,) )
            cr.set_source_rgb(1, 1, 1)
            cr.stroke()
            cr.restore()

