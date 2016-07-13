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
from gettext import gettext as _

from gi.repository import Gtk, Gdk

import gui.drawutils
import gui.tileddrawwidget

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

    def enum_current(self):
        """ Enum current assisted position, from fetched samples.
        This is the most important method of assistant class.
        This method should return a value with yield.
        """
        pass # By the way, this is melely stub.

    def fetch(self, x, y, pressure, time, button):
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

    ## Options presentor for assistant
    #  Return Gtk.Box type widget ,to fill freehand option presenter.
    def get_presenter(self):
        pass

class Averager(Assistbase):
    """ Averager Stabilizer class, which fetches 
    gtk.event x/y position as a sample,and return 
    the average of recent samples.
    """
    name = _("Averager")
    STABILIZE_START_MAX = 24

    def __init__(self, app):
        super(Stabilizer, self).__init__()
        self._stabilize_cnt = None
        self.app = app

    def enum_current(self):
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

        self._prev_rx = rx
        self._prev_ry = ry
        self._prev_rp = rp

        # Heading / Trailing glitch workaround
        if self._last_button == 1:
            if (self._prev_button == None):
                self._stabilize_cnt = 0
                yield (rx, ry, 0.0)
                yield (rx, ry, self._get_initial_pressure(rp))
                raise StopIteration
            elif self._stabilize_cnt < self.STABILIZE_START_MAX:
                rp = self._get_initial_pressure(rp)
        elif self._last_button == None:
            if (self._prev_button != None):
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
        self._last_button = None
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

    def fetch(self, x, y, pressure, time, button):
        """Fetch samples"""
        self._latest_pressure = pressure
        self._prev_button = self._last_button
        self._last_button = button
        
        # To reject extreamly slow and near samples
        if self._prev_time == None or time - self._prev_time > 8:
            px = self._get_stabilized_x(0)
            py = self._get_stabilized_x(0)
            if math.hypot(x - px, y - py) > 4:
                super(self.__class__, self).fetch(x, y, pressure, time)
            self._prev_time = time


class Stabilizer(Assistbase):
    """ Stabilizer class, like Krita's one. 

    This stablizer actually 'average angle'.
    """
    name = _("Stabilizer")
    STABILIZE_RADIUS = 48 # Stabilizer radius, in DISPLAY pixel.

    def __init__(self, app):
        super(Stabilizer, self).__init__()
        self._stabilize_cnt = None
        self.app = app
        self._raw_x = None
        self._raw_y = None
        self._last_button = None
        self._prev_button = None
        self._initial = False
        self._average_previous = True
        self._prev_dx = self._prev_dy = None
        self._presenter = None

    @property
    def _enabled(self):
        return self._last_button != None

    def enum_current(self):

        if self._last_button == 1:
            if (self._prev_button == None):
                self._cx = self._px = self._raw_x
                self._cy = self._py = self._raw_y
                self._initial = True
               #self._prev_pressure = 0.0
                yield (self._cx , self._cy , 0.0)
                raise StopIteration

            cx = self._cx
            cy = self._cy

            dx = self._raw_x - cx
            dy = self._raw_y - cy
            cur_length = math.hypot(dx, dy)

            if cur_length <= self.STABILIZE_RADIUS:
                raise StopIteration

            if self._average_previous:
                if self._prev_dx != None:
                    dx = (dx + self._prev_dx) / 2.0
                    dy = (dy + self._prev_dy) / 2.0
                self._prev_dx = dx
                self._prev_dy = dy

            move_length = cur_length - self.STABILIZE_RADIUS
            mx = (dx / cur_length) * move_length
            my = (dy / cur_length) * move_length


            self._px = cx
            self._py = cy
            self._cx = cx + mx
            self._cy = cy + my

            
            if self._initial:
                yield (self._cx , self._cy , 0.0) # To avoid heading glitch
                self._initial = False
        
           #yield (cx , cy , self._prev_pressure) # To avoid initial glitch
           #self._prev_pressure = self._latest_pressure
            yield (self._cx , self._cy , self._latest_pressure)

        elif self._last_button == None:
            if (self._prev_button != None):
                if self._latest_pressure > 0.0:
                    # button is released but
                    # still remained some pressure...
                    # rare case,but possible.
                    yield (self._cx, self._cy, self._latest_pressure)
                yield (self._cx, self._cy, 0.0)
            yield (self._raw_x, self._raw_y, 0.0) # We need this for avoid trailing glitch


        raise StopIteration



    def reset(self):
        super(Stabilizer, self).reset()
        self._px = None
        self._py = None
        self._cx = None
        self._cy = None
        self._latest_pressure = None
        self._prev_dx = None

    def fetch(self, x, y, pressure, time, button):
        """Fetch samples"""

        self._prev_button = self._last_button
        self._last_button = button

        self._latest_pressure = pressure
        self._raw_x = x
        self._raw_y = y
        

    ## Overlay drawing related

    def queue_draw_area(self, tdw):
        """ Queue draw area for overlay """
        if self._enabled:
            full_rad = (self.STABILIZE_RADIUS + 2) * 2
            half_rad = full_rad / 2
            tdw.queue_draw_area(self._cx - half_rad, self._cy - half_rad,
                    full_rad, full_rad)

    def draw_overlay(self, cr):
        """ Drawing overlay """
        if self._enabled:
            cr.save()
            cr.set_line_width(1)
            cr.arc( self._cx, self._cy,
                    self.STABILIZE_RADIUS,
                    0.0,
                    2*math.pi)
            cr.stroke_preserve()
            cr.set_dash( (10,) )
            cr.set_source_rgb(1, 1, 1)
            cr.stroke()
            cr.restore()


    ## Options presenter for assistant
    def get_presenter(self):
        if self._presenter == None:
            self._presenter = Optionpresenter_Stabilizer(self)
        return self._presenter.get_box_widget()

## Option presenters for assistants

class _Presenter_Mixin(object):
    """ Base Mixin of assistants option presenter"""

    def force_redraw_overlay(self, area=None):
        for tdw in gui.tileddrawwidget.TiledDrawWidget.get_visible_tdws():
            if area:
                tdw.queue_draw_area(*area)
            else:
                tdw.queue_draw()

class Optionpresenter_Stabilizer(_Presenter_Mixin):
    """ Optionpresenter for Stabilizer assistant.
    """

    def __init__(self, assistant):
        self.assistant = assistant
        grid = Gtk.Grid(column_spacing=6, row_spacing=4)
        grid.set_hexpand_set(True)

        checkbox = Gtk.CheckButton(_("Average direction:"),
            hexpand_set=True, hexpand=True, halign=Gtk.Align.FILL)
        checkbox.haligh = Gtk.Align.FILL
        grid.attach(checkbox,0,0,2,1)
        
        label = Gtk.Label(halign=Gtk.Align.START)
        label.set_text(_("Range:"))
        grid.attach(label,0,1,1,1)

        scale = Gtk.HScale(hexpand_set=True, hexpand=True, halign=Gtk.Align.FILL)
        scale.set_range(32,64)
        scale.set_increments(1,1)
        scale.set_value(48)
        scale.set_value_pos(Gtk.PositionType.RIGHT)
        scale.set_hexpand_set(True)
        grid.attach(scale,1,1,1,1)

        grid.show_all()
        self._grid = grid

    def get_box_widget(self):
        return self._grid


