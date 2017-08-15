# This file is part of MyPaint.
# Copyright (C) 2017 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""Freehand drawing modes"""

## Imports
from __future__ import division, print_function

import math
import logging
from collections import deque
logger = logging.getLogger(__name__)
import weakref
import time

from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GLib
from gettext import gettext as _
import numpy as np

import gui.mode
from drawutils import spline_4p
from lib import mypaintlib
import lib.helpers
import freehand_assisted
from gui.ui_utils import *
from lib.helpers import clamp


## Module settings

## Class defs
class _Phase:
    INVALID = -1
    INIT = 0
    DRAW = 1

class StabilizedFreehandMode (freehand_assisted.AssistedFreehandMode):
    """Freehand drawing mode with stablizer

    How stabilizer work:

    Stabilizer is one of 'angle-averager', which has 
    ignoring circle.
    when pointer move inside ignoring circle, no stroke
    drawn. 
    when pointer get out from that circle, stroke drawn
    from the center point of circle.
    the stroke direction is center point to pointer,
    and stroke length is the distance of from edge of circle 
    to pointer.
    """

    ## Class constants & instance defaults

    ACTION_NAME = 'StabilizedFreehandMode'
    permitted_switch_actions = set()   # Any action is permitted

    _X_TILT_OFFSET = 0.0    # XXX Class global tilt offsets, to
    _Y_TILT_OFFSET = 0.0    # enable change tilt parameters for
                            # non-tilt-sensible pen stylus.

    # Stabilizer constants.
    #
    # In Stabilizer, self._phase decides how enum_samples()
    # work. 
    # Usually, it enumerate nothing (MODE_INVALID)
    # so no any strokes drawn.
    # When the device is pressed, self._phase is set as 
    # MODE_INIT to initialize stroke at enum_samples().
    # After that, self._phase is set as MODE_DRAW.
    # and enum_samples() yields some modified device positions,
    # to draw strokes.
    # Finally, when the device is released,self._phase is set
    # as MODE_FINALIZE, to avoid trailing stroke glitches.
    # And self._phase returns its own normal state, i.e. MODE_INVALID.
    
    _AVERAGE_PREF_KEY = "assisted.stabilizer.average_previous"
    _STABILIZE_RANGE_KEY = "assisted.stabilizer.range"
    _ACTIVATE_BY_MOD_KEY = "assisted.stabilizer.activate_by_modifier"
    _FORCE_TAPERING_KEY = "assisted.stabilizer.force_tapering"

    _TAPERING_LENGTH = 32.0 
    _TIMER_PERIOD = 500.0

    _average_previous = None
    _stabilize_range = None
    _range_switcher = None
    _force_tapering = None

    # About Stabilizer attributes:
    #   _rx,_ry == Raw input of pointer, 
    #              the 'current' position of input, .i.e event.x&y
    #              stroke should be drawn TO (not from) this point.
    #
    #   _cx, _cy == Current center of stabilizer circle
    #               These also represent the previous end point of stroke.
    #
    #   _prev_dx, _prev_dy == Previously calculated differencial of
    #                         _rx - _cx & _ry - _cy.
    #                         This is used for averaging direction.
    #
    #   _current_range == The stabilizer range. it is variable, to
    #                     implement "Range switcher" functionality.
    #                     This might be 0(disabled) or 
    #                     half of _stabilize_range or same as _stabilize_range.


    ## Initialization

    def __init__(self, ignore_modifiers=True, **args):

        # Initialize class attributes.
        if self._average_previous is None:
            pref = self.app.preferences
            cls = self.__class__
            cls._average_previous = pref.get(self._AVERAGE_PREF_KEY, True)
            cls._stabilize_range = pref.get(self._STABILIZE_RANGE_KEY, 48)
            cls._range_switcher = pref.get(self._ACTIVATE_BY_MOD_KEY, True)
            cls._force_tapering = pref.get(self._FORCE_TAPERING_KEY, False)

        # Ignore the additional arg that flip actions feed us
        super(StabilizedFreehandMode, self).__init__(**args)

    ## Metadata

    @classmethod
    def get_name(cls):
        return _(u"Freehand Drawing with Stabilizer")

    def get_usage(self):
        return _(u"Paint free-form brush strokes with stabilizer")

    ## Properties
    
    ## Input handlers
    def drag_start_cb(self, tdw, event, pressure):
        self._start_time = event.time
        self._last_time = event.time
        self._rx = event.x
        self._ry = event.y
        self._drag_x = event.x
        self._drag_y = event.y

        self._init_stabilize() # this should be called after _rx/_ry is set.

        if self.overrided:
            if (self._range_switcher
                    and not event.state & Gdk.ModifierType.SHIFT_MASK):
                self._current_range = self._stabilize_range / 2
            else:
                self._current_range = self._stabilize_range

    def drag_update_cb(self, tdw, event, pressure):
        """ motion notify callback for assisted freehand drawing
        :return : boolean flag, True to CANCEL entire freehand motion 
                  handler and call motion_notify_cb of super-superclass.
        """

        # Adding drag length to detect drawing speed.
        self._drag_length += math.hypot(event.x - self._drag_x, 
                                        event.y - self._drag_y)
        self._drag_x = event.x
        self._drag_y = event.y

        # If Alt key pressed when stabilizer is in range_switcher mode
        # and not overrided, then expand the stabilizer range immidiately.
        if (not self.overrided 
                and self._current_range > 0
                and self._current_range < self._stabilize_range
                and event.state & Gdk.ModifierType.MOD1_MASK):
            self._current_range = self._stabilize_range
            self.queue_draw_ui(tdw)

        # If the stylus cursor is inside stabilize circle,
        # all drawing events should be ignored.
        length = math.hypot(self._cx - event.x, self._cy - event.y)
        return self._current_range > length

    def motion_notify_cb(self, tdw, event, fakepressure=None):

        if self.last_button is None:
            # XXX This also needed to eliminate stroke glitches.
            x, y = tdw.display_to_model(event.x, event.y)
            self.queue_motion(tdw, event.time, x, y)
            return True
        return super(StabilizedFreehandMode, self).motion_notify_cb(
                tdw, event, fakepressure)

    def drag_stop_cb(self, tdw, event):
        self._phase = _Phase.INVALID
        x, y = tdw.display_to_model(self._cx, self._cy)
        self.queue_motion(tdw, 
                          event.time, 
                          x, y)

        if self._range_switcher:
            self._drag_length = 0
            self._current_range = 0
            self._drag_x = 0
            self._drag_y = 0

    ## Mode options

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        cls = self.__class__
        if cls._OPTIONS_WIDGET is None:
            widget = StabilizerOptionsWidget(self)
            cls._OPTIONS_WIDGET = widget
        else:
            cls._OPTIONS_WIDGET.set_mode(self)
        return cls._OPTIONS_WIDGET

                
    def enum_samples(self):

        if self._phase == _Phase.INIT:
            # Drawing initial pressure, to avoid heading glitch.
            # That value would be 0.0 in normal.
            # However, When auto-range adjust is enabled,
            # drawing stroke is already undergo. in such case,
            # 'initial pressure' would be the pressure of current stylus input,
            # not 0.0. Therefore, not use constant(0.0) for initial pressure.
            # And,after this cycle, proceed normal stabilized stage.
            self._phase = _Phase.DRAW       
            yield (self._cx , self._cy , self._initial_pressure)        
        elif self._phase == _Phase.DRAW:
            # Normal stabilize stage.

            cx = self._cx
            cy = self._cy
            x = self._rx
            y = self._ry

            dx = x - cx
            dy = y - cy
            cur_length = math.hypot(dx, dy)

            if cur_length <= self._current_range:
                raise StopIteration

            if (self._current_range > 0.0 and 
                    self._average_previous):
                if self._prev_dx is not None:
                    dx = (dx + self._prev_dx) / 2.0
                    dy = (dy + self._prev_dy) / 2.0
                self._prev_dx = dx
                self._prev_dy = dy

            move_length = cur_length - self._current_range
            mx = (dx / cur_length) * move_length
            my = (dy / cur_length) * move_length

            self._actual_drawn_length += move_length
            
            self._cx = cx + mx
            self._cy = cy + my

            pressure = (self._latest_pressure + self._initial_pressure) * 0.5
            self._initial_pressure = self._latest_pressure
             
            if (self._force_tapering and 
                    self._actual_drawn_length < self._TAPERING_LENGTH):
                adj = min(1.0, self._actual_drawn_length / self._TAPERING_LENGTH)
                yield (self._cx , self._cy , pressure * adj)
            else:
                yield (self._cx , self._cy , pressure)
        else:
            # Set empty stroke, as usual.
            yield (self._rx, self._ry, 0.0)

        raise StopIteration
        

    def reset_assist(self):
        # We need special initialization for resetting this class
        # So do not supercall
        self._initialize_attrs(_Phase.INVALID,
                               0.0, 0.0, 0.0)

    def _init_stabilize(self):
        self._initialize_attrs(_Phase.INIT,
                               self._rx, self._ry, 0.0)

    def _initialize_attrs(self, phase, x, y, pressure):
        self._latest_pressure = pressure
        self._cx = x
        self._cy = y
        self._initial_pressure = 0.0
        self._prev_dx = None
        self._phase = phase
        self._actual_drawn_length = 0.0
        self._drag_length = 0.0
        if self._range_switcher:
            self._start_time = time.time()
            self._current_range = 0.0
        else:
            self._current_range = self._stabilize_range

    def fetch(self, x, y, pressure, time):
        """ Fetch samples(i.e. current stylus input datas) 
        into attributes.
        This method would be called each time motion_notify_cb is called.

        Explanation of attributes which are used at here:
        
        """

        self._latest_pressure = pressure
        self._rx = x
        self._ry = y

    ## Overlay related

    def _generate_overlay(self, tdw):
        return _Overlay_Stabilizer(self, tdw)

    def queue_draw_ui(self, tdw):
        """ Queue draw area for overlay """
        if tdw is None:
            for tdw in self._overlays:
                self.queue_draw_ui(tdw)
            return

        if self._ready:
            half_rad = int(self._current_range+ 2)
            full_rad = half_rad * 2
            tdw.queue_draw_area(self._cx - half_rad, self._cy - half_rad,
                    full_rad, full_rad)

    ## Stabilizer configuration related
    @property
    def _ready(self):
        return (self._phase in (_Phase.DRAW, _Phase.INIT) and
                self._current_range > 0)

    @property
    def average_previous(self):
        return self._average_previous
    
    @average_previous.setter
    def average_previous(self, flag):
        cls = self.__class__
        cls._average_previous = flag
        self.app.preferences[self._AVERAGE_PREF_KEY] = flag

    @property
    def stabilize_range(self):
        return self._stabilize_range
    
    @stabilize_range.setter
    def stabilize_range(self, value):
        cls = self.__class__
        cls._stabilize_range = value
        if self._current_range > value:
            self._current_range = value
        self.app.preferences[self._STABILIZE_RANGE_KEY] = value

    @property
    def range_switcher(self):
        return self._range_switcher
    
    @range_switcher.setter
    def range_switcher(self, flag):
        cls = self.__class__
        cls._range_switcher = flag
        self.app.preferences[self._ACTIVATE_BY_MOD_KEY] = flag

    @property
    def force_tapering(self):
        return self._force_tapering

    @force_tapering.setter
    def force_tapering(self, flag):
        cls = self.__class__
        cls._force_tapering = flag
        self.app.preferences[self._FORCE_TAPERING_KEY] = flag


class StabilizerOptionsWidget (freehand_assisted.AssistantOptionsWidget):
    """Configuration widget for freehand mode"""

    def __init__(self, mode):
        super(StabilizerOptionsWidget, self).__init__(mode)

    def init_specialized_widgets(self, row):
        self._updating_ui = True
        row = super(StabilizerOptionsWidget, self).init_specialized_widgets(row)

        def create_slider(label, handler, 
                          value, min_adj, max_adj, step_incr=1,
                          digits=1 ):
            labelobj = Gtk.Label(halign=Gtk.Align.START)
            labelobj.set_text(label)
            self.attach(labelobj, 0, row, 1, 1)

            adj = Gtk.Adjustment(value, min_adj, max_adj, 
                                 step_incr=step_incr)
            adj.connect('value-changed', handler)

            scale = Gtk.HScale(hexpand_set=True, hexpand=True, 
                    halign=Gtk.Align.FILL, adjustment=adj, digits=digits)
            scale.set_value_pos(Gtk.PositionType.RIGHT)
            self.attach(scale, 1, row, 1, 1)
            return scale

        create_slider(_("X Tilt Offset:"), 
                      self.x_tilt_offset_adj_changed_cb,
                      0.0, -1.0, 1.0, 0.01
                      )
        row += 1

        create_slider(_("Y Tilt Offset:"), 
                      self.y_tilt_offset_adj_changed_cb,
                      0.0, -1.0, 1.0, 0.01
                      )
        row += 1

        # Add VBox for Assistant area
        mode = self.mode_ref()
        assert mode is not None

        # Scale slider for range circle setting.
        create_slider(_("Range:"), 
                      self._range_changed_cb,
                      mode.stabilize_range, 32, 64)
        row += 1

        # Checkbox for average direction.
        checkbox = Gtk.CheckButton(_("Average direction"),
            hexpand_set=True, hexpand=True, halign=Gtk.Align.FILL)
        checkbox.set_active(mode.average_previous)
        checkbox.connect('toggled', self._average_toggled_cb)
        self.attach(checkbox, 0, row, 2, 1)
        row += 1

        # Checkbox for use 'range switcher' feature.
        checkbox = Gtk.CheckButton(_("Activate by modifier"),
            hexpand_set=True, hexpand=True, halign=Gtk.Align.FILL)
        checkbox.set_active(mode.range_switcher) 
        checkbox.connect('toggled', self._range_switcher_toggled_cb)
        self.attach(checkbox, 0, row, 2, 1)
        row += 1

        # Checkbox for use 'Force start tapering' feature.
        checkbox = Gtk.CheckButton(_("Force start tapering"),
            hexpand_set=True, hexpand=True, halign=Gtk.Align.FILL)
        checkbox.set_active(mode.force_tapering) 
        checkbox.connect('toggled', self._force_tapering_toggled_cb)
        self.attach(checkbox, 0, row, 2, 1)
        row += 1

        self._updating_ui = False
        return row

    def x_tilt_offset_adj_changed_cb(self, adj):
        FreehandMode._X_TILT_OFFSET = adj.get_value()

    def y_tilt_offset_adj_changed_cb(self, adj):
        FreehandMode._Y_TILT_OFFSET = adj.get_value()

    # Handlers
    def _average_toggled_cb(self, checkbox):
        if not self._updating_ui:
            self.mode.average_previous = checkbox.get_active()

    def _range_changed_cb(self, adj):
        if not self._updating_ui:
            self.mode.stabilize_range = adj.get_value()

    def _range_switcher_toggled_cb(self, checkbox):
        if not self._updating_ui:
            self.mode.range_switcher = checkbox.get_active()

    def _force_tapering_toggled_cb(self, checkbox):
        if not self._updating_ui:
            self.mode.force_tapering = checkbox.get_active()

class _Overlay_Stabilizer(gui.overlays.Overlay):
    """Overlay for stabilized freehand mode """

    def __init__(self, mode, tdw):
        super(_Overlay_Stabilizer, self).__init__()
        self._mode_ref = weakref.ref(mode)
        self._tdw_ref = weakref.ref(tdw)

    def paint(self, cr):
        """Draw brush size to the screen"""
        mode = self._mode_ref()
        if mode is not None and mode._ready:
            x, y = mode._cx, mode._cy
            self._draw_dashed_circle(cr, 
                    (x, y, mode._current_range))

            # XXX Drawing actual stroke point.
            # This should be same size as current brush radius,
            # but it would be a huge workload when the brush is 
            # extremely large.
            # so, currently this is fixed size, only shows the center
            # point of stroke.
            self._draw_dashed_circle(cr, 
                    (x, y, 2))

    @dashedline_wrapper
    def _draw_dashed_circle(self, cr, info):
        x, y, radius = info
        cr.arc( x, y,
                int(radius),
                0.0,
                2*math.pi)

