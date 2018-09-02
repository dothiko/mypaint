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
from lib.helpers import clamp
import logging
from collections import deque
logger = logging.getLogger(__name__)
import weakref

from gettext import gettext as _
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GLib

import numpy as np

import gui.mode
from drawutils import spline_4p, render_round_floating_color_chip

from lib import mypaintlib
import lib.helpers
import freehand_assisted
from gui.ui_utils import *
from gui.linemode import *

## Module settings
class _Phase:
    INVALID = -1
    DRAW = 0
    SET_POINT = 1
    INIT = 4
    FINALIZE = 5

class _Prefs:
    """Preference key constants"""
    LASTING_PREF_KEY = "assisted.centerpoint.context_lasting"
    DISTANCE_PREF_KEY = "assisted.centerpoint.context_distance"
    
    # Stroke context lasts when within 1 seconds 
    # from pen stylus detached previously.
    DEFAULT_LASTING_PREF = 1 

    # Stroke context lastes when the distance 
    # between previously released position is in this range.
    DEFAULT_DISTANCE_PREF = 32 

## Class defs
class CenterFreehandMode (freehand_assisted.AssistedFreehandMode):
    """Freehand drawing mode with centerpoint ruler.

    """

    ## Class constants & instance defaults
    ACTION_NAME = 'CenterFreehandMode'

    _initial_cursor = None
    _center_radius = 3

    # Centerpoint.
    _cx = None
    _cy = None


    ## Initialization

    def __init__(self, ignore_modifiers=True, **args):
        # Ignore the additional arg that flip actions feed us
        super(CenterFreehandMode, self).__init__(**args)
        self._sx = None
        if self.is_ready():
            self._phase = _Phase.INIT
        else:
            self._phase = _Phase.INVALID

    ## Metadata

    @classmethod
    def get_name(cls):
        return _(u"Freehand Centerpoint")

    def get_usage(self):
        return _(u"Paint free-form brush strokes with centerpoint ruler")

    ## Properties

    def is_ready(self):
        return (self._cx is not None)

    @property
    def initial_cursor(self):
        cursors = self.app.cursors
        cls = self.__class__
        if cls._initial_cursor == None:
            cls._initial_cursor = cursors.get_action_cursor(
                self.ACTION_NAME,
                gui.cursor.Name.CROSSHAIR_OPEN_PRECISE,
            )
        return cls._initial_cursor

    ## Mode stack & current mode
    def enter(self, doc, **kwds):
        """Enter freehand mode"""
        super(CenterFreehandMode, self).enter(doc, **kwds)
        if self.is_ready():
            self._ensure_overlay_for_tdw(doc.tdw)
            self.queue_draw_ui(doc.tdw)

    def leave(self, **kwds):
        super(CenterFreehandMode, self).leave(**kwds)
    
    
    ## Input handlers
    def drag_start_cb(self, tdw, event, pressure):
        self._latest_pressure = pressure
        self.start_x = event.x
        self.start_y = event.y
        self.last_x = event.x
        self.last_y = event.y

        if self.is_ready():
            hit_flag = self._is_hit_center(tdw, event.x, event.y)

            if hit_flag:
                # For centerpoint moving
                self._start_center_pt(tdw, event.x, event.y)
            else:
                # For drawing.
                self._rx, self._ry = tdw.display_to_model(event.x, event.y)
                self._sx, self._sy = self._rx, self._ry
                # To eliminate heading stroke glitch.
                self.queue_motion(tdw, event.time, self._rx, self._ry)
                self._update_center_vector()
                self._phase = _Phase.INIT

        elif self._phase == _Phase.INVALID:
            self._start_center_pt(tdw, event.x, event.y)
        else:
            print('other mode %d' % self._phase)

    def drag_update_cb(self, tdw, event, pressure):
        """ motion notify callback for assisted freehand drawing
        :return : boolean flag or None, True to CANCEL entire freehand motion 
                  handler and call motion_notify_cb of super-superclass.

        There is no mouse-hover(buttonless) event happen. 
        it can be detected only motion_notify_cb. 
        """
        if self._phase == _Phase.SET_POINT:
            self._start_center_pt(tdw, event.x, event.y)
            return True

    def motion_notify_cb(self, tdw, event, fakepressure=None):

        if self.last_button is None:
            cursor = None
            if self.is_ready():
                self.queue_draw_ui(tdw)
                if self._is_hit_center(tdw, event.x, event.y):
                    cursor = self.initial_cursor
            else:
                if self._phase == _Phase.INVALID:
                    cursor = self.initial_cursor

            if cursor is not None:
                tdw.set_override_cursor(cursor)
                self._overrided_cursor = cursor

            # XXX This also needed to eliminate stroke glitches.
            x, y = tdw.display_to_model(event.x, event.y)
            self.queue_motion(tdw, event.time, x, y)
            return True
        return super(CenterFreehandMode, self).motion_notify_cb(
                tdw, event, fakepressure)

    def drag_stop_cb(self, tdw, event):
        if self._phase == _Phase.DRAW:
            # To eliminate trailing stroke glitch.
            self.queue_motion(tdw, 
                              event.time,
                              self._sx, self._sy) 
            self._phase = _Phase.INIT
        elif self._phase == _Phase.SET_POINT:
            self._phase = _Phase.INIT
            self._update_positions(event.x, event.y)

    ## Mode options

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        cls = self.__class__
        if cls._OPTIONS_WIDGET is None:
            widget = CenterOptionsWidget(self)
            cls._OPTIONS_WIDGET = widget
        else:
            cls._OPTIONS_WIDGET.set_mode(self)
        return cls._OPTIONS_WIDGET

                
    def enum_samples(self, tdw):
        if not self.is_ready():
            raise StopIteration

        if self._phase == _Phase.DRAW:
            if self.is_ready():
                # All position attributes are in model coordinate.

                # _cx, _cy : center position 
                # _px, _py : previous position of stylus.
                # _sx, _sy : current position of 'stroke'. not stylus.
                # _vx, _vy : Identity vector of ruler direction.
                # _rx, _ry : raw point of stylus. This is display coord.

                # Calculate and reflect current stroking 
                # length and direction.
                length, nx, ny = length_and_normal(self._rx , self._ry, 
                        self._px, self._py)
                direction = cross_product(self._vy, -self._vx,
                        nx, ny)
                
                if length > 0:
                
                    if direction > 0.0:
                        length *= -1.0
                
                    tx = (length * self._vx) + self._sx
                    ty = (length * self._vy) + self._sy
                    self._sx = tx
                    self._sy = ty
                    yield (tx , ty , self._latest_pressure)
                    self._px, self._py = self._rx, self._ry

        elif self._phase == _Phase.INIT:
            if self.is_ready() and self.last_button is not None:
                # At here, we need to eliminate heading (a bit curved)
                # slightly visible stroke.
                # To do it, we need a point which is along ruler
                # but oppsite direction point.

                length, nx, ny = length_and_normal(self._rx , self._ry, 
                        self._cx, self._cy)
                direction = cross_product(self._vy, -self._vx,
                        nx, ny)

                tmp_length = 4.0 # practically enough length

                if length != 0 and direction < 0.0:
                    tmp_length *= -1.0

                tx = (tmp_length * self._vx) + self._rx
                ty = (tmp_length * self._vy) + self._ry

                yield (tx ,ty ,0.0)

                self._sx, self._sy = self._rx, self._ry
                self._px, self._py = self._rx, self._ry
                self._phase = _Phase.DRAW

        raise StopIteration

    def reset_assist(self):
        super(CenterFreehandMode, self).reset_assist()

        # _cx, _cy is center point.
        # all drawing stroke heads to/from this point.
        self._set_center_point(None, None)

        # _vx, _vy stores the identity vector of ruler, which is
        # from (_bx, _by) to (_dx, _dy) 
        # Each strokes should be parallel against this vector.
        # This attributes are set in _update_center_vector()
        self._vx = None
        self._vy = None

        # _px, _py is 'initially device pressed(started) point'
        # And they are updated each enum_samples() as
        # current end of stroke.
        self._px = None
        self._py = None

        # _rx, _ry is current pointer position,
        # in Display coordinate.
        self._rx = None
        self._ry = None

        self._phase = _Phase.INVALID

        self._overrided_cursor = None

    def fetch(self, tdw, x, y, pressure, time):
        """ Fetch samples(i.e. current stylus input datas) 
        into attributes.
        This method would be called each time motion_notify_cb is called.
        """
        if self.last_button is not None:
            self._last_time = time
            self._latest_pressure = pressure
            self._rx, self._ry = x, y

    def _update_positions(self, x, y):
        self._set_center_point(x, y)
    
    def _set_center_point(self, cx, cy):
        cls = self.__class__
        cls._cx = cx
        cls._cy = cy

    ## Overlay related

    def _generate_overlay(self, tdw):
        return _Overlay_Center(self, tdw)

    def queue_draw_ui(self, tdw):
        """ Queue draw area for overlay """
        if tdw is None:
            for tdw in self._overlays.keys():
                self.queue_draw_ui(tdw)
            return
        x, y = tdw.model_to_display(self._cx, self._cy)
        radius = self._center_radius + gui.style.DROP_SHADOW_BLUR + 1
        tdw.queue_draw_area(x - radius, y - radius,
                            radius*2+1, radius*2+1)

    ## Centerpoint related

    def _start_center_pt(self, tdw, dx, dy):
        mpos = tdw.display_to_model(dx, dy)
        # In this time, _cx & _px & _sx are
        # same.
        self._cx, self._cy = mpos
        self._px, self._py = mpos
        self._sx, self._sy = mpos
        self._phase = _Phase.SET_POINT
        self.queue_draw_ui(tdw)

    def _update_center_vector(self):
        self._vx, self._vy = normal(self._cx, self._cy, 
                                    self._rx, self._ry)

    def _is_hit_center(self, tdw, dx, dy):
        cdx, cdy = tdw.model_to_display(self._cx, self._cy)
        diff = math.hypot(dx-cdx, dy-cdy)
        return (diff <= self._center_radius)


class CenterOptionsWidget (freehand_assisted.AssistantOptionsWidget):
    """Configuration widget for freehand mode"""

    def __init__(self, mode):
        super(CenterOptionsWidget, self).__init__(mode)

    def init_specialized_widgets(self, row):
        self._updating_ui = True
        row = super(CenterOptionsWidget, self).init_specialized_widgets(row)

        self._create_slider(
            row,
            _("Context lasting:"), 
            self._lasting_changed_cb,
            pref.get(_Prefs.LASTING_PREF_KEY, _Prefs.DEFAULT_LASTING_PREF),
            0, 
            3.0 # Maximum 3 seconds
        )
        row += 1

        self._create_slider(
            row,
            _("Allowed distance:"), 
            self._context_distance_changed_cb,
            pref.get(_Prefs.DISTANCE_PREF_KEY, _Prefs.DEFAULT_DISTANCE_PREF),
            16.0, 
            64.0 # Maximum 64 pixels
        )
        row += 1

        button = Gtk.Button(label = _("Clear ruler")) 
        button.connect('clicked', self._reset_clicked_cb)
        self.attach(button, 0, row, 2, 1)
        row += 1
        
        self._updating_ui = False
        return row

    def _reset_clicked_cb(self, button):
        if not self._updating_ui:
            # To discard current(old) overlay.
            mode = self.mode
            if mode:
                mode.queue_draw_ui(None) # To erase.
                mode.reset_assist()

    def _lasting_changed_cb(self, adj, data=None):
        if not self._updating_ui:
            mode = self.mode
            if mode:
                value = adj.get_value()
                self.mode.context_lasting = value
                self.app.preferences[_Prefs.LASTING_PREF_KEY] = value

    def _context_distance_changed_cb(self, adj, data=None):
        if not self._updating_ui:
            mode = self.mode
            if mode:
                value = adj.get_value()
                self.mode.context_distance = value
                self.app.preferences[_Prefs.DISTANCE_PREF_KEY] = value

class _Overlay_Center(gui.overlays.Overlay):
    """Overlay for stabilized freehand mode """

    def __init__(self, mode, tdw):
        super(_Overlay_Center, self).__init__()
        self._mode_ref = weakref.ref(mode)
        self._tdw_ref = weakref.ref(tdw)

    def paint(self, cr):
        """Draw brush size to the screen"""
        mode = self._mode_ref()
        if mode is not None:
            tdw = self._tdw_ref()
            assert tdw is not None
            if (mode.is_ready() or 
                    mode._phase == _Phase.SET_POINT): 
                x, y = tdw.model_to_display(mode._cx, mode._cy)
                render_round_floating_color_chip(
                        cr,
                        x, y,
                        gui.style.EDITABLE_ITEM_COLOR,
                        mode._center_radius)

