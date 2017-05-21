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
from drawutils import spline_4p

from lib import mypaintlib
import lib.helpers
import freehand_assisted
from gui.ui_utils import *
from gui.rulercontroller import *
from gui.linemode import *

## Module settings
class _Phase:
    INVALID = -1
    DRAW = 0
    SET_BASE = 1
    SET_DEST = 2
    INIT = 4
    RULER = 5

## Class defs
class ParallelFreehandMode (freehand_assisted.AssistedFreehandMode):
    """Freehand drawing mode with parallel ruler.

    """

    ## Class constants & instance defaults
    ACTION_NAME = 'ParallelFreehandMode'

    _initial_cursor = None
    _level_margin = 0.005 # in radian, practical value.

    ## Class variables

    # Level vector. This tuple means x and y of identity vector.
    # If the ruler have completely same angle with this,
    # 'level indicator' would be shown.
    _level_vector = (0.0, 1.0)

    ## Initialization

    def __init__(self, ignore_modifiers=True, **args):
        # Ignore the additional arg that flip actions feed us

        # Initialize ruler before calling super-constructor.
        # because it would call reset_assist.
        # that method refer self._ruler.
        self._ruler = RulerController(self.app)
        
        super(ParallelFreehandMode, self).__init__(**args)

    ## Metadata

    @classmethod
    def get_name(cls):
        return _(u"Freehand Drawing with Parallel ruler")

    def get_usage(self):
        return _(u"Paint free-form brush strokes with parallel ruler")

    ## Properties

    def is_ready(self):
        return (self._ruler.is_ready() 
                and self.last_button is not None)

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
        super(ParallelFreehandMode, self).enter(doc, **kwds)
        if self._ruler.is_ready():
            self._ensure_overlay_for_tdw(doc.tdw)
            self.queue_draw_ui(doc.tdw)

   #def leave(self, **kwds):
   #    """Leave freehand mode"""
   #    self.queue_draw_ui(None)
   #    super(AssistedFreehandMode, self).leave(**kwds)
   #    self._discard_overlays()
    
    ## Input handlers
    def drag_start_cb(self, tdw, event, pressure):
        self._tdw = tdw
        self._latest_pressure = pressure
        self.start_x = event.x
        self.start_y = event.y
        self.last_x = event.x
        self.last_y = event.y

        if self._ruler.is_ready():
            node_idx = self._ruler.hittest_node(tdw, event.x, event.y)

            if node_idx is not None:
                # For ruler moving
                self._phase = _Phase.RULER
                self._ruler.button_press_cb(self, tdw, event)
                self._ruler.drag_start_cb(self, tdw, event)
            else:
                # For drawing.
                self._update_positions(event.x, event.y, True)
                # To eliminate heading stroke glitch.
                self.queue_motion(tdw, event.time, self._cx, self._cy)

        elif self._phase == _Phase.INVALID:
            self._phase = _Phase.SET_BASE
            self._ruler.set_start_pos(tdw, (event.x, event.y))
        elif self._phase == _Phase.SET_DEST:
            self._ruler.set_end_pos(tdw, (event.x, event.y))
        else:
            print('other mode %d' % self._phase)

    def drag_update_cb(self, tdw, event, pressure):
        """ motion notify callback for assisted freehand drawing
        :return : boolean flag or None, True to CANCEL entire freehand motion 
                  handler and call motion_notify_cb of super-superclass.

        There is no mouse-hover(buttonless) event happen. 
        it can be detected only motion_notify_cb. 
        """
        if self._phase == _Phase.SET_BASE:
            self._ruler.set_start_pos(tdw, (event.x, event.y))
            self.queue_draw_ui(tdw)
            return True
        elif self._phase == _Phase.SET_DEST:
            self._ruler.set_end_pos(tdw, (event.x, event.y))
            self.queue_draw_ui(tdw)
            return True
        elif self._phase == _Phase.RULER:
            dx = event.x - self.last_x 
            dy = event.y - self.last_y 
            self.last_x = event.x
            self.last_y = event.y
            self._ruler.drag_update_cb(self, tdw, event, dx, dy)

            self.queue_draw_ui(tdw)
            return True

    def motion_notify_cb(self, tdw, event, fakepressure=None):

        if self.last_button is None:
            if self._ruler.is_ready():
                self.queue_draw_ui(tdw)
                self._ruler.update_zone_index(self, tdw, event.x, event.y)
                cursor = self._ruler.update_cursor_cb(tdw)
                if cursor != self._overrided_cursor:
                    tdw.set_override_cursor(cursor)
                self._overrided_cursor = cursor
            else:
                if self._phase == _Phase.INVALID:
                    cursor = self.initial_cursor
                    tdw.set_override_cursor(cursor)
                    self._overrided_cursor = cursor

            # XXX This also needed to eliminate stroke glitches.
            x, y = tdw.display_to_model(event.x, event.y)
            self.queue_motion(tdw, event.time, x, y)
            return True
        return super(ParallelFreehandMode, self).motion_notify_cb(
                tdw, event, fakepressure)

    def drag_stop_cb(self, tdw, event):
        if self._phase == _Phase.DRAW:
            # To eliminate trailing stroke glitch.
            self.queue_motion(tdw, 
                              event.time, 
                              self._sx, self._sy)
            self._phase = _Phase.INIT
        elif self._phase == _Phase.SET_BASE:
            self._phase = _Phase.SET_DEST
        elif self._phase == _Phase.SET_DEST:
            self._phase = _Phase.INIT
            self._update_ruler_vector()
            self.queue_draw_ui(tdw)
        elif self._phase == _Phase.INIT:
            # Initialize mode but nothing done
            # = simply return to initial state.
            self._phase = _Phase.INVALID
        elif self._phase == _Phase.RULER:
            self._ruler.button_release_cb(self, tdw, event)
            self._ruler.drag_stop_cb(self, tdw)
            self._phase = _Phase.INIT
            self._update_ruler_vector()
            self.queue_draw_ui(tdw)

        self._tdw = None

    ## Mode options

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        cls = self.__class__
        if cls._OPTIONS_WIDGET is None:
            widget = ParallelOptionsWidget(self)
            cls._OPTIONS_WIDGET = widget
        else:
            cls._OPTIONS_WIDGET.set_mode(self)
        return cls._OPTIONS_WIDGET

                
    def enum_samples(self):
        if not self.is_ready():
            raise StopIteration

        tdw = self._tdw
        assert tdw is not None

        if self._phase == _Phase.DRAW:
            if self.is_ready():
                # All position attributes are in model coordinate.

                # _cx, _cy : current position of stylus
                # _px, _py : previous position of stylus.
                # _sx, _sy : current position of 'stroke'. not stylus.
                # _vx, _vy : Identity vector of ruler direction.


                # Calculate and reflect current stroking 
                # length and direction.
                length, nx, ny = length_and_normal(self._cx , self._cy, 
                        self._px, self._py)
                direction = cross_product(self._vy, -self._vx,
                        nx, ny)
                
                if length > 0:
                
                    if direction > 0.0:
                        length *= -1.0
                
                    cx = (length * self._vx) + self._sx
                    cy = (length * self._vy) + self._sy
                    self._sx = cx
                    self._sy = cy
                    cx, cy = tdw.model_to_display(cx, cy)
                    yield (cx , cy , self._latest_pressure)
                    self._px, self._py = self._cx, self._cy

        elif self._phase == _Phase.INIT:
            if self._ruler.is_ready() and self.last_button is not None:
                # At here, we need to eliminate heading (a bit curved)
                # slightly visible stroke.
                # To do it, we need a point which is along ruler
                # but oppsite direction point.


                length, nx, ny = length_and_normal(self._cx , self._cy, 
                        self._px, self._py)
                direction = cross_product(self._vy, -self._vx,
                        nx, ny)

                tmp_length = 4.0 # practically enough length

                if length != 0 and direction < 0.0:
                    tmp_length *= -1.0

                cx = (tmp_length * self._vx) + self._px
                cy = (tmp_length * self._vy) + self._py

                cx, cy = tdw.model_to_display(cx, cy)
                yield (cx ,cy ,0.0)
                cx, cy = tdw.model_to_display(self._px, self._py)
                yield (cx ,cy ,0.0)

                self._phase = _Phase.DRAW

        raise StopIteration

    def reset_assist(self):
        super(ParallelFreehandMode, self).reset_assist()

        # _vx, _vy stores the identity vector of ruler, which is
        # from (_bx, _by) to (_dx, _dy) 
        # Each strokes should be parallel against this vector.
        # This attributes are set in _update_ruler_vector()
        self._vx = None
        self._vy = None

        # _px, _py is 'initially device pressed(started) point'
        # And they are updated each enum_samples() as
        # current end of stroke.
        self._px = None
        self._py = None

        self._tdw = None

        self._ruler.reset()
        if self._ruler.is_ready():
            self._phase = _Phase.INIT
        else:
            self._phase = _Phase.INVALID

        self._overrided_cursor = None

    def fetch(self, x, y, pressure, time):
        """ Fetch samples(i.e. current stylus input datas) 
        into attributes.
        This method would be called each time motion_notify_cb is called.
        """
        if self.last_button is not None:
            self._last_time = time
            self._latest_pressure = pressure
            self._update_positions(x, y, False)

    ## Overlay related

    def _generate_overlay(self, tdw):
        return _Overlay_Parallel(self, tdw)

    def queue_draw_ui(self, tdw):
        """ Queue draw area for overlay """
        if tdw is None:
            for tdw in self._overlays.keys():
                self.queue_draw_ui(tdw)
            return
        self._ruler.queue_redraw(tdw)

    ## Ruler related

    def _update_positions(self, dx, dy, starting):
        """ update current positions from pointer position 
        of display coordinate.
        """
        assert self._tdw is not None
        mpos = self._tdw.display_to_model(dx, dy)

        if starting:
            if self._phase == _Phase.INIT:
                self._sx, self._sy = mpos
            self._px, self._py = mpos

        self._cx, self._cy = mpos

    def _update_ruler_vector(self):
        self._vx, self._vy = self._ruler.identity_vector

    ## Level angle related

    def set_ruler_as_level(self, vec):
        """Set current ruler as level 
        """
        cls = self.__class__
        cls._level_vector = self._ruler.identity_vector
        self.queue_draw_ui(None)

    def is_level_or_cross(self):
        if self._ruler.is_ready():
            vx, vy = self._ruler.identity_vector
            lx, ly = self._level_vector
            margin = self._level_margin
            
            return (self._ruler.is_level(lx, ly, margin) or 
                        self._ruler.is_level(ly, lx, margin))
        return False

    def snap_ruler_to_level(self):
        if self._ruler.is_ready():
            lx, ly = self._level_vector
            margin = self._level_margin

            ans = self._ruler.is_level(lx, ly, margin)

            if ans != 0:
                pass
            else:
                ans = self._ruler.is_level(ly, lx, margin)
                if ans != 0:
                    lx, ly = ly, lx
                else:
                    return

            self.queue_draw_ui(None)

            if ans < 0:
                lx *= -1.0
                ly *= -1.0

            self._ruler.snap(lx, ly)
            self.queue_draw_ui(None)

class ParallelOptionsWidget (freehand_assisted.AssistantOptionsWidget):
    """Configuration widget for freehand mode"""

    def __init__(self, mode):
        super(ParallelOptionsWidget, self).__init__(mode)

    def init_specialized_widgets(self, row):
        self._updating_ui = True
        row = super(ParallelOptionsWidget, self).init_specialized_widgets(row)

        button = Gtk.Button(label = _("Snap to level")) 
        button.connect('clicked', self._snap_level_clicked_cb)
        self.attach(button, 1, row, 1, 1)
        row += 1

        button = Gtk.Button(label = _("current as level")) 
        button.connect('clicked', self._current_level_clicked_cb)
        self.attach(button, 1, row, 1, 1)
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

    def _snap_level_clicked_cb(self, button):
        if not self._updating_ui:
            # To discard current(old) overlay.
            mode = self.mode
            if mode:
                if mode.is_level_or_cross():
                    mode.snap_ruler_to_level()

    def _current_level_clicked_cb(self, button):
        if not self._updating_ui:
            # To discard current(old) overlay.
            mode = self.mode
            if mode:
                mode.set_ruler_as_level()

class _Overlay_Parallel(gui.overlays.Overlay):
    """Overlay for stabilized freehand mode """

    def __init__(self, mode, tdw):
        super(_Overlay_Parallel, self).__init__()
        self._mode_ref = weakref.ref(mode)
        self._tdw_ref = weakref.ref(tdw)

    def paint(self, cr):
        """Draw brush size to the screen"""
        mode = self._mode_ref()
        if mode is not None:
            tdw = self._tdw_ref()
            assert tdw is not None
            ruler = mode._ruler
            if (ruler.is_ready() or 
                    mode._phase in (_Phase.SET_BASE, 
                                    _Phase.SET_DEST)):
                ruler.paint(cr, tdw, mode.is_level_or_cross())

