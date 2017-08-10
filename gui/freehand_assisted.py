# This file is part of MyPaint.
# Copyright (C) 2017 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""Assisted Freehand drawing base class"""

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
import freehand

## Module settings

## Class defs
class AssistedFreehandMode (freehand.FreehandMode,
                            gui.mode.OverlayMixin):
    """Assisted Freehand drawing mode

    This is a base class for assisted freehand mode.
    which is for assisting drawing with stabilier,
    parallel ruler, etc.
    """

    ## Class constants & instance defaults

    permitted_switch_actions = set()   # Any action is permitted

    _OPTIONS_WIDGET = None

    
    _X_TILT_OFFSET = 0.0    # XXX Class global tilt offsets, to
    _Y_TILT_OFFSET = 0.0    # enable change tilt parameters for
                            # non-tilt-sensible pen stylus.

    _app = None

    # The modifier key, used to activate assistant.
    # This might be alt(MOD1_MASK) or Windows-key(SUPER_MASK)
    # Mac command key(META_MASK).
    # Shift and ctrl are used in mypaint already and frequently, 
    # so should not used for assistant.
    ASSITANT_MODIFIER = Gdk.ModifierType.MOD1_MASK 

    ## Initialization

    def __init__(self, ignore_modifiers=True, **args):
        # Ignore the additional arg that flip actions feed us
        super(AssistedFreehandMode, self).__init__(**args)

        self.do_assist = True
        self._override_assist = False
        self._last_button = None
        self._prev_button = None
        self.reset_assist()

    ## Properties
    
    @property
    def app(self):
        cls = self.__class__
        if cls._app is None:
            from application import get_app
            cls._app = get_app() 
        return cls._app

    @property
    def last_button(self):
        return self._last_button

    @last_button.setter
    def last_button(self, button):
        self._prev_button = self._last_button
        self._last_button = button

    @property
    def prev_button(self):
        return self._prev_button

    ## Override (temporally enable) assitant feature. 
    @property
    def overrided(self):
        """This property intended to mistakenly overwrite the flag.
        """
        return self._override_assist

    ## Mode stack & current mode

    def enter(self, doc, **kwds):
        """Enter freehand mode"""
        super(AssistedFreehandMode, self).enter(doc, **kwds)
        self.get_options_widget() # This sets current mode as target
                                  # of options widget.
               
    def leave(self, **kwds):
        """Leave freehand mode"""
        self.queue_draw_ui(None)
        super(AssistedFreehandMode, self).leave(**kwds)
        self._discard_overlays()

    ## Input handlers

    def button_press_cb(self, tdw, event):
        # If alt key pressed, overriding and enable stabilizer
        # immidiately.
        current_layer = tdw.doc.layer_stack.current
        if (current_layer.get_paintable() and event.button == 1):
            #         and event.type == Gdk.EventType.BUTTON_PRESS):
            # event.type check removed for modifier.

            if event.state & self.ASSITANT_MODIFIER:
                # Override Assistant feature by ALT key modifier.
                self._override_assist = self.do_assist
                self.do_assist = True
            else:
                self._override_assist = None

            if self.do_assist:
                self._ensure_overlay_for_tdw(tdw)
                self.last_button = event.button
                if not self.drag_start_cb(tdw, event, 
                                          event.get_axis(Gdk.AxisUse.PRESSURE)):
                    self.queue_draw_ui(tdw) 
                else:
                    self.last_button = None

        return super(AssistedFreehandMode, self).button_press_cb(
                tdw, event)


    def button_release_cb(self, tdw, event):
        current_layer = tdw.doc.layer_stack.current
        if current_layer.get_paintable() and event.button == 1:

            if self.do_assist:
                self.queue_draw_ui(tdw) # To erase. call this first.
                if self.last_button is not None:
                    self.drag_stop_cb(tdw, event)

            if self._override_assist is not None:
                self.do_assist = self._override_assist
                self._override_assist = None

            self.last_button = None
        return super(AssistedFreehandMode, self).button_release_cb(
                tdw, event)

    def motion_notify_cb(self, tdw, event, fakepressure=None):
        """Motion event handler: queues raw input and returns

        :param tdw: The TiledDrawWidget receiving the event
        :param event: the MotionNotify event being handled
        :param fakepressure: fake pressure to use if no real pressure

        Fake pressure is passed with faked motion events, e.g.
        button-press and button-release handlers for mouse events.

        """

        # FIXME This Method almost copied from freehand.py.
        # so, when freehand.py is updated, this file should be 
        # updated in same way.

        # Do nothing if painting is inactivated
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False

        # If the device has changed and the last pressure value from the
        # previous device is not equal to 0.0, this can leave a visible
        # stroke on the layer even if the 'new' device is not pressed on
        # the tablet and has a pressure axis == 0.0.  Reseting the brush
        # when the device changes fixes this issue, but there may be a
        # much more elegant solution that only resets the brush on this
        # edge-case.
        same_device = True
        if tdw.app is not None:
            device = event.get_source_device()
            same_device = tdw.app.device_monitor.device_used(device)
            if not same_device:
                tdw.doc.brush.reset()

        # Extract the raw readings for this event
        x = event.x
        y = event.y

        time = event.time
        pressure = event.get_axis(Gdk.AxisUse.PRESSURE)
        xtilt = event.get_axis(Gdk.AxisUse.XTILT)
        ytilt = event.get_axis(Gdk.AxisUse.YTILT)
        state = event.state

        # Workaround for buggy evdev behaviour.
        # Events sometimes get a zero raw pressure reading when the
        # pressure reading has not changed. This results in broken
        # lines. As a workaround, forbid zero pressures if there is a
        # button pressed down, and substitute the last-known good value.
        # Detail: https://github.com/mypaint/mypaint/issues/29
        drawstate = self._get_drawing_state(tdw)
        if drawstate.button_down is not None:
            if pressure == 0.0:
                pressure = drawstate.last_good_raw_pressure
            elif pressure is not None and np.isfinite(pressure):
                drawstate.last_good_raw_pressure = pressure

        # Ensure each event has a defined pressure
        if pressure is not None:
            # Using the reported pressure. Apply some sanity checks
            if not np.isfinite(pressure):
                # infinity/nan: use button state (instead of clamping in
                # brush.hpp) https://gna.org/bugs/?14709
                pressure = None
            else:
                pressure = clamp(pressure, 0.0, 1.0)
            drawstate.last_event_had_pressure = True

        # Fake the pressure if we have none, or if infinity was reported
        if pressure is None:
            if fakepressure is not None:
                pressure = clamp(fakepressure, 0.0, 1.0)
            else:
                pressure = (
                    (state & Gdk.ModifierType.BUTTON1_MASK) and 0.5 or 0.0)
            drawstate.last_event_had_pressure = False

        # Check whether tilt is present.  For some tablets without
        # tilt support GTK reports a tilt axis with value nan, instead
        # of None.  https://gna.org/bugs/?17084
        if xtilt is None or ytilt is None or not np.isfinite(xtilt + ytilt):
            xtilt = 0.0
            ytilt = 0.0

        # Switching from a non-tilt device to a device which reports
        # tilt can cause GDK to return out-of-range tilt values, on X11.
        xtilt = clamp(xtilt, -1.0, 1.0)
        ytilt = clamp(ytilt, -1.0, 1.0)

        # Evdev workaround. X and Y tilts suffer from the same
        # problem as pressure for fancier devices.
        if drawstate.button_down is not None:
            if xtilt == 0.0:
                xtilt = drawstate.last_good_raw_xtilt
            else:
                drawstate.last_good_raw_xtilt = xtilt
            if ytilt == 0.0:
                ytilt = drawstate.last_good_raw_ytilt
            else:
                drawstate.last_good_raw_ytilt = ytilt

        # Tilt inputs are assumed to be relative to the viewport,
        # but the canvas may be rotated or mirrored, or both.
        # Compensate before passing them to the brush engine.
        # https://gna.org/bugs/?19988
        if tdw.mirrored:
            xtilt *= -1.0
        if tdw.rotation != 0:
            tilt_angle = math.atan2(ytilt, xtilt) - tdw.rotation
            tilt_magnitude = math.sqrt((xtilt**2) + (ytilt**2))
            xtilt = tilt_magnitude * math.cos(tilt_angle)
            ytilt = tilt_magnitude * math.sin(tilt_angle)

        # HACK: color picking, do not paint
        # TEST: Does this ever happen now?
        # XXX Modified codes.
        # Original code disables MOD1_MASK(alt key) here.
        # But for stabilizer, Alt key used "Temporary enable stabilizer"
        if (state & Gdk.ModifierType.CONTROL_MASK):
            # Don't simply return; this is a workaround for unwanted
            # lines in https://gna.org/bugs/?16169
            pressure = 0.0

        # Apply pressure mapping if we're running as part of a full
        # MyPaint application (and if there's one defined).
        if tdw.app is not None and tdw.app.pressure_mapping:
            pressure = tdw.app.pressure_mapping(pressure)

        # Apply any configured while-drawing cursor
        if pressure > 0:
            self._hide_drawing_cursor(tdw)
        else:
            self._reinstate_drawing_cursor(tdw)


        # HACK: straight line mode?
        # TEST: Does this ever happen?
        if state & Gdk.ModifierType.SHIFT_MASK:
            pressure = 0.0

        # XXX Added codes for assistant feature.
        # Assisted freehands cannot draw zero pressure
        # stroke all the time.
        # Because they (would) draw a different place
        # from the actual stylus position.
        # Thus, placing pressure 0 stroke at current place would 
        # result as heading / trailing glitches.
        if self.do_assist and self.last_button==1:

            self.queue_draw_ui(tdw)

            # drag_update_cb cancels this event when
            # it return True. not False or None. 
            if self.drag_update_cb(tdw, event, pressure):
                # Call super-superclass method.
                # Caution, not superclass method.because superclass might
                # draw a stroke!
                return super(freehand.FreehandMode, self).motion_notify_cb(
                        tdw, event)

            # Assitant event position fetch and queue motion
            self.fetch(x, y, pressure, time)
            for x, y, p in self.enum_samples():
                x, y = tdw.display_to_model(x, y)
                event_data = (time, x, y, p, xtilt, ytilt)
                drawstate.queue_motion(event_data)

            # New positioned assistant overlay should be drawn here.
            if fakepressure is None:
                self.queue_draw_ui(tdw)
        else:
            # Ordinary event queuing

            # Queue this event
            x, y = tdw.display_to_model(x, y)
            event_data = (time, x, y, pressure, xtilt, ytilt)
            drawstate.queue_motion(event_data)

        # Start the motion event processor, if it isn't already running
        if not drawstate.motion_processing_cbid:
            cbid = GLib.idle_add(
                self._motion_queue_idle_cb,
                tdw,
                priority = self.MOTION_QUEUE_PRIORITY,
            )
            drawstate.motion_processing_cbid = cbid

    ## Mode options

    def get_options_widget(self):
        """Get the (class singleton) options widget
        This is called on-demand.
        """
        cls = self.__class__
        if cls._OPTIONS_WIDGET is None:
            widget = StabilizerOptionsWidget(self)
            cls._OPTIONS_WIDGET = widget
        else:
            cls._OPTIONS_WIDGET.set_mode(self)
        return cls._OPTIONS_WIDGET

                
    ## Signal Handlers

    def drag_start_cb(self, tdw, event, pressure):
        """Dragging start point for assisted freehand drawing.
        When drag is started(from button_press_cb), this is called.
        :return : boolean flag or None. 
                  True to CANCEL assistant feature.
        """
        return False

    def drag_update_cb(self, tdw, event, pressure):
        """Motion notify callback for assisted freehand drawing

        CAUTION: This handler is not for assistant feature itself.
        This is mainly for controlling overlay GUI.

        :return : boolean flag or None. 
                  True to CANCEL entire freehand motion handler 
                  and no strokes are drawn.
        """
        return False

    def drag_stop_cb(self, tdw, event):
        """Dragging end point for assisted freehand drawing.
        When drag is end(at button_release_cb), this is called.

        :return : nothing
        """
        pass

    ## Assistant methods
    def reset_assist(self):
        """ Reset assist attributes.
        This might be called from constructor.
        """
        pass
    
    def fetch(self, x, y, pressure, time):
        """Fetch samples(i.e. current stylus input datas) into instance. 
        That samples are used for generating new modified point datas 
        in enum_samples."""
        pass

    def enum_samples(self):
        """Iterate a tuple of (x, y, pressure) to pass modified datas
        into stroke engine, when assitant is enabled.

        Iteration is done by yield statement.

        x and y are display coordinate.
        """
        raise StopIteration

    ## Overlay related

    def _generate_overlay(self, tdw):
        """Generate overlay class instance for tdw here.
        This is called on-demand.
        """
        pass

    def queue_draw_ui(self, tdw):
        """ Queue draw area for overlay """
        pass

    ## Utility method
    def queue_motion(self, tdw, time, x, y, 
                     pressure=0.0, xtilt=0.0, ytilt=0.0):
        """
        :param x,y: stroke position, in MODEL coodinate.
        """
        drawstate = self._get_drawing_state(tdw)
        event_data = (time, x, y, pressure, xtilt, ytilt)
        drawstate.queue_motion(event_data)

        if not drawstate.motion_processing_cbid:
            cbid = GLib.idle_add(
                self._motion_queue_idle_cb,
                tdw,
                priority = self.MOTION_QUEUE_PRIORITY,
            )
            drawstate.motion_processing_cbid = cbid

class AssistantOptionsWidget (freehand.FreehandOptionsWidget):
    """Configuration widget for freehand mode"""

    def __init__(self, mode):
        self.set_mode(mode) # call this first, because
                            # __init__ calls init_specialized_widgets
                            # and it refers self.mode_ref attr.
        super(AssistantOptionsWidget, self).__init__()

    def set_mode(self, mode):
        self.mode_ref = weakref.ref(mode)

    @property
    def mode(self):
        mode = self.mode_ref()
        assert mode is not None
        return mode

   #def init_specialized_widgets(self, row):
   #    self._updating_ui = True
   #    row = super(StabilizerOptionsWidget, self).init_specialized_widgets(row)
   #
   #    def create_slider(label, handler, 
   #                      value, min_adj, max_adj, step_incr=1,
   #                      digits=1 ):
   #        labelobj = Gtk.Label(halign=Gtk.Align.START)
   #        labelobj.set_text(label)
   #       #self._attach_grid(labelobj, col=0, width=1)
   #        self.attach(labelobj, 0, row, 1, 1)
   #
   #        adj = Gtk.Adjustment(value, min_adj, max_adj, 
   #                             step_incr=step_incr)
   #        adj.connect('value-changed', handler)
   #
   #        scale = Gtk.HScale(hexpand_set=True, hexpand=True, 
   #                halign=Gtk.Align.FILL, adjustment=adj, digits=digits)
   #        scale.set_value_pos(Gtk.PositionType.RIGHT)
   #       #self._attach_grid(scale, col=1, width=1)
   #        self.attach(scale, 1, row, 1, 1)
   #        return scale
   #
   #    create_slider(_("X Tilt Offset:"), 
   #                  self.x_tilt_offset_adj_changed_cb,
   #                  0.0, -1.0, 1.0, 0.01
   #                  )
   #    row += 1
   #
   #    self._updating_ui = False
   #    return row

