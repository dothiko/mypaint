# This file is part of MyPaint.
# Copyright (C) 2017 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GLib
import cairo
import math

from lib import helpers
import windowing
import drawutils
import quickchoice
import gui.cursor
import gui.style
from lib.color import HCYColor, RGBColor

"""Brush Size change popup."""


## Module constants
FONT_SIZE = 20
CANVAS_SIZE = 224
RADIUS = CANVAS_SIZE / 2
LINE_WIDTH = 32
TIMEOUT_LEAVE = int(0.7 * 1000) # Auto timeout, when once change size.
MARGIN = LINE_WIDTH / 4

## Class definitions
class _Zone:
    INVALID = 0
    CIRCLE = 1

class SizePopup (windowing.PopupWindow, 
                 windowing.TransparentMixin):
    """Brush Size change popup

    This window is normally popup when hover the cursor
    over a layer in layerlist.
    """

    outside_popup_timeout = 0

    _mask = None
    _active_cursor = None
    _normal_cursor = None

    _bar_color = None
    _base_color = None

    def __init__(self, app, prefs_id=quickchoice._DEFAULT_PREFS_ID):
        super(SizePopup, self).__init__(app)
        # FIXME: This duplicates stuff from the PopupWindow
        self.set_position(Gtk.WindowPosition.MOUSE)
        self.app = app
        self.app.kbm.add_window(self)
        self.set_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                        Gdk.EventMask.BUTTON_RELEASE_MASK |
                        Gdk.EventMask.ENTER_NOTIFY_MASK |
                        Gdk.EventMask.LEAVE_NOTIFY_MASK |
                        Gdk.EventMask.POINTER_MOTION_MASK 
                        )
        self.connect("button-release-event", self.button_release_cb)
        self.connect("button-press-event", self.button_press_cb)
        self.connect("leave-notify-event", self.popup_leave_cb)
        self.connect("enter-notify-event", self.popup_enter_cb)
        self.connect("motion-notify-event", self.motion_cb)

        self.connect("draw", self.draw_cb)

        self.set_size_request(CANVAS_SIZE, CANVAS_SIZE)

        self._brush_normal = None
        self._button = None
        self._close_timer_id = None
        self._zone = _Zone.INVALID
        self._current_cursor = None

        # Transparent window initialize..
        # CAUTION: Place this line here (after call set_size_request)
        # to determine exact dimension of this popup(window).
        self.setup_background()
        
    @property
    def active_cursor(self):
        cls = self.__class__
        if cls._active_cursor is None:
            cursors = self.app.cursors
            cursor = cursors._get_overlay_cursor(
                        None, 
                        gui.cursor.Name.CROSSHAIR_OPEN) 
            cls._active_cursor = cursor
        return cls._active_cursor

    @property
    def normal_cursor(self):
        cls = self.__class__
        if cls._normal_cursor is None:
            cursors = self.app.cursors
            cursor = cursors._get_overlay_cursor(
                        None, 
                        gui.cursor.Name.ARROW
                    ) 
            cls._normal_cursor = cursor
        return cls._normal_cursor

    @property
    def current_cursor(self):
        return self._current_cursor

    @current_cursor.setter
    def current_cursor(self, cursor):
        window = self.get_window()
        window.set_cursor(cursor)
        self._current_cursor = cursor

    @property
    def base_color(self):
        cls = self.__class__
        if cls._base_color is None:
            rgbc = RGBColor(color=gui.style.EDITABLE_ITEM_COLOR)
            cls._base_color = rgbc.get_rgb()
        return cls._base_color

    @property
    def bar_color(self):
        cls = self.__class__
        if cls._bar_color is None:
            rgbc = RGBColor(color=gui.style.ACTIVE_ITEM_COLOR)
            cls._bar_color = rgbc.get_rgb()
        return cls._bar_color

    def display_to_local(self, dx, dy):
        return (dx - RADIUS, dy - RADIUS)

    def enter(self):
        # Called when popup this window.
        x, y = self.get_position()
        self.move(x, y)
        self.show_all()

        adj = self.app.brush_adjustment['radius_logarithmic']
        self._adj = adj
        self._leave_cancel = True

        self._brush_normal = self.get_brushsize_normalized_from_adj()
 
    def leave(self, reason):
        if self._brush_normal is not None:
            # Restore `normalized` brush size into logarithmic.
             self.set_brushsize_normalized_to_adj(self._brush_normal)
        self._brush_normal = None
        self._close_timer_id = None
        self._button = None
        self.hide()

    def button_press_cb(self, widget, event):
        self.update_zone(event.x, event.y)
        self._button = event.button
        if event.button == 1:

            if self._close_timer_id:
                GLib.source_remove(self._close_timer_id)

            if self._zone == _Zone.CIRCLE:
                self._brush_normal = self.get_brushsize_normalized(
                                        event.x, event.y)
            elif self._zone == _Zone.INVALID:
                self._brush_normal = None
                self.leave("aborted")
                    
        self.queue_redraw(widget)

    def button_release_cb(self, widget, event):
        if self._button != None:
            if self._close_timer_id:
                GLib.source_remove(self._close_timer_id)
            self._close_timer_id = GLib.timeout_add(
                    TIMEOUT_LEAVE,
                    self.leave,
                    'timer')

        self._button = None

    def popup_enter_cb(self, widget, event):
        if self._leave_cancel:
            self._leave_cancel = False
            
    def popup_leave_cb(self, widget, event):
        if not self._leave_cancel:
            self.leave('outside')

    def motion_cb(self, widget, event):
        self.update_zone(event.x, event.y)
        if self._button == 1:
            if self._zone == _Zone.CIRCLE:
                self._brush_normal = self.get_brushsize_normalized(
                                        event.x, event.y)
                self.queue_redraw(widget)

    def update_zone(self, x, y):
        x, y = self.display_to_local(x, y)
        zone = _Zone.INVALID
        cursor = self.normal_cursor
        r = RADIUS
        sub_r = r - LINE_WIDTH
        inner_r = (sub_r - LINE_WIDTH / 2) - MARGIN
        outer_r = (sub_r + LINE_WIDTH / 2) + MARGIN

        dist = math.hypot(x , y)
        if dist >= inner_r and dist <= outer_r:
            zone = _Zone.CIRCLE
            cursor = self.active_cursor

        self.current_cursor = cursor
        self._zone = zone

    @property
    def brush_size(self):
        """Get brush size in pixels"""
        adj = self._adj
        min_s = adj.get_lower()
        max_s = adj.get_upper()
        val = self._brush_normal * (max_s - min_s) + min_s
        return math.exp(val)
        
    def get_brushsize_normalized_from_adj(self):
        """ Get normalized brush size from brush_adjustment"""
        adj = self._adj
        min_s = adj.get_lower()
        max_s = adj.get_upper()
        cur = adj.get_value() 
        return (cur - min_s) / (max_s - min_s)

    def set_brushsize_normalized_to_adj(self, value):
        """ Set normalized brush size to brush_adjustment"""
        adj = self._adj
        min_s = adj.get_lower()
        max_s = adj.get_upper()
        adj.set_value((max_s - min_s) * value + min_s)

    def get_brushsize_normalized(self, x, y):
        """Get normalized brush size from local coordinate.
        i.e. get clockwise angle of pressed point.
        :param x: window event.x
        :param y: window event.y
        """
        x, y = self.display_to_local(x, y)
        r = RADIUS
        a = math.acos(-y / math.hypot(x,y))
        rad = math.pi * 2
        if x < 0:
            a = rad - a
        return 1.0 - (a / rad) # Reverse, to increase right direction.

    def draw_cb(self, widget, cr):
       #cr.set_source_rgba(0.9, 0.9, 0.9, 1.0)
       #cr.paint()
        cr.save()

        self.draw_background(cr)

        r = (CANVAS_SIZE / 2)
        sub_r = r - LINE_WIDTH
        angle_offset = -(math.pi / 2)
        brush_angle = (1.0 - self._brush_normal) * math.pi * 2

        cr.translate(r, r)

        # As first, draw label background.
        cr.set_line_width(1)
        cr.set_source_rgba(1.0, 1.0, 1.0, 0.6)
        w = sub_r
        h = FONT_SIZE * 0.7
        cr.rectangle(-w, -h, w*2, h*2) 
        cr.fill()

        # Draw base circle.
        cr.set_line_width(LINE_WIDTH)
        cr.set_source_rgba(*self.base_color)
        cr.arc(0, 0, sub_r, angle_offset, math.pi*2 +angle_offset)
        drawutils.render_drop_shadow(cr)
        cr.stroke()

        # Draw base `empty` circle, with a bit narrower line.
        cr.set_line_width(LINE_WIDTH * 0.5)
        cr.set_source_rgba(0.0, 0.0, 0.0, 0.5)
        cr.arc(0, 0, sub_r, angle_offset, math.pi*2 +angle_offset)
        cr.stroke()

        # Draw colored circle, this shows the size setting.
        cr.set_line_width(LINE_WIDTH * 0.6)
        cr.set_source_rgba(*self.bar_color)
        cr.arc_negative(0, 0, sub_r, angle_offset, brush_angle+angle_offset)
        cr.stroke()

        # Draw current brush size label.
        cr.set_source_rgba(0.0, 0.0, 0.0, 1.0)
        cr.set_font_size(FONT_SIZE)
        txt = "%.3f" % self.brush_size
        x_bear, y_bear, width, height, x_adv, y_adv = cr.text_extents(txt)
        cr.move_to(-width / 2, height / 2)
        cr.show_text(txt)

        cr.restore()
        return True

    def advance(self):
        """Currently,nothing to do."""
        pass

    def backward(self):
        """Currently,nothing to do."""
        pass

