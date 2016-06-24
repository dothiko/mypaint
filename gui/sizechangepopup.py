# This file is part of MyPaint.
# Copyright (C) 2016 by dothiko <dothiko@gmail.com>
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


"""Brush Size change popup."""


## Module constants

MARGIN = 8
MAX_SAMPLES = 5
SIZE_SPACE = 20
CANVAS_SIZE = 80
POPUP_HEIGHT = CANVAS_SIZE + MARGIN * 2
POPUP_WIDTH = CANVAS_SIZE + SIZE_SPACE + (MARGIN * 3)
TIMEOUT_LEAVE = int(0.7 * 1000) # Auto timeout, when once change size.

## Class definitions

class SizePopup (windowing.PopupWindow):
    """Brush Size change popup

    This window is normally popup when hover the cursor
    over a layer in layerlist.
    """

    outside_popup_timeout = 0

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
        self.connect("leave-notify-event", self.leave_cb)
        self.connect("motion-notify-event", self.motion_cb)

        self.connect("draw", self.draw_cb)

        self.set_size_request(POPUP_WIDTH, POPUP_HEIGHT)

        self._user_size = None
        self._button = None
        self._close_timer_id = None

    def popup(self):
        self.enter()


    def enter(self):
        # popup placement
        x, y = self.get_position()
        self.move(x, y)
        self.show_all()

        window = self.get_window()
       #cursor = Gdk.Cursor.new_for_display(
       #    window.get_display(), Gdk.CursorType.CROSSHAIR)
       #window.set_cursor(cursor)

        # Normalize brush size into 0.0 - 1.0
        adj = self.app.brush_adjustment['radius_logarithmic']
        cur_val = adj.get_value() - adj.get_lower()
        range = adj.get_upper() - adj.get_lower()
        self._user_size = cur_val / range

    def leave(self, reason):
        if self._user_size:
            adj = self.app.brush_adjustment['radius_logarithmic']
            cur_val = adj.get_value() - adj.get_lower()
            range = adj.get_upper() - adj.get_lower()
            cur_val = self._user_size * range
            adj.set_value(cur_val + adj.get_lower())
        self._user_size = None
        self._close_timer_id = None
        self._button = None
        self.hide()


    def button_press_cb(self, widget, event):
        self._button = event.button
        if event.button == 1:

            if self._close_timer_id:
                GLib.source_remove(self._close_timer_id)

            if not self._direct_setting_cb(event.x, event.y):
                # Otherwise, if pressed position is 
                # 'predefined shortcut samples', 
                # change size from it.
                y = event.y
                left = MARGIN * 2 + CANVAS_SIZE
                if (left <= event.x <= left + SIZE_SPACE and
                        MARGIN < y < MARGIN + CANVAS_SIZE):
                    segment = float(CANVAS_SIZE) / MAX_SAMPLES
                    step = math.floor((y - MARGIN) / segment) + 1
                    self._user_size = step / float(MAX_SAMPLES)  
                    self._button = -1 
                    # -1 is Dummy value, to indicate invoke timer, 
                    # but not respond motion event.
                else:
                    self._button = None

        self.redraw(widget)

    def button_release_cb(self, widget, event):
        if self._button != None:
            if self._close_timer_id:
                GLib.source_remove(self._close_timer_id)
            self._close_timer_id = GLib.timeout_add(
                    TIMEOUT_LEAVE,
                    self.leave,
                    'timer')
        self._button = None

    def leave_cb(self, widget, event):
        self.leave('outside')
        pass

    def motion_cb(self, widget, event):
        if self._button == 1:
            self._direct_setting_cb(event.x, event.y)
            self.redraw(widget)


    def redraw(self, widget):
        a = widget.get_allocation()
        t = widget.queue_draw_area(a.x, a.y, a.width, a.height)       

    def _direct_setting_cb(self, x, y):
        """ Direct setting callback for pointer action.
        if pointer is off from size circle,this callback
        returns False.
        Otherwise this updates internal size value and returns True.

        :param x: the x coordinate of pointer
        :param y: the y coordinate of pointer
        """ 
        radius = CANVAS_SIZE / 2
        cx = MARGIN + radius
        cy = MARGIN + radius

        length = math.hypot(x - cx, y - cy)
        # If pressed position is inside 'size circle',
        # user change it directly
        if length <= radius:
            self._user_size = length / radius
            return True
        return False

    def draw_cb(self, widget, cr):
        cr.set_source_rgb(0.9, 0.9, 0.9)
        cr.paint()

        # Drawing background of presets
        cr.save()
        rad = min(4, MARGIN / 2)
        drawutils.create_rounded_rectangle_path(cr, 
                MARGIN * 2 + CANVAS_SIZE - rad, MARGIN - rad,
                SIZE_SPACE + rad*2, CANVAS_SIZE + rad*2, 
                rad)
        cr.set_source_rgb(0.4, 0.4, 0.4)
        cr.fill()
        cr.restore()

        cr.save()
        cr.translate(MARGIN + CANVAS_SIZE / 2, MARGIN + CANVAS_SIZE / 2)
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.arc(0, 0, CANVAS_SIZE / 2, 0, math.pi * 2)
        cr.fill()

        if self._user_size:
            cr.set_source_rgb(0, 0, 0)
            cr.arc(0, 0, (self._user_size * CANVAS_SIZE) / 2, 0, math.pi * 2)
            cr.fill()

        cr.restore()

        cr.save()
        cr.set_source_rgb(1.0, 1.0, 1.0)
        center = CANVAS_SIZE + (MARGIN * 2) + (SIZE_SPACE / 2)
        max_size = MAX_SAMPLES
        size_step = 1.0 / max_size
        segment_size = CANVAS_SIZE / float(max_size)
        cr.translate(center, MARGIN + segment_size / 2)
        for i in xrange(max_size):
            cursize = ((i+1) * size_step) * min(SIZE_SPACE, segment_size)
            cr.set_source_rgb(1.0, 1.0, 1.0)
            cr.arc(0, 0, cursize / 2.0, 0, math.pi * 2)
            cr.fill()
            cr.translate(0, segment_size)
        cr.restore()

        return True
