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
FONT_SIZE = 10
SIZE_SPACE = 20
CANVAS_SIZE = 200
ZOOM_SPACE = 28
POPUP_HEIGHT = CANVAS_SIZE + MARGIN * 2 + ZOOM_SPACE
POPUP_WIDTH = CANVAS_SIZE + SIZE_SPACE + (MARGIN * 3)
TIMEOUT_LEAVE = int(0.7 * 1000) # Auto timeout, when once change size.
RELATIVE_THRESHOLD = 10
MAX_RANGE_MIN = 80

INDICATOR_HEIGHT = 22
INDICATOR_WIDTH = CANVAS_SIZE - INDICATOR_HEIGHT 

PRESETS = (1.0, 3.0, 6.0, 10.0, 32.0, 50.0)

class _Zone:
    INVALID = -1
    CANVAS = 0
    PRESET = 1
    INDICATOR = 2

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
        self.connect("leave-notify-event", self.popup_leave_cb)
        self.connect("enter-notify-event", self.popup_enter_cb)
        self.connect("motion-notify-event", self.motion_cb)

        self.connect("draw", self.draw_cb)

        self.set_size_request(POPUP_WIDTH, POPUP_HEIGHT)

        self._brush_size = None
        self._button = None
        self._close_timer_id = None
        self._doubleclicked = False
        self._zone = _Zone.INVALID
        self._original_size = None
        
    @property
    def canvas_center(self):
        return MARGIN + (CANVAS_SIZE / 2)
        
    @property
    def max_range(self):
        return self._max_range
        
    @max_range.setter
    def max_range(self, value):
        self._max_range = min(max(MAX_RANGE_MIN, value), self.max_radius)
        self._canvas_ratio = CANVAS_SIZE / float(self._max_range)
        
    @property
    def max_range_level(self):
        return (self._max_range - MAX_RANGE_MIN) / (self.max_radius - MAX_RANGE_MIN)
        
    @property
    def max_radius(self):
        return math.exp(self._adj.get_upper())

    def get_normalized_brush_size(self, pixel_size):
        brush_range = self._adj.get_upper() - self._adj.get_lower()
        return (math.log(pixel_size) - self._adj.get_lower()) / brush_range
            
    def get_zone(self, x, y):
        """ Get zone information, to know where user clicked.
        """
        canvas_bottom = MARGIN + CANVAS_SIZE
        
        if (MARGIN < x < MARGIN + CANVAS_SIZE and
                MARGIN < y < canvas_bottom):
            return _Zone.CANVAS
        
        left = MARGIN * 2 + CANVAS_SIZE
        if (left <= x <= left + SIZE_SPACE and
                MARGIN < y < MARGIN + CANVAS_SIZE):
            return _Zone.PRESET
            
        canvas_bottom += MARGIN    
        if (MARGIN < x < MARGIN + INDICATOR_WIDTH and
                canvas_bottom < y < canvas_bottom + INDICATOR_HEIGHT):
            return _Zone.INDICATOR            
            
        return _Zone.INVALID
        
    def popup(self):
        self.enter()

    def enter(self):
        # Popup this window, in current position.
        x, y = self.get_position()
        self.move(x, y)
        self.show_all()

        window = self.get_window()
        cursor = Gdk.Cursor.new_for_display(
            window.get_display(), Gdk.CursorType.CROSSHAIR)
        window.set_cursor(cursor)

        adj = self.app.brush_adjustment['radius_logarithmic']
        self._adj = adj
        self._brush_size = math.exp(adj.get_value()) 
        self._initial_size = self._brush_size
        self.max_range = MAX_RANGE_MIN
        self._leave_cancel = True

    def leave(self, reason):
        if self._button == None:
            if self._brush_size:
                self._adj.set_value(math.log(self._brush_size))
            self._brush_size = None
            self._close_timer_id = None
            self._button = None
            self.hide()

    def button_press_cb(self, widget, event):
        self._button = event.button
        if event.button == 1:

            if self._close_timer_id:
                GLib.source_remove(self._close_timer_id)
                
            x, y = event.x, event.y
            self._zone = self.get_zone(x, y)
            
            if self._zone == _Zone.CANVAS:                                    
                if event.type == Gdk.EventType.BUTTON_PRESS:
                    self.base_x = x
                    self.base_y = y   
                    # Save pre-click brush size,
                    # for double click relative mode  
                    self._prev_size = self._original_size           
                    self._original_size = self._brush_size
                                   
                    self._direct_set_radius(x, y)

            elif self._zone == _Zone.PRESET:
                segment = float(CANVAS_SIZE) / len(PRESETS)
                idx = int(math.floor((y - MARGIN) / segment))
                # NOTE: Preset is DIAMETER, not radius.
                self.brush_size = PRESETS[idx] / 2.0
            elif self._zone == _Zone.INDICATOR:
                pass
            else:
                self._button = None
                
            if event.type == Gdk.EventType._2BUTTON_PRESS:
                self._doubleclicked = True
                # Reset brush size as efore doubleclick relative mode
                assert self._prev_size != None
                self._brush_size = self._prev_size
                    
        self.queue_redraw(widget)

    def button_release_cb(self, widget, event):
        if self._button != None:
            if self._zone == _Zone.CANVAS:

                if self._close_timer_id:
                    GLib.source_remove(self._close_timer_id)
                self._close_timer_id = GLib.timeout_add(
                        TIMEOUT_LEAVE,
                        self.leave,
                        'timer')
                        
                if self._doubleclicked:
                    if ( event.x < 0 or event.y < 0 or
                        event.x >= POPUP_WIDTH or event.y >= POPUP_HEIGHT):
                        self._leave_cancel = True
                    self._doubleclicked = False
                    self.queue_redraw(widget)
                    self._original_size = None

        self._button = None

    def popup_enter_cb(self, widget, event):
        if self._leave_cancel:
            self._leave_cancel = False
            
    def popup_leave_cb(self, widget, event):
        if not self._leave_cancel:
            self.leave('outside')

    def motion_cb(self, widget, event):
        if self._button == 1:
            if self._zone == _Zone.CANVAS:
                if self._doubleclicked:
                    self._relative_set_radius(event.x - self.base_x, 
                        event.y - self.base_y)
                else:
                    self._direct_set_radius(event.x, event.y)

                self.queue_redraw(widget)
                self.base_x = event.x
                self.base_y = event.y
            elif self._zone == _Zone.INDICATOR:
                pos = (event.x - MARGIN) / float(INDICATOR_WIDTH)
                max_indicator = self.max_radius - MAX_RANGE_MIN
                self.max_range = MAX_RANGE_MIN + (pos * max_indicator)
                self.queue_redraw(widget)


    def queue_redraw(self, widget):
        a = widget.get_allocation()
        t = widget.queue_draw_area(a.x, a.y, a.width, a.height)       
        
    @property
    def brush_size(self):
        return self._brush_size
        
    @brush_size.setter
    def brush_size(self, new_size):
        new_size = max (math.exp(self._adj.get_lower()), new_size)
        new_size = min (math.exp(self._adj.get_upper()), new_size)
        self._brush_size = new_size

    def _relative_set_radius(self, x, y):
        """ Set brush radius relatively.
        if pointer is off from size circle,this callback
        returns False.
        Otherwise this updates internal size value and returns True.

        :param x: the raw x difference of pointer, in pixel.
        :param y: the raw y difference of pointer, in pixel.
        """ 
        half_canvas = CANVAS_SIZE / 2
        self.brush_size = self.brush_size + \
            x * 0.1 * math.exp(self._brush_size / 100.0)      

                
    def _direct_set_radius(self, x, y):
        """ Set brush radius relatively.
        if pointer is off from size circle,this callback
        returns False.
        Otherwise this updates internal size value and returns True.

        :param x: the raw x coordinate of pointer, in pixel.
        :param y: the raw y coordinate of pointer, in pixel.
        """ 
        center = MARGIN + (CANVAS_SIZE / 2)
        x -= center
        y -= center
        
        length = math.hypot(x, y)
        self.brush_size = length / self._canvas_ratio

    def draw_cb(self, widget, cr):
        cr.set_source_rgb(0.9, 0.9, 0.9)
        cr.paint()
        cr.set_font_size(12)

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
        # Draw zoom indicator
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.translate(MARGIN, 
            CANVAS_SIZE + (ZOOM_SPACE - INDICATOR_HEIGHT / 2))
            
        drawutils.create_rounded_rectangle_path(cr, 
                0, 0, INDICATOR_WIDTH, INDICATOR_HEIGHT,
                rad) 
        cr.clip_preserve()
        cr.fill()
        
        cr.set_source_rgb(1.0, 0.5, 0.0)
        cr.rectangle(0, 0,
            self.max_range_level * INDICATOR_WIDTH, INDICATOR_HEIGHT)
        cr.fill()
        
        cr.set_source_rgb(0.0, 0.0, 0.0)
        cr.move_to(MARGIN, 
            (INDICATOR_HEIGHT / 2) + (FONT_SIZE / 2))
        cr.show_text("%.2f / max %.2f" % 
            (self._brush_size * 2.0, # Shows diameter.
             self._max_range))
        cr.restore()

        # Draw base canvas circle
        cr.save()
        cr.translate(MARGIN + CANVAS_SIZE / 2, MARGIN + CANVAS_SIZE / 2)
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.arc(0, 0, CANVAS_SIZE / 2, 0, math.pi * 2)
        
        cr.clip_preserve() # Important: clip following draw calls.
        cr.fill()

        assert self._brush_size != None
        # Drawing current brush size
        if self._doubleclicked:
            cr.set_source_rgb(1.0, 0, 0)
        else:
            cr.set_source_rgb(0, 0, 0)
        cr.arc(0, 0, self._brush_size * self._canvas_ratio,
            0, math.pi * 2)
        cr.fill()

        # Drawing initial brush size
        cr.set_source_rgb(0, 0.3, 0.7)
        cr.arc(0, 0, self._initial_size * self._canvas_ratio,
            0, math.pi * 2)
        cr.stroke()

        # Drawing gauge
        cr.set_source_rgb(0.6, 0.6, 0.6)
        cr.set_line_width(1)
        for i in (1, 10, 100, 250):
            rad = i * self._canvas_ratio
            cr.arc(0, 0, rad * 0.5, 0, math.pi * 2)
            cr.stroke()
        cr.restore()

        # Draw presets
        cr.save()
        cr.set_source_rgb(1.0, 1.0, 1.0)
        center = CANVAS_SIZE + (MARGIN * 2) + (SIZE_SPACE / 2)
        cnt = float(len(PRESETS))
        segment_size = CANVAS_SIZE / cnt
        max_radius = min(SIZE_SPACE, segment_size)
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.set_line_width(1.0)
        cr.translate(CANVAS_SIZE + (MARGIN * 2), MARGIN)
        for i, size in enumerate(PRESETS):
            cr.move_to(0, (segment_size / 2) + (FONT_SIZE / 2))
            cr.show_text("%d" % size)
            cr.move_to(0, segment_size - 1)
            cr.line_to(0 + SIZE_SPACE, segment_size - 1)
            cr.translate(0, segment_size)
        cr.restore()

        return True

    def advance(self):
        """Currently,nothing to do."""
        pass

    def backward(self):
        """Currently,nothing to do."""
        pass

