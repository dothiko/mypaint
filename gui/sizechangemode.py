# This file is part of MyPaint.
# Copyright (C) 2016 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# NOTE: This file based on linemode.py ,freehand.py, inktool.py

## Imports
import math
import logging
logger = logging.getLogger(__name__)
import weakref

from gettext import gettext as _
from gi.repository import Gtk, Gdk
from gui.curve import CurveWidget

import gui.mode
import gui.cursor
import gui.overlays
import gui.ui_utils
import gui.linemode
import lib.helpers


## Module constants

## Interaction modes for making lines

class SizechangeMode(gui.mode.ScrollableModeMixin,
                   #gui.mode.BrushworkModeMixin,
                    gui.mode.OneshotDragMode,
                    gui.mode.OverlayMixin):
    """Oncanvas brush Size change mode"""

    ## Class constants
    ACTION_NAME = "OncanvasSizeMode"
    _OPTIONS_WIDGET = None
    _PIXEL_PRECISION = 120.0 # Divider, per pixel precision. practical value.
    _BLANK_CURSOR = Gdk.Cursor(Gdk.CursorType.BLANK_CURSOR)
    
    ## Class configuration.
    permitted_switch_actions = set([
        "PanViewMode",
        "ZoomViewMode",
        "RotateViewMode",
    ])

    pointer_behavior = gui.mode.Behavior.EDIT_OBJECTS
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW

    @property
    def active_cursor(self):
        return self._cursor
    
    @property
    def inactive_cursor(self):
        return self._cursor  
          
    @classmethod
    def get_name(cls):
        return _(u"On-canvas brush size changer")

    def get_usage(self):
        return _(u"Change brush size on canvas.when drag toward up/left,size is decreased.down/right for increasing.vertical movement changes size largely.")

    unmodified_persist = True
    permitted_switch_actions = set(
        ['RotateViewMode', 'ZoomViewMode', 'PanViewMode']
        + gui.mode.BUTTON_BINDING_ACTIONS)

    ## Initialization

    def __init__(self, **kwds):
        """Initialize"""
        super(SizechangeMode, self).__init__(**kwds)
        self.app = None
        self._cursor = None
        
    ## InteractionMode/DragMode implementation

    def enter(self, doc, **kwds):
        """Enter the mode.
        """
        super(SizechangeMode, self).enter(doc, **kwds)
        self.app = self.doc.app
        self.base_x = None
        
        if not self._is_active():
            # This mode is not in modestack - this means 
            # XXX never called?
            self._discard_overlays()

    def leave(self, **kwds):
        if not self._is_active():
            # This mode is in modestack - this means 
            # This is temporary mode.
            self._discard_overlays()

        return super(SizechangeMode, self).leave(**kwds)

    def _is_active(self):
        return self in self.doc.modes

    def get_cursor_radius(self,tdw):
        #FIXME nearly Code duplication from 
        #      gui/tiledrawwidget.py:_get_cursor_info
        b = tdw.doc.brush.brushinfo
        base_radius = math.exp(b.get_base_value('radius_logarithmic'))
        r = base_radius
        r *= tdw.scale
        return r
        
    def button_press_cb(self, tdw, event):
        # getting returning point of cursor,in screen coordinate
        if self.base_x is None:
            self._cursor = self._BLANK_CURSOR
            self._initial_radius = self.get_cursor_radius(tdw)
            self.base_x = event.x
            self.base_y = event.y
        
        return super(SizechangeMode, self).button_press_cb(tdw, event)

    def drag_start_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        self._queue_draw_brush(self.base_x, self.base_y) 

        # Storing original cursor position(in screen coodinate)
        # and some needed objects.
        dev = event.get_device()
        self._dev = dev
        pos = dev.get_position()
        self._scr = event.get_screen()
        self._ox = pos.x
        self._oy = pos.y
        super(SizechangeMode, self).drag_start_cb(tdw, event)

    def drag_update_cb(self, tdw, event, dx, dy):
        self._ensure_overlay_for_tdw(tdw)

        self._queue_draw_brush(self.base_x, self.base_y) # To erase old circle
        adj = self.app.brush_adjustment['radius_logarithmic']
        diff_val = (dx / self._PIXEL_PRECISION)
        # Faster change along with y direction move.
        diff_val += ((dy / self._PIXEL_PRECISION) * 2)
        adj.set_value(adj.get_value() + diff_val)

        self._queue_draw_brush(self.base_x, self.base_y)
        super(SizechangeMode, self).drag_update_cb(tdw, event, dx, dy)

    def drag_stop_cb(self, tdw):
        self._ensure_overlay_for_tdw(tdw)
        self._queue_draw_brush(self.base_x, self.base_y)
        self.start_drag = False
         
        self.base_x = None
        self.base_y = None

        # Reset cursor positon at start position of dragging.
        Gdk.Device.warp(self._dev, self._scr, self._ox, self._oy)
        
        # Restore the cursor. 
        # (But, DragMode baseclass automatically restore it?)
        self._cursor = None

        # Reset mode as default
        super(SizechangeMode, self).drag_stop_cb(tdw)

    ## Overlays

    def _generate_overlay(self, tdw):
        return _Overlay(self, tdw)
    
    def _queue_draw_brush(self, x, y):
        for tdw, overlay in self._overlays.items():
            cur_radius = self.get_cursor_radius(tdw)
            gui.ui_utils.queue_circular_area(
                tdw,
                x, y,
                cur_radius
            )

    def _draw_dashed(self, cr):        
        """Drawing (preserved) stroke as dashed one."""
        cr.save()
        cr.set_source_rgb(1, 1, 1)
        cr.set_dash( (8,) )
        cr.stroke()
        cr.restore()
        
    def draw_ui(self, tdw, cr):
        """Called from Overlay """
        cr.set_source_rgb(0, 0, 0)
        cr.set_line_width(1)            
        cr.arc( self.base_x,
                self.base_y,
                self.get_cursor_radius(tdw),
                0.0,
                2*math.pi)
        cr.stroke_preserve()
        self._draw_dashed(cr)

    ## Mode options
    
    def _generate_options_widget(self):
        """Called from get_options_widget. for code-sharing. """
        return _SizeChangerOptionWidget()

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        cls = self.__class__
        if cls._OPTIONS_WIDGET is None:
            widget = self._generate_options_widget()
            cls._OPTIONS_WIDGET = widget
        return cls._OPTIONS_WIDGET

class TiltchangeMode(SizechangeMode):
    """Oncanvas tilt-offset change mode"""

    ACTION_NAME = "OncanvasTiltMode"   
    _TILT_RADIUS = 64.0   
    _OPTIONS_WIDGET = None # This class needs same name attribute.
    
    @classmethod
    def get_name(cls):
        return _(u"On-canvas tilt offset changer")

    def get_usage(self):
        return _(u"Change tilt offset on canvas. This emulates tilt-able pen stlus.")
            
    def __init__(self, **kwds):
        """Initialize"""
        super(TiltchangeMode, self).__init__(**kwds)
        
    @property
    def tilt_x_value(self):
        if self.app is not None:
            adj = self.app.brush_adjustment['tilt_offset_x']
            return adj.get_value()
        return 0.0
        
    @property
    def tilt_y_value(self):
        if self.app is not None:
            adj = self.app.brush_adjustment['tilt_offset_y']
            return adj.get_value()
        return 0.0
                    
    @tilt_x_value.setter
    def tilt_x_value(self, val):
        if self.app is not None:
            adj = self.app.brush_adjustment['tilt_offset_x']
            adj.set_value(val)
        
    @tilt_y_value.setter
    def tilt_y_value(self, val):
        if self.app is not None:
            adj = self.app.brush_adjustment['tilt_offset_y']
            adj.set_value(val)
            
    def _queue_draw_brush(self, x, y):
        space = 2 # I'm unsure why this number brought good result.
                  # might be line width * 2?
        margin = 3
        for tdw, overlay in self._overlays.items():
            if self.base_x != None:
                t_rad = self._TILT_RADIUS + margin
                tdw.queue_draw_area(
                        x - t_rad,
                        y - t_rad,
                        t_rad * 2,
                        t_rad * 2)
                        
    def drag_update_cb(self, tdw, event, dx, dy):
        self._ensure_overlay_for_tdw(tdw)
 
        self._queue_draw_brush(self.base_x, self.base_y)

        p = self._PIXEL_PRECISION
        cur_value = self.tilt_x_value + (dx / p)
        self.tilt_x_value = cur_value
        cur_value = self.tilt_y_value + (dy / p)
        self.tilt_y_value = cur_value
        self._queue_draw_brush(self.base_x, self.base_y)
        
        # Call parent of `SizechangeMode`(parent class) method.
        # Not parent class method, we need to bypass that callback.
        super(SizechangeMode, self).drag_update_cb(tdw, event, dx, dy)
        
    def _generate_options_widget(self):
        """Called from get_options_widget. for code-sharing. """
        return _TiltOffsetOptionWidget()
    
    def draw_ui(self, tdw, cr):
        cr.set_source_rgb(0, 0, 0)
        cr.set_line_width(1)
        cr.translate(self.base_x, self.base_y)
        r = self._TILT_RADIUS
        
        # Draw base rectangle
        #cr.rectangle(  -rad, -rad, rad*2, rad*2)
        cr.arc( 
            0,
            0,
            r,
            0.0,
            2*math.pi
        )
        cr.stroke_preserve()
        self._draw_dashed(cr)
        
        # Drawing tilt crosshair
        cr.move_to(0, -r)
        cr.line_to(0, r)
        cr.stroke_preserve()
        self._draw_dashed(cr)
        cr.move_to(-r, 0)
        cr.line_to(r, 0)
        cr.stroke_preserve()
        self._draw_dashed(cr)
        
        # Drawing tilt indicator
        cr.save()
        cr.set_line_width(2)
        cr.move_to(0, 0)
        x = self.tilt_x_value
        y = self.tilt_y_value
        # Remap input values into circle
        nx, ny = gui.linemode.normal(0, 0, x, y)
        s = gui.linemode.get_radian(1.0, 0, nx, ny)
        cr.line_to(
            x * abs(math.cos(s)) * r, # `abs` is to show inverted axis value.
            y * math.sin(s) * r
        )
        cr.stroke_preserve()
        self._draw_dashed(cr)
        cr.restore()


class _Overlay (gui.overlays.Overlay):
    """Overlay for an SizechangeMode's brushsize"""

    def __init__(self, sizemode, tdw):
        super(_Overlay, self).__init__()
        self._sizemode = weakref.ref(sizemode)
        self._tdw = weakref.ref(tdw)
               
    def paint(self, cr):
        """Draw brush size to the screen"""

        cr.save()
       #color = gui.style.ACTIVE_ITEM_COLOR
        mode = self._sizemode()
        tdw = self._tdw()

        if mode is not None and tdw is not None and mode.base_x is not None:
            cr.save()
            mode.draw_ui(tdw, cr)
            cr.restore()
        
        cr.restore()
                
class _SizeChangerOptionWidget(gui.mode.PaintingModeOptionsWidgetBase):
    """ Because OncanvasSizeMode use from dragging + modifier
    combination, thus this option widget mostly unoperatable.
    but I think user would feel some 'reliability' when there are 
    value displaying scale and label.
    """

    def __init__(self):
        # Overwrite self._COMMON_SETTINGS
        # to use(show) only 'radius_logarithmic' scale.
        for cname, text in self._COMMON_SETTINGS:
            if cname == 'radius_logarithmic':
                self._COMMON_SETTINGS = [ (cname, text) ]
                break
        
        # And then,call superclass method
        super(_SizeChangerOptionWidget, self).__init__()

    def init_reset_widgets(self, row):
        """To cancel creating 'reset setting' button"""
        pass
        
class _TiltOffsetOptionWidget(gui.mode.PaintingModeOptionsWidgetBase):
    def __init__(self):
        # Overwrite self._COMMON_SETTINGS
        # to use(show) only 'radius_logarithmic' scale.
        self._COMMON_SETTINGS = (
            ('tilt_offset_x', _("Tilt X:")),
            ('tilt_offset_y', _("Tilt Y:")),
        )   
        # And then,call superclass method
        super(_TiltOffsetOptionWidget, self).__init__()

    def init_reset_widgets(self, row):
        """To cancel creating 'reset setting' button"""
        pass


