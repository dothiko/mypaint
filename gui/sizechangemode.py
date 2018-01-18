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
from curve import CurveWidget

import gui.mode
import gui.cursor
import gui.overlays
import gui.ui_utils
import gui.linemode


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
        return self._cursor  # Completely blank cursor
    
    @classmethod
    def get_name(cls):
        return _(u"On-canvas brush size changer")

    def get_usage(self):
        return _(u"Change brush size on canvas.when drag toward up/left,size is decreased.down/right for increasing.vertical movement changes size largely.")

    @property
    def inactive_cursor(self):
        return self._cursor  # Completely blank cursor


    unmodified_persist = True
    permitted_switch_actions = set(
        ['RotateViewMode', 'ZoomViewMode', 'PanViewMode']
        + gui.mode.BUTTON_BINDING_ACTIONS)

    ## Mode related
    @property
    def setting_mode(self):
        return self._mode
        
    @setting_mode.setter
    def setting_mode(self, modenum):
        if self._mode != modenum:
            self._queue_draw_brush()
            self._mode = modenum
                                       
    ## Initialization

    def __init__(self, **kwds):
        """Initialize"""
        super(SizechangeMode, self).__init__(**kwds)
        self.app = None
        self._cursor = Gdk.Cursor(Gdk.CursorType.BLANK_CURSOR)

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
       #r += 2 * base_radius * b.get_base_value('offset_by_random')
        r *= tdw.scale
        return r
        
    def button_press_cb(self, tdw, event):
        # getting returning point of cursor,in screen coordinate
        self._queue_draw_brush() # erase previous brush circle (if exists)
        if self.base_x is None:
            self._initial_radius = self.get_cursor_radius(tdw)
            self.base_x = event.x
            self.base_y = event.y
        
        return super(SizechangeMode, self).button_press_cb(tdw, event)

    def button_release_cb(self, tdw, event):
        # Some button(such as Bamboo pad) might not report release event!
        result = super(SizechangeMode, self).button_release_cb(tdw, event)
        return result

    def drag_start_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        self._queue_draw_brush()

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

        self._queue_draw_brush()
        adj = self.app.brush_adjustment['radius_logarithmic']
        cur_value = adj.get_value() + (dx / 120.0)
        adj.set_value(cur_value)

        self._queue_draw_brush()
        super(SizechangeMode, self).drag_update_cb(tdw, event, dx, dy)

    def drag_stop_cb(self, tdw):
        self._ensure_overlay_for_tdw(tdw)
        self._queue_draw_brush()
        self.start_drag = False
         
        self.base_x = None
        self.base_y = None

        # Reset cursor positon at start position of dragging.
        Gdk.Device.warp(self._dev, self._scr, self._ox, self._oy)

        # Reset mode as default
        super(SizechangeMode, self).drag_stop_cb(tdw)

    ## Overlays

    def _generate_overlay(self, tdw):
        return _Overlay(self, tdw)
    
    def _queue_draw_brush(self):
        space = 2 # I'm unsure why this number brought good result.
                  # might be line width * 2?
        margin = 3
        for tdw, overlay in self._overlays.items():
            if self.base_x != None:
                cur_radius = self.get_cursor_radius(tdw)
                areasize = cur_radius*2 + space*2 +1
                if areasize < 32:
                    tdw.queue_draw_area(
                            self.base_x - cur_radius - space,  
                            self.base_y - cur_radius - space,
                            areasize, areasize)
                else:
                    # To reduce redrawing area
                    dx = cur_radius * math.sin(math.pi / 4.0)
                    dw = dx * 2
                    dh = cur_radius - dx
                    dw += margin * 2
                    dh += margin * 2
                    # Top
                    tdw.queue_draw_area(
                            self.base_x - dx - margin,
                            self.base_y - cur_radius - margin,
                            dw, dh)
                    # Left
                    tdw.queue_draw_area(
                            self.base_x - cur_radius - margin,
                            self.base_y - dx - margin,
                            dh, dw)
                    # Right
                    tdw.queue_draw_area(
                            self.base_x + dx - margin,
                            self.base_y - dx - margin,
                            dh, dw)
                    # Bottom
                    tdw.queue_draw_area(
                            self.base_x - dx - margin,
                            self.base_y + dx - margin,
                            dw, dh)

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
    _TILT_RADIUS = 64    
    
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
            
    def _queue_draw_brush(self):
        space = 2 # I'm unsure why this number brought good result.
                  # might be line width * 2?
        margin = 3
        for tdw, overlay in self._overlays.items():
            if self.base_x != None:
                t_rad = self._TILT_RADIUS + margin
                tdw.queue_draw_area(
                        self.base_x - t_rad,
                        self.base_y - t_rad,
                        t_rad * 2,
                        t_rad * 2)
                        
    def drag_update_cb(self, tdw, event, dx, dy):
        self._ensure_overlay_for_tdw(tdw)
 
        self._queue_draw_brush()
        
        
        
        cur_value = self.tilt_x_value + (dx / 120.0)
        self.tilt_x_value = cur_value
        cur_value = self.tilt_y_value + (dy / 120.0)
        self.tilt_y_value = cur_value
        self._queue_draw_brush()
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
        # Remap input values into 
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
        overridden_settings = []
        for cname, text in self._COMMON_SETTINGS:
            if cname.startswith('tilt_offset'):
                overridden_settings.append((cname, text))
        self._COMMON_SETTINGS = overridden_settings
        
        # And then,call superclass method
        super(_TiltOffsetOptionWidget, self).__init__()

    def init_reset_widgets(self, row):
        """To cancel creating 'reset setting' button"""
        pass


