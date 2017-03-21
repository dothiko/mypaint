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


## Module constants



## Interaction modes for making lines

class SizechangeMode(gui.mode.ScrollableModeMixin,
                    gui.mode.BrushworkModeMixin,
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
        
        # _in_nonpainting flag :
        # 
        # This flag means 'the behavior of device 
        # which activate this mode is set as NON_PAINTING'
        # If so, the gui.document does not fire the 
        # button_release_cb. Thus you cannot exit this
        # mode even release the device button.
        # So when non_painting behavior detected,
        # This mode actually does not do anything
        # until PAINTING device button1 pressed.
        #
        # This flag is THREE-STATE,
        # when True, nothing would be done.
        # for False, size-setting is done.(it is normal behavior)
        # for None, size-setting is done, and pop up when 
        # the another device(i.e. stylus button) released.
        self._in_non_painting = False

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
        #r += 0.5
        return r
        
    def button_press_cb(self, tdw, event):
        # getting returning point of cursor,in screen coordinate
        self._queue_draw_brush() # erase previous brush circle (if exists)
        if self.base_x is None:
            self._initial_radius = self.get_cursor_radius(tdw)
            self.base_x = event.x
            self.base_y = event.y

        mon = self.app.device_monitor
        dev = event.get_source_device()
        dev_settings = mon.get_device_settings(dev)
        if not (dev_settings.usage_mask & self.pointer_behavior):
            self._in_non_painting = True
        else:
            if self._in_non_painting == True:
                self._in_non_painting = None
            else:
                self._in_non_painting = False

        return super(SizechangeMode, self).button_press_cb(tdw, event)

    def button_release_cb(self, tdw, event):
        # Some button(such as Bamboo pad) might not report release event!
        result = super(SizechangeMode, self).button_release_cb(tdw, event)
        if self._in_non_painting is None:
            self._in_non_painting = False
            self.doc.modes.pop()
        return result

    def drag_start_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        self._queue_draw_brush()
        super(SizechangeMode, self).drag_start_cb(tdw, event)

    def drag_update_cb(self, tdw, event, dx, dy):
        self._ensure_overlay_for_tdw(tdw)
            
        if not self._in_non_painting:
            self._queue_draw_brush()
            adj = self.app.brush_adjustment['radius_logarithmic']
            cur_value = adj.get_value() + (dx / 120.0)
            adj.set_value(cur_value)
            self._queue_draw_brush()
        super(SizechangeMode, self).drag_update_cb(tdw, event, dx, dy)

    def drag_stop_cb(self, tdw):
        self._ensure_overlay_for_tdw(tdw)
        if not self._in_non_painting:
            self._queue_draw_brush()
            self.start_drag = False
             
            self.base_x = None
            self.base_y = None
        super(SizechangeMode, self).drag_stop_cb(tdw)
        

    ## Overlays

    def _generate_overlay(self, tdw):
        return _Overlay(self, tdw)
    
    def _queue_draw_brush(self):
        space = 2 # I'm unsure why this number brought good result.
                  # might be line width * 2?
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
                    margin = 3
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

    ## Mode options

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        cls = self.__class__
        if cls._OPTIONS_WIDGET is None:
            widget = _SizeChangerOptionWidget()
            cls._OPTIONS_WIDGET = widget
        return cls._OPTIONS_WIDGET


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

        if mode is not None and tdw is not None:
            cr.set_source_rgb(0, 0, 0)
            cr.set_line_width(1)
            if mode.base_x != None:
                cr.arc( mode.base_x,
                        mode.base_y,
                        mode.get_cursor_radius(tdw),
                        0.0,
                        2*math.pi)
            cr.stroke_preserve()
            cr.set_dash( (10,) )
            cr.set_source_rgb(1, 1, 1)
            cr.stroke()
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

