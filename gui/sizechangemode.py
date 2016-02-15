# This file is part of MyPaint.
# Copyright (C) 2016 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# NOTE: This file based on linemode.py



## Imports

import math
import logging
logger = logging.getLogger(__name__)
import weakref

import gtk2compat
import gtk
from gtk import gdk
from gettext import gettext as _
import gobject
from curve import CurveWidget

import gui.mode
import gui.cursor
import gui.overlays


## Module constants



## Interaction modes for making lines

class SizechangeMode(gui.mode.ScrollableModeMixin,
                    gui.mode.BrushworkModeMixin,
                    gui.mode.OneshotDragMode):
    """Oncanvas brush Size change mode"""

    ## Class constants

    ACTION_NAME = "SizechangeMode"

    _OPTIONS_WIDGET = None

    ## Class configuration.

    permitted_switch_actions = set([
        "PanViewMode",
        "ZoomViewMode",
        "RotateViewMode",
    ])

    pointer_behavior = gui.mode.Behavior.PAINT_CONSTRAINED
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW

    @property
    def active_cursor(self):
        return self._cursor  # Completely blank cursor
    
    @classmethod
    def get_name(cls):
        return _(u"Size changer")

    def get_usage(self):
        return _(u"Change brush size.when move to up/left,size decreased.down/right,size increased.")

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
        self._cursor = gdk.Cursor(gdk.CursorType.BLANK_CURSOR)
        self._overlays = {}  # keyed by tdw
        self.start_drag = False

    ## InteractionMode/DragMode implementation

    def enter(self, doc, **kwds):
        """Enter the mode.
        """
        super(SizechangeMode, self).enter(doc, **kwds)
        self.app = self.doc.app
        self.base_x = None
        if not self._is_active():
            self._discard_overlays()

    def leave(self, **kwds):
        if not self._is_active():
            self._discard_overlays()

        return super(SizechangeMode, self).leave(**kwds)

    def _is_active(self):
        for mode in self.doc.modes:
            if mode is self:
                return True
        return False


    def get_cursor_radius(self,tdw):
        #FIXME nearly Code duplication from 
        #      gui/tiledrawwidget.py:_get_cursor_info
        b = tdw.doc.brush.brushinfo
        base_radius = math.exp(b.get_base_value('radius_logarithmic'))
        r = base_radius
        r += 2 * base_radius * b.get_base_value('offset_by_random')
        r *= tdw.scale
        r += 0.5
        return r


    def drag_start_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        self._queue_draw_brush() # erase previous brush circle

        self.pressed_x, self.pressed_y = \
                tdw.display_to_model(event.x, event.y)
        if self.base_x == None:
            self.base_x = self.pressed_x
            self.base_y = self.pressed_y
            # getting returning point of cursor
            disp = gdk.Display.get_default()
            screen, self.start_screen_x , self.start_screen_y ,mod = \
                    disp.get_pointer()
        self._queue_draw_brush()
        self.start_drag = True
        super(SizechangeMode, self).drag_start_cb(tdw, event)

    def drag_update_cb(self, tdw, event, dx, dy):

        self._ensure_overlay_for_tdw(tdw)
        if self.start_drag:
            dx, dy = tdw.display_to_model(event.x, event.y)
            cx = dx - self.pressed_x
            cy = dy - self.pressed_y
            cs = math.hypot(cx, cy)
            if cs > 0.0:
                nx = cx / cs
                ny = cy / cs
                angle = math.acos(ny)  # Getting angle
                diff = cs / 200.0  # 200.0 is not theorical number,it's my feeling

                if math.pi / 4 < angle < math.pi / 4 + math.pi / 2:
                    if nx < 0.0:
                        diff *= -1
                elif ny < 0.0:
                    diff *= -1


                self._queue_draw_brush()
                adj = self.app.brush_adjustment['radius_logarithmic']
                adj.set_value(adj.get_value() + diff)
                self._queue_draw_brush()

                # refresh pressed position
                self.pressed_x = dx
                self.pressed_y = dy

        return super(SizechangeMode, self).drag_update_cb(tdw, event, dx, dy)

    def drag_stop_cb(self, tdw):
        self._ensure_overlay_for_tdw(tdw)
        if self.start_drag:
            self._queue_draw_brush()
        self.start_drag = False

        # return cursor to staring point.
        d=tdw.get_display()
        d.warp_pointer(d.get_default_screen(),self.start_screen_x,self.start_screen_y)
        return super(SizechangeMode, self).drag_stop_cb(tdw)



    ## Overlays
    #  FIXME: mostly copied from gui/inktool.py
    #  should I make it something mixin?
    
    def _ensure_overlay_for_tdw(self, tdw):
        overlay = self._overlays.get(tdw)
        if not overlay:
            overlay = _Overlay(self, tdw)
            tdw.display_overlays.append(overlay)
            self._overlays[tdw] = overlay
        return overlay

    def _discard_overlays(self):
        for tdw, overlay in self._overlays.items():
            tdw.display_overlays.remove(overlay)
            tdw.queue_draw()
        self._overlays.clear()

    def _queue_draw_brush(self):
        space = 2 # I'm unsure why this number brought good result.
                  # might be line width * 2?
        for tdw, overlay in self._overlays.items():
            if self.base_x != None:
                cur_radius = self.get_cursor_radius(tdw)
                sx, sy = tdw.model_to_display(self.base_x, self.base_y)
                areasize = cur_radius*2 + space*2 +1
                tdw.queue_draw_area(
                        sx - cur_radius - space,  
                        sy - cur_radius - space,
                        areasize, areasize)

class _Overlay (gui.overlays.Overlay):
    """Overlay for an SizechangeMode's brushsize"""

    def __init__(self, sizemode, tdw):
        super(_Overlay, self).__init__()
        self._sizemode = weakref.proxy(sizemode)
        self._tdw = weakref.proxy(tdw)

    def paint(self, cr):
        """Draw brush size to the screen"""

        cr.save()
        color = gui.style.ACTIVE_ITEM_COLOR
        cr.set_source_rgb(0, 0, 0)
        sx, sy = self._tdw.model_to_display(
                self._sizemode.base_x, 
                self._sizemode.base_y)
        cr.set_line_width(1)
        cr.arc( sx, sy,
                self._sizemode.get_cursor_radius(self._tdw),
                0.0,
                2*math.pi)
        cr.stroke()
        cr.restore()

