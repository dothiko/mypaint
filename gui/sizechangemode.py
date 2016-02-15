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
                    gui.mode.DragMode):
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
        return self._cursor
       #cursor_name = gui.cursor.Name.PENCIL
       #return self.doc.app.cursors.get_action_cursor(
       #    self.ACTION_NAME,
       #    cursor_name
       #)
    
    @classmethod
    def get_name(cls):
        return _(u"Size changer")

    def get_usage(self):
        return _(u"Change brush size.when move to up/left,size decreased.down/right,size increased.")

    @property
    def inactive_cursor(self):
        return self._cursor

   #@property
   #def inactive_cursor(self):
   #    cursor_name = gui.cursor.Name.CROSSHAIR_OPEN_PRECISE
   #    return self.doc.app.cursors.get_action_cursor(
   #        self.ACTION_NAME,
   #        cursor_name
   #    )

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
        self.cursor_radius = 0.0
       #self.idle_srcid = None

    ## InteractionMode/DragMode implementation

    def enter(self, doc, **kwds):
        """Enter the mode.

        If modifiers are held when the mode is entered, the mode is a oneshot
        mode and is popped from the mode stack automatically at the end of the
        drag. Without modifiers, line modes may be continued, and some
        subclasses offer additional options for adjusting control points.

        """
        super(SizechangeMode, self).enter(doc, **kwds)
        self.app = self.doc.app
        rootstack = self.doc.model.layer_stack
        self.base_x = None
       #self._update_cursors()
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

   #def _update_cursors(self, *_ignored):
   #    pass
   #    if self.in_drag:
   #        return   # defer update to the end of the drag
   #    layer = self.doc.model.layer_stack.current
   #    self._line_possible = (layer.get_paintable() and
   #                           layer.visible and not layer.locked)
   #    self.doc.tdw.set_override_cursor(self._cursor)

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

    def set_cursor(self,tdw):
        radius = self.get_cursor_radius(tdw)
        self._cursor = gui.cursor.get_brush_cursor(radius, gui.cursor.BRUSH_CURSOR_STYLE_NORMAL, self.app.preferences)
        tdw.get_window().set_cursor(self._cursor)

    def drag_start_cb(self, tdw, event):
        super(SizechangeMode, self).drag_start_cb(tdw, event)
        self._queue_draw_brush() # erase previous 

        self.pressed_x, self.pressed_y = \
                tdw.display_to_model(event.x, event.y)
        if self.base_x == None:
            self.base_x = self.pressed_x
            self.base_y = self.pressed_y
        self.cursor_radius = self.get_cursor_radius(tdw)
        self._queue_draw_brush()

    def drag_update_cb(self, tdw, event, dx, dy):

        self._ensure_overlay_for_tdw(tdw)

        dx, dy = tdw.display_to_model(event.x, event.y)
        cx = dx - self.pressed_x
        cy = dy - self.pressed_y
        cs = math.sqrt(cx * cx + cy * cy)
        if cs > 0.0:
            nx = cx / cs
            ny = cy / cs
            angle = math.acos(ny)  # Getting angle
            diff = cs / 1000.0  # 128.0 is not theorical number,it's my feeling
            diff = 0.1  # test

            if math.pi / 4 < angle < math.pi / 4 + math.pi / 2:
                if nx < 0.0:
                    diff *= -1
            elif ny < 0.0:
                diff *= -1


            self._queue_draw_brush()
            adj = self.app.brush_adjustment['radius_logarithmic']
            adj.set_value(adj.get_value() + diff)
            self.cursor_radius = self.get_cursor_radius(tdw)
            self._queue_draw_brush()

        return super(SizechangeMode, self).drag_update_cb(tdw, event, dx, dy)

    def drag_stop_cb(self, tdw):
        self._queue_draw_brush()
        return super(SizechangeMode, self).drag_stop_cb(tdw)

    def _drag_idle_cb(self):
        # Updates the on-screen line during drags.
        pass

    ## Overlays
    #  taken from gui/inktool.py
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
        space = 2
        for tdw, overlay in self._overlays.items():
            if self.base_x != None:
                sx, sy = tdw.display_to_model(self.base_x, self.base_y)
                areasize = self.cursor_radius*2 + space*2 +1
                tdw.queue_draw_area(
                        sx - self.cursor_radius - space, 
                        sy - self.cursor_radius - space,
                        areasize, areasize)

class _Overlay (gui.overlays.Overlay):
    """Overlay for an SizechangeMode's brushsize"""

    def __init__(self, sizemode, tdw):
        super(_Overlay, self).__init__()
        self._sizemode = weakref.proxy(sizemode)
        self._tdw = weakref.proxy(tdw)

    def paint(self, cr):
        """Draw brush size to the screen"""

        if self._sizemode.cursor_radius > 0:
            cr.save()
            color = gui.style.ACTIVE_ITEM_COLOR
            cr.set_source_rgb(0, 0, 0)#*color.get_rgb())
            cr.set_line_width(1)
            cr.arc( self._sizemode.base_x,
                    self._sizemode.base_y,
                    self._sizemode.cursor_radius,
                    0.0,
                    2*math.pi)
            cr.stroke()
            cr.restore()

