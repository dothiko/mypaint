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

from gi.repository import Gtk, Gdk
from gettext import gettext as _
import gobject
from curve import CurveWidget
import array

import gui.mode
import gui.cursor
import gui.overlays
import gui.ui_utils

""" 
HOW TO ADD Selection FUNCTIONALITY TO MODE

 1. add 'SelectionMode' to 'permitted_switch_actions' 

```
    permitted_switch_actions = set(gui.mode.BUTTON_BINDING_ACTIONS).union([
            'RotateViewMode',
            'ZoomViewMode',
            'PanViewMode',
            'SelectionMode',
        ])
```

 2. add callback 'select_area(self, selection_mode)' to mode.
    argument selection_mode is the SelectionMode,it has
    'is_inside_model/is_inside_display' method to 
    distinguish whether the point(s) are inside selection or not.

```
    def select_area(self, selection_mode):
        modified = False
        for idx,cn in enumerate(self.nodes):
            if selection_mode.is_inside_model(cn.x, cn.y):
                if not idx in self.selected_nodes:
                    self.selected_nodes.append(idx)
                    modified = True
        if modified:
            self._queue_redraw_all_nodes()
```  

 3. Just completed!

"""


## Module constants



## Interaction modes for making lines

class SelectionMode(gui.mode.ScrollableModeMixin,
                    gui.mode.OneshotDragMode):
    """Oncanvas area selection mode"""

    ## Class constants

    ACTION_NAME = "SelectionMode"
    _OPTIONS_WIDGET = None
    LINE_WIDTH = 1

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
        return _(u"On-canvas item selector")

    def get_usage(self):
        return _(u"Select on-canvas items such as inktool nodes or area of layer by dragging pointer.")

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
        super(SelectionMode, self).__init__(**kwds)
        self.app = None
        self._cursor = Gdk.Cursor(Gdk.CursorType.BLANK_CURSOR)
        self._overlays = {}  # keyed by tdw
        self._x = array.array('f',(0,0,0,0))
        self._y = array.array('f',(0,0,0,0))
        self._mx = array.array('f',(0,0,0,0))
        self._my = array.array('f',(0,0,0,0))

    def reset(self):
        self._x[0] = self._x[1] = self._x[2] = self._x[3] = 0
        self._y[0] = self._y[1] = self._y[2] = self._y[3] = 0
        self._mx[0] = self._mx[1] = self._mx[2] = self._mx[3] = 0
        self._my[0] = self._my[1] = self._my[2] = self._my[3] = 0


    def _modelnized(self, tdw):
        for i in xrange(4):
            self._mx[i], self._my[i] = tdw.display_to_model(
                    self._x[i], self._y[i])

    def _get_min_max_pos(self):
        min_x = self._x[0]
        max_x = min_x
        min_y = self._y[0]
        max_y = min_y
        for i in xrange(3):
            min_x = min(min_x, self._x[i+1])
            max_x = max(max_x, self._x[i+1])
            min_y = min(min_y, self._y[i+1])
            max_y = max(max_y, self._y[i+1])

        return (min_x, min_y, max_x, max_y)

    ## InteractionMode/DragMode implementation

    def enter(self, doc, **kwds):
        """Enter the mode.
        """
        super(SelectionMode, self).enter(doc, **kwds)
        self.app = self.doc.app
        if not self._is_active():
            self._discard_overlays()

    def leave(self, **kwds):
        if not self._is_active():
            self._discard_overlays()
        
        if self.is_valid():
            returning_mode = self.doc.modes.top
            if hasattr(returning_mode, 'select_area'):
                returning_mode.select_area(self)

        return super(SelectionMode, self).leave(**kwds)

    def _is_active(self):
        for mode in self.doc.modes:
            if mode is self:
                return True
        return False

    def button_press_cb(self, tdw, event):
        super(SelectionMode, self).button_press_cb(tdw, event)
    
    def button_release_cb(self, tdw, event):
        return super(SelectionMode, self).button_release_cb(tdw, event)

    def drag_start_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        self.start(tdw, event.x, event.y)
        self.is_addition = (event.state & Gdk.ModifierType.CONTROL_MASK)
        self._queue_draw_selection_rect() # to start

        super(SelectionMode, self).drag_start_cb(tdw, event)

    def drag_update_cb(self, tdw, event, dx, dy):

        self._ensure_overlay_for_tdw(tdw)
        self._queue_draw_selection_rect() # to erase
        self.drag(tdw, event.x, event.y)
        self._queue_draw_selection_rect()
        return super(SelectionMode, self).drag_update_cb(tdw, event, dx, dy)

    def drag_stop_cb(self, tdw):
        self._ensure_overlay_for_tdw(tdw)
        self._queue_draw_selection_rect()

        self._discard_overlays()
        return super(SelectionMode, self).drag_stop_cb(tdw)



    ## Overlays
    #  FIXME: mostly copied from gui/inktool.py
    #  Would it be better there is something mixin?
    #  (OverlayedMixin??)
    
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

    def _queue_draw_selection_rect(self):
        """Redraws selection area"""
        for tdw, overlay in self._overlays.items():
            tdw.queue_draw_area(
                    *self._get_update_rect(tdw))

    def _get_update_rect(self, tdw):
        """Get update 'rect' for update(erase) tdw"""
        csx, csy, cex, cey = self._get_min_max_pos()
        margin = self.LINE_WIDTH * 2
        return (csx - margin,
                csy - margin,
                (cex - csx + 1) + margin * 2,
                (cey - csy + 1) + margin * 2)



    ## Selection related methods

    def start(self, tdw, x, y):
        self._x[0] = self._x[1] = self._x[2] = self._x[3] = x
        self._y[0] = self._y[1] = self._y[2] = self._y[3] = y
        self._modelnized(tdw)
        

    def drag(self, tdw, x, y):
        self._x[1] = self._x[2] = x
        self._y[2] = self._y[3] = y

        self._modelnized(tdw)

    def get_display_point(self, tdw, i):
        return tdw.model_to_display(self._mx[i], self._my[i])

    def get_model_point(self, i):
        return (self._mx[i], self._my[i])

    def is_inside_model(self, mx, my):
        """ check whether mx,my is inside selected rectangle.
        this method needs model coordinate point.
        """
        if not gui.ui_utils.is_inside_triangle( 
                mx, my, ( (self._mx[0], self._my[0]),
                        (self._mx[1], self._my[1]),
                        (self._mx[2], self._my[2]))):
            return gui.ui_utils.is_inside_triangle(
                    mx, my, ( (self._mx[0], self._my[0]),
                            (self._mx[2], self._my[2]),
                            (self._mx[3], self._my[3])))
        else:
            return True

    def is_valid(self):
        return not (self._x[0] == self._x[1] == self._x[2] == self._x[3] and 
            self._y[0] == self._y[1] == self._y[2] == self._y[3]) 





class _Overlay (gui.overlays.Overlay):
    """Overlay for an SizechangeMode's brushsize"""

    def __init__(self, mode, tdw):
        super(_Overlay, self).__init__()
        self._mode = weakref.proxy(mode)
        self._tdw = weakref.proxy(tdw)

    def draw_selection_rect(self, cr):
        tdw = self._tdw
        cr.save()
        cr.set_source_rgb(0,0,0)
        cr.set_line_width(self._mode.LINE_WIDTH)

        cr.new_path()
        cr.move_to(*self._mode.get_display_point(tdw, 0))
        cr.line_to(*self._mode.get_display_point(tdw, 1))
        cr.line_to(*self._mode.get_display_point(tdw, 2))
        cr.line_to(*self._mode.get_display_point(tdw, 3))
        cr.close_path()
        cr.stroke_preserve()

        cr.set_source_rgb(1,1,1)
        cr.set_dash( (3.0, ) )
        cr.stroke()
        cr.restore()

    def paint(self, cr):
        """Draw selection rectangle to the screen"""
        self.draw_selection_rect(cr)


