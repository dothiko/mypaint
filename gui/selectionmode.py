# This file is part of MyPaint.
# Copyright (C) 2016 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.




## Imports

import math
import logging
logger = logging.getLogger(__name__)
import weakref
import array

from gi.repository import Gtk, Gdk
from gettext import gettext as _
from gui. curve import CurveWidget

import gui.mode
import gui.cursor
import gui.overlays
import gui.ui_utils

""" 
HOW TO ADD Selection FUNCTIONALITY TO (YOUR) MODE CLASS

 1. add 'SelectionMode' to 'permitted_switch_actions' of the mode class.

```
    permitted_switch_actions = set(gui.mode.BUTTON_BINDING_ACTIONS).union([
            'RotateViewMode',
            'ZoomViewMode',
            'PanViewMode',
            'SelectionMode',
        ])
```

 2. add callback 'select_area_cb(self, selection_mode)' to the mode class.
    argument selection_mode is the SelectionMode itself.
    it has utility methods such as 'is_inside_model/is_inside_display',
    to distinguish whether the point(s) are inside the selection area or not.

    CAUTION:
    you cannot access your modeclass's 'self.doc' inside the callback, 
    because doc attribute is invalidated by modestack,
    until self.enter() is called.
    so you must use parameter selection_mode,
    i.e. 'selection_mode.doc', instead of 'self.doc'.

```
    def select_area_cb(self, selection_mode):
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

SEE ALSO:
    With document.select_all_cb(self, action) of gui/document.py and
    GtkAction "SelectAll" / "DeselectAll" of gui/resource.xml
    you can implement 'select_all()' or 'deselect_all()' for your mode. 
    In conjection with SelectionMode, you can implement standard
    selection interface.

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

    RECTANGLE_INDEX = ( (0,0), (1,0), (1,1), (0,1) )

    ## Initialization

    def __init__(self, **kwds):
        """Initialize"""
        super(SelectionMode, self).__init__(**kwds)
        self.app = None
        self._cursor = Gdk.Cursor(Gdk.CursorType.BLANK_CURSOR)
        self._overlays = {}  # keyed by tdw
        self._x = array.array('f',(0,0))
        self._y = array.array('f',(0,0))
        self._mx = array.array('f',(0,0))
        self._my = array.array('f',(0,0))

    def reset(self):
        self._x[0] = self._x[1] = 0
        self._y[0] = self._y[1] = 0
        self._mx[0] = self._mx[1] = 0
        self._my[0] = self._my[1] = 0
        
    #def debug_disp(self):
        #print("screen: sx,sy:%.2f,%.2f / ex,ey:%.2f,%.2f" % (self._x[0], self._y[0], self._x[1], self._y[1]))
        #print("model: sx,sy:%.2f,%.2f / ex,ey:%.2f,%.2f" % (self._mx[0], self._my[0], self._mx[1], self._my[1]))

    def _refresh_model_coords(self, tdw):
        for i in xrange(2):
            self._mx[i], self._my[i] = tdw.display_to_model(
                    self._x[i], self._y[i])
        self._justify_coords()
            
    def _get_min_max_pos(self, tdw, margin=2):
        return gui.ui_utils.get_outmost_area(tdw,
                self._mx[0], self._my[0], self._mx[1], self._my[1], 
                margin=margin)

    def get_min_max_pos_model(self, margin=0):
        return gui.ui_utils.get_outmost_area(None,
                self._mx[0], self._my[0], self._mx[1], self._my[1], 
                margin=margin)
                
    def _justify_coords(self):
        if self._mx[0] > self._mx[1]:
            self._mx[0] , self._mx[1] = self._mx[1], self._mx[0]
        if self._my[0] > self._my[1]:
            self._my[0] , self._my[1] = self._my[1], self._my[0]


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
            if hasattr(returning_mode, 'select_area_cb'):
                returning_mode.select_area_cb(self)

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
        return self._get_min_max_pos(tdw)



    ## Selection related methods

    def start(self, tdw, x, y):
        self._x[0] = self._x[1] = x
        self._y[0] = self._y[1] = y
        self._refresh_model_coords(tdw)
        

    def drag(self, tdw, x, y):
        self._x[1] = x
        self._y[1] = y

        self._refresh_model_coords(tdw)

    def get_display_point(self, tdw, i):
        xi, yi = self.RECTANGLE_INDEX[i]
        return tdw.model_to_display(
                self._mx[xi], self._my[yi])

    def get_model_point(self, i):
        xi, yi = self.RECTANGLE_INDEX[i]
        return (self._mx[xi], self._my[xi])

    def is_inside_model(self, mx, my):
        """ check whether mx,my is inside selected rectangle.
        this method needs model coordinate point.
        """
        return (self._mx[0] <= mx <= self._mx[1] and
                self._my[0] <= my <= self._my[1])

    def is_valid(self):
        return not (self._x[0] == self._x[1] or
            self._y[0] == self._y[1]) 


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
        cr.set_dash((), 0)
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


