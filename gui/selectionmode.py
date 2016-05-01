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

 2. add callback 'select_area(sx, sy, ex, ey)' to mode.
    (sx, sy) is left-top , (ex, ey) is right-bottom of 
    selected area, in model coordinate.

```
    def select_area(self, sx, sy, ex, ey):
        for idx,cn in enumerate(self.nodes):
            if sx <= cn.x <= ex and sy <= cn.y <= ey:
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
        self.reset()

    def reset(self):
        self.sx = 0
        self.sy = 0
        self.ex = 0
        self.ey = 0

        # workaround flag, because drag_stop_cb does not have
        # event parameter.
        self.is_addition = False

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
        
        returning_mode = self.doc.modes.top
        if hasattr(returning_mode, 'select_area'):
            sx, sy, ex, ey = self.get_sorted_area()
            if sx != ex and sy != ey:
                returning_mode.select_area(sx, sy, ex, ey)

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
        mx, my = tdw.display_to_model(event.x, event.y)

        self.start(mx, my)
        self.is_addition = (event.state & Gdk.ModifierType.CONTROL_MASK)
        self._queue_draw_selection_rect() # to start

        super(SelectionMode, self).drag_start_cb(tdw, event)

    def drag_update_cb(self, tdw, event, dx, dy):

        self._ensure_overlay_for_tdw(tdw)
        mx, my = tdw.display_to_model(event.x, event.y)

        self._queue_draw_selection_rect() # to erase
        self.drag(mx, my)
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
                    *self.get_update_rect(tdw))


    ## Mode options

   #def get_options_widget(self):
   #    """Get the (class singleton) options widget"""
   #    cls = self.__class__
   #    if cls._OPTIONS_WIDGET is None:
   #        widget = _SizeChangerOptionWidget()
   #        cls._OPTIONS_WIDGET = widget
   #    return cls._OPTIONS_WIDGET

    ## Selection related methods

    def start(self,x,y):
        self.sx = x
        self.sy = y
        self.ex = x
        self.ey = y
        
        SelectionMode.finalized_rect = None

    def drag(self,x,y):
        self.ex = x
        self.ey = y

    def get_sorted_area(self):
        """
        Get sorted selection area, in model coordinate.
        """
        return (min(self.sx, self.ex),
                min(self.sy, self.ey),
                max(self.sx, self.ex),
                max(self.sy, self.ey))


    def get_update_rect(self,tdw):
        """Get update 'rect' for update(erase) tdw"""

        c_area = self.get_sorted_area()
        csx, csy = tdw.model_to_display(c_area[0], c_area[1])
        cex, cey = tdw.model_to_display(c_area[2], c_area[3])
        margin = self.LINE_WIDTH * 2
        return (csx - margin,
                csy - margin,
                (cex - csx + 1) + margin * 2,
                (cey - csy + 1) + margin * 2)

    def is_enabled(self):
        return self.sx != None

    def is_inside(self, x, y):
        """ check whether x,y is inside selected rectangle.
        this method needs screen coordinate point.
        """
        sx, sy, ex, ey = self.get_sorted_area()
        return (x >= sx and x <= ex and y >= sy and y <= ey)

    def get_display_offset(self, tdw):
        sx, sy, ex, ey = self.get_display_area(tdw)
        return (ex - sx, ey - sy)

    def get_model_offset(self):
        return (self.ex - self.sx, self.ey - self.sy)

    def get_display_area(self, tdw):
        sx,sy = tdw.model_to_display(self.sx, self.sy)
        ex,ey = tdw.model_to_display(self.ex, self.ey)
        return (sx, sy, ex, ey)

    def get_offset(self):
        return (self.ex - self.sx, self.ey - self.sy)




class _Overlay (gui.overlays.Overlay):
    """Overlay for an SizechangeMode's brushsize"""

    def __init__(self, mode, tdw):
        super(_Overlay, self).__init__()
        self._mode = weakref.proxy(mode)
        self._tdw = weakref.proxy(tdw)

    def draw_selection_rect(self, cr):
        sx, sy, ex, ey = self._mode.get_display_area(self._tdw)

        cr.save()
        for color in ( (0,0,0) , (1,1,1) ):
            cr.set_source_rgb(*color)
            cr.set_line_width(self._mode.LINE_WIDTH)
            cr.new_path()
            cr.move_to(sx, sy)
            cr.line_to(ex, sy)
            cr.line_to(ex, ey)
            cr.line_to(sx, ey)
            cr.close_path()
            cr.stroke()
            cr.set_dash( (3.0, ) )
        cr.restore()

    def paint(self, cr):
        """Draw selection rectangle to the screen"""
        self.draw_selection_rect(cr)



#class _SizeChangerOptionWidget(gui.mode.PaintingModeOptionsWidgetBase):
#    """ Because OncanvasSizeMode use from dragging + modifier
#    combination, thus this option widget mostly unoperatable.
#    but I think user would feel some 'reliability' when there are 
#    value displaying scale and label.
#    """
#
#    def __init__(self):
#        # Overwrite self._COMMON_SETTINGS
#        # to use(show) only 'radius_logarithmic' scale.
#        for cname, text in self._COMMON_SETTINGS:
#            if cname == 'radius_logarithmic':
#                self._COMMON_SETTINGS = [ (cname, text) ]
#                break
#        
#        # And then,call superclass method
#        super(_SizeChangerOptionWidget, self).__init__()
#
#    def init_reset_widgets(self, row):
#        """To cancel creating 'reset setting' button"""
#        pass
#
