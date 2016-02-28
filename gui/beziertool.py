# This file is part of MyPaint.
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


## Imports

import math
from numpy import isfinite
import collections
import weakref
import os.path
from logging import getLogger
logger = getLogger(__name__)

from gettext import gettext as _
import gi
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GLib

import gui.mode
import gui.overlays
import gui.style
import gui.drawutils
import lib.helpers
import gui.cursor
import lib.observable

from inktool import *

## Class defs

class _Control_Handle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
class _Node_Bezier (object):
    """Recorded control point.
    In bezier curve,nodes are frequently rewritten
    and each nodes own control handles.
    so this class is object, not a namedtuple.

    _Node_Bezier thave the following 6 fields, in order

    * x, y: model coords, float
    * pressure: float in [0.0, 1.0]
    * xtilt, ytilt: float in [-1.0, 1.0]
    * control_handle: two _Control_Handle objects.
    """
    
            
            
    def __init__(self,x,y,pressure=None,xtilt=0.0,ytilt=0.0,
            control_handles=None,even=True):
        self.x = x
        self.y = y
        self.pressure = pressure
        self.xtilt = xtilt
        self.ytilt = ytilt
        self.time = 0
        if control_handles:
            self.control_handles = control_handles
        else:
            self.control_handles = (_Control_Handle(x, y), _Control_Handle(x, y))

        self._even = False
        #self.set_even(even)

    def set_even(self, flag):
        if self._even != flag and flag:
            self.control_handles[1].x = -self.control_handles[0].x
            self.control_handles[1].y = -self.control_handles[0].y
        self._even = flag

    def _replace(self, **kwarg):
        ox = self.x
        oy = self.y
        for ck in kwarg:
            if ck in self.__dict__:
                self.__dict__[ck] = kwarg[ck]
        self._refresh_control_handles(ox, oy, self.x, self.y)
        return self

    def _refresh_control_handles(self, ox, oy, nx, ny):
        """ refresh control-handle position,when node position changed.
        :param ox: old x position,in model
        :param oy: old y position,in model
        :param nx: new x position,in model
        :param ny: new y position,in model
        """
        for i in (0,1):
            self.control_handles[i].x = nx + (self.control_handles[i].x - ox)
            self.control_handles[i].y = ny + (self.control_handles[i].y - oy)

    def move(self,x, y):
        """Move this node to (x,y).
        :param x: destination x position,in model
        :param y: destination y position,in model

        Use this method to move control points simultaneously.
        """
        self._refresh_control_handles(self.x, self.y, x, y)
        self.x = x
        self.y = y
    
    
    
class _EditZone_Bezier:
    """Enumeration of what the pointer is on in the ADJUST phase"""
    EMPTY_CANVAS = 0  #: Nothing, empty space
    CONTROL_NODE = 1  #: Any control node; see target_node_index
    REJECT_BUTTON = 2  #: On-canvas button that abandons the current line
    ACCEPT_BUTTON = 3  #: On-canvas button that commits the current line
    CONTROL_HANDLE = 4 #: Control handle of bezier

class _Phase:
    """Enumeration of the states that an BezierCurveMode can be in"""
    INITIAL = 0           # Initial Phase,creating node.
    MOVE_NODE = 1         # Moving node(s)
    ADJUST_PRESSURE = 2   # Changing nodes pressure
    ADJUST_SELECTING = 3  # Nodes Area Selecting
    ADJUST_HANDLE = 5     # change control-handle position
    INIT_HANDLE = 6       # initialize control handle,right after create a node
    
def _bezier_iter(seq):
    """Converts an list of control point tuples to interpolatable arrays

    :param list seq: Sequence of tuples of nodes
    :param list selected: Selected control points index list
    :param list offset: An Offset for selected points,a tuple of (x,y).
    :returns: Iterator producing (p-1, p0, p1, p2)

    The resulting sequence of 4-tuples is intended to be fed into
    spline_4p().  The start and end points are therefore normally
    doubled, producing a curve that passes through them, along a vector
    aimed at the second or penultimate point respectively.

    """
    cint = [None, None, None, None]

    for idx,cn in enumerate(seq[:-1]):
        nn = seq[idx+1]
        cint[0] = cn
        cint[1] = cn.control_handles[1]
        cint[2] = nn.control_handles[0]
        cint[3] = nn
        yield cint

class BezierMode (InkingMode):

    ## Metadata properties

    ACTION_NAME = "BezierCurveMode"


    ## Metadata methods

    @classmethod
    def get_name(cls):
        return _(u"Bezier")

    def get_usage(self):
        return _(u"Draw, and then adjust smooth lines with bezier curve")

    @property
    def inactive_cursor(self):
        return None
        
    @property
    def active_cursor(self):
        if self.phase == _Phase.INITIAL:
            return self._crosshair_cursor
        elif self.phase == _Phase.MOVE_NODE:
            if self.zone == _EditZone_Bezier.CONTROL_NODE:
                return self._crosshair_cursor
            elif self.zone != _EditZone_Bezier.EMPTY_CANVAS: # assume button
                return self._arrow_cursor

        elif self.phase == _Phase.ADJUST_PRESSURE:
            if self.zone == _EditZone_Bezier.CONTROL_NODE:
                return self._cursor_move_nw_se

        elif self.phase == _Phase.ADJUST_SELECTING:
            return self._crosshair_cursor
        return None  

    ## Class config vars


    ## Other class vars

    _OPTIONS_PRESENTER = None   #: Options presenter singleton

    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):
        super(BezierMode, self).__init__(**kwargs)

        self.zone = _EditZone_Bezier.EMPTY_CANVAS

    def _reset_adjust_data(self):
        super(BezierMode, self)._reset_adjust_data()
        self.current_handle_index = None

    def _ensure_overlay_for_tdw(self, tdw):
        overlay = self._overlays.get(tdw)
        if not overlay:
            overlay = OverlayBezier(self, tdw)
            tdw.display_overlays.append(overlay)
            self._overlays[tdw] = overlay
        return overlay
        
    def _update_zone_and_target(self, tdw, x, y):
        """Update the zone and target node under a cursor position"""
        ## FIXME mostly copied from inktool.py
        ## the difference is control handle processing
        self._ensure_overlay_for_tdw(tdw)
        new_zone = _EditZone_Bezier.EMPTY_CANVAS
        if not self.in_drag:
            if self.phase == _Phase.MOVE_NODE or self.phase == _Phase.INITIAL:
                new_target_node_index = None
                
                # Test buttons for hits
                overlay = self._ensure_overlay_for_tdw(tdw)
                hit_dist = gui.style.FLOATING_BUTTON_RADIUS
                button_info = [
                    (_EditZone_Bezier.ACCEPT_BUTTON, overlay.accept_button_pos),
                    (_EditZone_Bezier.REJECT_BUTTON, overlay.reject_button_pos),
                ]
                for btn_zone, btn_pos in button_info:
                    if btn_pos is None:
                        continue
                    btn_x, btn_y = btn_pos
                    d = math.hypot(btn_x - x, btn_y - y)
                    if d <= hit_dist:
                        new_target_node_index = None
                        new_zone = btn_zone
                        break
                # Test nodes for a hit, in reverse draw order
                if new_zone == _EditZone_Bezier.EMPTY_CANVAS:
                    hit_dist = gui.style.DRAGGABLE_POINT_HANDLE_SIZE + 12
                    new_target_node_index = None
                    for i, node in reversed(list(enumerate(self.nodes))):
                        node_x, node_y = tdw.model_to_display(node.x, node.y)
                        d = math.hypot(node_x - x, node_y - y)
                        if d > hit_dist:
                            continue
                        new_target_node_index = i
                        new_zone = _EditZone_Bezier.CONTROL_NODE
                        break
                
                # ADDED PORTION:
                # New target node is not hit.
                # But, pointer might hit control handles
                if (new_target_node_index is None and 
                        self.current_node_index is not None):
                    c_node = self.nodes[self.current_node_index]
                    self.current_handle_index = None
                    for i, handle in enumerate(c_node.control_handles):
                        hx, hy = tdw.model_to_display(handle.x, handle.y)
                        d = math.hypot(hx - x, hy - y)
                        if d > hit_dist:
                            continue
                        new_target_node_index = self.current_node_index
                        self.current_handle_index = i
                        new_zone = _EditZone_Bezier.CONTROL_HANDLE
                        break         
                    
                    
                    
                # Update the prelit node, and draw changes to it
                if new_target_node_index != self.target_node_index:
                    if self.target_node_index is not None:
                        self._queue_draw_node(self.target_node_index)
                    self.target_node_index = new_target_node_index
                    if self.target_node_index is not None:
                        self._queue_draw_node(self.target_node_index)

                # Disable override modes when node targetted
                InkingMode.enable_switch_actions(new_target_node_index == None)
                        
                   

        elif self.phase == _Phase.ADJUST_PRESSURE:
            # Always control node,in pressure editing.
            new_zone = _EditZone_Bezier.CONTROL_NODE 

        # Update the zone, and assume any change implies a button state
        # change as well (for now...)
        if self.zone != new_zone:
            self.zone = new_zone
            self._ensure_overlay_for_tdw(tdw)
            if len(self.nodes) > 1:
                self._queue_previous_draw_buttons()
               #self._queue_draw_buttons()
        # Update the "real" inactive cursor too:
        if not self.in_drag:
            cursor = None
            if self.phase in (_Phase.INITIAL, _Phase.MOVE_NODE, _Phase.ADJUST_PRESSURE):
                if self.zone == _EditZone_Bezier.CONTROL_NODE:
                    cursor = self._crosshair_cursor
                elif self.zone != _EditZone_Bezier.EMPTY_CANVAS: # assume button
                    cursor = self._arrow_cursor
            if cursor is not self._current_override_cursor:
                tdw.set_override_cursor(cursor)
                self._current_override_cursor = cursor

    ## Redraws
    
    def _queue_draw_node(self, i):
        """Redraws a specific control node on all known view TDWs"""
        node = self.nodes[i]
        dx,dy = self.selection_motion.get_model_offset()
        
        def get_area(node_idx, nx, ny, size, area=None):
            if node_idx in self.selected_nodes:
                x, y = tdw.model_to_display(
                        nx + dx, ny + dy)
            else:
                x, y = tdw.model_to_display(nx, ny)
            x = math.floor(x)
            y = math.floor(y)
            sx = x-size-2
            sy = y-size-2
            ex = x+size+2
            ey = y+size+2
            if not area:
                return (sx, sy, ex, ey)
            else:
                return (min(area[0], sx),
                        min(area[1], sy),
                        max(area[2], ex),
                        max(area[3], ey))

        size = math.ceil(gui.style.DRAGGABLE_POINT_HANDLE_SIZE)
        for tdw in self._overlays:
            area = get_area(i, node.x, node.y, size)
            for hi in (0,1):
                if (hi == 0 and i > 0) or (hi == 1 and i <= len(self.nodes)-1):
                    area = get_area(i, 
                        node.control_handles[hi].x, node.control_handles[hi].y,
                        size, area)

            tdw.queue_draw_area(area[0], area[1], 
                    area[2] - area[0] + 1, 
                    area[3] - area[1] + 1)


    def _queue_redraw_curve(self,step = 0.05):
        """Redraws the entire curve on all known view TDWs
        :param step: rendering step of curve.
        The lower this value is,the higher quality bezier curve rendered.
        default value is for draft/editing, 
        to be decreased when render final stroke.
        """
        self._stop_task_queue_runner(complete=False)
        for tdw in self._overlays:
            model = tdw.doc
            if len(self.nodes) < 2:
                continue
            self._queue_task(self.brushwork_rollback, model)
            self._queue_task(
                self.brushwork_begin, model,
                description=_("Bezier"),
                abrupt=True,
            )
            for p0, p1, p2, p3 in _bezier_iter(self.nodes):
                self._queue_task(
                    self._draw_curve_segment,
                    model,
                    p0, p1, p2, p3, step
                )
        self._start_task_queue_runner()

    def _queue_draw_buttons(self):
        # to surpress exception
        if len(self.nodes) >= 2:
            super(BezierMode, self)._queue_draw_buttons()

    def _queue_previous_draw_buttons(self):
        """ Queue previous (current) button position to draw.
        It means erase old position buttons.
        BezierCurveMode changes button position with its 
        node selection state,so we miss calcurate it in some case.
        """

        for tdw, overlay in self._overlays.items():
            for pos in (overlay.accept_button_pos,
                         overlay.reject_button_pos):
                # FIXME duplicate code:from gui.inktool.queue_draw_buttons
                if pos is None:
                    continue
                r = gui.style.FLOATING_BUTTON_ICON_SIZE
                r += max(
                    gui.style.DROP_SHADOW_X_OFFSET,
                    gui.style.DROP_SHADOW_Y_OFFSET,
                )
                r += gui.style.DROP_SHADOW_BLUR
                x, y = pos
                tdw.queue_draw_area(x-r, y-r, 2*r+1, 2*r+1)

    def _draw_curve_segment(self, model, p0, p1, p2, p3, step):
        """Draw the curve segment between the middle two points
        :param step: rendering step of curve.
        """
        
        def get_pt(v0, v1, step):
            return v0 + ((v1-v0) * step)
        
        cur_step = 0.0
        pressure = 1.0 # for testing
        xtilt = 0.0
        ytilt = 0.0
        while cur_step <= 1.0:
            xa = get_pt(p0.x, p1.x, cur_step)
            ya = get_pt(p0.y, p1.y, cur_step)
            xb = get_pt(p1.x, p2.x, cur_step)
            yb = get_pt(p1.y, p2.y, cur_step)
            xc = get_pt(p2.x, p3.x, cur_step)
            yc = get_pt(p2.y, p3.y, cur_step)
            
            xa = get_pt(xa, xb, cur_step)
            ya = get_pt(ya, yb, cur_step)
            xb = get_pt(xb, xc, cur_step)
            yb = get_pt(yb, yc, cur_step)
            
            x = get_pt(xa, xb, cur_step)
            y = get_pt(ya, yb, cur_step)
            
            #t_abs = max(last_t_abs, t_abs)
            #dtime = t_abs - last_t_abs
            dtime = 1.0
            
            self.stroke_to(
                model, dtime, x, y, 
                lib.helpers.clamp(
                    p0.pressure + ((p3.pressure - p0.pressure) * cur_step),
                    0.0, 1.0), 
                xtilt, 
                ytilt,
                auto_split=False,
            )
                        
            cur_step += step

    ## Raw event handling (prelight & zone selection in adjust phase)
    def button_press_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False
        self._update_zone_and_target(tdw, event.x, event.y)
        self._update_current_node_index()
       #if self.phase == _Phase.ADJUST:
       #    button = event.button
       #    if (self.current_node_index is not None and 
       #            button == 1 and
       #            self.phase == _Phase.ADJUST and 
       #            event.state & self.__class__._PRESSURE_MOD_MASK == 
       #            self.__class__._PRESSURE_MOD_MASK):
       #        
       #        # Entering On-canvas Pressure Adjustment Phase!
       #        self.phase = _Phase.ADJUST_PRESSURE
       #
       #        # And do not forget,this can be a node selection.
       #        if not self.current_node_index in self.selected_nodes:
       #            # To avoid old selected nodes still lit.
       #            self._queue_draw_selected_nodes() 
       #            self._reset_selected_nodes(self.current_node_index)
       #        else:
       #            # The node is already included to self.selected_nodes
       #            pass
       #
       #        # FALLTHRU: *do* start a drag 
       #
       #    else:
       #        # Normal ADJUST Phase.
       #        
       #        if self.zone in (_EditZone_Bezier.REJECT_BUTTON,
       #                         _EditZone_Bezier.ACCEPT_BUTTON):
       #            if (button == 1 and 
       #                    event.type == Gdk.EventType.BUTTON_PRESS):
       #                self._click_info = (button, self.zone)
       #                return False
       #            # FALLTHRU: *do* allow drags to start with other buttons
       #        elif self.zone == _EditZone_Bezier.CONTROL_HANDLE:
       #            self.phase = _Phase.ADJUST_HANDLE
       #            
       #        elif self.zone == _EditZone_Bezier.EMPTY_CANVAS:
       #            if (event.state & Gdk.ModifierType.SHIFT_MASK):
       #                # selection box dragging start!!
       #                self.phase = _Phase.ADJUST_SELECTING
       #                self.selection_motion.start(
       #                        *tdw.display_to_model(event.x, event.y))
       #            else:
       #                self._start_new_capture_phase(rollback=False)
       #                assert self.phase == _Phase.INITIAL
       #
       #            # FALLTHRU: *do* start a drag
       #        else:
       #            # clicked a node.
       #
       #            if button == 1:
       #                do_reset = False
       #                if (event.state & Gdk.ModifierType.CONTROL_MASK):
       #                    # Holding CONTROL key = adding or removing a node.
       #                    # But it is done at button_release_cb for now,
       #                    pass
       #
       #                else:
       #                    # no CONTROL Key holded.
       #                    # If new solo node clicked without holding 
       #                    # CONTROL key,then reset all selected nodes.
       #
       #                    assert self.current_node_index != None
       #
       #                    do_reset = ((event.state & Gdk.ModifierType.MOD1_MASK) != 0)
       #                    do_reset |= not (self.current_node_index in self.selected_nodes)
       #
       #                if do_reset:
       #                    # To avoid old selected nodes still lit.
       #                    self._queue_draw_selected_nodes() 
       #                    self._reset_selected_nodes(self.current_node_index)
       #
       #            # FALLTHRU: *do* start a drag
       #
       #
       #elif self.phase == _Phase.INITIAL:
        print('pressing')
        if self.phase == _Phase.INITIAL:
            # XXX Not sure what to do here.
            # XXX Click to append nodes?
            # XXX  but how to stop that and enter the adjust phase?
            # XXX Click to add a 1st & 2nd (=last) node only?
            # XXX  but needs to allow a drag after the 1st one's placed.
            if self.zone == _EditZone_Bezier.CONTROL_HANDLE:
                self.phase = _Phase.ADJUST_HANDLE
            elif self.zone == _EditZone_Bezier.CONTROL_NODE:

                button = event.button
                print('coming...')
                if (self.current_node_index is not None and 
                        button == 1 and
                        event.state & self.__class__._PRESSURE_MOD_MASK == 
                        self.__class__._PRESSURE_MOD_MASK):
                    
                    # Entering On-canvas Pressure Adjustment Phase!
                    self.phase = _Phase.ADJUST_PRESSURE
                    print('pressure!!!')
            
                    # And do not forget,this can be a node selection.
                    if not self.current_node_index in self.selected_nodes:
                        # To avoid old selected nodes still lit.
                        self._queue_draw_selected_nodes() 
                        self._reset_selected_nodes(self.current_node_index)
                    else:
                        # The node is already included to self.selected_nodes
                        pass
            
                    # FALLTHRU: *do* start a drag 
                else:
                    self.phase = _Phase.MOVE_NODE

       #elif self.phase == _Phase.ADJUST_PRESSURE:
       #    # XXX Not sure what to do here.
       #    pass
        elif self.phase == _Phase.ADJUST_SELECTING:
            # XXX Not sure what to do here.
            pass
        elif self.phase in (_Phase.ADJUST_HANDLE, _Phase.INIT_HANDLE):
            pass
        else:
            raise NotImplementedError("Unrecognized zone %r", self.zone)
        # Update workaround state for evdev dropouts
        self._button_down = event.button
        # Supercall: start drags etc
        return super(InkingMode, self).button_press_cb(tdw, event) 

    def button_release_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False

       #if self.phase == _Phase.ADJUST:
       #    if self._click_info:
       #        button0, zone0 = self._click_info
       #        if event.button == button0:
       #            if self.zone == zone0:
       #                if zone0 == _EditZone_Bezier.REJECT_BUTTON:
       #                    self._start_new_capture_phase(rollback=True)
       #                    assert self.phase == _Phase.INITIAL
       #                elif zone0 == _EditZone_Bezier.ACCEPT_BUTTON:
       #                    self._start_new_capture_phase(rollback=False)
       #                    assert self.phase == _Phase.INITIAL
       #            self._click_info = None
       #            self._update_zone_and_target(tdw, event.x, event.y)
       #            self._update_current_node_index()
       #            return False
       #    else:
       #        # Clicked node and button released.
       #        # Add or Remove selected node
       #        # when control key is pressed
       #        if event.button == 1:
       #            if event.state & Gdk.ModifierType.CONTROL_MASK:
       #                tidx = self.target_node_index
       #                if tidx != None:
       #                    if not tidx in self.selected_nodes:
       #                        self.selected_nodes.append(tidx)
       #                    else:
       #                        self.selected_nodes.remove(tidx)
       #                        self.target_node_index = None
       #                        self.current_node_index = None
       #            else:
       #                # Single node click. 
       #                pass
       #
       #            ## fall throgh
       #
       #        self._update_zone_and_target(tdw, event.x, event.y)
       #
       #    # (otherwise fall through and end any current drag)
       #elif self.phase == _Phase.ADJUST_PRESSURE:
        if self.phase == _Phase.MOVE_NODE:
           #self._update_zone_and_target(tdw, event.x, event.y)
            pass

        elif self.phase == _Phase.ADJUST_PRESSURE:
            self.options_presenter.target = (self, self.current_node_index)
            InkingMode.enable_switch_actions(True)
            self.phase = _Phase.INITIAL
        elif self.phase == _Phase.ADJUST_SELECTING:
            # XXX Not sure what to do here.
            pass
        elif self.phase in (_Phase.ADJUST_HANDLE, _Phase.INIT_HANDLE):
            pass
        elif self.phase == _Phase.INITIAL:
            if self.zone == _EditZone_Bezier.REJECT_BUTTON:
                self._start_new_capture_phase(rollback=True)
            elif self.zone == _EditZone_Bezier.ACCEPT_BUTTON:
                self._queue_redraw_curve(0.01) # Redraw with hi-fidely curve
                self._start_new_capture_phase(rollback=False)
        else:
            raise NotImplementedError("Unrecognized phase %r", self.phase)
        # Update workaround state for evdev dropouts
        self._button_down = None
        # Initialize pressed position as invalid for hold-and-modify
        self._pressed_x = None
        self._pressed_y = None
        # Supercall: stop current drag
        return super(InkingMode, self).button_release_cb(tdw, event)
        
    ## Drag handling (both capture and adjust phases)
    def drag_start_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        dx, dy = tdw.display_to_model(event.x, event.y)

        self._queue_previous_draw_buttons() # To erase button,and avoid glitch

        # Basically,all sections should do fall-through.
        if self.phase == _Phase.INITIAL:

            if self.zone == _EditZone_Bezier.EMPTY_CANVAS:
                if event.state != 0:
                    # To activate some mode override
                    self._last_event_node = None
                    return super(InkingMode, self).drag_start_cb(tdw, event)
                else:
                    # New node added!
                    node = self._get_event_data(tdw, event)
                    self.nodes.append(node)
                    self._last_event_node = node
                    self.phase = _Phase.INIT_HANDLE
                    self.current_node_index=len(self.nodes)-1
                    if len(self.nodes) == 1:
                        self.current_handle_index = 1
                    else:
                        self.current_handle_index = 0

                    self._queue_draw_node(self.current_node_index)

        elif self.phase == _Phase.MOVE_NODE:
            self._node_dragged = False
            if self.target_node_index is not None:
                node = self.nodes[self.target_node_index]
                self._dragged_node_start_pos = (node.x, node.y)
                
                # Use selection_motion class as offset-information
                self.selection_motion.start(dx, dy)
        
        elif self.phase == _Phase.ADJUST_PRESSURE:
            if self.current_node_index is not None:
                node = self.nodes[self.current_node_index]
                self._pressed_pressure = node.pressure
                self._pressed_x, self._pressed_y = \
                        tdw.display_to_model(dx, dy)
        elif self.phase == _Phase.ADJUST_SELECTING:
            self.selection_motion.start(dx, dy)
            self.selection_motion.is_addition = (event.state & Gdk.ModifierType.CONTROL_MASK)
        elif self.phase == _Phase.ADJUST_HANDLE:
            self._last_event_node = self.nodes[self.target_node_index]
            pass
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)


    def drag_update_cb(self, tdw, event, dx, dy):
        self._ensure_overlay_for_tdw(tdw)
        mx, my = tdw.display_to_model(event.x, event.y)
        if self.phase == _Phase.INITIAL:
            pass
            
        elif self.phase in (_Phase.ADJUST_HANDLE, _Phase.INIT_HANDLE):
            node = self._last_event_node
            if self._last_event_node:
                self._queue_draw_node(self.current_node_index)# to erase
                handle = node.control_handles[self.current_handle_index]
                handle.x = mx
                handle.y = my
                self._queue_draw_node(self.current_node_index)
            self._queue_redraw_curve()
                
        elif self.phase == _Phase.MOVE_NODE:
            if self._dragged_node_start_pos:
                x0, y0 = self._dragged_node_start_pos
                disp_x, disp_y = tdw.model_to_display(x0, y0)
                disp_x += event.x - self.start_x
                disp_y += event.y - self.start_y
                x, y = tdw.display_to_model(disp_x, disp_y)
                self.update_node(self.target_node_index, x=x, y=y)
        
        elif self.phase == _Phase.ADJUST_PRESSURE:
            if self._pressed_pressure is not None:
                self._adjust_pressure_with_motion(mx, my)
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)

    def drag_stop_cb(self, tdw):
        self._ensure_overlay_for_tdw(tdw)
        if self.phase == _Phase.INITIAL:
            node = self._last_event_node
            self._reset_adjust_data()
            self._queue_redraw_all_nodes()
            self._queue_redraw_curve()
            if len(self.nodes) > 1:
                self._queue_draw_buttons()
                
            self.phase = _Phase.INITIAL
            
        elif self.phase in (_Phase.ADJUST_HANDLE, _Phase.INIT_HANDLE):
            node = self._last_event_node
      
            if self.phase == _Phase.INIT_HANDLE and len(self.nodes) > 1:
                node.control_handles[1].x = node.x - (node.control_handles[0].x - node.x)
                node.control_handles[1].y = node.y - (node.control_handles[0].y - node.y)

           #self._reset_adjust_data()
            self._queue_redraw_all_nodes()
            self._queue_redraw_curve()
            if len(self.nodes) > 1:
                self._queue_draw_buttons()
                
            self.phase = _Phase.INITIAL
        elif self.phase == _Phase.MOVE_NODE:
            self._dragged_node_start_pos = None
            self._queue_redraw_curve()
            self._queue_draw_buttons()
            self.phase = _Phase.INITIAL
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)

    ## Interrogating events

    def _get_event_data(self, tdw, event):
        x, y = tdw.display_to_model(event.x, event.y)
        xtilt, ytilt = self._get_event_tilt(tdw, event)
        return _Node_Bezier(
            x=x, y=y,
            pressure=lib.helpers.clamp(
                    self._get_event_pressure(event),
                    0.3, 1.0), 
            xtilt=xtilt, ytilt=ytilt
            )
        

    ## Node editing

    

    @property
    def options_presenter(self):
        """MVP presenter object for the node editor panel"""
        cls = self.__class__
        if cls._OPTIONS_PRESENTER is None:
            cls._OPTIONS_PRESENTER = OptionsPresenter_Bezier()
        return cls._OPTIONS_PRESENTER






    def insert_node(self, i):
        """Insert a node, and issue redraws & updates"""
        assert self.can_insert_node(i), "Can't insert back of the endpoint"
        # Redraw old locations of things while the node still exists
        self._queue_draw_buttons()
        self._queue_draw_node(i)
        # Create the new node
        cn = self.nodes[i]
        nn = self.nodes[i+1]

        newnode = _Node_Bezier(
            x=(cn.x + nn.x)/2.0, y=(cn.y + nn.y) / 2.0,
            pressure=(cn.pressure + nn.pressure) / 2.0,
            xtilt=(cn.xtilt + nn.xtilt) / 2.0, 
            ytilt=(cn.ytilt + nn.ytilt) / 2.0           
        )
        self.nodes.insert(i+1,newnode)

        # Issue redraws for the changed on-canvas elements
        self._queue_redraw_curve()
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()


class OverlayBezier (Overlay):
    """Overlay for an BezierMode's adjustable points"""

    def __init__(self, mode, tdw):
        super(OverlayBezier, self).__init__(mode, tdw)
        
    def update_button_positions(self):
        """Recalculates the positions of the mode's buttons."""
        # FIXME mostly copied from inktool.Overlay.update_button_positions
        # difference is for-loop of nodes
        nodes = self._inkmode.nodes
        num_nodes = len(nodes)
        if num_nodes == 0:
            self.reject_button_pos = None
            self.accept_button_pos = None
            return

        button_radius = gui.style.FLOATING_BUTTON_RADIUS
        margin = 1.5 * button_radius
        alloc = self._tdw.get_allocation()
        view_x0, view_y0 = alloc.x, alloc.y
        view_x1, view_y1 = view_x0+alloc.width, view_y0+alloc.height

        # Force-directed layout: "wandering nodes" for the buttons'
        # eventual positions, moving around a constellation of "fixed"
        # points corresponding to the nodes the user manipulates.
        fixed = []

        for i, node in enumerate(nodes):
            x, y = self._tdw.model_to_display(node.x, node.y)
            fixed.append(_LayoutNode(x, y))
            # ADDED PORTION:to avoid overwrap on control handles,
            # treat control handles as nodes,when it is visible.
            if i == self._inkmode.current_node_index:
                for t in (0,1):
                    fixed.append(_LayoutNode(node.control_handles[t].x, node.control_handles[t].y))

        # The reject and accept buttons are connected to different nodes
        # in the stroke by virtual springs.
        stroke_end_i = len(fixed)-1
        stroke_start_i = 0
        stroke_last_quarter_i = int(stroke_end_i * 3.0 // 4.0)
        assert stroke_last_quarter_i < stroke_end_i
        reject_anchor_i = stroke_start_i
        accept_anchor_i = stroke_end_i

        # Classify the stroke direction as a unit vector
        stroke_tail = (
            fixed[stroke_end_i].x - fixed[stroke_last_quarter_i].x,
            fixed[stroke_end_i].y - fixed[stroke_last_quarter_i].y,
        )
        stroke_tail_len = math.hypot(*stroke_tail)
        if stroke_tail_len <= 0:
            stroke_tail = (0., 1.)
        else:
            stroke_tail = tuple(c/stroke_tail_len for c in stroke_tail)

        # Initial positions.
        accept_button = _LayoutNode(
            fixed[accept_anchor_i].x + stroke_tail[0]*margin,
            fixed[accept_anchor_i].y + stroke_tail[1]*margin,
        )
        reject_button = _LayoutNode(
            fixed[reject_anchor_i].x - stroke_tail[0]*margin,
            fixed[reject_anchor_i].y - stroke_tail[1]*margin,
        )

        # Constraint boxes. They mustn't share corners.
        # Natural hand strokes are often downwards,
        # so let the reject button to go above the accept button.
        reject_button_bbox = (
            view_x0+margin, view_x1-margin,
            view_y0+margin, view_y1-2.666*margin,
        )
        accept_button_bbox = (
            view_x0+margin, view_x1-margin,
            view_y0+2.666*margin, view_y1-margin,
        )

        # Force-update constants
        k_repel = -25.0
        k_attract = 0.05

        # Let the buttons bounce around until they've settled.
        for iter_i in xrange(100):
            accept_button \
                .add_forces_inverse_square(fixed, k=k_repel) \
                .add_forces_inverse_square([reject_button], k=k_repel) \
                .add_forces_linear([fixed[accept_anchor_i]], k=k_attract)
            reject_button \
                .add_forces_inverse_square(fixed, k=k_repel) \
                .add_forces_inverse_square([accept_button], k=k_repel) \
                .add_forces_linear([fixed[reject_anchor_i]], k=k_attract)
            reject_button \
                .update_position() \
                .constrain_position(*reject_button_bbox)
            accept_button \
                .update_position() \
                .constrain_position(*accept_button_bbox)
            settled = [(p.speed<0.5) for p in [accept_button, reject_button]]
            if all(settled):
                break
        self.accept_button_pos = accept_button.x, accept_button.y
        self.reject_button_pos = reject_button.x, reject_button.y

    
    def paint(self, cr):
        """Draw adjustable nodes to the screen"""
        # Control nodes
        mode = self._inkmode
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        alloc = self._tdw.get_allocation()
        dx, dy = mode.selection_motion.get_display_offset(self._tdw)
        for i, node, x, y in self._get_onscreen_nodes():
            color = gui.style.EDITABLE_ITEM_COLOR
            if mode.phase in (_Phase.INITIAL, _Phase.MOVE_NODE, _Phase.ADJUST_HANDLE, _Phase.INIT_HANDLE):
                if i == mode.current_node_index:
                    color = gui.style.ACTIVE_ITEM_COLOR
                    x += dx
                    y += dy
              
                    # Drawing control handle
                    cr.save()
                   #cr.move_to(x, y)
                    cr.set_source_rgb(0,0,1)
                    cr.set_line_width(1)
                    for hi in (0,1):                        
                        if ((hi == 0 and i > 0) or
                                (hi == 1 and i <= len(self._inkmode.nodes)-1)): 
                            ch = node.control_handles[hi]
                            hx, hy = self._tdw.model_to_display(ch.x, ch.y)
                            hx += dx
                            hy += dy
                            gui.drawutils.render_square_floating_color_chip(
                                cr, hx, hy,
                                color, radius, 
                                fill=(hi==self._inkmode.current_handle_index)) 
                            cr.move_to(x, y)
                            cr.line_to(hx, hy)
                            cr.stroke()
                    cr.restore()
                              
                elif i == mode.target_node_index:
                    color = gui.style.PRELIT_ITEM_COLOR
                    x += dx
                    y += dy
      
            gui.drawutils.render_round_floating_color_chip(
                cr=cr, x=x, y=y,
                color=color,
                radius=radius,
            )
            
    
                
        # Buttons
        if not mode.in_drag and len(self._inkmode.nodes) > 1:
            self.update_button_positions()
            radius = gui.style.FLOATING_BUTTON_RADIUS
            button_info = [
                (
                    "mypaint-ok-symbolic",
                    self.accept_button_pos,
                    _EditZone_Bezier.ACCEPT_BUTTON,
                ),
                (
                    "mypaint-trash-symbolic",
                    self.reject_button_pos,
                    _EditZone_Bezier.REJECT_BUTTON,
                ),
            ]
            for icon_name, pos, zone in button_info:
                if pos is None:
                    continue
                x, y = pos
                if mode.zone == zone:
                    color = gui.style.ACTIVE_ITEM_COLOR
                else:
                    color = gui.style.EDITABLE_ITEM_COLOR
                icon_pixbuf = self._get_button_pixbuf(icon_name)
                gui.drawutils.render_round_floating_button(
                    cr=cr, x=x, y=y,
                    color=color,
                    pixbuf=icon_pixbuf,
                    radius=radius,
                )
        

                
                
                
                
class _LayoutNode (object):
    """Vertex/point for the button layout algorithm."""

    def __init__(self, x, y, force=(0.,0.), velocity=(0.,0.)):
        self.x = float(x)
        self.y = float(y)
        self.force = tuple(float(c) for c in force[:2])
        self.velocity = tuple(float(c) for c in velocity[:2])

    def __repr__(self):
        return "_LayoutNode(x=%r, y=%r, force=%r, velocity=%r)" % (
            self.x, self.y, self.force, self.velocity,
        )

    @property
    def pos(self):
        return (self.x, self.y)

    @property
    def speed(self):
        return math.hypot(*self.velocity)

    def add_forces_inverse_square(self, others, k=20.0):
        """Adds inverse-square components to the effective force.

        :param sequence others: _LayoutNodes affecting this one
        :param float k: scaling factor
        :returns: self

        The forces applied are proportional to k, and inversely
        proportional to the square of the distances. Examples:
        gravity, electrostatic repulsion.

        With the default arguments, the added force components are
        attractive. Use negative k to simulate repulsive forces.

        """
        fx, fy = self.force
        for other in others:
            if other is self:
                continue
            rsquared = (self.x-other.x)**2 + (self.y-other.y)**2
            if rsquared == 0:
                continue
            else:
                fx += k * (other.x - self.x) / rsquared
                fy += k * (other.y - self.y) / rsquared
        self.force = (fx, fy)
        return self

    def add_forces_linear(self, others, k=0.05):
        """Adds linear components to the total effective force.

        :param sequence others: _LayoutNodes affecting this one
        :param float k: scaling factor
        :returns: self

        The forces applied are proportional to k, and to the distance.
        Example: springs.

        With the default arguments, the added force components are
        attractive. Use negative k to simulate repulsive forces.

        """
        fx, fy = self.force
        for other in others:
            if other is self:
                continue
            fx += k * (other.x - self.x)
            fy += k * (other.y - self.y)
        self.force = (fx, fy)
        return self

    def update_position(self, damping=0.85):
        """Updates velocity & position from total force, then resets it.

        :param float damping: Damping factor for velocity/speed.
        :returns: self

        Calling this method should be done just once per iteration,
        after all the force components have been added in. The effective
        force is reset to zero after calling this method.

        """
        fx, fy = self.force
        self.force = (0., 0.)
        vx, vy = self.velocity
        vx = (vx + fx) * damping
        vy = (vy + fy) * damping
        self.velocity = (vx, vy)
        self.x += vx
        self.y += vy
        return self

    def constrain_position(self, x0, x1, y0, y1):
        vx, vy = self.velocity
        if self.x < x0:
            self.x = x0
            vx = 0
        elif self.x > x1:
            self.x = x1
            vx = 0
        if self.y < y0:
            self.y = y0
            vy = 0
        elif self.y > y1:
            self.y = y1
            vy = 0
        self.velocity = (vx, vy)
        return self


class OptionsPresenter_Bezier (OptionsPresenter):
    """Presents UI for directly editing point values etc."""

    def __init__(self):
        super(OptionsPresenter_Bezier, self).__init__()

    def _ensure_ui_populated(self):
        super(OptionsPresenter_Bezier, self)._ensure_ui_populated()
        if self._options_grid is not None:
            return

  
