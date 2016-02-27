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
    
    
    
class _EditZone_Bezier:
    """Enumeration of what the pointer is on in the ADJUST phase"""
    EMPTY_CANVAS = 0  #: Nothing, empty space
    CONTROL_NODE = 1  #: Any control node; see target_node_index
    REJECT_BUTTON = 2  #: On-canvas button that abandons the current line
    ACCEPT_BUTTON = 3  #: On-canvas button that commits the current line
    CONTROL_HANDLE = 4 #: Control handle of bezier

class _Phase:
    """Enumeration of the states that an BezierCurveMode can be in"""
    CAPTURE = 0
    ADJUST = 1
    ADJUST_PRESSURE = 2
    ADJUST_SELECTING = 3
    CREATE_NODE = 4
    ADJUST_HANDLE = 5
    
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
        if self.phase == _Phase.CAPTURE:
            return self._crosshair_cursor
        elif self.phase == _Phase.ADJUST:
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

    def enter(self, doc, **kwds):
        """Enters the mode: called by `ModeStack.push()` etc."""
        super(BezierMode, self).enter(doc, **kwds)

        self._arrow_cursor = self.doc.app.cursors.get_action_cursor(
            self.ACTION_NAME,
            gui.cursor.Name.ARROW,
        )
        self._crosshair_cursor = self.doc.app.cursors.get_action_cursor(
            self.ACTION_NAME,
            gui.cursor.Name.CROSSHAIR_OPEN_PRECISE,
        )
        self._cursor_move_nw_se = self.doc.app.cursors.get_action_cursor(
            self.ACTION_NAME,
            gui.cursor.Name.MOVE_NORTHWEST_OR_SOUTHEAST,
        )

    def _ensure_overlay_for_tdw(self, tdw):
        overlay = self._overlays.get(tdw)
        if not overlay:
            overlay = OverlayBezier(self, tdw)
            tdw.display_overlays.append(overlay)
            self._overlays[tdw] = overlay
        return overlay
        
    def _update_zone_and_target(self, tdw, x, y):
        """Update the zone and target node under a cursor position"""
        self._ensure_overlay_for_tdw(tdw)
        new_zone = _EditZone_Bezier.EMPTY_CANVAS
        if not self.in_drag:
            if self.phase == _Phase.ADJUST or self.phase == _Phase.CAPTURE:
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
                        
                   

        elif self.phase == _Phase.ADJUST_PRESSURE:
            # Always control node,in pressure editing.
            new_zone = _EditZone_Bezier.CONTROL_NODE 

        # Update the zone, and assume any change implies a button state
        # change as well (for now...)
        if self.zone != new_zone:
            self.zone = new_zone
            self._ensure_overlay_for_tdw(tdw)
            if len(self.nodes) > 1:
                self._queue_draw_buttons()
        # Update the "real" inactive cursor too:
        if not self.in_drag:
            cursor = None
            if self.phase in (_Phase.CAPTURE, _Phase.ADJUST, _Phase.ADJUST_PRESSURE):
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
        
        def do_queue_area(i, nx, ny):
            if i in self.selected_nodes:
                x, y = tdw.model_to_display(
                        nx + dx, ny + dy)
            else:
                x, y = tdw.model_to_display(nx, ny)
            x = math.floor(x)
            y = math.floor(y)
            size = math.ceil(gui.style.DRAGGABLE_POINT_HANDLE_SIZE * 2)
            tdw.queue_draw_area(x-size, y-size, size*2+1, size*2+1)
                    
        for tdw in self._overlays:
            do_queue_area(i, node.x, node.y)
            if i > 0:
                do_queue_area(i, 
                    node.control_handles[0].x, node.control_handles[0].y)
            if i < len(self.nodes)-1:
                do_queue_area(i, 
                    node.control_handles[1].x, node.control_handles[1].y)                              

    def _queue_redraw_curve(self,step = 0.1):
        """Redraws the entire curve on all known view TDWs"""
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

    def _draw_curve_segment(self, model, p0, p1, p2, p3, step):
        """Draw the curve segment between the middle two points"""
        
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
                model, dtime, x, y, pressure, xtilt, ytilt,
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
        if self.phase == _Phase.ADJUST:
            button = event.button
            if (self.current_node_index is not None and 
                    button == 1 and
                    self.phase == _Phase.ADJUST and 
                    event.state & self.__class__._PRESSURE_MOD_MASK == 
                    self.__class__._PRESSURE_MOD_MASK):
                
                # Entering On-canvas Pressure Adjustment Phase!
                self.phase = _Phase.ADJUST_PRESSURE

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
                # Normal ADJUST Phase.
                
                if self.zone in (_EditZone.REJECT_BUTTON,
                                 _EditZone.ACCEPT_BUTTON):
                    if (button == 1 and 
                            event.type == Gdk.EventType.BUTTON_PRESS):
                        self._click_info = (button, self.zone)
                        return False
                    # FALLTHRU: *do* allow drags to start with other buttons
                elif self.zone == _EditZone.EMPTY_CANVAS:
                    if (event.state & Gdk.ModifierType.SHIFT_MASK):
                        # selection box dragging start!!
                        self.phase = _Phase.ADJUST_SELECTING
                        self.selection_motion.start(
                                *tdw.display_to_model(event.x, event.y))
                    else:
                        self._start_new_capture_phase(rollback=False)
                        assert self.phase == _Phase.CAPTURE

                    # FALLTHRU: *do* start a drag
                else:
                    # clicked a node.

                    if button == 1:
                        do_reset = False
                        if (event.state & Gdk.ModifierType.CONTROL_MASK):
                            # Holding CONTROL key = adding or removing a node.
                            # But it is done at button_release_cb for now,
                            pass

                        else:
                            # no CONTROL Key holded.
                            # If new solo node clicked without holding 
                            # CONTROL key,then reset all selected nodes.

                            assert self.current_node_index != None

                            do_reset = ((event.state & Gdk.ModifierType.MOD1_MASK) != 0)
                            do_reset |= not (self.current_node_index in self.selected_nodes)

                        if do_reset:
                            # To avoid old selected nodes still lit.
                            self._queue_draw_selected_nodes() 
                            self._reset_selected_nodes(self.current_node_index)

                    # FALLTHRU: *do* start a drag


        elif self.phase == _Phase.CAPTURE:
            # XXX Not sure what to do here.
            # XXX Click to append nodes?
            # XXX  but how to stop that and enter the adjust phase?
            # XXX Click to add a 1st & 2nd (=last) node only?
            # XXX  but needs to allow a drag after the 1st one's placed.
            pass    
        elif self.phase == _Phase.ADJUST_PRESSURE:
            # XXX Not sure what to do here.
            pass
        elif self.phase == _Phase.ADJUST_SELECTING:
            # XXX Not sure what to do here.
            pass
        elif self.phase == _Phase.ADJUST_HANDLE:
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

        if self.phase == _Phase.ADJUST:
            if self._click_info:
                button0, zone0 = self._click_info
                if event.button == button0:
                    if self.zone == zone0:
                        if zone0 == _EditZone.REJECT_BUTTON:
                            self._start_new_capture_phase(rollback=True)
                            assert self.phase == _Phase.CAPTURE
                        elif zone0 == _EditZone.ACCEPT_BUTTON:
                            self._start_new_capture_phase(rollback=False)
                            assert self.phase == _Phase.CAPTURE
                    self._click_info = None
                    self._update_zone_and_target(tdw, event.x, event.y)
                    self._update_current_node_index()
                    return False
            else:
                # Clicked node and button released.
                # Add or Remove selected node
                # when control key is pressed
                if event.button == 1:
                    if event.state & Gdk.ModifierType.CONTROL_MASK:
                        tidx = self.target_node_index
                        if tidx != None:
                            if not tidx in self.selected_nodes:
                                self.selected_nodes.append(tidx)
                            else:
                                self.selected_nodes.remove(tidx)
                                self.target_node_index = None
                                self.current_node_index = None
                    else:
                        # Single node click. 
                        pass

                    ## fall throgh

                self._update_zone_and_target(tdw, event.x, event.y)

            # (otherwise fall through and end any current drag)
        elif self.phase == _Phase.ADJUST_PRESSURE:
            self.options_presenter.target = (self, self.current_node_index)
        elif self.phase == _Phase.ADJUST_SELECTING:
            # XXX Not sure what to do here.
            pass
        elif self.phase in (_Phase.CAPTURE, _Phase.CREATE_NODE):
            # Update options_presenter when capture phase end
            self.options_presenter.target = (self, None)
        elif self.phase == _Phase.ADJUST_HANDLE:
            pass
        else:
            raise NotImplementedError("Unrecognized zone %r", self.zone)
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
        if self.phase == _Phase.CAPTURE:
            #self._reset_nodes()
            #self._reset_capture_data()
            #self._reset_adjust_data()
            #self.zone == _EditZone_Bezier.CONTROL_NODE
            if self.zone == _EditZone_Bezier.CONTROL_NODE:
                self._last_event_node = self.nodes[self.current_node_index]
            elif self.zone == _EditZone_Bezier.EMPTY_CANVAS:
                if event.state != 0:
                    # To activate some mode override
                    self._last_event_node = None
                    return super(InkingMode, self).drag_start_cb(tdw, event)
                else:
                    self.phase = _Phase.CREATE_NODE
                    node = self._get_event_data(tdw, event)
                    self.nodes.append(node)
                    self._queue_draw_node(0)
                    self._last_event_node = node

        elif self.phase == _Phase.ADJUST:
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
            self._queue_draw_buttons() # To erase button!
        elif self.phase == _Phase.ADJUST_HANDLE:
            pass
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)

    def drag_update_cb(self, tdw, event, dx, dy):
        self._ensure_overlay_for_tdw(tdw)
        mx, my = tdw.display_to_model(event.x, event.y)
        if self.phase in (_Phase.CREATE_NODE, _Phase.CAPTURE):
            node = self._last_event_node
            if self._last_event_node:
                self._queue_draw_node(len(self.nodes)-1) # to erase
                node.x = mx
                node.y = my

                
     
            
            self._queue_draw_node(len(self.nodes)-1)
            self._queue_redraw_curve()
                
            #node = self._get_event_data(tdw, event)
            #evdata = (event.x, event.y, event.time)
            #if not self._last_node_evdata: # e.g. after an undo while dragging
                #append_node = True
            #elif evdata == self._last_node_evdata:
                #logger.debug(
                    #"Capture: ignored successive events "
                    #"with identical position and time: %r",
                    #evdata,
                #)
                #append_node = False
            #else:
                #dx = event.x - self._last_node_evdata[0]
                #dy = event.y - self._last_node_evdata[1]
                #dist = math.hypot(dy, dx)
                #dt = event.time - self._last_node_evdata[2]
                #max_dist = self.MAX_INTERNODE_DISTANCE_MIDDLE
                #if len(self.nodes) < 2:
                    #max_dist = self.MAX_INTERNODE_DISTANCE_ENDS
                #append_node = (
                    #dist > max_dist and
                    #dt > self.MAX_INTERNODE_TIME
                #)
            #if append_node:
                #self.nodes.append(node)
                #self._queue_draw_node(len(self.nodes)-1)
                #self._queue_redraw_curve()
                #self._last_node_evdata = evdata
            #self._last_event_node = node
            pass
        elif self.phase == _Phase.ADJUST:
            if self._dragged_node_start_pos:
                x0, y0 = self._dragged_node_start_pos
                disp_x, disp_y = tdw.model_to_display(x0, y0)
                disp_x += event.x - self.start_x
                disp_y += event.y - self.start_y
                x, y = tdw.display_to_model(disp_x, disp_y)
                self.update_node(self.target_node_index, x=x, y=y)
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)

    def drag_stop_cb(self, tdw):
        self._ensure_overlay_for_tdw(tdw)
        if self.phase in (_Phase.CAPTURE, _Phase.CREATE_NODE):
            node = self._last_event_node
            
            if self.phase == _Phase.CREATE_NODE:
               
                if len(self.nodes) > 1:
                    pn = self.nodes[-2]
                    hx = (node.x - pn.x) / 2.0
                    hy = (node.y - pn.y) / 2.0
                    node.control_handles[0].x = node.x - hy
                    node.control_handles[0].y = node.y + hx
                
                    pn.control_handles[1].x = node.x + hy
                    pn.control_handles[1].y = node.y - hx
                    
            self._reset_adjust_data()
            self._queue_redraw_all_nodes()
            self._queue_redraw_curve()
            if len(self.nodes) > 1:
                self._queue_draw_buttons()
                
            self.phase = _Phase.CAPTURE
        elif self.phase == _Phase.ADJUST:
            self._dragged_node_start_pos = None
            self._queue_redraw_curve()
            self._queue_draw_buttons()
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)

    ## Interrogating events

    def _get_event_data(self, tdw, event):
        x, y = tdw.display_to_model(event.x, event.y)
        xtilt, ytilt = self._get_event_tilt(tdw, event)
        return _Node_Bezier(
            x=x, y=y,
            pressure=self._get_event_pressure(event),
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
        
    
    def paint(self, cr):
        """Draw adjustable nodes to the screen"""
        # Control nodes
        mode = self._inkmode
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        alloc = self._tdw.get_allocation()
        for i, node, x, y in self._get_onscreen_nodes():
            color = gui.style.EDITABLE_ITEM_COLOR
            if mode.phase in (_Phase.CAPTURE, _Phase.CREATE_NODE, _Phase.ADJUST):
                if i == mode.current_node_index:
                    color = gui.style.ACTIVE_ITEM_COLOR
              
                    # Drawing control handle
                    cr.save()
                    cr.move_to(x, y)
                    cr.set_source_rgb(0,0,1)
                    cr.set_line_width(1)
                    for hi in (0,1):                        
                        if ((hi == 0 and i > 0) or
                                (hi == 1 and i < len(self._inkmode.nodes)-1)):
                            ch = node.control_handles[hi]
                            dx, dy = self._tdw.model_to_display(ch.x, ch.y)
                            gui.drawutils.render_square_floating_color_chip(
                                cr, dx, dy,
                                color, radius, 
                                fill=(hi==self._inkmode.current_handle_index)) 
                            cr.move_to(x, y)
                            cr.line_to(dx, dy)
                    cr.restore()
                              
                elif i == mode.target_node_index:
                    color = gui.style.PRELIT_ITEM_COLOR

      
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

  
