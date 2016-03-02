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
from inktool import _LayoutNode, _Phase, _EditZone

## Class defs

class _Control_Handle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
class _Node_Bezier (object):
    """Recorded control point.
    In bezier curve,nodes are frequently rewritten
    and each nodes own control handles.
    each control handles need to be adjusted,when
    node moves.
    so this class is object, not a namedtuple.

    _Node_Bezier thave the following 6 fields, in order

    * x, y: model coords, float
    * pressure: float in [0.0, 1.0]
    * xtilt, ytilt: float in [-1.0, 1.0]
    * control_handle: two _Control_Handle objects.
    """
    
            
            
    def __init__(self,x,y,pressure=None,xtilt=0.0,ytilt=0.0,
            control_handles=None,curve=True):
        self.x = x
        self.y = y
        self.pressure = pressure
        self.xtilt = xtilt
        self.ytilt = ytilt
        self.time = 0
        if control_handles:
            self._control_handles = control_handles
        else:
            self._control_handles = (_Control_Handle(x, y), _Control_Handle(x, y))

        self._curve = curve

    @property
    def curve(self):
        return self._curve

    @curve.setter
    def curve(self, flag):
        self._curve = flag
        if self._curve:
            self.set_control_handle(0, 
                    self._control_handles[0].x,
                    self._control_handles[0].y)

    def set_control_handle(self, idx, x, y):
        """Use this method to set control handle.
        This method refers self._curve flag,
        and if it is True,automatically make handles
        as symmetry = 'Curved bezier control point'
        """

        dx = x - self.x
        dy = y - self.y
        self._control_handles[idx].x = x 
        self._control_handles[idx].y = y

        if self._curve:
            tidx = (idx + 1) % 2
            self._control_handles[tidx].x = self.x - dx
            self._control_handles[tidx].y = self.y - dy

    def get_control_handle(self,idx):
        assert 0 <= idx <= 1
        return self._control_handles[idx]

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
            self._control_handles[i].x = nx + (self._control_handles[i].x - ox)
            self._control_handles[i].y = ny + (self._control_handles[i].y - oy)

    def move(self,x, y):
        """Move this node to (x,y).
        :param x: destination x position,in model
        :param y: destination y position,in model

        Use this method to move control points simultaneously.
        """
        self._refresh_control_handles(self.x, self.y, x, y)
        self.x = x
        self.y = y

    def copy(self, node, dx=0, dy=0):
        """Copy all information to node,with (dx,dy) offset.
        :param node: the _Node_Bezier object copy to
        :param dx:offset x,in model
        :param dy:offset y,in model
        """
        node.x = self.x + dx
        node.y = self.y + dy
        node.pressure = self.pressure
        node.xtilt = self.xtilt
        node.ytilt = self.ytilt
        node.time = self.time
        node._curve = self._curve
        for i in (0,1):
            node._control_handles[i].x = self._control_handles[i].x + dx
            node._control_handles[i].y = self._control_handles[i].y + dy
    
    
    
class _EditZone_Bezier(_EditZone):
    """Enumeration of what the pointer is on in the ADJUST phase"""
    CONTROL_HANDLE = 104 #: Control handle of bezier

    # _EditZone-enum also used,they are defined at gui.inktool._EditZone
   #EMPTY_CANVAS   #: Nothing, empty space
   #CONTROL_NODE   #: Any control node; see target_node_index
   #REJECT_BUTTON  #: On-canvas button that abandons the current line
   #ACCEPT_BUTTON  #: On-canvas button that commits the current line

class _PhaseBezier(_Phase):
    """Enumeration of the states that an BezierCurveMode can be in"""
    INITIAL = _Phase.CAPTURE     #: Initial Phase,creating node.
    CREATE_PATH = _Phase.ADJUST  #: Main Phase.creating path(adding node)
                                 # THIS MEMBER MUST SAME AS _Phase.ADJUST
                                 # because for InkingMode.scroll_cb()
    MOVE_NODE = 100         #: Moving node(s)
    ADJUST_HANDLE = 103     #: change control-handle position
    INIT_HANDLE = 104       #: initialize control handle,right after create a node
    PLACE_NODE = 105        #: place a new node into clicked position on current
                            # stroke,when you click with holding CTRL key

   # ADJUST_PRESSURE
   # ADJUST_SELECTING
   # are also used,it is defined at gui.inktool._Phase


    
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
        cint[1] = cn.get_control_handle(1)
        cint[2] = nn.get_control_handle(0)
        cint[3] = nn
        yield cint

def _bezier_iter_offset(seq, selected, offset):
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
    dx, dy = offset
    bidx = 0
    def copy_to_buffered_node(bidx,node):
        node.copy(_bezier_iter_offset.nodebuf[bidx], dx, dy)
        return ((bidx+1) % 2, _bezier_iter_offset.nodebuf[bidx])

    cn = seq[0]
    if 0 in selected:
        bidx, cn = copy_to_buffered_node(bidx, cn)

    for idx,nn in enumerate(seq[1:]):
        if idx+1 in selected:
            bidx, nn = copy_to_buffered_node(bidx, nn)

        cint[0] = cn
        cint[1] = cn.get_control_handle(1)
        cint[2] = nn.get_control_handle(0)
        cint[3] = nn
        yield cint
        cn = nn
_bezier_iter_offset.nodebuf = (_Node_Bezier(0,0), _Node_Bezier(0,0))

def _get_bezier_pt(v0, v1, step):
    return v0 + ((v1-v0) * step)

def _get_bezier_segment(p0, p1, p2, p3, step):

    xa = _get_bezier_pt(p0.x, p1.x, step)
    ya = _get_bezier_pt(p0.y, p1.y, step)
    xb = _get_bezier_pt(p1.x, p2.x, step)
    yb = _get_bezier_pt(p1.y, p2.y, step)
    xc = _get_bezier_pt(p2.x, p3.x, step)
    yc = _get_bezier_pt(p2.y, p3.y, step)
    
    xa = _get_bezier_pt(xa, xb, step)
    ya = _get_bezier_pt(ya, yb, step)
    xb = _get_bezier_pt(xb, xc, step)
    yb = _get_bezier_pt(yb, yc, step)
    
    return (_get_bezier_pt(xa, xb, step),
            _get_bezier_pt(ya, yb, step))

            

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
        if self.phase in (_PhaseBezier.INITIAL, _PhaseBezier.CREATE_PATH):
            if self.zone == _EditZone_Bezier.CONTROL_NODE:
                return self._crosshair_cursor
        elif self.phase == _PhaseBezier.MOVE_NODE:
            if self.zone == _EditZone_Bezier.CONTROL_NODE:
                return self._crosshair_cursor
            elif self.zone != _EditZone_Bezier.EMPTY_CANVAS: # assume button
                return self._arrow_cursor

        elif self.phase == _PhaseBezier.ADJUST_PRESSURE:
            if self.zone == _EditZone_Bezier.CONTROL_NODE:
                return self._cursor_move_nw_se

        elif self.phase == _PhaseBezier.ADJUST_SELECTING:
            return self._crosshair_cursor
        return None  

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, new_phase):
        if new_phase == _PhaseBezier.INITIAL:
            InkingMode.enable_switch_actions(True)
        else:
            InkingMode.enable_switch_actions(False)
        self._phase = new_phase

    ## Class config vars



    ## Other class vars

    _OPTIONS_PRESENTER = None   #: Options presenter singleton

    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):
        super(BezierMode, self).__init__(**kwargs)

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

    ## Update inner states related methods

    def _update_current_node_index(self):
        """Updates current_node_index from target_node_index & redraw"""
        new_index = self.target_node_index
        old_index = self.current_node_index
        if new_index == old_index:
            return

        # In BezierMode class,changing current_node_index
        # might cause re-positioning of buttons.
        # so we need to queue drawing button.
        self._queue_draw_buttons() 

        self.current_node_index = new_index
        self.current_node_changed(new_index)
        self.options_presenter.target = (self, new_index)
        for i in (old_index, new_index):
            if i is not None:
                self._queue_draw_node(i)

        self._queue_draw_buttons()
        
    def _update_zone_and_target(self, tdw, x, y):
        """Update the zone and target node under a cursor position"""
        ## FIXME mostly copied from inktool.py
        ## the differences are 'control handle processing' and
        ## 'cursor changing' and, queuing buttons draw 
        ## to follow current_node_index
        self._ensure_overlay_for_tdw(tdw)
        new_zone = _EditZone_Bezier.EMPTY_CANVAS
        if not self.in_drag:
            if self.phase in (_PhaseBezier.MOVE_NODE, _PhaseBezier.INITIAL, _PhaseBezier.CREATE_PATH):
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
                
                ## CHANGED CODES for beziertool:
                # New target node is not hit.
                # But, pointer might hit control handles
                if (new_target_node_index is None and 
                        self.current_node_index is not None):
                    c_node = self.nodes[self.current_node_index]
                    self.current_handle_index = None
                    for i in (0,1):
                        handle = c_node.get_control_handle(i)
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


        elif self.phase == _PhaseBezier.ADJUST_PRESSURE:
            # Always control node,in pressure editing.
            new_zone = _EditZone_Bezier.CONTROL_NODE 

        # Update the zone, and assume any change implies a button state
        # change as well (for now...)
        if self.zone != new_zone:
            self.zone = new_zone
            self._ensure_overlay_for_tdw(tdw)
            if len(self.nodes) > 1:
                self._queue_previous_draw_buttons()

        # Update the "real" inactive cursor too:
        # these codes also a little changed from inktool.
        if not self.in_drag:
            cursor = None
            if self.phase in (_PhaseBezier.INITIAL, _PhaseBezier.CREATE_PATH,
                    _PhaseBezier.MOVE_NODE, _PhaseBezier.ADJUST_PRESSURE):
                if self.zone == _EditZone_Bezier.CONTROL_NODE:
                    cursor = self._crosshair_cursor
                elif self.zone != _EditZone_Bezier.EMPTY_CANVAS: # assume button
                    cursor = self._arrow_cursor
            if cursor is not self._current_override_cursor:
                tdw.set_override_cursor(cursor)
                self._current_override_cursor = cursor

    def _detect_on_stroke(self, x, y, allow_distance = 4.0):
        """ detect the assigned coordinate is on stroke or not
        :param x: cursor x position in MODEL coord
        :param y: cursor y position in MODEL coord
        """

        for i,cn in enumerate(self.nodes[:-1]):
            nn = self.nodes[i+1]
            sx = min(cn.x, nn.x)
            ex = max(cn.x, nn.x)
            sy = min(cn.y, nn.y)
            ey = max(cn.y, nn.y)
            for t in (0,1):
                cx = cn.get_control_handle(t).x
                nx = nn.get_control_handle(t).x
                sx = min(min(sx, cx), nx)
                ex = max(max(sx, cx), nx)
                cy = cn.get_control_handle(t).y
                ny = nn.get_control_handle(t).y
                sy = min(min(sy, cy), ny)
                ey = max(max(sy, cy), ny)

            if sx <= x <= ex and sy <= y <= ey:

                def detect_step(start_step, end_step, increase_step):
                    ox, oy = _get_bezier_segment(cn, cn.get_control_handle(1),
                                nn.get_control_handle(0), nn, start_step)
                    cur_step = start_step

                    while cur_step <= end_step:
                        cx, cy = _get_bezier_segment(cn, cn.get_control_handle(1),
                                nn.get_control_handle(0), nn, cur_step)
                        sx = min(ox, cx) - allow_distance
                        ex = max(ox, cx) + allow_distance
                        sy = min(oy, cy) - allow_distance
                        ey = max(oy, cy) + allow_distance
                        if sx <= x <= ex and sy <= y <= ey:
                            # vpx/vpy : vector of assigned point
                            # vsx/vsy : vector of segment
                            # TODO this is same as 'simplify nodes'
                            # so these can be commonize.
                            vpx = x - ox
                            vpy = y - oy
                            vsx = cx - ox
                            vsy = cy - oy
                            scaler_s = math.sqrt(vsx**2 + vsy**2)
                            if scaler_s > 0:
                                nsx = vsx / scaler_s
                                nsy = vsy / scaler_s
                                dot_vp_v = nsx * vpx + nsy * vpy
                                vsx = (vsx * dot_vp_v) / scaler_s
                                vsy = (vsy * dot_vp_v) / scaler_s
                                vsx -= vpx
                                vsy -= vpy
                            else:
                                vsx= vpx
                                vsy= vpy

                            # now,vsx/vsy is the vector of distance between
                            # vpx/vpx and vsx/vsy
                            distance = math.sqrt(vsx**2 + vsy**2)


                            if 0 < distance < allow_distance:
                                return cur_step

                        ox = cx
                        oy = cy
                        cur_step += increase_step
                        

                detected = detect_step(0.0, 1.0 , 0.1)
                # final detection
                if detected:
                    prev_detected = detected
                    detected = detect_step(detected, detected + 0.1, 0.01)
                
                    if detected:
                        return (i,detected)
                    else:
                        return (i,prev_detected)
                

                # We need search entire the stroke     
                # because it might be intersected.

        return None

    ## Redraws
    
    def _queue_draw_node(self, i):
        """Redraws a specific control node on all known view TDWs"""
        node = self.nodes[i]
        dx,dy = self.selection_rect.get_model_offset()
        
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

            # Active control handles also should be queued.
            for hi in (0,1):
                # But,the first node only shows 2nd(index 1) handle.
                if (hi == 0 and i > 0) or (hi == 1 and i <= len(self.nodes)-1):
                    handle = node.get_control_handle(hi)
                    area = get_area(i, 
                        handle.x, handle.y,
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
            dx, dy = self.selection_rect.get_model_offset()
            for p0, p1, p2, p3 in _bezier_iter_offset(self.nodes,
                    self.selected_nodes,
                    (dx,dy)):
                self._queue_task(
                    self._draw_curve_segment,
                    model,
                    p0, p1, p2, p3, step
                )
        self._start_task_queue_runner()

    def _queue_draw_buttons(self):
        # To surpress exception
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
        
        
        cur_step = 0.0
        pressure = 1.0 # for testing
        xtilt = 0.0
        ytilt = 0.0
        while cur_step <= 1.0:
            x, y = _get_bezier_segment(p0, p1, p2, p3, cur_step)

            #t_abs = max(last_t_abs, t_abs)
            #dtime = t_abs - last_t_abs
            # TODO This is the big problem.
            # how we decide the speed from static node list?
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

        if self.phase in (_PhaseBezier.INITIAL, _PhaseBezier.CREATE_PATH):
            # Initial state - everything starts here!
       
            if self.zone == _EditZone_Bezier.CONTROL_HANDLE:
                self.phase = _PhaseBezier.ADJUST_HANDLE
            elif self.zone == _EditZone_Bezier.CONTROL_NODE:
                # Grabbing a node...
                button = event.button
                if (self.current_node_index is not None and 
                        button == 1 and
                        event.state & self.__class__._PRESSURE_MOD_MASK == 
                        self.__class__._PRESSURE_MOD_MASK):
                    # It's 'Entering On-canvas Pressure Adjustment Phase'!

                    self.phase = _PhaseBezier.ADJUST_PRESSURE
            
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
                    # normal move node start
                    self.phase = _PhaseBezier.MOVE_NODE

                    if button == 1:
                        if (event.state & Gdk.ModifierType.CONTROL_MASK):
                            # Holding CONTROL key = adding or removing a node.
                            if self.current_node_index in self.selected_nodes:
                                self.selected_nodes.remove(self.current_node_index)
                            else:
                                self.selected_nodes.append(self.current_node_index)
        
                            self._queue_draw_selected_nodes() 
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

            elif self.zone == _EditZone_Bezier.EMPTY_CANVAS:
                
                if (len(self.nodes) > 0): 

                    if (event.state & Gdk.ModifierType.SHIFT_MASK):
                        # selection box dragging start!!
                        self.phase = _PhaseBezier.ADJUST_SELECTING
                        self.selection_rect.start(
                                *tdw.display_to_model(event.x, event.y))
                    elif (event.state & Gdk.ModifierType.CONTROL_MASK):
                        mx, my = tdw.display_to_model(event.x, event.y)
                        pressed_segment = self._detect_on_stroke(mx, my)
                        if pressed_segment:
                            # pressed_segment is a tuple which contains
                            # (node index of start of segment, stroke step)

                            # To erase buttons 
                           #self._queue_draw_buttons() 

                            self._divide_bezier(*pressed_segment)

                            # queue new node here.
                            self._queue_draw_node(pressed_segment[0] + 1)
                            
                            self.phase = _PhaseBezier.PLACE_NODE
                            return False # Cancel drag event



            if self.phase == _PhaseBezier.INITIAL:
               #BezierMode.change_switch_actions(_PermitAction.ENABLE)
                self.phase = _PhaseBezier.CREATE_PATH
            # FALLTHRU: *do* start a drag 

        elif self.phase == _PhaseBezier.ADJUST_SELECTING:
            # XXX Not sure what to do here.
            pass
        elif self.phase in (_PhaseBezier.ADJUST_HANDLE, _PhaseBezier.INIT_HANDLE):
            pass
        else:
            raise NotImplementedError("Unrecognized zone %r", self.zone)
        # Update workaround state for evdev dropouts
        self._button_down = event.button

        # Super-Supercall(not supercall) would invoke drag-related callbacks.
        return super(InkingMode, self).button_press_cb(tdw, event) 

    def button_release_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False

        if self.phase == _PhaseBezier.CREATE_PATH:
            if self.zone == _EditZone_Bezier.REJECT_BUTTON:
                self._start_new_capture_phase(rollback=True)
            elif self.zone == _EditZone_Bezier.ACCEPT_BUTTON:
                self._queue_redraw_curve(0.01) # Redraw with hi-fidely curve
                self._start_new_capture_phase(rollback=False)
        if self.phase == _PhaseBezier.PLACE_NODE:
           #self._queue_draw_buttons() 
            self._queue_redraw_curve() 
            self.phase = _PhaseBezier.CREATE_PATH
            pass

        # Update workaround state for evdev dropouts
        self._button_down = None
        # Initialize pressed position as invalid for hold-and-modify
        self._pressed_x = None
        self._pressed_y = None



        # Super-Supercall(not supercall) would invoke drag_stop_cb signal.
        return super(InkingMode, self).button_release_cb(tdw, event)
        

    ## Drag handling (both capture and adjust phases)
    def drag_start_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        dx, dy = tdw.display_to_model(event.x, event.y)

        self._queue_previous_draw_buttons() # To erase button,and avoid glitch

        # Basically,all sections should do fall-through.
        if self.phase == _PhaseBezier.CREATE_PATH:

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
                    self.phase = _PhaseBezier.INIT_HANDLE
                    self.current_node_index=len(self.nodes)-1
                    # Important: with setting initial control handle 
                    # as the 'next' (= index 1) one,it brings us
                    # inkscape-like node creation.
                    self.current_handle_index = 1 

                    self._queue_draw_node(self.current_node_index)

        elif self.phase == _PhaseBezier.MOVE_NODE:
            if len(self.selected_nodes) > 0:
                # Use selection_rect class as offset-information
                self.selection_rect.start(dx, dy)
        
        elif self.phase == _PhaseBezier.ADJUST_PRESSURE:
            if self.current_node_index is not None:
                node = self.nodes[self.current_node_index]
                self._pressed_pressure = node.pressure
                self._pressed_x, self._pressed_y = \
                        tdw.display_to_model(dx, dy)
        elif self.phase == _PhaseBezier.ADJUST_SELECTING:
            self.selection_rect.start(dx, dy)
            self.selection_rect.is_addition = (event.state & Gdk.ModifierType.CONTROL_MASK)
            self._queue_draw_selection_rect() # to start
        elif self.phase == _PhaseBezier.ADJUST_HANDLE:
            self._last_event_node = self.nodes[self.target_node_index]
            pass
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)


    def drag_update_cb(self, tdw, event, dx, dy):
        self._ensure_overlay_for_tdw(tdw)
        mx, my = tdw.display_to_model(event.x, event.y)
        if self.phase == _PhaseBezier.CREATE_PATH:
            pass
            
        elif self.phase in (_PhaseBezier.ADJUST_HANDLE, _PhaseBezier.INIT_HANDLE):
            node = self._last_event_node
            if self._last_event_node:
                self._queue_draw_node(self.current_node_index)# to erase
                node.set_control_handle(self.current_handle_index,
                        mx, my)

                self._queue_draw_node(self.current_node_index)
            self._queue_redraw_curve()
                
        elif self.phase == _PhaseBezier.MOVE_NODE:
            if len(self.selected_nodes) > 0:
                self._queue_draw_selected_nodes()
                self.selection_rect.drag(mx, my)
                self._queue_draw_selected_nodes()
                self._queue_redraw_curve()
        elif self.phase == _PhaseBezier.ADJUST_PRESSURE:
            if self._pressed_pressure is not None:
                self._adjust_pressure_with_motion(mx, my)
        elif self.phase == _PhaseBezier.ADJUST_SELECTING:
            self._queue_draw_selection_rect() # to erase
            self.selection_rect.drag(mx, my)
            self._queue_draw_selection_rect()
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)

    def drag_stop_cb(self, tdw):
        self._ensure_overlay_for_tdw(tdw)
        if self.phase == _PhaseBezier.CREATE_PATH:
            node = self._last_event_node
            self._reset_adjust_data()
            self._queue_redraw_all_nodes()
            self._queue_redraw_curve()
            if len(self.nodes) > 1:
                self._queue_draw_buttons()
                
            
        elif self.phase in (_PhaseBezier.ADJUST_HANDLE, _PhaseBezier.INIT_HANDLE):
            node = self._last_event_node
      
            # At initialize handle phase, even if the node is not 'curve'
            # Set the handles as symmetry.
            if (self.phase == _PhaseBezier.INIT_HANDLE and 
                    len(self.nodes) > 1 and node.curve == False):
                node.curve = True 
                node.curve = False

            self._queue_redraw_all_nodes()
            self._queue_redraw_curve()
            if len(self.nodes) > 1:
                self._queue_draw_buttons()
                
            self.phase = _PhaseBezier.CREATE_PATH
        elif self.phase == _PhaseBezier.MOVE_NODE:
            dx, dy = self.selection_rect.get_model_offset()

            for idx in self.selected_nodes:
                cn = self.nodes[idx]
                cn.move(cn.x + dx, cn.y + dy)

            self.selection_rect.reset()
            self._dragged_node_start_pos = None
            self._queue_redraw_curve()
            self._queue_draw_buttons()
            self.phase = _PhaseBezier.CREATE_PATH
        elif self.phase == _PhaseBezier.ADJUST_SELECTING:
            ## Nodes selection phase
            self._queue_draw_selection_rect()

            modified = False
            if not self.selection_rect.is_addition:
                self._reset_selected_nodes()
                modified = True

            for idx,cn in enumerate(self.nodes):
                if self.selection_rect.is_inside(cn.x, cn.y):
                    if not idx in self.selected_nodes:
                        self.selected_nodes.append(idx)
                        modified = True

            if modified:
                self._queue_redraw_all_nodes()

            self._queue_draw_buttons() # buttons erased while selecting
            self.selection_rect.reset()
            self.phase = _PhaseBezier.CREATE_PATH

        elif self.phase == _PhaseBezier.ADJUST_PRESSURE:
            self.phase = _PhaseBezier.CREATE_PATH

        # Common processing
        if self.current_node_index != None:
            self.options_presenter.target = (self, self.current_node_index)

    ## Interrogating events

    def _get_event_data(self, tdw, event):
        # almost same as inktool,but we needs generate _Node_Bezier object
        # not _Node object
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

    def _divide_bezier(self, index, step):
        """ Divide (insert a node intermidiate stroke)
        to current active bezier stroke 
        without shape change.
        """
        assert index < len(self.nodes)-1
        cn = self.nodes[index]
        nn = self.nodes[index+1]
        p0 = cn
        p1 = cn.get_control_handle(1)
        p2 = nn.get_control_handle(0)
        p3 = nn

        xa = _get_bezier_pt(p0.x, p1.x, step)
        ya = _get_bezier_pt(p0.y, p1.y, step)
        xb = _get_bezier_pt(p1.x, p2.x, step)
        yb = _get_bezier_pt(p1.y, p2.y, step)
        xc = _get_bezier_pt(p2.x, p3.x, step)
        yc = _get_bezier_pt(p2.y, p3.y, step)

        xd = _get_bezier_pt(xa, xb, step)
        yd = _get_bezier_pt(ya, yb, step)
        xe = _get_bezier_pt(xb, xc, step)
        ye = _get_bezier_pt(yb, yc, step)
    

        # The nodes around a new node changed to 'not curve' node,
        # to retain original shape.
        cn.curve = False
        cn.set_control_handle(1, xa, ya)
        new_node = _Node_Bezier(
                    _get_bezier_pt(xd, xe, step), _get_bezier_pt(yd, ye, step),
                    pressure = cn.pressure + ((nn.pressure - cn.pressure) * step),
                    xtilt = cn.xtilt + (nn.xtilt - cn.xtilt) * step,
                    ytilt = cn.ytilt + (nn.ytilt - cn.ytilt) * step,
                    curve = False)
        new_node.set_control_handle(0, xd, yd)
        new_node.set_control_handle(1, xe, ye)
        self.nodes.insert(index + 1, new_node)

        nn.curve = False
        nn.set_control_handle(0, xc, yc)


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
        # The difference is for-loop of nodes , to deal with control handles.
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
            # ADDED CODES FOR BEZIERTOOL: 
            # to avoid buttons are overwrap on control handles,
            # treat control handles as nodes,when it is visible.
            if i == self._inkmode.current_node_index:
                for t in (0,1):
                    handle = node.get_control_handle(t)
                    fixed.append(_LayoutNode(handle.x, handle.y))
            # ADDED CODES END.

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
        dx, dy = mode.selection_rect.get_display_offset(self._tdw)
        for i, node, x, y in self._get_onscreen_nodes():
            color = gui.style.EDITABLE_ITEM_COLOR
            if mode.phase in (_PhaseBezier.INITIAL, _PhaseBezier.MOVE_NODE, 
                    _PhaseBezier.CREATE_PATH, _PhaseBezier.ADJUST_HANDLE, _PhaseBezier.INIT_HANDLE):
                if i == mode.current_node_index:
                    color = gui.style.ACTIVE_ITEM_COLOR
                    x += dx
                    y += dy
              
                    # Drawing control handle
                    cr.save()
                    cr.set_source_rgb(0,0,1)
                    cr.set_line_width(1)
                    for hi in (0,1):                        
                        if ((hi == 0 and i > 0) or
                                (hi == 1 and i <= len(self._inkmode.nodes)-1)): 
                            ch = node.get_control_handle(hi)
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
                elif i in mode.selected_nodes:
                    color = gui.style.POSTLIT_ITEM_COLOR
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
        

        # Selection Rectangle
        if mode.phase == _PhaseBezier.ADJUST_SELECTING:
            self.draw_selection_rect(cr)
                


class OptionsPresenter_Bezier (OptionsPresenter):
    """Presents UI for directly editing point values etc."""

    def __init__(self):
        super(OptionsPresenter_Bezier, self).__init__()

    def _ensure_ui_populated(self):
        if self._options_grid is not None:
            return
        builder_xml = os.path.splitext(__file__)[0] + ".glade"
        builder = Gtk.Builder()
        builder.set_translation_domain("mypaint")
        builder.add_from_file(builder_xml)
        builder.connect_signals(self)
        self._options_grid = builder.get_object("options_grid")
        self._point_values_grid = builder.get_object("point_values_grid")
        self._point_values_grid.set_sensitive(False)
        self._pressure_adj = builder.get_object("pressure_adj")
        self._xtilt_adj = builder.get_object("xtilt_adj")
        self._ytilt_adj = builder.get_object("ytilt_adj")
        self._dtime_adj = builder.get_object("dtime_adj")
        self._dtime_label = builder.get_object("dtime_label")
        self._dtime_scale = builder.get_object("dtime_scale")
        self._insert_button = builder.get_object("insert_point_button")
        self._insert_button.set_sensitive(False)
        self._delete_button = builder.get_object("delete_point_button")
        self._delete_button.set_sensitive(False)
        self._check_curvepoint= builder.get_object("checkbutton_curvepoint")
        self._check_curvepoint.set_sensitive(False)
        self._average_pressure_button = builder.get_object("average_pressure_button")
        self._average_pressure_button.set_sensitive(False)

    @property
    def target(self):
        return super(OptionsPresenter_Bezier, self).target

    @target.setter
    def target(self, targ):
        beziermode, cn_idx = targ
        beziermode_ref = None
        if beziermode:
            beziermode_ref = weakref.ref(beziermode)
        self._target = (beziermode_ref, cn_idx)
        # Update the UI
        if self._updating_ui:
            return
        self._updating_ui = True
        try:
            self._ensure_ui_populated()
            if 0 <= cn_idx < len(beziermode.nodes):
                cn = beziermode.nodes[cn_idx]
                self._pressure_adj.set_value(cn.pressure)
                self._xtilt_adj.set_value(cn.xtilt)
                self._ytilt_adj.set_value(cn.ytilt)
                if cn_idx > 0:
                    sensitive = True
                    dtime = beziermode.get_node_dtime(cn_idx)
                else:
                    sensitive = False
                    dtime = 0.0
                for w in (self._dtime_scale, self._dtime_label):
                    w.set_sensitive(sensitive)
                self._dtime_adj.set_value(dtime)
                self._point_values_grid.set_sensitive(True)
                self._check_curvepoint.set_sensitive(True)
                self._check_curvepoint.set_active(cn.curve)
            else:
                self._point_values_grid.set_sensitive(False)
                self._check_curvepoint.set_sensitive(False)

            self._insert_button.set_sensitive(beziermode.can_insert_node(cn_idx))
            self._delete_button.set_sensitive(beziermode.can_delete_node(cn_idx))
            self._average_pressure_button.set_sensitive(len(beziermode.nodes) > 2)
        finally:
            self._updating_ui = False                               

    def _checkbutton_curvepoint_toggled_cb(self, button):
        beziermode, node_idx = self.target
        if beziermode:
            if 0 <= node_idx < len(beziermode.nodes):
                beziermode._queue_draw_node(node_idx) 
                beziermode._queue_redraw_curve()
                beziermode.nodes[node_idx].curve = button.get_active()
                beziermode._queue_draw_node(node_idx) 
                beziermode._queue_redraw_curve()

    ## Other handlers are as implemented in superclass.  
