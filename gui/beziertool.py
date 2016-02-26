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
        self.xtilt = xtilt
        self.ytilt = ytilt
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

  

    ## Class config vars


    ## Other class vars

    _OPTIONS_PRESENTER = None   #: Options presenter singleton

    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):
        super(BezierMode, self).__init__(**kwargs)






    ## Redraws

    def _queue_redraw_curve(self):
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
                    p0, p1, p2, p3,
                )
        self._start_task_queue_runner()

    def _draw_curve_segment(self, model, p0, p1, p2, p3):
        """Draw the curve segment between the middle two points"""
        
        def get_pt(v0, v1, step):
            return v0 + ((v1-v0) * step)
        
        step = 0.0
        pressure = 1.0 # for testing
        xtilt = 0.0
        ytilt = 0.0
        while step <= 1.0:
            xa = get_pt(p0.x, p1.x, step)
            ya = get_pt(p0.y, p1.y, step)
            xb = get_pt(p1.x, p2.x, step)
            yb = get_pt(p1.y, p2.y, step)
            xc = get_pt(p2.x, p3.x, step)
            yc = get_pt(p2.y, p3.y, step)
            
            xa = get_pt(xa, xb, step)
            ya = get_pt(ya, yb, step)
            xb = get_pt(xb, xc, step)
            yb = get_pt(yb, yc, step)
            
            x = get_pt(xa, xb, step)
            y = get_pt(ya, yb, step)
            
            #t_abs = max(last_t_abs, t_abs)
            #dtime = t_abs - last_t_abs
            dtime = 1.0
            
            self.stroke_to(
                model, dtime, x, y, pressure, xtilt, ytilt,
                auto_split=False,
            )
                        
            step += 0.01


    ## Drag handling (both capture and adjust phases)
    def drag_start_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        dx, dy = tdw.display_to_model(event.x, event.y)
        if self.phase == _Phase.CAPTURE:
            #self._reset_nodes()
            #self._reset_capture_data()
            #self._reset_adjust_data()

            if event.state != 0:
                # To activate some mode override
                self._last_event_node = None
                return super(InkingMode, self).drag_start_cb(tdw, event)
            else:
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
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)

    def drag_update_cb(self, tdw, event, dx, dy):
        self._ensure_overlay_for_tdw(tdw)
        if self.phase == _Phase.CAPTURE:
            mx, my = tdw.display_to_model(event.x, event.y)
            if self._last_event_node:
              
                self._queue_draw_node(len(self.nodes)-1) # to erase
                self._last_event_node.x = mx
                self._last_event_node.y = my
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
        if self.phase == _Phase.CAPTURE:
            node = self._last_event_node
            #self.nodes.append(node)
            #print self.nodes

            self._reset_adjust_data()
            self._queue_redraw_all_nodes()
            self._queue_redraw_curve()
            if len(self.nodes) > 1:
                self._queue_draw_buttons()
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

    def __init__(self, inkmode, tdw):
        super(OverlayBezier, self).__init__()
        
    
    def paint(self, cr):
        """Draw adjustable nodes to the screen"""
        # Control nodes
        mode = self._beziermode
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        alloc = self._tdw.get_allocation()
        for i, node, x, y in self._get_onscreen_nodes():
            color = gui.style.EDITABLE_ITEM_COLOR
            if mode.phase == _Phase.ADJUST:
                if i == mode.current_node_index:
                    color = gui.style.ACTIVE_ITEM_COLOR
                elif i == mode.target_node_index:
                    color = gui.style.PRELIT_ITEM_COLOR
            gui.drawutils.render_round_floating_color_chip(
                cr=cr, x=x, y=y,
                color=color,
                radius=radius,
            )
        # Buttons
        if mode.phase == _Phase.ADJUST and not mode.in_drag:
            self.update_button_positions()
            radius = gui.style.FLOATING_BUTTON_RADIUS
            button_info = [
                (
                    "mypaint-ok-symbolic",
                    self.accept_button_pos,
                    _EditZone.ACCEPT_BUTTON,
                ),
                (
                    "mypaint-trash-symbolic",
                    self.reject_button_pos,
                    _EditZone.REJECT_BUTTON,
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

  
