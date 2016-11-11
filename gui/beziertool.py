# This file is part of MyPaint.
# Copyright (C) 2016 by dothiko <dothiko@gmail.com>
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
import array
import time

from gettext import gettext as _
import gi
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GLib
import cairo

import gui.mode
import gui.overlays
import gui.style
import gui.drawutils
import lib.helpers
import gui.cursor
import lib.observable
from gui.inktool import *
from gui.inktool import _LayoutNode, _Phase
from gui.linemode import *
from gui.oncanvas import *

## Class defs

class _Control_Handle(object):
    def __init__(self, x, y):
        self._array = array.array('f',(x,y))

    def __getitem__(self, idx):
        return self._array[idx]

    @property
    def x(self):
        return self._array[0]
    @x.setter
    def x(self,x):
        self._array[0]=x

    @property
    def y(self):
        return self._array[1]
    @y.setter
    def y(self,y):
        self._array[1]=y
    
class _Node_Bezier (_Control_Handle):
    """Node (Control point) class,with handle.

    In bezier curve,nodes would be frequently rewritten.
    Evenmore, each nodes have its own control handle.
    This control handle should be adjusted automatically,
    in certain situation.
    So this class is object, not a namedtuple.

    _Node_Bezier thave the following 6 fields, in order

    * x, y: model coords, float
    * pressure: float in [0.0, 1.0]
    * xtilt, ytilt: float in [-1.0, 1.0]
    * control_handle: two _Control_Handle objects.
    """
    
            
            
    def __init__(self,x,y,pressure=1.0,xtilt=0.0,ytilt=0.0,dtime=0.5,
            control_handles=None,curve=True):
        self.pressure = pressure
        self.xtilt = xtilt
        self.ytilt = ytilt
        self.time = dtime
        super(_Node_Bezier, self).__init__(x, y)

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
                    self._control_handles[0].y,
                    False)

    def set_control_handle(self, idx, x, y, invert_curve_flag):
        """Use this method to set control handle.
        This method refers self._curve flag,
        and if it is True,automatically make handles
        as symmetry = 'Curved bezier control point'
        """

        dx = x - self.x
        dy = y - self.y
        self._control_handles[idx].x = x 
        self._control_handles[idx].y = y

        curve = self._curve
        if invert_curve_flag:
            curve = not curve

        if curve:
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
        return node

    def __getitem__(self, idx):
        return self._array[idx]
    
class _EditZone(EditZone_Mixin):
    """Enumeration of what the pointer is on in the ADJUST phase"""
    CONTROL_HANDLE = 104 #: Control handle of bezier

    # _EditZone-enum also used,they are defined at gui.inktool._EditZone
   #EMPTY_CANVAS   #: Nothing, empty space
   #CONTROL_NODE   #: Any control node; see target_node_index
   #REJECT_BUTTON  #: On-canvas button that abandons the current line
   #ACCEPT_BUTTON  #: On-canvas button that commits the current line

class _Phase(Phase_Mixin):
    """Enumeration of the states that an BezierCurveMode can be in"""
    ADJUST_PRESSURE = 2
    ADJUST_PRESSURE_ONESHOT = 4
    CHANGE_PHASE = 5
    INITIAL = Phase_Mixin.CAPTURE     #: Initial Phase,before creating any node.
                                 #  This is needed when change changing brush size 
                                 #  before create path.

    CREATE_PATH = Phase_Mixin.ADJUST  #: Main Phase.creating path(adding node)
                                 # THIS MEMBER MUST SAME AS _Phase.ADJUST
                                 # because for InkingMode.scroll_cb()
    MOVE_NODE = 100         #: Moving node(s)
    ADJUST_HANDLE = 103     #: change control-handle position
    INIT_HANDLE = 104       #: initialize control handle,right after create a node
    PLACE_NODE = 105        #: place a new node into clicked position on current
                            # stroke,when you click with holding CTRL key
    CALL_BUTTONS = 106      #: call buttons around clicked point. 


class PressureMap(object):
    """ PressureMap wrapper object, to mapping 'pressure-variation'
    configuration to current stroke.

    With this mechanism, even the stroke has only 2 nodes,
    the variation mapped entire stroke.
    old node-based pressure mapping code,it cannot 
    process pressure correctly in 2 node curve.
    """

    def __init__(self, source):
        """
        :param source: source curve widget.
        """
        self.curve_widget = source

    def get_pressure(self, step):
        return self.curve_widget.get_pressure_value(step)

class StrokeHistory(object):
    """ Stroke History class.this is singleton,stored as 
    class attribute of BezierMode
    """
    def __init__(self, maxcount):
        self._max = maxcount
        self._nodes = Gtk.ListStore(str,object)
        self._nodes.append( ('------', None) )
        
    def register(self, nodes):
        lt = time.localtime()
        timestr = '%02d:%02d:%02d-%d' % (lt.tm_hour, lt.tm_min,
                lt.tm_sec, len(nodes))
        self._nodes.insert(1, (timestr, nodes))
        if len(self._nodes) > self._max:
            del self._nodes[self._max - 1]

    @property
    def liststore(self):
        return self._nodes

    def get_and_place_nodes(self, idx, x, y):
        """ generate a copy of stroke and
        'place' (move) it to (x,y)

        :param idx: the REVERSED index(most recent one is 0) of nodes.
        """
        assert 1 <= idx < len(self._nodes)
        src = self._nodes[idx][1]
        fn = src[0]
        ret = []
        for cn in src:
            if x != None and y != None:
                nx = (cn.x - fn.x) + x
                ny = (cn.y - fn.y) + y
            else:
                nx = cn.x
                ny = cn.y

            new_node=_Node_Bezier(
                    nx,ny,
                    cn.pressure,
                    cn.xtilt,cn.ytilt,
                    cn.time,
                    None,
                    cn.curve)

            for i in (0, 1):
                handle = cn.get_control_handle(i)
                if x != None and y != None:
                    nx = (handle.x - fn.x) + x 
                    ny = (handle.y - fn.y) + y
                else:
                    nx = handle.x
                    ny = handle.y
                new_node.set_control_handle(i, nx, ny, False)
                if cn.curve:
                    break

            ret.append(new_node)

        return ret
        



class BezierMode (OncanvasEditMixin):

    ## Metadata properties
    ACTION_NAME = "BezierCurveMode"

    ## Metadata methods

    @classmethod
    def get_name(cls):
        return _(u"Bezier")

    def get_usage(self):
        return _(u"Draw, and then adjust smooth lines with bezier curve")


    @property
    def active_cursor(self):
        if self.phase in (_Phase.INITIAL, _Phase.CREATE_PATH):
            if self.zone in (_EditZone.CONTROL_NODE,
                    _EditZone.CONTROL_HANDLE):
                return self._crosshair_cursor
        elif self.phase == _Phase.MOVE_NODE:
            if self.zone == _EditZone.CONTROL_NODE:
                return self._crosshair_cursor
            elif self.zone != _EditZone.EMPTY_CANVAS: # assume button
                return self._arrow_cursor

        elif self.phase == _Phase.ADJUST_PRESSURE:
            if self.zone == _EditZone.CONTROL_NODE:
                return self._cursor_move_nw_se

        return None  

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, new_phase):
        if new_phase == _Phase.INITIAL:
            self.enable_switch_actions(True)
        else:
            self.enable_switch_actions(False)
        self._phase = new_phase

    @classmethod
    def set_default_dtime(cls, new_dtime):
        cls._DEFAULT_DTIME = new_dtime

    @classmethod
    def set_default_pressure(cls, new_pressure):
        cls._DEFAULT_PRESSURE = new_pressure


    ## Class config vars

    DRAFT_STEP = 0.01 # Draft(Editing) Bezier curve stroke step.
    FINAL_STEP = 0.005 # Final output stroke Bezier-curve step.

    _DEFAULT_PRESSURE = 0.5 # default bezier pressure,this is fixed value.
                            # because it is hard to control pressure 
                            # for human hand at node creation.

    _DEFAULT_DTIME = 0.5 # default dtime value

    stroke_history = StrokeHistory(6) # stroke history

    DEFAULT_POINT_CORNER = False       # default point is curve,not corner

    ## Other class vars

    _PRESSURE_MAP = None #: Pressure mapping object singleton


    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):
        super(BezierMode, self).__init__(**kwargs)
        self._stroke_from_history = False
        self.forced_button_pos = None


    def _reset_adjust_data(self):
        super(BezierMode, self)._reset_adjust_data()
        self.current_handle_index = None

    
    def _generate_overlay(self, tdw):
        return OverlayBezier(self, tdw)

    ## Options presenter
    def _generate_presenter(self):
        return OptionsPresenter_Bezier()

    ## Update inner states methods

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
        
    def _update_zone_and_target(self, tdw, x, y, ignore_handle=False):
        """Update the zone and target node under a cursor position"""
        ## FIXME mostly copied from inktool.py
        ## the differences are 'control handle processing' and
        ## 'cursor changing' and, queuing buttons draw 
        ## to follow current_node_index


        self._ensure_overlay_for_tdw(tdw)
        new_zone = _EditZone.EMPTY_CANVAS
        if not self.in_drag and len(self.nodes) > 0:
            if self.phase in (_Phase.MOVE_NODE, 
                    _Phase.ADJUST_PRESSURE, 
                    _Phase.CREATE_PATH):

                new_target_node_index = None
                
                # Test buttons for hits
                overlay = self._ensure_overlay_for_tdw(tdw)
                hit_dist = gui.style.FLOATING_BUTTON_RADIUS

                if len(self.nodes) > 1:
                   #button_info = [
                   #    (_EditZone.ACCEPT_BUTTON, overlay.get_button_pos(_EditZone.ACCEPT_BUTTON)),
                   #    (_EditZone.REJECT_BUTTON, overlay.get_button_pos(_EditZone.REJECT_BUTTON)),
                   #]
                    for btn_zone in (_EditZone.ACCEPT_BUTTON, _EditZone.REJECT_BUTTON):
                        btn_pos = overlay.get_button_pos(btn_zone)
                        if btn_pos is None:
                            continue
                        btn_x, btn_y = btn_pos
                        d = math.hypot(btn_x - x, btn_y - y)
                        if d <= hit_dist:
                            new_target_node_index = None
                            new_zone = btn_zone
                            break


                if (new_zone == _EditZone.EMPTY_CANVAS):

                    if self.phase == _Phase.ADJUST_PRESSURE:
                        new_target_node_index = self._search_target_node(tdw, x, y)
                    else:
                        # Checking Control handles first:
                        # because when you missed setting control handle 
                        # at node creation stage,if node zone detection
                        # is prior to control handle, they are unoperatable.
                        if (self.current_node_index is not None and 
                                ignore_handle == False):
                            c_node = self.nodes[self.current_node_index]
                            self.current_handle_index = None
                            if self.current_node_index == 0:
                                seq = (1,)
                            else:
                                seq = (0, 1)
                            for i in seq:
                                handle = c_node.get_control_handle(i)
                                hx, hy = tdw.model_to_display(handle.x, handle.y)
                                d = math.hypot(hx - x, hy - y)
                                if d > hit_dist:
                                    continue
                                new_target_node_index = self.current_node_index
                                self.current_handle_index = i
                                new_zone = _EditZone.CONTROL_HANDLE
                                break         

                        # Test nodes for a hit, in reverse draw order
                        if new_target_node_index == None:
                            new_target_node_index = self._search_target_node(tdw, x, y)

                    if (new_target_node_index != None and 
                        new_zone == _EditZone.EMPTY_CANVAS):
                        new_zone = _EditZone.CONTROL_NODE
                    
                    
                # Update the prelit node, and draw changes to it
                if new_target_node_index != self.target_node_index:
                    if self.target_node_index is not None:
                        self._queue_draw_node(self.target_node_index)
                    self.target_node_index = new_target_node_index
                    if self.target_node_index is not None:
                        self._queue_draw_node(self.target_node_index)

                ## Fallthru below


        elif self.phase == _Phase.ADJUST_PRESSURE_ONESHOT:
            # Always control node,in pressure editing.
            new_zone = _EditZone.CONTROL_NODE 

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
            if self.phase in (_Phase.INITIAL, _Phase.CREATE_PATH,
                    _Phase.MOVE_NODE, _Phase.ADJUST_PRESSURE):
                if self.zone == _EditZone.CONTROL_NODE:
                    cursor = self._crosshair_cursor
                elif self.zone != _EditZone.EMPTY_CANVAS: # assume button
                    cursor = self._arrow_cursor
            if cursor is not self._current_override_cursor:
                tdw.set_override_cursor(cursor)
                self._current_override_cursor = cursor

    def _start_new_capture_phase(self, rollback=False):
        if rollback:
            pass
        else:
            if not self._stroke_from_history:
                # This MUST be called prior to supercall.
                # because current stroke data will be erased 
                # at that method.
                self.stroke_history.register(self.nodes)

        super(BezierMode, self)._start_new_capture_phase(rollback)
        self._stroke_from_history = False
        self.options_presenter.reset_stroke_history()
        self.forced_button_pos = None

    ## Stroke related
    def _detect_on_stroke(self, x, y, allow_distance = 4.0):
        """ detect the assigned coordinate is on stroke or not
        currently implemented very silly way (scan entire
        bezier segment until find most closer point)

        it must done with newton's method

        :param x: cursor x position in MODEL coord
        :param y: cursor y position in MODEL coord
        """

        for i,cn in enumerate(self.nodes[:-1]):
            # Get boundary rectangle,to reduce processing segment
            nn = self.nodes[i+1]
            p0 = cn.x
            p1 = cn.get_control_handle(1).x
            p2 = nn.get_control_handle(0).x
            p3 = nn.x

            q0 = cn.y
            q1 = cn.get_control_handle(1).y
            q2 = nn.get_control_handle(0).y
            q3 = nn.y

            sx, ex = gui.drawutils.get_minmax_bezier(p0, p1, p2, p3)
            sy, ey = gui.drawutils.get_minmax_bezier(q0, q1, q2, q3)
            
            if sx <= x <= ex and sy <= y <= ey:
                # first, detect from 
                limitDt = 0.001
                t = 0
                loop = 0
                A = (3.0*p1 + p3 - 3.0*p2 - p0)
                B = 3.0 * (p0 - 2*p1 + p2)
                C = 3.0 * (p1 - p0)
                while loop < 2:
                    cnt = 0
                    while cnt < 10:
                        dx = gui.drawutils.get_cubic_bezier(
                                p0, p1, p2, p3, t) - x
                                
                        
                        if (abs(dx) < limitDt):
                            break
                        
                        slope = 3.0 * A * (t**2) + 2.0 * B * t + C
                        
                        if slope != 0.0:
                            t = (slope*t - dx)/slope;
                        else:
                            t += 0.001

                        cnt+=1

                    if (cnt < 10 and 0.0 <= t <= 1.0): 
                        dy = gui.drawutils.get_cubic_bezier(
                                q0, q1, q2, q3, t)
                        if dy - allow_distance < y < dy + allow_distance:
                            return (i, t)

                    t = 1.0
                    loop+=1

                ## Failthrogh, and doing full search
                t = 0
                candidate_t = None
                least_diff = None
                while t < 1.0:
                    cx = gui.drawutils.get_cubic_bezier(
                            p0, p1, p2, p3, t) 
                    cy = gui.drawutils.get_cubic_bezier(
                            q0, q1, q2, q3, t)

                    cur_diff = math.hypot(cx-x,cy-y)
                   #if ((cy - allow_distance < y < cy + allow_distance) and
                   #        (cx - allow_distance < x < cx + allow_distance)):
                   #    cur_diff = math.hypot(cx-x,cy-y)
                    if cur_diff < allow_distance:
                        if least_diff != None:
                            if (least_diff > cur_diff):
                                least_diff = cur_diff
                                candidate_t = t
                        else:
                            least_diff = cur_diff
                            candidate_t = t

                    t += 0.001

                if candidate_t:
                    return (i, candidate_t)


    ## Redraws
    
    def _queue_draw_node(self, i):
        """Redraws a specific control node on all known view TDWs
        """
        node = self.nodes[i]
        dx,dy = self.drag_offset.get_model_offset()
        
        def get_area(node_idx, nx, ny, radius, area=None):
            if node_idx in self.selected_nodes:
                x, y = tdw.model_to_display(
                        nx + dx, ny + dy)
            else:
                x, y = tdw.model_to_display(nx, ny)

            x = math.ceil(x)
            y = math.ceil(y)
            sx = x-radius-2
            sy = y-radius-2
            ex = x+radius+2
            ey = y+radius+2
            if not area:
                return (sx, sy, ex, ey)
            else:
                return (min(area[0], sx),
                        min(area[1], sy),
                        max(area[2], ex),
                        max(area[3], ey))

        radius = math.ceil(gui.style.DRAGGABLE_POINT_HANDLE_SIZE)
        for tdw in self._overlays:
            area = get_area(i, node.x, node.y, radius)

            # Active control handles also should be queued.
            for hi in (0,1):
                # But,the first node only shows 2nd(index 1) handle.
                if self.is_drawn_handle(i, hi):
                    handle = node.get_control_handle(hi)
                    area = get_area(i, 
                        handle.x, handle.y,
                        radius, area)

            tdw.queue_draw_area(area[0], area[1], 
                    area[2] - area[0] + 1, 
                    area[3] - area[1] + 1)

    def is_drawn_handle(self, node_idx, hndl_idx):
        return ((hndl_idx == 0 and node_idx > 0) or 
                (hndl_idx == 1 and node_idx <= len(self.nodes)-1))


    def redraw_curve_cb(self, erase=False):
        """ Frontend method,to redraw curve from outside this class"""
        if erase:
            for tdw in self._overlays:
                model = tdw.doc
                self._queue_task(self.brushwork_rollback, model)
                self._queue_task(
                    self.brushwork_begin, model,
                    description=_("Bezier"),
                    abrupt=True,
                )
        else:
            self._queue_redraw_curve()


    def _queue_redraw_curve(self, step = 0.05, pressure_obj=None):
        """Redraws the entire curve on all known view TDWs
        :param step: rendering step of curve.
        The lower this value is,the higher quality bezier curve rendered.
        default value is for draft/editing, 
        to be decreased when render final stroke.
        :param pressure_obj: a pressure-mapping object or None
        """
        self._stop_task_queue_runner(complete=False)

        if pressure_obj == None:
            pressure_obj = self.pressure_map

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
            dx,dy = self.drag_offset.get_model_offset()
            idx = 0
            cnt = float(len(self.nodes) - 1)
            while idx < len(self.nodes) - 1:
                self._queue_task(
                    self._draw_curve_segment,
                    model,
                    idx, idx+1, dx, dy,step,
                    pressure_obj,
                    (idx / cnt, (idx+1) / cnt)
                )
                idx+=1
        self._start_task_queue_runner()

    def _queue_draw_buttons(self):
        # To surpress exception
        if len(self.nodes) >= 2:
            super(BezierMode, self)._queue_draw_buttons()

    def _queue_previous_draw_buttons(self):
        """ Queue previous (current) button position to draw.
        It means erase old position buttons.
        BezierCurveMode changes dramatically button position with its 
        node selection state (due to display control handles),
        so we might miss calcurate button location in some case.
        """

        for tdw, overlay in self._overlays.items():
            for pos in (overlay.get_button_pos(_EditZone.ACCEPT_BUTTON),
                         overlay.get_button_pos(_EditZone.REJECT_BUTTON)):
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


    def _draw_curve_segment(self, model, sidx, eidx, dx, dy, step,
            pressure_src=None, internode_steps=None):
        """Draw the curve segment between the middle two points
        :param step: Rendering step of curve.
        :param pressure_src: Pressure source object,which has 
            get_pressure(step) method.
            this is used for 'pressure variation mapping' feature.
        :param internode_steps: A tuple of 
            (the step of start point in entire stroke,
             the step of end point in entire stroke)
            or None. 
        if either pressure_src or internode_steps is none,
        the 'pressure variation mapping' feature disabled.
        """
        
        
        cur_step = 0.0
        xtilt = 0.0
        ytilt = 0.0
        dtime = 0.5
        # TODO dtime is the big problem.
        # how we decide the speed from static node list?
        
        try:
            p0 = self.nodes[sidx]
            p3 = self.nodes[eidx]
        except IndexError:
            return
        p1 = p0.get_control_handle(1)
        p2 = p3.get_control_handle(0)
        
        offset = (dx, dy)
        if sidx in self.selected_nodes:
            o0 = offset
        else:
            o0 = (0, 0)
        
        if eidx in self.selected_nodes:
            o3 = offset
        else:
            o3 = (0, 0)
        
        def draw_single_segment(cur_step):
            
           #x, y = gui.drawutils.get_cubic_bezier_segment(
           #    p0, p1, p2, p3, cur_step, o0, o3)
            x = gui.drawutils.get_cubic_bezier(
               p0.x + o0[0] , p1.x + o0[0], 
               p2.x + o3[0] , p3.x + o3[0], cur_step)
            y = gui.drawutils.get_cubic_bezier(
               p0.y + o0[1], p1.y + o0[1], 
               p2.y + o3[1], p3.y + o3[1], cur_step)


            if pressure_src and internode_steps:
                pressure_map = pressure_src.get_pressure(
                        gui.drawutils.linear_interpolation(
                            internode_steps[0], internode_steps[1], cur_step)
                        )
            else:
                pressure_map = 1.0
            
            self.stroke_to(
                model, 
                dtime,
                x, y, 
                lib.helpers.clamp(
                    gui.drawutils.linear_interpolation(
                    p0.pressure, p3.pressure, cur_step) * pressure_map,
                    0.0, 1.0), 
                gui.drawutils.linear_interpolation(
                    p0.xtilt, p3.xtilt, cur_step),
                gui.drawutils.linear_interpolation(
                    p0.ytilt, p3.ytilt, cur_step),
                auto_split=False,
            )

        while cur_step < 1.0:
            draw_single_segment(cur_step)            
            cur_step += step

        draw_single_segment(1.0) # Ensure draw the last segment          



    ### Event handling

    ## Raw event handling (prelight & zone selection in adjust phase)
    def button_press_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False
        self._update_zone_and_target(tdw, event.x, event.y,
                event.state & Gdk.ModifierType.MOD1_MASK)
        self._update_current_node_index()

        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
        ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK

        if self.phase == _Phase.INITIAL: 
            self.phase = _Phase.CREATE_PATH
            # FALLTHRU: *do* start a drag 
        elif self.phase in (_Phase.CREATE_PATH,
                _Phase.ADJUST_PRESSURE):
            # Initial state - everything starts here!
       
            if (self.zone in (_EditZone.REJECT_BUTTON, 
                        _EditZone.ACCEPT_BUTTON)):
                if (event.button == 1 and 
                        event.type == Gdk.EventType.BUTTON_PRESS):

                        # To avoid some of visual glitches,
                        # we need to process button here.
                        if self.zone == _EditZone.REJECT_BUTTON:
                            self.discard_edit()
                        elif self.zone == _EditZone.ACCEPT_BUTTON:
                            self.accept_edit()
                        return False
                    
                    
            elif self.zone == _EditZone.CONTROL_NODE:
                # Grabbing a node...
                if self.phase == _Phase.CREATE_PATH:

                    if (self.current_node_index is not None and 
                            event.button == 1 and
                            shift_state):
                        # It's 'Entering On-canvas Pressure Adjustment Phase'!

                        self._returning_phase = self.phase
                        self.phase = _Phase.ADJUST_PRESSURE_ONESHOT
                
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
                        self.phase = _Phase.MOVE_NODE

                        if event.button == 1:
                            if ctrl_state:
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

            elif self.zone == _EditZone.EMPTY_CANVAS:
                
                if self.phase == _Phase.CREATE_PATH:
                    if (len(self.nodes) > 0 and event.button == 1): 
                        
                        if shift_state and ctrl_state:
                            self._queue_draw_buttons() 
                            self.forced_button_pos = (event.x, event.y)
                            self.phase = _Phase.CHANGE_PHASE 
                            self._returning_phase = _Phase.CREATE_PATH
                        elif ctrl_state:
                            mx, my = tdw.display_to_model(event.x, event.y)
                            pressed_segment = self._detect_on_stroke(mx, my)
                            if pressed_segment:
                                # pressed_segment is a tuple which contains
                                # (node index of start of segment, stroke step)

                                # To erase buttons 
                                self._queue_draw_buttons() 

                                self._divide_bezier(*pressed_segment)

                                # queue new node here.
                                self._queue_draw_node(pressed_segment[0] + 1)
                                
                                self.phase = _Phase.PLACE_NODE
                                return False # Cancel drag event

                        ## Otherwise, after this callback
                        ## a new node created in drag_stop_cb.

                elif self.phase == _Phase.ADJUST_PRESSURE:
                    if self._returning_phase == None:
                        self._returning_phase = _Phase.CREATE_PATH
                        self._queue_redraw_all_nodes()
                        self._queue_draw_buttons()
                    self.phase = _Phase.CHANGE_PHASE

            elif self.zone == _EditZone.CONTROL_HANDLE:
                if self.phase == _Phase.CREATE_PATH:
                    self.phase = _Phase.ADJUST_HANDLE


            # FALLTHRU: *do* start a drag 

        elif self.phase in (_Phase.ADJUST_HANDLE, _Phase.INIT_HANDLE):
            pass
        elif self.phase in (_Phase.ADJUST_PRESSURE, 
                _Phase.ADJUST_PRESSURE_ONESHOT):
            # XXX in some cases,ADJUST_PRESSURE phase come here
            # without reaching drag_stop_cb.(it might due to pen tablet...)
            # so ignore this for now,or something should be done here?
            pass 
        elif self.phase in (_Phase.MOVE_NODE, _Phase.CHANGE_PHASE):
            # THIS CANNOT BE HAPPEN...might be an evdev dropout.through it.
            pass
        else:
            raise NotImplementedError("Unrecognized phase %r", self.phase)
        # Update workaround state for evdev dropouts
        self._button_down = event.button

        return super(BezierMode, self).button_press_cb(tdw, event) 

    def button_release_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False

        # Here is 'button_release_cb',which called 
        # prior to drag_stop_cb.
        # so, in this method, changing self._phase
        # is very special case. 
        if self.phase == _Phase.PLACE_NODE:
            self._queue_redraw_curve() 
            self.phase = _Phase.CREATE_PATH
            pass

        # Update workaround state for evdev dropouts
        self._button_down = None

        return super(BezierMode, self).button_release_cb(tdw, event)
        

    ## Drag handling (both capture and adjust phases)
    def drag_start_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        mx, my = tdw.display_to_model(event.x, event.y)

        self._queue_previous_draw_buttons() # To erase button,and avoid glitch

        # Basically,all sections should do fall-through.
        if self.phase == _Phase.CREATE_PATH:

            if self.zone == _EditZone.EMPTY_CANVAS:
                if event.state != 0:
                    # To activate some mode override
                    self._last_event_node = None
                    return super(BezierMode, self).drag_start_cb(tdw, event)
                else:
                    # New node added!
                    node = self._get_event_data(tdw, event)
                    self.nodes.append(node)
                    self._last_event_node = node
                    self.phase = _Phase.INIT_HANDLE
                    self.current_node_index=len(self.nodes)-1
                    self._reset_selected_nodes(self.current_node_index)
                    # Important: with setting initial control handle 
                    # as the 'next' (= index 1) one,it brings us
                    # inkscape-like node creation.
                    self.current_handle_index = 1 

                    self._queue_draw_node(self.current_node_index)

        elif self.phase == _Phase.MOVE_NODE:
            if len(self.selected_nodes) > 0:
                # Use selection_rect class as offset-information
                self.drag_offset.start(mx, my)
        
        elif self.phase in (_Phase.ADJUST_PRESSURE, 
                _Phase.ADJUST_PRESSURE_ONESHOT):
            pass
        elif self.phase == _Phase.ADJUST_HANDLE:
            self._last_event_node = self.nodes[self.target_node_index]
            pass
        elif self.phase == _Phase.CHANGE_PHASE:
            # DO NOT DO ANYTHING.
            pass
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)


    def drag_update_cb(self, tdw, event, dx, dy):
        self._ensure_overlay_for_tdw(tdw)
        mx, my = tdw.display_to_model(event.x, event.y)
        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK

        if self.phase == _Phase.CREATE_PATH:
            pass
            
        elif self.phase in (_Phase.ADJUST_HANDLE, _Phase.INIT_HANDLE):
            node = self._last_event_node
            if self._last_event_node:
                self._queue_draw_node(self.current_node_index)# to erase
                node.set_control_handle(self.current_handle_index,
                        mx, my,
                        shift_state)

                self._queue_draw_node(self.current_node_index)
            self._queue_redraw_curve()
                
        elif self.phase == _Phase.MOVE_NODE:
            if len(self.selected_nodes) > 0:
                self._queue_draw_selected_nodes()
                self.drag_offset.end(mx, my)
                self._queue_draw_selected_nodes()
                self._queue_redraw_curve()
        elif self.phase in (_Phase.ADJUST_PRESSURE,
                _Phase.ADJUST_PRESSURE_ONESHOT):
            self._adjust_pressure_with_motion(dx, dy)
        elif self.phase == _Phase.CHANGE_PHASE:
            # DO NOT DO ANYTHING.
            pass
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)

    def drag_stop_cb(self, tdw):
        self._ensure_overlay_for_tdw(tdw)
        if self.phase == _Phase.CREATE_PATH:
            self._reset_adjust_data()
            if len(self.nodes) > 0:
                self._queue_redraw_curve()
                self._queue_redraw_all_nodes()
                if len(self.nodes) > 1:
                    self._queue_draw_buttons()
                
            
        elif self.phase in (_Phase.ADJUST_HANDLE, _Phase.INIT_HANDLE):
            node = self._last_event_node
      
            # At initialize handle phase, even if the node is not 'curve'
            # Set the handles as symmetry.
            if (self.phase == _Phase.INIT_HANDLE and 
                    len(self.nodes) > 1 and node.curve == False):
                node.curve = True 
                node.curve = False

            self._queue_redraw_all_nodes()
            self._queue_redraw_curve()
            if len(self.nodes) > 1:
                self._queue_draw_buttons()
                
            self.phase = _Phase.CREATE_PATH
        elif self.phase == _Phase.MOVE_NODE:
            dx, dy = self.drag_offset.get_model_offset()

            for idx in self.selected_nodes:
                cn = self.nodes[idx]
                cn.move(cn.x + dx, cn.y + dy)

            self.drag_offset.reset()
            self._dragged_node_start_pos = None
            self._queue_redraw_curve()
            self._queue_draw_buttons()
            self.phase = _Phase.CREATE_PATH
        elif self.phase in (_Phase.ADJUST_PRESSURE, 
                _Phase.ADJUST_PRESSURE_ONESHOT):
            self._queue_draw_buttons()
        elif self.phase == _Phase.CHANGE_PHASE:
            pass


        # Common processing
        if self.current_node_index != None:
            self.options_presenter.target = (self, self.current_node_index)

        if self._returning_phase != None:
            self.phase = self._returning_phase
            self._returning_phase = None
            self._queue_draw_buttons() 

    ## Interrogating events

    def _get_event_data(self, tdw, event):
        """ Overriding mixin method.
        
        almost same as inktool,but we needs generate _Node_Bezier object
        not _Node object
        """
        x, y = tdw.display_to_model(event.x, event.y)
        xtilt, ytilt = self._get_event_tilt(tdw, event)
        return _Node_Bezier(
            x=x, y=y,
            pressure=self._DEFAULT_PRESSURE,
            xtilt=xtilt, ytilt=ytilt,
            dtime=self._DEFAULT_DTIME,
            )


    ## Node editing


    @property
    def pressure_map(self):
        """pressure map object for stroke drawing with pressure mapping"""
        cls = self.__class__
        if cls._PRESSURE_MAP is None:
            cls._PRESSURE_MAP = PressureMap(self.options_presenter.curve)
        return cls._PRESSURE_MAP

    def _divide_bezier(self, index, step):
        """ Divide (insert a node intermidiate stroke)
        to current active bezier stroke 
        without shape change.
        """
        assert index < len(self.nodes)-1
        cn = self.nodes[index]
        nn = self.nodes[index+1]
        

        xa, xb, xc=gui.drawutils.get_cubic_bezier_raw(
                cn.x, cn.get_control_handle(1).x,
                nn.get_control_handle(0).x, nn.x,
                step)
        ya, yb, yc=gui.drawutils.get_cubic_bezier_raw(
                cn.y, cn.get_control_handle(1).y,
                nn.get_control_handle(0).y, nn.y,
                step)

        xd, xe=gui.drawutils.get_bezier_raw(xa, xb, xc, step)
        yd, ye=gui.drawutils.get_bezier_raw(ya, yb, yc, step)


        # The nodes around a new node changed to 'not curve' node,
        # to retain original shape.
        cn.curve = False
        cn.set_control_handle(1, xa, ya, False)
        new_node = _Node_Bezier(
                    gui.drawutils.linear_interpolation(xd, xe, step), 
                    gui.drawutils.linear_interpolation(yd, ye, step),
                    pressure = cn.pressure + ((nn.pressure - cn.pressure) * step),
                    xtilt = cn.xtilt + (nn.xtilt - cn.xtilt) * step,
                    ytilt = cn.ytilt + (nn.ytilt - cn.ytilt) * step,
                    dtime = self._DEFAULT_DTIME,
                    curve = False)
        new_node.set_control_handle(0, xd, yd, False)
        new_node.set_control_handle(1, xe, ye, False)
        self.nodes.insert(index + 1, new_node)

        nn.curve = False
        nn.set_control_handle(0, xc, yc, False)


    def can_delete_node(self, idx):
        """ differed from InkingMode,
        BezierMode can delete the last node.
        """ 
        return 1 <= idx < len(self.nodes)

    def _adjust_current_node_index(self):
        """ Adjust self.current_node_index
        child classes might have different behavior
        from Inktool about current_node_index.
        """
        if self.current_node_index >= len(self.nodes):
            self.current_node_index = None
            self.current_node_changed(
                    self.current_node_index)
                                                
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
            ytilt=(cn.ytilt + nn.ytilt) / 2.0,
            dtime=self._DEFAULT_DTIME,
            curve=not self.DEFAULT_POINT_CORNER
        )
        self.nodes.insert(i+1,newnode)

        # Issue redraws for the changed on-canvas elements
        self._queue_redraw_curve()
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()

    def apply_pressure_from_curve_widget(self):
        """ apply pressure reprenting points
        from StrokeCurveWidget.
        """

        # We need smooooth value, so treat the points
        # as Bezier-curve points.


        if len(self.nodes) < 2:
            return

        assert hasattr(self.options_presenter,'curve')
        curve = self.options_presenter.curve

        self._queue_redraw_curve()

        # First of all, get the entire stroke length
        # to normalize stroke.

        node_length=[]
        total_length = 0.0

        for idx, cn in enumerate(self.nodes[:-1]):
            nn = self.nodes[idx + 1]
            ox, oy = gui.drawutils.get_cubic_bezier_segment(cn, cn.get_control_handle(1),
                        nn.get_control_handle(0), nn, 0)
            cur_step = BezierMode.DRAFT_STEP
            length = 0.0 
            while cur_step < 1.0:
                cx, cy = gui.drawutils.get_cubic_bezier_segment(cn, cn.get_control_handle(1),
                            nn.get_control_handle(0), nn, cur_step)
                length += vector_length(cx - ox, cy - oy)
                cur_step += BezierMode.DRAFT_STEP
                ox = cx
                oy = cy

            node_length.append(length)
            total_length+=length

        node_length.append(total_length) # this is sentinel


        # use control handle class temporary to get smooth pressures.
        cur_length = 0.0
        for idx,cn in enumerate(self.nodes):
            cn.pressure = curve.get_curve_value(cur_length / total_length)
            cur_length += node_length[idx]
        
        self._queue_redraw_curve()


    def delete_selected_nodes(self):
        """ Beziertool can delete any nodes...
        even first / last one!
        """

        # First of all,queue redraw area.
        self._queue_draw_buttons()
        for idx in self.selected_nodes:
            self._queue_draw_node(idx)

        self._queue_redraw_curve()

        # after then,delete it.
        new_nodes = []
        for idx,cn in enumerate(self.nodes):
            if idx in self.selected_nodes:
                if self.current_node_index == idx:
                    self.current_node_index = None
            else:
                new_nodes.append(cn)

        self.nodes = new_nodes
        self._reset_selected_nodes()
        self._adjust_current_node_index()
        self.target_node_index = None

        # Issue redraws for the changed on-canvas elements
        if len(self.nodes) <= 1:
            if len(self.nodes) == 0:
                self.phase = _Phase.INITIAL
            self.redraw_curve_cb(True)
        else:
            self._queue_redraw_curve()

        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()

    def recall_nodes(self, idx):
        """ recall nodes from history
        """
        if 0 < idx < len(self.stroke_history.liststore):
            self._queue_draw_buttons()
            self._queue_redraw_all_nodes()
            self._queue_redraw_curve()

            self._stroke_from_history = True

            x = y = None
            if len(self.nodes) > 0:
                x = self.nodes[0].x
                y = self.nodes[0].y

            self.nodes = self.stroke_history.get_and_place_nodes(
                    idx, x, y)

            self._queue_redraw_curve()
            self._queue_redraw_all_nodes()
            self._queue_draw_buttons()
            self.phase = _Phase.CREATE_PATH

    def toggle_current_node_curve(self):
        """ mainly called from Callback of gui.document
        """
        if (self.phase in (_Phase.CREATE_PATH, ) and
                len(self.nodes) > 0 and
                self.current_node_index != None):
            cn = self.nodes[self.current_node_index]
            cn.curve = not cn.curve
            self.options_presenter.set_checkbutton_curvepoint(
                    cn.curve)


    ## Generic Oncanvas-editing handler

    def accept_edit(self):
        if (self.phase in (_Phase.ADJUST ,_Phase.ADJUST_PRESSURE) and
                len(self.nodes) > 1):
            self._queue_redraw_curve(BezierMode.FINAL_STEP) # Redraw with hi-fidely curve
            self._start_new_capture_phase(rollback=False)

    def discard_edit(self):
        if (self.phase in (_Phase.ADJUST ,_Phase.ADJUST_PRESSURE)):
            self._start_new_capture_phase(rollback=True)

class OverlayBezier (OverlayOncanvasMixin):
    """Overlay for an BezierMode's adjustable points"""

    def __init__(self, mode, tdw):
        super(OverlayBezier, self).__init__(mode, tdw)
        self._draw_initial_handle_both = False
        
    def update_button_positions(self, ignore_control_handle=False):
        """Recalculates the positions of the mode's buttons."""
        # FIXME mostly copied from inktool.Overlay.update_button_positions
        # The difference is for-loop of nodes , to deal with control handles.
        mode = self._mode
        nodes = mode.nodes
        num_nodes = len(nodes)
        if num_nodes == 0:
            self.button_pos[_EditZone.ACCEPT_BUTTON] = None
            self.button_pos[_EditZone.REJECT_BUTTON] = None
            return False

        button_radius = gui.style.FLOATING_BUTTON_RADIUS
        alloc = self._tdw.get_allocation()
        view_x0, view_y0 = alloc.x, alloc.y
        view_x1, view_y1 = view_x0+alloc.width, view_y0+alloc.height
        
        def adjust_button_inside(cx, cy, radius):
            if cx + radius > view_x1:
                cx = view_x1 - radius
            elif cx - radius < view_x0:
                cx = view_x0 + radius
            
            if cy + radius > view_y1:
                cy = view_y1 - radius
            elif cy - radius < view_y0:
                cy = view_y0 + radius
            return cx, cy

        if mode.forced_button_pos:
            # User deceided button position 
            cx, cy = mode.forced_button_pos
            margin = 1.5 * button_radius
            area_radius = 64 + margin #gui.style.FLOATING_TOOL_RADIUS

            cx, cy = adjust_button_inside(cx, cy, area_radius)

            pos_list = []
            count = 2
            for i in range(count):
                rad = (math.pi / count) * 2.0 * i
                x = - area_radius*math.sin(rad)
                y = area_radius*math.cos(rad)
                pos_list.append( (x + cx, - y + cy) )

            self._button_pos[_EditZone.ACCEPT_BUTTON] = \
                    (pos_list[0][0], pos_list[0][1])
            self._button_pos[_EditZone.REJECT_BUTTON] = \
                    (pos_list[1][0], pos_list[1][1])
        else:
            # Usually, Bezier tool needs to keep extending control points.
            # So when buttons placed around the tail(newest) node, 
            # it is something frastrating to manipulate new node...
            # Thus,different to Inktool, place buttons around 
            # the first(oldest) nodes.
            
            node = nodes[0]
            cx, cy = self._tdw.model_to_display(node.x, node.y)
            margin = 2.0 * button_radius
               
            if ignore_control_handle:
                dx = margin
                dy = margin
            else:
                handle = node.get_control_handle(1)
                nx, ny = self._tdw.model_to_display(handle.x, handle.y)

                vx = nx - cx
                vy = ny - cy
                s  = math.hypot(vx, vy)
                if s > 0.0:
                    vx /= s
                    vy /= s
                else:
                    vx = 0.0
                    vy = 1.0


                # reverse vx, vy, to get right-angled position.
                dx = vy * margin
                dy = vx * margin
            
            self._button_pos[_EditZone.ACCEPT_BUTTON] = \
                    adjust_button_inside(cx + dx, cy - dy, button_radius * 1.5)
            self._button_pos[_EditZone.REJECT_BUTTON] = \
                    adjust_button_inside(cx - dx, cy + dy, button_radius * 1.5)

        return True

    def paint_control_handle(self, cr, i, node, x, y, dx, dy, draw_line):
        """ Paint Control Handles
        :param x,y: center(node) position,in display coordinate
        :param dx,dy: delta x/y, these are used for moving control handle
        :param boolean draw_line: draw line of handle, when this is True
        """
        cr.save()
        cr.set_line_width(1)
        for hi in (0,1):                        
            if ((hi == 0 and i > 0) or
                    (hi == 1 and i <= len(self._mode.nodes)-1) or 
                    self._draw_initial_handle_both): 
                ch = node.get_control_handle(hi)
                hx, hy = self._tdw.model_to_display(ch.x, ch.y)
                hx += dx
                hy += dy
                gui.drawutils.render_square_floating_color_chip(
                    cr, hx, hy,
                    gui.style.ACTIVE_ITEM_COLOR, 
                    gui.style.DRAGGABLE_POINT_HANDLE_SIZE,
                    fill=(hi==self._mode.current_handle_index)) 
                if draw_line:
                    cr.set_source_rgb(0,0,0)
                    cr.set_dash((), 0)
                    cr.move_to(x, y)
                    cr.line_to(hx, hy)
                    cr.stroke_preserve()
                    cr.set_source_rgb(1,1,1)
                    cr.set_dash((3, ))
                    cr.stroke()

        cr.restore()

    
    def paint(self, cr, draw_buttons=True):
        """Draw adjustable nodes to the screen"""
        mode = self._mode
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        alloc = self._tdw.get_allocation()
        dx, dy = mode.drag_offset.get_display_offset(self._tdw)
        fill_flag = not mode.phase in (_Phase.ADJUST_PRESSURE,
                _Phase.ADJUST_PRESSURE_ONESHOT)

        for i, node, x, y in self._get_onscreen_nodes():
            show_node = not mode.hide_nodes
            show_handle = False
            color = gui.style.EDITABLE_ITEM_COLOR

            if mode.phase in (_Phase.INITIAL, 
                    _Phase.MOVE_NODE, 
                    _Phase.CREATE_PATH, 
                    _Phase.ADJUST_PRESSURE,
                    _Phase.ADJUST_PRESSURE_ONESHOT,
                    _Phase.ADJUST_HANDLE, 
                    _Phase.INIT_HANDLE):
                if show_node:
                    if i == mode.current_node_index:
                        color = gui.style.ACTIVE_ITEM_COLOR
                        show_handle = True                 
                                  
                    elif i == mode.target_node_index:
                        color = gui.style.PRELIT_ITEM_COLOR
                    elif i in mode.selected_nodes:
                        color = gui.style.POSTLIT_ITEM_COLOR

                else:
                    if i == mode.current_node_index:
                        # Drawing control handle
                        if mode.zone == _EditZone.CONTROL_HANDLE:
                            show_node = True
                            color = gui.style.ACTIVE_ITEM_COLOR
                                  
                    # not 'elif' ... because target_node_index
                    # and current_node_index maight be same.
                    if i == mode.target_node_index:
                        show_node = True
                        show_handle = True
                        color = gui.style.PRELIT_ITEM_COLOR

                if (color != gui.style.EDITABLE_ITEM_COLOR and
                        mode.phase in (_Phase.CREATE_PATH,
                                       _Phase.MOVE_NODE,
                                       _Phase.ADJUST)):
                    x += dx
                    y += dy
      
            if show_node:
                gui.drawutils.render_round_floating_color_chip(
                    cr=cr, x=x, y=y,
                    color=color,
                    radius=radius,
                    fill=fill_flag
                )
                if show_handle and fill_flag:
                    self.paint_control_handle(cr, i, node, 
                            x, y, dx, dy, True)
                                    
    
                
        # Buttons
        if (draw_buttons and 
                not mode.in_drag and len(self._mode.nodes) > 1):
            self.update_button_positions()
            button_info = [
                (
                    "mypaint-ok-symbolic",
                    self._button_pos[_EditZone.ACCEPT_BUTTON],
                    _EditZone.ACCEPT_BUTTON,
                ),
                (
                    "mypaint-trash-symbolic",
                    self._button_pos[_EditZone.REJECT_BUTTON],
                    _EditZone.REJECT_BUTTON,
                ),
            ]
            self._draw_buttons(cr, button_info)
        

                


class OptionsPresenter_Bezier (OptionsPresenter):
    """Presents UI for directly editing point values etc."""

    def __init__(self):
        super(OptionsPresenter_Bezier, self).__init__()

    def _ensure_ui_populated(self):
        if self._options_grid is not None:
            return
        self._updating_ui = True
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

        self._default_dtime_scale = builder.get_object("default_dtime_scale")
        self._default_dtime_scale.set_sensitive(True)
        self._default_pressure_scale = builder.get_object("default_pressure_scale")
        self._default_pressure_scale.set_sensitive(True)
        self._default_dtime_adj = builder.get_object("default_dtime_adj")
        self._default_pressure_adj = builder.get_object("default_pressure_adj")

        self._default_dtime_adj.set_value(self._app.preferences.get(
            "beziertool.default_dtime", BezierMode._DEFAULT_DTIME))
        self._default_pressure_adj.set_value(self._app.preferences.get(
            "beziertool.default_pressure", BezierMode._DEFAULT_PRESSURE))

        combo = builder.get_object('stroke_history_combobox')
        combo.set_model(BezierMode.stroke_history.liststore)
        cell = Gtk.CellRendererText()
        combo.pack_start(cell,True)
        combo.add_attribute(cell,'text',0)
        self._stroke_history_combo = combo

        base_grid = builder.get_object("points_editing_grid")
        self.init_linecurve_widget(0, base_grid)
        self.init_variation_preset_combo(1, base_grid)
        self._updating_ui = False

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
        finally:
            self._updating_ui = False                               

    def set_checkbutton_curvepoint(self, flag):
        """ called from BezierMode object,
        when current node curve status changed.
        """

        # avoid cancelling other ongoing ui updating
        entering_updating_ui = self._updating_ui

        self._updating_ui = True
        self._check_curvepoint.set_active(flag)
        self._updating_ui = entering_updating_ui

   #def set_default_values(self, dtime, pressure):
   #    """ called from BezierMode object,
   #    when default dtime or pressure changed.
   #    """
   #
   #    # avoid cancelling other ongoing ui updating
   #    entering_updating_ui = self._updating_ui
   #
   #    self._updating_ui = True
   #    if dtime:
   #        self._default_dtime_adj.set_value(dtime)
   #    if pressure:
   #        self._default_pressure_adj.set_value(pressure)
   #    self._updating_ui = entering_updating_ui

    def set_checkbutton_curvepoint(self, flag):
        """ called from BezierMode object,
        when current node curve status changed.
        """

        # avoid cancelling other ongoing ui updating
        if self._updating_ui: 
            self._check_curvepoint.set_active(flag)
            return

        self._updating_ui = True
        self._check_curvepoint.set_active(flag)
        self._updating_ui = False

    def reset_stroke_history(self):
        self._updating_ui = True
        self._stroke_history_combo.set_active(0)
        self._updating_ui = False

    def checkbutton_curvepoint_toggled_cb(self, button):
        if self._updating_ui:
            return
        beziermode, node_idx = self.target
        if beziermode:
            if 0 <= node_idx < len(beziermode.nodes):
                beziermode._queue_draw_node(node_idx) 
                beziermode._queue_redraw_curve()
                beziermode.nodes[node_idx].curve = button.get_active()
                beziermode._queue_draw_node(node_idx) 
                beziermode._queue_redraw_curve()


    def _variation_preset_combo_changed_cb(self, widget):
        if self._updating_ui:
            return
        super(OptionsPresenter_Bezier, self)._variation_preset_combo_changed_cb(widget)
        beziermode, node_idx = self.target
        if beziermode:
            beziermode.redraw_curve_cb()

    def _default_dtime_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        self._app.preferences['beziertool.default_dtime'] = adj.get_value()
        BezierMode.set_default_dtime(adj.get_value())

    def _default_pressure_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        self._app.preferences['beziertool.default_pressure'] = adj.get_value()
        BezierMode.set_default_pressure(adj.get_value())
    
    def stroke_history_combobox_changed_cb(self, widget):
        if self._updating_ui:
            return
        beziermode, node_idx = self.target
        if beziermode:
            beziermode.recall_nodes(self._stroke_history_combo.get_active())

    ## Other handlers are as implemented in superclass.  
