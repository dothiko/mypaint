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
import struct # XXX for `node pick` 
import zlib

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
from gui.exinktool import *
from gui.exinktool import _LayoutNode
from gui.linemode import *
from gui.oncanvas import *
import gui.pickable as pickable

## Function defs

def detect_on_stroke(nodes, x, y, allow_distance = 4.0):
    """Detecting pressed point is on the stroke currently editing.
    To share this code between other modules, this is implemented
    as global function, not a class method.
    
    :param nodes: a sequence of nodes. 
                  nodes must have get_control_handle method.
    :param x: cursor x position in MODEL coord
    :param y: cursor y position in MODEL coord
    :param allow_distance: the allowed distance from stroke.
    :return : a tuple of (the index of 'previous' node, time parameter of stroke)
    :rtype : a tuple when the pressed point is on stroke, otherwise
             None.
    
    """

    # XXX Transplant from https://gist.github.com/MadRabbit/996893
    def find_x_for(p0, p1, p2, p3, tx, init):
        t=init
        x=t 
        i=0
        while i < 5: # making 5 iterations max
            z = gui.drawutils.get_cubic_bezier(
                    p0, p1, p2, p3, x) - tx

            if abs(z) < 0.0000001:
                break # if already got close enough

            dx = gui.drawutils.get_diff_cubic_bezier(
                  p0, p1, p2, p3, x)
            if dx == 0.0:
                break

            x = x - z / dx
            i+=1

        return x # try any of x

    for i,cn in enumerate(nodes[:-1]):
        # Get boundary rectangle,to reduce processing segment
        nn = nodes[i+1]
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
            # cursor is inside the bezier segment.
            c=0
            t=1.0
            while c < 2:
                t = find_x_for(p0, p1, p2, p3, x, t)
                cy = gui.drawutils.get_cubic_bezier(q0, q1, q2, q3, t)
                if abs(y - cy) < allow_distance:
                    # the timepoint Found!
                    return (i, t)
                t = 0.0
                c+=1

    # Fallthrough: return None when failed.

## Class defs

class _Control_Handle(object):
    """To store Control handle positon. 
    Also this works as `memory view` of array object in _Node_Bezier.
    """
    def __init__(self, source_info):
        self._array, self._offset = source_info

    def __getitem__(self, idx):
        return self._array[idx+self._offset]

    @property
    def x(self):
        return self._array[0+self._offset]
    @x.setter
    def x(self,x):
        self._array[0+self._offset]=x

    @property
    def y(self):
        return self._array[1+self._offset]
    @y.setter
    def y(self,y):
        self._array[1+self._offset]=y
    
class _Node_Bezier (object):
    """Node (Control point) class,with handle.

    In bezier curve,nodes would be frequently rewritten.
    So this class is object, not a namedtuple.

    _Node_Bezier have the following 6 fields as array object, in order

    * x, y: model coords, float
    * pressure: float in [0.0, 1.0]
    * xtilt, ytilt: float in [-1.0, 1.0]
    * time : time for brushstroke. We cannot use actual bezier node edted time,
             so this is psuedo value.
    
    In addition to these datas, there are control node handle datas. 
    """

    # XXX for `node pick`
    # Node type id. This is used for node pickable strokemap. 
    # This MUST be unique, 32bit unsigned value.
    # And type_id 0 means `generic namedtuple node`,
    # which is used in inktool.
    type_id = 0x00000001
    # XXX for `node pick` end
            
    def __init__(self, 
                 x=0.0, y=0.0, pressure=1.0, 
                 xtilt=0.0, ytilt=0.0, dtime=0.5,
                 curve=True, source=None):
        """
        :param source: A sequence of 10 double values, used as initial value.
        :param source_arrays: Not initial values, that arrays used internal 
                              buffer of this object.
                              This param is tuple of three array objects.
        """

        # No supercall, create this class own array.
        if source is not None:
            _array = array.array('d', source)
        else:
            _array = array.array(
                'd', 
                (x, y, pressure, xtilt, ytilt, dtime, x, y, x, y)
            )

        self._array = _array
        self._control_handles = (
            _Control_Handle(source_info=(_array, 6)),
            _Control_Handle(source_info=(_array, 8))
        )
        self._curve = curve

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

    @property
    def pressure(self):
        return self._array[2]
    @pressure.setter
    def pressure(self, v):
        self._array[2] = v

    @property
    def xtilt(self):
        return self._array[3]
    @xtilt.setter
    def xtilt(self, v):
        self._array[3] = v

    @property
    def ytilt(self):
        return self._array[4]
    @ytilt.setter
    def ytilt(self, v):
        self._array[4] = v

    @property
    def time(self):
        return self._array[5]

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

    def set_control_handle_offset(self, idx, dx, dy, invert_curve_flag):
        """Use this method to set control handle.
        This method refers self._curve flag,
        and if it is True,automatically make handles
        as symmetry = 'Curved bezier control point'
        """

        self._control_handles[idx].x += dx 
        self._control_handles[idx].y += dy

        curve = self._curve
        if invert_curve_flag:
            curve = not curve

        if curve:
            tidx = (idx + 1) % 2
            self._control_handles[tidx].x -= dx
            self._control_handles[tidx].y -= dy

    def get_control_handle(self,idx):
        assert 0 <= idx <= 1
        return self._control_handles[idx]

   #def _replace(self, **kwarg):
   #    ox = self.x
   #    oy = self.y
   #    for ck in kwarg:
   #        if ck in self.__dict__:
   #            self.__dict__[ck] = kwarg[ck]
   #    self._refresh_control_handles(ox, oy, self.x, self.y)
   #    return self

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

    def move(self,x, y, relative=False):
        """Move this node to (x,y).
        :param x: destination x position,in model
        :param y: destination y position,in model

        Use this method to move control points simultaneously.
        """
        if relative:
            self.x += x
            self.y += y
            for i in (0,1):
                self._control_handles[i].x += x
                self._control_handles[i].y += y
        else:
            self._refresh_control_handles(self.x, self.y, x, y)
            self.x = x
            self.y = y

    def copy(self, dx=0, dy=0, node=None):
        """Copy all information to node,with (dx,dy) offset.
        :param dx:offset x,in model
        :param dy:offset y,in model
        :param node: the _Node_Bezier object copy to
        """
        if node is None:
            node = _Node_Bezier(0, 0, source_array=self._array)
        node.x += dx
        node.y += dy
        node._curve = self._curve
        for i in (0,1):
            node._control_handles[i].x = self._control_handles[i].x + dx
            node._control_handles[i].y = self._control_handles[i].y + dy
        return node

    def __getitem__(self, idx):
        return self._array[idx]

    def serialize(self):
        """Serialize node datas as byte string.
        Used for saving strokenode into a file.
        Compression should be done against entire stroke node datas
        for better compression rate,
        therefore just make bytestring here.
        """

        # We need to use struct.pack, instead of array.tostring.
        # Because this data would be used by not only picking nodes
        # also saving into file.
        # We must ensure endian to save them into file.
        data = struct.pack(">10d", *self._array.tolist())
        data += struct.pack(">b", 1 if self._curve else 0)
        return data

    @classmethod
    def fromstring(cls, raw_node_bytes, idx):
        data_length = 10 * 8
        a = struct.unpack(">10d", raw_node_bytes[idx: idx+data_length])
        curve = struct.unpack(">b", raw_node_bytes[idx+data_length]) != 0
        node = _Node_Bezier(curve=curve, source=a)
        return (data_length+1, node)


_EditZone = EditZoneMixin # Same as basic EditZoneMixin at gui/oncanvas.py

class _Phase(PressPhase):
    """Enumeration of the states that an BezierCurveMode can be in"""
    ADJUST_HANDLE = 103     #: change control-handle position

    INIT_HANDLE = 104       #: Initialize control handle phase,right after create a node.
                            #  This phase looks like completely same as ADJUST_HANDLE,
                            #  but this phase needed for very important post processing,
                            #  'After new node created, curve flag should be set'.
                            #  To do this,we need to distinguish whether current phase
                            #  is new node created or existing node edited.

    PLACE_NODE = 105        #: place a new node into clicked position on current
                            # stroke,when you click with holding CTRL key
    CALL_BUTTONS = 106      #: show action buttons around the clicked point. 


class _ActionButton(ActionButtonMixin):
    """Enumeration for the action button of ExInkingMode
    """
    EDIT=2 # To toggle editing node-pressure phase.

class _Prefname:
    """Constants of the keys for app.preferences.
    To share this constants between multiple classes."""
    INITIAL_PRESSURE = "beziertool.initial_pressure"
    DEFAULT_DTIME = 'beziertool.default_dtime'
        
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


class BezierMode (PressureEditableMixin,
                  HandleNodeUserMixin):

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
        """Setting drag-related handler cursor.
        Called by DragMode._start_drag() method. """
        if self.phase == _Phase.ADJUST:
            if self.zone == _EditZone.CONTROL_NODE:
                return self._crosshair_cursor
        elif self.phase == _Phase.ADJUST_POS:
            if self.zone == _EditZone.CONTROL_NODE:
                return self._crosshair_cursor
            elif self.zone != _EditZone.EMPTY_CANVAS: # assume button
                return self._arrow_cursor
    
        elif self.phase == _Phase.ADJUST_PRESSURE:
            if self.zone == _EditZone.CONTROL_NODE:
                return self._cursor_move_nw_se

        # There is no cursor setting for phases which does not support drag.
        # Such phase will need to setup at self.update_cursor_cb()
        return None  

    ## Action buttons
    buttons = {
        ActionButtonMixin.ACCEPT : ('mypaint-ok-symbolic', 
            'accept_button_cb'), 
        _ActionButton.REJECT : ('mypaint-trash-symbolic', 
            'reject_button_cb'),
        _ActionButton.EDIT : ('mypaint-edit-symbolic', 
            'edit_button_cb') 
    }

    ## Phase related
    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, new_phase):
        self._phase = new_phase

    def enter_insert_node_phase(self):
        """ Entering insert node phase from Gtk.Action.
        """
        if len(self.nodes) > 2:
            if self.phase == _Phase.INSERT_NODE:
                self.phase = _Phase.ADJUST
                self.app.show_transient_message(_("Toggled to adjust phase."))
            else:
                self.phase = _Phase.INSERT_NODE
                self.app.show_transient_message(_("Entering insert node phase."))

            for tdw in self._overlays:
                self._update_cursor(tdw) 
        else:
            self.app.show_transient_message(_("There is no stroke.Cannot enter insert phase."))

    def is_adjusting_phase(self):
        return self.phase in (_Phase.ADJUST,
                              _Phase.ADJUST_POS,
                              _Phase.ADJUST_PRESSURE,
                              _Phase.ADJUST_PRESSURE_ONESHOT,
                              _Phase.ADJUST_HANDLE,
                              _Phase.ADJUST_ITEM)

    def is_pressure_modifying(self):
        return self.phase in (_Phase.ADJUST_PRESSURE,
                              _Phase.ADJUST_PRESSURE_ONESHOT,
                              _Phase.ADJUST_ITEM)

    @classmethod
    def set_default_dtime(cls, new_dtime):
        cls._DEFAULT_DTIME = new_dtime

    @classmethod
    def set_initial_pressure(cls, new_pressure):
        cls._INITIAL_PRESSURE = new_pressure

    @property
    def initial_pressure(self):
        cls = self.__class__
        if cls._INITIAL_PRESSURE is None:
            cls._INITIAL_PRESSURE = \
                    self.app.preferences.get(_Prefname.INITIAL_PRESSURE, 
                                             1.0)
        return cls._INITIAL_PRESSURE

    def enter(self, doc, **kwds):
        """Enters the mode: called by `ModeStack.push()` etc."""
        super(BezierMode, self).enter(doc, **kwds)
        cursors = self.app.cursors
        self._insert_cursor = cursors.get_action_cursor(
            self.ACTION_NAME,
            gui.cursor.Name.ADD,
        )

    ## Class config vars

    DRAFT_STEP = 0.01 # Draft(Editing) Bezier curve stroke step.
    FINAL_STEP = 0.005 # Final output stroke Bezier-curve step.

    _INITIAL_PRESSURE = None # default bezier pressure,this is fixed value.
                             # because it is hard to control pressure 
                             # for human hand at node creation.

    _DEFAULT_DTIME = 0.5 # default dtime value

    DEFAULT_POINT_CORNER = False       # default point is curve,not corner

    INITIAL_NODE_HANDLE_RANGE = (1, ) # Overriding HandleNodeUserMixin class attr
                                      # to Ignore the first control handle
    ## Other class vars

    _PRESSURE_MAP = None #: Pressure mapping object singleton

    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):
        super(BezierMode, self).__init__(**kwargs)
       #self._init_recall()
        self.forced_button_pos = None

    def _reset_capture_data(self):
        """ Reset capture related data.
        This method would be called the initializing codes of
        capture phase.
        """
        super(BezierMode, self)._reset_capture_data()
        self.phase = _Phase.ADJUST
        self._pressure_modified = False

    def _reset_adjust_data(self):
        """ Reset adjust related data.
        This method would be called each time the adjust phase has reached to
        node_drag_stop_cb()
        """
        super(BezierMode, self)._reset_adjust_data()
        self.current_node_handle = None
    
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

        if self.current_node_index != None:
            self._queue_draw_node(None, self.current_node_index) # To Erase

        self.current_node_index = new_index
        self.current_node_changed(new_index)
        self.options_presenter.target = (self, new_index)
        for i in (old_index, new_index):
            if i is not None:
                self._queue_draw_node(None, i)

        self._queue_draw_buttons()

    def _update_zone_and_target(self, tdw, x, y):
        """Update the zone and target node under a cursor position
        """
        super(BezierMode, self)._update_zone_and_target(
                tdw, x, y)

        if self.phase == _Phase.INSERT_NODE:
            self._update_cursor(tdw) 

    def update_cursor_cb(self, tdw): 
        """  Update the cursor for not dragging mode.
        """
        cursor = None
       #if self.is_editing_phase():
        if self.is_adjusting_phase():
            if self.zone != _EditZone.EMPTY_CANVAS: # assume button
                cursor = self._arrow_cursor
            else:
                cursor = self._crosshair_cursor
        elif self.phase == _Phase.INSERT_NODE:
            cursor = self._insert_cursor

        return cursor

    def _start_new_capture_phase(self, rollback=False):
        super(BezierMode, self)._start_new_capture_phase(rollback)
        self.forced_button_pos = None

    ## Redraws
    def _queue_draw_node(self, tdw, i, offsets=None):
        """This method might called from baseclass,
        so we need to call HandleNodeUserMixin method explicitly.
        """
        if offsets is None:
            if i in self.selected_nodes:
                offsets = self.drag_offset.get_model_offset() 

        return self._queue_draw_handle_node(tdw, i, offsets)

    def _queue_draw_selected_nodes(self, tdw):
        if tdw is None:
            for tdw in self._overlays:
                self._queue_draw_selected_nodes(tdw)
            return

        offsets = self.drag_offset.get_model_offset() 

        for i in self.selected_nodes:
            self._queue_draw_node(tdw, i, offsets)

    def redraw_item_cb(self, erase=False):
        """ Frontend method,to redraw item (for example, it is stroke curve
        in ExInkingMode) from outside this class"""
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
            self._queue_redraw_item()

    def _queue_redraw_item(self, step = 0.05, pressure_obj=None):
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
        if len(self.nodes) >= 2:
            super(BezierMode, self)._queue_draw_buttons()

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
            
            viewzoom = self.doc.tdw.scale
            viewrotation = self.doc.tdw.rotation  
                        
            self.stroke_to(
                model, dtime,
                x, y, 
                lib.helpers.clamp(
                    gui.drawutils.linear_interpolation(
                    p0.pressure, p3.pressure, cur_step) * pressure_map,
                    0.0, 1.0), 
                gui.drawutils.linear_interpolation(
                    p0.xtilt, p3.xtilt, cur_step),
                gui.drawutils.linear_interpolation(
                    p0.ytilt, p3.ytilt, cur_step),
                viewzoom, viewrotation,
                auto_split=False,
            )

        while cur_step < 1.0:
            draw_single_segment(cur_step)            
            cur_step += step

        draw_single_segment(1.0) # Ensure draw the last segment          

    ### Event handling

    ## Raw event handling (prelight & zone selection in adjust phase)
    def mode_button_press_cb(self, tdw, event):

        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
        ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK
        mod_state = event.state & self.ONCANVAS_MODIFIER_MASK
        if self.phase == _Phase.CAPTURE:
            # Quick hack.
            # In this class, different from inktool, 
            # we can add nodes everytime we want.
            # so actually there is no 'CAPTURE' phase.
            self._start_new_capture_phase(rollback=False)
            self.phase = _Phase.ADJUST

        if self.phase in (_Phase.ADJUST, _Phase.ADJUST_POS): 
            # Important: phase can be set as _Phase.ADJUST_POS
            # at the button_press handler of base mixin.
                    
            if self.zone == _EditZone.EMPTY_CANVAS:
                
                if (len(self.nodes) > 0 and event.button == 1): 
                    
                    if shift_state and ctrl_state:
                        # In this case, call(place) buttons into this position.
                        self._queue_draw_buttons() 
                        self.forced_button_pos = (event.x, event.y)
                        self._bypass_phase(_Phase.ADJUST)

                    elif mod_state:
                        self.phase = _Phase.INSERT_NODE

                    ## Otherwise,fallthrough to supercall line. 
                    ## Then, a new node created and phase changed. 

            elif self.zone == _EditZone.CONTROL_NODE:
                assert len(self.nodes) > 0 
                if event.button == 1: 
                    if self.current_node_handle != None:
                        self.phase = _Phase.ADJUST_HANDLE
                        return # Do drag,but not supercall

            # FALLTHRU: *do* start a drag 
        
        if self.phase == _Phase.INSERT_NODE:
            mx, my = tdw.display_to_model(event.x, event.y)
            pressed_segment = detect_on_stroke(self.nodes, mx, my)
            if pressed_segment:
                # pressed_segment is a tuple which contains
                # (node index of start of segment, stroke step)

                # To erase buttons 
                self._queue_draw_buttons() 

                self._divide_bezier(*pressed_segment)

                self._bypass_phase(_Phase.ADJUST)
                self.app.show_transient_message(_("Create a new node on stroke"))
                return True # Cancel drag event
            else:
                self.app.show_transient_message(_("There is no stroke on clicked point.Creating node is failed."))
        
        # Supercall : inside inherited class, basic operation (such as
        # moving nodes) would be done.
        return super(BezierMode, self).mode_button_press_cb(tdw, event) 

    ## Drag handling (both capture and adjust phases)
    def node_drag_start_cb(self, tdw, event):
        mx, my = tdw.display_to_model(event.x, event.y)

        self._queue_draw_buttons() # To erase button,and avoid glitch

        # Basically,all sections should do fall-through.
        if self.phase == _Phase.ADJUST:

            if self.zone == _EditZone.EMPTY_CANVAS:
                # Workaround for numlock status.
                # When numlock is on, always GDK_MOD2_MASK is on!
                if event.state != 0 and event.state != Gdk.ModifierType.MOD2_MASK:
                    # To activate some mode override
                    self._last_event_node = None
                    return 
                else:
                    # New node added!
                    node = self._get_event_data(tdw, event)
                    self.nodes.append(node)
                    if not self._pressure_modified:
                        self.apply_pressure_from_curve_widget()
                    self._last_event_node = node
                    self.phase = _Phase.INIT_HANDLE
                    idx = len(self.nodes) - 1
                    self.select_node(idx, exclusive=True)
                    #elf._reset_selected_nodes(self.current_node_index)
                    # Important: with setting initial control handle 
                    # as the 'next' (= index 1) one,it brings us
                    # inkscape-like node creation.
                    self.current_node_handle = 1 

                    self._queue_draw_node(tdw, idx)

                    # Actually, this drag_offset.start() call is not to start,
                    # but to disable offset during handle manipulation.
                    # so, both of _Phase.ADJUST_HANDLE and INIT_HANDLE,
                    # we do not call drag_offset.end().
                    self.drag_offset.start(mx, my) 

        elif self.phase == _Phase.ADJUST_HANDLE:
            self._last_event_node = self.nodes[self.current_node_index]
            self.drag_offset.start(mx, my) # Same as new node added case.
        elif self.phase == _Phase.ADJUST_ITEM:
            self._queue_draw_selected_nodes(tdw)
        else:
            super(BezierMode, self).node_drag_start_cb(tdw, event)

    def node_drag_update_cb(self, tdw, event, dx, dy):
        mx, my = tdw.display_to_model(event.x, event.y)
        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK

        if self.phase == _Phase.ADJUST:
            pass
        elif self.phase in (_Phase.ADJUST_HANDLE, _Phase.INIT_HANDLE):
            node = self._last_event_node
            if self._last_event_node:
                self._queue_draw_node(tdw, self.current_node_index)# to erase
                node.set_control_handle(self.current_node_handle,
                        mx, my,
                        shift_state)
                self._queue_draw_node(tdw, self.current_node_index)# to update
            self._queue_redraw_item()
        else:
            super(BezierMode, self).node_drag_update_cb(tdw, event, dx, dy)

    def node_drag_stop_cb(self, tdw):
        if self.phase == _Phase.ADJUST:
            self._reset_adjust_data()
            if len(self.nodes) > 0:
                self._queue_redraw_item()
                self._queue_redraw_all_nodes()
                if len(self.nodes) > 1:
                    self._queue_draw_buttons()
            
        elif self.phase in (_Phase.ADJUST_HANDLE, _Phase.INIT_HANDLE):
            node = self._last_event_node
      
            # When initializing handle phase, 
            # even if the node is NOT 'curved' one,
            # the handles must be set as symmetry.
            # Otherwise, it might be rather confusing or unoperatable.
            if (self.phase == _Phase.INIT_HANDLE and 
                    len(self.nodes) > 1 and node.curve == False):
                # Setting 'curve' property of node twice.

                # The first one is to create initial symmetry handle.
                node.curve = True 

                # After that, set curve property to the desired
                # initial value = False.
                node.curve = False

            self._queue_draw_node(tdw, self.current_node_index) # to erase

            self._queue_redraw_item()
            if len(self.nodes) > 1:
                self._queue_draw_buttons()
            self.phase = _Phase.ADJUST
            self._queue_draw_node(tdw, self.current_node_index) 
        
        elif self.phase == _Phase.ADJUST_POS:
            self._queue_draw_selected_nodes(tdw) # to ensure erase them
            dx, dy = self.drag_offset.get_model_offset()
        
            for idx in self.selected_nodes:
                cn = self.nodes[idx]
                cn.move(cn.x + dx, cn.y + dy)
        
            self.drag_offset.reset()
            self._dragged_node_start_pos = None
            self._queue_redraw_item()
            self._queue_draw_buttons()
            self._queue_draw_selected_nodes(tdw) 
            self.phase = _Phase.ADJUST
        elif self.phase in (_Phase.ADJUST_PRESSURE, 
                            _Phase.ADJUST_PRESSURE_ONESHOT,
                            _Phase.ADJUST_ITEM): 
            self._pressure_modified = True
            self._queue_draw_buttons()
            super(BezierMode, self).node_drag_stop_cb(tdw)
        else:
            super(BezierMode, self).node_drag_stop_cb(tdw)

        # Common processing
        if self.current_node_index != None:
            self.options_presenter.target = (self, self.current_node_index)

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
            pressure=self.initial_pressure,
            xtilt=xtilt, ytilt=ytilt,
            dtime=self._DEFAULT_DTIME,
            )

    ## Node editing
    def set_node_pressure(self, idx, pressure):
        assert idx < len(self.nodes)
        self.nodes[idx].pressure=pressure

    def set_node_pos(self, idx, x, y):
        assert idx < len(self.nodes)
        cn = self.nodes[idx]
        cn.x = x
        cn.y = y

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
        self._queue_draw_node(None, i)
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
        self._queue_redraw_item()
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

        self._queue_redraw_item()

        # First of all, get the entire bezier stroke length.
        # That length is used to normalize stroke later.

        node_length=[]
        total_length = 0.0
        initial_pressure = self.initial_pressure
        for idx, cn in enumerate(self.nodes[:-1]):
            nn = self.nodes[idx + 1]
            ox = gui.drawutils.get_cubic_bezier(
                    cn.x, cn.get_control_handle(1).x,
                    nn.get_control_handle(0).x, nn.x, 0)
            oy = gui.drawutils.get_cubic_bezier(
                    cn.y, cn.get_control_handle(1).y,
                    nn.get_control_handle(0).y, nn.y, 0)

            cur_step = BezierMode.DRAFT_STEP
            length = 0.0 
            while cur_step < 1.0:
                cx = gui.drawutils.get_cubic_bezier(
                        cn.x, cn.get_control_handle(1).x,
                        nn.get_control_handle(0).x, nn.x, cur_step)
                cy = gui.drawutils.get_cubic_bezier(
                        cn.y, cn.get_control_handle(1).y,
                        nn.get_control_handle(0).y, nn.y, cur_step)
                length += vector_length(cx - ox, cy - oy)
                cur_step += BezierMode.DRAFT_STEP
                ox = cx
                oy = cy

            node_length.append(length)
            total_length+=length

        node_length.append(total_length) 

        # use curve class, to get smooth pressures.
        cur_length = 0.0
        pressure_map = self.pressure_map
        for idx,cn in enumerate(self.nodes):
            pressure = pressure_map.get_pressure(cur_length / total_length)
            pressure *= initial_pressure
            cn.pressure = pressure
            cur_length += node_length[idx]
        self._queue_redraw_item()

    def delete_selected_nodes(self):
        """ Beziertool can delete any nodes...
        even first / last one!
        """

        # First of all,queue redraw area.
        self._queue_draw_buttons()
        for idx in self.selected_nodes:
            self._queue_draw_node(None, idx)

        self._queue_redraw_item()

        # after then,delete it.
        new_nodes = []
        for idx,cn in enumerate(self.nodes):
            if idx in self.selected_nodes:
                if self.current_node_index == idx:
                    self.current_node_index = None
            else:
                new_nodes.append(cn)

        self.nodes = new_nodes
        self.select_node(-1)
        self._adjust_current_node_index()
        self.target_node_index = None

        # Issue redraws for the changed on-canvas elements
        if len(self.nodes) <= 1:
            if len(self.nodes) == 0:
                self.phase = _Phase.INITIAL
            self.redraw_item_cb(True)
        else:
            self._queue_redraw_item()

        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()

    def toggle_current_node_curve(self):
        """ mainly called from Callback of gui.document
        """
        if (self.phase in (_Phase.ADJUST, ) and
                len(self.nodes) > 0 and
                self.current_node_index != None):
            cn = self.nodes[self.current_node_index]
            cn.curve = not cn.curve
            self.options_presenter.set_checkbutton_curvepoint(
                    cn.curve)

    def update_node(self, i, **kwargs):
        """Updates properties of a node, and redraws it.
        
        :param i: node index
        :param kwargs: keyword arguments of nodes. incoming datas, such as
                       'x', 'y', or 'pressure' are stored as keyworded data.
        """
        assert i < len(self.nodes)
        node = self.nodes[i]
        if 'pressure' in kwargs:
            node.pressure = kwargs['pressure']
        if 'xtilt' in kwargs:
            node.pressure = kwargs['xtilt']
        if 'ytilt' in kwargs:
            node.pressure = kwargs['ytilt']
        if 'time' in kwargs:
            node.pressure = kwargs['time']

        self._queue_draw_node(None, i)
        self._queue_redraw_item()

    ## Action button handlers

    def accept_button_cb(self, tdw):
        if (len(self.nodes) > 1):
            self._queue_redraw_item(BezierMode.FINAL_STEP) # Redraw with hi-fidely curve
            self._start_new_capture_phase(rollback=False)

    def reject_button_cb(self, tdw):
        self._start_new_capture_phase(rollback=True)

    def edit_button_cb(self, tdw):
        if self.phase != _Phase.ADJUST_ITEM:
            self.phase = _Phase.ADJUST_ITEM
        else:
            self.phase = _Phase.ADJUST
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()

    # XXX for `info pick`
    ## Node pick
    def _apply_info(self, si, offset):
        """Apply nodes from compressed bytestring.
        """
        info = pickable.extract_info(si.get_info())
        nodes = self._unpack_info(info)

        # Note: This clears offset data in StrokeNode.
        assert offset is not None
        if offset != (0, 0):
            dx, dy = offset
            for n in nodes:
                n.move(dx, dy, relative=True)

        self.inject_nodes(nodes)
        self._erase_old_stroke(si)

        # For oncanvas editing tool, it automatically
        # registers offsetted nodes into strokemap.
        # So reset offset each time nodes picked.
        si.reset_offset() 

    def _match_info(self, infotype):
        return infotype == pickable.Infotype.BEZIER

    def _pack_info(self):
        nodes = self.nodes
        datas = struct.pack(">I", len(nodes))
        for n in nodes:
            datas += n.serialize()
        return (zlib.compress(datas), pickable.Infotype.BEZIER)

    def _unpack_info(self, info):
        raw_data = zlib.decompress(info)
        idx = 4
        count = struct.unpack('>I', raw_data[:idx])[0]
        nodes = []
        for i in range(count):
            data_length, node = _Node_Bezier.fromstring(raw_data, idx)
            nodes.append(node)
            idx += data_length
        return nodes
    # XXX for `info pick` end


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
            self._button_pos[_ActionButton.REJECT] = None
            self._button_pos[_ActionButton.ACCEPT] = None
            return False

        button_radius = gui.style.FLOATING_BUTTON_RADIUS
        alloc = self._tdw.get_allocation()
        view_x0, view_y0 = alloc.x, alloc.y
        view_x1, view_y1 = view_x0+alloc.width, view_y0+alloc.height
        
        def constrain_button(cx, cy, radius):
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

            ac_x, ac_y = pos_list[0][0], pos_list[0][1]
            rj_x, rj_y = pos_list[1][0], pos_list[1][1]
        else:
            # Usually, Bezier tool needs to keep extending control points
            # to drawing(constructing) a stroke. 
            # So when buttons placed around the tail(newest) node, 
            # it is something frastrating to place new node...
            # Thus,different from Inktool, place buttons around 
            # the first(oldest) node of a stroke.
            
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

            ac_x, ac_y = cx + dx, cy - dy
            rj_x, rj_y = cx - dx, cy + dy

        margin = 2.0 * button_radius
        ac_x, ac_y = constrain_button(ac_x, ac_y, margin)
        rj_x, rj_y = constrain_button(rj_x, rj_y, margin)
        self._button_pos[_ActionButton.ACCEPT] = (ac_x, ac_y)
        self._button_pos[_ActionButton.REJECT] = (rj_x, rj_y)

        # XXX Code duplication from exinktool.py
        # `Edit button` placed at the center of the line between accept_button 
        # to reject_button
        l, nx, ny = length_and_normal(ac_x, ac_y, rj_x, rj_y)
        ml = l / 2.0
        bx = nx * ml 
        by = ny * ml 

        mx = -ny * margin + bx + ac_x 
        my = nx * margin + by + ac_y
        mx, my = constrain_button(mx, my, margin)
        self._button_pos[_ActionButton.EDIT] = (mx, my)
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
                    fill=(hi==self._mode.current_node_handle)) 
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
        fill_flag = not mode.is_pressure_modifying()

        for i, node, x, y in self._get_onscreen_nodes():
            show_node = not mode.hide_nodes
            show_handle = False
            color = gui.style.EDITABLE_ITEM_COLOR

            if mode.phase in (_Phase.ADJUST_POS, 
                    _Phase.ADJUST, 
                    _Phase.ADJUST_PRESSURE,
                    _Phase.ADJUST_PRESSURE_ONESHOT,
                    _Phase.ADJUST_HANDLE, 
                    _Phase.INIT_HANDLE,
                    _Phase.ADJUST_ITEM):
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
                        if (mode.zone == _EditZone.CONTROL_NODE
                                and mode.current_node_handle != None):
                            show_node = True
                            color = gui.style.ACTIVE_ITEM_COLOR
                                  
                    # not 'elif' ... because target_node_index
                    # and current_node_index might be same.
                    if i == mode.target_node_index:
                        show_node = True
                        show_handle = True
                        color = gui.style.PRELIT_ITEM_COLOR

                # If current node is moving by drag...
                # At here, we use color of node to check 
                # whether the current node is target of editing or not. 
                # If that color is not `normal`, that node should be moved.
                if (color != gui.style.EDITABLE_ITEM_COLOR and
                        mode.phase in (_Phase.ADJUST_POS,
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
            self._draw_mode_buttons(cr)


class OptionsPresenter_Bezier (OptionsPresenter_ExInking):
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
            _Prefname.DEFAULT_DTIME, BezierMode._DEFAULT_DTIME))
        self._default_pressure_adj.set_value(self._app.preferences.get(
            _Prefname.INITIAL_PRESSURE, 1.0))

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

    def checkbutton_curvepoint_toggled_cb(self, button):
        if self._updating_ui:
            return
        beziermode, node_idx = self.target
        if beziermode:
            if 0 <= node_idx < len(beziermode.nodes):
                beziermode._queue_draw_node(None, node_idx) 
                beziermode._queue_redraw_item()
                beziermode.nodes[node_idx].curve = button.get_active()
                beziermode._queue_draw_node(None, node_idx) 
                beziermode._queue_redraw_item()

    def _variation_preset_combo_changed_cb(self, widget):
        if self._updating_ui:
            return
        super(OptionsPresenter_Bezier, self)._variation_preset_combo_changed_cb(widget)
        beziermode, node_idx = self.target
        if beziermode:
            beziermode.apply_pressure_from_curve_widget()

    def _default_dtime_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        self._app.preferences[_Prefname.DEFAULT_DTIME] = adj.get_value()
        BezierMode.set_default_dtime(adj.get_value())

    def _default_pressure_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        self._app.preferences[_Prefname.INITIAL_PRESSURE] = adj.get_value()
        BezierMode.set_initial_pressure(adj.get_value())

    ## Other handlers are as implemented in superclass.  
