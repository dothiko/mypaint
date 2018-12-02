# This file is part of MyPaint.
# Copyright (C) 2016 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


# EX-Inktool (EXperimental Inktool)
#
# This is for experimental version of Inkingtool, mostly based on
# inktool.py.
# This file is for, not only 'by remaining inktool.py, I can keep 
# using surely working/trusted inktool even adding any experimental 
# feature', but also  'to avoid many of conflicts and correcting them
# when git-pulling master'

## Imports
from __future__ import print_function

import math
import collections
import weakref
import os.path
from logging import getLogger
logger = getLogger(__name__)
import struct
import zlib

from gettext import gettext as _
import gi
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GLib
import numpy as np

import gui.mode
import gui.overlays
import gui.style
import gui.drawutils
import lib.helpers
import gui.cursor
import lib.observable
import gui.curve
import gui.widgets
from gui.linemode import *
import gui.ui_utils
from gui.oncanvas import *
from gui.inktool import _Node, _NODE_FIELDS
import gui.pickable as pickable

## Module constants

# Default Pressure variations.
# display-name, list of 4 control points.
# actually x-axis value of points[0] and points[3] are fixed.
_PRESSURE_VARIATIONS = [
        ('Default', [(0.0, 0.7), (0.3, 0.0), (0.7, 0.0), (1.0, 0.7)] ),
        ('Flurent', [(0.0, 0.9), (0.20, 0.4), (0.8, 0.4), (1.0, 0.9)] ),
        ('Thick'  , [(0.0, 0.4), (0.25, 0.2), (0.75, 0.2), (1.0, 0.4)] ),
        ('Thin'   , [(0.0, 0.9), (0.25, 0.7), (0.75, 0.7), (1.0, 0.9)] ),
        ('Head'   , [(0.0, 0.4), (0.25, 0.1), (0.75, 0.4), (1.0, 0.6)] ),
        ('Tail'   , [(0.0, 0.6), (0.25, 0.4), (0.75, 0.1), (1.0, 0.4)] ),
        ]


## Function defs

def _nodes_deletion_decorator(method):
    """ Decorator for deleting multiple nodes methods
    """
    def _decorator(self, *args):
        # To ensure redraw entire overlay,avoiding glitches.
        self._queue_redraw_item()
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()

        # the method should return deleted nodes count
        result = method(self, *args)
        assert type(result) == int

        if result > 0:
            self.options_presenter.target = (self, self.current_node_index)
            self._queue_redraw_item()
            self._queue_redraw_all_nodes()
            self._queue_draw_buttons()
        return result
    return _decorator


## Class defs

# _Phase is currently same as base PressPhase
_Phase = PressPhase

# _EditZone is currently same as base EditZoneMixin
_EditZone = EditZoneMixin

class _Prefs:
    """Constants for app.preferences and its default values.
    """
    AUTOAPPLY_PREF = 'exinktool.autoapply'
    THRESHOLD_PREF = 'exinktool.autoapply_threshold'

    DEFAULT_AUTOAPPLY = None
    DEFAULT_THRESHOLD = 5


class _ActionButton(ActionButtonMixin):
    """Enumeration for the action button of ExInkingMode
    """
    EDIT=2 # To toggle editing node-pressure phase.


# _Node object is imported from original inktool.

class ExInkingMode (PressureEditableMixin, 
                    NodeUserMixin):
    """Experimental Inking mode
    to test new feature.
    """

    ## Metadata properties

    ACTION_NAME = "ExInkingMode"
    pointer_behavior = gui.mode.Behavior.PAINT_FREEHAND
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW

    # Redefined buttons
    buttons = {
        _ActionButton.ACCEPT : ('mypaint-ok-symbolic', 
            'accept_button_cb'), 
        _ActionButton.REJECT : ('mypaint-trash-symbolic', 
            'reject_button_cb'),
        _ActionButton.EDIT : ('mypaint-edit-symbolic', 
            'edit_button_cb') 
    }

    ## Metadata methods

    @classmethod
    def get_name(cls):
        return _(u"ExpInk")

    def get_usage(self):
        return _(u"Draw, and then adjust smooth lines")


    @property
    def active_cursor(self):
        if self.phase == _Phase.ADJUST_POS:
            if self.zone == _EditZone.CONTROL_NODE:
                return self._crosshair_cursor
            elif self.zone != _EditZone.EMPTY_CANVAS: # assume button
                return self._arrow_cursor
        elif self.phase in (_Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT):
            if self.zone == _EditZone.CONTROL_NODE:
                return self._cursor_move_nw_se

        return None

    ## Class config vars
    
    # Captured input nodes are then interpolated with a spline.
    # The code tries to make nice smooth input for the brush engine,
    # but avoids generating too much work.
    INTERPOLATION_MAX_SLICE_TIME = 1/200.0   # seconds
    INTERPOLATION_MAX_SLICE_DISTANCE = 20   # model pixels
    INTERPOLATION_MAX_SLICES = \
            OncanvasEditMixin.MAX_INTERNODE_DISTANCE_MIDDLE * 5
    # In other words, limit to a set number of interpolation slices
    # per display pixel at the time of stroke capture.


    ## Other class vars

    _OPTIONS_PRESENTER = None   #: Options presenter singleton


    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):

        super(ExInkingMode, self).__init__(**kwargs)

        self._last_good_raw_pressure = 0.0
        self._last_good_raw_xtilt = 0.0
        self._last_good_raw_ytilt = 0.0

        # We need to copy the original __active_brushwork
       #self.__active_brushwork = self._BrushworkModeMixin__active_brushwork 
        self._sshot_before = None
        self._entered_cmd = None

        # For autoapply
        prefs = self.app.preferences
        self._autoapply = prefs.get(_Prefs.AUTOAPPLY_PREF, 
                                    _Prefs.DEFAULT_AUTOAPPLY)
        self._autoapply_threshold = prefs.get(_Prefs.THRESHOLD_PREF, 
                                              _Prefs.DEFAULT_THRESHOLD)

    def _reset_capture_data(self):
        super(ExInkingMode,self)._reset_capture_data()
        self._last_event_node = None  # node for the last event
        self._last_node_evdata = None  # (xdisp, ydisp, tmilli) for nodes[-1]

    def _reset_adjust_data(self):
        super(ExInkingMode, self)._reset_adjust_data()
        self._node_dragged = False

    def is_adjusting_phase(self):
        return self.phase in (_Phase.ADJUST,
                              _Phase.ADJUST_POS,
                              _Phase.ADJUST_PRESSURE,
                              _Phase.ADJUST_PRESSURE_ONESHOT,
                              _Phase.ADJUST_ITEM)

    def is_pressure_modifying(self):
        return self.phase in (_Phase.ADJUST_PRESSURE,
                              _Phase.ADJUST_PRESSURE_ONESHOT,
                              _Phase.ADJUST_ITEM)

    def _start_new_capture_phase(self, rollback=False):
        super(ExInkingMode, self)._start_new_capture_phase(rollback)
        self._reset_capture_data()

    def _generate_overlay(self, tdw):
        return Overlay(self, tdw)

    def _generate_presenter(self):
        return OptionsPresenter_ExInking()

    def update_cursor_cb(self, tdw):
        """ Called from _update_zone_and_target()
        to update cursors
        """
        cursor = None
        if self.is_adjusting_phase():
            if self.zone != _EditZone.EMPTY_CANVAS: # assume button
                cursor = self._arrow_cursor
            else:
                cursor = self._crosshair_cursor
        elif self.phase == _Phase.INSERT_NODE:
            cursor = self._insert_cursor

        return cursor

    def enter_insert_node_phase(self):
        if len(self.nodes) >= 2:
            if self.phase == _Phase.INSERT_NODE:
                self.phase = _Phase.ADJUST
                self.doc.app.show_transient_message(_("Toggled to adjust phase."))
            else:
                self.phase = _Phase.INSERT_NODE
                self.doc.app.show_transient_message(_("Entering insert node phase."))

            for tdw in self._overlays:
                self._update_cursor(tdw) 
        else:
            self.doc.app.show_transient_message(_("There is no stroke.Cannot enter insert phase."))

    def enter(self, doc, **kwds):
        """Enters the mode: called by `ModeStack.push()` etc."""
        super(ExInkingMode, self).enter(doc, **kwds)
        cursors = self.doc.app.cursors
        self._insert_cursor = cursors.get_action_cursor(
            self.ACTION_NAME,
            gui.cursor.Name.ADD,
        )

    ## Offset vector
    def _generate_offset_vector(self):
        """Returns offset vector(dragging vector, tuple of x, y)
        depend on current phase.
        """
        if self.phase == _Phase.ADJUST_POS:
            return self.drag_offset.get_model_offset()
        return None

    ## Redraws

    def _queue_draw_ink_node(self, tdw, i, base_node, offset_vec):
        """Redraws a specific control node on all known view TDWs
        :param node_ctx: node queuing context information.
        """

        pos = self._get_node_position(i, base_node, offset_vec)

        x, y = tdw.model_to_display(*pos)
        x = math.floor(x)
        y = math.floor(y)
        size = math.ceil(gui.style.DRAGGABLE_POINT_HANDLE_SIZE * 2)
        tdw.queue_draw_area(x-size, y-size, size*2+1, size*2+1)

    def _queue_draw_node(self, tdw, idx):
        """ For compatibility """
        if tdw is None:
            for tdw in self._overlays:
                self._queue_draw_node(tdw, idx)
            return 

        if self.current_node_index != None:
            basept = self.nodes[self.current_node_index]
        else:
            basept = None

        self._queue_draw_ink_node(tdw, idx, basept,
            self._generate_offset_vector())

    def _queue_draw_selected_nodes(self, tdw):
        """ Override mixin """
        if tdw is None:
            for tdw in self._overlays:
                self._queue_draw_selected_nodes(tdw)
            return

        if len(self._overlays) > 0:
            if self.current_node_index != None:
                basept = self.nodes[self.current_node_index]
            else:
                basept = None

            offset_vec = self._generate_offset_vector()

            for i in self.selected_nodes:
                self._queue_draw_ink_node(tdw, i, basept, offset_vec)

    def _queue_redraw_all_nodes(self):
        """ Override mixin :
        Redraws all nodes on all known view TDWs"""
        if len(self.nodes) > 0:
            if self.current_node_index != None:
                basept = self.nodes[self.current_node_index]
            else:
                basept = None
            offset_vec = self._generate_offset_vector()

            for tdw in self._overlays:
                for i in range(len(self.nodes)):
                    self._queue_draw_ink_node(tdw, i, basept, offset_vec)

    def _get_node_position(self, idx, base_node, offset_vec):
        """ Utility method, For unifying all node-editing codes. 
        """
        node = self.nodes[idx]
        if offset_vec is None or len(self.selected_nodes)==0:
            return (node.x, node.y)
        else:
            if idx in self.selected_nodes:
                ox , oy = offset_vec
                pos = (node.x + ox, node.y + oy)
            else:
                pos = (node.x, node.y)
        return pos

    def _queue_redraw_item(self):
        """Redraws the entire curve on all known view TDWs"""
        self._stop_task_queue_runner(complete=False)
        if self.current_node_index != None:
            base_node = self.nodes[self.current_node_index]
        else:
            base_node = None

        offset_vec = self._generate_offset_vector()

        for tdw in self._overlays:
            model = tdw.doc
            if len(self.nodes) < 2:
                continue
            self._queue_task(self.brushwork_rollback, model)
            self._queue_task(
                self.brushwork_begin, model,
                description=_("Inking"),
                abrupt=True,
            )
            interp_state = {"t_abs": self.nodes[0].time}
            
            for p_1, p0, p1, p2 in gui.drawutils.spline_iter_3(
                                    self.nodes,
                                    base_node,
                                    self.selected_nodes,
                                    offset_vec=offset_vec):
                self._queue_task(
                    self._draw_curve_segment,
                    model,
                    p_1, p0, p1, p2,
                    state=interp_state
                )
        self._start_task_queue_runner()

    def _draw_curve_segment(self, model, p_1, p0, p1, p2, state): 
        """Draw the curve segment between the middle two points"""
        last_t_abs = state["t_abs"]
        dtime_p0_p1_real = p1[-1] - p0[-1]
        steps_t = dtime_p0_p1_real / self.INTERPOLATION_MAX_SLICE_TIME
        dist_p1_p2 = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
        steps_d = dist_p1_p2 / self.INTERPOLATION_MAX_SLICE_DISTANCE
        steps_max = float(self.INTERPOLATION_MAX_SLICES)
        steps = math.ceil(min(steps_max, max([2, steps_t, steps_d])))
        for i in range(int(steps) + 1):
            t = i / steps
            point = gui.drawutils.spline_4p(t, p_1, p0, p1, p2)
            x, y, pressure, xtilt, ytilt, t_abs, viewzoom, viewrotation = point
            pressure = lib.helpers.clamp(pressure, 0.0, 1.0)
            xtilt = lib.helpers.clamp(xtilt, -1.0, 1.0)
            ytilt = lib.helpers.clamp(ytilt, -1.0, 1.0)
            t_abs = max(last_t_abs, t_abs)
            dtime = t_abs - last_t_abs
            viewzoom = self.doc.tdw.scale
            viewrotation = self.doc.tdw.rotation   
            self.stroke_to(
                model, dtime, 
                x, y, 
                pressure, 
                xtilt, ytilt,
                viewzoom, viewrotation,
                auto_split=False,
            )

            last_t_abs = t_abs
        state["t_abs"] = last_t_abs

    ## Raw event handling (prelight & zone selection in adjust phase)
    def mode_button_press_cb(self, tdw, event):
        mod_state = event.state & self.ONCANVAS_MODIFIER_MASK
        detected_stroke = None

        if self.phase in (_Phase.ADJUST, _Phase.ADJUST_ITEM):
            if event.button == 1:
                if self.zone == EditZoneMixin.EMPTY_CANVAS:
                    if not mod_state:
                        self._start_new_capture_phase(rollback=False)
                        assert self.phase == PhaseMixin.CAPTURE
                    else:
                        mx, my = tdw.display_to_model(event.x, event.y)
                        detected_stroke = self._detect_on_stroke(mx, my)
                        if detected_stroke:
                            self.phase = _Phase.INSERT_NODE

                    # Fallthrough

        # Not `elif`. In above codes, `detected_stroke` might become True.
        if self.phase == _Phase.INSERT_NODE or detected_stroke:
            if detected_stroke is None:
                mx, my = tdw.display_to_model(event.x, event.y)
                detected_info = self._detect_on_stroke(mx, my)

            if detected_stroke:
                # pressed_segment is a tuple which contains
                # (node index of start of segment, stroke step)

                # To erase buttons 
                self._queue_draw_buttons() 

                idx, t, x, y = detected_stroke
                new_pressure = self.nodes[idx].pressure
                new_xtilt = self.nodes[idx].xtilt
                new_ytilt = self.nodes[idx].ytilt
                new_time = self.nodes[idx].time

                if idx < len(self.nodes) - 1:
                    new_pressure += self.nodes[idx+1].pressure
                    new_xtilt += self.nodes[idx+1].xtilt
                    new_ytilt += self.nodes[idx+1].ytilt
                    new_time += self.nodes[idx+1].time

                    new_pressure *= 0.5
                    new_xtilt *= 0.5
                    new_ytilt *= 0.5
                    new_time *= 0.5


                self.nodes.insert(
                    idx + 1, # insert method requires the inserted position. 
                    _Node(
                        x=x, y=y,
                        pressure=new_pressure,
                        xtilt=new_xtilt, ytilt=new_ytilt,
                        time=new_time,
                        viewzoom = self.doc.tdw.scale,
                        viewrotation = self.doc.tdw.rotation,
                    )
                )

                # queue new node here.
                
                self._bypass_phase(_Phase.ADJUST)
                self.doc.app.show_transient_message(_("Create a new node on stroke"))
                return True # Cancel drag event
            else:
                self.doc.app.show_transient_message(_("There is no stroke on clicked point.Creating node is failed."))
                
        super(ExInkingMode, self).mode_button_press_cb(tdw, event)
            
        # Update workaround state for evdev dropouts
        self._button_down = event.button
        self._last_good_raw_pressure = 0.0
        self._last_good_raw_xtilt = 0.0
        self._last_good_raw_ytilt = 0.0

    def mode_button_release_cb(self, tdw, event):
        super(ExInkingMode, self).mode_button_release_cb(tdw, event)

        # Update workaround state for evdev dropouts
        self._button_down = None
        self._last_good_raw_pressure = 0.0
        self._last_good_raw_xtilt = 0.0
        self._last_good_raw_ytilt = 0.0

    ## Drag handling (both capture and adjust phases)

    ## Capture handling

    def node_drag_start_cb(self, tdw, event):
        self._node_dragged = False
        if self.phase == _Phase.CAPTURE:
            assert len(self.nodes) == 0
            # Workaround for numlock status.
            # When numlock is on, always GDK_MOD2_MASK is on!
            if event.state != 0 and event.state != Gdk.ModifierType.MOD2_MASK:
                # To activate some mode override
                self._last_event_node = None
                return super(ExInkingMode, self).node_drag_start_cb(tdw, event)
            else:
                node = self._get_event_data(tdw, event)
                self.nodes.append(node)
                self._queue_draw_node(tdw, 0)
                self._last_node_evdata = (event.x, event.y, event.time)
                self._last_event_node = node
        elif self.phase == _Phase.ADJUST_ITEM:
            self._queue_draw_selected_nodes(tdw)
        else:
            super(ExInkingMode, self).node_drag_start_cb(tdw, event)

    def node_drag_update_cb(self, tdw, event, dx, dy):
        """ User dragging within capturing phase.
 
        Some deriving class like Inktool generates
        additional new nodes inside this callback.
        """
        if self.phase == _Phase.CAPTURE:
            if self._last_event_node:
                node = self._get_event_data(tdw, event)
                evdata = (event.x, event.y, event.time)
                if not self._last_node_evdata: # e.g. after an undo while dragging
                    append_node = True
                elif evdata == self._last_node_evdata:
                    logger.debug(
                        "Capture: ignored successive events "
                        "with identical position and time: %r",
                        evdata,
                    )
                    append_node = False
                else:
                    dx = event.x - self._last_node_evdata[0]
                    dy = event.y - self._last_node_evdata[1]
                    dist = math.hypot(dy, dx)
                    dt = event.time - self._last_node_evdata[2]
                    max_dist = self.MAX_INTERNODE_DISTANCE_MIDDLE
                    if len(self.nodes) < 2:
                        max_dist = self.MAX_INTERNODE_DISTANCE_ENDS
                    append_node = (
                        dist > max_dist and
                        dt > self.MAX_INTERNODE_TIME
                    )
                if append_node:
                    self.nodes.append(node)
                    self._queue_draw_node(tdw, len(self.nodes)-1)
                    self._queue_redraw_item()
                    self._last_node_evdata = evdata
                self._last_event_node = node
                return True

        elif self.phase == _Phase.ADJUST_POS:
            if self._dragged_node_start_pos:
                self._node_dragged = True
                doff = self.drag_offset

                # To erase old-positioned nodes.
                self._queue_draw_selected_nodes(tdw)
                x, y = tdw.display_to_model(event.x, event.y)
                doff.end(x, y)
                self._queue_draw_selected_nodes(tdw)
                self._queue_redraw_item()
        else:
            super(ExInkingMode, self).node_drag_update_cb(tdw, event, dx, dy)
 
    def node_drag_stop_cb(self, tdw):
        """ User ends capturing the selected node(s).
        """
        if self.phase == _Phase.CAPTURE:
            if not self.nodes or self._last_event_node == None:
                return 

            node = self._last_event_node
            if self.nodes[-1] is not node:
                # When too close against last captured node,
                # delete it.
                d = math.hypot(self.nodes[-1].x - node.x,
                        self.nodes[-1].y - node.y)
                mid_d = gui.ui_utils.display_to_model_distance(tdw, 
                        self.MAX_INTERNODE_DISTANCE_MIDDLE)
                # For now, I define nodes are 'too close' 
                # when their distance is less than MAX_INTERNODE_DISTANCE_MIDDLE / 5
                if d < mid_d / 5.0:
                    self._queue_draw_node(tdw, len(self.nodes)-1) # To avoid glitch
                    del self.nodes[-1]
            
                self.nodes.append(node)


            if (self._autoapply is not None 
                    and len(self.nodes) > self._autoapply_threshold):
                if self._autoapply == 'cull':
                    self._queue_redraw_all_nodes() # To erase
                    self._cull_nodes()
                elif self._autoapply == 'simple':
                    self._queue_redraw_all_nodes() # To erase
                    self._simplify_nodes()

            if len(self.nodes) > 1:
                self.phase = _Phase.ADJUST
                self._queue_redraw_all_nodes()
                self._queue_redraw_item()
                self._queue_draw_buttons()
            else:
                # Should enter capture phase again
                self.nodes = []
                self._reset_capture_data()
                tdw.queue_draw()

        elif self.phase == _Phase.ADJUST_POS:
            # Finalize dragging motion to selected nodes.
            if self._node_dragged:
 
                self._queue_draw_selected_nodes(tdw) # to ensure erase them
 
                for i, cn, x, y in self.nodes_position_iter(tdw, convert_to_display=False):
                    if cn.x != x or cn.y != y:
                        self.nodes[i] = cn._replace(x=x, y=y)
 
                self.drag_offset.reset()
 
            self._dragged_node_start_pos = None

            self.phase = _Phase.ADJUST

        elif self.phase == _Phase.ADJUST_ITEM:
            # Finalize dragging motion as pressure changing.
            if self._node_dragged:
                self.drag_offset.reset()
            self._dragged_node_start_pos = None
            self._queue_draw_buttons()
        else:
            super(ExInkingMode, self).node_drag_stop_cb(tdw)

    ## Node editing

    def set_node_pressure(self, idx, pressure):
        cn = self.nodes[idx]
        self.nodes[idx] = cn._replace(pressure=pressure)

    def set_node_pos(self, idx, x, y):
        cn = self.nodes[idx]
        self.nodes[idx] = cn._replace(x=x, y=y)

    def _detect_on_stroke(self, x, y, allow_distance = 4.0):
        """Detecting pressed point is on the stroke currently editing.
        
        :param x: cursor x position in MODEL coord
        :param y: cursor y position in MODEL coord
        :param allow_distance: the allowed distance from stroke.
        :return : a tuple of (the index of 'previous' node, 
                time parameter of stroke,
                x of new_node, y of new_node)
        :rtype : a tuple when the pressed point is on stroke, otherwise
                 None.
        
        """

        # XXX Transplant from https://gist.github.com/MadRabbit/996893
        def find_x_for(p_1, p0, p1, p2, tx, init):
            x=init 
            i=0
            while i < 5: # making 5 iterations max
                z = gui.drawutils.spline_4p(x, p_1, p0, p1, p2) - tx

                if abs(z) < 0.0000001:
                    break # if already got close enough

                dx = gui.drawutils.get_diff_spline_4p(x, p_1, p0, p1, p2)
                if dx == 0.0:
                    break 

                x = x - z / dx
                i+=1

            return x # try any x

        for idx in range(len(self.nodes)-1):
            pt0 = self.nodes[idx]
            pt1 = self.nodes[idx+1]
            if gui.drawutils.is_inside_segment(pt0, pt1, x, y):

                if idx == 0:
                    double_first=True
                    line_list = [ pt0, pt1 ] 
                else:
                    double_first = False
                    line_list = [ self.nodes[idx-1], pt0, pt1 ] 

                if idx >= len(self.nodes) - 2:
                    double_last = True
                else:
                    double_last = False
                    line_list.append(self.nodes[idx+2])

                for p_1, p0, p1, p2 in gui.drawutils.spline_iter(line_list,
                        double_first=double_first, double_last=double_last):
                    c=0
                    t=1.0
                    while c < 2:
                        t = find_x_for(p_1[0], p0[0], p1[0], p2[0], x, t)
                        cpos = gui.drawutils.spline_4p(t, p_1, p0, p1, p2)
                        if abs(y - cpos[1]) < allow_distance:
                            # the timepoint Found!
                            return (idx, t, cpos[0], cpos[1])
                        t = 0.0
                        c+=1

    def _get_event_data(self, tdw, event):
        x, y = tdw.display_to_model(event.x, event.y)
        xtilt, ytilt = self._get_event_tilt(tdw, event)
        return _Node(
            x=x, y=y,
            pressure=self._get_event_pressure(event),
            xtilt=xtilt, ytilt=ytilt,
            time=(event.time / 1000.0),
            viewzoom = self.doc.tdw.scale,
            viewrotation = self.doc.tdw.rotation,
        )

    def nodes_position_iter(self, tdw, convert_to_display=True):
        """ Enumerate nodes and its screen coordinate 
        with considering pointer dragging offsets and range.
        """
        mx, my = self.drag_offset.get_model_offset()
        for i, node in enumerate(self.nodes):
            if i in self.selected_nodes:
                x, y = node.x + mx, node.y + my
            else:
                x, y = node.x, node.y

            if convert_to_display:
                x, y = tdw.model_to_display(x, y)

            yield (i, node, x, y)

    def update_node(self, i, **kwargs):
        """Updates properties of a node, and redraws it"""
        changing_pos = bool({"x", "y"}.intersection(kwargs))
        oldnode = self.nodes[i]
        if changing_pos:
            self._queue_draw_node(None, i)
        self.nodes[i] = oldnode._replace(**kwargs)
        # FIXME: The curve redraw is a bit flickery.
        #   Perhaps dragging to adjust should only draw an
        #   armature during the drag, leaving the redraw to
        #   the stop handler.
        self._queue_redraw_item()
        if changing_pos:
            self._queue_draw_node(None, i)

    def can_delete_node(self, i):
        """ Override mixin method.
        """
        return 0 < i < len(self.nodes)-1

    def _adjust_current_node_index(self):
        """ Adjust self.current_node_index.
        This behavior is Inktool unique.
        """
        if self.current_node_index >= len(self.nodes):
            self.current_node_index = len(self.nodes) - 2
            if self.current_node_index < 0:
                self.current_node_index = None
            self.current_node_changed(
                    self.current_node_index)

    def delete_node(self, i):
        """Delete a node, and issue redraws & updates"""
        assert self.can_delete_node(i), "Can't delete endpoints"
        # Redraw old locations of things while the node still exists
        self._queue_draw_buttons()
        self._queue_draw_node(None, i)

        self._pop_node(i)
        self.options_presenter.target = (self, self.current_node_index)

        # Issue redraws for the changed on-canvas elements
        self._queue_redraw_item()
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()

    def delete_selected_nodes(self):

        self._queue_draw_buttons()
        for idx in self.selected_nodes:
            self._queue_draw_node(None, idx)

        new_nodes = [self.nodes[0]]
        for idx,cn in enumerate(self.nodes[1:-1]):
            t_idx = idx + 1
            if t_idx in self.selected_nodes:
                if self.current_node_index == t_idx:
                    self.current_node_index = None
            else:
                new_nodes.append(cn)

        new_nodes.append(self.nodes[-1])
        self.nodes = new_nodes
        self.select_node(-1)
        self.target_node_index = None

        # Issue redraws for the changed on-canvas elements
        self._queue_redraw_item()
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()


    def insert_node(self, i):
        """Insert a node, and issue redraws & updates"""
        assert self.can_insert_node(i), "Can't insert back of the endpoint"
        # Redraw old locations of things while the node still exists
        self._queue_draw_buttons()
        self._queue_draw_node(None, i)
        # Create the new node
        cn = self.nodes[i]
        nn = self.nodes[i+1]

        newnode = _Node(
            x=(cn.x + nn.x)/2.0, y=(cn.y + nn.y) / 2.0,
            pressure=(cn.pressure + nn.pressure) / 2.0,
            xtilt=(cn.xtilt + nn.xtilt) / 2.0,
            ytilt=(cn.ytilt + nn.ytilt) / 2.0,
            time=(cn.time + nn.time) / 2.0,
            viewzoom=(cn.viewzoom + nn.viewzoom) / 2.0,
            viewrotation=(cn.viewrotation + nn.viewrotation) / 2.0,
        )
        self.nodes.insert(i+1,newnode)

        # Issue redraws for the changed on-canvas elements
        self._queue_redraw_item()
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()

    def insert_current_node(self):
        if self.can_insert_node(self.current_node_index):
            self.insert_node(self.current_node_index)

    def _pop_node(self, idx):
        """ wrapper method of popping(delete) node.
        to ensure not included in self.selected_nodes.
        """
        if idx in self.selected_nodes:
            self.selected_nodes.remove(idx)

        for i, sidx  in enumerate(self.selected_nodes):
            if sidx > idx:
                self.selected_nodes[i] = sidx - 1

        def adjust_index(cur_idx, targ_idx):
            if cur_idx == targ_idx:
                cur_idx = -1
            elif cur_idx > targ_idx:
                cur_idx -= 1

            if cur_idx < 0:
                return None
            return cur_idx


        self.current_node_index = adjust_index(self.current_node_index,idx)
        self.target_node_index = adjust_index(self.target_node_index,idx)

        return self.nodes.pop(idx)

    def _simplify_nodes(self, tolerance):
        """Internal method of simplify nodes.

        """

        # Algorithm: Reumann-Witkam.
        i=0
        oldcnt=len(self.nodes)
        while i<len(self.nodes)-2:
            try:
                vsx=self.nodes[i+1].x-self.nodes[i].x
                vsy=self.nodes[i+1].y-self.nodes[i].y
                ss=math.sqrt(vsx*vsx + vsy*vsy)
                nsx=vsx/ss
                nsy=vsy/ss
                while i+2<len(self.nodes):
                    vex=self.nodes[i+2].x-self.nodes[i].x
                    vey=self.nodes[i+2].y-self.nodes[i].y
                    es=math.sqrt(vex*vex + vey*vey)
                    px=nsx*es
                    py=nsy*es
                    dp=(px*(vex/es)+py*(vey/es)) / es
                    hx=(vex*dp)-px
                    hy=(vey*dp)-py

                    if math.sqrt(hx*hx + hy*hy) < tolerance:
                        self._pop_node(i+1)
                    else:
                        break

            except ValueError:
                pass
            except ZeroDivisionError:
                pass
            finally:
                i+=1

        return oldcnt-len(self.nodes)

    def _cull_nodes(self):
        """Internal method of cull nodes."""
        curcnt=len(self.nodes)
        idx = 1
        while idx < len(self.nodes)-1:
            self._pop_node(idx)
            idx+=1
        return curcnt-len(self.nodes)

    def _queue_all_visual_redraw(self):
        """Redraw all overlay objects"""
        self._queue_redraw_item()
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()

    @_nodes_deletion_decorator
    def simplify_nodes(self):
        """User interface method of simplify nodes."""
        # XXX For now, parameter is fixed value.
        # tolerance is 8, in model coords.
        # this value should be configureable...?
        return self._simplify_nodes(8)

    @_nodes_deletion_decorator
    def cull_nodes(self):
        """User interface method of cull nodes."""
        return self._cull_nodes()


    ## nodes average
    def average_nodes_angle(self):
        """Average nodes angle.
        Treat stroke as a sequence of vector,and
        average all nodes angle,except for first and last.

        The meaning of 'angle' referred here is,
        for example,there is 3 nodes in order of A,B and C,

            __ -- B ---___
        A --              --C

        when 'average angle of B',it means
        'To half the angle of between A-B and A-C'

        This method affects to selected nodes,
        but when the only one node is selected,
        entire nodes (except for the first and last) affected.
        """

        if len(self.nodes) > 2:

            # Redraw to erase old nodes
            self._queue_all_visual_redraw()

            idx = 1
            nodes = self.nodes

            def _do_average_nodes(idx):
                while idx < len(self.nodes) - 1:
                    pn = nodes[idx-1]
                    cn = nodes[idx]
                    nn = nodes[idx+1]
                    # Limit affected nodes with selection list.
                    # if only one node is selected,
                    # entire nodes are averaged.
                    if (len(self.selected_nodes) == 0 or
                            idx in self.selected_nodes):
                        try:
                            s = math.hypot(cn.x - pn.x, cn.y - pn.y)
                            # avx, avy is identity vector of current-prev node
                            # bvx, bvy is identity vector of next-prev node
                            avx, avy = normal(pn.x, pn.y, cn.x, cn.y)
                            bvx, bvy = normal(pn.x, pn.y, nn.x, nn.y)
                            avx=(avx + bvx) / 2.0
                            avy=(avy + bvy) / 2.0
                            avx*=s
                            avy*=s
                            nodes[idx]=(cn._replace(x=avx+pn.x, y=avy+pn.y))
                        except ZeroDivisionError:
                            # This means 'two nodes at same place'.
                            # abort averaging for this node.
                            pass
                    else:
                        pass

                    idx += 2

            # first time, it targets the odd index nodes.
            _do_average_nodes(1)

            # then next time, it targets the even index nodes.
            _do_average_nodes(2)

            # redraw new nodes
            self._queue_all_visual_redraw()

    def average_nodes_distance(self):
        """Average nodes distance.
        Treat stroke as a sequence of vector,and
        average(to half) all nodes distance,
        except for first and last.

        this method affects entire self.nodes,
        regardless of how many nodes selected.
        """

        if len(self.nodes) > 2:

            # Redraw to erase old nodes
            self._queue_all_visual_redraw()

            # get entire vector length
            entire_length = 0
            for idx,cn in enumerate(self.nodes[:-1]):
                nn = self.nodes[idx+1]
                entire_length += math.hypot(cn.x - nn.x, cn.y - nn.y)

            segment_length = entire_length / (len(self.nodes) - 1)
            new_nodes = [self.nodes[0],]

            # creating entire new nodes list.
            cur_segment = segment_length
            sidx = 1 # source node idx,it is not equal to idx.
            for idx,cn in enumerate(self.nodes[:-1]):
                nn = self.nodes[idx+1]
                cur_length = math.hypot(cn.x - nn.x, cn.y - nn.y)

                if cur_segment == cur_length:
                    # it is rare,next node completely fit
                    # to segment.
                    new_nodes.append(self.nodes[sidx])
                    sidx += 1
                    cur_segment = segment_length
                elif cur_segment < cur_length:
                    # segment end.need for adding a node.
                    try:
                        avx, avy = normal(cn.x, cn.y, nn.x, nn.y)
                        avx *= cur_segment
                        avy *= cur_segment
                        new_nodes.append(self.nodes[sidx]._replace(
                            x=avx+cn.x, y=avy+cn.y))
                        cur_segment = segment_length - (cur_length - cur_segment)
                        sidx += 1
                    except ZeroDivisionError:
                        # this means 'current length is 0'.
                        # so ignore.
                        pass
                else:
                    # segment continues
                    cur_segment -= cur_length

            assert sidx == len(self.nodes) - 1

            new_nodes.append(self.nodes[-1])
            self.nodes = new_nodes

            # redraw new nodes
            self._queue_all_visual_redraw()

    def average_nodes_pressure(self):
        """Average nodes pressure.

        This method affects to selected nodes,
        but when the only one node is selected,
        entire nodes (except for the first and last) affected.
        """

        if len(self.nodes) > 2:

            new_nodes = []

            for idx,cn in enumerate(self.nodes):
                if (idx > 0 and idx < len(self.nodes) - 1 and
                        (len(self.selected_nodes) == 0 or
                            idx in self.selected_nodes) ):
                    pn = self.nodes[idx-1]
                    nn = self.nodes[idx+1]

                    # not simple average,weighted one
                    new_pressure = (pn.pressure * 0.25 +
                                    cn.pressure * 0.5 +
                                    nn.pressure * 0.25)

                    cn = cn._replace(pressure = new_pressure)

                new_nodes.append(cn)

            self.nodes = new_nodes
            self._queue_redraw_item()


    ## Node selection
    def select_all(self):
        self.selected_nodes = range(0, len(self.nodes))
        self._queue_redraw_all_nodes()

    def deselect_all(self):
        """ Utility method.
        Actually, same as self.select_node(-1)
        """
        self.select_node(-1)
        self._queue_redraw_all_nodes()

    def select_area_cb(self, selection_mode):
        """ Selection handler called from SelectionMode.
        This handler never called when no selection executed.
        """
        modified = False
        for idx,cn in enumerate(self.nodes):
            if selection_mode.is_inside_model(cn.x, cn.y):
                if not idx in self.selected_nodes:
                    self.selected_nodes.append(idx)
                    modified = True
        if modified:
            self._queue_redraw_all_nodes()
       

    ## Action button related
    def accept_button_cb(self, tdw):
        if len(self.nodes) > 1:
            self._start_new_capture_phase(rollback=False)
            assert self.phase == _Phase.CAPTURE

    def reject_button_cb(self, tdw):
        self._start_new_capture_phase(rollback=True)
        assert self.phase == _Phase.CAPTURE

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
            for i,n in enumerate(nodes):
                nodes[i] = n._replace(x=n.x+dx, y=n.y+dy)

        self.inject_nodes(nodes)
        self._erase_old_stroke(si)

        # For oncanvas editing tool, it automatically
        # registers offsetted nodes into strokemap.
        # So reset offset each time nodes picked.
        si.reset_offset() 

    def _match_info(self, infotype):
        return infotype == pickable.Infotype.TUPLE

    def _pack_info(self):
        nodes = self.nodes
        datas = struct.pack(">I", len(nodes))
        fmt=">%dd" % len(nodes[0]) 
        for n in nodes:
            datas += struct.pack(fmt, *n)
        return (zlib.compress(datas), pickable.Infotype.TUPLE)

    def _unpack_info(self, nodesinfo):
        raw_data = zlib.decompress(nodesinfo)
        idx = 4
        count = struct.unpack('>I', raw_data[:idx])[0]
        nodes = []
        field_cnt = len(_NODE_FIELDS)
        fmt = ">%dd" % field_cnt
        data_length = field_cnt * 8
        for i in range(count):
            a = struct.unpack(fmt, raw_data[idx: idx+data_length])
            node = _Node(*a)
            nodes.append(node)
            idx += data_length
        return nodes
    # XXX for `info pick` end

    ## Autoapply 
    def set_autoapply(self, method):
        """Set autoapply method.

        That method is executed when right after the new nodes stroke
        is created.

        :param method: None or 'cull' or 'simple'
        """
        self._autoapply = method

    def set_autoapply_threshold(self, count):
        """Set autoapply threshould node count.

        If generated nodes count is lower than this count,
        `autoapply` is not executed.
        """
        self._autoapply_threshold = count


class Overlay (OverlayOncanvasMixin):
    """Overlay for an ExInkingMode's adjustable points"""

    def update_button_positions(self):
        """Recalculates the positions of the mode's buttons."""
        nodes = self._mode.nodes
        num_nodes = len(nodes)
        if num_nodes == 0:
            self._button_pos[_ActionButton.REJECT] = None
            self._button_pos[_ActionButton.ACCEPT] = None
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

        # The reject and accept buttons are connected to different nodes
        # in the stroke by virtual springs.
        stroke_end_i = len(fixed)-1
        stroke_start_i = 0
        stroke_last_quarter_i = int(stroke_end_i * 3.0 // 4.0)
        assert stroke_last_quarter_i < stroke_end_i
        reject_anchor_i = stroke_start_i
        accept_anchor_i = stroke_end_i
        mode_anchor_i = ((stroke_end_i - stroke_start_i) / 2) + stroke_start_i

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
        edit_button = _LayoutNode(
            fixed[mode_anchor_i].x - stroke_tail[0]*margin,
            fixed[mode_anchor_i].y - stroke_tail[1]*margin,
        )

        # Constraint boxes. They mustn't share corners.
        # Natural hand strokes are often downwards,
        # so let the reject button to go above the accept button.
        t_margin = 2.666 * margin
        reject_button_bbox = (
            view_x0+margin, view_x1-margin,
            view_y0+margin, view_y1-t_margin,
        )
        accept_button_bbox = (
            view_x0+margin, view_x1-margin,
            view_y0+t_margin, view_y1-margin,
        )

        edit_button_bbox = (
            view_x0+t_margin, view_x1-t_margin,
            view_y0+t_margin, view_y1-t_margin,
        )

        # Force-update constants
        k_repel = -25.0
        k_attract = 0.05

        # Let the buttons bounce around until they've settled.
        for iter_i in range(100):
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

        self._button_pos[_ActionButton.ACCEPT] = accept_button.x, accept_button.y
        self._button_pos[_ActionButton.REJECT] = reject_button.x, reject_button.y

        # `Edit button` placed at the center of the line between accept_button 
        # to reject_button
        l, nx, ny = length_and_normal(accept_button.x, accept_button.y,
                                      reject_button.x, reject_button.y)

        # Get center vector from accept_button
        ml = l / 2.0
        bx = nx * ml 
        by = ny * ml 

        ml = 0 # Recycle variable `ml`
        ax = accept_button.x
        ay = accept_button.y
        mx = bx + ax
        my = by + ay
        tdw = self._tdw

        # Check edit_button distance against every nodes.
        # CAUTION: nodes are in model, but buttons are in display.
        for iter_i in range(100):
            modified = False
            for cn in nodes:
                cx, cy = tdw.model_to_display(cn.x, cn.y)
                dist = math.hypot(cx-mx, 
                                  cy-my)
                if dist < t_margin:
                    ml += t_margin
                    # Get a bit far, right-angled vector
                    # From accept-button to reject-button
                    mx = -ny * ml + bx + ax
                    my = nx * ml + by + ay
                    modified = True
                    break
            if not modified:
                break

        edit_button.x = mx
        edit_button.y = my
        edit_button.constrain_position(*edit_button_bbox)
        self._button_pos[_ActionButton.EDIT] = edit_button.x, edit_button.y

    def _get_onscreen_nodes(self):
        """ Override mixin method.
        Iterates across only the on-screen nodes.
        with range-based adjustment.
        """
        mode = self._mode
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        alloc = self._tdw.get_allocation()
        for i, node, x, y in mode.nodes_position_iter(self._tdw):
            node_on_screen = (
                x > alloc.x - radius*2 and
                y > alloc.y - radius*2 and
                x < alloc.x + alloc.width + radius*2 and
                y < alloc.y + alloc.height + radius*2
            )
            if node_on_screen:
                yield (i, node, x, y)

    def paint(self, cr):
        """Draw adjustable nodes to the screen"""
        # Control nodes
        mode = self._mode
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        alloc = self._tdw.get_allocation()
        dx,dy = mode.drag_offset.get_display_offset(self._tdw)
        fill_flag = not mode.is_pressure_modifying()
        for i, node, x, y in self._get_onscreen_nodes():
            color = gui.style.EDITABLE_ITEM_COLOR
            show_node = not mode.hide_nodes
            if mode.phase in (_Phase.ADJUST, 
                    _Phase.ADJUST_POS,
                    _Phase.ADJUST_PRESSURE,
                    _Phase.ADJUST_PRESSURE_ONESHOT,
                    _Phase.CHANGE_PHASE,
                    _Phase.ADJUST_ITEM):
                if show_node:
                    if i == mode.current_node_index:
                        color = gui.style.ACTIVE_ITEM_COLOR
                    elif i == mode.target_node_index:
                        color = gui.style.PRELIT_ITEM_COLOR
                    elif i in mode.selected_nodes:
                        color = gui.style.POSTLIT_ITEM_COLOR
                else:
                    if i == mode.target_node_index:
                        show_node = True
                        color = gui.style.PRELIT_ITEM_COLOR

            if show_node:
                gui.drawutils.render_round_floating_color_chip(
                    cr=cr, x=x, y=y,
                    color=color,
                    radius=radius,
                    fill=fill_flag)

        if mode.is_adjusting_phase():
            
            # Drawing Buttons when not in drag.
            if not mode.in_drag:
                self._draw_mode_buttons(cr)
                


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


class StrokePressureSettings (object):
    """Manage GtkAdjustments for tweaking Inktool StrokeCurveWidget settings.

    An instance resides in the main application singleton. Changes to the
    adjustments are reflected into the app preferences.

    this class is originated from LineModeSettings of gui/linemode.py
    """

    ## Class Constants
    _PREF_KEY_BASE = "inkmode.pressure"

    def __init__(self, app):
        """Initializer; initial settings are loaded from the app prefs"""
        object.__init__(self)
        self.app = app
        self.observers = []  #: List of callbacks
        self._idle_srcid = None
        self._changed_settings = set()
        self._settings = {}
        custom_pressures = self.app.preferences.get(
                StrokePressureSettings.get_pref_key('settings'),
                _PRESSURE_VARIATIONS)

        for cname, pressure_list in custom_pressures:
            self._settings[cname] = pressure_list
        
        # Ensure 'Default' exists
        if not 'Default' in self._settings:
            self._settings['Default'] = self.get_default_setting()


    @classmethod
    def get_pref_key(cls, name):
        return "%s.%s" % (cls._PREF_KEY_BASE, name)

    @property
    def settings(self):
        return self._settings

    @property
    def current_setting(self):
        prefs_key = StrokePressureSettings.get_pref_key('last_used')
        return self.app.preferences.get(prefs_key, 'Default')

    @current_setting.setter
    def current_setting(self, name):
        prefs_key = StrokePressureSettings.get_pref_key('last_used')
        self.app.preferences[prefs_key] = name

    @property
    def last_used_setting(self):
        prefs_key = StrokePressureSettings.get_pref_key('last_used')
        return self.app.preferences.get(prefs_key, 'Default')

    def _set_last_used(self, name):
        prefs_key = StrokePressureSettings.get_pref_key('last_used')
        self.app.preferences[prefs_key] = name

    def get_default_setting(self):
        """ Not the named 'Default' setting,
        return the application default setting.
        """
        return _PRESSURE_VARIATIONS[0][1]

    def finalize(self):
        """ Finalize current settings into app.preference
        """
        save_settings = []

        for cname in self._settings:
            save_settings.append((cname, self._settings[cname]))

        prefs_key = StrokePressureSettings.get_pref_key('settings')
        self.app.preferences[prefs_key] = save_settings


    def points_changed_cb(self, curve):
        """ callback for when StrokeCurveWidget point has been changed.
        """

        setting = self._settings[self.current_setting]
        for i in range(min(len(curve.points),4)):
            setting[i] = curve.points[i] # no copy needed,because it is tuple.

        if self._idle_srcid is None:
            self._idle_srcid = GLib.idle_add(self._values_changed_idle_cb)

    def _values_changed_idle_cb(self):
        # Aggregate, idle-state callback for multiple adjustments being changed
        # in a single event. Queues redraws, and runs observers. The curve sets
        # multiple settings at once, and we might as well not queue too many
        # redraws.
        if self._idle_srcid is not None:
            current_mode = self.app.doc.modes.top
            if hasattr(current_mode, 'redraw_item_cb'):
                # Redraw last_line when settings are adjusted in the adjustment Curve
                GLib.idle_add(current_mode.redraw_item_cb)
            for func in self.observers:
                func(self._changed_settings)
            self._changed_settings = set()
            self._idle_srcid = None
        return False


class StrokeCurveWidget (gui.curve.CurveWidget):
    """Graph of pressure by distance, tied to the central LineModeSettings"""

    ## Class constants

    _CURVE_STEP = 0.05 # The smoothness of curve(0.0 - 1.0).
                       # lower value is smoother.

    def __init__(self):
        from application import get_app
        self.app = get_app()
        super(StrokeCurveWidget, self).__init__(npoints=4, 
                             changed_cb=self._changed_cb)

        self.setting_changed_cb()
        self._update()


    def setting_changed_cb(self):
        name = self.app.stroke_pressure_settings.current_setting
        preset_seq = self.app.stroke_pressure_settings.settings[name]
        for i, value in enumerate(preset_seq):
            if i >= 4:
                break
            self.set_point(i, value)
        self.queue_draw()

    def _update(self):
        # we needs this method ,called from superclass 
        self.queue_draw()


    def _changed_cb(self, curve):
        """Updates the linemode pressure settings when the curve is altered"""
        self.app.stroke_pressure_settings.points_changed_cb(self)

    def draw_cb(self, widget, cr):

        super(StrokeCurveWidget, self).draw_cb(widget, cr)

        width, height = self.get_display_area()
        if width <= 0 or height <= 0:
            return

        def get_disp(x, y):
            return (x * width + gui.curve.RADIUS, 
                    y * height + gui.curve.RADIUS)

        
        cr.save()

        # [TODO] we need choose color which is friendly with
        # the theme which is used by end-user.
        cr.set_source_rgb(0.4,0.4,0.8)

        ox, oy = get_disp(*self._get_curve_value(0.0))
        cr.move_to(ox, oy)
        cur_step = self._CURVE_STEP
        while cur_step < 1.0:
            cx, cy = get_disp(*self._get_curve_value(cur_step))
            cr.line_to(cx, cy)
            cr.stroke()
            cr.move_to(cx, cy)
            cur_step+= self._CURVE_STEP

        # don't forget draw final segment
        cx, cy = get_disp(*self._get_curve_value(1.0))
        cr.line_to(cx, cy)
        cr.stroke()
        cr.restore()
        return True

    def _get_curve_value(self, step):
        """ Treat 4 points of self.points as bezier-control-points
        and get curve interpolated value.
        but to get minimum value(it is reversed to maximum pressure)
        does not treat self.points as single cubic bezier curve,
        but two connected bezier curve.
        if we use cubic one,it never reachs the top.
        """
        bx,by = self.points[1]
        cx,cy = self.points[2]

        xp = (bx + (cx - bx) / 2,
              by + (cy - by) / 2)

        if step <= 0.5:
            t_step = step * 2
            ap = self.points[0]
            bp = self.points[1]
            cp = xp

        else:
            t_step = (step - 0.5) * 2
            ap = xp
            bp = self.points[2]
            cp = self.points[3]

        return ( gui.drawutils.get_bezier(
                    ap[0], bp[0], cp[0], t_step),
                 gui.drawutils.get_bezier(
                    ap[1], bp[1], cp[1], t_step))
        

    def get_pressure_value(self, step):
        junk, value = self._get_curve_value(step)
        return lib.helpers.clamp(1.0 - value, 0.0, 1.0)

class OptionsPresenter_ExInking (object):
    """Presents UI for directly editing point values etc."""

    variation_preset_store = None

    @classmethod
    def init_variation_preset_store(cls):
        if cls.variation_preset_store == None:
            from application import get_app
            _app = get_app()
            store = Gtk.ListStore(str, int)
            for i,name in enumerate(_app.stroke_pressure_settings.settings):
                store.append((name,i))
            cls.variation_preset_store = store

    def __init__(self):
        super(OptionsPresenter_ExInking, self).__init__()
        from application import get_app
        self._app = get_app()
        self._options_grid = None
        self._point_values_grid = None
        self._pressure_adj = None
        self._xtilt_adj = None
        self._ytilt_adj = None
        self._dtime_adj = None
        self._dtime_label = None
        self._dtime_scale = None
        self._insert_button = None
        self._delete_button = None
        self._apply_variation_button = None
        self._variation_preset_combo = None

        self._updating_ui = False
        self._target = (None, None)

        OptionsPresenter_ExInking.init_variation_preset_store()
        

    def _ensure_ui_populated(self):
        if self._options_grid is not None:
            return
        app = self._app
        prefs = app.preferences
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
        self._hide_nodes_check = builder.get_object("hide_nodes_checkbutton")
        # Radiobuttons for auto-apply.
        n_radio = builder.get_object("autoapply_none_radio")
        c_radio = builder.get_object("autoapply_cull_radio")
        s_radio = builder.get_object("autoapply_simple_radio")
        radios = {
            n_radio : None,
            c_radio : "cull",
            s_radio : "simple"
        }
        r_label = prefs.get(_Prefs.AUTOAPPLY_PREF, _Prefs.DEFAULT_AUTOAPPLY)
        for cr, cl in radios.items():
            if cl == r_label:
                cr.set_active(True)
                break
        self._autoapply_radios = radios
        # Thershold slider for auto-apply.
        t_scale = builder.get_object("autoapply_threshold_scale")
        t_scale.set_digits(0) # Cant we set this from glade??
        t_adj = builder.get_object("autoapply_threshold_adj")
        t_adj.set_value(
            prefs.get(_Prefs.THRESHOLD_PREF, _Prefs.DEFAULT_THRESHOLD)
        )
        self._autoapply_threshold_adj = t_adj
        # Pressure variation.
        apply_btn = builder.get_object("apply_variation_button")
        apply_btn.set_sensitive(False)
        self._apply_variation_button = apply_btn
        base_grid = builder.get_object("points_editing_grid")
        toolbar = gui.widgets.inline_toolbar(
            app,
            [
                ("SimplifyNodes", "mypaint-layer-group-new-symbolic"),
                ("CullNodes", "mypaint-add-symbolic"),
                ("AverageNodesAngle", "mypaint-remove-symbolic"),
                ("AverageNodesDistance", "mypaint-up-symbolic"),
                ("AverageNodesPressure", "mypaint-down-symbolic"),
            ]
        )
        style = toolbar.get_style_context()
        style.set_junction_sides(Gtk.JunctionSides.TOP)
        base_grid.attach(toolbar, 0, 0, 2, 1)

        self.init_linecurve_widget(1, base_grid)
        self.init_variation_preset_combo(2, base_grid,
                self._apply_variation_button)

    def init_linecurve_widget(self, row, box):

        # XXX code duplication from gui.linemode.LineModeOptionsWidget
        curve = StrokeCurveWidget()
        curve.set_size_request(175, 125)
        self.curve = curve
        exp = Gtk.Expander()
        exp.set_label(_("Pressure variation..."))
        exp.set_use_markup(False)
        exp.add(curve)
        box.attach(exp, 0, row, 2, 1)
        exp.set_expanded(True)

    def init_variation_preset_combo(self, row, box, ref_button=None):
        combo = Gtk.ComboBox.new_with_model(
                OptionsPresenter_ExInking.variation_preset_store)
        cell = Gtk.CellRendererText()
        combo.pack_start(cell,True)
        combo.add_attribute(cell,'text',0)
        combo.set_active(0)
        combo.set_sensitive(True) # variation preset always can be changed
        if ref_button:
            combo.set_margin_top(ref_button.get_margin_top())
            combo.set_margin_right(4)
            combo.set_margin_bottom(ref_button.get_margin_bottom())
            box.attach(combo, 0, row, 1, 1)
        else:
            box.attach(combo, 0, row, 2, 1)
        combo.connect('changed', self._variation_preset_combo_changed_cb)
        self._variation_preset_combo = combo

        # set last active setting.
        last_used = self._app.stroke_pressure_settings.last_used_setting
        def walk_combo_cb(model, path, iter, user_data):
            if self.variation_preset_store[iter][0] == last_used:
                combo.set_active_iter(iter)
                return True

        self.variation_preset_store.foreach(walk_combo_cb,None)



    @property
    def widget(self):
        self._ensure_ui_populated()
        return self._options_grid

    @property
    def target(self):
        """The active mode and its current node index

        :returns: a pair of the form (inkmode, node_idx)
        :rtype: tuple

        Updating this pair via the property also updates the UI.
        The target mode most be an InkingTool instance.

        """
        mode_ref, node_idx = self._target
        mode = None
        if mode_ref is not None:
            mode = mode_ref()
        return (mode, node_idx)

    @target.setter
    def target(self, targ):
        inkmode, cn_idx = targ
        inkmode_ref = None
        if inkmode:
            inkmode_ref = weakref.ref(inkmode)
        self._target = (inkmode_ref, cn_idx)
        # Update the UI
        if self._updating_ui:
            return
        self._updating_ui = True
        try:
            self._ensure_ui_populated()
            if 0 <= cn_idx < len(inkmode.nodes):
                cn = inkmode.nodes[cn_idx]
                self._pressure_adj.set_value(cn.pressure)
                self._xtilt_adj.set_value(cn.xtilt)
                self._ytilt_adj.set_value(cn.ytilt)
                if cn_idx > 0:
                    sensitive = True
                    dtime = inkmode.get_node_dtime(cn_idx)
                else:
                    sensitive = False
                    dtime = 0.0
                for w in (self._dtime_scale, self._dtime_label):
                    w.set_sensitive(sensitive)
                self._dtime_adj.set_value(dtime)
                self._point_values_grid.set_sensitive(True)
            else:
                self._point_values_grid.set_sensitive(False)
            self._insert_button.set_sensitive(inkmode.can_insert_node(cn_idx))
            self._delete_button.set_sensitive(inkmode.can_delete_node(cn_idx))
            #self._period_adj.set_value(self._app.preferences.get(
                #"inktool.capture_period_factor", 1))
            self._apply_variation_button.set_sensitive(len(inkmode.nodes) > 2)
        finally:
            self._updating_ui = False

    def _variation_preset_combo_changed_cb(self, widget):
        iter = self._variation_preset_combo.get_active_iter()
        # TODO app.stroke_pressure_settings...? 
        # it sounds rather strange...
        self._app.stroke_pressure_settings.current_setting = \
                self.variation_preset_store[iter][0]
        self.curve.setting_changed_cb()

    def _pressure_adj_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        inkmode, node_idx = self.target
        inkmode.update_node(node_idx, pressure=float(adj.get_value()))

    def _dtime_adj_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        inkmode, node_idx = self.target
        inkmode.set_node_dtime(node_idx, adj.get_value())

    def _xtilt_adj_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        inkmode, node_idx = self.target
        inkmode.update_node(node_idx, xtilt=float(adj.get_value()))

    def _ytilt_adj_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        inkmode, node_idx = self.target
        inkmode.update_node(node_idx, ytilt=float(adj.get_value()))

    def _insert_point_button_clicked_cb(self, button):
        inkmode, node_idx = self.target
        if inkmode.can_insert_node(node_idx):
            inkmode.insert_node(node_idx)

    def _delete_point_button_clicked_cb(self, button):
        inkmode, node_idx = self.target
        if inkmode.can_delete_node(node_idx):
            inkmode.delete_node(node_idx)

    def _simplify_points_button_clicked_cb(self, button):
        inkmode, node_idx = self.target
        if len(inkmode.nodes) > 3:
            inkmode.simplify_nodes()

    def _cull_points_button_clicked_cb(self, button):
        inkmode, node_idx = self.target
        if len(inkmode.nodes) > 2:
            inkmode.cull_nodes()

    def _average_angle_clicked_cb(self,button):
        inkmode, node_idx = self.target
        if inkmode:
            inkmode.average_nodes_angle()

    def _average_distance_clicked_cb(self,button):
        inkmode, node_idx = self.target
        if inkmode:
            inkmode.average_nodes_distance()

    def _average_pressure_clicked_cb(self,button):
        inkmode, node_idx = self.target
        if inkmode:
            inkmode.average_nodes_pressure()

    def _apply_variation_button_cb(self, button):
        inkmode, node_idx = self.target
        if inkmode:
            if len(inkmode.nodes) > 1:
                # To LineModeCurveWidget,
                # we can access control points as "points" attribute.
                inkmode.apply_pressure_from_curve_widget()

    def _autoapply_radio_toggled_cb(self, button):
        inkmode, node_idx = self.target
        if inkmode:
            radios = self._autoapply_radios
            assert button in radios
            if button.get_active():
                value = radios[button]
                inkmode.set_autoapply(value)
                prefs = self._app.preferences
                prefs[_Prefs.AUTOAPPLY_PREF] = value

    def _autoapply_threshold_adj_value_changed_cb(self, adj):
        inkmode, node_idx = self.target
        if inkmode:
            inkmode.set_autoapply_threshold(adj.get_value())
            prefs[_Prefs.THRESHOLD_PREF] = adj.get_value()





