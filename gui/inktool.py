# This file is part of MyPaint.
# Copyright (C) 2015 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


## Imports
from __future__ import print_function

import math
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
        self._queue_redraw_curve()
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()

        # the method should return deleted nodes count
        result = method(self, *args)
        assert type(result) == int

        if result > 0:
            self.options_presenter.target = (self, self.current_node_index)
            self._queue_redraw_curve()
            self._queue_redraw_all_nodes()
            self._queue_draw_buttons()
        return result
    return _decorator


## Class defs



class _Phase:
    """Enumeration of the states that an InkingMode can be in"""
    CAPTURE = 0
    ADJUST = 1
    ADJUST_PRESSURE = 2
    ADJUST_PRESSURE_ONESHOT = 4
    CHANGE_PHASE = 5


_NODE_FIELDS = ("x", "y", "pressure", "xtilt", "ytilt", "time")


class _Node (collections.namedtuple("_Node", _NODE_FIELDS)):
    """Recorded control point, as a namedtuple.

    Node tuples have the following 6 fields, in order

    * x, y: model coords, float
    * pressure: float in [0.0, 1.0]
    * xtilt, ytilt: float in [-1.0, 1.0]
    * time: absolute seconds, float
    """


class _EditZone:
    """Enumeration of what the pointer is on in the ADJUST phase"""
    EMPTY_CANVAS = 0  #: Nothing, empty space
    CONTROL_NODE = 1  #: Any control node; see target_node_index
    REJECT_BUTTON = 2  #: On-canvas button that abandons the current line
    ACCEPT_BUTTON = 3  #: On-canvas button that commits the current line

class _CapturePeriodSetting(object):
    """Capture Period Setting class,to ease for user to customize it"""
   #BASE_INTERNODE_DISTANCE_MIDDLE = 30   # display pixels
   #BASE_INTERNODE_DISTANCE_ENDS = 10   # display pixels

   #INTERPOLATION_MAX_SLICE_TIME = 1/200.0   # seconds
   #INTERPOLATION_MAX_SLICE_DISTANCE = 20   # model pixels
   #INTERPOLATION_MAX_SLICES = MAX_INTERNODE_DISTANCE_MIDDLE * 5

    @property
    def internode_distance_middle(self):
        return 30 * self.factor # display pixels

    @property
    def internode_distance_ends(self):
        return 10 * self.factor # display pixels

    @property
    def max_internode_time(self):
        return 1/100.0 #1/(100.0 / self.factor) # default MAX TIME is 1/100.0

    @property
    def min_internode_time(self):
        return 1/200.0 #1/(200.0 / self.factor) # default MIN TIME is 1/200.0

    # Captured input nodes are then interpolated with a spline.
    # The code tries to make nice smooth input for the brush engine,
    # but avoids generating too much work.

    @property
    def interpolation_max_slices(self):
        return self.internode_distance_middle * 5

    @property
    def interpolation_max_slice_distance(self):
        return 20# * self.factor # model pixels

    @property
    def interpolation_max_slice_time(self):
        return 1/200.0 #1/(200.0 / self.factor)

        # In other words, limit to a set number of interpolation slices
        # per display pixel at the time of stroke capture.

    def __init__(self):
        self.factor=1.0

    def set_factor(self, value):
        self.factor = value




class InkingMode (gui.mode.ScrollableModeMixin,
                  gui.mode.BrushworkModeMixin,
                  gui.mode.DragMode):

    ## Metadata properties

    ACTION_NAME = "InkingMode"
    pointer_behavior = gui.mode.Behavior.PAINT_FREEHAND
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW

    ## Metadata methods

    @classmethod
    def get_name(cls):
        return _(u"Inking")

    def get_usage(self):
        return _(u"Draw, and then adjust smooth lines")

    @property
    def inactive_cursor(self):
        return None

    @property
    def active_cursor(self):
        if self.phase == _Phase.ADJUST:
            if self.zone == _EditZone.CONTROL_NODE:
                return self._crosshair_cursor
            elif self.zone != _EditZone.EMPTY_CANVAS: # assume button
                return self._arrow_cursor

        elif self.phase in (_Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT):
            if self.zone == _EditZone.CONTROL_NODE:
                return self._cursor_move_nw_se

        return None

    ## Override action
    permitted_switch_actions = None
    _enable_switch_actions = set()   # Any action is permitted,for now.
    _disable_switch_actions=set(gui.mode.BUTTON_BINDING_ACTIONS).union([
            'RotateViewMode',
            'ZoomViewMode',
            'PanViewMode',
            "SelectionMode",
        ])
    @classmethod
    def enable_switch_actions(cls, flag):
        if flag:
            cls.permitted_switch_actions = cls._enable_switch_actions
        else:
            cls.permitted_switch_actions = cls._disable_switch_actions



    ## Class config vars

    # Input node capture settings:
   #MAX_INTERNODE_DISTANCE_MIDDLE = 30   # display pixels
   #MAX_INTERNODE_DISTANCE_ENDS = 10   # display pixels
   #MAX_INTERNODE_TIME = 1/100.0   # seconds

    # Captured input nodes are then interpolated with a spline.
    # The code tries to make nice smooth input for the brush engine,
    # but avoids generating too much work.
   #INTERPOLATION_MAX_SLICE_TIME = 1/200.0   # seconds
   #INTERPOLATION_MAX_SLICE_DISTANCE = 20   # model pixels
   #INTERPOLATION_MAX_SLICES = MAX_INTERNODE_DISTANCE_MIDDLE * 5
        # In other words, limit to a set number of interpolation slices
        # per display pixel at the time of stroke capture.

    # Node value adjustment settings
   #MIN_INTERNODE_TIME = 1/200.0   # seconds (used to manage adjusting)

    CAPTURE_SETTING = _CapturePeriodSetting()

    ## Other class vars

    _OPTIONS_PRESENTER = None   #: Options presenter singleton

    drag_offset = gui.ui_utils.DragOffset()


    ## Pressure oncanvas edit settings

    # Pressure editing key modifiers,single node and with nearby nodes.
    # these can be hard-coded,but we might need some customizability later.
    _PRESSURE_MOD_MASK = Gdk.ModifierType.SHIFT_MASK
    _PRESSURE_NEARBY_MOD_MASK = Gdk.ModifierType.CONTROL_MASK

    _PRESSURE_WHEEL_STEP = 0.025 # pressure modifying step,for mouse wheel


    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):
        super(InkingMode, self).__init__(**kwargs)

        #+ initialize selected nodes - 
        #+ place this prior to _reset_nodes()
        self.selected_nodes=[]

        self.phase = _Phase.CAPTURE
        self.zone = _EditZone.EMPTY_CANVAS
        self.current_node_index = None  #: Node active in the options ui
        self.target_node_index = None  #: Node that's prelit
        self._overlays = {}  # keyed by tdw
        self._reset_nodes()
        self._reset_capture_data()
        self._reset_adjust_data()
        self._task_queue = collections.deque()  # (cb, args, kwargs)
        self._task_queue_runner_id = None
        self._click_info = None   # (button, zone)
        self._current_override_cursor = None
        # Button pressed while drawing
        # Not every device sends button presses, but evdev ones
        # do, and this is used as a workaround for an evdev bug:
        # https://github.com/mypaint/mypaint/issues/223
        self._button_down = None
        self._last_good_raw_pressure = 0.0
        self._last_good_raw_xtilt = 0.0
        self._last_good_raw_ytilt = 0.0


        #+ Hiding nodes functionality
        self._hide_nodes = False

        #+ returning phase.for special phase changing case.
        self._returning_phase = None

        #+ previous scroll event time.
        #  in some environment, Gdk.ScrollDirection.UP/DOWN/LEFT/RIGHT
        #  and Gdk.ScrollDirection.SMOOTH might happen at same time.
        #  to reject such event, this attribute needed.
        self._prev_scroll_time = None
        
        #+ affecting range related.
        self._range_radius = None # Invalid values, set later inside property.
        self._range_factor = None
       #self.set_range_radius(self.doc.app.preferences.get(
       #                            "inktool.adjust_range_radius", 0))
       #self.set_range_factor(self.doc.app.preferences.get(
       #                            "inktool.adjust_range_factor", 0))
    
    @property
    def range_factor(self):
        if self._range_factor == None:
            self._range_factor = self.doc.app.preferences.get(
                                    "inktool.adjust_range_factor", 0)


    def _reset_nodes(self):
        self.nodes = []  # nodes that met the distance+time criteria
        self._reset_selected_nodes(None)


    def _reset_capture_data(self):
        self._last_event_node = None  # node for the last event
        self._last_node_evdata = None  # (xdisp, ydisp, tmilli) for nodes[-1]

    def _reset_adjust_data(self):
        self.zone = _EditZone.EMPTY_CANVAS
        self.current_node_index = None
        self.target_node_index = None
        self._dragged_node_start_pos = None
        self.drag_offset.reset()

        # Multiple selected nodes.
        # This is a index list of node from self.nodes
        self._reset_selected_nodes()

        self.hide_nodes = False


    def _reset_selected_nodes(self, initial_idx=None):
        """ Resets selected_nodes list and assign
        initial index,if needed.

        :param initial_idx: initial node index.in most case,
                            node will manipurate by solo.
                            it might be inefficient to
                            generate list each time s solo node
                            is moved,so use this parameter in such case.
        """

        if initial_idx == None:
            if len(self.selected_nodes) > 0:
                self.selected_nodes=[]
        elif len(self.selected_nodes) == 0:
            self.selected_nodes.append(initial_idx)
        elif len(self.selected_nodes) == 1:
            self.selected_nodes[0] = initial_idx
        else:
            self.selected_nodes = [initial_idx, ]




    def _ensure_overlay_for_tdw(self, tdw):
        overlay = self._overlays.get(tdw)
        if not overlay:
            overlay = Overlay(self, tdw)
            tdw.display_overlays.append(overlay)
            self._overlays[tdw] = overlay
        return overlay

    def _is_active(self):
        for mode in self.doc.modes:
            if mode is self:
                return True
        return False

    def _discard_overlays(self):
        for tdw, overlay in self._overlays.items():
            tdw.display_overlays.remove(overlay)
            tdw.queue_draw()
        self._overlays.clear()

    def enter(self, doc, **kwds):
        """Enters the mode: called by `ModeStack.push()` etc."""
        super(InkingMode, self).enter(doc, **kwds)
        if not self._is_active():
            self._discard_overlays()
        self._ensure_overlay_for_tdw(self.doc.tdw)
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
        self.drag_offset.reset()
        InkingMode.enable_switch_actions(True)

    def leave(self, **kwds):
        """Leaves the mode: called by `ModeStack.pop()` etc."""
        if not self._is_active():
            self._discard_overlays()
        self._stop_task_queue_runner(complete=True)
        InkingMode.enable_switch_actions(False)
        super(InkingMode, self).leave(**kwds)  # supercall will commit

    def checkpoint(self, flush=True, **kwargs):
        """Sync pending changes from (and to) the model

        If called with flush==False, this is an override which just
        redraws the pending stroke with the current brush settings and
        color. This is the behavior our testers expect:
        https://github.com/mypaint/mypaint/issues/226

        When this mode is left for another mode (see `leave()`), the
        pending brushwork is committed properly.

        """
        if flush:
            # Commit the pending work normally
            self._start_new_capture_phase(rollback=False)
            super(InkingMode, self).checkpoint(flush=flush, **kwargs)
        else:
            # Queue a re-rendering with any new brush data
            # No supercall
            self._stop_task_queue_runner(complete=False)
            self._queue_draw_buttons()
            self._queue_redraw_all_nodes()
            self._queue_redraw_curve()

    def _start_new_capture_phase(self, rollback=False):
        """Let the user capture a new ink stroke"""
        if rollback:
            self._stop_task_queue_runner(complete=False)
            self.brushwork_rollback_all()
        else:
            self._stop_task_queue_runner(complete=True)
            self.brushwork_commit_all()
        self.options_presenter.target = (self, None)
        self._queue_draw_buttons()
        self._queue_redraw_all_nodes()
        self._reset_nodes()
        self._reset_capture_data()
        self._reset_adjust_data()
        self.phase = _Phase.CAPTURE
        InkingMode.enable_switch_actions(True)

    ## Raw event handling (prelight & zone selection in adjust phase)
    def button_press_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False
        self._update_zone_and_target(tdw, event.x, event.y)
        self._update_current_node_index()
        if self.phase in (_Phase.ADJUST, _Phase.ADJUST_PRESSURE):
            button = event.button
            if (self.current_node_index is not None and
                    button == 1 and
                    self.phase == _Phase.ADJUST and
                    event.state & self.__class__._PRESSURE_MOD_MASK ==
                    self.__class__._PRESSURE_MOD_MASK):
            
                # Entering On-canvas Pressure Adjustment Phase!
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
            # Normal ADJUST/ADJUST_PRESSURE Phase.

                if self.zone in (_EditZone.REJECT_BUTTON,
                                 _EditZone.ACCEPT_BUTTON):
                    if (button == 1 and
                            event.type == Gdk.EventType.BUTTON_PRESS):
                        self._click_info = (button, self.zone)
                        return False
                    # FALLTHRU: *do* allow drags to start with other buttons
                elif self.zone == _EditZone.EMPTY_CANVAS:
                    if self.phase == _Phase.ADJUST_PRESSURE:
                        self.phase = _Phase.ADJUST
                        self._queue_redraw_all_nodes()
                    else:
                        self._start_new_capture_phase(rollback=False)
                        assert self.phase == _Phase.CAPTURE

                    # FALLTHRU: *do* start a drag
                else:
                    # clicked a node.

                    if button == 1:
                        # 'do_reset' is a selection reset flag
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
        elif self.phase == _Phase.ADJUST_PRESSURE_ONESHOT:
            # XXX Not sure what to do here.
            pass
        else:
            raise NotImplementedError("Unrecognized zone %r", self.zone)
        # Update workaround state for evdev dropouts
        self._button_down = event.button
        self._last_good_raw_pressure = 0.0
        self._last_good_raw_xtilt = 0.0
        self._last_good_raw_ytilt = 0.0

        # Supercall: start drags etc
        return super(InkingMode, self).button_press_cb(tdw, event)

    def button_release_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False

        if self.phase in (_Phase.ADJUST , _Phase.ADJUST_PRESSURE):
            if self._click_info:
                button0, zone0 = self._click_info
                if event.button == button0:
                    if self.zone == zone0:
                        if zone0 == _EditZone.REJECT_BUTTON:
                            self.accept_edit()
                        elif zone0 == _EditZone.ACCEPT_BUTTON:
                            self.discard_edit()
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
        elif self.phase == _Phase.CAPTURE:
            # Update options_presenter when capture phase end
            self.options_presenter.target = (self, None)

        # Update workaround state for evdev dropouts
        self._button_down = None
        self._last_good_raw_pressure = 0.0
        self._last_good_raw_xtilt = 0.0
        self._last_good_raw_ytilt = 0.0


        # other processing 
        if self.phase in (_Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT):
            self.options_presenter.target = (self, self.current_node_index)

        # Supercall: stop current drag
        return super(InkingMode, self).button_release_cb(tdw, event)

    def motion_notify_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False

        self._update_zone_and_target(tdw, event.x, event.y)
        return super(InkingMode, self).motion_notify_cb(tdw, event)

    def _update_current_node_index(self):
        """Updates current_node_index from target_node_index & redraw"""
        new_index = self.target_node_index
        old_index = self.current_node_index
        if new_index == old_index:
            return
        self.current_node_index = new_index
        self.current_node_changed(new_index)
        self.options_presenter.target = (self, new_index)
        for i in (old_index, new_index):
            if i is not None:
                self._queue_draw_node(i)

    @lib.observable.event
    def current_node_changed(self, index):
        """Event: current_node_index was changed"""

    def _search_target_node(self, tdw, x, y):
        """ utility method: to commonize processing,
        even in inherited classes.
        """
        hit_dist = gui.style.DRAGGABLE_POINT_HANDLE_SIZE + 12
        new_target_node_index = None
        for i, node in reversed(list(enumerate(self.nodes))):
            node_x, node_y = tdw.model_to_display(node.x, node.y)
            d = math.hypot(node_x - x, node_y - y)
            if d > hit_dist:
                continue
            new_target_node_index = i
            break
        return new_target_node_index

    def _update_zone_and_target(self, tdw, x, y):
        """Update the zone and target node under a cursor position"""

        self._ensure_overlay_for_tdw(tdw)
        new_zone = _EditZone.EMPTY_CANVAS

        if not self.in_drag:
            if self.phase in (_Phase.ADJUST,
                    _Phase.ADJUST_PRESSURE):

                new_target_node_index = None
                # Test buttons for hits
                overlay = self._ensure_overlay_for_tdw(tdw)
                hit_dist = gui.style.FLOATING_BUTTON_RADIUS
                button_info = [
                    (_EditZone.ACCEPT_BUTTON, overlay.accept_button_pos),
                    (_EditZone.REJECT_BUTTON, overlay.reject_button_pos),
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
                if new_zone == _EditZone.EMPTY_CANVAS:
                    new_target_node_index = self._search_target_node(tdw, x, y)
                    if new_target_node_index != None:
                        new_zone = _EditZone.CONTROL_NODE

                # Update the prelit node, and draw changes to it
                if new_target_node_index != self.target_node_index:
                    if self.target_node_index is not None:
                        self._queue_draw_node(self.target_node_index)
                    self.target_node_index = new_target_node_index
                    if self.target_node_index is not None:
                        self._queue_draw_node(self.target_node_index)


            elif self.phase ==  _Phase.ADJUST_PRESSURE_ONESHOT:
                self.target_node_index = self._search_target_node(tdw, x, y)
                if self.target_node_index != None:
                    new_zone = _EditZone.CONTROL_NODE

        elif self.phase in (_Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT):
            # Always control node,in pressure editing.
            new_zone = _EditZone.CONTROL_NODE

        # Update the zone, and assume any change implies a button state
        # change as well (for now...)
        if self.zone != new_zone:
            self.zone = new_zone
            self._ensure_overlay_for_tdw(tdw)
            self._queue_draw_buttons()
        # Update the "real" inactive cursor too:
        if not self.in_drag:
            cursor = None
            if self.phase in (_Phase.ADJUST, _Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT):
                if self.zone == _EditZone.CONTROL_NODE:
                    cursor = self._crosshair_cursor
                elif self.zone != _EditZone.EMPTY_CANVAS: # assume button
                    cursor = self._arrow_cursor
            if cursor is not self._current_override_cursor:
                tdw.set_override_cursor(cursor)
                self._current_override_cursor = cursor


    ## Redraws

    def _queue_draw_buttons(self):
        """Redraws the accept/reject buttons on all known view TDWs"""
        for tdw, overlay in self._overlays.items():
            overlay.update_button_positions()
            positions = (
                overlay.reject_button_pos,
                overlay.accept_button_pos,
            )
            for pos in positions:
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


    def _queue_draw_node(self, i, offset_vec=None):
        """Redraws a specific control node on all known view TDWs"""
        node = self.nodes[i]
        if self.current_node_index != None:
            basept = self.nodes[self.current_node_index]
        else:
            basept = None

        if offset_vec == None:
            offset_vec = self._generate_offset_vector()

        for tdw in self._overlays:
           #if len(self.selected_nodes) > 1:
           #    if i in self.selected_nodes:
           #        offsets = gui.drawutils.calc_ranged_offset(basept, node,
           #                self.range_radius, self.range_factor,
           #                offset_vec)
           #    else:
           #        offsets = (node.x, node.y)
           #else:
            offsets = gui.drawutils.calc_ranged_offset(basept, node,
                    self.range_radius, self.range_factor,
                    offset_vec)

            x, y = tdw.model_to_display(*offsets)
            x = math.floor(x)
            y = math.floor(y)
            size = math.ceil(gui.style.DRAGGABLE_POINT_HANDLE_SIZE * 2)
            tdw.queue_draw_area(x-size, y-size, size*2+1, size*2+1)

    def _queue_draw_selected_nodes(self):
        offset_vec = self._generate_offset_vector()
        for i in self.selected_nodes:
            self._queue_draw_node(i, offset_vec = offset_vec)

    def _queue_redraw_all_nodes(self):
        """Redraws all nodes on all known view TDWs"""
        offset_vec = self._generate_offset_vector()
        for i in xrange(len(self.nodes)):
            self._queue_draw_node(i, offset_vec = offset_vec)

    def _generate_offset_vector(self):
        """ Generates node-moving offset vector.
        """ 
        dx,dy = self.drag_offset.get_model_offset()
        offset_len = math.hypot(dx, dy)
        if offset_len > 0.0:
            return (offset_len,
                    dx / offset_len,
                    dy / offset_len)




    def _queue_redraw_curve(self):
        """Redraws the entire curve on all known view TDWs"""
        self._stop_task_queue_runner(complete=False)
        if self.current_node_index != None:
            base_node = self.nodes[self.current_node_index]
        else:
            base_node = None
        offset_vec = self._generate_offset_vector()
        model_radius = None # Initial value, to be rewritten at first loop.
        
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
            
            #for p_1, p0, p1, p2 in gui.drawutils.spline_iter_2(
                        #self.nodes,
                        #self.selected_nodes,
                        #(dx,dy)):
                        
            if model_radius is None:
                model_radius, junk = tdw.display_to_model(self.range_radius, 0)
                model_radius = abs(model_radius)
            
            for p_1, p0, p1, p2 in gui.drawutils.spline_iter_3(
                                    self.nodes,
                                    base_node,
                                    model_radius,
                                    self.range_factor,
                                    offset_vec):
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
        steps_t = dtime_p0_p1_real / self.CAPTURE_SETTING.interpolation_max_slice_time#self.INTERPOLATION_MAX_SLICE_TIME
        dist_p1_p2 = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
        steps_d = dist_p1_p2 / self.CAPTURE_SETTING.interpolation_max_slice_distance #self.INTERPOLATION_MAX_SLICE_DISTANCE
        steps_max = float(self.CAPTURE_SETTING.interpolation_max_slices)#self.INTERPOLATION_MAX_SLICES)
        steps = math.ceil(min(steps_max, max([2, steps_t, steps_d])))
        for i in xrange(int(steps) + 1):
            t = i / steps
            point = gui.drawutils.spline_4p(t, p_1, p0, p1, p2)
            x, y, pressure, xtilt, ytilt, t_abs = point
            pressure = lib.helpers.clamp(pressure, 0.0, 1.0)
            xtilt = lib.helpers.clamp(xtilt, -1.0, 1.0)
            ytilt = lib.helpers.clamp(ytilt, -1.0, 1.0)
            t_abs = max(last_t_abs, t_abs)
            dtime = t_abs - last_t_abs
            self.stroke_to(
                model, dtime, x, y, pressure, xtilt, ytilt,
                auto_split=False,
            )
            last_t_abs = t_abs
        state["t_abs"] = last_t_abs

    def _queue_task(self, callback, *args, **kwargs):
        """Append a task to be done later in an idle cycle"""
        self._task_queue.append((callback, args, kwargs))

    def _start_task_queue_runner(self):
        """Begin processing the task queue, if not already going"""
        if self._task_queue_runner_id is not None:
            return
        idler_id = GLib.idle_add(self._task_queue_runner_cb)
        self._task_queue_runner_id = idler_id

    def _stop_task_queue_runner(self, complete=True):
        """Halts processing of the task queue, and clears it"""
        if self._task_queue_runner_id is None:
            return
        if complete:
            for (callback, args, kwargs) in self._task_queue:
                callback(*args, **kwargs)
        self._task_queue.clear()
        GLib.source_remove(self._task_queue_runner_id)
        self._task_queue_runner_id = None

    def _task_queue_runner_cb(self):
        """Idle runner callback for the task queue"""
        try:
            callback, args, kwargs = self._task_queue.popleft()
        except IndexError:  # queue empty
            self._task_queue_runner_id = None
            return False
        else:
            callback(*args, **kwargs)
            return True

    ## Drag handling (both capture and adjust phases)

    def drag_start_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        mx, my = tdw.display_to_model(event.x, event.y)
        if self.phase == _Phase.CAPTURE:
            self._reset_nodes()
            self._reset_capture_data()
            self._reset_adjust_data()

            if event.state != 0:
                # To activate some mode override
                self._last_event_node = None
                return super(InkingMode, self).drag_start_cb(tdw, event)
            else:
                node = self._get_event_data(tdw, event)
                self.nodes.append(node)
                self._queue_draw_node(0)
                self._last_node_evdata = (event.x, event.y, event.time)
                self._last_event_node = node

        elif self.phase == _Phase.ADJUST:
            self._node_dragged = False
            if self.target_node_index is not None:
                node = self.nodes[self.target_node_index]
                self._dragged_node_start_pos = (node.x, node.y)

                self.drag_offset.start(mx, my)

        elif self.phase in (_Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT):
            pass
        elif self.phase == _Phase.CHANGE_PHASE:
            pass
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)


    def drag_update_cb(self, tdw, event, dx, dy):
        self._ensure_overlay_for_tdw(tdw)
        mx, my = tdw.display_to_model(event.x ,event.y)
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
                    max_dist = self.CAPTURE_SETTING.internode_distance_middle #MAX_INTERNODE_DISTANCE_MIDDLE
                    if len(self.nodes) < 2:
                        max_dist = self.CAPTURE_SETTING.internode_distance_ends #MAX_INTERNODE_DISTANCE_ENDS
                    append_node = (
                        dist > max_dist and
                        dt > self.CAPTURE_SETTING.max_internode_time #MAX_INTERNODE_TIME
                    )
                if append_node:
                    self.nodes.append(node)
                    self._queue_draw_node(len(self.nodes)-1)
                    self._queue_redraw_curve()
                    self._last_node_evdata = evdata
                self._last_event_node = node
            else:
                super(InkingMode, self).drag_update_cb(tdw, event, dx, dy)

        elif self.phase == _Phase.ADJUST:
            if self._dragged_node_start_pos:
                self._node_dragged = True
                x0, y0 = self._dragged_node_start_pos

                if len(self.selected_nodes) == 0:
                    disp_x, disp_y = tdw.model_to_display(x0, y0)
                    disp_x += event.x - self.start_x
                    disp_y += event.y - self.start_y
                    x, y = tdw.display_to_model(disp_x, disp_y)
                    self.update_node(self.target_node_index, x=x, y=y)
                else:
                    self._queue_draw_selected_nodes()
                    self.drag_offset.end(mx, my)
                    self._queue_draw_selected_nodes()

                self._queue_redraw_curve()

        elif self.phase in (_Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT):
            self._adjust_pressure_with_motion(dx, dy)
        elif self.phase == _Phase.CHANGE_PHASE:
            pass
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)

    def drag_stop_cb(self, tdw):
        self._ensure_overlay_for_tdw(tdw)
        if self.phase == _Phase.CAPTURE:

            if not self.nodes or self._last_event_node == None:
                return super(InkingMode, self).drag_stop_cb(tdw)

            node = self._last_event_node
            if self.nodes[-1] is not node:
                # When too close against last captured node,
                # delete it.
                d = math.hypot(self.nodes[-1].x - node.x,
                        self.nodes[-1].y - node.y)
                mid_d = tdw.display_to_model(
                        self.CAPTURE_SETTING.internode_distance_middle, 0)[0]
                # 'too close' means less than internode_distance_middle / 5
                if d < mid_d / 5.0:
                    self._queue_draw_node(len(self.nodes)-1) # To avoid glitch
                    del self.nodes[-1]
            
                self.nodes.append(node)


            self._reset_capture_data()
            self._reset_adjust_data()
            if len(self.nodes) > 1:
                self.phase = _Phase.ADJUST
                InkingMode.enable_switch_actions(False)
                self._queue_redraw_all_nodes()
                self._queue_redraw_curve()
                self._queue_draw_buttons()
            else:
                self._reset_nodes()
                tdw.queue_draw()
        elif self.phase == _Phase.ADJUST:

            # Finalize dragging motion to selected nodes.
            if self._node_dragged:

                self._queue_draw_selected_nodes() # to ensure erase them

               #dx, dy = self.drag_offset.get_model_offset()
               #
               #for idx in self.selected_nodes:
               #    cn = self.nodes[idx]
               #    self.nodes[idx] = cn._replace(x=cn.x + dx,
               #            y=cn.y + dy)

                for i, cn, x, y in self.enum_nodes_coord(tdw, convert_to_display=False):
                    if cn.x != x or cn.y != y:
                        self.nodes[i] = cn._replace(x=x, y=y)

                self.drag_offset.reset()

            self._dragged_node_start_pos = None
            self._queue_redraw_curve()
            self._queue_draw_buttons()
        elif self.phase in (_Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT):
            ## Pressure editing phase end.
            # Return to ADJUST phase.
            # simple but very important,to ensure entering normal editing.
            if self.phase == _Phase.ADJUST_PRESSURE_ONESHOT:
                self.phase = _Phase.ADJUST
            self._queue_redraw_curve()
            self._queue_draw_buttons()
        elif self.phase == _Phase.CHANGE_PHASE:
            pass
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)

        if self._returning_phase == None:
            self._returning_phase = self.phase

    def scroll_cb(self, tdw, event):
        """Handles scroll-wheel events, to adjust pressure."""
        if (self.phase in (_Phase.ADJUST, 
                _Phase.ADJUST_PRESSURE, 
                _Phase.ADJUST_PRESSURE_ONESHOT) 
                and self.target_node_index != None):

            if self._prev_scroll_time != event.time:
               #if len(self.selected_nodes) == 0:
               #    targets = (self.target_node_index,)
               #else:
               #    targets = self.selected_nodes
                targets = (self.target_node_index,)

                for idx in targets:
                    node = self.nodes[idx]
                    new_pressure = node.pressure

                    junk, y = gui.ui_utils.get_scroll_delta(event, self._PRESSURE_WHEEL_STEP)
                    new_pressure += y

                    if new_pressure != node.pressure:
                        self.nodes[idx]=node._replace(pressure=new_pressure)

                    if idx == self.target_node_index:
                        self.options_presenter.target = (self, idx)

                self._queue_redraw_curve()

            self._prev_scroll_time = event.time
        else:
            return super(InkingMode, self).scroll_cb(tdw, event)



    ## Interrogating events

    def _get_event_data(self, tdw, event):
        x, y = tdw.display_to_model(event.x, event.y)
        xtilt, ytilt = self._get_event_tilt(tdw, event)
        return _Node(
            x=x, y=y,
            pressure=self._get_event_pressure(event),
            xtilt=xtilt, ytilt=ytilt,
            time=(event.time / 1000.0),
        )

    def _get_event_pressure(self, event):
        # FIXME: CODE DUPLICATION: copied from freehand.py
        pressure = event.get_axis(Gdk.AxisUse.PRESSURE)
        if pressure is not None:
            if not np.isfinite(pressure):
                pressure = None
            else:
                pressure = lib.helpers.clamp(pressure, 0.0, 1.0)

        if pressure is None:
            pressure = 0.0
            if event.state & Gdk.ModifierType.BUTTON1_MASK:
                pressure = 0.5

        # Workaround for buggy evdev behaviour.
        # Events sometimes get a zero raw pressure reading when the
        # pressure reading has not changed. This results in broken
        # lines. As a workaround, forbid zero pressures if there is a
        # button pressed down, and substitute the last-known good value.
        # Detail: https://github.com/mypaint/mypaint/issues/223
        if self._button_down is not None:
            if pressure == 0.0:
                pressure = self._last_good_raw_pressure
            elif pressure is not None and np.isfinite(pressure):
                self._last_good_raw_pressure = pressure
        return pressure

    def _get_event_tilt(self, tdw, event):
        # FIXME: CODE DUPLICATION: copied from freehand.py
        xtilt = event.get_axis(Gdk.AxisUse.XTILT)
        ytilt = event.get_axis(Gdk.AxisUse.YTILT)
        if xtilt is None or ytilt is None or not np.isfinite(xtilt + ytilt):
            return (0.0, 0.0)

        # Switching from a non-tilt device to a device which reports
        # tilt can cause GDK to return out-of-range tilt values, on X11.
        xtilt = lib.helpers.clamp(xtilt, -1.0, 1.0)
        ytilt = lib.helpers.clamp(ytilt, -1.0, 1.0)

        # Evdev workaround. X and Y tilts suffer from the same
        # problem as pressure for fancier devices.
        if self._button_down is not None:
            if xtilt == 0.0:
                xtilt = self._last_good_raw_xtilt
            else:
                self._last_good_raw_xtilt = xtilt
            if ytilt == 0.0:
                ytilt = self._last_good_raw_ytilt
            else:
                self._last_good_raw_ytilt = ytilt

        # Tilt inputs are assumed to be relative to the viewport,
        # but the canvas may be rotated or mirrored, or both.
        # Compensate before passing them to the brush engine.
        # https://gna.org/bugs/?19988
        if tdw.mirrored:
            xtilt *= -1.0
        if tdw.rotation != 0:
            tilt_angle = math.atan2(ytilt, xtilt) - tdw.rotation
            tilt_magnitude = math.sqrt((xtilt**2) + (ytilt**2))
            xtilt = tilt_magnitude * math.cos(tilt_angle)
            ytilt = tilt_magnitude * math.sin(tilt_angle)

        return (xtilt, ytilt)

    ## Node editing

    @property
    def options_presenter(self):
        """MVP presenter object for the node editor panel"""
        cls = self.__class__
        if cls._OPTIONS_PRESENTER is None:
            cls._OPTIONS_PRESENTER = OptionsPresenter()
        return cls._OPTIONS_PRESENTER

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        return self.options_presenter.widget

    def update_node(self, i, **kwargs):
        """Updates properties of a node, and redraws it"""
        changing_pos = bool({"x", "y"}.intersection(kwargs))
        oldnode = self.nodes[i]
        if changing_pos:
            self._queue_draw_node(i)
        self.nodes[i] = oldnode._replace(**kwargs)
        # FIXME: The curve redraw is a bit flickery.
        #   Perhaps dragging to adjust should only draw an
        #   armature during the drag, leaving the redraw to
        #   the stop handler.
        self._queue_redraw_curve()
        if changing_pos:
            self._queue_draw_node(i)

    def get_node_dtime(self, i):
        if not (0 < i < len(self.nodes)):
            return 0.0
        n0 = self.nodes[i-1]
        n1 = self.nodes[i]
        dtime = n1.time - n0.time
        dtime = max(dtime, self.CAPTURE_SETTING.min_internode_time)
        return dtime

    def set_node_dtime(self, i, dtime):
        dtime = max(dtime, self.CAPTURE_SETTING.min_internode_time)
        nodes = self.nodes
        if not (0 < i < len(nodes)):
            return
        old_dtime = nodes[i].time - nodes[i-1].time
        for j in range(i, len(nodes)):
            n = nodes[j]
            new_time = n.time + dtime - old_dtime
            self.update_node(j, time=new_time)

    def can_delete_node(self, i):
        return 0 < i < len(self.nodes)-1

    def _adjust_current_node_index(self):
        """ Adjust self.current_node_index
        child classes might have different behavior
        from Inktool about current_node_index.
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
        self._queue_draw_node(i)

        self._pop_node(i)

        # Limit the current node.
        # this processing may vary in inherited classes,
        # so wrap this.
       #self._adjust_current_node_index()

        self.options_presenter.target = (self, self.current_node_index)
        # Issue redraws for the changed on-canvas elements
        self._queue_redraw_curve()
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()


    def _adjust_pressure_with_motion(self, dx, dy):
        """Adjust pressure of current selected node,
        and it may affects nearby nodes

        :param dx/dy: currently dragging position,in display coord.
        """
        direction,length = gui.ui_utils.get_drag_direction(
                0, 0, dx, dy)

        if direction >= 0:
            if direction in (0 , 1):
                diff = length / 128.0
            else:
                diff = length / 256.0

            if direction in (0 , 3):
                diff *= -1

            # XXX divide dragged length with 64pixel(large change)/128pixel
            # to convert dragged length into pressure differencial.
            # This valus is not theorical.so this should be configured
            # by user...?
            # Also it might be change at HiDPI system
                
           #if (self.target_node_index != None and 
           #        not self.target_node_index in self.selected_nodes):
           #    cn = self.nodes[self.target_node_index]
           #    self.nodes[self.target_node_index] = \
           #            cn._replace(pressure = lib.helpers.clamp(
           #                cn.pressure + diff,0.0, 1.0))
           #else:
           #    for idx in self.selected_nodes:
           #        cn = self.nodes[idx]
           #        self.nodes[idx] = \
           #                cn._replace(pressure = lib.helpers.clamp(
           #                    cn.pressure + diff,0.0, 1.0))

            if (self.target_node_index != None):
                cn = self.nodes[self.target_node_index]
                self.nodes[self.target_node_index] = \
                        cn._replace(pressure = lib.helpers.clamp(
                            cn.pressure + diff,0.0, 1.0))

        self._queue_redraw_curve()

    def delete_selected_nodes(self):

        self._queue_draw_buttons()
        for idx in self.selected_nodes:
            self._queue_draw_node(idx)

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
        self._reset_selected_nodes()
        self.target_node_index = None

        # Issue redraws for the changed on-canvas elements
        self._queue_redraw_curve()
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()

    def can_insert_node(self, i):
        return 0 <= i < len(self.nodes)-1

    def insert_node(self, i):
        """Insert a node, and issue redraws & updates"""
        assert self.can_insert_node(i), "Can't insert back of the endpoint"
        # Redraw old locations of things while the node still exists
        self._queue_draw_buttons()
        self._queue_draw_node(i)
        # Create the new node
        cn = self.nodes[i]
        nn = self.nodes[i+1]

        newnode = _Node(
            x=(cn.x + nn.x)/2.0, y=(cn.y + nn.y) / 2.0,
            pressure=(cn.pressure + nn.pressure) / 2.0,
            xtilt=(cn.xtilt + nn.xtilt) / 2.0,
            ytilt=(cn.ytilt + nn.ytilt) / 2.0,
            time=(cn.time + nn.time) / 2.0
        )
        self.nodes.insert(i+1,newnode)

        # Issue redraws for the changed on-canvas elements
        self._queue_redraw_curve()
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
        for i in xrange(len(self.nodes)/2):
            self._pop_node(idx)
            idx+=1
        return curcnt-len(self.nodes)


    def _queue_all_visual_redraw(self):
        """Redraw all overlay objects"""
        self._queue_redraw_curve()
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

            new_nodes = [self.nodes[0]]
            idx = 1
            pn = self.nodes[0]
            cn = self.nodes[idx]
            while idx < len(self.nodes) - 1:
                nn = self.nodes[idx+1]
                # Limit affected nodes with selection list.
                # if only one node is selected,
                # entire nodes are averaged.
                if (len(self.selected_nodes) == 0 or
                        idx in self.selected_nodes):
                    try:
                        # avx, avy is identity vector of current-prev node
                        # bvx, bvy is identity vector of next-prev node
                        avx, avy = normal(pn.x, pn.y, cn.x, cn.y)
                        bvx, bvy = normal(pn.x, pn.y, nn.x, nn.y)
                        avx=(avx + bvx) / 2.0
                        avy=(avy + bvy) / 2.0
                        s = math.hypot(cn.x - pn.x, cn.y - pn.y)
                        avx*=s
                        avy*=s
                        new_nodes.append(cn._replace(x=avx+pn.x, y=avy+pn.y))
                    except ZeroDivisionError:
                        # This means 'two nodes at same place'.
                        # abort averaging for this node.
                        new_nodes.append(cn)
                else:
                    new_nodes.append(cn)

                pn = cn
                cn = nn
                idx += 1

            new_nodes.append(self.nodes[-1])
            self.nodes = new_nodes
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
            self._queue_redraw_curve()



    ## Node selection
    def select_all(self):
        self.selected_nodes = range(0, len(self.nodes))
        self._queue_redraw_all_nodes()

    def deselect_all(self):
        self._reset_selected_nodes(None)
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
       
    def apply_pressure_from_curve_widget(self):
        """ apply pressure reprenting points
        from StrokeCurveWidget.
        Mostly resembles as BezierMode.apply_pressure_points,
        but inktool stroke calculartion is not same as
        BezierMode.
        """

        # We need smooooth value, so treat the points
        # as Bezier-curve points.

        # first of all, get the entire stroke length
        # to normalize stroke.

        if len(self.nodes) < 2:
            return

        assert hasattr(self.options_presenter,'curve')
        curve = self.options_presenter.curve

        self._queue_redraw_curve()

        # Getting entire stroke(vector) length
        node_length=[]
        total_length = 0.0

        for idx, cn in enumerate(self.nodes[:-1]):
            nn = self.nodes[idx + 1]
            length = math.sqrt((cn.x - nn.x) ** 2 + (cn.y - nn.y) ** 2)
            node_length.append(length)
            total_length+=length

        node_length.append(total_length) # this is sentinel



        # use control handle class temporary to get smooth pressures.
        cur_length = 0.0
        new_nodes=[]
        for idx,cn in enumerate(self.nodes):
            val = curve.get_pressure_value(cur_length / total_length)
            new_nodes.append(cn._replace(pressure=val)) 
            cur_length += node_length[idx]

        self.nodes = new_nodes

        self._queue_redraw_curve()

    ## Nodes hide
    @property
    def hide_nodes(self):
        return self._hide_nodes

    @hide_nodes.setter
    def hide_nodes(self, flag):
        self._hide_nodes = flag
        self._queue_redraw_all_nodes()

    def enter_pressure_phase(self):
        if self.phase == _Phase.ADJUST:
            self.phase = _Phase.ADJUST_PRESSURE
            self._queue_redraw_all_nodes()
        elif self.phase == _Phase.ADJUST_PRESSURE:
            self.phase = _Phase.ADJUST
            self._queue_redraw_all_nodes()

    ## Generic Oncanvas-editing handler
    def delete_item(self):
        self.delete_selected_nodes()

    def accept_edit(self):
        if (self.phase in (_Phase.ADJUST ,_Phase.ADJUST_PRESSURE) and
                len(self.nodes) > 1):
            self._start_new_capture_phase(rollback=True)
            assert self.phase == _Phase.CAPTURE

    def discard_edit(self):
        if (self.phase in (_Phase.ADJUST ,_Phase.ADJUST_PRESSURE)):
            self._start_new_capture_phase(rollback=False)
            assert self.phase == _Phase.CAPTURE
            
    ## Editing range related
    def set_range_radius(self, radius):
        self._range_radius = radius
    
    def set_range_factor(self, factor):
        self._range_factor_source = factor
        if factor < 0.0:
            self._range_factor = (1.0 / math.gamma(factor+1.000001))
        else:
            self._range_factor = math.gamma(factor+0.000001)
        
    @property
    def range_radius(self):
        if self._range_radius == None:
            self._range_radius = self.doc.app.preferences.get(
                                    "inktool.adjust_range_radius", 0)
        return self._range_radius
    
    @property
    def range_factor(self):
        if self._range_factor == None:
            self.set_range_factor(self.doc.app.preferences.get(
                                    "inktool.adjust_range_factor", 0))
        return self._range_factor

    def enum_nodes_coord(self, tdw, convert_to_display=True):
        """ Enumerate nodes screen coordinate with offsets.
        """
        if self.current_node_index != None:
            basept = self.nodes[self.current_node_index]
            model_radius, junk = tdw.display_to_model(self.range_radius, 0)
            model_radius = abs(model_radius)
        else:
            basept = None
            model_radius = 0

        offset_vec = self._generate_offset_vector()

        for i in xrange(len(self.nodes)):
            node = self.nodes[i]

            offsets = gui.drawutils.calc_ranged_offset(basept, node,
                    model_radius, self.range_factor,
                    offset_vec)

            if convert_to_display:
                x, y = tdw.model_to_display(*offsets)
            else:
                x, y = offsets

            yield (i, node, x, y)


class Overlay (gui.overlays.Overlay):
    """Overlay for an InkingMode's adjustable points"""

    def __init__(self, inkmode, tdw):
        super(Overlay, self).__init__()
        self._inkmode = weakref.proxy(inkmode)
        self._tdw = weakref.proxy(tdw)
        self._button_pixbuf_cache = {}
        self.accept_button_pos = None
        self.reject_button_pos = None

    def update_button_positions(self):
        """Recalculates the positions of the mode's buttons."""
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

    def _get_button_pixbuf(self, name):
        """Loads the pixbuf corresponding to a button name (cached)"""
        cache = self._button_pixbuf_cache
        pixbuf = cache.get(name)
        if not pixbuf:
            pixbuf = gui.drawutils.load_symbolic_icon(
                icon_name=name,
                size=gui.style.FLOATING_BUTTON_ICON_SIZE,
                fg=(0, 0, 0, 1),
            )
            cache[name] = pixbuf
        return pixbuf

    def _get_onscreen_nodes(self):
        """Iterates across only the on-screen nodes."""
        mode = self._inkmode
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        alloc = self._tdw.get_allocation()
       #for i, node in enumerate(mode.nodes):
       #    x, y = self._tdw.model_to_display(node.x, node.y)
       #    node_on_screen = (
       #        x > alloc.x - radius*2 and
       #        y > alloc.y - radius*2 and
       #        x < alloc.x + alloc.width + radius*2 and
       #        y < alloc.y + alloc.height + radius*2
       #    )
       #    if node_on_screen:
       #        yield (i, node, x, y)
        for i, node, x, y in mode.enum_nodes_coord(self._tdw):
           #x, y = self._tdw.model_to_display(node.x, node.y)
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
        mode = self._inkmode
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        alloc = self._tdw.get_allocation()
        dx,dy = mode.drag_offset.get_display_offset(self._tdw)
        fill_flag = not mode.phase in (_Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT)
        for i, node, x, y in self._get_onscreen_nodes():
            color = gui.style.EDITABLE_ITEM_COLOR
            show_node = not mode.hide_nodes
            if (mode.phase in
                    (_Phase.ADJUST,
                     _Phase.ADJUST_PRESSURE,
                     _Phase.ADJUST_PRESSURE_ONESHOT
                     )):
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

               #if (color != gui.style.EDITABLE_ITEM_COLOR and
               #        mode.phase == _Phase.ADJUST):
               #    x += dx
               #    y += dy

            if show_node:
                gui.drawutils.render_round_floating_color_chip(
                    cr=cr, x=x, y=y,
                    color=color,
                    radius=radius,
                    fill=fill_flag)

        # Buttons
        if (mode.phase in
                (_Phase.ADJUST,
                 _Phase.ADJUST_PRESSURE,
                 _Phase.ADJUST_PRESSURE_ONESHOT) and
                not mode.in_drag):
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
            if hasattr(current_mode, 'redraw_curve_cb'):
                # Redraw last_line when settings are adjusted in the adjustment Curve
                GLib.idle_add(current_mode.redraw_curve_cb)
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

class OptionsPresenter (object):
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
        super(OptionsPresenter, self).__init__()
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

        OptionsPresenter.init_variation_preset_store()
        

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
        #self._period_adj = builder.get_object("period_adj")
        #self._period_scale = builder.get_object("period_scale")
        #self._period_adj.set_value(self._app.preferences.get(
            #"inktool.capture_period_factor", 1))
        self._range_radius_adj = builder.get_object("range_radius_adj")
        self._range_radius_scale = builder.get_object("range_radius_scale")
        self._range_radius_adj.set_value(self._app.preferences.get(
            "inktool.adjust_range_radius", 0))
            
        self._range_factor_adj = builder.get_object("range_factor_adj")
        self._range_factor_scale = builder.get_object("range_factor_scale")
        self._range_factor_adj.set_value(self._app.preferences.get(
            "inktool.adjust_range_factor", 0))
            
        self._hide_nodes_check = builder.get_object("hide_nodes_checkbutton")

        apply_btn = builder.get_object("apply_variation_button")
        apply_btn.set_sensitive(False)
        self._apply_variation_button = apply_btn

        base_grid = builder.get_object("points_editing_grid")
        toolbar = gui.widgets.inline_toolbar(
            self._app,
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

       #hide_nodes_act = self._app.find_action('HideNodes')
       #if hide_nodes_act:
       #    self._app.ui_manager.do_connect_proxy(
       #            hide_nodes_act,
       #            self._hide_nodes_check)
       #            
       #   #hide_nodes_act.connect_proxy(self._hide_nodes_check)
       #else:
       #    print('no such hide!')



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
                OptionsPresenter.variation_preset_store)
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
            self._range_radius_adj.set_value(self._app.preferences.get(
                "inktool.adjust_range_radius", 0))
            self._range_factor_adj.set_value(self._app.preferences.get(
                "inktool.adjust_range_factor", 0))            
            self._apply_variation_button.set_sensitive(len(inkmode.nodes) > 2)
        finally:
            self._updating_ui = False

    def _variation_preset_combo_changed_cb(self, widget):
        iter = self._variation_preset_combo.get_active_iter()
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

    def _range_radius_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        self._app.preferences['inktool.adjust_range_radius'] = adj.get_value()
        inkmode, node_idx = self.target
        if inkmode:
            inkmode.set_range_radius(adj.get_value())
        
    def _range_factor_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        self._app.preferences['inktool.adjust_range_factor'] = adj.get_value()
        inkmode, node_idx = self.target
        if inkmode:
            inkmode.set_range_factor(adj.get_value())

    def _range_radius_scale_format_value_cb(self, scale, value):
        return "%dpx" % value

    def _range_factor_scale_format_value_cb(self, scale, value):
        return "%.1fx" % value
        
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

