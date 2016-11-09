#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2016 by Dothiko <a.t.dothiko@gmail.com>
# Most part of this file is transplanted from
# original gui/inktool.py, re-organized it as Mixin.
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


## Function defs

#def _nodes_deletion_decorator(method):
#    """ Decorator for deleting multiple nodes methods
#    """
#    def _decorator(self, *args):
#        # To ensure redraw entire overlay,avoiding glitches.
#        self._queue_redraw_curve()
#        self._queue_redraw_all_nodes()
#        self._queue_draw_buttons()
#
#        # the method should return deleted nodes count
#        result = method(self, *args)
#        assert type(result) == int
#
#        if result > 0:
#            self.options_presenter.target = (self, self.current_node_index)
#            self._queue_redraw_curve()
#            self._queue_redraw_all_nodes()
#            self._queue_draw_buttons()
#        return result
#    return _decorator


## Class defs

class Phase_Mixin:
    """Enumeration of the states that an InkingMode can be in"""
    CAPTURE = 0
    ADJUST = 1

#
##_NODE_FIELDS = ("x", "y", "pressure", "xtilt", "ytilt", "time")
##
##
##class _Node (collections.namedtuple("_Node", _NODE_FIELDS)):
##    """Recorded control point, as a namedtuple.
##
##    Node tuples have the following 6 fields, in order
##
##    * x, y: model coords, float
##    * pressure: float in [0.0, 1.0]
##    * xtilt, ytilt: float in [-1.0, 1.0]
##    * time: absolute seconds, float
##    """
#
#
class EditZone_Mixin:
    """Enumeration of what the pointer is on in the ADJUST phase"""

    EMPTY_CANVAS = 0  #: Nothing, empty space
    CONTROL_NODE = 1  #: Any control node; see target_node_index
    ACCEPT_BUTTON = 2  #: On-canvas action button. Also used when drawing overlay.
    REJECT_BUTTON = 3  #: On-canvas action button. Also used when drawing overlay.


class OncanvasEditMixin(gui.mode.ScrollableModeMixin,
                        gui.mode.BrushworkModeMixin,
                        gui.mode.DragMode):
    """ Mixin for modes which have on-canvas node editing ability.

    Actually, to create new oncanvas editable mode,
    you will need other useful mixins such as

    gui.mode.ScrollableModeMixin,
    gui.mode.BrushworkModeMixin,
    and gui.mode.DragMode.

    This mixin adds the ability to edit on-canvas nodes.
    The `accept()` finalize the node editing result
    And `cancel()` discards current editing.
    """



    ## Metadata properties
    pointer_behavior = gui.mode.Behavior.PAINT_FREEHAND
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW



    ## Metadata methods

    @property
    def inactive_cursor(self):
        return None
 
#   @property
#   def active_cursor(self):
#       if self.phase == _Phase.ADJUST:
#           if self.zone == _EditZone.CONTROL_NODE:
#               return self._crosshair_cursor
#           elif self.zone != _EditZone.EMPTY_CANVAS: # assume button
#               return self._arrow_cursor
#
#       elif self.phase in (_Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT):
#           if self.zone == _EditZone.CONTROL_NODE:
#               return self._cursor_move_nw_se
#
#       return None
 
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
    MAX_INTERNODE_DISTANCE_MIDDLE = 30   # display pixels
    MAX_INTERNODE_DISTANCE_ENDS = 10   # display pixels
    MAX_INTERNODE_TIME = 1/100.0   # seconds

    # Node value adjustment settings
    MIN_INTERNODE_TIME = 1/200.0   # seconds (used to manage adjusting)


    ## Class attributes
 
    _OPTIONS_PRESENTER = None   #: Options presenter singleton
 
    drag_offset = gui.ui_utils.DragOffset() #: Dragging management class singleton

 
    ## Initialization & lifecycle methods
 
    def __init__(self, **kwargs):

        super(OncanvasEditMixin, self).__init__(**kwargs)
 
        #+ initialize selected nodes - 
        #+ place this prior to _reset_nodes()
        self.selected_nodes=[]
 
        self.phase = Phase_Mixin.CAPTURE
        self.zone = EditZone_Mixin.EMPTY_CANVAS
        self.current_node_index = None  #: Node active in the options ui
        self.target_node_index = None  #: Node that's prelit
        self._overlays = {}  # keyed by tdw
        self._reset_nodes()
        self._reset_adjust_data()
        self._task_queue = collections.deque()  # (cb, args, kwargs)
        self._task_queue_runner_id = None
        self._clicked_action_button = None  # id of self._BUTTONS
        self._current_override_cursor = None
        # Button pressed while drawing
        # Not every device sends button presses, but evdev ones
        # do, and this is used as a workaround for an evdev bug:
        # https://github.com/mypaint/mypaint/issues/223
        self._button_down = None
 
        #+ Hiding nodes functionality
        self._hide_nodes = False
 
        #+ Previous scroll event time.
        #  in some environment, Gdk.ScrollDirection.UP/DOWN/LEFT/RIGHT
        #  and Gdk.ScrollDirection.SMOOTH might happen at same time.
        #  to reject such event, this attribute needed.
        self._prev_scroll_time = None
 
        #+ Sub phase, use in deriving class.
        self.subphase = None

        #+ returning phase.for special phase changing case.
        self._returning_phase = None
 
 
    def _reset_nodes(self):
        self.nodes = []  # nodes that met the distance+time criteria
        self._reset_selected_nodes(None)
 
 
    def _reset_adjust_data(self):
        self.zone = EditZone_Mixin.EMPTY_CANVAS
        self.current_node_index = None
        self.target_node_index = None
        self._dragged_node_start_pos = None
        self.drag_offset.reset()
 
        # Multiple selected nodes.
        # This is a index list of node from self.nodes
        self._reset_selected_nodes()
 
        self.hide_nodes = False

        self._click_info = None
 
 
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
 
    def _is_active(self):
        """ To know whether this mode is active or not. 
        """
        for mode in self.doc.modes:
            if mode is self:
                return True
        return False
 
    ## Overlay related

    def _ensure_overlay_for_tdw(self, tdw):
        overlay = self._overlays.get(tdw)
        if not overlay:
            overlay = self._generate_overlay(tdw)
            tdw.display_overlays.append(overlay)
            self._overlays[tdw] = overlay
        return overlay

    def _discard_overlays(self):
        for tdw, overlay in self._overlays.items():
            tdw.display_overlays.remove(overlay)
            tdw.queue_draw()
        self._overlays.clear()

    def _generate_overlay(self, tdw):
        """ generate mode own overlay.
        This is placeholder, should be implemented 
        in child class.
        """
        pass
 

    ##  Option presenter related
    @property
    def options_presenter(self):
        """MVP presenter object for the node editor panel"""
        cls = self.__class__
        if cls._OPTIONS_PRESENTER is None:
            cls._OPTIONS_PRESENTER = self._generate_presenter()
        return cls._OPTIONS_PRESENTER

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        return self.options_presenter.widget

    ## Internal states

    def _start_new_capture_phase(self, rollback=False):
        """Let the user capture a new stroke"""
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
        self._reset_adjust_data()
        self.phase = Phase_Mixin.CAPTURE
        self.enable_switch_actions(True)

    def enter(self, doc, **kwds):
        """Enters the mode: called by `ModeStack.push()` etc."""
        super(OncanvasEditMixin, self).enter(doc, **kwds)
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
        self.enable_switch_actions(True)

    def leave(self, **kwds):
        """Leaves the mode: called by `ModeStack.pop()` etc."""
        if not self._is_active():
            self._discard_overlays()
        self._stop_task_queue_runner(complete=True)
        self.enable_switch_actions(False)
        super(OncanvasEditMixin, self).leave(**kwds)  # supercall will commit

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
            super(OncanvasEditMixin, self).checkpoint(flush=flush, **kwargs)
        else:
            # Queue a re-rendering with any new brush data
            # No supercall
            self._stop_task_queue_runner(complete=False)
            self._queue_draw_buttons()
            self._queue_redraw_all_nodes()
            self._queue_redraw_curve()
 
 
 
#   def _update_zone_and_target(self, tdw, x, y):
#       """Update the zone and target node under a cursor position"""
#
#       self._ensure_overlay_for_tdw(tdw)
#       new_zone = _EditZone.EMPTY_CANVAS
#
#       if not self.in_drag:
#           if self.phase == _Phase.ADJUST):
#
#               new_target_node_index = None
#               self._action_button = None
#               # Test buttons for hits
#               overlay = self._ensure_overlay_for_tdw(tdw)
#               hit_dist = gui.style.FLOATING_BUTTON_RADIUS
#              #button_info = [
#              #    (_EditZone.ACCEPT_BUTTON, overlay.accept_button_pos),
#              #    (_EditZone.REJECT_BUTTON, overlay.reject_button_pos),
#              #]
#               for btn_id in self._BUTTONS.keys():
#                   btn_pos = overlay.get_button_pos(btn_id)
#                   if btn_pos is None:
#                       continue
#                   btn_x, btn_y = btn_pos
#                   d = math.hypot(btn_x - x, btn_y - y)
#                   if d <= hit_dist:
#                       new_target_node_index = None
#                       new_zone = _EditZone.ACTION_BUTTON
#                       self._action_button = btn_id
#                       break
#
#               # Test nodes for a hit, in reverse draw order
#               if new_zone == _EditZone.EMPTY_CANVAS:
#                   new_target_node_index = self._search_target_node(tdw, x, y)
#                   if new_target_node_index != None:
#                       new_zone = _EditZone.CONTROL_NODE
#
#               # Update the prelit node, and draw changes to it
#               if new_target_node_index != self.target_node_index:
#                   # Redrawing old target node.
#                   if self.target_node_index is not None:
#                       self._queue_draw_node(self.target_node_index)
#                       self.node_leave_cb(self.nodes[self.target_node_index]) 
#
#                   self.target_node_index = new_target_node_index
#                   if self.target_node_index is not None:
#                       self._queue_draw_node(self.target_node_index)
#                       self.node_enter_cb(self.nodes[self.target_node_index]) 
#
#
#       # Update the zone, and assume any change implies a button state
#       # change as well (for now...)
#       if self.zone != new_zone:
#           self.zone = new_zone
#           self._ensure_overlay_for_tdw(tdw)
#           self._queue_draw_buttons()
#
#       # Update the "real" inactive cursor too:
#       #f not self.in_drag:
#       #   cursor = None
#       #   if self.phase in (_Phase.ADJUST, _Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT):
#       #       if self.zone == _EditZone.CONTROL_NODE:
#       #           cursor = self._crosshair_cursor
#       #       elif self.zone != _EditZone.EMPTY_CANVAS: # assume button
#       #           cursor = self._arrow_cursor
#       #   if cursor is not self._current_override_cursor:
#       #       tdw.set_override_cursor(cursor)
#       #       self._current_override_cursor = cursor
#
#
#   ## Redraws

    def _queue_draw_buttons(self):
        """Redraws the accept/reject buttons on all known view TDWs"""
        for tdw, overlay in self._overlays.items():
            overlay.update_button_positions()
            positions = (
                overlay.get_button_pos(EditZone_Mixin.REJECT_BUTTON),
                overlay.get_button_pos(EditZone_Mixin.ACCEPT_BUTTON),
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

    def _queue_draw_node(self, idx):
        """ queue an oncanvas item to draw."""

    def _queue_draw_selected_nodes(self):
        for i in self.selected_nodes:
            self._queue_draw_node(i)

    def _queue_redraw_all_nodes(self):
        """Redraws all nodes on all known view TDWs"""
        for i in xrange(len(self.nodes)):
            self._queue_draw_node(i)

 
#   def _queue_redraw_display(self):
#       """ queue redraw entire(not only overlay) displaying items.
#       """
#       self._queue_redraw_contents()
#       self._queue_redraw_all_nodes()
#       self._queue_draw_buttons()
#
#   def _queue_redraw_contents(self):
#       """ Redraws class specific drawing contents
#       like a stroke in Inktool.
#       """
#       pass
#

    ## Redrawing task management

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
 
#   ## Editing result commit/cancel callbacks
#
#   def finalize_editing_cb(self, commited):
#       """ Finalize(commit) editing or discard it.
#       This callback is called from `self._start_new_capture_phase()`
#
#       Clearing node, or resetting overlay, should be done automatically
#       from mixin method, so no need to do them.
#
#       :param committed: The editting nodes should be commited(=True) 
#                           or cancelled(=False).
#       """
#       pass
#
    ## Raw event handling (prelight & zone selection in adjust phase)

    def motion_notify_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False
 
        self._update_zone_and_target(tdw, event.x, event.y)
        return super(OncanvasEditMixin, self).motion_notify_cb(tdw, event)

#   def button_press_cb(self, tdw, event):
#       self._ensure_overlay_for_tdw(tdw)
#       current_layer = tdw.doc._layers.current
#       if not (tdw.is_sensitive and current_layer.get_paintable()):
#           return False
#       self._update_zone_and_target(tdw, event.x, event.y)
#       self._update_current_node_index()
#       if self.phase == _Phase.ADJUST:
#           button = event.button
#           if button == 1:
#               if self.zone == _EditZone.ACTION_BUTTON:
#                   assert self._action_button != None
#                   self._clicked_action_button = self._action_button
#                   return False
#                   # FALLTHRU: *do* allow drags to start with other buttons
#               elif self.zone == _EditZone.EMPTY_CANVAS:
#                   self._start_new_capture_phase(rollback=False)
#                   assert self.phase == _Phase.CAPTURE
#                   # FALLTHRU: *do* start a drag
#               else:
#                   # clicked a node.
#                   # FALLTHRU: *do* start a drag
#
#
#       elif self.phase == _Phase.CAPTURE:
#           # XXX Not sure what to do here.
#           # XXX Click to append nodes?
#           # XXX  but how to stop that and enter the adjust phase?
#           # XXX Click to add a 1st & 2nd (=last) node only?
#           # XXX  but needs to allow a drag after the 1st one's placed.
#           pass
#       else:
#           raise NotImplementedError("Unrecognized zone %r", self.zone)
#
#       # Update workaround state for evdev dropouts
#       self._button_down = event.button
#
#       # Supercall: start drags etc
#       return super(OncanvasEditMixin, self).button_press_cb(tdw, event)
#
#   def button_release_cb(self, tdw, event):
#       self._ensure_overlay_for_tdw(tdw)
#       current_layer = tdw.doc._layers.current
#       if not (tdw.is_sensitive and current_layer.get_paintable()):
#           return False
#
#       if self.phase == _Phase.ADJUST:
#           assert self._button_down >= 1
#           if self._clicked_action_button != None:
#               handler = self._BUTTONS[self._clicked_action_button]
#               rollback = False
#               if handler:
#                   # TODO We should think the third type button
#                   # which is not both of accept/cancel,
#                   # does noting to with current editing.
#                   self._start_new_capture_phase(handler(self))
#               self._clicked_action_button = None
#               self._update_zone_and_target(tdw, event.x, event.y)
#               self._update_current_node_index()
#               return False  # NO `drag_stop_cb` activated.
#           else:
#               # Clicked node and button released.
#               # Add or Remove selected node
#               # when control key is pressed
#               if event.button == 1:
#                   if event.state & Gdk.ModifierType.CONTROL_MASK:
#                       tidx = self.target_node_index
#                       if tidx != None:
#                           if not tidx in self.selected_nodes:
#                               self.selected_nodes.append(tidx)
#                           else:
#                               self.selected_nodes.remove(tidx)
#                               self.target_node_index = None
#                               self.current_node_index = None
#                   else:
#                       # Single node click.
#                       pass
#
#                   ## fall throgh
#
#               self._update_zone_and_target(tdw, event.x, event.y)
#
#       elif self.phase == _Phase.CAPTURE:
#           # Update options_presenter when capture phase end
#           self.options_presenter.target = (self, None)
#
#       # Update workaround state for evdev dropouts
#       self._button_down = None
#
#       # Supercall: stop current drag
#       return super(OncanvasEditMixin, self).button_release_cb(tdw, event)
#
#   ## Drag handling (both capture and adjust phases)
#   # 
#   #  We did not need to implement them in mixin-user-class.
#   #  Instead of it, implement `node_capture_*_cb()`
#   #  and `node_adjust_*_cb()`
#
#   def drag_start_cb(self, tdw, event):
#       self._ensure_overlay_for_tdw(tdw)
#       if self.phase == _Phase.CAPTURE:
#           call_super = self.node_capture_start_cb(tdw, event)
#       elif self.phase == _Phase.ADJUST:
#           call_super = self.node_adjust_start_cb(tdw, event)
#       else:
#           call_super = True
#
#       if call_super:
#           return super(OncanvasEditMixin, self).drag_start_cb(tdw, event)
#
#   def drag_update_cb(self, tdw, event, dx, dy):
#       self._ensure_overlay_for_tdw(tdw)
#       if self.phase == _Phase.CAPTURE:
#           call_super = self.node_capture_update_cb(tdw, event, dx, dy)
#       elif self.phase == _Phase.ADJUST:
#           call_super = self.node_adjust_update_cb(tdw, event, dx, dy)
#       else:
#           call_super = True
#
#       if call_super:
#           return super(OncanvasEditMixin, self).drag_update_cb(tdw, event,
#                   dx, dy)
#
#   def drag_stop_cb(self, tdw):
#       self._ensure_overlay_for_tdw(tdw)
#       if self.phase == _Phase.CAPTURE:
#
#           if not self.nodes or self._last_event_node == None:
#               return super(OncanvasEditMixin, self).drag_stop_cb(tdw)
#
#           self.node_capture_stop_cb(tdw, node)
#          #node = self._last_event_node
#          #if self.nodes[-1] is not node:
#          #    self.node_capture_stop_cb(tdw, node)
#          #   #self.nodes.append(node)
#
#           self._reset_capture_data()
#           self._reset_adjust_data()
#           if len(self.nodes) > 1:
#               self.phase = _Phase.ADJUST
#               OncanvasEditMixin.enable_switch_actions(False)
#               self._queue_redraw_display()
#           else:
#               self._reset_nodes()
#               tdw.queue_draw()
#
#       elif self.phase == _Phase.ADJUST:
#
#           # Finalize dragging motion to selected nodes.
#           if self._node_dragged:
#
#               self._queue_draw_selected_nodes() # to ensure erase them
#
#               dx, dy = self.drag_offset.get_model_offset()
#               self.node_adjust_stop_cb(tdw, dx, dy)
#              #for idx in self.selected_nodes:
#              #    cn = self.nodes[idx]
#              #    self.nodes[idx] = cn._replace(x=cn.x + dx,
#              #            y=cn.y + dy)
#
#               self.drag_offset.reset()
#
#           self._dragged_node_start_pos = None
#           self._queue_redraw_curve()
#           self._queue_draw_buttons()
#
#
#   def scroll_cb(self, tdw, event):
#       if (self.phase == _Phase.ADJUST 
#               and self.target_node_index != None):
#
#           redraw = False
#
#           if self._prev_scroll_time != event.time:
#               if len(self.selected_nodes) == 0:
#                   targets = (self.target_node_index,)
#               else:
#                   targets = self.selected_nodes
#
#               for idx in targets:
#                   node = self.nodes[idx]
#                   new_pressure = node.pressure
#
#                   x, y = gui.ui_utils.get_scroll_delta(event, self._PRESSURE_WHEEL_STEP)
#                   redraw |= self.node_scroll_cb(tdw, node, x, y)
#                   new_pressure += y
#
#                   if new_pressure != node.pressure:
#                       self.nodes[idx]=node._replace(pressure=new_pressure)
#
#                   if idx == self.target_node_index:
#                       self.options_presenter.target = (self, self.target_node_index)
#
#               if redraw:
#                   self._queue_redraw_contents()
#
#           self._prev_scroll_time = event.time
#       else:
#           return super(OncanvasEditMixin, self).scroll_cb(tdw, event)
#
#   ## Node Editing event callbacks
#
#   def node_enter_cb(self, tdw, node):
#       """ The cursor hovers over a node
#       You can change cursor if needed.
#       """
#       pass
#
#   def node_leave_cb(self, tdw, node):
#       """ The cursor hovers outside a node.
#       You MUST reset cursor if change it.
#       """
#       pass
#
#   def node_adjust_start_cb(self, tdw, event):
#       """ User starts dragging the selected node(s) right now.
#       """
#       pass
#
#   def node_adjust_update_cb(self, tdw, dx, dy):
#       """ User dragging the selected node(s) now.
#
#       :param dx: the draging delta of x , in MODEL coordinate.
#       :param dy: the draging delta of y , in MODEL coordinate.
#       """
#       pass
#
#   def node_adjust_stop_cb(self, tdw, dx, dy):
#       """ User ends dragging the selected node(s).
#       The deriving class should add(or manipulate)
#       the `dx, dy` value into node.
#       This mixin only supply target node and the delta values.
#
#       :param dx: the draging delta of x , in MODEL coordinate.
#       :param dy: the draging delta of y , in MODEL coordinate.
#       """
#       pass
#
#   def node_scroll_cb(self, tdw, node, x_delta, y_delta):
#       """ The scroll wheel activated over a node.
#       This handler might be called for multiple time
#       (for each of the selected nodes) from single scroll event.
#
#       :param x_delta: x_delta value of scroll wheel.
#       :param y_delta: y_delta value of scroll wheel.
#       """
#       pass
#
#   ## Capturing (Generating) event handlers
#
#   def node_capture_start_cb(self, tdw, event):
#       """ User starts capturing phase right now.
#
#       Generally deriving class should generate
#       a new node in this callback.
#       """
#       pass
#
#   def node_capture_update_cb(self, tdw, event, dx, dy):
#       """ User dragging within capturing phase.
#
#       Some deriving class like Inktool generates
#       additional new nodes inside this callback.
#       """
#       pass
#
#   def node_capture_stop_cb(self, tdw):
#       """ User ends capturing the selected node(s).
#       """
#       pass
#
#
    ## Interrogating events
 
 
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
 
    ## Node related

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
 
    def _search_target_node(self, tdw, x, y, margin=12):
        """ utility method: to commonize node searching codes
        """
        hit_dist = gui.style.DRAGGABLE_POINT_HANDLE_SIZE + margin
        new_target_node_index = None
        for i, node in reversed(list(enumerate(self.nodes))):
            node_x, node_y = tdw.model_to_display(node.x, node.y)
            d = math.hypot(node_x - x, node_y - y)
            if d > hit_dist:
                continue
            new_target_node_index = i
            break
        return new_target_node_index

    def get_node_dtime(self, i):
        if not (0 < i < len(self.nodes)):
            return 0.0
        n0 = self.nodes[i-1]
        n1 = self.nodes[i]
        dtime = n1.time - n0.time
        dtime = max(dtime, self.MIN_INTERNODE_TIME)
        return dtime
 
    def set_node_dtime(self, i, dtime):
        dtime = max(dtime, self.MIN_INTERNODE_TIME)
        nodes = self.nodes
        if not (0 < i < len(nodes)):
            return
        old_dtime = nodes[i].time - nodes[i-1].time
        for j in range(i, len(nodes)):
            n = nodes[j]
            new_time = n.time + dtime - old_dtime
            self.update_node(j, time=new_time)

 
    def can_insert_node(self, i):
        return 0 <= i < len(self.nodes)-1
 
    def can_delete_node(self, i):
        pass

#  #def _adjust_current_node_index(self):
#  #    """ Adjust self.current_node_index
#  #    child classes might have different behavior
#  #    from Inktool about current_node_index.
#  #    """
#  #    if self.current_node_index >= len(self.nodes):
#  #        self.current_node_index = len(self.nodes) - 2
#  #        if self.current_node_index < 0:
#  #            self.current_node_index = None
#  #        self.current_node_changed(
#  #                self.current_node_index)
#
#   def delete_node(self, i):
#       """Delete a node, and issue redraws & updates"""
#       assert self.can_delete_node(i), "Can't delete endpoints"
#       # Redraw old locations of things while the node still exists
#       self._queue_draw_buttons()
#       self._queue_draw_node(i)
#
#       self._pop_node(i)
#
#       # Limit the current node.
#       # this processing may vary in inherited classes,
#       # so wrap this.
#      #self._adjust_current_node_index()
#
#       self.options_presenter.target = (self, self.current_node_index)
#       # Issue redraws for the changed on-canvas elements
#       self._queue_redraw_display()
#
#
#   def delete_selected_nodes(self):
#
#       self._queue_draw_buttons()
#       for idx in self.selected_nodes:
#           self._queue_draw_node(idx)
#
#       new_nodes = [self.nodes[0]]
#       for idx,cn in enumerate(self.nodes[1:-1]):
#           t_idx = idx + 1
#           if t_idx in self.selected_nodes:
#               if self.current_node_index == t_idx:
#                   self.current_node_index = None
#           else:
#               new_nodes.append(cn)
#
#       new_nodes.append(self.nodes[-1])
#       self.nodes = new_nodes
#       self._reset_selected_nodes()
#       self.target_node_index = None
#
#       # Issue redraws for the changed on-canvas elements
#       self._queue_redraw_display()
#
 
#   def insert_node(self, i):
#       """Insert a node, and issue redraws & updates"""
#       assert self.can_insert_node(i), "Can't insert back of the endpoint"
#       # Redraw old locations of things while the node still exists
#       self._queue_draw_buttons()
#       self._queue_draw_node(i)
#       # Create the new node
#       cn = self.nodes[i]
#       nn = self.nodes[i+1]
#
#       newnode = _Node(
#           x=(cn.x + nn.x)/2.0, y=(cn.y + nn.y) / 2.0,
#           pressure=(cn.pressure + nn.pressure) / 2.0,
#           xtilt=(cn.xtilt + nn.xtilt) / 2.0,
#           ytilt=(cn.ytilt + nn.ytilt) / 2.0,
#           time=(cn.time + nn.time) / 2.0
#       )
#       self.nodes.insert(i+1,newnode)
#
#       # Issue redraws for the changed on-canvas elements
#       self._queue_redraw_display()
#
#   def insert_current_node(self):
#       if self.can_insert_node(self.current_node_index):
#           self.insert_node(self.current_node_index)
#
#   def _pop_node(self, idx):
#       """ wrapper method of popping(delete) node.
#       to ensure not included in self.selected_nodes.
#       """
#       if idx in self.selected_nodes:
#           self.selected_nodes.remove(idx)
#
#       for i, sidx  in enumerate(self.selected_nodes):
#           if sidx > idx:
#               self.selected_nodes[i] = sidx - 1
#
#       def adjust_index(cur_idx, targ_idx):
#           if cur_idx == targ_idx:
#               cur_idx = -1
#           elif cur_idx > targ_idx:
#               cur_idx -= 1
#
#           if cur_idx < 0:
#               return None
#           return cur_idx
#
#
#       self.current_node_index = adjust_index(self.current_node_index,idx)
#       self.target_node_index = adjust_index(self.target_node_index,idx)
#
#       return self.nodes.pop(idx)
#
#
#   ## Node selection
#   def select_all(self):
#       self.selected_nodes = range(0, len(self.nodes))
#       self._queue_redraw_all_nodes()
#
#   def deselect_all(self):
#       self._reset_selected_nodes(None)
#       self._queue_redraw_all_nodes()
#
#   def select_area_cb(self, selection_mode):
#       """ Selection handler called from SelectionMode.
#       This handler never called when no selection executed.
#       """
#       modified = False
#       for idx,cn in enumerate(self.nodes):
#           if selection_mode.is_inside_model(cn.x, cn.y):
#               if not idx in self.selected_nodes:
#                   self.selected_nodes.append(idx)
#                   modified = True
#       if modified:
#           self._queue_redraw_all_nodes()
#      
#   def apply_pressure_from_curve_widget(self, curve):
#       """ Apply pressure reprenting points
#       from StrokeCurveWidget.
#       Mostly resembles as BezierMode.apply_pressure_points,
#       but inktool stroke calculartion is not same as
#       BezierMode.
#       """
#
#       # We need smooooth value, so treat the points
#       # as Bezier-curve points.
#
#       # first of all, get the entire stroke length
#       # to normalize stroke.
#
#       if len(self.nodes) < 2:
#           return
#
#       self._queue_redraw_curve()
#
#       # Getting entire stroke(vector) length
#       node_length=[]
#       total_length = 0.0
#
#       for idx, cn in enumerate(self.nodes[:-1]):
#           nn = self.nodes[idx + 1]
#           length = math.sqrt((cn.x - nn.x) ** 2 + (cn.y - nn.y) ** 2)
#           node_length.append(length)
#           total_length+=length
#
#       node_length.append(total_length) # this is sentinel
#
#
#
#       # use control handle class temporary to get smooth pressures.
#       cur_length = 0.0
#       new_nodes=[]
#       for idx,cn in enumerate(self.nodes):
#           val = curve.get_pressure_value(cur_length / total_length)
#           new_nodes.append(cn._replace(pressure=val)) 
#           cur_length += node_length[idx]
#
#       self.nodes = new_nodes
#
#
    ## Nodes hide
    @property
    def hide_nodes(self):
        return self._hide_nodes
 
    @hide_nodes.setter
    def hide_nodes(self, flag):
        self._hide_nodes = flag
        self._queue_redraw_all_nodes()
 
    ## Generic Oncanvas-editing interface

    def delete_item(self):
        self.delete_selected_nodes()
 
#   ## Editing finalize / discarding
#   def accept(self):
#       if self.phase == _Phase.ADJUST:
#               len(self.nodes) > 1):
#           self._start_new_capture_phase(rollback=True)
#
#   def cancel(self):
#       if self.phase == _Phase.ADJUST:
#           self._start_new_capture_phase(rollback=False)
#



class OverlayOncanvasMixin(gui.overlays.Overlay):
    """ The mixin of overlay for Oncanvas-editing mixin.
    """
 
    def __init__(self, mode, tdw):
        super(OverlayOncanvasMixin, self).__init__()
        self._mode = weakref.proxy(mode)
        self._tdw = weakref.proxy(tdw)
        self._button_pixbuf_cache = {}
        self._button_pos = {}
 
    def get_button_pos(self, button_id):
        return self._button_pos.get(button_id, None)
 
    def update_button_positions(self):
        """Recalculates the positions of the mode's buttons.
        
        Normally the class uses this mixin should override
        this method, to reject when `self._mode.nodes` is
        not ready to display action buttons yet.
        """
        pass
 
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
        mode = self._mode
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        alloc = self._tdw.get_allocation()
        for i, node in enumerate(mode.nodes):
            x, y = self._tdw.model_to_display(node.x, node.y)
            node_on_screen = (
                x > alloc.x - radius*2 and
                y > alloc.y - radius*2 and
                x < alloc.x + alloc.width + radius*2 and
                y < alloc.y + alloc.height + radius*2
            )
            if node_on_screen:
                yield (i, node, x, y)

    def _draw_buttons(self, cr, button_info):
        radius = gui.style.FLOATING_BUTTON_RADIUS
        for icon_name, pos, zone in button_info:
            if pos is None:
                continue
            x, y = pos
            if self._mode.zone == zone:
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
 
    def paint(self, cr):
        """Draw adjustable nodes to the screen"""
        pass
        
