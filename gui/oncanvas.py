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
import time
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
#        self._queue_redraw_item()
#        self._queue_redraw_all_nodes()
#        self._queue_draw_buttons()
#
#        # the method should return deleted nodes count
#        result = method(self, *args)
#        assert type(result) == int
#
#        if result > 0:
#            self.options_presenter.target = (self, self.current_node_index)
#            self._queue_redraw_item()
#            self._queue_redraw_all_nodes()
#            self._queue_draw_buttons()
#        return result
#    return _decorator


## Class defs

class PhaseMixin:
    """Enumeration of the states that an InkingMode can be in"""
    CAPTURE = 0 # Initial phase, it should be 'capture input'.
    ADJUST = 1  # Adjust phase, but nothing has begun yet.
    ADJUST_POS = 2  # The phase of adjusting node position.
                    # this is common for all oncanvas class.
    ACTION = 3  # Action button or something.

    CHANGE_PHASE = -1 # Changing phase - bypassing other phases


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
class EditZoneMixin:
    """Enumeration of where the pointer is on,in the ADJUST phase"""

    EMPTY_CANVAS = 0  #: Nothing, empty space
    CONTROL_NODE = 1  #: Any control node; see target_node_index
    CONTROL_HANDLE = 2 #: Control handle index to manipulate node onscreen, 
                       #  not all mixin user use this constant.
    ACTION_BUTTON = 3  #: On-canvas action button. Also used when drawing overlay.


class ActionButtonMixin:
    """Enumeration for the action button definition. """
    ACCEPT=0
    REJECT=1

class OncanvasEditMixin(gui.mode.ScrollableModeMixin,
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
 
 
    ## Override action
    #
    #  This is mainly use to avoid conflicting shift/ctrl keys.
    #  Only listed action modes here (or completely empty for all modes)
    #  are permitted.
    #  But, these are default setting... if some user change shift/ctrl
    #  keybinds, this might not work well.
    permitted_switch_actions = None
    _enable_switch_actions = set()   # This means 'Any action is permitted'
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

    _WHEEL_STEP = 0.025 # mouse wheel delta step,for scroll event
 
    drag_offset = gui.ui_utils.DragOffset() #: Dragging management class singleton

    ## Button configration
    _buttons = None
 
    ## Initialization & lifecycle methods
 
    def __init__(self, **kwargs):

        super(OncanvasEditMixin, self).__init__(**kwargs)
 
        #+ initialize selected nodes - 
        #+ place this prior to _reset_nodes()
        self.selected_nodes=[]
 
        self.phase = PhaseMixin.CAPTURE
        self.subphase = None
        self.zone = EditZoneMixin.EMPTY_CANVAS
        self.current_node_index = None  #: Node active in the options ui
        self.target_node_index = None  #: Node that's prelit
        self._overlays = {}  # keyed by tdw
        self.nodes = []
        self._reset_adjust_data()
        self._task_queue = collections.deque()  # (cb, args, kwargs)
        self._task_queue_runner_id = None
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

        #+ Current button information
        self.current_button_id = None # id of focused button
        self._clicked_buttn_id = None  # id of clicked button
 
    def _reset_capture_data(self):
        # XXX To be obsoluted, almost same as _reset_nodes()
        self.select_node(-1)
        self.nodes = []
 
    def _reset_adjust_data(self):
        """ Reset all datas about adjusting nodes.
        This should be called at entirely new phase starts.
        """
        self.zone = EditZoneMixin.EMPTY_CANVAS
        self.current_node_index = None
        self.target_node_index = None
        self._dragged_node_start_pos = None
        self.drag_offset.reset()
 
        # Multiple selected nodes.
        # This is a index list of node from self.nodes
        self.select_node(-1)
 
        self._hide_nodes = False

        self._click_info = None

        self.subphase = None
 
 
   #def _reset_selected_nodes(self, initial_idx=None):
   #    """ Resets selected_nodes list and assign
   #    initial index,if needed.
   #
   #    :param initial_idx: initial node index.in most case,
   #                        node will manipurate by solo.
   #                        it might be inefficient to
   #                        generate list each time s solo node
   #                        is moved,so use this parameter in such case.
   #    """
   #
   #    if initial_idx == None:
   #        if len(self.selected_nodes) > 0:
   #            self.selected_nodes=[]
   #    elif len(self.selected_nodes) == 0:
   #        self.selected_nodes.append(initial_idx)
   #    elif len(self.selected_nodes) == 1:
   #        self.selected_nodes[0] = initial_idx
   #    else:
   #        self.selected_nodes = [initial_idx, ]
 
    def _is_active(self):
        """ To know whether this mode is active or not. 
        """
        return self in self.doc.modes
       #for mode in self.doc.modes:
       #    if mode is self:
       #        return True
       #return False

    def is_adjusting_phase(self):
        """To know whether current phase is node adjusting phase.
        this method should be overriden in deriving classes.
        """
        return self.phase in (PhaseMixin.ADJUST,
                              PhaseMixin.ADJUST_POS)

    ## Buttons property 
    
    buttons = {
        ActionButtonMixin.ACCEPT : ('mypaint-ok-symbolic', 
            'accept_button_cb'), 
        ActionButtonMixin.REJECT : ('mypaint-trash-symbolic', 
            'reject_button_cb') 
    }

 
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
        # Queue current node to erase
        self._queue_draw_buttons()
        self._queue_redraw_all_nodes()
       #self._reset_nodes()
        self._reset_capture_data()
        self._reset_adjust_data()
        self.phase = PhaseMixin.CAPTURE
        self.enable_switch_actions(True)

    def enter(self, doc, **kwds):
        """Enters the mode: called by `ModeStack.push()` etc."""
        super(OncanvasEditMixin, self).enter(doc, **kwds)
        if not self._is_active():
            self._discard_overlays()
        self._ensure_overlay_for_tdw(self.doc.tdw)
        cursors = self.doc.app.cursors
        self._arrow_cursor = cursors.get_action_cursor(
            self.ACTION_NAME,
            gui.cursor.Name.ARROW,
        )
        self._crosshair_cursor = cursors.get_action_cursor(
            self.ACTION_NAME,
            gui.cursor.Name.CROSSHAIR_OPEN_PRECISE,
        )
        self._cursor_move_nw_se = cursors.get_action_cursor(
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
            self._queue_redraw_item()
 
    def _update_zone_and_target(self, tdw, x, y):
        """Update the zone and target node under a cursor position"""
 
        overlay = self._ensure_overlay_for_tdw(tdw)
        new_zone = EditZoneMixin.EMPTY_CANVAS
        new_handle_idx = None
 
        if not self.in_drag:
           #if self.phase in (PhaseMixin.CAPTURE, PhaseMixin.ADJUST):
           #if self.phase == PhaseMixin.ADJUST:
            if self.is_actionbutton_ready():
                new_target_node_index = None
                self.current_button_id = None
                # Test buttons for hits
                hit_dist = gui.style.FLOATING_BUTTON_RADIUS

                for btn_id in self.buttons.keys():
                    btn_pos = overlay.get_button_pos(btn_id)
                    if btn_pos is None:
                        continue
                    btn_x, btn_y = btn_pos
                    d = math.hypot(btn_x - x, btn_y - y)
                    if d <= hit_dist:
                        new_target_node_index = None
                        new_zone = EditZoneMixin.ACTION_BUTTON
                        self.current_button_id = btn_id
                        break

            # Test nodes for a hit, in reverse draw order
            if new_zone == EditZoneMixin.EMPTY_CANVAS:
                new_target_node_index, new_handle_idx = \
                        self._search_target_node(tdw, x, y)
                if new_target_node_index != None:
                    new_zone = EditZoneMixin.CONTROL_NODE

            # Update the prelit node, and draw changes to it
            if new_target_node_index != self.target_node_index:
                # Redrawing old target node.
                if self.target_node_index is not None:
                    self._queue_draw_node(self.target_node_index)
                    self.node_leave_cb(tdw, self.nodes[self.target_node_index]) 

                self.target_node_index = new_target_node_index
                if self.target_node_index is not None:
                    self._queue_draw_node(self.target_node_index)
                    self.node_enter_cb(tdw, self.nodes[self.target_node_index]) 

            self.current_node_handle = new_handle_idx
 
        # Update the zone, and assume any change implies a button state
        # change as well (for now...)
        self._enter_new_zone(tdw, new_zone)

    def _enter_new_zone(self, tdw, new_zone):
        """Entering new zone. This is actually postprocess of 
        _update_zone_and_target().

        This is useful when customize _update_zone_and_target()
        in deriving classes.
        """
        if self.zone != new_zone:
           #if not self.in_drag:
            self._enter_zone_cb(new_zone)
            self.zone = new_zone
            self._ensure_overlay_for_tdw(tdw)
            self._queue_draw_buttons()

        if not self.in_drag:
            self._update_cursor(tdw)

    def _update_cursor(self, tdw): 
        """ Call update_cursor_cb() of subclass
        and decide current cursor,according to zone and phase.

        The callback is 

        def update_cursor_cb(self, tdw)

        It should return a cursor object, or None.
        """

        cursor = self.update_cursor_cb(tdw)

        if cursor is not None and cursor is not self._current_override_cursor:
            tdw.set_override_cursor(cursor)
            self._current_override_cursor = cursor

    def _bypass_phase(self, next_phase):
        """ Bypass(cancel) follwing processing of current phase 
        and enter next_phase.

        This would be called from dragging/button press
        callbacks.
        """
        self.phase = PhaseMixin.CHANGE_PHASE
        self._returning_phase = next_phase

    def _enter_zone_cb(self, new_zone):
        """ Callback to notify entering new zone.
        Might be overridden in child class
        if needed.

        This callback (and _update_zone_and_target) would not be called
        when mode instance is in drag state.
        """
        app = self.doc.app
        if new_zone == EditZoneMixin.ACTION_BUTTON:
            id = self.current_button_id
            if id == ActionButtonMixin.ACCEPT:
                app.show_transient_message(_("Accept Button; Commit current edit."))
            elif id == ActionButtonMixin.REJECT:
                app.show_transient_message(_("Reject Button; Discard current edit."))
 
    ## Redraws
    def _queue_redraw_item(self):
        """ Just a placeholder, to be overridden at child class.
        """
        pass

    def _queue_draw_buttons(self):
        """Redraws the accept/reject buttons on all known view TDWs"""
        for tdw, overlay in self._overlays.items():
            overlay.update_button_positions()
            for id in self.buttons:
                pos = overlay.get_button_pos(id)
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

    def _queue_draw_node(self, idx, offsets=None):
        """ queue an oncanvas item to draw."""
        cn = self.nodes[idx]
        if offsets == None:
            dx=dy=0
        else:
            dx,dy = offsets
        for tdw in self._overlays:
            x, y = tdw.model_to_display(cn.x + dx, cn.y + dy)
            x = math.floor(x)
            y = math.floor(y)
            size = math.ceil(gui.style.DRAGGABLE_POINT_HANDLE_SIZE * 2)
            tdw.queue_draw_area(x-size, y-size, size*2+1, size*2+1)

    def _queue_draw_selected_nodes(self):
        offsets = self.drag_offset.get_model_offset()
        for i in self.selected_nodes:
            self._queue_draw_node(i, offsets=offsets) 

    def _queue_redraw_all_nodes(self):
        """Redraws all nodes on all known view TDWs"""
        offsets = self.drag_offset.get_model_offset()
        for i in xrange(len(self.nodes)):
            if i in self.selected_nodes:
                self._queue_draw_node(i, offsets=offsets)
            else:
                self._queue_draw_node(i)

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
 
    ## Raw event handling (prelight & zone selection in adjust phase)

    def motion_notify_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False
 
        self._update_zone_and_target(tdw, event.x, event.y)
        return super(OncanvasEditMixin, self).motion_notify_cb(tdw, event)

    def button_press_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False
        self._update_zone_and_target(tdw, event.x, event.y)
        self._update_current_node_index()

        # Update workaround state for evdev dropouts
        self._button_down = event.button

        shift_state =  event.state & Gdk.ModifierType.SHIFT_MASK != 0
        ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK != 0

        if self.zone == EditZoneMixin.ACTION_BUTTON:
            button = event.button
            if button == 1:
                buttons = self.buttons
                assert self.current_button_id != None
                assert self.current_button_id in buttons

                # We need the id of 'pressed' button
                # current_button_id is the 'focused' button,
                # not pressed.
                self._clicked_button_id = self.current_button_id
                self.phase = PhaseMixin.ACTION

                return False
        elif self.phase in (PhaseMixin.CAPTURE, PhaseMixin.ADJUST):

            if (self.phase == PhaseMixin.CAPTURE
                and self.zone == EditZoneMixin.EMPTY_CANVAS):
                    self._start_new_capture_phase(rollback=False)

            if self.zone == EditZoneMixin.CONTROL_NODE:
                # clicked a node.
                mx, my = tdw.display_to_model(event.x, event.y)
                self.drag_offset.start(mx, my)
                self.phase = PhaseMixin.ADJUST_POS
                
                targidx = self.target_node_index
                if ctrl_state:
                    # Node selection editing mode.
                    self.select_node(targidx, not ctrl_state)
                else:
                    # Ordinary selection of a node.
                    # When the target node is already in selected_nodes list,
                    # only activate it (i.e. it becomes self.current_node_index).
                    # Otherwise, only the node is selected, other selected nodes
                    # should be cleared.
                    self.select_node(targidx ,targidx)


            else:
                pass

            # FALLTHRU: *do* start a drag

        if not self.mode_button_press_cb(tdw, event):
            # Supercall: start drags etc
            return super(OncanvasEditMixin, self).button_press_cb(tdw, event)

    def mode_button_press_cb(self, tdw, event):
        """ Intermidiate button press callback.
        This method is for avoiding same processing
        (such as _ensure_overlay, _update_zone_and_target, etc)
        even when child class call superclass method.
        """
        pass
    
    def button_release_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        try:
            # Use try-finally to ensure self._button_down is set
            # in any case, even return from this method. 
            if not (tdw.is_sensitive and current_layer.get_paintable()):
                return False

            if self.phase == PhaseMixin.ACTION:
                assert self._button_down >= 1
                assert self._clicked_button_id != None
                self._call_action_button(self._clicked_button_id, tdw)
                # Inside action button handler, self.phase would be set as 
                # PhaseMixin.CAPTURE in many case
                # (by calling _start_new_capture_phase).
                # But, it is not mandatory.Some action button might
                # do something without changing phase.
                self._clicked_button_id = None
                self._update_zone_and_target(tdw, event.x, event.y)
                self.phase = PhaseMixin.CAPTURE
                return False  # NO `drag_stop_cb` activated.

            elif self.phase == PhaseMixin.CHANGE_PHASE:
                self.phase = self._returning_phase

            elif self.mode_button_release_cb(tdw, event):
                # mode_button_release_cb returned True
                # so bypass supercall
                # CAUTION: bypassing supercalling button_release_cb
                # is NOT RECOMMENDED, annoying subeffect might happen.
                return False  

            # Supercall: stop current drag
            return super(OncanvasEditMixin, self).button_release_cb(tdw, event)

        finally:
            # Update workaround state for evdev dropouts
            self._button_down = None

    def mode_button_release_cb(self, tdw, event):
        pass

 
    ## Drag handling (both capture and adjust phases)
    # 
    #  We did not need to implement them in mixin-user-class.
    #  Instead of it, implement `node_capture_*_cb()`
    #  and `node_adjust_*_cb()`
 
    def drag_start_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        if not self.node_drag_start_cb(tdw, event):
            return super(OncanvasEditMixin, self).drag_start_cb(tdw, event)
 
    def drag_update_cb(self, tdw, event, dx, dy):
        self._ensure_overlay_for_tdw(tdw)
        if not self.node_drag_update_cb(tdw, event, dx, dy):
            return super(OncanvasEditMixin, self).drag_update_cb(tdw, event,
                    dx, dy)
 
    def drag_stop_cb(self, tdw):
        self._ensure_overlay_for_tdw(tdw)

        if not self.node_drag_stop_cb(tdw):
            return super(OncanvasEditMixin, self).drag_stop_cb(tdw)

        self.drag_offset.reset()
 
    def scroll_cb(self, tdw, event):
        if self._prev_scroll_time != event.time:

            dx, dy = gui.ui_utils.get_scroll_delta(event, self._WHEEL_STEP)
            if not self.node_scroll_cb(tdw, dx, dy):
                return super(OncanvasEditMixin, self).scroll_cb(tdw, event)

        self._prev_scroll_time = event.time
 
    ## Node Editing event callbacks
 
    def node_enter_cb(self, tdw, node):
        """ The cursor hovers over a node
        You can change cursor if needed.
        """
        pass
 
    def node_leave_cb(self, tdw, node):
        """ The cursor hovers outside a node.
        You MUST reset cursor if change it.
        """
        pass
 
    def node_drag_start_cb(self, tdw, event):
        """ User starts dragging the selected node(s) right now.
        """
        if self.phase == PhaseMixin.ADJUST_POS:
            if self.target_node_index is not None:
                node = self.nodes[self.target_node_index]
                self._dragged_node_start_pos = (node.x, node.y)
                x, y = tdw.display_to_model(event.x, event.y)
                self.drag_offset.start(x, y)
 
    def node_drag_update_cb(self, tdw, event, dx, dy):
        """ User dragging the selected node(s) now.
 
        :param dx: the draging delta of x , in MODEL coordinate.
        :param dy: the draging delta of y , in MODEL coordinate.
        """
        if self.phase == PhaseMixin.ADJUST_POS:
            if self._dragged_node_start_pos:

                # To erase old-positioned nodes.
                self._queue_draw_selected_nodes()

                x, y = tdw.display_to_model(event.x, event.y)
                self.drag_offset.end(x, y)
                self._queue_draw_selected_nodes()

                self._queue_redraw_item()

 
    def node_drag_stop_cb(self, tdw):
        """ User ends dragging the selected node(s).
        The deriving class should add(or manipulate)
        the `dx, dy` value into node.
        This mixin only supply target node and the delta values.
 
        :param dx: the draging delta of x , in MODEL coordinate.
        :param dy: the draging delta of y , in MODEL coordinate.
        """
        if self.phase == PhaseMixin.ADJUST_POS:
            # Finalize dragging motion to selected nodes.
            if self._dragged_node_start_pos:
 
                self._queue_draw_selected_nodes() # to ensure erase them
                dx, dy = self.drag_offset.get_model_offset()

                if dx != 0 or dy != 0:
                    for i in self.selected_nodes:
                        cn = self.nodes[i]
                        self.nodes[i] = cn._replace(x=cn.x+dx, y=cn.y+dy)
 
                self.drag_offset.reset()

                self._queue_draw_selected_nodes() # to ensure erase them
                self._queue_redraw_item()
                self._queue_draw_buttons()
 
            self._dragged_node_start_pos = None
            self.phase = PhaseMixin.ADJUST
 
    def node_scroll_cb(self, tdw, dx, dy):
        """ The scroll wheel activated over a node.
        This handler might be called for multiple time
        (for each of the selected nodes) from single scroll event.
 
        :param dx: x delta value of scroll(tilt) wheel.
        :param dy: y delta value of scroll wheel.
        """
        pass
 
 
    ## Interrogating events
 
 
    def _get_event_pressure(self, event):
        # FIXME: CODE DUPLICATION: copied from freehand.py
        # However, this class has no parent-child relationship 
        # with freehand. so I copied this from it.
        # (or we need to create something mixin...?)
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
    def select_node(self, idx, exclusive=False, update_gui=True, ):
        """Select a node, i.e. add idx of node into selected_nodes list.

        :param idx: the selected node index. 
                    when this is -1, parameter exclusive is ignored, and
                    all nodes deselected.
                    
        :param bool or int exclusive: if this is True, only the 'idx' node is selected,
                                      all other nodes are removed from selected_nodes list.

                                      Otherwise, if exclusive is integer and equal to idx,
                                      this means 'Do not remove this node when the node
                                      is in self.selected_nodes,only activate it. Otherwise, 
                                      only the 'idx' node should be in the list.'

        :param update_gui: Assign to update overlay gui.
                           When many nodes iterated and selected 
                           over and over again, set False to this flag 
                           and update overlay after all selection has been done.
        """
        if update_gui:
           #self._queue_draw_selected_nodes() # To erase
            self._queue_redraw_all_nodes() # To erase

        if idx == -1:
            self.selected_nodes = []
        else:
            if exclusive == True:
                self.selected_nodes = [idx]
            elif exclusive == idx:
                # Special case : do not remove from selection,
                # Just append the node when it is not in selection.
                if not idx in self.selected_nodes:
                    self.selected_nodes = [idx]
            else:
                if idx in self.selected_nodes:
                    self.selected_nodes.remove(idx)
                else:
                    self.selected_nodes.append(idx)

            if update_gui:
                self._queue_draw_node(idx)

        if len(self.selected_nodes) == 0:
            self.target_node_index = None
            self.current_node_index = None
        else:
            self.current_node_index = idx


    def _update_current_node_index(self):
        """Updates current_node_index from target_node_index & redraw"""
        new_index = self.target_node_index
        old_index = self.current_node_index
        if new_index == old_index:
            return

        if self.current_node_index != None:
            # Impotant line, to erase current node once.
            # Without this, garbage pixel remained screen
            # when there is some other visual objects (such as 'bezier handle')
            # around current node.
            self._queue_draw_node(self.current_node_index) 

        self.current_node_index = new_index
        self.current_node_changed(new_index)
        self.options_presenter.target = (self, new_index)
        for i in (old_index, new_index):
            if i is not None:
                self._queue_draw_node(i)
 
    @lib.observable.event
    def current_node_changed(self, index):
        """Event: current_node_index was changed"""
 

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

    @property
    def hide_nodes(self):
        return self._hide_nodes

    @hide_nodes.setter
    def hide_nodes(self, flag):
        if self._hide_nodes != flag:
            if not self._hide_nodes:
                self._queue_redraw_all_nodes()

            self._hide_nodes = flag

            if not self._hide_nodes:
                self._queue_redraw_all_nodes()


    def update_node(self, i, **kwargs):
        """Updates properties of a node, and redraws it.
        
        :param i: node index
        :param kwargs: keyword arguments of nodes. incoming datas, such as
                       'x', 'y', or 'pressure' are stored as keyworded data.
        """
        pass # placeholder, to be implemented in derived class.


       #changing_pos = bool({"x", "y"}.intersection(kwargs))
       #oldnode = self.nodes[i]
       #if changing_pos:
       #    self._queue_draw_node(i)
       #self.nodes[i] = oldnode._replace(**kwargs)
       ## FIXME: The curve redraw is a bit flickery.
       ##   Perhaps dragging to adjust should only draw an
       ##   armature during the drag, leaving the redraw to
       ##   the stop handler.
       #self._queue_redraw_item()
       #if changing_pos:
       #    self._queue_draw_node(i)

    ## Action button related

    def is_actionbutton_ready(self):
        """customize whether the action buttons are ready to display.
        This method can be overridden from child classes,
        to change phase detection behavior.
        """
        return (len(self.nodes) > 0 and 
                self.phase in (PhaseMixin.CAPTURE, PhaseMixin.ADJUST))

    def _call_action_button(self, id, tdw):
        """ Call action button, from id (i.e. _ActionButton constants)

        Internally, this method get method from string 
        which is defined as class attribute of button structure
        and call it.
        """
        try:
            junk, handler_name = self.buttons[id]
            method = getattr(self, handler_name)
            method(tdw)
        except KeyError as e:
            logger.error("Action button %d does not exist" % id)
        except AttributeError as e:
            logger.error("Action button %d handler named %s, but it does not exist" % (id, handler_name))
            print(str(e))





class PressPhase(PhaseMixin):
    ADJUST_PRESSURE = 4
    ADJUST_PRESSURE_ONESHOT = 5
    INSERT_NODE = 6         #: Inserting node phase, with clicking on stroke,
                            #  a node inserted at the point of stroke.
                            #  this phase is one-shot, immidiately return to 
                            #  _Phase.ADJUST.

class PressureEditableMixin(OncanvasEditMixin,
                            gui.mode.BrushworkModeMixin):
    """ The mixin for oncanvas pressure editable class.
    """

    ## Pressure oncanvas edit settings

    # Pressure editing key modifiers,single node and with nearby nodes.
    # these can be hard-coded,but we might need some customizability later.
    _PRESSURE_MOD_MASK = Gdk.ModifierType.SHIFT_MASK
    _ADD_SELECTION_MASK = Gdk.ModifierType.CONTROL_MASK


    def is_pressure_modifying(self):
        return self.phase in (PressPhase.ADJUST_PRESSURE,
                    PressPhase.ADJUST_PRESSURE_ONESHOT)

    def __init__(self, **kwargs):
        super(PressureEditableMixin, self).__init__(**kwargs)
        self._sshot_before = None
        self._pending_cmd = None

    def _detect_on_stroke(self, x, y, allow_distance = 4.0):
        """Detecting pressed point is on the stroke currently editing.
        
        :param x: cursor x position in MODEL coord
        :param y: cursor y position in MODEL coord
        :param allow_distance: the allowed distance from stroke.
        :return : a tuple of (the index of 'previous' node, time parameter of stroke)
        :rtype : a tuple when the pressed point is on stroke, otherwise
                 None.
        
        """
        return None

    def _enter_zone_cb(self, new_zone):
        """ Entering zone callback
        """
        app = self.doc.app
        if new_zone == EditZoneMixin.EMPTY_CANVAS:
            pass
           #app.show_transient_message(_(""))
        elif new_zone == EditZoneMixin.CONTROL_NODE:
            app.show_transient_message(_("Node editing; Shift change pressure, Ctrl edit selection group."))
        else:
            super(PressureEditableMixin, self)._enter_zone_cb(new_zone)

    ## Event handlers

    def mode_button_press_cb(self, tdw, event):
        if self.is_adjusting_phase():
            button = event.button
            if self.phase != PressPhase.ADJUST_PRESSURE:

                cls = self.__class__

                if button == 1 and self.zone == EditZoneMixin.CONTROL_NODE:

                    if self.current_node_index is not None:
                        if (event.state & cls._PRESSURE_MOD_MASK ==
                            cls._PRESSURE_MOD_MASK):
                        
                            self.phase = PressPhase.ADJUST_PRESSURE_ONESHOT
                            # Fallthrough
                        else:
                            assert self.phase == PressPhase.ADJUST_POS 

            
                # FALLTHRU: *do* start a drag
        else:
            super(PressureEditableMixin, self).mode_button_press_cb(tdw, event)

    def mode_button_release_cb(self, tdw, event):

        if self.is_adjusting_phase():

            if self.is_pressure_modifying():
                # When pressure has changed on canvas,
                # refrect it to presenter.
                self.options_presenter.target = (self, self.current_node_index)

                ## FALLTHRU

            self._update_zone_and_target(tdw, event.x, event.y)

        else:
            super(PressureEditableMixin, self).mode_button_press_cb(tdw, event)


    def node_drag_start_cb(self, tdw, event):
        if self.phase == PressPhase.ADJUST_PRESSURE_ONESHOT:
                self._queue_draw_selected_nodes()
        elif self.phase == PressPhase.ADJUST_PRESSURE:
                self._queue_redraw_all_nodes()
        elif self.phase == PressPhase.CAPTURE:
            # Update options_presenter when capture phase end
            self.options_presenter.target = (self, None)
        else:
            super(PressureEditableMixin, self).node_drag_start_cb(tdw, event)

    def node_drag_update_cb(self, tdw, event, dx, dy):
        if self.phase in (PressPhase.ADJUST_PRESSURE_ONESHOT,
                PressPhase.ADJUST_PRESSURE):
            self._node_dragged = True
            self._adjust_pressure_with_motion(dx, dy)
        else:
            super(PressureEditableMixin, self).node_drag_update_cb(tdw, 
                    event, dx, dy)

                
    def node_drag_stop_cb(self, tdw):
        if self.is_pressure_modifying():
            if self._node_dragged:

                ## Pressure editing phase end.
                if self.phase == PressPhase.ADJUST_PRESSURE_ONESHOT:
                    self.phase = PressPhase.ADJUST

                self._dragged_node_start_pos = None

                self._queue_redraw_all_nodes()
                self._queue_redraw_item()
                self._queue_draw_buttons()
        else:
            super(PressureEditableMixin, self).node_drag_stop_cb(tdw)


    def node_scroll_cb(self, tdw, dx, dy):
        """Handles scroll-wheel events, to adjust pressure.
        
        :param dx: delta x of (tilt) wheel scroll
        :param dy: delta y of wheel scroll
        """
        if self.phase == PressPhase.ADJUST and self.target_node_index != None:

            if len(self.selected_nodes) == 0:
                targets = (self.target_node_index,)
            else:
                targets = self.selected_nodes
 
            for idx in targets:
                node = self.nodes[idx]
                new_pressure = node.pressure
                new_pressure += dy

                if new_pressure != node.pressure:
                    self.nodes[idx]=node._replace(pressure=new_pressure)

                if idx == self.target_node_index:
                    self.options_presenter.target = (self, self.target_node_index)

            self._queue_redraw_item()

            return True # To suppress invoke superclass handler

    def node_enter_cb(self, tdw, node):
        self.enable_switch_actions(False)

    def node_leave_cb(self, tdw, node):
        self.enable_switch_actions(True)

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

            if len(self.selected_nodes) == 0:
                targets = (self.target_node_index, )
            else:
                targets = self.selected_nodes

            for i in targets:
                if i != None:
                    cn = self.nodes[i]
                    self.nodes[i] = \
                            cn._replace(pressure = lib.helpers.clamp(
                                cn.pressure + diff,0.0, 1.0))

        self._queue_redraw_item()

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

        self._queue_redraw_item()

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

        self._queue_redraw_item()


    def enter_pressure_phase(self):
        """ To enter/force pressure editing phase
        from outside of inktool.
        """
        # FIXME this is experimental code.
        # I'm unsure whether this functionality is 
        # useful or not.
        if self.phase == PressPhase.ADJUST_PRESSURE:
            self.phase = PressPhase.ADJUST
        elif self.phase == PressPhase.ADJUST:
            self.phase = PressPhase.ADJUST_PRESSURE
        self._queue_redraw_all_nodes()
        for tdw in self._overlays:
            self._update_cursor(tdw) 

    ## Brushwork related, to enable undo node operations.

    def brushwork_begin(self, model, description=None, abrupt=False):
        """Begins a new segment of active brushwork for a model
    
        :param lib.document.Document model: The model to begin work on
        :param unicode description: Optional description of the work
        :param bool abrupt: Tail out/in abruptly with faked zero pressure.
    
        Any current segment of brushwork is committed, and a new segment
        is begun.
    
        Passing ``None`` for the description is suitable for freehand
        drawing modes.  This method will be called automatically with
        the default options by `stroke_to()` if needed, so not all
        subclasses need to use it.
    
        The first segment of brushwork begin by a newly created
        BrushworkMode objects always starts abruptly.
        The second and subsequent segments are assumed to be
        continuations by default. Set abrupt=True to break off any
        existing segment cleanly, and start the new segment cleanly.
    
        """
        # Commit any previous work for this model
        cmd = self._active_brushwork.get(model)
        if cmd is not None:
            self.brushwork_commit(model, abrupt=abrupt)
    
        # New segment of brushwork
        layer_path = model.layer_stack.current_path
        cmd = lib.command.Nodework(
            model, layer_path,
            self.doc, self.nodes,
            self.__class__,
            override_sshot_before=self._sshot_before,
            description=description,
            abrupt_start=(abrupt or self.__first_begin),
        )
        self._sshot_before = None
        self.__first_begin = False
        cmd.__last_pos = None
        self._active_brushwork[model] = cmd

    def brushwork_commit(self, model, abrupt=False):
        """Commits any active brushwork for a model to the command stack

        :param lib.document.Document model: The model to commit work to
        :param bool abrupt: End with a faked zero pressure "stroke_to()"

        This only makes a new entry on the command stack if
        the currently active brushwork segment made
        any changes to the model.

        See also `brushwork_rollback()`.
        """
        cmd = self._active_brushwork.pop(model, None)
        if cmd is None:
            return
        if abrupt and cmd.__last_pos is not None:
            x, y, xtilt, ytilt = cmd.__last_pos
            pressure = 0.0
            dtime = 0.0
            cmd.stroke_to(dtime, x, y, pressure, xtilt, ytilt)
        changed = cmd.stop_recording(revert=False)
        if changed:
            if self._pending_cmd:
                model.command_stack.remove_command(self._pending_cmd)
                self._pending_cmd = None
            model.do(cmd)


    def undo_nodes_cb(self, cmd, nodes, sshot_before):
        """ called from lib.command.Nodework.undo().
        
        This callback is called when a work using this mode
        is undone.
        Nodework command detects current operation mode and only when
        current mode is this mode, call this callback.
        """
        self.nodes = nodes
        self.phase = PhaseMixin.ADJUST
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()
        self._sshot_before = sshot_before
        self._pending_cmd = cmd
        
    def redo_nodes_cb(self, cmd, nodes):
        """ called from lib.command.Nodework.redo().
        
        This callback is called when a command which using this mode
        is redone.
        Nodework command detects current operation mode and only when
        current mode is this mode, call this callback.
        """

        if len(nodes) >= 2:
            self.nodes = nodes
            self.phase = PhaseMixin.ADJUST
            self._queue_redraw_all_nodes()
            self._queue_draw_buttons()
            self._pending_cmd = cmd
        else:
            logger.warning('redo notified, but node count %d is not suffcient.' % len(nodes))


class NodeUserMixin(object):
    """ The Mixin which use oncanvas node object.
    this Mixin is created assuming a node which is made from
    namedtuple object.
    """

    def _search_target_node(self, tdw, x, y, margin=12):
        """ utility method: to commonize node searching codes
        This class 
        Since this Mixin is created assuming a node 
        without a control handle, it always returns -1 for handles.
        So you might need to override this method.

        :return : a tuple of (target-node-index, control-handle-idx)
                  if no target node found, that index should be None.
                  also if no control handle found, that index should be None.
        :rtype tuple:
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
        return (new_target_node_index, None)

class HandleNodeUserMixin(NodeUserMixin):
    """ The mixin which use a sort of node class
    which has "get_control_handle" method.
    """

    INITIAL_NODE_HANDLE_RANGE = (0, 1)

    def _search_target_node(self, tdw, x, y, margin=12):
        """ utility method: to commonize node searching codes
        This class 
        Since this Mixin is created assuming a node 
        without a control handle, it always returns -1 for handles.
        So you might need to override this method.

        :return : a tuple of (new target node index, new control handle idx)
        :rtype tuple:
        """
        targidx, node_handle = \
                super(HandleNodeUserMixin, self)._search_target_node(tdw, x, y, margin)
        if targidx == None:
            for n in xrange(len(self.nodes)):
                c_node = self.nodes[n]
                hit_dist = gui.style.FLOATING_BUTTON_RADIUS
                if n == 0:
                    seq = self.INITIAL_NODE_HANDLE_RANGE #(1,)
                else:
                    seq = (0, 1)
                for i in seq:
                    handle = c_node.get_control_handle(i)
                    hx, hy = tdw.model_to_display(handle.x, handle.y)
                    d = math.hypot(hx - x, hy - y)
                    if d > hit_dist:
                        continue
                    node_handle = i

                if node_handle != None:
                    return (n, node_handle)

        return (targidx, node_handle)

    def _queue_draw_handle_node(self, i, offsets=None, tdws=None):
        """Redraws a specific control node on all known view TDWs

        You must call this method from your class of _queue_draw_node(),
        to override OncanvasEditMixin._queue_draw_node()

        Because this mixin has nothing to do with OncanvasEditMixin,
        Python Multiple inheritance cannot override methods of OncanvasEditMixin
        with this mixin.
        """

        def get_area(nx, ny, radius, old_area=None):
            x, y = tdw.model_to_display(nx, ny)

            x = math.ceil(x)
            y = math.ceil(y)
            sx = x-radius-2
            sy = y-radius-2
            ex = x+radius+2
            ey = y+radius+2
            if not old_area:
                return (sx, sy, ex, ey)
            else:
                return (min(old_area[0], sx),
                        min(old_area[1], sy),
                        max(old_area[2], ex),
                        max(old_area[3], ey))


        node = self.nodes[i]
        radius = math.ceil(gui.style.DRAGGABLE_POINT_HANDLE_SIZE)
        dx = dy = 0

        if i in self.selected_nodes:
            if offsets == None:
                dx, dy = self.drag_offset.get_model_offset()
            else:
                dx, dy = offsets

        if tdws==None:
            tdws=self._overlays

        for tdw in tdws:
            area = get_area(node.x+dx, node.y+dy, radius)

            # Only CURRENT NODE can be manipulated control handles.
            # NOT SELECTED NODES.
            if i == self.current_node_index:
            
                # Active control handles also should be queued.
                for hi in (0,1):
                    # But,the first node only shows 2nd(index 1) handle.
                    if self.is_drawn_handle(i, hi):
                        handle = node.get_control_handle(hi)
                        area = get_area( 
                            handle.x+dx, handle.y+dy, 
                            radius, area)

            tdw.queue_draw_area(area[0], area[1], 
                    area[2] - area[0] + 1, 
                    area[3] - area[1] + 1)


    def is_drawn_handle(self, node_idx, hndl_idx):
        return ((hndl_idx == 0 and node_idx > 0) or 
                (hndl_idx == 1 and node_idx <= len(self.nodes)-1))


class OverlayOncanvasMixin(gui.overlays.Overlay):
    """ The mixin of overlay for Oncanvas-editing mixin.
    """
 
    def __init__(self, mode, tdw):
        super(OverlayOncanvasMixin, self).__init__()
        self._mode = weakref.proxy(mode)
        self._tdw = weakref.proxy(tdw)
        self._button_pixbuf_cache = {}

        # Button position dictionary.
        # The key is _ActionButton Enumeration attributes
        # of each oncanvas editing class.
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

    def _draw_button(self, cr, pos, resource, radius, active):
        """ Drawing a button
        :param cr: Cairo context
        :param pos: a tuple of button position
        :param resource: the resource name 
        :param radius: button size(radius)
        :param active: button is active or not
        """
        if pos is None:
            return

        x, y = pos
        if active:
            color = gui.style.ACTIVE_ITEM_COLOR
        else:
            color = gui.style.EDITABLE_ITEM_COLOR
        icon_pixbuf = self._get_button_pixbuf(resource)
        gui.drawutils.render_round_floating_button(
            cr=cr, x=x, y=y,
            color=color,
            pixbuf=icon_pixbuf,
            radius=radius,
        )

    def _draw_mode_buttons(self, cr):
        mode = self._mode
        self.update_button_positions()
        radius = gui.style.FLOATING_BUTTON_RADIUS
        for id in mode.buttons:
            resource, junk = mode.buttons[id]
            if id in self._button_pos and self._button_pos[id] != None:
                self._draw_button(cr, 
                        self._button_pos[id], resource, radius, 
                        mode.current_button_id == id)
 
    def paint(self, cr):
        """Draw adjustable nodes to the screen"""
        pass

## Recallable on-canvas nodes mixin
#  And related classes.

class StrokeHistory(object):
    """ Stroke History class.this is singleton,stored as 
    class attribute of RecallableNodeMixin
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
        if x != None and y != None:
            x = -fn.x + x
            y = -fn.y + y
        else:
            x = 0
            y = 0

        ret = []
        for cn in src:

            if hasattr(cn, 'copy'):
                # It would be a dedicated node class.
                new_node = cn.copy(x, y)
            elif hasattr(cn, '_replace'):
                # This should be collection.namedtuple node
                new_node = cn._replace(x=cn.x+dx, y=cn.y+dy) 

            ret.append(new_node)

        return ret
        
class RecallableNodeMixin(object):
    """Node recall feature, which is to record nodes position
    when final rendering a oncanvas-editing item, and later, 
    we can recall node and re-edit it from that records.
    """

    stroke_history = StrokeHistory(6) # stroke history

    def _init_recall(self):
        self._stroke_from_history = False

    def recall_nodes(self, idx):
        """ recall nodes from history

        CAUTION: this method reset the user class phase
        into PhaseMixin.ADJUST.
        """
        if 0 < idx < len(self.stroke_history.liststore):
            self._queue_draw_buttons()
            self._queue_redraw_all_nodes()
            self._queue_redraw_item()

            self._stroke_from_history = True

            x = y = None
            if len(self.nodes) > 0:
                x = self.nodes[0].x
                y = self.nodes[0].y

            self.nodes = self.stroke_history.get_and_place_nodes(
                    idx, x, y)

            self._queue_redraw_item()
            self._queue_redraw_all_nodes()
            self._queue_draw_buttons()
            self.phase = PhaseMixin.ADJUST
    

class RecallableNodePresnterMixin(object):
    """Options presenter mixin for node recallable mixin

    This mixin needs 'self._updating_ui' attributes, to show whether
    user class is updating/creating user interface.
    """

    def create_recall_combobox(self, builder, name, cls):
        """
        :param cls: metaclass object which uses RecallableNodeMixin
        """
        combo = builder.get_object(name)
        combo.set_model(cls.stroke_history.liststore)
        cell = Gtk.CellRendererText()
        combo.pack_start(cell,True)
        combo.add_attribute(cell,'text',0)
        self._stroke_history_combo = combo

    def stroke_history_combobox_changed_cb(self, widget):
        if self._updating_ui:
            return
        mode, node_idx = self.target
        if mode:
            mode.recall_nodes(self._stroke_history_combo.get_active())
    