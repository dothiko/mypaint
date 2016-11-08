# This file is part of MyPaint.
# Copyright (C) 2016 by dothiko <a.t.dothiko@gmail.com>

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""Freehand drawing modes"""

## Imports

import math
from numpy import array
from numpy import isfinite
from lib.helpers import clamp
import logging
from collections import deque
logger = logging.getLogger(__name__)
import random
import json

from gettext import gettext as _
from lib import brushsettings
from gi.repository import GLib
from gi.repository import GdkPixbuf
import cairo

import gui.mode
from drawutils import spline_4p
from lib import mypaintlib
from gui.inktool import *
from gui.inktool import _LayoutNode, _Phase
from gui.linemode import *
from lib.command import Command
from gui.ui_utils import *
from gui.stamps import *
from lib.color import HCYColor, RGBColor
import lib.helpers
from gui.oncanvas import *

## Module settings

## Functions



## Class defs

_NODE_FIELDS = ("x", "y", "angle", "scale_x", "scale_y", "tile_index")

class _StampNode (collections.namedtuple("_StampNode", _NODE_FIELDS)):
    """Recorded control point, as a namedtuple.

    Node tuples have the following 6 fields, in order

    * x, y: model coords, float
    * angle: float in [-math.pi, math.pi] (radian)
    * scale_w: float in [0.0, 3.0]
    * scale_h: float in [0.0, 3.0]
    """
class _PhaseStamp(_Phase):
    """Enumeration of the states that an BezierCurveMode can be in"""
    MOVE   = 100         #: Moving stamp
    ROTATE = 101         #: Rotate with mouse drag 
    SCALE  = 102         #: Scale with mouse drag 
    ROTATE_BY_HANDLE = 103 #: Rotate with handle of GUI
    SCALE_BY_HANDLE = 104  #: Scale  with handle of GUI
    CALL_BUTTONS = 106     #: call buttons around clicked point. 
    ADJUST_SOURCE = 107    #: Adjust source target area
    ADJUST_SOURCE_BY_HANDLE = 108    #: Adjust source target area with handle.

class _EditZone(EditZone_Mixin):
    """Enumeration of what the pointer is on in phases"""
    CONTROL_HANDLE_0 = 100
    CONTROL_HANDLE_1 = 101
    CONTROL_HANDLE_2 = 102
    CONTROL_HANDLE_3 = 103
    CONTROL_HANDLE_BASE = 100
    SOURCE_AREA = 110
    SOURCE_AREA_HANDLE = 111
    SOURCE_TRASH_BUTTON = 112


class DrawStamp(Command):
    """Command : Draw a stamp(pixbuf) on the current layer"""

    display_name = _("Draw stamp(s)")

    def __init__(self, model, stamp, nodes, bbox, **kwds):
        """
        :param bbox: boundary rectangle,in model coordinate.
        """
        super(DrawStamp, self).__init__(model, **kwds)
        self.nodes = nodes
        self._stamp = stamp
        self.bbox = bbox
        self.snapshot = None

    def redo(self):
        # Pick a source
        target = self.doc.layer_stack.current
        assert target is not None
        self.snapshot = target.save_snapshot()
        target.autosave_dirty = True
        # Draw stamp at each nodes location 
        render_stamp_to_layer(target,
                self._stamp, self.nodes, self.bbox)

    def undo(self):
        layers = self.doc.layer_stack
        assert self.snapshot is not None
        layers.current.load_snapshot(self.snapshot)
        self.snapshot = None





class StampMode (InkingMode, OncanvasEditMixin):

    ## Metadata properties

    ACTION_NAME = "StampMode"

    permitted_switch_actions = set(gui.mode.BUTTON_BINDING_ACTIONS).union([
            'RotateViewMode',
            'ZoomViewMode',
            'PanViewMode',
            'SelectionMode',
        ])

    ## Metadata methods

    @classmethod
    def get_name(cls):
        return _(u"Stamp")

    def get_usage(self):
        return _(u"Place, and then adjust predefined stamps")

    @property
    def inactive_cursor(self):
        return None

    _OPTIONS_PRESENTER = None

    @property
    def options_presenter(self):
        """MVP presenter object for the node editor panel"""
        cls = self.__class__
        if cls._OPTIONS_PRESENTER is None:
            cls._OPTIONS_PRESENTER = OptionsPresenter_Stamp()
        return cls._OPTIONS_PRESENTER

   #@property
   #def active_cursor(self):
   #    if self.phase == _Phase.ADJUST:
   #        if self.zone == _EditZone.CONTROL_NODE:
   #            return self._crosshair_cursor
   #        elif self.zone != _EditZone.EMPTY_CANVAS: # assume button
   #            return self._arrow_cursor
   #
   #    elif self.phase in (_Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT):
   #        if self.zone == _EditZone.CONTROL_NODE:
   #            return self._cursor_move_nw_se
   #


    ## Class config vars


    ## Class variable & objects

    _stamp = None

    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):
        super(StampMode, self).__init__(**kwargs)
        self.current_handle_index = -1
        self.forced_button_pos = False

        self.scaleval=1.0

    def _reset_adjust_data(self):
        super(StampMode, self)._reset_adjust_data()
        self.target_area_index = None
        self.target_area_handle = None
        self.show_area_trash_button = False

    @property
    def stamp(self):
        return StampMode._stamp

    @property
    def target_area_index(self):
        return self._target_area_index

    @target_area_index.setter
    def target_area_index(self, index):
        self._target_area_index = index
       #self.current_node_index = None
       #self.target_node_index = None



    def set_stamp(self, stamp):
        """ Called from OptionPresenter, 
        This is to make stamp property as if it is read-only.
        """
        old_stamp = StampMode._stamp
        if old_stamp:
            old_stamp.leave(self.doc)
        StampMode._stamp = stamp
        stamp.enter(self.doc)
        stamp.initialize_phase(self)

    ## Status methods

    def enter(self, doc, **kwds):
        """Enters the mode: called by `ModeStack.push()` etc."""
        self._blank_cursor = doc.app.cursors.get_action_cursor(
            self.ACTION_NAME,
            gui.cursor.Name.ADD
        )
        self.options_presenter.target = (self, None)
        if self.stamp:
            self.stamp.initialize_phase(self)
        super(StampMode, self).enter(doc, **kwds)

    def leave(self, **kwds):
        """Leaves the mode: called by `ModeStack.pop()` etc."""
        super(StampMode, self).leave(**kwds)  # supercall will commit

    def checkpoint(self, flush=True, **kwargs):
        """Sync pending changes from (and to) the model

        If called with flush==False, this is an override which just
        redraws the pending stroke with the current brush settings and
        color. This is the behavior our testers expect:
        https://github.com/mypaint/mypaint/issues/226

        When this mode is left for another mode (see `leave()`), the
        pending brushwork is committed properly.

        """

        # FIXME almost copyied from polyfilltool.py
        if flush:
            # Commit the pending work normally
            super(InkingMode, self).checkpoint(flush=flush, **kwargs) # call super-superclass method
        else:
            # Queue a re-rendering with any new brush data
            # No supercall
            self._stop_task_queue_runner(complete=False)
            self._queue_draw_buttons()
            self._queue_redraw_all_nodes()
            self._queue_redraw_curve()
        
    def _commit_all(self):
        # We need that the target layer(current layer)
        # has surface and not locked. 
        bbox = self._stamp.get_bbox(None, self.nodes[0])
        if bbox:
            sx, sy, w, h = bbox

            ex = sx + w
            ey = sy + h
            for cn in self.nodes[1:]:
                tsx, tsy, tw, th = self._stamp.get_bbox(None, cn)
                sx = min(sx, tsx)
                sy = min(sy, tsy)
                ex = max(ex, tsx + tw)
                ey = max(ey, tsy + th)

            if hasattr(self._stamp, 'pixbuf'):
                # This means 'Current stamp is dynamic'. 
                # Therefore we need save its current content 
                # during draw command exist.
                stamp = ProxyStamp('', self._stamp.pixbuf)
            else:
                stamp = self._stamp


            cmd = DrawStamp(self.doc.model,
                    stamp,
                    self.nodes,
                    (sx, sy, abs(ex-sx)+1, abs(ey-sy)+1))
            self.doc.model.do(cmd)
        else:
            logger.warning("stamptool.commit_all encounter enpty bbox")

    def _start_new_capture_phase(self, rollback=False):
        """Let the user capture a new ink stroke"""
        if rollback:
            self._stop_task_queue_runner(complete=False)
            # Currently, this tool uses overlay(cairo) to preview stamps.
            # so we need no rollback action to the current layer.
        else:
            self._stop_task_queue_runner(complete=True)
            self._commit_all()

        if self.stamp:
            self.stamp.finalize_phase(self)
            
        self.options_presenter.target = (self, None)
        self._queue_redraw_curve(force_margin=True)  # call this before reset node
        self._queue_draw_buttons()
        self._queue_redraw_all_nodes()
        self._reset_nodes()
        self._reset_capture_data()
        self._reset_adjust_data()
        self.phase = _Phase.CAPTURE

        if self.stamp:
            self.stamp.initialize_phase(self)

    def _ensure_overlay_for_tdw(self, tdw):
        overlay = self._overlays.get(tdw)
        if not overlay:
            overlay = Overlay_Stamp(self, tdw)
            tdw.display_overlays.append(overlay)
            self._overlays[tdw] = overlay
        return overlay


    def _update_zone_and_target(self, tdw, x, y):
        """Update the zone and target node under a cursor position"""
        if not self._stamp:
            return
        else:
            stamp = self._stamp

        self._ensure_overlay_for_tdw(tdw)
        new_zone = _EditZone.EMPTY_CANVAS

        if not self.in_drag:
            if self.phase in (_Phase.ADJUST, _Phase.CAPTURE):

                new_target_node_index = None
                new_target_area_index = None
                new_target_area_handle = None
                # Test buttons for hits
                overlay = self._ensure_overlay_for_tdw(tdw)
                hit_dist = gui.style.FLOATING_BUTTON_RADIUS
               #button_info = [
               #    (_EditZone.ACCEPT_BUTTON, overlay.accept_button_pos),
               #    (_EditZone.REJECT_BUTTON, overlay.reject_button_pos),
               #]
               #for btn_zone, btn_pos in button_info:
                for btn_zone in (_EditZone.ACCEPT_BUTTON, 
                                _EditZone.REJECT_BUTTON):
                    btn_pos = overlay.get_button_pos(btn_zone)
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
                    new_target_node_index, control_node_idx = \
                            self._search_target_node(tdw, x, y)
                    if new_target_node_index != None:
                        if  0<= control_node_idx <= 3:
                            new_zone = _EditZone.CONTROL_HANDLE_BASE + \
                                        control_node_idx
                        else:
                            new_zone = _EditZone.CONTROL_NODE

                        if new_zone != self.zone:
                            self._queue_draw_node(new_target_node_index)

                        self.target_area_index = None
                        self.target_area_handle = None

                    elif stamp.is_support_selection:
                        margin = gui.style.DRAGGABLE_POINT_HANDLE_SIZE 
                        for i, area in stamp.enum_visible_selection_areas(tdw):
                            for  t, tx, ty in enum_area_point(*area):
                                if (tx-margin <= x <= tx+margin and 
                                        ty-margin <= y <= ty+margin):
                                    new_zone = _EditZone.SOURCE_AREA_HANDLE
                                    new_target_area_handle = t
                                    new_target_area_index = i
                                    break

                            # If 'handle-check' failed, but cursor might be 
                            # on the source-targetting rect.
                            if new_zone == _EditZone.EMPTY_CANVAS:
                                sx, sy, ex, ey = area
                                hit_dist = gui.style.FLOATING_BUTTON_RADIUS / 2

                                if sx <= x <= ex and sy <= y <= ey:
                                    new_zone = _EditZone.SOURCE_AREA
                                    new_target_area_index = i

                                    if self.show_area_trash_button:
                                        btn_x = sx + (ex - sx) / 2
                                        btn_y = sy + (ey - sy) / 2
                                        d = math.hypot(btn_x - x, btn_y - y)
                                        if d <= hit_dist:
                                            new_zone = _EditZone.SOURCE_TRASH_BUTTON

                            # Check zone again.
                            # Either cursor is on handle or on rect,
                            # loop should be broken.
                            if new_zone != _EditZone.EMPTY_CANVAS:
                                break

                        if (new_target_area_index != self.target_area_index or
                                new_target_area_handle != self.target_area_handle):
                            self._queue_selection_area(tdw, 
                                    indexes=(new_target_area_index, 
                                        self.target_area_index))
                            self.target_area_index = new_target_area_index
                            self.target_area_handle = new_target_area_handle


                # Update the prelit node, and draw changes to it
                if new_target_node_index != self.target_node_index:
                    if self.target_node_index is not None:
                        self._queue_draw_node(self.target_node_index)
                    self.target_node_index = new_target_node_index
                    if self.target_node_index is not None:
                        self._queue_draw_node(self.target_node_index)



        # Update the zone, and assume any change implies a button state
        # change as well (for now...)
        if self.zone != new_zone:
            self.zone = new_zone
            self._ensure_overlay_for_tdw(tdw)
            self._queue_draw_buttons()
            self._queue_selection_area(tdw)
        # Update the "real" inactive cursor too:
        if not self.in_drag:
            cursor = None
            if self.phase in (_Phase.ADJUST, _Phase.CAPTURE):
                if self.zone == _EditZone.CONTROL_NODE:
                    cursor = self._crosshair_cursor
                elif self.zone != _EditZone.EMPTY_CANVAS: # assume button
                    cursor = self._arrow_cursor
                else:
                    cursor = self._blank_cursor
            else:
                cursor = self._blank_cursor

            if cursor is not self._current_override_cursor:
                tdw.set_override_cursor(cursor)
                self._current_override_cursor = cursor

    def _notify_stamp_changed(self):
        """
        Common processing stamp changed,
        or target area added/removed
        """
        for tdw in self._overlays:
            self._queue_selection_area(tdw)

        self.options_presenter.refresh_tile_count()

    def select_area_cb(self, selection_mode):
        """ Selection handler called from SelectionMode.
        This handler never called when no selection executed.

        CAUTION: you can not access the self.doc attribute here
        (it is disabled as None, with modestack facility)
        so you must use 'selection_mode.doc', instead of it.
        """
        if self.phase in (_Phase.CAPTURE, _Phase.ADJUST):
            if self.stamp and self.stamp.is_support_selection:
                self.stamp.set_selection_area(-1,
                        selection_mode.get_min_max_pos_model(margin=0),
                        selection_mode.doc.model.layer_stack.current)

                self.stamp.initialize_phase(self)
                self._notify_stamp_changed()
        else:
            # Ordinary selection, which means 'node selection'
            modified = False
            for idx,cn in enumerate(self.nodes):
                if selection_mode.is_inside_model(cn.x, cn.y):
                    if not idx in self.selected_nodes:
                        self.selected_nodes.append(idx)
                        modified = True
            if modified:
                self._queue_redraw_all_nodes()

    ## Redraws

    def _search_target_node(self, tdw, x, y):
        """ utility method: to commonize processing,
        even in inherited classes.
        """
        hit_dist = gui.style.DRAGGABLE_POINT_HANDLE_SIZE + 12
        new_target_node_index = None
        handle_idx = -1
        stamp = self._stamp
       #mx, my = tdw.display_to_model(x, y)
        for i, node in reversed(list(enumerate(self.nodes))):
            handle_idx = stamp.get_handle_index(tdw, x, y, node,
                   gui.style.DRAGGABLE_POINT_HANDLE_SIZE)
            if handle_idx >= 0:
                new_target_node_index = i
                if handle_idx >= 4:
                    handle_idx = -1
                break
        return new_target_node_index, handle_idx

    def _queue_draw_node(self, i, force_margin=False):
        """Redraws a specific control node on all known view TDWs"""
        if i in self.selected_nodes:
            dx,dy = self.drag_offset.get_model_offset()
        else:
            dx = dy = 0.0

        if i == self.target_node_index:
            force_margin = True

        for tdw in self._overlays:
            self._queue_draw_node_internal(tdw, self.nodes[i], dx, dy, force_margin)


    def _queue_draw_node_internal(self, tdw, node, dx, dy, add_margin):
        if not self._stamp:
            return

        if add_margin:
            margin = gui.style.DRAGGABLE_POINT_HANDLE_SIZE + 4
        else:
            margin = 4

        bbox = self._stamp.get_bbox(tdw, node, dx, dy, margin=margin)

        if bbox:
            tdw.queue_draw_area(*bbox)

    def _queue_redraw_curve(self, force_margin=False):
        """Redraws the entire curve on all known view TDWs"""
        dx, dy = self.drag_offset.get_model_offset()
        
        for tdw in self._overlays:
            for i, cn in enumerate(self.nodes):
                targetted = (i == self.current_node_index)
                if i in self.selected_nodes:
                    self._queue_draw_node_internal(tdw, cn, dx, dy, 
                            targetted or force_margin)
                else:
                    self._queue_draw_node_internal(tdw, cn, 0.0, 0.0, 
                            targetted or force_margin)

            self._queue_selection_area(tdw)

    def _queue_selection_area(self, tdw, indexes=None):
        dx, dy = self.drag_offset.get_model_offset()
        stamp = self.stamp

        if stamp and stamp.is_support_selection:
            for i, junk in stamp.enum_visible_selection_areas(tdw, indexes=indexes):
                area = stamp.get_selection_area(i)
                area = self.adjust_selection_area(i, area)
                sx, sy, ex, ey = gui.ui_utils.get_outmost_area(tdw, *area, 
                        margin=gui.style.DRAGGABLE_POINT_HANDLE_SIZE+4)
                tdw.queue_draw_area(sx, sy, 
                        abs(ex - sx) + 1, abs(ey - sy) + 1)

    ## Raw event handling (prelight & zone selection in adjust phase)

    def button_press_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False

        if not self._stamp or not self._stamp.is_ready:
            return super(InkingMode, self).button_press_cb(tdw, event)

        self._update_zone_and_target(tdw, event.x, event.y)
        self._update_current_node_index()
        self.drag_offset.reset()


        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
        ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK

        if self.phase in (_Phase.ADJUST, _Phase.CAPTURE):
            button = event.button
            # Normal ADJUST/ADJUST_PRESSURE Phase.

            if self.zone in (_EditZone.REJECT_BUTTON,
                             _EditZone.ACCEPT_BUTTON,
                             _EditZone.SOURCE_TRASH_BUTTON):
                if (button == 1 and
                        event.type == Gdk.EventType.BUTTON_PRESS):
                    self._click_info = (button, self.zone)
                    return False
                # FALLTHRU: *do* allow drags to start with other buttons
            elif self.zone == _EditZone.EMPTY_CANVAS:
                self.phase = _Phase.CAPTURE
                self._queue_draw_buttons() # To erase button!
                self._queue_redraw_curve()

                # FALLTHRU: *do* start a drag
            elif self.zone == _EditZone.CONTROL_NODE:
                # clicked a node.

                if button == 1:
                    # 'do_reset' is a selection reset flag
                    # With clicking node with holding shift/ctrl key,
                    # we can select/cancel the node.
                    # but that routine is placed at button_release_cb()
                    # at here, we confirm whether selecting nodes or not.
                    do_reset = False
                    self.phase = _Phase.ADJUST
                     
                    if not (shift_state or ctrl_state):
                        do_reset = ((event.state & Gdk.ModifierType.MOD1_MASK) != 0)
                        do_reset |= not (self.current_node_index in self.selected_nodes)

                    if do_reset:
                        # To avoid old selected nodes still lit.
                        self._queue_draw_selected_nodes()
                        self._reset_selected_nodes(self.current_node_index)

                # FALLTHRU: *do* start a drag

            elif (_EditZone.CONTROL_HANDLE_0 <= 
                    self.zone <= _EditZone.CONTROL_HANDLE_3):
                if button == 1:

                    self.current_handle_index = \
                            self.zone - _EditZone.CONTROL_HANDLE_BASE

                    if ctrl_state:
                        self.phase = _PhaseStamp.ROTATE_BY_HANDLE
                    else:
                        self.phase = _PhaseStamp.SCALE_BY_HANDLE
            elif self.zone in (_EditZone.SOURCE_AREA,
                               _EditZone.SOURCE_AREA_HANDLE):
                if button == 1:
                    if self.zone == _EditZone.SOURCE_AREA:
                        self.phase = _PhaseStamp.ADJUST_SOURCE
                    else:
                        self.phase = _PhaseStamp.ADJUST_SOURCE_BY_HANDLE
            else:
                raise NotImplementedError("Unrecognized zone %r", self.zone)


        elif self.phase in (_PhaseStamp.ROTATE,
                            _PhaseStamp.SCALE,
                            _PhaseStamp.ROTATE_BY_HANDLE,
                            _PhaseStamp.SCALE_BY_HANDLE):
            # XXX Not sure what to do here.
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

        if not self._stamp or not self._stamp.is_ready:
            return super(InkingMode, self).button_release_cb(tdw, event)

        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
        ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK

        if self.phase in (_Phase.ADJUST, _Phase.CAPTURE):
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
                        elif zone0 == _EditZone.SOURCE_TRASH_BUTTON:
                            assert self.stamp.is_support_selection
                            assert self.target_area_handle == None
                            self._queue_selection_area(tdw)
                            self.stamp.remove_selection_area(self.target_area_index)
                            self.target_area_index = None

                    self._click_info = None
                    self._update_zone_and_target(tdw, event.x, event.y)
                    self._update_current_node_index()
                    return False
            else:
                # Clicked node and button released.
                # Add or Remove selected node
                # when control key is pressed
                if event.button == 1:
                    if ctrl_state:
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
        
        # Update workaround state for evdev dropouts
        self._button_down = None


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

        if self._stamp and self._stamp.is_ready:
            shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
            ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK
            self._update_zone_and_target(tdw, event.x, event.y)
            prev_state = self.show_area_trash_button

            if not self.in_drag:
                if self.phase in (_Phase.CAPTURE, _Phase.ADJUST):
                    if (self._stamp != None and 
                            self.stamp.is_support_selection and
                            shift_state):
                        
                        self.show_area_trash_button = \
                                (self.zone in (_EditZone.SOURCE_AREA,
                                               _EditZone.SOURCE_TRASH_BUTTON))
                    else:
                        self.show_area_trash_button = False

            if prev_state != self.show_area_trash_button:
                self._queue_selection_area(tdw)
                    

        # call super-superclass callback
        return super(InkingMode, self).motion_notify_cb(tdw, event)

    ## Drag handling (both capture and adjust phases)

    def drag_start_cb(self, tdw, event):
        if not self._stamp or not self._stamp.is_ready:
            super(InkingMode, self).drag_start_cb(tdw, event)

        self._ensure_overlay_for_tdw(tdw)
        mx, my = tdw.display_to_model(event.x, event.y)
        if self.phase == _Phase.CAPTURE:

            if event.state != 0:
                # here when something go wrong,and cancelled.
                self.current_node_index = None
                return super(InkingMode, self).drag_start_cb(tdw, event)
            elif self.stamp.tile_count > 0:
                node = _StampNode(mx, my, 
                        self._stamp.default_angle,
                        self._stamp.default_scale_x,
                        self._stamp.default_scale_y,
                        self._stamp.latest_tile_index)
                self.nodes.append(node)
                self.target_node_index = len(self.nodes) -1
                self._update_current_node_index()
                self.selected_nodes = [self.target_node_index, ]
                self._queue_draw_node(0)
                self.drag_offset.start(mx, my)

        elif self.phase == _Phase.ADJUST:
            self._node_dragged = False
            if self.target_node_index is not None:
                node = self.nodes[self.target_node_index]
                self._dragged_node_start_pos = (node.x, node.y)

                # Use selection_rect class as offset-information
                self.drag_offset.start(mx, my)

        elif self.phase in (_Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT):
            pass
        elif self.phase == _Phase.CHANGE_PHASE:
            pass
        elif self.phase in (_PhaseStamp.ROTATE,
                            _PhaseStamp.SCALE,
                            _PhaseStamp.ROTATE_BY_HANDLE,
                            _PhaseStamp.SCALE_BY_HANDLE):
            pass
        elif self.phase in (_PhaseStamp.ADJUST_SOURCE,
                            _PhaseStamp.ADJUST_SOURCE_BY_HANDLE):
            self.drag_offset.start(mx, my)
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)


    def drag_update_cb(self, tdw, event, dx, dy):
        if not self._stamp or not self._stamp.is_ready:
            super(InkingMode, self).drag_update_cb(tdw, event, dx, dy)

        self._ensure_overlay_for_tdw(tdw)

        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
        ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK
        mx, my = tdw.display_to_model(event.x ,event.y)

        def override_scale_and_rotate():
            if ctrl_state:
                self.phase = _PhaseStamp.ROTATE
            else:
                self.phase = _PhaseStamp.SCALE
            # Re-enter drag operation again
            self.drag_update_cb(tdw, event, dx, dy)
            self.phase = _Phase.CAPTURE
            self._queue_draw_node(self.current_node_index) 

        if self.phase == _Phase.CAPTURE:
            if self.current_node_index != None:
                self._queue_draw_node(self.current_node_index) 
                if shift_state:
                    override_scale_and_rotate()
                else:
                    self.drag_offset.end(mx, my)
                    self._queue_draw_node(len(self.nodes)-1) 
        elif self.phase == _Phase.ADJUST:
            if shift_state:
                override_scale_and_rotate()
            else:
                self._queue_redraw_curve()
                super(StampMode, self).drag_update_cb(tdw, event, dx, dy)
        elif self.phase in (_PhaseStamp.SCALE,
                            _PhaseStamp.ROTATE):
            assert self.current_node_index is not None
            self._queue_redraw_curve()
            node = self.nodes[self.current_node_index]
            bx, by = tdw.model_to_display(node.x, node.y)
            dir, length = get_drag_direction(
                    self.start_x, self.start_y,
                    event.x, event.y)

            if dir >= 0:
                if self.phase == _PhaseStamp.ROTATE:
                    rad = length * 0.005
                    if dir in (0, 3):
                        rad *= -1
                    node = node._replace(angle = node.angle + rad)
                else:
                    scale = length * 0.005
                    if dir in (0, 3):
                        scale *= -1
                    node = node._replace(scale_x = node.scale_x + scale,
                            scale_y = node.scale_y + scale)

                self.nodes[self.current_node_index] = node
                self._queue_redraw_curve()
                self.start_x = event.x
                self.start_y = event.y
        elif self.phase == _PhaseStamp.SCALE_BY_HANDLE:
            assert self.target_node_index is not None
            self._queue_redraw_curve()
            node = self.nodes[self.target_node_index]
            pos = self._stamp.get_boundary_points(node)

            # At here, we consider the movement of control handle(i.e. cursor)
            # as a Triangle from origin.

            # 1. Get new(=cursor position) vector from origin(node.x,node.y)
            mx, my = tdw.display_to_model(event.x, event.y)
            length, nx, ny = length_and_normal(
                    node.x, node.y,
                    mx, my)

            bx = mx - node.x 
            by = my - node.y

            orig_pos = self._stamp.get_boundary_points(node, 
                    no_scale=True)

            ti = self.current_handle_index
            nlen, nx, ny = length_and_normal(node.x, node.y, mx, my)

            # get original side and top ridge length
            # and its identity vector
            si,ei = (2, 1) if ti in (0, 1) else (1, 2)
            side_length, snx, sny = length_and_normal(
                    orig_pos[si][0], orig_pos[si][1],
                    orig_pos[ei][0], orig_pos[ei][1])

            si,ei = (0, 1) if ti in (1, 2) else (1, 0)
            top_length, tnx, tny = length_and_normal(
                    orig_pos[si][0], orig_pos[si][1],
                    orig_pos[ei][0], orig_pos[ei][1])

            # get the 'leg' of new vectors
            dp = dot_product(snx, sny, bx, by)
            vx = dp * snx 
            vy = dp * sny
            v_length = vector_length(vx, vy) * 2
            
            # 4. Then, get another leg of triangle.
            hx = bx - vx
            hy = by - vy
            h_length = vector_length(hx, hy) * 2

            scale_x = h_length / top_length 
            scale_y = v_length / side_length 

            # Also, scaling might be inverted(mirrored).
            # it can detect from 'psuedo' cross product
            # between side and top vector.
            cp = cross_product(tnx, tny, nx, ny)
            if ((ti in (1, 3) and cp > 0.0)
                    or (ti in (0, 2) and cp < 0.0)):
                scale_y = -scale_y

            cp = cross_product(snx, sny, nx, ny)
            if ((ti in (1, 3) and cp < 0.0)
                    or (ti in (0, 2) and cp > 0.0)):
                scale_x = -scale_x

            self.nodes[self.target_node_index] = node._replace(
                    scale_x=scale_x,
                    scale_y=scale_y)

            self._queue_redraw_curve()

        elif self.phase == _PhaseStamp.ROTATE_BY_HANDLE:
            assert self.target_node_index is not None
            self._queue_redraw_curve()
            node = self.nodes[self.target_node_index]
            #pos = self._stamp.get_boundary_points(node)
            #mx, my = tdw.display_to_model(event.x, event.y)
            ndx, ndy = tdw.model_to_display(node.x, node.y)
            junk, bx, by = length_and_normal(
                    self.last_x, self.last_y,
                    ndx, ndy)                    
            
            junk, cx, cy = length_and_normal(
                    event.x, event.y,
                    ndx, ndy)
                    

            rad = get_radian(cx, cy, bx, by)

            # 3. Get a cross product of them to
            # identify which direction user want to rotate.
            if cross_product(cx, cy, bx, by) >= 0.0:
                rad = -rad

            self.nodes[self.target_node_index] = node._replace(
                    angle = node.angle + rad)
                      
            self._queue_redraw_curve()
            
        elif self.phase in (_PhaseStamp.ADJUST_SOURCE,
                            _PhaseStamp.ADJUST_SOURCE_BY_HANDLE):

            self._queue_selection_area(tdw)
            self.drag_offset.end(mx, my)
            self._queue_selection_area(tdw)
        else:
            super(StampMode, self).drag_update_cb(tdw, event, dx, dy)


    def drag_stop_cb(self, tdw):
        if not self._stamp or not self._stamp.is_ready:
            return super(InkingMode, self).drag_stop_cb(tdw)
        

        self._ensure_overlay_for_tdw(tdw)
        if self.phase == _Phase.CAPTURE:

            if not self.nodes or self.current_node_index == None:
                # Cancelled drag event (and current capture phase)
                # call super-superclass directly to bypass this phase
                self._reset_capture_data()
                self._reset_adjust_data()
                return super(InkingMode, self).drag_stop_cb(tdw) 

            node = self.nodes[self.current_node_index]
            dx, dy = self.drag_offset.get_model_offset()
            self.nodes[self.current_node_index] = \
                    node._replace( x=node.x + dx, y=node.y + dy)
            self.drag_offset.reset()

            if len(self.nodes) > 0:
                self.phase = _Phase.ADJUST
                self._update_zone_and_target(tdw, self.last_x, self.last_y)
                self._queue_redraw_all_nodes()
                self._queue_redraw_curve()
                self._queue_draw_buttons()
            else:
                self._reset_nodes()
                tdw.queue_draw()

        elif self.phase == _Phase.ADJUST:
            super(StampMode, self).drag_stop_cb(tdw)
            self._queue_redraw_all_nodes()
            self._queue_redraw_curve()
            self._queue_draw_buttons()

        elif self.phase in (_PhaseStamp.ROTATE,
                            _PhaseStamp.SCALE,
                            _PhaseStamp.ROTATE_BY_HANDLE,
                            _PhaseStamp.SCALE_BY_HANDLE):
            self.phase = _Phase.ADJUST

        elif self.phase in (_PhaseStamp.ADJUST_SOURCE,
                            _PhaseStamp.ADJUST_SOURCE_BY_HANDLE):
            area = self.adjust_selection_area(
                            self.target_area_index,
                            self.stamp.get_selection_area(self.target_area_index))
            self.stamp.set_selection_area(self.target_area_index, area,
                    self.doc.model.layer_stack.current)
            self.stamp.refresh_surface(self.target_area_index)

            self.phase = _Phase.ADJUST
            self._queue_selection_area(tdw)
            self._reset_adjust_data()
        else:
            return super(StampMode, self).drag_stop_cb(tdw)

    def scroll_cb(self, tdw, event):
        """Handles scroll-wheel events, to adjust rotation/scale/tile_index."""

        if (self.phase in (_Phase.ADJUST,) 
                and self.zone == _EditZone.CONTROL_NODE
                and self.current_node_index != None):

            if self._prev_scroll_time != event.time:
                shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
                ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK
                junk, step = get_scroll_delta(event, 1.0)
                node = self.nodes[self.current_node_index]
                redraw = True

                if shift_state:
                    self._queue_draw_node(self.current_node_index) 

                    scale_step = 0.05
                    node = node._replace(scale_x = node.scale_x + step * scale_step,
                            scale_y = node.scale_y + step * scale_step)
                    self.nodes[self.current_node_index] = node
                elif ctrl_state:
                    self._queue_draw_node(self.current_node_index) 

                    node = node._replace(angle = node.angle + step * (math.pi * 0.05))
                    self.nodes[self.current_node_index] = node
                else:
                    if self.stamp and self.stamp.tile_count > 1:
                        self._queue_draw_node(self.current_node_index) 

                        new_tile_index = lib.helpers.clamp(node.tile_index + int(step),
                                0, self.stamp.tile_count - 1)

                        if new_tile_index != node.tile_index:
                            self.nodes[self.current_node_index] = \
                                    node._replace(tile_index = new_tile_index)
                    else:
                        redraw = False

                if redraw:
                    self._queue_draw_node(self.current_node_index) 

            self._prev_scroll_time = event.time
        else:
            # calling super-supreclass handler, to invoke original scroll events.
            return super(InkingMode, self).scroll_cb(tdw, event)


    ## Interrogating events

    def stamp_tile_deleted_cb(self, tile_index):
        """
        A notification callback when stamp tile has been changed
        (deleted)
        """

        # First of all, queue redraw the nodes to be deleted.
        for i, cn in enumerate(self.nodes):
            if cn.tile_index == tile_index:
                self._queue_draw_node(i, force_margin=True) 

        # After that, delete it.
        for i, cn in enumerate(self.nodes[:]):
            if cn.tile_index == tile_index:
                self.nodes.remove(cn)
                if i in self.selected_nodes:
                    self.selected_nodes.remove(i)
                if i == self.current_node_index:
                    self.current_node_index = None
                if i == self.target_node_index:
                    self.target_node_index = None

        if len(self.nodes) == 0:
            self.phase == _Phase.CAPTURE
            logger.info('stamp tile deleted, and all nodes deleted')

    ## Node editing
    def update_node(self, i, **kwargs):
        self._queue_draw_node(i, force_margin=True) 
        self.nodes[i] = self.nodes[i]._replace(**kwargs)
        self._queue_draw_node(i, force_margin=True) 

    ## Utility methods
    def adjust_selection_area(self, index, area):
        """
        Adjust selection(source-target) area
        against user interaction(i.e. dragging selection area).

        CAUTION: THIS METHOD ACCEPTS ONLY MODEL COORDINATE.

        :param index: area index of LayerStamp source area
        :param area: area, i.e. a tuple of (sx, sy, ex, ey)
                     this is model coordinate.
        """
        if self.target_area_index == index:
            dx, dy = self.drag_offset.get_model_offset()
            sx, sy, ex, ey = area 

            if self.target_area_handle == None:
                return (sx+dx, sy+dy, ex+dx, ey+dy)
            else:
                if self.target_area_handle in (0, 3):
                    sx += dx
                else:
                    ex += dx

                if self.target_area_handle in (0, 1):
                    sy += dy
                else:
                    ey += dy

                return (sx, sy, ex, ey)

        return area

    def delete_selected_nodes(self):
        self._queue_draw_buttons()
        for idx in self.selected_nodes:
            self._queue_draw_node(idx)

        new_nodes = []
        for idx,cn in enumerate(self.nodes):
            if not idx in self.selected_nodes:
                new_nodes.append(cn)

        self.nodes = new_nodes
        self._reset_selected_nodes()
        self.current_node_index = None
        self.target_node_index = None

        self._queue_redraw_curve()
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()




class Overlay_Stamp (OverlayOncanvasMixin):
    """Overlay for an StampMode's adjustable points"""

    SELECTED_COLOR = \
            RGBColor(color=gui.style.ACTIVE_ITEM_COLOR).get_rgb()
    SELECTED_AREA_COLOR = \
            RGBColor(color=gui.style.POSTLIT_ITEM_COLOR).get_rgb()


    def __init__(self, mode, tdw):
        super(Overlay_Stamp, self).__init__(mode, tdw)


    def _get_onscreen_nodes(self):
        """Iterates across only the on-screen nodes."""
        mode = self._mode
        alloc = self._tdw.get_allocation()
        dx,dy = mode.drag_offset.get_model_offset()
        for i, node in enumerate(mode.nodes):

            if i in mode.selected_nodes:
                tx = dx
                ty = dy
            else:
                tx = ty = 0.0
            bbox = mode.stamp.get_bbox(self._tdw, node, tx, ty)

            if bbox:
                x, y, w, h = bbox
                node_on_screen = (
                    x > alloc.x - w and
                    y > alloc.y - h and
                    x < alloc.x + alloc.width + w and
                    y < alloc.y + alloc.height + h
                )

                if node_on_screen:
                    yield (i, node)

    def update_button_positions(self):
        """Recalculates the positions of the mode's buttons."""

        mode = self._mode
        nodes = mode.nodes
        num_nodes = len(nodes)
        if num_nodes == 0:
            self._button_pos[_EditZone.REJECT_BUTTON] = None
            self._button_pos[_EditZone.ACCEPT_BUTTON] = None
            return False

        button_radius = gui.style.FLOATING_BUTTON_RADIUS
        margin = 1.5 * button_radius
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
            area_radius = 64 + margin #gui.style.FLOATING_TOOL_RADIUS

            cx, cy = adjust_button_inside(cx, cy, area_radius)

            pos_list = []
            count = 2
            for i in range(count):
                rad = (math.pi / count) * 2.0 * i
                x = - area_radius*math.sin(rad)
                y = area_radius*math.cos(rad)
                pos_list.append( (x + cx, - y + cy) )

            self._button_pos[_EditZone.ACCEPT_BUTTON] = pos_list[0][0], pos_list[0][1]
            self._button_pos[_EditZone.REJECT_BUTTON] = pos_list[1][0], pos_list[1][1]
        else:
            # Usually, these tool needs to keep extending control points.
            # So when buttons placed around the tail(newest) node, 
            # it is something frastrating to manipulate new node...
            # Thus,different to Inktool, place buttons around 
            # the first(oldest) nodes.
            
            node = nodes[0]
            cx, cy = self._tdw.model_to_display(node.x, node.y)

            nx = cx + button_radius
            ny = cx - button_radius

            vx = nx-cx
            vy = ny-cy
            s  = math.hypot(vx, vy)
            if s > 0.0:
                vx /= s
                vy /= s
            else:
                pass

            margin = 4.0 * button_radius
            dx = vx * margin
            dy = vy * margin
            
            self._button_pos[_EditZone.ACCEPT_BUTTON] = adjust_button_inside(
                    cx + dy, cy - dx, button_radius * 1.5)
            self._button_pos[_EditZone.REJECT_BUTTON] = adjust_button_inside(
                    cx - dy, cy + dx, button_radius * 1.5)

        return True


    def draw_stamp(self, cr, idx, node, dx, dy, colors):
        """ Draw a stamp as overlay preview.

        :param idx: index of node in mode.nodes[] 
        :param node: current node.this holds some information
                     such as stamp scaling ratio and rotation.
        :param dx, dy: display coordinate position of node.
        :param color: color of stamp rectangle
        """
        mode = self._mode
        pos = mode.stamp.get_boundary_points(node, tdw=self._tdw)
        x, y = self._tdw.model_to_display(node.x, node.y)
        normal_color, selected_color = colors

        if idx == mode.current_node_index or idx in mode.selected_nodes:
            self.draw_stamp_rect(cr, idx, dx, dy, selected_color, position=pos)
            x+=dx
            y+=dy
            for i, pt in enumerate(pos):
                handle_idx = _EditZone.CONTROL_HANDLE_BASE + i
                gui.drawutils.render_square_floating_color_chip(
                    cr, pt[0] + dx, pt[1] + dy,
                    gui.style.ACTIVE_ITEM_COLOR, 
                    gui.style.DRAGGABLE_POINT_HANDLE_SIZE,
                    fill=(handle_idx==mode.zone)) 
        else:
            self.draw_stamp_rect(cr, idx, 0, 0, normal_color, position=pos)

        mode.stamp.draw(self._tdw, cr, x, y, node, True)

    def draw_stamp_rect(self, cr, idx, dx, dy, color, position=None):
        cr.save()
        mode = self._mode
        cr.set_line_width(1)

        cr.set_source_rgb(0, 0, 0)

        if not position:
            position = mode.stamp.get_boundary_points(
                    mode.nodes[idx], tdw=self._tdw)

        cr.move_to(position[0][0] + dx,
                position[0][1] + dy)
        for lx, ly in position[1:]:
            cr.line_to(lx+dx, ly+dy)
        cr.close_path()
        cr.stroke_preserve()

        cr.set_dash( (3.0, ) )
        cr.set_source_rgb(*color)
        cr.stroke()
        cr.restore()

    def draw_selection_area(self, cr, dx, dy, right_color, other_color):
        """ Drawing LayerStamp's source-target rectangle.
        """
        cr.save()
        cr.set_line_width(1)
        tdw = self._tdw
        mode = self._mode
        icon_pixbuf = None
        current_layer = tdw.doc.layer_stack.current
        assert mode.stamp and hasattr(mode.stamp, "enum_visible_selection_areas")

        for i, junk in mode.stamp.enum_visible_selection_areas(tdw): 

            # To modify some corners of selection rectangle,
            # use StampMode.adjust_selection_area() method.
            # that method accepts only model coodinate area,
            # so we cannot use the yielded areas of 
            # enum_visible_selection_areas().
            area = mode.stamp.get_selection_area(i)
            sx, sy, ex, ey = mode.adjust_selection_area(i, area)
            sx, sy = tdw.model_to_display(sx, sy)
            ex, ey = tdw.model_to_display(ex, ey)

            # We MUST consider rotation, to draw rectangle

            # _get_onscreen_areas() returns display coordinate area
            # (with offseted one when user move it by dragging)
            # so use it. 
            #
            # passing None to tdw parameter here, because the area
            # is already in display coordinate.
            gui.drawutils.draw_rectangle_follow_canvas(cr, None,
                    sx, sy, ex, ey) 
            cr.set_dash((), 0)
            cr.set_source_rgb(0, 0, 0)
            cr.stroke_preserve()

            cr.set_dash( (3.0, ) )
            target_layer = mode.stamp.get_layer_for_area(i)
            if target_layer == current_layer:
                cr.set_source_rgb(*right_color)
            else:
                cr.set_source_rgb(*other_color)
            cr.stroke()

            if (i == mode.target_area_index):
               #and 
               #    mode.target_area_handle != None):

                for i, x, y in enum_area_point(sx, sy, ex, ey):
                    gui.drawutils.render_square_floating_color_chip(
                        cr, x, y,
                        gui.style.ACTIVE_ITEM_COLOR, 
                        gui.style.DRAGGABLE_POINT_HANDLE_SIZE,
                        fill=(i==mode.target_area_handle)) 

                if mode.show_area_trash_button:
                    if icon_pixbuf == None:
                        icon_pixbuf = self._get_button_pixbuf("mypaint-trash-symbolic")
                        radius = gui.style.FLOATING_BUTTON_RADIUS / 2

                    if mode.zone == _EditZone.SOURCE_TRASH_BUTTON:
                        btn_color = gui.style.ACTIVE_ITEM_COLOR
                    else:
                        btn_color = gui.style.EDITABLE_ITEM_COLOR

                    gui.drawutils.render_round_floating_button(
                        cr=cr, x=sx+abs(ex-sx)/2, y=sy+abs(ey-sy)/2,
                        color=btn_color,
                        pixbuf=icon_pixbuf,
                        radius=radius,
                    )

        cr.restore()

    def paint(self, cr):
        """Draw adjustable nodes to the screen"""

        mode = self._mode
        if mode.stamp == None:
            return

        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        alloc = self._tdw.get_allocation()
        dx,dy = mode.drag_offset.get_display_offset(self._tdw)
        fill_flag = not mode.phase in (_Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT)
        mode.stamp.initialize_draw(cr)

        colors = ( (1, 1, 1), self.SELECTED_COLOR)

        for i, node in self._get_onscreen_nodes():
           #color = gui.style.EDITABLE_ITEM_COLOR
            show_node = not mode.hide_nodes

            if show_node:
                self.draw_stamp(cr, i, node, dx, dy, colors)
            else:
                self.draw_stamp_rect(cr, i, node, dx, dy, colors)

        mode.stamp.finalize_draw(cr)

        # Selection areas
        if mode.stamp.is_support_selection:
            # TODO right_color and other_color should be
            # correctly configured at gui/style.py
            if mode.stamp.tile_count > 0:
                self.draw_selection_area(cr, 
                        dx, dy,
                        self.SELECTED_AREA_COLOR, (1, 0 ,0) )

        # Buttons
        if (mode.phase in (_Phase.ADJUST,)
                and
                len(mode.nodes) > 0 and
                not mode.in_drag):
            self.update_button_positions()
            button_info = [
                (
                    "mypaint-ok-symbolic",
                    self.get_button_pos(_EditZone.ACCEPT_BUTTON),
                    _EditZone.ACCEPT_BUTTON,
                ),
                (
                    "mypaint-trash-symbolic",
                    self.get_button_pos(_EditZone.REJECT_BUTTON),
                    _EditZone.REJECT_BUTTON,
                ),
            ]
            self._draw_buttons(cr, button_info)


class OptionsPresenter_Stamp (object):
    """Presents UI for directly editing point values etc."""

    variation_preset_store = None

    def __init__(self):
        super(OptionsPresenter_Stamp, self).__init__()
        from application import get_app
        self._app = get_app()
        self._options_grid = None
        self._point_values_grid = None
        self._angle_adj = None
        self._xscale_adj = None
        self._yscale_adj = None
        self._tile_adj = None
        self._tile_label = None
        self._tile_scale = None
        self._random_tile_button = None
        self._stamp_preset_view = None

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
        self._angle_adj = builder.get_object("angle_adj")
        self._xscale_adj = builder.get_object("xscale_adj")
        self._yscale_adj = builder.get_object("yscale_adj")
        self._tile_adj = builder.get_object("tile_adj")
        self._tile_scale = builder.get_object("tile_scale")
       #self._random_tile_button = builder.get_object("random_tile_button")
       #self._random_tile_button.set_sensitive(False)

        base_grid = builder.get_object("preset_editing_grid")
        self._init_stamp_preset_view(2, base_grid)

        base_grid = builder.get_object("additional_button_grid")
        self._init_toolbar(0, base_grid)

    def _init_stamp_preset_view(self, row, box):
        # XXX we'll need reconsider fixed value 
        # such as item width of 48 or icon size of 32 
        # in hidpi environment
        liststore = self._app.stamp_manager.initialize_icon_store()
        iconview = Gtk.IconView.new()
        iconview.set_model(liststore)
        iconview.set_pixbuf_column(0)
        iconview.set_text_column(1)
        iconview.set_item_width(48) 
        iconview.connect('selection-changed', self._iconview_item_changed_cb)
        self._stamps_store = liststore

        sw = Gtk.ScrolledWindow()
        sw.set_margin_top(4)
        sw.set_shadow_type(Gtk.ShadowType.ETCHED_IN)
        sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)            
        sw.set_hexpand(True)
        sw.set_vexpand(True)
        sw.set_halign(Gtk.Align.FILL)
        sw.set_valign(Gtk.Align.FILL)
        sw.add(iconview)
        box.attach(sw, 0, row, 2, 1)
        self.preset_view = iconview

    def _init_toolbar(self, row, box):
        toolbar = gui.widgets.inline_toolbar(
            self._app,
            [
                ("StampRandomize", "mypaint-up-symbolic"),
                ("DeleteItem", "mypaint-remove-symbolic"),
                ("AcceptEdit", "mypaint-ok-symbolic"),
                ("DiscardEdit", "mypaint-trash-symbolic"),
            ]
        )
        style = toolbar.get_style_context()
        style.set_junction_sides(Gtk.JunctionSides.TOP)
        box.attach(toolbar, 0, row, 1, 1)


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
        mode, cn_idx = targ
        mode_ref = None
        if mode:
            mode_ref = weakref.ref(mode)
        self._target = (mode_ref, cn_idx)
        # Update the UI
        if self._updating_ui:
            return

        self._updating_ui = True
        try:
            self._ensure_ui_populated()

            if 0 <= cn_idx < len(mode.nodes):
                cn = mode.nodes[cn_idx]
                self._angle_adj.set_value(
                        clamp(math.degrees(cn.angle),-180.0, 180.0))
                self._xscale_adj.set_value(cn.scale_x)
                self._yscale_adj.set_value(cn.scale_y)
                self._point_values_grid.set_sensitive(True)

                if self.refresh_tile_count():
                   #self._random_tile_button.set_sensitive(True)
                    ti = mode.stamp.get_rawindex_from_tileindex(cn.tile_index)
                    self._tile_adj.set_value(ti)
                else:
                   #self._random_tile_button.set_sensitive(False)
                    pass
            else:
                self._point_values_grid.set_sensitive(False)

           #self._delete_button.set_sensitive(len(mode.nodes) > 0)
        finally:
            self._updating_ui = False

    def refresh_tile_count(self):
        mode, node_idx = self.target

        if mode.stamp:
            if mode.stamp.tile_count > 1:
                self._tile_adj.set_upper(mode.stamp.tile_count-1)
                return True
            else:
                self._tile_adj.set_upper(0)
        return False

    ## Widgets Handlers

    def _angle_adj_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        mode, node_idx = self.target
        mode.update_node(node_idx, angle=math.radians(adj.get_value()))

    def _tile_adj_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        mode, node_idx = self.target
        ti = mode.stamp.get_tileindex_from_rawindex(int(adj.get_value()))
        mode.update_node(node_idx, tile_index=ti)

    def _xscale_adj_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        value = float(adj.get_value())
        mode, node_idx = self.target
        if value == 0:
            value = 0.001  
        mode.update_node(node_idx, scale_x=value)

    def _yscale_adj_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        value = adj.get_value()
        mode, node_idx = self.target
        if value == 0:
            value = 0.001
        mode.update_node(node_idx, scale_y=value)

    def _random_tile_button_clicked_cb(self, button):
        mode, node_idx = self.target
        if mode.stamp.get_tile_count() > 1:
            mode.update_node(node_idx, 
                    tile_index=random.randint(0, mode.stamp.get_tile_count()))

    def _delete_point_button_clicked_cb(self, button):
        mode, node_idx = self.target
        if mode.can_delete_node(node_idx):
            mode.delete_node(node_idx)

    def _iconview_item_changed_cb(self, iconview):
        mode, node_idx = self.target
        if mode:
            if len(iconview.get_selected_items()) > 0:
                path = iconview.get_selected_items()[0]
                iter = self._stamps_store.get_iter(path)
                mode.set_stamp(self._stamps_store.get(iter, 2)[0])



if __name__ == '__main__':
    pass
