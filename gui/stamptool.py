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
from gui.inktool import _LayoutNode
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
class _Phase(PhaseMixin):
    """Enumeration of the states that an BezierCurveMode can be in"""
    MOVE   = 100         #: Moving stamp
    ROTATE = 101         #: Rotate with mouse drag 
    SCALE  = 102         #: Scale with mouse drag 
    ROTATE_BY_HANDLE = 103 #: Rotate with handle of GUI
    SCALE_BY_HANDLE = 104  #: Scale  with handle of GUI
    CALL_BUTTONS = 106     #: call buttons around clicked point. 

class _EditZone(EditZoneMixin):
    """Enumeration of what the pointer is on in phases"""
    CONTROL_HANDLE_0 = 100
    CONTROL_HANDLE_1 = 101
    CONTROL_HANDLE_2 = 102
    CONTROL_HANDLE_3 = 103
    CONTROL_HANDLE_BASE = 100
    SOURCE_AREA = 110
   #SOURCE_AREA_HANDLE = 111
   #SOURCE_TRASH_BUTTON = 112

class _ActionButton(ActionButtonMixin):
    IMPORT_LAYER = 100
    REJECT_SELECTION = 101
    IMPORT_CANVAS = 102


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





class StampMode (OncanvasEditMixin):

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


    ## Class config vars

    buttons = {
        _ActionButton.ACCEPT : ('mypaint-ok-symbolic', 
            'accept_button_cb'), 
        _ActionButton.REJECT : ('mypaint-trash-symbolic', 
            'reject_button_cb'), 
        _ActionButton.IMPORT_LAYER : ('mypaint-ok-symbolic', 
            'import_selection_layer_cb'), 
        _ActionButton.REJECT_SELECTION: ('mypaint-trash-symbolic', 
            'reject_selection_cb'), 
        _ActionButton.IMPORT_CANVAS: ('mypaint-about-symbolic', 
            'import_selection_canvas_cb'), 
    }                 

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
        self._selection_area = None

    @property
    def stamp(self):
        return StampMode._stamp

    @property
    def target_area_index(self):
        return self._target_area_index

    @target_area_index.setter
    def target_area_index(self, index):
        self._target_area_index = index



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

   #def leave(self, **kwds):
   #    """Leaves the mode: called by `ModeStack.pop()` etc."""
   #    super(StampMode, self).leave(**kwds)  # supercall will commit


   #def checkpoint(self, flush=True, **kwargs):
   #    """Sync pending changes from (and to) the model
   #
   #    If called with flush==False, this is an override which just
   #    redraws the pending stroke with the current brush settings and
   #    color. This is the behavior our testers expect:
   #    https://github.com/mypaint/mypaint/issues/226
   #
   #    When this mode is left for another mode (see `leave()`), the
   #    pending brushwork is committed properly.
   #
   #    """
   #
   #    # FIXME almost copyied from polyfilltool.py
   #    if flush:
   #        # Commit the pending work normally
   #        super(StampMode, self).checkpoint(flush=flush, **kwargs) 
   #    else:
   #        # Queue a re-rendering with any new brush data
   #        # No supercall
   #        self._stop_task_queue_runner(complete=False)
   #        self._queue_draw_buttons()
   #        self._queue_redraw_all_nodes()
   #        self._queue_redraw_curve()
        
    def stampwork_commit_all(self, abrupt=False):
        """ abrupt is ignored.
        """
        # We need that the target layer(current layer)
        # has surface and not locked. 
        if len(self.nodes) > 0:
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

                cmd = DrawStamp(self.doc.model,
                        self._stamp,
                        self.nodes,
                        (sx, sy, abs(ex-sx)+1, abs(ey-sy)+1))
                # Important: without this, stamps drawn twice.
                # this _reset_nodes() should called prior to 
                # do(cmd)
                self._reset_nodes() 

                self.doc.model.do(cmd)
            else:
                logger.warning("stamptool.commit_all encounter enpty bbox")


    def _start_new_capture_phase(self, rollback=False):
        """Let the user capture a new ink stroke"""
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()
        self._queue_redraw_curve(force_margin=True)  # call this before reset node

        if rollback:
            self._stop_task_queue_runner(complete=False)
            # Currently, this tool uses overlay(cairo) to preview stamps.
            # so we need no rollback action to the current layer.
            self._reset_nodes()
        else:
            self._stop_task_queue_runner(complete=True)
            self.stampwork_commit_all()

        if self.stamp:
            self.stamp.finalize_phase(self)
            
        self.options_presenter.target = (self, None)
        self._reset_adjust_data()
        self.phase = _Phase.CAPTURE

        if self.stamp:
            self.stamp.initialize_phase(self)


    def _generate_overlay(self, tdw):
        return Overlay_Stamp(self, tdw)

    def _generate_presenter(self):
        return OptionsPresenter_Stamp()

    def _update_zone_and_target(self, tdw, x, y):
        """ Update the zone and target node under a cursor position 
        """
        if not self._stamp:
            return
        else:
            stamp = self._stamp


        new_zone = _EditZone.EMPTY_CANVAS

        if not self.in_drag:
           #overlay = self._ensure_overlay_for_tdw(tdw)
            if self.phase in (_Phase.ADJUST, _Phase.CAPTURE):
                super(StampMode, self)._update_zone_and_target(tdw, x, y)
            elif self.phase in (_Phase.ROTATE, _Phase.SCALE, 
                    _Phase.ROTATE_BY_HANDLE, _Phase.SCALE_BY_HANDLE):
                # XXX In these phase cannot be updated zone and target...?
                pass
            else:
                super(StampMode, self)._update_zone_and_target(tdw, x, y)

    def update_cursor_cb(self, tdw):
        """ Called from _update_zone_and_target()
        to update cursors according to current zone and phase.

        :return : the cursor object or None
                  when returned None, the default cursor
                  self._blank_cursor should be apply.
        :rtype cursor object:
        
        If there is override cursor (self._current_override_cursor)
        returned cursor will overrided by base Mixin.
        """
        cursor = None
        if self.phase in (_Phase.ADJUST, _Phase.CAPTURE):
            if self._selection_area:
                if self.zone == _EditZone.SOURCE_AREA:
                    cursor = self._arrow_cursor
                elif self.zone == _EditZone.ACTION_BUTTON:
                    cursor = self._arrow_cursor
                else:
                    cursor = self._blank_cursor
            else:
                if self.zone == _EditZone.CONTROL_NODE:
                    cursor = self._crosshair_cursor
                elif self.zone != _EditZone.EMPTY_CANVAS: # assume button
                    cursor = self._arrow_cursor
                else:
                    cursor = self._blank_cursor

        return cursor

    def _notify_stamp_changed(self):
        """
        Common processing stamp changed,
        or target area added/removed
        """
        self.options_presenter.refresh_tile_count()

    def select_area_cb(self, selection_mode):
        """ Selection handler called from SelectionMode.
        This handler never called when no selection executed.

        CAUTION: you can not access the self.doc attribute here
        (it is disabled as None, with modestack facility)
        so you must use 'selection_mode.doc', instead of it.
        """
        if self.stamp:
            if self.phase in (_Phase.CAPTURE, _Phase.ADJUST):
                self._selection_area = selection_mode.get_min_max_pos_model(margin=0)

                # Important:Action buttons are operational in ADJUST phase only. 
                self.phase = _Phase.ADJUST 

                self._queue_draw_buttons()

                

    ## Redraws

    def _search_target_node(self, tdw, x, y):
        """ utility method: to commonize 'search target node' processing,
        even in inherited classes.
        """
        index, junk = self._search_target_tile(tdw, x, y)
        return index

    def _search_target_tile(self, tdw, x, y):
        """Search a tile which placed at (x, y) of screen. 
        If (x, y) is on a transformation handle of the tile,
        return index of that handle too.
        Otherwise (pointer hovers only on the tile, not handle), 
        return -1 as handle index.

        :return : a index of tile, with its handle index.
        :rtype tuple:
        """
        hit_dist = gui.style.DRAGGABLE_POINT_HANDLE_SIZE + 12
        new_target_node_index = None
        handle_idx = -1
        stamp = self._stamp
        for i, node in reversed(list(enumerate(self.nodes))):
            handle_idx = stamp.get_handle_index(tdw, x, y, node,
                   gui.style.DRAGGABLE_POINT_HANDLE_SIZE)
            if handle_idx >= 0:
                new_target_node_index = i
                if handle_idx >= 4:
                    handle_idx = -1
                break
        return new_target_node_index, handle_idx

    def _queue_draw_node(self, i, dx=0, dy=0, force_margin=False):
        """Redraws a specific control node on all known view TDWs"""

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

    def _queue_draw_buttons(self):

        super(StampMode, self)._queue_draw_buttons()

        if self._selection_area:
            for tdw in self._overlays:
                sx, sy, ex, ey = gui.ui_utils.get_outmost_area(tdw, 
                        *self._selection_area, 
                        margin=gui.style.DRAGGABLE_POINT_HANDLE_SIZE+4)
                tdw.queue_draw_area(sx, sy, 
                        abs(ex - sx) + 1, abs(ey - sy) + 1)

    ## Raw event handling (prelight & zone selection in adjust phase)

    def mode_button_press_cb(self, tdw, event):

        if not self._stamp or not self._stamp.is_ready:
            return 

        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
        ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK

        if self.phase in (_Phase.ADJUST, _Phase.CAPTURE):
            button = event.button

            if (_EditZone.CONTROL_HANDLE_0 <= 
                    self.zone <= _EditZone.CONTROL_HANDLE_3):
                if button == 1:

                    self.current_handle_index = \
                            self.zone - _EditZone.CONTROL_HANDLE_BASE

                    if ctrl_state:
                        self.phase = _Phase.ROTATE_BY_HANDLE
                    else:
                        self.phase = _Phase.SCALE_BY_HANDLE

            else:
                return super(StampMode, self).mode_button_press_cb(tdw, event)

        elif self.phase in (_Phase.ROTATE,
                            _Phase.SCALE,
                            _Phase.ROTATE_BY_HANDLE,
                            _Phase.SCALE_BY_HANDLE):
            # XXX Not sure what to do here.
            pass
        else:
            return super(StampMode, self).mode_button_press_cb(tdw, event)

    def mode_button_release_cb(self, tdw, event):

        if not self._stamp or not self._stamp.is_ready:
            return 

        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
        ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK

        if self.phase in (_Phase.ROTATE,
                            _Phase.SCALE,
                            _Phase.ROTATE_BY_HANDLE,
                            _Phase.SCALE_BY_HANDLE):
            # XXX Future use. currently does nothing.
            pass
        else:
            return super(StampMode, self).mode_button_release_cb(tdw, event)



    ## Drag handling (both capture and adjust phases)

    def node_drag_start_cb(self, tdw, event):

        mx, my = tdw.display_to_model(event.x, event.y)
        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
        ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK

        if self.phase in (_Phase.CAPTURE, _Phase.ADJUST):

            if self.zone == _EditZone.EMPTY_CANVAS:
                if self.stamp.tile_count > 0:
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

            elif self.zone == _EditZone.CONTROL_NODE:
                super(StampMode, self).node_drag_start_cb(tdw, event)

        elif self.phase in (_Phase.ROTATE,
                            _Phase.SCALE,
                            _Phase.ROTATE_BY_HANDLE,
                            _Phase.SCALE_BY_HANDLE):
            pass
        else:
            super(StampMode, self).node_drag_start_cb(tdw, event)


    def drag_update_cb(self, tdw, event, dx, dy):
        if not self._stamp or not self._stamp.is_ready:
            super(StampMode, self).drag_update_cb(tdw, event, dx, dy)

        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
        ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK
        mx, my = tdw.display_to_model(event.x ,event.y)

        def override_scale_and_rotate():
            if ctrl_state:
                self.phase = _Phase.ROTATE
            else:
                self.phase = _Phase.SCALE
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
        elif self.phase in (_Phase.ADJUST, _Phase.ADJUST_POS):
            if shift_state:
                override_scale_and_rotate()
            else:
                return super(StampMode, self).node_drag_update_cb(tdw, event, dx, dy)
                
        elif self.phase in (_Phase.SCALE,
                            _Phase.ROTATE):
            assert self.current_node_index is not None
            self._queue_redraw_curve()
            node = self.nodes[self.current_node_index]
            bx, by = tdw.model_to_display(node.x, node.y)
            dir, length = get_drag_direction(
                    self.start_x, self.start_y,
                    event.x, event.y)

            if dir >= 0:
                if self.phase == _Phase.ROTATE:
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
        elif self.phase == _Phase.SCALE_BY_HANDLE:
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

        elif self.phase == _Phase.ROTATE_BY_HANDLE:
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
            
        else:
            super(StampMode, self).node_drag_update_cb(tdw, event, dx, dy)


    def drag_stop_cb(self, tdw):
        if not self._stamp or not self._stamp.is_ready:
            return super(StampMode, self).drag_stop_cb(tdw)
        
        if self.phase == _Phase.CAPTURE:

            if not self.nodes or self.current_node_index == None:
                # Cancelled drag event (and current capture phase)
                # call super-superclass directly to bypass this phase
                self._reset_adjust_data()
                return super(StampMode, self).drag_stop_cb(tdw) 

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
            self._queue_draw_selected_nodes() # to ensure erase them

            ox, oy = self.drag_offset.get_model_offset()
            for i in self.selected_nodes:
                cn = self.nodes[i]
                self.nodes[i] = cn._replace(x=cn.x + ox, 
                                            y=cn.y + oy)
            self.drag_offset.reset()
            
            self._queue_redraw_all_nodes()
            self._queue_redraw_curve()
            self._queue_draw_buttons()

        elif self.phase in (_Phase.ROTATE,
                            _Phase.SCALE,
                            _Phase.ROTATE_BY_HANDLE,
                            _Phase.SCALE_BY_HANDLE):
            self.phase = _Phase.ADJUST

        else:
            return super(StampMode, self).node_drag_stop_cb(tdw)

    def node_scroll_cb(self, tdw, event):
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
            return super(StampMode, self).node_scroll_cb(tdw, event)


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

    ## Action button related
    def accept_button_cb(self, tdw):
        if len(self.nodes) > 0:
            self._start_new_capture_phase(rollback=False)

    def reject_button_cb(self, tdw):
        if len(self.nodes) > 0:
            self._start_new_capture_phase(rollback=True)

    def _capture_layer_to_stamp(self, layer):
        sx, sy, ex, ey = [int(x) for x in self._selection_area]
        pixbuf = layer.render_as_pixbuf(sx, sy, 
                abs(ex-sx)+1, abs(ey-sy)+1, alpha=True)
        self.stamp.set_surface_from_pixbuf(-1, pixbuf)

    def import_selection_layer_cb(self, tdw):
        assert self._selection_area != None
        layer = self.doc.model.layer_stack.current
        self._capture_layer_to_stamp(layer)

        # Then, erase all selection-area related gui.
        # It is same as reject_selection_cb()
        self.reject_selection_cb(tdw, 
                msg=_("Import a part of layer into the stamp."))


    def reject_selection_cb(self, tdw, msg=None):
        self._queue_draw_buttons() # To erase
        self._selection_area = None

        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()

        if not msg:
            msg = _("Cancelled to import stamp picture.")

        self.doc.app.show_transient_message(msg)

        self.phase = _Phase.CAPTURE

    def import_selection_canvas_cb(self, tdw):
        assert self._selection_area != None
        layer = self.doc.model.layer_stack
        self._capture_layer_to_stamp(layer)

        # Then, erase all selection-area related gui.
        # It is same as reject_selection_cb()
        self.reject_selection_cb(tdw, 
                msg=_("Import a part of canvas into the stamp."))

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
        tdw = self._tdw
        alloc = tdw.get_allocation()
        view_x0, view_y0 = alloc.x, alloc.y
        view_x1, view_y1 = view_x0+alloc.width, view_y0+alloc.height
        button_radius = gui.style.FLOATING_BUTTON_RADIUS

        for ck in self._button_pos:
            self._button_pos[ck] = None

        if mode._selection_area != None:
            # When selection area is active (not None),
            # buttons for selection area activated.

            area = mode._selection_area
            for i, x, y in gui.ui_utils.enum_area_point(*area):
                x, y = tdw.model_to_display(x, y)
                if i == 0:
                    sx = ex = x
                    sy = ey = y
                else:
                    sx = min(x, sx)
                    sy = min(y, sy)
                    ex = max(x, ex)
                    ey = max(y, ey)

            cx = sx + (ex - sx) / 2
            cy = sy + (ey - sy) / 2

            entire_width = (button_radius * 2) * 3 + button_radius

            # Reuse sx, sy for button placement
            sx = cx - (entire_width / 2)
            sy = cy - (button_radius / 2) 

            # Then, sx,sy means 'the first button position'.
            # Adjust them into outmost position of buttons array.
            sx -= button_radius
            sy -= button_radius

            if sx < view_x0:
                sx = view_x0 + button_radius
            elif sx + entire_width > view_x1:
                sx = view_x1 - entire_width

            if sy < view_y0:
                sy = view_y0
            elif sy + button_radius * 2 > view_y1:
                sy = view_y1 - button_radius * 2

            # Restore them into center position of the first button.
            sx += button_radius
            sy += button_radius

            for id in (_ActionButton.IMPORT_LAYER,
                       _ActionButton.IMPORT_CANVAS,
                       _ActionButton.REJECT_SELECTION):
                self._button_pos[id] = (sx, sy)
                sx += button_radius * 3

        else:
            # Otherwise, normal editing buttons are activated.

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


            nodes = mode.nodes
            num_nodes = len(nodes)
            if num_nodes == 0:
                return False

            button_radius = gui.style.FLOATING_BUTTON_RADIUS
            margin = 1.5 * button_radius

            if mode.forced_button_pos:
                # User deceided button position 
                cx, cy = mode.forced_button_pos
                area_radius = gui.style.FLOATING_TOOL_RADIUS * 3

                cx, cy = adjust_button_inside(cx, cy, area_radius)

                pos_list = []
                count = 2
                # TODO : 
                #  Although this is 'arranging buttons in a circle', 
                #  in consideration of increasing the number of buttons, 
                #  but experience seems to be enough for two buttons for
                #  normal editing. 
                #  So it may be unnecessary processing.
                for i in range(count):
                    rad = (math.pi / count) * 2.0 * i
                    x = - area_radius*math.sin(rad)
                    y = area_radius*math.cos(rad)
                    pos_list.append( (x + cx, - y + cy) )

                self._button_pos[_ActionButton.ACCEPT] = pos_list[0][0], pos_list[0][1]
                self._button_pos[_ActionButton.REJECT] = pos_list[1][0], pos_list[1][1]
            else:
                # InkingMode is consisted from two completely different phase, 
                # to capture and to adjust.
                # it cannot extend nodes in adjust phase, so no problem.
                #
                # But, usually, other oncanvas-editing tools can extend
                # new nodes in adjust phase.
                # So when the action button is placed around the last node,
                # it might be frastrating, because we might not place
                # a new node at the area where button already occupied.
                # 
                # Thus,different from Inktool, place buttons around 
                # the first(oldest) nodes.
                
                node = nodes[0]
                cx, cy = tdw.model_to_display(node.x, node.y)

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
                
                self._button_pos[_ActionButton.ACCEPT] = adjust_button_inside(
                        cx + dy, cy - dx, button_radius * 1.5)
                self._button_pos[_ActionButton.REJECT] = adjust_button_inside(
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

    def draw_selection_area(self, cr):

        tdw = self._tdw
        area = self._mode._selection_area
        assert area != None

        # all area component is in model coordinate.
        # therefore, convert them into screen coordinate.

        cr.save()
        for i, x, y in gui.ui_utils.enum_area_point(*area):
            x, y = tdw.model_to_display(x, y)
            if i == 0:
                cr.move_to(x, y)
            else:
                cr.line_to(x, y)

        cr.close_path()

        cr.set_dash((), 0)
        cr.set_source_rgb(0, 0, 0)
        cr.stroke_preserve()

        cr.set_dash( (3.0, ) )
        cr.set_source_rgb(*self.SELECTED_AREA_COLOR)
        cr.stroke()

        cr.restore()


    def paint(self, cr):
        """Draw adjustable nodes to the screen"""

        mode = self._mode
        if mode.stamp == None:
            return

        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        alloc = self._tdw.get_allocation()
        dx,dy = mode.drag_offset.get_display_offset(self._tdw)
        fill_flag = True
       #mode.stamp.initialize_draw(cr)

        colors = ( (1, 1, 1), self.SELECTED_COLOR)

        for i, node in self._get_onscreen_nodes():
           #color = gui.style.EDITABLE_ITEM_COLOR
            show_node = not mode.hide_nodes

            if show_node:
                self.draw_stamp(cr, i, node, dx, dy, colors)
            else:
                self.draw_stamp_rect(cr, i, node, dx, dy, colors)

       #mode.stamp.finalize_draw(cr)

        # Selection areas
        if mode._selection_area != None:
            self.draw_selection_area(cr)
       #if mode.stamp.is_support_selection:
       #    # TODO this is for stamp manager shows source area
       #    # or right after selection tool activated.
       #    if mode.stamp.tile_count > 0:
       #        self.draw_selection_area(cr, 
       #                dx, dy,
       #                self.SELECTED_AREA_COLOR, (1, 0 ,0) )

        # Buttons
        adjust_phase_flag = (mode.phase == _Phase.ADJUST and
                len(mode.nodes) > 0 )
        select_phase_flag = (mode._selection_area != None)

        if (not mode.in_drag and (adjust_phase_flag or select_phase_flag)):
            self._draw_mode_buttons(cr)


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
                manager = self._app.stamp_manager
               #manager.set_current_iter(iter)
                mode.set_stamp(self._stamps_store.get(iter, 2)[0])
               #mode.set_stamp(manager.current)



if __name__ == '__main__':
    pass
