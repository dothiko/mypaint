# This file is part of MyPaint.
# Copyright (C) 2008-2013 by Martin Renold <martinxyz@gmx.ch>
# Copyright (C) 2013-2016 by the MyPaint Development Team.
#
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

import gtk2compat
from gettext import gettext as _
import gtk
from gtk import gdk
from gtk import keysyms
from libmypaint import brushsettings
from gi.repository import GLib
from gi.repository import GdkPixbuf
import cairo

import gui.mode
from drawutils import spline_4p
from lib import mypaintlib
from gui.inktool import *
from gui.inktool import _LayoutNode, _Phase, _EditZone
from gui.linemode import *
from lib.command import Command
from gui.ui_utils import *
from gui.stamps import *

## Module settings

# Which workarounds to allow for motion event compression
EVCOMPRESSION_WORKAROUND_ALLOW_DISABLE_VIA_API = True
EVCOMPRESSION_WORKAROUND_ALLOW_EVHACK_FILTER = True

# Consts for the style of workaround in use
EVCOMPRESSION_WORKAROUND_DISABLE_VIA_API = 1
EVCOMPRESSION_WORKAROUND_EVHACK_FILTER = 2
EVCOMPRESSION_WORKAROUND_NONE = 999

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
    CALL_BUTTONS = 106      #: call buttons around clicked point. 

class _EditZone_Stamp:
    """Enumeration of what the pointer is on in phases"""
    CONTROL_HANDLE_0 = 100
    CONTROL_HANDLE_1 = 101
    CONTROL_HANDLE_2 = 102
    CONTROL_HANDLE_3 = 103
    CONTROL_HANDLE_BASE = 100


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
        draw_stamp_to_layer(target,
                self._stamp, self.nodes, self.bbox)

    def undo(self):
        layers = self.doc.layer_stack
        assert self.snapshot is not None
        layers.current.load_snapshot(self.snapshot)
        self.snapshot = None





class StampMode (InkingMode):

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



    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):
        super(StampMode, self).__init__(**kwargs)

        self._stamp = None
        #! test code
        self.current_handle_index = -1
        self.forced_button_pos = False

    @property
    def stamp(self):
        return self._stamp

    def set_stamp(self, stamp):
        """ Called from OptionPresenter, 
        This is to make stamp property as if it is read-only.
        """
        self._stamp = stamp
        self._stamp.initialize_phase()


    def enter(self, doc, **kwds):
        """Enters the mode: called by `ModeStack.push()` etc."""
        self._blank_cursor = doc.app.cursors.get_action_cursor(
            self.ACTION_NAME,
            gui.cursor.Name.ADD
        )
        self.options_presenter.target = (self, None)
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
        bbox = self._stamp.get_bbox(None, self.nodes[0])
        if bbox:
            sx, sy, w, h = bbox

            ex = sx+w
            ey = sy+h
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
                stamp = PixbufStamp('', self._stamp.pixbuf)
            else:
                stamp = self._stamp

            cmd = DrawStamp(self.doc.model,
                    stamp,
                    self.nodes,
                    (sx, sy, ex - sx + 1, ey - sy + 1))
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
            self.stamp.finalize_phase()
            
        self.options_presenter.target = (self, None)
        self._queue_redraw_curve(force_margin=True)  # call this before reset node
        self._queue_draw_buttons()
        self._queue_redraw_all_nodes()
        self._reset_nodes()
        self._reset_capture_data()
        self._reset_adjust_data()
        self.phase = _Phase.CAPTURE

        if self.stamp:
            self.stamp.initialize_phase()

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
                    new_target_node_index, control_node_idx = \
                            self._search_target_node(tdw, x, y)
                    if new_target_node_index != None:
                        if  0<= control_node_idx <= 3:
                            new_zone = _EditZone_Stamp.CONTROL_HANDLE_BASE + \
                                        control_node_idx
                        else:
                            new_zone = _EditZone.CONTROL_NODE

                        if new_zone != self.zone:
                            self._queue_draw_node(new_target_node_index)

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
                else:
                    cursor = self._blank_cursor
            else:
                cursor = self._blank_cursor

            if cursor is not self._current_override_cursor:
                tdw.set_override_cursor(cursor)
                self._current_override_cursor = cursor


    ## Redraws


   #def _queue_draw_selected_nodes(self):
   #    for i in self.selected_nodes:
   #        self._queue_draw_node(i)
   #
   #def _queue_redraw_all_nodes(self):
   #    """Redraws all nodes on all known view TDWs"""
   #    for i in xrange(len(self.nodes)):
   #        self._queue_draw_node(i)
   #
    def _search_target_node(self, tdw, x, y):
        """ utility method: to commonize processing,
        even in inherited classes.
        """
        hit_dist = gui.style.DRAGGABLE_POINT_HANDLE_SIZE + 12
        new_target_node_index = None
        handle_idx = -1
        stamp = self._stamp
        mx, my = tdw.display_to_model(x, y)
        for i, node in reversed(list(enumerate(self.nodes))):
            handle_idx = stamp.get_handle_index(mx, my, node,
                   gui.style.DRAGGABLE_POINT_HANDLE_SIZE)
           #if stamp.is_inside(mx, my, node):
            if handle_idx >= 0:
                new_target_node_index = i
                if handle_idx >= 4:
                    handle_idx = -1
                break
        return new_target_node_index, handle_idx

    def _queue_draw_node(self, i, force_margin=False):
        """Redraws a specific control node on all known view TDWs"""
       #node = self.nodes[i]
       #
       #if i == self.target_node_index:
       #    margin = gui.style.DRAGGABLE_POINT_HANDLE_SIZE + 4
       #else:
       #    margin = 4
       #
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
                    self._queue_draw_node_internal(tdw, cn, dx, dy, targetted)
                else:
                    self._queue_draw_node_internal(tdw, cn, 0.0, 0.0, targetted)
        

    ## Raw event handling (prelight & zone selection in adjust phase)
    def button_press_cb(self, tdw, event):
        if not self._stamp:
            return super(InkingMode, self).button_press_cb(tdw, event)

        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False
        self._update_zone_and_target(tdw, event.x, event.y)
        self._update_current_node_index()

        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
        ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK

        if self.phase in (_Phase.ADJUST, _Phase.ADJUST_PRESSURE):
            button = event.button
            # Normal ADJUST/ADJUST_PRESSURE Phase.

            if self.zone in (_EditZone.REJECT_BUTTON,
                             _EditZone.ACCEPT_BUTTON):
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
                    do_reset = False
                    if shift_state:
                        # Holding SHIFT key
                        if ctrl_state:
                            self.phase = _PhaseStamp.ROTATE
                        else:
                            self.phase = _PhaseStamp.SCALE
                            
                        self._queue_redraw_curve()
                        
                    elif ctrl_state:
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

            elif (_EditZone_Stamp.CONTROL_HANDLE_0 <= 
                    self.zone <= _EditZone_Stamp.CONTROL_HANDLE_3):
                if button == 1:

                    self.current_handle_index = \
                            self.zone - _EditZone_Stamp.CONTROL_HANDLE_BASE

                    if ctrl_state:
                        self.phase = _PhaseStamp.ROTATE_BY_HANDLE
                    else:
                        self.phase = _PhaseStamp.SCALE_BY_HANDLE
            else:
                raise NotImplementedError("Unrecognized zone %r", self.zone)


        elif self.phase == _Phase.CAPTURE:
            # XXX Not sure what to do here.
            # XXX Click to append nodes?
            # XXX  but how to stop that and enter the adjust phase?
            # XXX Click to add a 1st & 2nd (=last) node only?
            # XXX  but needs to allow a drag after the 1st one's placed.
            pass
        elif self.phase in (_Phase.ADJUST_PRESSURE_ONESHOT,
                            _PhaseStamp.ROTATE,
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
        if not self._stamp:
            return super(InkingMode, self).button_release_cb(tdw, event)

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
                            self._start_new_capture_phase(rollback=True)
                            assert self.phase == _Phase.CAPTURE
                        elif zone0 == _EditZone.ACCEPT_BUTTON:
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
            # Updating options_presenter is done at drag_stop_cb()
            pass

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

        self._update_zone_and_target(tdw, event.x, event.y)
        return super(InkingMode, self).motion_notify_cb(tdw, event)
    ## Drag handling (both capture and adjust phases)

    def drag_start_cb(self, tdw, event):
        if not self._stamp:
            return super(StampMode, self).drag_start_cb(tdw, event)

        self._ensure_overlay_for_tdw(tdw)
        mx, my = tdw.display_to_model(event.x, event.y)
        if self.phase == _Phase.CAPTURE:

            if event.state != 0:
                # To activate some mode override
                self._last_event_node = None
                return super(InkingMode, self).drag_start_cb(tdw, event)
            else:
                node = _StampNode(mx, my, 
                        self._stamp.default_angle,
                        self._stamp.default_scale_x,
                        self._stamp.default_scale_y,
                        0)
                self.nodes.append(node)
                self._queue_draw_node(0)
                self._last_node_evdata = (event.x, event.y, event.time)
                self._last_event_node = node

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
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)


    def drag_update_cb(self, tdw, event, dx, dy):
        if not self._stamp:
            super(StampMode, self).drag_update_cb(tdw, event, dx, dy)

        self._ensure_overlay_for_tdw(tdw)
        mx, my = tdw.display_to_model(event.x ,event.y)
        if self.phase == _Phase.CAPTURE:

            self._queue_redraw_curve()

             # [TODO] below line can be reformed to minimize redrawing
            self._queue_draw_node(len(self.nodes)-1) 
        elif self.phase == _Phase.ADJUST:
            self._queue_redraw_curve()
            super(StampMode, self).drag_update_cb(tdw, event, dx, dy)
        elif self.phase in (_PhaseStamp.SCALE,
                            _PhaseStamp.ROTATE):
            assert self.target_node_index is not None
            self._queue_redraw_curve()
            node = self.nodes[self.target_node_index]
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

                self.nodes[self.target_node_index] = node
                self._queue_redraw_curve()
                self.start_x = event.x
                self.start_y = event.y
        elif self.phase in (_PhaseStamp.SCALE_BY_HANDLE,
                            _PhaseStamp.ROTATE_BY_HANDLE):

            assert self.target_node_index is not None
            self._queue_redraw_curve()
            node = self.nodes[self.target_node_index]
            pos = self._stamp.get_boundary_points(node, 
                    no_transform=True)

            # At here, we consider the movement of control handle(i.e. cursor)
            # as a Triangle from origin.

            # 1. Get new(=cursor position) vector from origin(node.x,node.y)
            mx, my = tdw.display_to_model(event.x, event.y)
            length, nx, ny = length_and_normal(
                    node.x, node.y,
                    mx, my)

            bx = node.x - mx
            by = node.y - my

            if self.phase == _PhaseStamp.SCALE_BY_HANDLE:

                # 2. Get 'original vertical leg' of triangle,
                # it is equal the half of 'side' ridge of stamp rectangle
                side_length, snx, sny = length_and_normal(
                        pos[2][0], pos[2][1],
                        pos[1][0], pos[1][1])
                side_length /= 2.0


                # 3. Use the identity vector of side ridge from above
                # to get 'current' base leg of triangle.
                dp = dot_product(snx, sny, bx, by)
                vx = dp * snx
                vy = dp * sny
                v_length = vector_length(vx, vy)

                # 4. Then, get another leg of triangle.
                hx = bx - vx
                hy = by - vy
                h_length = vector_length(hx, hy)

                # 5. Finally, we can get the new scaling ratios.
                top_length = vector_length(pos[1][0] - pos[0][0],
                        pos[1][1] - pos[0][1]) / 2.0

                # Replace the attributes and redraw.
                assert top_length != 0.0
                assert side_length != 0.0
                self.nodes[self.target_node_index] = node._replace(
                        scale_x=h_length / top_length,
                        scale_y=v_length / side_length)

            else:
                # 2. Get angle between current cursor position
                # to the 'origin - handle vector'.

                ox, oy = pos[self.current_handle_index]
                
                ox, oy = normal(node.x, node.y, ox, oy)
                cx, cy = normal(node.x, node.y, mx, my)

                rad = get_radian(cx, cy, ox, oy)

                # 3. Get a cross product of them to
                # identify which direction user want to rotate.
                if cross_product(cx, cy, ox, oy) >= 0.0:
                    rad = -rad

                self.nodes[self.target_node_index] = node._replace(
                        angle = rad)
                      

            self._queue_redraw_curve()

        else:
            super(StampMode, self).drag_update_cb(tdw, event, dx, dy)


    def drag_stop_cb(self, tdw):
        if not self._stamp:
            return super(StampMode, self).drag_stop_cb(tdw)

        self._ensure_overlay_for_tdw(tdw)
        if self.phase == _Phase.CAPTURE:

            if not self.nodes or self._last_event_node == None:
                # call super-superclass directly to bypass this phase
                return super(InkingMode, self).drag_stop_cb(tdw) 



            self._reset_capture_data()
            self._reset_adjust_data()
            if len(self.nodes) > 0:
                self.phase = _Phase.ADJUST
                self.target_node_index = len(self.nodes) -1
                self._update_current_node_index()
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
        else:
            return super(StampMode, self).drag_stop_cb(tdw)

    def scroll_cb(self, tdw, event):
        """Handles scroll-wheel events, to adjust pressure."""
        if (self.phase in (_Phase.ADJUST, 
                _Phase.ADJUST_PRESSURE, 
                _Phase.ADJUST_PRESSURE_ONESHOT) 
                and self.target_node_index != None):
            return

        else:
            return super(StampMode, self).scroll_cb(tdw, event)


    ## Interrogating events

    ## Node editing
    def update_node(self, i, **kwargs):
        self._queue_draw_node(i, force_margin=True) 
        self.nodes[i] = self.nodes[i]._replace(**kwargs)
        self._queue_draw_node(i, force_margin=True) 


class Overlay_Stamp (Overlay):
    """Overlay for an StampMode's adjustable points"""

    def __init__(self, mode, tdw):
        super(Overlay_Stamp, self).__init__(mode, tdw)

    def _get_onscreen_nodes(self):
        """Iterates across only the on-screen nodes."""
        mode = self._inkmode
        alloc = self._tdw.get_allocation()
        dx,dy = mode.drag_offset.get_display_offset(self._tdw)
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
                    x > alloc.x  and
                    y > alloc.y  and
                    x < alloc.x + alloc.width + w and
                    y < alloc.y + alloc.height + h
                )

                if node_on_screen:
                    yield (i, node)

    def update_button_positions(self):
        """Recalculates the positions of the mode's buttons."""
        # FIXME mostly copied from inktool.Overlay.update_button_positions
        # The difference is for-loop of nodes , to deal with control handles.
        mode = self._inkmode
        nodes = mode.nodes
        num_nodes = len(nodes)
        if num_nodes == 0:
            self.reject_button_pos = None
            self.accept_button_pos = None
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

            self.accept_button_pos = pos_list[0][0], pos_list[0][1]
            self.reject_button_pos = pos_list[1][0], pos_list[1][1]
        else:
            # Usually, Bezier tool needs to keep extending control points.
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
            
            self.accept_button_pos = adjust_button_inside(
                    cx + dy, cy - dx, button_radius * 1.5)
            self.reject_button_pos = adjust_button_inside(
                    cx - dy, cy + dx, button_radius * 1.5)

        return True


    def draw_stamp(self, cr, idx, node, dx, dy):
        """ Draw a stamp as overlay preview.

        :param idx: index of node in mode.nodes[] 
        :param node: current node.this holds some information
                     such as stamp scaling ratio and rotation.
        :param x,y: display coordinate position of node.
        """
        mode = self._inkmode
        pos = mode.stamp.get_boundary_points(node, tdw=self._tdw)
        x, y = self._tdw.model_to_display(node.x, node.y)

        if idx == mode.current_node_index or idx in mode.selected_nodes:
            self.draw_stamp_rect(cr, idx, dx, dy, position=pos)
            x+=dx
            y+=dy
            for i, pt in enumerate(pos):
                handle_idx = _EditZone_Stamp.CONTROL_HANDLE_BASE + i
                gui.drawutils.render_square_floating_color_chip(
                    cr, pt[0] + dx, pt[1] + dy,
                    gui.style.ACTIVE_ITEM_COLOR, 
                    gui.style.DRAGGABLE_POINT_HANDLE_SIZE,
                    fill=(handle_idx==mode.zone)) 
        else:
            self.draw_stamp_rect(cr, idx, 0, 0, position=pos)

        mode.stamp.draw(self._tdw, cr, x, y,
                node, True)

    def draw_stamp_rect(self, cr, idx, dx, dy, position=None):
        cr.save()
        mode = self._inkmode
        cr.set_line_width(1)
        if idx == mode.current_node_index:
            cr.set_source_rgb(1, 0, 0)
        else:
            cr.set_source_rgb(0, 0, 0)

        if not position:
            position = mode.stamp.get_boundary_points(
                    mode.nodes[idx], tdw=self._tdw)

        cr.move_to(position[0][0] + dx,
                position[0][1] + dy)
        for lx, ly in position[1:]:
            cr.line_to(lx+dx, ly+dy)
        cr.close_path()
        cr.stroke()
        cr.restore()

    def paint(self, cr):
        """Draw adjustable nodes to the screen"""
        # Control nodes
        mode = self._inkmode
        if mode.stamp == None:
            return

        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        alloc = self._tdw.get_allocation()
        dx,dy = mode.drag_offset.get_display_offset(self._tdw)
        fill_flag = not mode.phase in (_Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT)
        mode.stamp.initialize_draw(cr)

        for i, node in self._get_onscreen_nodes():
            color = gui.style.EDITABLE_ITEM_COLOR
            show_node = not mode.hide_nodes
            if (mode.phase in
                    (_Phase.ADJUST,
                     _Phase.ADJUST_PRESSURE,
                     _Phase.ADJUST_PRESSURE_ONESHOT)):
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
                self.draw_stamp(cr, i, node, dx, dy)
            else:
                self.draw_stamp_rect(cr, i, node, dx, dy)


        mode.stamp.finalize_draw(cr)

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


class OptionsPresenter_Stamp (object):
    """Presents UI for directly editing point values etc."""

    variation_preset_store = None

    @classmethod
    def init_stamp_presets(cls):
        return
       #if cls.variation_preset_store == None:
       #    from application import get_app
       #    _app = get_app()
       #    store = Gtk.ListStore(str, int)
       #    for i,name in enumerate(_app.stroke_pressure_settings.settings):
       #        store.append((name,i))
       #    cls.variation_preset_store = store

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
        self._random_tile_button = builder.get_object("random_tile_button")
        self._random_tile_button.set_sensitive(False)
        base_grid = builder.get_object("preset_editing_grid")

        self.init_stamp_preset_view(1, base_grid)

    def init_stamp_preset_view(self, row, box):
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
       #liststore.connect('changed', self._iconview_item_changed_cb)

        sw = Gtk.ScrolledWindow()
        sw.set_margin_top(4)
        sw.set_shadow_type(Gtk.SHADOW_ETCHED_IN)
        sw.set_policy(Gtk.POLICY_AUTOMATIC, Gtk.POLICY_AUTOMATIC)            
        sw.set_hexpand(True)
        sw.set_vexpand(True)
        sw.set_halign(Gtk.Align.FILL)
        sw.set_valign(Gtk.Align.FILL)
        sw.add(iconview)
        box.attach(sw, 0, row, 2, 1)
       #combo.connect('changed', self._variation_preset_combo_changed_cb)
        self.preset_view = iconview
        return #! REMOVEME

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
            tile_adj = None
            if mode.stamp != None:
                if mode.stamp.tile_count > 1:
                    self._tile_adj.set_upper(mode.stamp.tile_count-1)
                    tile_adj = self._tile_adj
                else:
                    self._tile_adj.set_upper(0)

                self._random_tile_button.set_sensitive(
                        mode.stamp.tile_count > 1)
            else:
                self._random_tile_button.set_sensitive(False)


            if 0 <= cn_idx < len(mode.nodes):
                cn = mode.nodes[cn_idx]
                self._angle_adj.set_value(
                        clamp(math.degrees(cn.angle),-180.0, 180.0))
                self._xscale_adj.set_value(cn.scale_x)
                self._yscale_adj.set_value(cn.scale_y)
                self._point_values_grid.set_sensitive(True)
                if tile_adj:
                    tile_adj.set_value(cn.tile_index)
            else:
                self._point_values_grid.set_sensitive(False)

           #self._delete_button.set_sensitive(len(mode.nodes) > 0)
        finally:
            self._updating_ui = False


    def _angle_adj_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        mode, node_idx = self.target
        mode.update_node(node_idx, angle=math.radians(adj.get_value()))

    def _tile_adj_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        mode, node_idx = self.target
        mode.set_node_tile(node_idx, adj.get_value())

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
            mode.set_current_tile(
                    random.randint(0, mode.stamp.get_tile_count()))

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
