# This file is part of MyPaint.
# Copyright (C) 2015 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


## Imports
from __future__ import division, print_function

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
import gui.drawwindow
import gui.cursor
import lib.helpers
import lib.observable
import lib.layer
import lib.surface
import lib.command 
from gui.inktool import _LayoutNode

## Constants

_TILE_SIZE = lib.mypaintlib.TILE_SIZE

## Function defs

## Class defs


class _Phase:
    """Enumeration of the states that an InkingMode can be in"""
    CAPTURE = 0
    ADJUST = 1

_NODE_FIELDS = ("x", "y", )


class _Node (collections.namedtuple("_Node", _NODE_FIELDS)):
    """Recorded control point, as a namedtuple.

    Node tuples have the following 6 fields, in order

    * x, y: model coords, float
    """


class _EditZone:
    """Enumeration of what the pointer is on in the ADJUST phase"""
    EMPTY_CANVAS = 0  #: Nothing, empty space
    CONTROL_NODE = 1  #: Any control node; see target_node_index
    REJECT_BUTTON = 2  #: On-canvas button that abandons the current line
    ACCEPT_BUTTON = 3  #: On-canvas button that commits the current line

class _FillMethod:
    """Constants for fill method.
    """
    FLOOD_FILL = 0
    CLOSED_AREA_FILL = 1
    LASSO_FILL = 2

    LABELS = {
        0 : "Flood fill",
        1 : "Closed area fill",
        2 : "Lasso fill"
    }

class _Prefs:
    """Constants of preferences.
       This class looks _FillMethod constants, 
       so must be defined after _FillMethod defined.
    """

    # These PREF constants cannot be used without PREFIX.
    TOLERANCE_PREF = 'tolerance'
    GAP_LEVEL_PREF = 'gap_level'
    DILATION_SIZE_PREF = 'dilate_size'
    SAMPLE_MERGED_PREF = 'sample_merged'
    FILL_IMMIDIATELY_PREF = 'fill_immidiately'
    REJECT_FACTOR_PREF = 'reject_factor'
    FILL_METHOD_PREF = 'fill_method'
    SHARE_SETTING_PREF = 'share_setting'
    ALPHA_THRESHOLD_PREF = 'alpha_threshold'
    FILL_ALL_HOLES_PREF = 'fill_all_holes'

    DEFAULT_TOLERANCE = 0.2
    DEFAULT_GAP_LEVEL = 3
    DEFAULT_DILATION_SIZE = 2
    DEFAULT_SAMPLE_MERGED = True
    DEFAULT_MAKE_NEW_LAYER = False
    DEFAULT_FILL_IMMIDIATELY = False
    DEFAULT_REJECT_FACTOR = 2.0
    DEFAULT_FILL_METHOD = _FillMethod.FLOOD_FILL
    DEFAULT_SHARE_SETTING = True
    DEFAULT_ALPHA_THRESHOLD = 0.0156
    DEFAULT_FILL_ALL_HOLES = False

    PREFIX = {
        _FillMethod.FLOOD_FILL : "flood_fill",
        _FillMethod.CLOSED_AREA_FILL : "closed_area_fill",
        _FillMethod.LASSO_FILL : "lasso_fill",
    }



class ClosefillMode (gui.mode.ScrollableModeMixin,
                       gui.mode.DragMode,
                       gui.mode.SingleClickMode):
    """FIXME:Mostly copied from inktool.py.
    """      

    ## Metadata properties

    ACTION_NAME = "CloseFillMode"
    pointer_behavior = gui.mode.Behavior.PAINT_FREEHAND
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW
    permitted_switch_actions = (
        set(gui.mode.BUTTON_BINDING_ACTIONS).union([
            'RotateViewMode',
            'ZoomViewMode',
            'PanViewMode',
        ])
    )

    ## Metadata methods

    @classmethod
    def get_name(cls):
        return _(u"CloseFill")

    def get_usage(self):
        return _(u"Fill areas with color. Holding Shift changes fill method temporary. Holding Alt fill with transparent color.")

    ## Class config vars

    # Input node capture settings:
   #MAX_INTERNODE_DISTANCE_MIDDLE = 30   # display pixels
   #MAX_INTERNODE_DISTANCE_ENDS = 10   # display pixels

   # with 15 & 5, processing time 0.00678086280823
   #MAX_INTERNODE_DISTANCE_MIDDLE = 15   # display pixels
   #MAX_INTERNODE_DISTANCE_ENDS = 5   # display pixels

    # 8 & 3, processing time 0.00526189804077
                        
    MAX_INTERNODE_DISTANCE_MIDDLE = 5   # display pixels
    MAX_INTERNODE_DISTANCE_ENDS = 2   # display pixels

    ## Cursors
    _cursors = {}
    _CURSOR_ARROW = 0
    _CURSOR_CROSS = 1
    _CURSOR_PENCIL = 2
    _CURSOR_ERASER = 3
    _CURSOR_CROSS_ERASER = 4

    ## Other class vars

    _OPTIONS_PRESENTER = None   #: Options presenter singleton
    _MODIFIER_CHANGE_FILL = (Gdk.KEY_Shift_L, Gdk.KEY_Shift_R)

    ## Class methods
    @classmethod
    def _init_cursors(cls, app):
        cursor_dict = cls._cursors
        cursors = app.cursors
        name = gui.cursor.Name
        c = cursors.get_action_cursor(
            cls.ACTION_NAME,
            name.ARROW,
        )
        cursor_dict[cls._CURSOR_ARROW] = c

        c = cursors.get_action_cursor(
            cls.ACTION_NAME,
            name.CROSSHAIR_OPEN_PRECISE,
        )
        cursor_dict[cls._CURSOR_CROSS] = c

        c = cursors.get_action_cursor(
            cls.ACTION_NAME,
            name.PENCIL,
        )
        cursor_dict[cls._CURSOR_PENCIL] = c

        # Use eraser icon. so, using get_icon_cursor method.
        eraser_icon_name = "mypaint-eraser-symbolic" 
        c = cursors.get_icon_cursor(
            eraser_icon_name, 
            name.PENCIL,
        )
        cursor_dict[cls._CURSOR_ERASER] = c

        c = cursors.get_icon_cursor(
            eraser_icon_name, 
            name.CROSSHAIR_OPEN_PRECISE,
        )
        cursor_dict[cls._CURSOR_CROSS_ERASER] = c

    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):
        super(ClosefillMode, self).__init__(**kwargs) 
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

        self._overridden_fill_method = None

    def _reset_nodes(self):
        self.nodes = []  # nodes that met the distance+time criteria

    def _reset_capture_data(self):
        self._last_event_node = None  # node for the last event

    def _reset_adjust_data(self):
        self.zone = _EditZone.EMPTY_CANVAS
        self.current_node_index = None
        self.target_node_index = None
        self._dragged_node_start_pos = None

        # marker shares _dragged_node_start_pos with nodes.

    def _is_active(self):
        for mode in self.doc.modes:
            if mode is self:
                return True
        return False

    def _ensure_overlay_for_tdw(self, tdw):
        overlay = self._overlays.get(tdw)
        if not overlay:
            overlay = Overlay(self, tdw)
            tdw.display_overlays.append(overlay)
            self._overlays[tdw] = overlay
        return overlay

    def _discard_overlays(self):
        for tdw, overlay in self._overlays.items():
            tdw.display_overlays.remove(overlay)
            tdw.queue_draw()
        self._overlays.clear()

    def enter(self, doc, **kwds):
        """Enters the mode: called by `ModeStack.push()` etc."""
        super(ClosefillMode, self).enter(doc, **kwds)
        if not self._is_active():
            self._discard_overlays()
        self._ensure_overlay_for_tdw(self.doc.tdw)
        opt = self.options_presenter
        opt.target = self

        app = self.doc.app
        cls = self.__class__
        if len(cls._cursors) == 0:
            cls._init_cursors(app)

        app.brushmodifier.blend_mode_changed += self._blend_mode_changed_cb

        # Needed this when overrided by scroll-mode or something.
        self._current_override_cursor = None
        self._update_cursor(self.doc.tdw)
        
    def leave(self, **kwds):
        """Leaves the mode: called by `ModeStack.pop()` etc."""
        if not self._is_active():
            self._discard_overlays()

        app = self.doc.app
        app.brushmodifier.blend_mode_changed -= self._blend_mode_changed_cb
       #self._stop_task_queue_runner(complete=True)
        super(ClosefillMode, self).leave(**kwds)  # supercall will commit

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
            super(ClosefillMode, self).checkpoint(flush=flush, **kwargs)
        else:
            # Queue a re-rendering with any new brush data
            # No supercall
            self._queue_draw_buttons()
            self._queue_redraw_all_nodes()

    def _start_new_capture_phase(self, rollback=False):
        """Let the user capture a new ink stroke"""
        if rollback:
            # TODO: for future expansion. something needed here.
           #self._stop_task_queue_runner(complete=False)
           pass
        else:
           #self._stop_task_queue_runner(complete=True)
           pass

        self.options_presenter.target = self
        self._queue_draw_buttons()
        self._queue_redraw_all_nodes()
        self._reset_nodes()
        self._reset_capture_data()
        self._reset_adjust_data()
        self.phase = _Phase.CAPTURE

    ## Properties

    # Cursor properties: Needed when device cursor move out from canvas.
    @property
    def inactive_cursor(self):
        return None

    @property
    def active_cursor(self):
        return self._current_override_cursor

    ## Raw event handling (prelight & zone selection in adjust phase)

    def button_press_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        opts = self.get_options_widget()
        if not ((tdw.is_sensitive and current_layer.get_paintable()) 
                    or opts.make_new_layer):
            return False

        if self.fill_method_option == _FillMethod.FLOOD_FILL:
            # Flood-fill does not need drag facility.
            return False

        self._update_zone_and_target(tdw, event.x, event.y)
        self._update_current_node_index()
        if self.phase == _Phase.ADJUST:
            if self.zone in (_EditZone.REJECT_BUTTON,
                             _EditZone.ACCEPT_BUTTON):
                button = event.button
                if button == 1 and event.type == Gdk.EventType.BUTTON_PRESS:
                    self._click_info = (button, self.zone)
                    return False
                # FALLTHRU: *do* allow drags to start with other buttons
            elif self.zone == _EditZone.EMPTY_CANVAS:
                if (len(self.nodes) > 2):
                    self.do_fill_operation(
                        tdw.display_to_model(event.x, event.y),
                        self.fill_method_option
                    )
                self._start_new_capture_phase(rollback=False)
                assert self.phase == _Phase.CAPTURE
                self._click_info = None
                self._update_zone_and_target(tdw, event.x, event.y)
                self._update_current_node_index()
                return False
                # FALLTHRU: *do* start a drag
        elif self.phase == _Phase.CAPTURE:
            # XXX Not sure what to do here.
            # XXX Click to append nodes?
            # XXX  but how to stop that and enter the adjust phase?
            # XXX Click to add a 1st & 2nd (=last) node only?
            # XXX  but needs to allow a drag after the 1st one's placed.
            pass
        else:
            raise NotImplementedError("Unrecognized zone %r", self.zone)
        # Update workaround state for evdev dropouts
        self._button_down = event.button
        # Supercall: start drags etc
        return super(ClosefillMode, self).button_press_cb(tdw, event)

    def button_release_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        opts = self.get_options_widget()
        if not ((tdw.is_sensitive and current_layer.get_paintable()) 
                    or opts.make_new_layer):
            return False

        if self.fill_method_option == _FillMethod.FLOOD_FILL:
            self.do_fill_operation(
                tdw.display_to_model(event.x, event.y),
                self.fill_method_option
            )
            return False # cancel dragging
        else:
            if self.phase == _Phase.ADJUST:
                if self._click_info:
                    button0, zone0 = self._click_info
                    if event.button == button0:
                        if self.zone == zone0:
                            if zone0 == _EditZone.REJECT_BUTTON:
                                self._start_new_capture_phase(rollback=True)
                                assert self.phase == _Phase.CAPTURE
                            elif zone0 == _EditZone.ACCEPT_BUTTON:
                                self.do_fill_operation(
                                    None,
                                    self.fill_method_option
                                )
                                self._start_new_capture_phase(rollback=False)
                                assert self.phase == _Phase.CAPTURE
                        self._click_info = None
                        self._update_zone_and_target(tdw, event.x, event.y)
                        self._update_current_node_index()
                        return False
                # (otherwise fall through and end any current drag)
            elif self.phase == _Phase.CAPTURE:
                # XXX Not sure what to do here: see above
                # Update options_presenter when capture phase end
                self.options_presenter.target = self
            else:
                raise NotImplementedError("Unrecognized zone %r", self.zone)
            # Update workaround state for evdev dropouts
            self._button_down = None

            # We need dragging facility. use supercall here.
            return super(ClosefillMode, self).button_release_cb(tdw, event)

    def motion_notify_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False
        self._update_zone_and_target(tdw, event.x, event.y, event.state)
        return super(ClosefillMode, self).motion_notify_cb(tdw, event)

    def key_press_cb(self, win ,tdw, event):
        """Keyboard press handler, for overriding fill method
        and show settings to user, before actually fill(click) pixels.
        Other keyboard interactions should be done as Gtk.Action.
        """
        # We cannot detect shift key pressed from event.state, 
        # so use KEY_ constants.
        if event.keyval in self._MODIFIER_CHANGE_FILL:
            opts = self.options_presenter
            self._overridden_fill_method = opts.fill_method
            opts.override_fill_method(self.fill_method_option)
            self._update_cursor(tdw)

    def key_release_cb(self, win ,tdw, event):
        """Keyboard release handler, for overriding fill method.
        """
        if event.keyval in self._MODIFIER_CHANGE_FILL:
            assert self._overridden_fill_method is not None
            opts = self.options_presenter
            opts.override_fill_method(self._overridden_fill_method)
            self._overridden_fill_method = None
            self._update_cursor(tdw)

    def _blend_mode_changed_cb(self, old_mode, new_mode):
        """ Notify the blend mode has changed.
        mode arguments might be None, it means 'BlendModeNormal'
        :param new_blend_mode: newly entered mode. 
        """
        self._update_cursor(None, blendmode=new_mode)

    def _update_current_node_index(self):
        """Updates current_node_index from target_node_index & redraw"""
        new_index = self.target_node_index
        old_index = self.current_node_index
        if new_index == old_index:
            return
        self.current_node_index = new_index
        self.current_node_changed(new_index)
        self.options_presenter.target = self
        for i in (old_index, new_index):
            if i is not None:
                self._queue_draw_node(i)

    @lib.observable.event
    def current_node_changed(self, index):
        """Event: current_node_index was changed"""

    def _update_zone_and_target(self, tdw, x, y, state=0):
        """Update the zone and target node under a cursor position"""
        self._ensure_overlay_for_tdw(tdw)
        new_zone = _EditZone.EMPTY_CANVAS
        if self.phase == _Phase.ADJUST and not self.in_drag:
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
                    new_zone = btn_zone
                    break
            # Test nodes for a hit, in reverse draw order
            if new_zone == _EditZone.EMPTY_CANVAS:
                hit_dist = gui.style.DRAGGABLE_POINT_HANDLE_SIZE + 12
               #new_target_node_index = None
                for i, node in reversed(list(enumerate(self.nodes))):
                    node_x, node_y = tdw.model_to_display(node.x, node.y)
                    d = math.hypot(node_x - x, node_y - y)
                    if d > hit_dist:
                        continue
                    new_target_node_index = i
                    new_zone = _EditZone.CONTROL_NODE
                    break
            # Test End.

            if new_zone != _EditZone.CONTROL_NODE:
                new_target_node_index = None
            
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
        # Update the "real" inactive cursor too:
        self._update_cursor(tdw)

    def _update_cursor(self, tdw, blendmode=None):
        """Update self._override_cursor.
        :param blendmode: Used when blendmode callback.
                          At the time when blendmode changed callback,
                          we cannot get actual blendmode state from
                          brushinfo object. So use this.
        """
        cursors = self._cursors
        cursor = None
        fill_method = self.fill_method_option

        if blendmode:
            erase_pixel = blendmode.get_name() == "BlendModeEraser"
        else:
            erase_pixel = self.doc.model.brush.brushinfo.is_eraser()

        if fill_method == _FillMethod.FLOOD_FILL:
            if erase_pixel:
                cursor = cursors[self._CURSOR_CROSS_ERASER]
            else:
                cursor = cursors[self._CURSOR_CROSS]
        else:
            if self.phase == _Phase.ADJUST:
                if self.zone == _EditZone.CONTROL_NODE: 
                    cursor = cursors[self._CURSOR_ARROW]
                else:
                    if erase_pixel:
                        cursor = cursors[self._CURSOR_CROSS_ERASER]
                    else:
                        cursor = cursors[self._CURSOR_CROSS]
            elif self.phase == _Phase.CAPTURE:
                # Without this, ordinary brush cursor(circle) shown
                if erase_pixel:
                    cursor = cursors[self._CURSOR_ERASER]
                else:
                    cursor = cursors[self._CURSOR_PENCIL]

        print(cursor)

        if cursor is not self._current_override_cursor:
            if tdw is not None:
                tdws = (tdw)
            else:
                tdws = self._overlays.keys()

            for tdw in tdws:
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

    def _queue_draw_node(self, i):
        """Redraws a specific control node on all known view TDWs"""
        if len(self.nodes) < 2:
            return

        m = 3 # margin

        if i == len(self.nodes) - 1:
            cn = self.nodes[i - 1]
            nn = self.nodes[i] 
        elif i == len(self.nodes):
            # Used from `redraw_all_nodes`
            # to redraw the final(closing) segment.
            cn = self.nodes[i - 1]
            nn = self.nodes[0] 
        else:
            cn = self.nodes[i]
            nn = self.nodes[i+1]

        for tdw in self._overlays:
            cx, cy = tdw.model_to_display(cn.x, cn.y)
            nx, ny = tdw.model_to_display(nn.x, nn.y)
            if cx > nx:
                cx, nx = nx, cx
            if cy > ny:
                cy, ny = ny, cy

            x = math.floor(cx)
            y = math.floor(cy)
            w = math.ceil(nx - cx)
            h = math.ceil(ny - cy)
            tdw.queue_draw_area(x-m, y-m, w+m*2, h+m*2)
            
    def _queue_redraw_all_nodes(self):
        """Redraws all nodes on all known view TDWs"""

        # We need to redraw the final(closing) segment
        # of target area. so add +1 to `self.nodes` length.
        for i in xrange(len(self.nodes) + 1):
            self._queue_draw_node(i)

    ## Drag handling (both capture and adjust phases)

    def drag_start_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        if self.phase == _Phase.CAPTURE:
            self._reset_nodes()
            self._reset_capture_data()
            self._reset_adjust_data()
            node = self._get_event_data(tdw, event)
            self.nodes.append(node)
            self._queue_draw_node(0)
            self._last_node_evdata = (event.x, event.y)
            self._last_event_node = node
        elif self.phase == _Phase.ADJUST:
            if self.target_node_index is not None:
                node = self.nodes[self.target_node_index]
                self._dragged_node_start_pos = (node.x, node.y)
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)

    def drag_update_cb(self, tdw, event, dx, dy):
        self._ensure_overlay_for_tdw(tdw)
        if self.phase == _Phase.CAPTURE:
            node = self._get_event_data(tdw, event)
            evdata = (event.x, event.y)
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
                max_dist = self.MAX_INTERNODE_DISTANCE_MIDDLE
                if len(self.nodes) < 2:
                    max_dist = self.MAX_INTERNODE_DISTANCE_ENDS
                append_node = (
                    dist > max_dist 
                )
            if append_node:
                self.nodes.append(node)
                self._queue_draw_node(len(self.nodes)-1)
                self._last_node_evdata = evdata
            self._last_event_node = node
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
            opts = self.get_options_widget()

            if not self.nodes:
                return
            node = self._last_event_node
            # TODO: maybe rewrite the last node here so it's the right
            # TODO: distance from the end?
            if self.nodes[-1] is not node:
                self.nodes.append(node)

            if (len(self.nodes) > 2
                    and (opts.fill_immidiately
                        or opts.fill_method == _FillMethod.LASSO_FILL)):
                self.do_fill_operation(
                    None,
                    self.fill_method_option
                )
                self.nodes = []

            self._reset_capture_data()
            self._reset_adjust_data()
            if len(self.nodes) > 1:
                self.phase = _Phase.ADJUST
                self._queue_redraw_all_nodes()
                self._queue_draw_buttons()
            else:
                self._reset_nodes()
                tdw.queue_draw()
        elif self.phase == _Phase.ADJUST:
            self._dragged_node_start_pos = None
            self._queue_draw_buttons()
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)

    ## Interrogating events

    def _get_event_data(self, tdw, event):
        x, y = tdw.display_to_model(event.x, event.y)
        return _Node(
            x=x, y=y,
        )

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
        return self.options_presenter

    def update_node(self, i, **kwargs):
        """Updates properties of a node, and redraws it"""
        changing_pos = bool({"x", "y"}.intersection(kwargs))
        oldnode = self.nodes[i]
        if changing_pos:
            self._queue_draw_node(i)
        self.nodes[i] = oldnode._replace(**kwargs)
        if changing_pos:
            self._queue_draw_node(i)

    ## Close and fill
    @property
    def fill_method_option(self):
        """Wrapper property of fill method option
        """
        overridden = self._overridden_fill_method
        if overridden is None:
            opts = self.get_options_widget()
            return opts.fill_method
        elif overridden == _FillMethod.FLOOD_FILL:
            return _FillMethod.CLOSED_AREA_FILL
        else:
            return _FillMethod.FLOOD_FILL

    # Do close_and_fill or lasso_fill.
    # Flood_fill is done at button_release_cb.
    @gui.drawwindow.with_wait_cursor
    def do_fill_operation(self, targ_color_pos, fill_method):
        app = self.doc.app
        pref = app.preferences
        model = self.doc.model
        opts = self.options_presenter
        nodes = self.nodes
        color = app.brush_color_manager.get_color().get_rgb()
        erase_pixel = model.brush.brushinfo.is_eraser()
        make_new_layer = opts.make_new_layer

        if fill_method == _FillMethod.FLOOD_FILL:
            assert targ_color_pos is not None
            x, y = targ_color_pos
            model.flood_fill(
                x, y, color,
                tolerance=opts.tolerance,
                sample_merged=opts.sample_merged,
                make_new_layer=make_new_layer,
                # keyword arguments
                dilation_size=opts.dilation_size,
                progress_level=opts.gap_level,
                erase_pixel=erase_pixel,
                fill_all_holes=opts.fill_all_holes,
                # XXX Debug arguments
                show_flag=opts.show_flag,
                tile_output=opts.tile_output
            )
        elif fill_method == _FillMethod.CLOSED_AREA_FILL:
            assert(len(self.nodes) > 2)
            cmd = lib.command.ClosedAreaFill(
                model, nodes, 
                color,
                opts.tolerance,
                opts.sample_merged,
                make_new_layer,
                opts.dilation_size,
                # kwds params
                targ_color_pos=targ_color_pos,
                progress_level=opts.gap_level,
                erase_pixel=erase_pixel,
                reject_perimeter=opts.reject_perimeter,
                fill_all_holes=opts.fill_all_holes,
                # debug options
                show_flag = opts.show_flag,
                tile_output = opts.tile_output
            )
            model.do(cmd)
        elif fill_method == _FillMethod.LASSO_FILL:
            assert(len(self.nodes) > 2)
            cmd = lib.command.LassoFill(
                model, nodes, 
                color,
                opts.tolerance,
                opts.sample_merged,
                make_new_layer,
                opts.dilation_size,
                # kwds params
                erase_pixel=erase_pixel,
                fill_all_holes=opts.fill_all_holes,
                # debug options
                show_flag = opts.show_flag,
                tile_output = opts.tile_output
            )
            model.do(cmd)
        else:
            assert("unknown fill method:%s" % str(opts.fill_method))

        opts.make_new_layer = False

class Overlay (gui.overlays.Overlay):
    """Overlay for an InkingMode's adjustable points"""

    def __init__(self, mode, tdw):
        super(Overlay, self).__init__()
        self._mode = weakref.proxy(mode)
        self._tdw = weakref.proxy(tdw)
        self._button_pixbuf_cache = {}
        self.accept_button_pos = None
        self.reject_button_pos = None

    def update_button_positions(self):
        """XXX Code Duplication from inktool.py
        Recalculates the positions of the mode's buttons."""
        nodes = self._mode.nodes
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

        # XXX Modified part:
        # We now deal with unmodifiable closed region for closefill, 
        # so just enough to get extream position of nodes here.

        # Original Code:
       #for i, node in enumerate(nodes):
       #    x, y = self._tdw.model_to_display(node.x, node.y)
       #    fixed.append(_LayoutNode(x, y))

        node = nodes[0]
        min_x, min_y = self._tdw.model_to_display(node.x, node.y)
        max_x, max_y = min_x, min_y

        for node in nodes[1:]:
            x, y = self._tdw.model_to_display(node.x, node.y)
            min_x = min(x, min_x)
            min_y = min(y, min_y)
            max_x = max(x, max_x)
            max_y = max(y, max_y)

        fixed.append(_LayoutNode(min_x, min_y))
        fixed.append(_LayoutNode(max_x, max_y))
        # XXX Modified part ends.

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
        mode = self._mode
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        alloc = self._tdw.get_allocation()
        for i, node in enumerate(mode.nodes):
            x, y = self._tdw.model_to_display(node.x, node.y)
            yield (i, node, x, y)

    def paint(self, cr):
        """Draw adjustable nodes to the screen"""
        # Control nodes
        mode = self._mode
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        alloc = self._tdw.get_allocation()
        color = gui.style.EDITABLE_ITEM_COLOR
        cr.save()
        cr.set_source_rgb(*color.get_rgb())
        for i, node, x, y in self._get_onscreen_nodes():
            if i == 0:
                cr.move_to(x,y)
            else:
                cr.line_to(x,y)

        if mode.phase == _Phase.ADJUST: 
            cr.close_path()

        gui.drawutils.render_drop_shadow(cr)
        cr.stroke()
        cr.restore()

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

class OptionsPresenter(Gtk.Grid):
    """Configuration widget for the flood fill tool"""

    _SPACING = 6

    def __init__(self):
        Gtk.Grid.__init__(self)
        self._update_ui = True
        self.set_row_spacing(self._SPACING)
        self.set_column_spacing(self._SPACING)
        from application import get_app
        self.app = get_app()
        self._mode_ref = None
        prefs = self.app.preferences
        row = 0

        def generate_label(text, tooltip, row, grid=self, alignment=(1.0, 0.5)):
            # Generate a label and return it.
            label = Gtk.Label()
            label.set_markup(text)
            label.set_tooltip_text(tooltip)
            label.set_alignment(*alignment)
            label.set_hexpand(False)
            grid.attach(label, 0, row, 1, 1)
            return label
            
        def generate_spinbtn(row, grid, adj):
            # Generate a spinbtn, and return it.
            spinbtn = Gtk.SpinButton()
            spinbtn.set_hexpand(True)
            spinbtn.set_adjustment(adj)
            # We need spinbutton focus event callback, to disable/re-enable
            # Keyboard manager for them.            
            spinbtn.connect("focus-in-event", self._spin_focus_in_cb)
            spinbtn.connect("focus-out-event", self._spin_focus_out_cb)
            grid.attach(spinbtn, 1, row, 1, 1)
            return spinbtn
        
        label = generate_label(
            _("Fill method:"),
            _("The fill method to use"),
            0,
            self,
            (1.0, 0.1)
        )
        vbox = Gtk.VBox()
        label_list = _FillMethod.LABELS
        btndict = {}
        radio_base = None
        for i in range(len(label_list)):
            label = label_list[i]
            radio = Gtk.RadioButton.new_with_label_from_widget(
                radio_base,
                label
            )
            radio.connect("toggled", self._fillmethod_toggled_cb, i)
            vbox.pack_start(radio, False, False, 0)
            btndict[i] = radio
            if radio_base is None:
                radio_base = radio

        self.attach(vbox, 1, row, 1, 1)
        # `Fill method` and `Share setting between methods` 
        # should be set from app.preferences directly.
        # Otherwise, use get_pref_value wrapper method 
        # And they are refreshed at the end of this method.
        method = prefs.get(_Prefs.FILL_METHOD_PREF ,
                           _Prefs.DEFAULT_FILL_METHOD)
        active_btn = btndict.get(method, radio_base)
        active_btn.set_active(True)
        self._fillmethod_buttons = btndict

        row += 1
        label = generate_label(
            _("Dilation Size:"),
            _("How many pixels the filled area to be dilated."),
            row
        )

        adj = Gtk.Adjustment(
            value=_Prefs.DEFAULT_DILATION_SIZE, lower=0.0,
            upper=lib.mypaintlib.TILE_SIZE / 2 - 1,
            step_increment=1, page_increment=4,
            page_size=0
        )
        adj.connect("value-changed", self._dilation_size_changed_cb)
        self._dilation_size_adj = adj
        generate_spinbtn(row, self, adj)

        row += 1
        label = generate_label(
            _("Tolerance:"),
            _("How much pixel colors or transparency are allowed "
              "to vary from the start\n"
              "before Flood Fill will refuse to fill them"),
            row
        )

        adj = Gtk.Adjustment(
            value=_Prefs.DEFAULT_TOLERANCE, lower=0.0,
            upper=1.0,
            step_increment=0.05, page_increment=0.05,
            page_size=0
        )
        adj.connect("value-changed", self._tolerance_changed_cb)
        self._tolerance_adj = adj
        scale = Gtk.Scale()
        scale.set_hexpand(True)
        scale.set_adjustment(adj)
        scale.set_draw_value(False)
        self.attach(scale, 1, row, 1, 1)

        row += 1
        label = generate_label(
            _("Gap-closing level:"),
            _("Specifying the size of closing the gap of contour,in 6 level."),
            row
        )
        # Gap closing level is from 0 to 6.
        # It is actually exponent of power of 2, so, 
        # level 0 is 2^0==1px(no gap closing), and level 6(maximum) is 2^6==64px.
        # And, it is `radius`, so gap-closing-level 6 would stop 
        # maximum 127px diameter hole.
        adj = Gtk.Adjustment(
            value=_Prefs.DEFAULT_GAP_LEVEL, lower=0,
            upper=6,
            step_increment=1, page_increment=1,
            page_size=0
        )
        adj.connect("value-changed", self._gap_level_changed_cb)
        self._gap_level_adj = adj
        self._gap_level_spin = generate_spinbtn(row, self, adj)

        # XXX `Sample Merged` and `New Layer (once)` is 
        # almost same as fill.py
        # But closefill will merged into fill.py in future,
        # so this is not problem.
        row += 1
        label = generate_label(
            _("Source:"),
            _("Which visible layers should be filled"),
            row
        )
        text = _("Sample Merged")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            _("When considering which area to fill, use a\n"
              "temporary merge of all the visible layers\n"
              "underneath the current layer")
        )
        self.attach(checkbut, 1, row, 1, 1)
        checkbut.set_active(_Prefs.DEFAULT_SAMPLE_MERGED)
        checkbut.connect("toggled", self._sample_merged_toggled_cb)
        self._sample_merged_toggle = checkbut

        row += 1
        label = generate_label(
            _("Target:"),
            _("Where the output should go"),
            row
        )
        self.attach(label, 0, row, 1, 1)

        text = _("New Layer (once)")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            _("Create a new layer with the results of the fill.\n"
              "This is turned off automatically after use.")
        )
        self.attach(checkbut, 1, row, 1, 1)
        # Set default embedded value for `New layer (once)` always.
        # There is no user configurable value for it.
        checkbut.set_active(_Prefs.DEFAULT_MAKE_NEW_LAYER)
        self._make_new_layer_toggle = checkbut

        
        #### Advanced options
        
        row += 1
        exp = Gtk.Expander()
        adv_grid = Gtk.Grid()
        adv_grid.set_row_spacing(self._SPACING)
        adv_grid.set_column_spacing(self._SPACING)
        exp.set_label(_("Advanced Options..."))
        exp.set_use_markup(False)
        self.attach(exp, 0, row, 2, 1)
        
        ## Common advanced options 
        # "Fill all holes"
        adv_row = 0
        label = generate_label(
            _("General options:"),
            _("Option which is common between all fill methods."),
            adv_row,
            adv_grid
        )
        text = _("Fill all holes")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            _("Fill all small holes and detached contour area\n"
              "within large filled area.")
        )
        adv_grid.attach(checkbut, 1, adv_row, 1, 1)
        checkbut.set_active(_Prefs.DEFAULT_FILL_ALL_HOLES)
        checkbut.connect("toggled", self._fill_all_holes_toggled_cb)
        self._fill_all_holes_toggle = checkbut

        ## Dedicated advanced options
        # "Fill immidiately area"
        # This is only for close-and-fill. 
        adv_row += 1
        label = generate_label(
            _("Closed area fill:"),
            _("Dedicated options for `Closed area fill` method."),
            adv_row,
            adv_grid
        )
        text = _("Fill immidiately")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            _("Without any color pixel assignment, \n"
              "fill closed transparent pixels immidiately.")
        )
        adv_grid.attach(checkbut, 1, adv_row, 1, 1)
        checkbut.set_active(_Prefs.DEFAULT_FILL_IMMIDIATELY)
        checkbut.connect("toggled", self._fill_immidiately_toggled_cb)
        self._fill_immidiately_toggle = checkbut

        # Factor of pixel rejecting perimeter.
        adv_row += 1
        label = generate_label(
            _("Rejecting factor:"),
            _("Specifying the perimeter factor of to be removed pixels\n"
              "which is unintentionally filled around contour."),
            adv_row,
            adv_grid
        )
        adv_grid.attach(label, 0, adv_row, 1, 1)
        # As a default, the rejecting perimeter is , 
        # (1 << gap-closing-level) * 4 * 2.
        # This `2` is the `factor`, and 2 is default value.
        # Pixel areas which have smaller perimeter than 
        # this threshold perimeter should be rejected.
        # This default value would be enough in most case practically.
        # So, assigning too large factor would ruin filled result.
        adj = Gtk.Adjustment(
            value=_Prefs.DEFAULT_REJECT_FACTOR, lower=0,
            upper=4.0, 
            step_increment=1, page_increment=1,
            page_size=0
        )
        adj.connect("value-changed", self._reject_factor_changed_cb)
        self._reject_factor_adj = adj
        scale = Gtk.Scale()
        scale.set_hexpand(True)
        scale.set_adjustment(adj)
        scale.set_draw_value(False)
        adv_grid.attach(scale, 1, adv_row, 1, 1)
        self._reject_factor_scale = scale

        # Alpha transparency threshold.
        adv_row += 1
        label = generate_label(
            _("Alpha threshold:"),
            _("Specifying pixel transparency threshold.\n"
              "Some brush preset produce nearly invisible pixels,\n"
              "which is difficult to be removed with tolerance option.\n"),
            adv_row,
            adv_grid
        )
        adv_grid.attach(label, 0, adv_row, 1, 1)
        # As a default, the rejecting perimeter is , 
        # (1 << gap-closing-level) * 4 * 2.
        # This `2` is the `factor`, and 2 is default value.
        # Pixel areas which have smaller perimeter than 
        # this threshold perimeter should be rejected.
        # This default value would be enough in most case practically.
        # So, assigning too large factor would ruin filled result.
        adj = Gtk.Adjustment(
            value=_Prefs.DEFAULT_ALPHA_THRESHOLD, lower=0,
            upper=1.0, 
            step_increment=1, page_increment=1,
            page_size=0
        )
        adj.connect("value-changed", self._alpha_threshold_changed_cb)
        self._alpha_threshold_adj = adj
        scale = Gtk.Scale()
        scale.set_hexpand(True)
        scale.set_adjustment(adj)
        scale.set_draw_value(False)
        adv_grid.attach(scale, 1, adv_row, 1, 1)
        self._alpha_threshold_scale = scale

        # Preferences related.
        adv_row += 1
        label = generate_label(
            _("Preference:"),
            _("About preference values."),
            adv_row,
            adv_grid
        )
        text = _("Share between methods")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            _("Share same setting values between each filling methods.")
        )
        adv_grid.attach(checkbut, 1, adv_row, 1, 1)
        # This option access to app.preferences directly.
        active = prefs.get(_Prefs.SHARE_SETTING_PREF,
                           _Prefs.DEFAULT_SHARE_SETTING)
        checkbut.set_active(bool(active))
        checkbut.connect("toggled", self._share_setting_toggled_cb)
        self._share_setting_toggle = checkbut

        # XXX Debug buttons
        adv_row += 1
        label = Gtk.Label()
        label.set_markup(_("Debug:"))
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        adv_grid.attach(label, 0, adv_row, 1, 1)

        text = "Show flag image"
        checkbut = Gtk.CheckButton.new_with_label(text)
        adv_grid.attach(checkbut, 1, adv_row, 1, 1)
        self.show_flag_btn = checkbut

        adv_row += 1
        text = "Tile output"
        checkbut = Gtk.CheckButton.new_with_label(text)
        adv_grid.attach(checkbut, 1, adv_row, 1, 1)
        self.tile_output_btn = checkbut

        adv_row += 1
        btn = Gtk.Button()
        btn.set_label(_("Do debug"))
        btn.connect("clicked", self._debug_clicked_cb)
        btn.set_hexpand(True)
        adv_grid.attach(btn, 0, adv_row, 2, 1)
        # XXX Debug buttons End
        exp.add(adv_grid)

        self._update_ui = False

        # After all widgets are setup, refresh them.
        self._refresh_ui_from_fillmethod(self.fill_method, force=True)

    def _refresh_ui_from_fillmethod(self, method, force=False):
        """Refresh user interface widgets.
        You must modify this method when adding/removing
        option widgets.
        """
        prev = self._update_ui
        self._update_ui = True

        # Setting sensitive state of some options
        # which is not used paticular fill method.
        self._gap_level_spin.set_sensitive(method!=_FillMethod.LASSO_FILL)
        self._fill_immidiately_toggle.set_sensitive(
            method==_FillMethod.CLOSED_AREA_FILL
        )
        self._reject_factor_scale.set_sensitive(
            method!=_FillMethod.LASSO_FILL
        )

        if not self.share_setting or force:
            value = self.get_pref_value(
                _Prefs.DILATION_SIZE_PREF,
                _Prefs.DEFAULT_DILATION_SIZE
            )
            self._dilation_size_adj.set_value(float(value))

            value = self.get_pref_value(
                _Prefs.TOLERANCE_PREF,
                _Prefs.DEFAULT_TOLERANCE
            )
            self._tolerance_adj.set_value(float(value))

            value = self.get_pref_value(
                _Prefs.GAP_LEVEL_PREF,
                _Prefs.DEFAULT_GAP_LEVEL
            )
            self._gap_level_adj.set_value(int(value))

            active = self.get_pref_value(
                _Prefs.SAMPLE_MERGED_PREF,
                _Prefs.DEFAULT_SAMPLE_MERGED
            )
            self._sample_merged_toggle.set_active(bool(active))

            active = self.get_pref_value(
                _Prefs.FILL_IMMIDIATELY_PREF ,
                _Prefs.DEFAULT_FILL_IMMIDIATELY
            )
            self._fill_immidiately_toggle.set_active(bool(active))

            value = self.get_pref_value(
                _Prefs.REJECT_FACTOR_PREF,
                _Prefs.DEFAULT_REJECT_FACTOR
            )
            self._reject_factor_adj.set_value(float(value))
            
            value = self.get_pref_value(
                _Prefs.ALPHA_THRESHOLD_PREF,
                _Prefs.DEFAULT_ALPHA_THRESHOLD
            )
            self._alpha_threshold_adj.set_value(float(value))
            
            active = self.get_pref_value(
                _Prefs.FILL_ALL_HOLES_PREF,
                _Prefs.DEFAULT_FILL_ALL_HOLES
            )
            self._fill_all_holes_toggle.set_active(bool(active))

        self._update_ui = prev

    def _get_setting_name(self, pref_name):
        if not self.share_setting:
            prefix = _Prefs.PREFIX[self.fill_method]
        else:
            prefix = _Prefs.PREFIX[_FillMethod.FLOOD_FILL]
        return "%s.%s" % (prefix, pref_name)
                              
    def get_pref_value(self, pref_name, default_value):
        """Wrapper method for getting preference value.
        When `Save setting for each fill methods` is enabled,
        preference values would be different between methods.
        """
        setting_name = self._get_setting_name(pref_name)
        return self.app.preferences.get(
            setting_name, 
            default_value
        )

    def set_pref_value(self, pref_name, new_value):
        if not self._update_ui:
            setting_name = self._get_setting_name(pref_name)
            self.app.preferences[setting_name] = new_value 

    # XXX Code duplication : some handlers/properties just copied from
    # gui/fill.py

    @property
    def target(self):
        if self._mode_ref is not None:
            return self._mode_ref()
        else:
            print('no target')

    @target.setter
    def target(self, mode):
        self._mode_ref = weakref.ref(mode)

    @property
    def tolerance(self):
        return float(self._tolerance_adj.get_value())

    @property
    def make_new_layer(self):
        make_new_layer = self._make_new_layer_toggle.get_active()
        rootstack = self.app.doc.model.layer_stack
        if not rootstack.current.get_fillable():
            # Force make new layer when cannot fill.
            make_new_layer = True
        return make_new_layer

    @make_new_layer.setter
    def make_new_layer(self, value):
        self._make_new_layer_toggle.set_active(bool(value))

    @property
    def sample_merged(self):
        return self._sample_merged_toggle.get_active()

    @property
    def dilation_size(self):
        return int(math.floor(self._dilation_size_adj.get_value()))

    @property
    def fill_immidiately(self):
        return self._fill_immidiately_toggle.get_active()

    @property
    def fill_method(self):
        for idx, btn in self._fillmethod_buttons.iteritems():
            if btn.get_active():
                return idx

        # Return default if no any button activated.
        return self.app.preferences[_Prefs.FILL_METHOD_PREF]

    def override_fill_method(self, method):
        """For overriding fill method, only shows setting.
        """
        assert method >= 0
        assert method < 3
        btn = self._fillmethod_buttons[method]
        self._update_ui = True
        btn.set_active(True)
        # self._update_ui flag cancels callback of method button.
        # So, update ui widgets manually , with avoiding change
        # app.preferences.
        self._refresh_ui_from_fillmethod(method)
        self._update_ui = False

    @property
    def gap_level(self):
        return int(math.floor(self._gap_level_adj.get_value()))

    @property
    def reject_perimeter(self):
        f = self._reject_factor_adj.get_value()
        g = self.gap_level
        # 1 << gap_level == ridge length of progress level pixel.
        # ridge-length * 4 == maximum perimeter of progress level pixel
        # perimeter * factor(as a default, 2.0) == reject_perimeter
        return (1 << g) * 4 * f

    @property
    def share_setting(self):
        return self._share_setting_toggle.get_active()
        
    @property
    def alpha_threshold(self):
        return self._alpha_threshold_adj.get_value()
        
    @property
    def fill_all_holes(self):
        return self._fill_all_holes_toggle.get_active()

    # Option event handlers
    # These event should use self.set_pref_value method
    # to deal with `share setting` advanced option.
    def _tolerance_changed_cb(self, adj):
        self.set_pref_value(
            _Prefs.TOLERANCE_PREF,
            adj.get_value()
        )

    def _sample_merged_toggled_cb(self, btn):
        self.set_pref_value(
            _Prefs.SAMPLE_MERGED_PREF,
            btn.get_active()
        )

    def _dilation_size_changed_cb(self, adj):
        self.set_pref_value(
            _Prefs.DILATION_SIZE_PREF,
            adj.get_value()
        )

    def _gap_level_changed_cb(self, adj):
        self.set_pref_value(
            _Prefs.GAP_LEVEL_PREF,
            adj.get_value()
        )

    def _reject_factor_changed_cb(self, adj):
        self.set_pref_value(
            _Prefs.REJECT_FACTOR_PREF,
            adj.get_value()
        )

    def _fill_immidiately_toggled_cb(self, btn):
        self.set_pref_value(
            _Prefs.FILL_IMMIDIATELY_PREF,
            btn.get_active()
        )

    def _alpha_threshold_changed_cb(self, adj):
        self.set_pref_value(
            _Prefs.ALPHA_THRESHOLD_PREF,
            adj.get_value()
        )
        
    def _fill_all_holes_toggled_cb(self, btn):
        self.set_pref_value(
            _Prefs.FILL_ALL_HOLES_PREF,
            btn.get_active()
        )
        
    # `Fill method` and `Share setting between methods`
    # directly set app.preferences.
    def _fillmethod_toggled_cb(self, btn, idx):
        # Only active button should be refresh options.
        # Otherwise, the refresh process woule be done twice.
        if not self._update_ui and btn.get_active():
            self.app.preferences[_Prefs.FILL_METHOD_PREF] = idx
            self._refresh_ui_from_fillmethod(idx)
            target = self.target
            target.checkpoint(flush=True)# Reset overlay contents with this.
            target._update_cursor(None)  # Fill mode changed, so update cursor.

    def _share_setting_toggled_cb(self, btn):
        if not self._update_ui:
            self.app.preferences[_Prefs.SHARE_SETTING_PREF] = btn.get_active()
            self._refresh_ui_from_fillmethod(self.fill_method, force=True)

    def _spin_focus_in_cb(self, widget, event):
        if self._update_ui:
            return
        kbm = self.app.kbm
        kbm.enabled = False

    def _spin_focus_out_cb(self, widget, event):
        kbm = self.app.kbm
        kbm.enabled = True        

    def _reset_clicked_cb(self, button):
        # Use self.get_pref_value method 
        # to deal with `share setting` advanced option.
        self._tolerance_adj.set_value(
            self.get_pref_value(
                _Prefs.TOLERANCE_PREF,
                _Prefs.DEFAULT_TOLERANCE)
        )
        self._dilation_size_adj.set_value(
            self.get_pref_value(
                _Prefs.DILATION_SIZE_PREF,
                _Prefs.DEFAULT_DILATION_SIZE)
        )
        self._reject_factor_adj.set_value(
            self.get_pref_value(
                _Prefs.REJECT_FACTOR_PREF,
                _Prefs.DEFAULT_REJECT_FACTOR)
        )

        self._sample_merged_toggle.set_active(
            self.get_pref_value(
                _Prefs.SAMPLE_MERGED_PREF,
                _Prefs.DEFAULT_SAMPLE_MERGED)
        )

        # make_new_layer is independent option, not per method option.
        self._make_new_layer_toggle.set_active(
            _Prefs.DEFAULT_MAKE_NEW_LAYER)

        # `Fill method` cannot be resetted.

    def _debug_clicked_cb(self, btn):
        # XXX Debug button
        import json
        infobase = "/home/dothiko/python/test/mypainttest/fill_close"
        infodir = "%s/tiles" % infobase

        with open("%s/info" % infodir, 'r') as ifp:
            info = json.load(ifp)
       #min_x, min_y, w, h, tw, th = info['bbox']
        nodes = [_Node(*c) for c in info['nodes']]
        targ_pos = info.get('targ_color_pos', None)
        tolerance = info.get('tolerance', 0.05)
        level = info.get('level', 2)

        print("-- loaded nodes")
        print("targ_pos %s" % str(targ_pos))
        print("tolerance %s" % str(tolerance))

        mode = self.target
        mode.nodes = nodes
        mode.do_fill_operation(targ_pos)

    # XXX Debug property
    @property
    def show_flag(self):
        return self.show_flag_btn.get_active()

    @property
    def tile_output(self):
        return self.tile_output_btn.get_active()

