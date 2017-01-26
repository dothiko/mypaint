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
import weakref

from gettext import gettext as _
from lib import brushsettings
from gi.repository import GLib
from gi.repository import GdkPixbuf
import cairo

import gui.mode
from drawutils import spline_4p
from lib import mypaintlib
from gui.exinktool import *
from gui.exinktool import _LayoutNode
from gui.linemode import *
from lib.command import Command
from gui.ui_utils import *
from gui.stamps import *
from lib.color import HCYColor, RGBColor
import lib.helpers
from gui.oncanvas import *
import gui.stampeditor

## Module settings

## Functions

def _get_iconview_data(iconview, idx):
    """A utility method for iconview, to get currently selected item
    which is contained into current selected icon(store).

    :param iconview: the iconview
    :param idx: the index in a tuple. that tuple is registered
        to store instance with append() method.
    """
    path = iconview.get_selected_items()[0]
    store = iconview.get_model()
    iter = store.get_iter(path)
    return store.get(iter, idx)[0]

## Class defs

_NODE_FIELDS = ("x", "y", "angle", "scale_x", "scale_y", "picture_index")

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
    ROTATE = 101         #: Rotate with mouse drag 
    SCALE  = 102         #: Scale with mouse drag 
    ROTATE_BY_HANDLE = 103 #: Rotate with handle of GUI
    SCALE_BY_HANDLE = 104  #: Scale  with handle of GUI
    CALL_BUTTONS = 106     #: call buttons around clicked point. 
    CAPTURE_IMAGE = 107    #: capturing layer image(with select tool)

class _EditZone(EditZoneMixin):
    """Enumeration of what the pointer is on in phases"""
   #CONTROL_HANDLE_0 = 100
   #CONTROL_HANDLE_1 = 101
   #CONTROL_HANDLE_2 = 102
   #CONTROL_HANDLE_3 = 103
   #CONTROL_HANDLE_BASE = 100
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





class StampMode (OncanvasEditMixin,
        HandleNodeUserMixin):

    ## Metadata properties

    ACTION_NAME = "StampMode"

   #permitted_switch_actions = set(gui.mode.BUTTON_BINDING_ACTIONS).union([
   #        'RotateViewMode',
   #        'ZoomViewMode',
   #        'PanViewMode',
   #        'SelectionMode',
   #    ])
    _disable_switch_actions=set(gui.mode.BUTTON_BINDING_ACTIONS).union([
            'RotateViewMode',
            'ZoomViewMode',
            'PanViewMode',
            "SelectionMode",
        ])
    _enable_switch_actions = _disable_switch_actions

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


    # _stamp_param stores scale/angle value for
    # each picture. This attribute must be shared in instances.
    _stamp_param = {}

    # A flag indicates whether remember stamp parameter or not
    REMEMBER_PARAM_ENABLED = False

    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):
        super(StampMode, self).__init__(**kwargs)
        self.current_node_handle = -1
        self.forced_button_pos = False

        self.scaleval=1.0

        # _current_picture_id is the picture index within current stamp.
        # 
        # This is used when a new node created, nothing to do with
        # the picture index of currently active(already editing) node.
        self._current_picture_id = -1

    def _reset_adjust_data(self):
        super(StampMode, self)._reset_adjust_data()
        self._selection_area = None

    @property
    def stamp(self):
        return self._app.stamp_manager.get_current()

    @property
    def current_picture_id(self):
        if self.stamp != None:
            return self.stamp.validate_picture_id(self._current_picture_id)
        return -1

    @current_picture_id.setter
    def current_picture_id(self, new_id):
        self._current_picture_id = new_id


    def set_stamp(self, stamp):
        """ Called from OptionPresenter, 
        This is to make stamp property as if it is read-only.
        """
        old_stamp = self.stamp
        if old_stamp:
            old_stamp.leave(self.doc)

        self._app.stamp_manager.set_current(stamp)

        stamp.enter(self.doc)
        stamp.initialize_phase(self)

    ## Stamp default parameter related
    def record_current_stamp_param(self, scale_x, scale_y, angle,
            picture_id=-1):
        """ Called from node_drag_stop_cb, 
        when phase is _Phase.SCALE*/ANGLE*
        """
        assert self.stamp is not None
        if picture_id == -1:
            picture_id = self.current_picture_id

        cur_dic = StampMode._stamp_param.get(self.stamp, None)
        if cur_dic == None and StampMode.REMEMBER_PARAM_ENABLED:
            cur_dic = {}
            StampMode._stamp_param[self.stamp] = cur_dic

        if cur_dic != None:
            cur_dic[picture_id] = (scale_x, scale_y, angle)
        

    def get_default_stamp_param(self):
        """Get default stamp parameter from
        current stamp of current picture.
        
        This method is called from node_drag_start_cb, 
        when phase is _Phase.CAPTURE/ADJUST and
        Editzone is _EditZone.SOURCE_AREA/CONTROL_NODE
        """
        assert self.stamp is not None
        assert self.stamp.picture_count > 0

        cur_id = self.current_picture_id
        default_param = (1.0, 1.0, 0.0)
        
        cur_dic = StampMode._stamp_param.get(self.stamp, None)
        if cur_dic != None:
            return cur_dic.get(cur_id, default_param)
        
        return default_param

    ## Status methods

    def is_adjusting_phase(self):
        """To know whether current phase is node adjusting phase.
        this method should be overriden in deriving classes.
        """
        return self.phase in (_Phase.ADJUST,
                              _Phase.ADJUST_POS,
                              _Phase.ROTATE,
                              _Phase.SCALE,
                              _Phase.ROTATE_BY_HANDLE,
                              _Phase.SCALE_BY_HANDLE)

    def enter(self, doc, **kwds):
        """Enters the mode: called by `ModeStack.push()` etc."""
        self._app = doc.app
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
        
    def stampwork_commit_all(self, abrupt=False):
        """ abrupt is ignored.
        """
        # We need that the target layer(current layer)
        # has surface and not locked. 
        if len(self.nodes) > 0:
            bbox = self.stamp.get_bbox(None, self.nodes[0])
            if bbox:
                sx, sy, w, h = bbox

                ex = sx + w
                ey = sy + h
                for cn in self.nodes[1:]:
                    tsx, tsy, tw, th = self.stamp.get_bbox(None, cn)
                    sx = min(sx, tsx)
                    sy = min(sy, tsy)
                    ex = max(ex, tsx + tw)
                    ey = max(ey, tsy + th)

                cmd = DrawStamp(self.doc.model,
                        self.stamp,
                        self.nodes,
                        (sx, sy, abs(ex-sx)+1, abs(ey-sy)+1))
                # Important: clearing nodes.
                # without this, stamps drawn twice.
                # this should be done prior to 
                # do(cmd)
                self.nodes = []

                self.doc.model.do(cmd)
            else:
                logger.warning("stamptool.commit_all encounter enpty bbox")


    def _start_new_capture_phase(self, rollback=False):
        """Let the user capture a new ink stroke"""
        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()

        if rollback:
            self._stop_task_queue_runner(complete=False)
            # Currently, this tool uses overlay(cairo) to preview stamps.
            # so we need no rollback action to the current layer.
            self.nodes = []
        else:
            self._stop_task_queue_runner(complete=True)
            self.stampwork_commit_all()

        if self.stamp:
            self.stamp.finalize_phase(self, rollback)
            
        self.options_presenter.target = (self, None)
        self._reset_adjust_data()
        self.phase = _Phase.CAPTURE

       #if self.stamp:
       #    self.stamp.initialize_phase(self)


    def _generate_overlay(self, tdw):
        return Overlay_Stamp(self, tdw)

    def _generate_presenter(self):
        return OptionsPresenter_Stamp()

    def _update_zone_and_target(self, tdw, x, y):
        """ Update the zone and target node under a cursor position 
        """
        if not self.stamp:
            return
        else:
            stamp = self.stamp


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
        pass

    def select_area_cb(self, selection_mode):
        """ Selection handler called from SelectionMode.
        This handler never called when no selection executed.

        CAUTION: you can not access the self.doc attribute here
        (it is disabled as None, with modestack facility)
        so you must use 'selection_mode.doc', instead of it.
        """
        app = selection_mode.doc.app
        if self.stamp:
            if self.phase in (_Phase.CAPTURE, 
                    _Phase.ADJUST, 
                    _Phase.CAPTURE_IMAGE):

                self._queue_draw_buttons() # to erase

                self._selection_area = selection_mode.get_min_max_pos_model(margin=0)
                self.phase = _Phase.CAPTURE_IMAGE

                self._queue_draw_buttons()
        else:
            app.show_transient_message(
                _("There is no any stamp activated."))

                

    ## Redraws

    def _search_target_node(self, tdw, x, y):
        """Overriding HandleNodeUserMixin method.
        Search a node, (i.e. stamp picture) which placed at (x, y) of screen. 
        If (x, y) is on a transformation handle of the picture,
        return index of that handle too.
        Otherwise (pointer hovers only on the picture, not handle), 
        return -1 as handle index.

        :return : a index of picture, with its handle index.
        :rtype tuple:
        """
        hit_dist = gui.style.DRAGGABLE_POINT_HANDLE_SIZE + 12
        new_target_node_index = None
        handle_idx = None
        stamp = self.stamp
        # Use reversed index, because "the top covering picture
        # should be manipulated prior to below one"
        nmax = len(self.nodes) - 1
        for i in xrange(nmax + 1):
            ri = nmax - i
            node = self.nodes[ri]
            handle_idx = stamp.get_handle_index(tdw, x, y, node,
                   gui.style.DRAGGABLE_POINT_HANDLE_SIZE)
            if handle_idx >= 0:
                new_target_node_index = ri
                if handle_idx >= 4:
                    handle_idx = None
                break

        return (new_target_node_index, handle_idx)

    def _queue_draw_node(self, i, offsets=None, tdws=None):
        """Redraws a specific control node on all known view TDWs"""

        if offsets:
            dx, dy = offsets
        else:
            if i in self.selected_nodes:
                dx, dy = self.drag_offset.get_model_offset()
            else:
                dx = dy = 0

        if tdws == None:
            tdws = self._overlays

        for tdw in tdws:
            self._queue_draw_stamp_node(tdw, self.nodes[i], dx, dy, 
                    i in self.selected_nodes)


    def _queue_draw_stamp_node(self, tdw, node, dx, dy, add_margin):
        if not self.stamp:
            return

        if add_margin:
            margin = gui.style.DRAGGABLE_POINT_HANDLE_SIZE + 4
        else:
            margin = 4

        bbox = self.stamp.get_bbox(tdw, node, dx, dy, margin=margin)

        if bbox:
            tdw.queue_draw_area(*bbox)


    def _queue_draw_buttons(self):
        """queuing draw buttons, and,
        if there is selection area, also queue it.
        """

        super(StampMode, self)._queue_draw_buttons()

        if self._selection_area:
            for tdw in self._overlays:
                sx, sy, ex, ey = gui.ui_utils.get_outmost_area(tdw, 
                        *self._selection_area, 
                        margin=gui.style.DRAGGABLE_POINT_HANDLE_SIZE+4)
                tdw.queue_draw_area(sx, sy, 
                        abs(ex - sx) + 1, abs(ey - sy) + 1)



   #def _queue_redraw_item(self): this method is ignored,
   # leave as the placeholder of OncanvasEditMixin.
   # because, in this class, drawing nodes same as drawing items.

    ## Raw event handling (prelight & zone selection in adjust phase)

    def mode_button_press_cb(self, tdw, event):

        if not self.stamp:
            return 

        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
        ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK

        if self.phase == _Phase.CAPTURE:
            self.stamp.initialize_phase(self)
        elif self.phase == _Phase.ADJUST_POS:
            self._queue_draw_buttons()
            assert event.button == 1
            retval = super(StampMode, self).mode_button_press_cb(tdw, event)
            if self.current_node_handle != None:
                if ctrl_state:
                    self.phase = _Phase.ROTATE_BY_HANDLE
                else:
                    self.phase = _Phase.SCALE_BY_HANDLE
            return retval
        elif self.phase in (_Phase.ROTATE,
                            _Phase.SCALE,
                            _Phase.ROTATE_BY_HANDLE,
                            _Phase.SCALE_BY_HANDLE):
            # XXX Not sure what should be done here yet.
            # but remained for future use.
            pass
        elif self.phase == _Phase.CAPTURE_IMAGE:
            if self.zone in (_EditZone.CONTROL_NODE,
                             _EditZone.EMPTY_CANVAS):
                self._queue_draw_buttons() # To Erase
                if self.zone == _EditZone.EMPTY_CANVAS:
                    self.phase = _Phase.CAPTURE
                    self.stamp.initialize_phase(self)
                else:
                    self.phase = _Phase.ADJUST_POS
        else:
            return super(StampMode, self).mode_button_press_cb(tdw, event)

   #def mode_button_release_cb(self, tdw, event):
   # Currently nothing to do for this.

    ## Drag handling (both capture and adjust phases)

    def node_drag_start_cb(self, tdw, event):

        assert self.stamp is not None
        mx, my = tdw.display_to_model(event.x, event.y)
        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
        ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK

        if self.phase in (_Phase.CAPTURE, _Phase.ADJUST):

            if self.zone == _EditZone.EMPTY_CANVAS:
                if self.stamp.picture_count > 0:
                    new_picture_id = self.current_picture_id
                    scale_x, scale_y, angle = self.get_default_stamp_param()

                    node = _StampNode(mx, my, 
                           #self.stamp.default_angle,
                           #self.stamp.default_scale_x,
                           #self.stamp.default_scale_y,
                            angle,
                            scale_x,
                            scale_y,
                            new_picture_id)
                    self.nodes.append(node)
                    idx = len(self.nodes) -1
                    self._queue_draw_node(idx)
                    self.drag_offset.start(mx, my)
                    self.phase = _Phase.CAPTURE
                    self.select_node(idx, exclusive=True)
                    self._queue_draw_buttons() # to erase

        elif self.phase in (_Phase.ROTATE,
                            _Phase.SCALE,
                            _Phase.ROTATE_BY_HANDLE,
                            _Phase.SCALE_BY_HANDLE):
            pass
        else:
            super(StampMode, self).node_drag_start_cb(tdw, event)



    def node_drag_update_cb(self, tdw, event, dx, dy):

        assert self.stamp is not None

        # Utility closure definition
        def override_scale_and_rotate(original_phase):
            # Return True, if phase is overridden.
            # and if this closure return True,
            # immidiately exit from this handler method.

            # In older code, there were two modifier,
            # shift for scaling modifier, and ctrl for 
            # rotation modifiler.
            #
            # But currently, ctrl key would conflict
            # with node selection of baseclass(OncanvasEditMixin), 
            # so I decided to use shift as rotation modifier.
            # scaling modifier is obsoluted. Instead of modifier key,
            # use scaling handle.
            if shift_state:
                targ_phase = _Phase.ROTATE
            else:
                return False

            if original_phase != targ_phase:
                # Re-enter drag operation again
                # as different phase
                self.phase = targ_phase
                self.drag_update_cb(tdw, event, dx, dy)
                self.phase = original_phase #_Phase.CAPTURE
                self._queue_draw_node(self.current_node_index) 
                return True
            return False

        # Local variables setup
        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
        ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK
        mx, my = tdw.display_to_model(event.x ,event.y)

        # Setup for specific phases.
        if self.phase in (_Phase.ROTATE,
                            _Phase.SCALE,
                            _Phase.ROTATE_BY_HANDLE,
                            _Phase.SCALE_BY_HANDLE):
            if (shift_state or ctrl_state):
                if override_scale_and_rotate(self.phase):
                    return
                # Fallthrough

            assert self.current_node_index is not None
            idx = self.current_node_index
            self._queue_draw_node(idx)
            node = self.nodes[idx]



        # Phase processing
        if self.phase in (_Phase.ADJUST_POS, _Phase.CAPTURE):
            if override_scale_and_rotate(self.phase):
                return 

            # Stamps are drawn by cairo,
            # so we must care to redraw overlapped stamps
            # during dragging current stamp.
            # Therefore We needs to call _queue_redraw_all_nodes()
            # instead of _queue_draw_node(idx)
            self._queue_redraw_all_nodes()
            self.drag_offset.end(mx, my)
            self._queue_redraw_all_nodes()
                
        elif self.phase in (_Phase.SCALE,
                            _Phase.ROTATE):
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

                self.nodes[idx] = node
                self._queue_draw_node(idx)
                self.start_x = event.x
                self.start_y = event.y
        elif self.phase == _Phase.SCALE_BY_HANDLE:
            pos = self.stamp.get_boundary_points(node)

            # At here, we consider the movement of control handle(i.e. cursor)
            # as a Triangle from origin.

            # 1. Get new(=cursor position) vector from origin(node.x,node.y)
            mx, my = tdw.display_to_model(event.x, event.y)
            length, nx, ny = length_and_normal(
                    node.x, node.y,
                    mx, my)

            bx = mx - node.x 
            by = my - node.y

            orig_pos = self.stamp.get_boundary_points(node, 
                    no_scale=True)

            ti = self.current_node_handle
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

            self.nodes[idx] = node._replace(
                    scale_x=scale_x,
                    scale_y=scale_y)

            self._queue_draw_node(idx)

        elif self.phase == _Phase.ROTATE_BY_HANDLE:
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

            self.nodes[idx] = node._replace(
                    angle = node.angle + rad)
                      
            self._queue_draw_node(idx)
            
        else:
            super(StampMode, self).node_drag_update_cb(tdw, event, dx, dy)


    def node_drag_stop_cb(self, tdw):

        assert self.stamp is not None

        if self.phase == _Phase.CAPTURE:

            if not self.nodes or self.current_node_index == None:
                # Cancelled drag event (and current capture phase)
                # call super-superclass directly to bypass this phase
                self._reset_adjust_data()
                return super(StampMode, self).node_drag_stop_cb(tdw) 

            idx = self.current_node_index
            node = self.nodes[idx]
            dx, dy = self.drag_offset.get_model_offset()
            self.nodes[idx] = node._replace( x=node.x + dx, y=node.y + dy)
            self.drag_offset.reset()

            self.phase = _Phase.ADJUST
            self._update_zone_and_target(tdw, self.last_x, self.last_y)

            # Use _queue_redraw_all_nodes, not _queue_draw_node(idx)
            # Because we must redraw all stamps, including overlapped one. 
            self._queue_redraw_all_nodes()

            self._queue_draw_buttons()

        elif self.phase in (_Phase.ROTATE,
                            _Phase.SCALE,
                            _Phase.ROTATE_BY_HANDLE,
                            _Phase.SCALE_BY_HANDLE):
            assert self.current_node_index is not None
            node = self.nodes[self.current_node_index]
            self.record_current_stamp_param(
                    node.scale_x, node.scale_y, node.angle,
                    picture_id = node.picture_index)
            self.phase = _Phase.ADJUST

        else:
            return super(StampMode, self).node_drag_stop_cb(tdw)

    ## Interrogating events

    def stamp_picture_deleted_cb(self, picture_index):
        """
        A notification callback when stamp picture has been changed
        (deleted)
        """

        # First of all, queue redraw the nodes to be deleted.
        for i, cn in enumerate(self.nodes):
            if cn.picture_index == picture_index:
                self._queue_draw_node(i)#, force_margin=True) 

        # After that, delete it.
        for i, cn in enumerate(self.nodes[:]):
            if cn.picture_index == picture_index:
                self.nodes.remove(cn)
                if i in self.selected_nodes:
                    self.selected_nodes.remove(i)
                if i == self.current_node_index:
                    self.current_node_index = None
                if i == self.target_node_index:
                    self.target_node_index = None

        if len(self.nodes) == 0:
            self.phase == _Phase.CAPTURE
            logger.info('stamp picture deleted, and all nodes deleted')

    ## Node editing
    def update_node(self, i, **kwargs):
        self._queue_draw_node(i)#, force_margin=True) 
        self.nodes[i] = self.nodes[i]._replace(**kwargs)
        self._queue_draw_node(i)#, force_margin=True) 

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
        self.select_node(-1)
        self.current_node_index = None
        self.target_node_index = None

        self._queue_redraw_all_nodes()
        self._queue_draw_buttons()

    ## Action button related
    def is_actionbutton_ready(self):
        """customize whether the action buttons are ready to display.
        This method called from _update_zone_and_target() of
        base mixin.
        """
        flag = super(StampMode, self).is_actionbutton_ready()
        return (flag or self.phase == _Phase.CAPTURE_IMAGE)

    def accept_button_cb(self, tdw):
        if len(self.nodes) > 0:
            self._start_new_capture_phase(rollback=False)

    def reject_button_cb(self, tdw):
        if len(self.nodes) > 0:
            self._start_new_capture_phase(rollback=True)

    def _capture_layer_to_stamp(self, layer):
        sx, sy, ex, ey = [int(x) for x in self._selection_area]

        render_background = not self._app.preferences.get("StampMode.ignore_bg", False)
        pixbuf = layer.render_as_pixbuf(sx, sy, 
                abs(ex-sx)+1, abs(ey-sy)+1, alpha=True,
                render_background=render_background)
        self.stamp.set_surface_from_pixbuf(-1, pixbuf,
                (gui.stamps.PictureSource.CAPTURED, 
                    weakref.proxy(layer))
                )
        self.options_presenter.update_picture_store(False)

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
        sdx,sdy = mode.drag_offset.get_display_offset(self._tdw)
        dx,dy = mode.drag_offset.get_model_offset()
        for i, node in enumerate(mode.nodes):

            if i in mode.selected_nodes:
                tdx = dx
                tdy = dy
                tsdx = sdx
                tsdy = sdy
            else:
                tdx = tdy = 0.0
                tsdx = tsdy = 0.0

            bbox = mode.stamp.get_bbox(self._tdw, node, tdx, tdy)

            if bbox:
                x, y, w, h = bbox
                node_on_screen = (
                    x > alloc.x - w and
                    y > alloc.y - h and
                    x < alloc.x + alloc.width + w and
                    y < alloc.y + alloc.height + h
                )

                if node_on_screen:
                    yield (i, node, tsdx, tsdy)

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

            # Fallthrough

        # also, normal editing buttons are activated.

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


    def draw_stamp(self, cr, idx, node, sdx, sdy, colors):
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

        self.draw_stamp_rect(cr, idx, sdx, sdy, selected_color, position=pos)

        if idx == mode.current_node_index or idx in mode.selected_nodes:
            for i, pt in enumerate(pos):
               #handle_idx = _EditZone.CONTROL_HANDLE_BASE + i
               #handle_idx = _EditZone.CONTROL_HANDLE_BASE + i
                gui.drawutils.render_square_floating_color_chip(
                    cr, pt[0] + sdx, pt[1] + sdy,
                    gui.style.ACTIVE_ITEM_COLOR, 
                    gui.style.DRAGGABLE_POINT_HANDLE_SIZE,
                    fill=(i==mode.current_node_handle)) 
                   #fill=(handle_idx==mode.zone)) 

        mode.stamp.draw(self._tdw, cr, x+sdx, y+sdy, node, True)

    def draw_stamp_rect(self, cr, idx, sdx, sdy, color, position=None):
        cr.save()
        mode = self._mode
        cr.set_line_width(1)

        cr.set_source_rgb(0, 0, 0)

        if not position:
            position = mode.stamp.get_boundary_points(
                    mode.nodes[idx], tdw=self._tdw)

        cr.move_to(position[0][0] + sdx,
                position[0][1] + sdy)
        for lx, ly in position[1:]:
            cr.line_to(lx+sdx, ly+sdy)

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
        fill_flag = True

        colors = ( (1, 1, 1), self.SELECTED_COLOR)

        for i, node, dx, dy in self._get_onscreen_nodes():
           #color = gui.style.EDITABLE_ITEM_COLOR
            show_node = not mode.hide_nodes

            if show_node:
                self.draw_stamp(cr, i, node, dx, dy, colors)
            else:
                self.draw_stamp_rect(cr, i, node, dx, dy, colors)

        # Selection areas
        if mode._selection_area != None:
            self.draw_selection_area(cr)

        # Buttons
        adjust_phase_flag = (mode.phase == _Phase.ADJUST and
                len(mode.nodes) > 0 )
        select_phase_flag = (mode._selection_area != None)

        if (not mode.in_drag and (adjust_phase_flag or select_phase_flag)):
            self._draw_mode_buttons(cr)


class OptionsPresenter_Stamp (object):
    """Presents UI for directly editing point values etc."""

    variation_preset_store = None

    PRESET_ICON_SIZE = 40
    PICTURE_ICON_SIZE = 40

    def __init__(self):
        super(OptionsPresenter_Stamp, self).__init__()
        from application import get_app
        self._app = get_app()
        self._options_grid = None
        self._point_values_grid = None
        self._angle_adj = None
        self._xscale_adj = None
        self._yscale_adj = None
        self._stamp_preset_view = None
        self._stamp_picture_view = None
        self._current_stamp = None

        self._updating_ui = False
        self._target = (None, None)

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

        base_window = builder.get_object("stamp_scrolledwindow")
        self._init_stamp_preset_view(base_window)

        base_window = builder.get_object("picture_scrolledwindow")
        self._init_picture_view(base_window)

        self._init_popup_menus(builder)

        base_grid = builder.get_object("additional_button_grid")
        self._init_toolbar(0, base_grid)

        remember_check = builder.get_object("remember_checkbutton")
        remember_check.set_active(
                self._app.preferences.get("StampMode.remember_last_param", False))

        ignore_bg_check = builder.get_object("ignore_bg_checkbutton")
        ignore_bg_check.set_active(
                self._app.preferences.get("StampMode.ignore_bg", False))

    def _init_stamp_preset_view(self, sw):
        """Initialize stamp preset icon view.

        :param sw: the scroll widget, which contains the stamp preset iconview.
        """

        # XXX we'll need reconsider fixed value 
        # such as item width of 48 or icon size of 32 
        # in high-dpi environment
        manager = self._app.stamp_manager
        liststore = Gtk.ListStore(GdkPixbuf.Pixbuf, str, object)
        for id in manager.stamps:
            stamp = manager.stamps[id]
            iter = liststore.append(
                    (stamp.thumbnail, stamp.name, stamp))

        iconview = Gtk.IconView.new()
        iconview.set_model(liststore)
        iconview.set_pixbuf_column(0)
        iconview.set_text_column(1)
        iconview.set_item_width(self.PRESET_ICON_SIZE) 
        iconview.connect('selection-changed', self._stamp_preset_changed_cb)
        self._stamps_store = liststore

        sw.add(iconview)
        self._stamp_preset_view = iconview

    def _init_picture_view(self, sw):
        """Initialize stamp picture icon view.

        Stamp picture store(and its view) should be updated 
        until a stamp is selected.
        So there is no icon registration codes.

        :param sw: the scroll widget, which contains the stamp picture iconview.
        """

        # The contents of picture liststore is , (picture, picture-id)
        liststore = Gtk.ListStore(GdkPixbuf.Pixbuf, int)
        iconview = Gtk.IconView.new()
        iconview.set_model(liststore)
        iconview.set_pixbuf_column(0)
        iconview.set_item_width(self.PICTURE_ICON_SIZE) 
        iconview.connect('selection-changed', self._stamp_picture_changed_cb)
        iconview.connect('button-release-event', self._stamp_picture_button_release_cb)
        self._pictures_store = liststore

        sw.add(iconview)
        self._stamp_picture_view = iconview

    def _init_popup_menus(self, builder):
        agroup = builder.get_object("popup_actiongroup")
        clipmenu = builder.get_object("clipboard_popup_menu")

        self._delete_picture_action = agroup.get_action('delete_picture_action')
        self._stampicon_action = agroup.get_action('stampicon_action')

        # Creating Popup menu for normal stamp.
        # to share same menu handler between two popup menu.
        basemenu = Gtk.Menu()
        for i, ca in enumerate(agroup.list_actions()):
            basemenu.append(ca.create_menu_item())
            clipmenu.insert(ca.create_menu_item(),i)

        basemenu.show_all()
        clipmenu.show_all()
        self.popup_stamp = basemenu
        self.popup_clipboard = clipmenu

    def _init_toolbar(self, row, box):
        toolbar = gui.widgets.inline_toolbar(
            self._app,
            [
                ("DeleteItem", "mypaint-remove-symbolic"),
                ("AcceptEdit", "mypaint-ok-symbolic"),
                ("DiscardEdit", "mypaint-trash-symbolic"),
            ]
        )
        style = toolbar.get_style_context()
        style.set_junction_sides(Gtk.JunctionSides.TOP)
        box.attach(toolbar, 0, row, 1, 1)

    def update_picture_store(self, force_update):
        """Update picture store (and picture view)
        with current stamp.

        CAUTION: This method update only store.
        does not update iconview.
        """
        mode, node_idx = self.target
        if mode:
            assert mode.stamp is not None
            store = self._pictures_store
            if force_update or mode.stamp != self._current_stamp:
                store.clear()
                for id, icon in mode.stamp.picture_icon_iter():
                    store.append( (icon, id) )
                self._current_stamp = mode.stamp
            else:
                # Refresh current _pictures_store
                for id, icon in mode.stamp.picture_icon_iter():
                    iter = store.get_iter_first()
                    while iter:
                        old_icon = store.get_value(iter, 0) 
                        store_id = store.get_value(iter, 1)
                        if id == store_id:
                            # Already this stamp exist.
                            if old_icon != icon:
                                # Already this stamp exist, but icon changed.
                                # so update it, and exit loop.
                                store.set_value(iter, 0, icon)

                            # Clear 'id' variable to notify 
                            # 'current icon is already registered in the store'
                            id = None
                            break
                        else:
                            iter = store.iter_next(iter)
                            continue

                    if id != None:
                        # current icon is not registered into picture store
                        store.append( (icon, id) )

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

            else:
                self._point_values_grid.set_sensitive(False)

        finally:
            self._updating_ui = False

    def _enable_clipboard_menus(self, flag):
        for cm in self._clipboard_menus:
            cm.set_sensitive(flag)


    ## Widgets Handlers

    def _angle_adj_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        mode, node_idx = self.target
        mode.update_node(node_idx, angle=math.radians(adj.get_value()))

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

    def _delete_point_button_clicked_cb(self, button):
        mode, node_idx = self.target
        if mode.can_delete_node(node_idx):
            mode.delete_node(node_idx)

    def _stamp_preset_changed_cb(self, iconview):
        mode, node_idx = self.target
        if mode:
            if len(iconview.get_selected_items()) > 0:
                mode.set_stamp(
                        _get_iconview_data(iconview, 2)
                        )
                self.update_picture_store(False)

    def _stamp_picture_changed_cb(self, iconview):
        if not self._updating_ui:
            mode, node_idx = self.target
            if mode and mode.stamp is not None:
                enabled = len(iconview.get_selected_items()) > 0
                self._delete_picture_action.set_sensitive(enabled)
                self._stampicon_action.set_sensitive(enabled)

                if enabled:
                    picture_id = _get_iconview_data(iconview, 1)
                    mode.current_picture_id = picture_id

    def _stamp_picture_unselect_all_cb(self, iconview):
        mode, junk = self.target
        if mode and mode.stamp is not None:
            mode.current_picture_id = -1

    def _stamp_picture_button_release_cb(self, iconview, event):
        if event.button == Gdk.BUTTON_SECONDARY:
            mode, node_idx = self.target
            if mode and mode.stamp is not None:
                selected = len(iconview.get_selected_items()) > 0
               #self.stamp_icon_menu_item.set_sensitive(selected)

                if isinstance(mode.stamp, gui.stamps.ClipboardStamp):
                    clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
                    self._enable_clipboard_menus(
                        clipboard.wait_is_image_available())
                    self.popup_clipboard.popup(None, None, None, None,
                            event.button, event.time)
                else:
                    self.popup_stamp.popup(None, None, None, None,
                            event.button, event.time)


    def notebook_stamp_change_current_page_cb(self, note, id):
        print('notebook id %d' % id)
        pass

    def remember_checkbutton_toggled_cb(self, button):
        flag = button.get_active()

        if not self._updating_ui: 
            self._app.preferences["StampMode.remember_last_param"] = flag

        # This line don't care whether updating ui or not.
        StampMode.REMEMBER_PARAM_ENABLED = flag

    def ignore_bg_checkbutton_toggled_cb(self, button):
        if not self._updating_ui: 
            self._app.preferences["StampMode.ignore_bg"] = button.get_active()

    ## Popup menus handler for Stamp picture icon view.
    def _popup_clipboard_cb_base(self, stamp_id):
        mode, junk = self.target
        assert mode
        assert mode.stamp

        img = gui.stamps.load_clipboard_image()

        # Clipboard image might be disabled at this time,
        # from other process.
        if img:
            if isinstance(mode.stamp, gui.stamps.ClipboardStamp):
                mode.stamp.set_surface_from_pixbuf(stamp_id, img)
                self.update_picture_store(False)
        else:
            self._app.show_transient_message(_("There is no clipboard image available."))

    def refresh_clipboard_action_activate_cb(self, menuitem):
        """ Updating clipboard stamp.
        """
        # Call shared handler between menuitems
        self._popup_clipboard_cb_base(0)  

    def add_from_clipboard_action_activate_cb(self, menuitem):

        # Call shared handler between menuitems
        self._popup_clipboard_cb_base(-1)

    def call_editor_action_activate_cb(self, menuitem):
        mode, junk = self.target
        assert mode
        assert mode.stamp
        editor = self._app.stamp_editor_window
        editor.show()

    def stampicon_action_activate_cb(self, menuitem):
        mode, junk = self.target
        assert mode
        assert mode.stamp
        picview = self._stamp_picture_view
        presetview = self._stamp_preset_view
        assert len(picview.get_selected_items()) > 0
        assert len(presetview.get_selected_items()) > 0

        # First, change the internal icon(thumbnail) from the picture
        picture_id = _get_iconview_data(picview, 1)
        mode.stamp.generate_thumbnail(picture_id)

        # Next, refresh the internal icon to widget icon.
        path = presetview.get_selected_items()[0]
        store = presetview.get_model()
        iter = store.get_iter(path)
        store.set_value(iter, 0, mode.stamp.thumbnail)

    def delete_picture_action_activate_cb(self, menuitem):
        mode, junk = self.target
        assert mode
        assert mode.stamp
        iconview = self._stamp_picture_view
        model = iconview.get_model()
        for cpath in iconview.get_selected_items():
            iter = model.get_iter(cpath)
            picture_id = model.get(iter, 1)[0]
            model.remove(iter)
            # 'row-deleted' signal of Gtk.ListStore(Gtk.TreeModel)
            # is completely useless for this, because it is called
            # after the row deleted, there is no way to get
            # the deleted picture id at signal handler...
            # So, delete picture here manually.
            mode.stamp.remove(picture_id)


if __name__ == '__main__':
    pass
