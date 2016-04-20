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

## Module settings

# Which workarounds to allow for motion event compression
EVCOMPRESSION_WORKAROUND_ALLOW_DISABLE_VIA_API = True
EVCOMPRESSION_WORKAROUND_ALLOW_EVHACK_FILTER = True

# Consts for the style of workaround in use
EVCOMPRESSION_WORKAROUND_DISABLE_VIA_API = 1
EVCOMPRESSION_WORKAROUND_EVHACK_FILTER = 2
EVCOMPRESSION_WORKAROUND_NONE = 999

## Functions

def _draw_stamp_to_layer(target_layer, stamp, bbox):
    """
    :param bbox: boundary box, in model coordinate
    """
    sx, sy, ex, ey = bbox
    sx = int(sx)
    sy = int(sy)
    w = int(ex-sx+1)
    h = int(ey-sy+1)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    cr = cairo.Context(surf)

    stamp.draw(cr, sx, sy)
    surf.flush()
    pixbuf = Gdk.pixbuf_get_from_surface(surf, 0, 0, w, h)
    layer = lib.layer.PaintingLayer(name='')
    layer.load_surface_from_pixbuf(pixbuf, int(sx), int(sy))
    del surf, cr

    tiles = set()
    tiles.update(layer.get_tile_coords())
    dstsurf = target_layer._surface
    for tx, ty in tiles:
        with dstsurf.tile_request(tx, ty, readonly=False) as dst:
            layer.composite_tile(dst, True, tx, ty, mipmap_level=0)

    bbox = tuple(target_layer.get_full_redraw_bbox())
    target_layer.root.layer_content_changed(target_layer, *bbox)
    target_layer.autosave_dirty = True
    del layer



## Module constants


## Function defs

## Class defs

class Stamp(object):
    """
    holding stamp information, and draw it to "cairo surface",
    not mypaint surface, for now.

    and then after finalize, it converted to mypaint surface
    and merged into current layer.
    """

    def __init__(self):
        self._stamp_src = None
        self._mat = cairo.Matrix()

    def load_from_file(self, filename):
        self._stamp_src = GdkPixbuf.Pixbuf.new_from_file(filename)


    def draw(self, cr, x, y, angle, scale_x, scale_y, save_context=False):
        """ draw this stamp into cairo surface.
        cairo surface merged into a MyPaint surface later.
        """
        if save_context:
            cr.save()


        w = self._stamp_src.get_width() 
        h = self._stamp_src.get_height()

        # ---- non-work code
        ox = -(w / 2)
        oy = -(h / 2)

        cr.translate(x,y)
        if angle != 0.0:
            cr.rotate(angle)

        if scale_x != 1.0 and scale_y != 1.0:
            cr.scale(scale_x, scale_y)

        Gdk.cairo_set_source_pixbuf(cr, self._stamp_src, ox, oy)
        cr.rectangle(ox, oy, w, h) 

        cr.clip()
        cr.paint()
        
        if save_context:
            cr.restore()

    def get_boundary_positions(self, tdw, mx, my, angle, scale_x, scale_y):
        w = self._stamp_src.get_width() * scale_x 
        h = self._stamp_src.get_height() * scale_y
        sx = - w / 2
        sy = - h / 2
        ex = w+sx
        ey = h+sy

        if angle != 0.0:
            positions = [ (sx, sy),
                          (ex, sy),
                          (ex, ey),
                          (sx, ey) ]
            cos_s = math.cos(angle)
            sin_s = math.sin(angle)
            for i in xrange(4):
                x = positions[i][0]
                y = positions[i][1]
                tx = (cos_s * x - sin_s * y) + mx
                ty = (sin_s * x + cos_s * y) + my
                positions[i] = (tx, ty) 
        else:
            sx += mx
            ex += mx
            sy += my
            ey += my
            positions = [ (sx, sy),
                          (ex, sy),
                          (ex, ey),
                          (sx, ey) ]

        if tdw:
            positions = [ tdw.model_to_display(x,y) for x,y in positions ]

        return positions

    def get_bbox(self, tdw, mx, my, angle, scale_x, scale_y):
        pos = self.get_boundary_positions(None, mx, my, 
                angle, scale_x, scale_y)
        sx, sy = tdw.model_to_display(*pos[0])
        ex = sx
        ey = sy
        for x, y in pos[1:]:
            x, y = tdw.model_to_display(x, y)
            sx = min(sx, x)
            sy = min(sy, y)
            ex = max(ex, x)
            ey = max(ey, y)

        return (sx, sy, (ex - sx) + 1, (ey - sy) + 1)


_NODE_FIELDS = ("x", "y", "angle", "scale_x", "scale_y")

class _StampNode (collections.namedtuple("_StampNode", _NODE_FIELDS)):
    """Recorded control point, as a namedtuple.

    Node tuples have the following 6 fields, in order

    * x, y: model coords, float
    * angle: float in [-math.pi, math.pi] (radian)
    * scale_w: float in [0.0, 3.0]
    * scale_h: float in [0.0, 3.0]
    """






class StampMode (InkingMode):

    ## Metadata properties

    ACTION_NAME = "StampMode"

    ## Metadata methods

    @classmethod
    def get_name(cls):
        return _(u"Stamp")

    def get_usage(self):
        return _(u"Place, and then adjust predefined stamps")

    @property
    def inactive_cursor(self):
        return None

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
   #    elif self.phase == _Phase.ADJUST_SELECTING:
   #        return self._crosshair_cursor
   #    return None


    ## Class config vars



    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):
        super(StampMode, self).__init__(**kwargs)

        self._stamp = Stamp()
        #! test code
        self._stamp.load_from_file('/home/dull/python/src/mypaint/mypaint/stamptest.png')

    @property
    def stamp(self):
        return self._stamp



    def enter(self, doc, **kwds):
        """Enters the mode: called by `ModeStack.push()` etc."""
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
        return
       #if flush:
       #    # Commit the pending work normally
       #    self._start_new_capture_phase(rollback=False)
       #    super(InkingMode, self).checkpoint(flush=flush, **kwargs)
       #else:
       #    # Queue a re-rendering with any new brush data
       #    # No supercall
       #    self._stop_task_queue_runner(complete=False)
       #    self._queue_draw_buttons()
       #    self._queue_redraw_all_nodes()
       #    self._queue_redraw_stamps()

    def _ensure_overlay_for_tdw(self, tdw):
        overlay = self._overlays.get(tdw)
        if not overlay:
            overlay = Overlay_Stamp(self, tdw)
            tdw.display_overlays.append(overlay)
            self._overlays[tdw] = overlay
        return overlay

    ## Raw event handling (prelight & zone selection in adjust phase)
    def button_press_cb(self, tdw, event):
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
                if shift_state:
                    # selection box dragging start!!
                    if self._returning_phase == None:
                        self._returning_phase = self.phase
                    self.phase = _Phase.ADJUST_SELECTING
                    self.selection_rect.start(
                            *tdw.display_to_model(event.x, event.y))
                else:
                   #self._start_new_capture_phase(rollback=False)
                   #assert self.phase == _Phase.CAPTURE
                    self.phase = _Phase.CAPTURE
                    self._queue_draw_buttons() # To erase button!
                    self._queue_redraw_stamps()

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
        elif self.phase == _Phase.ADJUST_SELECTING:
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
        elif self.phase == _Phase.ADJUST_SELECTING:
            # XXX Not sure what to do here.
            pass
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


   #def _queue_draw_node(self, i):
   #    """Redraws a specific control node on all known view TDWs"""
   #    node = self.nodes[i]
   #    dx,dy = self.selection_rect.get_model_offset()
   #    for tdw in self._overlays:
   #        if i in self.selected_nodes:
   #            x, y = tdw.model_to_display(
   #                    node.x + dx, node.y + dy)
   #        else:
   #            x, y = tdw.model_to_display(node.x, node.y)
   #        x = math.floor(x)
   #        y = math.floor(y)
   #        size = math.ceil(gui.style.DRAGGABLE_POINT_HANDLE_SIZE * 2)
   #        tdw.queue_draw_area(x-size, y-size, size*2+1, size*2+1)

   #def _queue_draw_selected_nodes(self):
   #    for i in self.selected_nodes:
   #        self._queue_draw_node(i)
   #
   #def _queue_redraw_all_nodes(self):
   #    """Redraws all nodes on all known view TDWs"""
   #    for i in xrange(len(self.nodes)):
   #        self._queue_draw_node(i)
   #

    def _queue_redraw_stamps(self):
        """Redraws the entire curve on all known view TDWs"""
       #self._stop_task_queue_runner(complete=False)
        dx, dy = self.selection_rect.get_model_offset()
        for tdw in self._overlays:
           #model = tdw.doc
           #interp_state = {"t_abs": self.nodes[0].time}
            for i, cn in enumerate(self.nodes):
                if i in self.selected_nodes:
                    x, y, w, h = self.stamp.get_bbox(tdw, 
                            cn.x + dx , cn.y + dy, 
                            cn.angle, cn.scale_x, cn.scale_y)
                   #tdw.queue_draw_area(
                   #     *self.stamp.get_bbox(tdw, dx + cn.x, dy + cn.y))
                else:
                    x, y, w, h = self.stamp.get_bbox(tdw, cn.x , cn.y, 
                            cn.angle, cn.scale_x, cn.scale_y)
                   #tdw.queue_draw_area(
                   #     *self.stamp.get_bbox(tdw, cn.x, cn.y))
               #tdw.queue_draw_area(
               #        0, 0, w, h)
                tdw.queue_draw_area(
                        x, y, w, h)


       #self._start_task_queue_runner()


    ## Drag handling (both capture and adjust phases)

    def drag_start_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        mx, my = tdw.display_to_model(event.x, event.y)
        if self.phase == _Phase.CAPTURE:

            if event.state != 0:
                # To activate some mode override
                self._last_event_node = None
                return super(InkingMode, self).drag_start_cb(tdw, event)
            else:
                node = _StampNode(event.x, event.y, 1.6, 1.0, 1.0)
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
                self.selection_rect.start(mx, my)

        elif self.phase in (_Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT):
            pass
        elif self.phase == _Phase.ADJUST_SELECTING:
            self.selection_rect.start(mx, my)
            self.selection_rect.is_addition = (event.state & Gdk.ModifierType.CONTROL_MASK)
            self._queue_draw_buttons() # To erase button!
            self._queue_draw_selection_rect() # to start
        elif self.phase == _Phase.CHANGE_PHASE:
            pass
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)


    def drag_update_cb(self, tdw, event, dx, dy):
        self._ensure_overlay_for_tdw(tdw)
        mx, my = tdw.display_to_model(event.x ,event.y)
        if self.phase == _Phase.CAPTURE:
            node = _StampNode(event.x, event.y, 0.0, 1.0, 1.0)
            self._last_event_node = node

            self._queue_redraw_stamps()

             # [TODO] below line can be reformed to minimize redrawing
            self._queue_draw_node(len(self.nodes)-1) 
        else:
            super(StampMode, self).drag_update_cb(tdw, event, dx, dy)


    def drag_stop_cb(self, tdw):
        self._ensure_overlay_for_tdw(tdw)
        if self.phase == _Phase.CAPTURE:

            if not self.nodes or self._last_event_node == None:
                # call super-superclass directly to bypass this phase
                return super(InkingMode, self).drag_stop_cb(tdw) 

            self.nodes.append(self._last_event_node)


            self._reset_capture_data()
            self._reset_adjust_data()
            if len(self.nodes) > 0:
                self.phase = _Phase.ADJUST
                self._queue_redraw_all_nodes()
                self._queue_redraw_stamps()
                self._queue_draw_buttons()
            else:
                self._reset_nodes()
                tdw.queue_draw()
        elif self.phase == _Phase.ADJUST:
            super(StampMode, self).drag_stop_cb(tdw)
            self._queue_redraw_all_nodes()
            self._queue_redraw_stamps()
            self._queue_draw_buttons()
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

    @property
    def options_presenter(self):
        """MVP presenter object for the node editor panel"""
        cls = self.__class__
        if cls._OPTIONS_PRESENTER is None:
            cls._OPTIONS_PRESENTER = OptionsPresenter_Stamp()
        return cls._OPTIONS_PRESENTER





class Overlay_Stamp (Overlay):
    """Overlay for an StampMode's adjustable points"""

    def __init__(self, mode, tdw):
        super(Overlay_Stamp, self).__init__(mode, tdw)

    def draw_stamp(self, cr, node, x, y):
        mode = self._inkmode
        pos = mode.stamp.get_boundary_positions(self._tdw,
                node.x, node.y, node.angle, node.scale_x, node.scale_y) 
        cr.save()
        cr.set_line_width(1)
        cr.set_source_rgb(0, 0, 0)
        cr.move_to(pos[0][0], pos[0][1])
        for lx, ly in pos[1:]:
            cr.line_to(lx, ly)
        cr.line_to(pos[0][0], pos[0][1])
        cr.stroke()
        cr.restore()

        mode.stamp.draw(cr, x, y, 
                node.angle, node.scale_x, node.scale_y,
                True)

    def paint(self, cr):
        """Draw adjustable nodes to the screen"""
        # Control nodes
        mode = self._inkmode
        radius = gui.style.DRAGGABLE_POINT_HANDLE_SIZE
        alloc = self._tdw.get_allocation()
        dx,dy = mode.selection_rect.get_display_offset(self._tdw)
        fill_flag = not mode.phase in (_Phase.ADJUST_PRESSURE, _Phase.ADJUST_PRESSURE_ONESHOT)
        for i, node, x, y in self._get_onscreen_nodes():
            color = gui.style.EDITABLE_ITEM_COLOR
            show_node = not mode.hide_nodes
            if (mode.phase in
                    (_Phase.ADJUST,
                     _Phase.ADJUST_PRESSURE,
                     _Phase.ADJUST_PRESSURE_ONESHOT,
                     _Phase.ADJUST_SELECTING)):
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

                if (color != gui.style.EDITABLE_ITEM_COLOR and
                        mode.phase == _Phase.ADJUST):
                    x += dx
                    y += dy

            if show_node:
                gui.drawutils.render_round_floating_color_chip(
                    cr=cr, x=x, y=y,
                    color=color,
                    radius=radius,
                    fill=fill_flag)
                self.draw_stamp(cr, node, x, y)

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

        # Selection Rectangle
        if mode.phase == _Phase.ADJUST_SELECTING:
            self.draw_selection_rect(cr)


class OptionsPresenter_Stamp (object):
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
        super(OptionsPresenter_Stamp, self).__init__()
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
        self._period_adj = builder.get_object("period_adj")
        self._period_scale = builder.get_object("period_scale")
        self._period_adj.set_value(self._app.preferences.get(
            "inktool.capture_period_factor", 1))
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
        return #! REMOVEME

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
        return #! REMOVEME
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
        return #! REMOVEME
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
            self._period_adj.set_value(self._app.preferences.get(
                "inktool.capture_period_factor", 1))

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
        value = adj.get_value()
        inkmode, node_idx = self.target
        inkmode.update_node(node_idx, xtilt=float(adj.get_value()))

    def _ytilt_adj_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        value = adj.get_value()
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

    def _period_adj_value_changed_cb(self, adj):
        if self._updating_ui:
            return
        self._app.preferences['inktool.capture_period_factor'] = adj.get_value()
        InkingMode.CAPTURE_SETTING.set_factor(adj.get_value())

    def _period_scale_format_value_cb(self, scale, value):
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


