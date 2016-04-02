# This file is part of MyPaint.
# Copyright (C) 2016 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


## Imports

import math
from numpy import isfinite
import collections
import weakref
import os.path
from logging import getLogger
logger = getLogger(__name__)
import array
import time

from gettext import gettext as _
import gi
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GLib
from gi.repository import GObject
import cairo

import gui.mode
import gui.overlays
import gui.style
import gui.drawutils
import lib.helpers
import lib.layer
import gui.cursor
import lib.observable
from gui.inktool import *
from gui.linemode import *
from gui.beziertool import *
from gui.beziertool import _Control_Handle, _Node_Bezier, _EditZone_Bezier, _PhaseBezier
from lib.command import Command
import lib.mypaintlib

## Module constants

POLYFILLMODES = (
    (lib.mypaintlib.CombineNormal, _("Normal")),
    (lib.mypaintlib.CombineDestinationOut, _("Erase")),
    (lib.mypaintlib.CombineDestinationIn, _("Erase Outside")),
    (lib.mypaintlib.CombineSourceAtop, _("Clipped")),
    )

## Enum defs


## Module funcs

def _draw_node_polygon(cr, tdw, nodes, selected_nodes=None, 
        color=None, gradient=None,
        dx=0, dy=0, ox=0, oy=0, stroke=False, fill=True):
    """ draw cairo polygon
    :param color: color object of mypaint
    :param gradient: Gradient object. cairo.LinearGradient or something

    if both of color and gradient are null,polygon is not filled.

    :param selected_nodes: list of INDEX of selected nodes 
    :param dx,dy: offset position of selected nodes
    :param ox,oy: polygon origin position
    """
    if len(nodes) > 1:
        cr.save()
        cr.set_line_width(1)
        if color:
            cr.set_source_rgb(*color.get_rgb())
        elif gradient:
            cr.set_source(gradient)

        for i, node in enumerate(nodes):#self._get_onscreen_nodes():

            if i==len(nodes)-1 and len(nodes) < 3:
                break

            if tdw:
                x, y = tdw.model_to_display(node.x, node.y)
            else:
                x, y = node

            x-=ox
            y-=oy

            n = (i+1) % len(nodes)

            if tdw:
                x1, y1 = tdw.model_to_display(*node.get_control_handle(1))
                x2, y2 = tdw.model_to_display(*nodes[n].get_control_handle(0))
                x3, y3 = tdw.model_to_display(nodes[n].x, nodes[n].y)
            else:
                x1, y1 = node.get_control_handle(1)
                x2, y2 = nodes[n].get_control_handle(0)
                x3, y3 = nodes[n].x, nodes[n].y

            x1-=ox
            x2-=ox
            x3-=ox
            y1-=oy
            y2-=oy
            y3-=oy

            if selected_nodes:

                if i in selected_nodes:
                    x += dx
                    y += dy
                    x1 += dx
                    y1 += dy

                if n in selected_nodes:
                    x2 += dx
                    y2 += dy
                    x3 += dx
                    y3 += dy

            if fill or i < len(nodes)-1:
                if i==0:
                    cr.move_to(x,y)

                cr.curve_to(x1, y1, x2, y2, x3, y3) 



        if fill and len(nodes) > 2 and (gradient or color):
            cr.close_path()
            cr.fill_preserve()

        if stroke:
            def draw_dashed(space):
                cr.set_source_rgb(1,1,1)
                cr.stroke_preserve()
                cr.set_source_rgb(0,0,0)
                cr.set_dash( (space, ) )
                cr.stroke()

            draw_dashed(3.0)

            if len(nodes) > 2:
                cr.move_to(x,y)
                cr.curve_to(x1, y1, x2, y2, x3, y3) 
                draw_dashed(8.0)


        cr.restore()

def _draw_polygon_to_layer(model, target_layer,nodes, 
        color, gradient, bbox, mode=lib.mypaintlib.CombineNormal):
    """
    :param bbox: boundary box, in model coordinate
    :param mode: polygon drawing mode(enum _POLYDRAWMODE). 
                 * NOT layer composite mode*
    """
    sx, sy, ex, ey = bbox
    sx = int(sx)
    sy = int(sy)
    w = int(ex-sx+1)
    h = int(ey-sy+1)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    cr = cairo.Context(surf)

   #linpat = cairo.LinearGradient(0, 0, w, h);
   #linpat.add_color_stop_rgba(0.20,  1, 0, 0, 1);
   #linpat.add_color_stop_rgba(0.50,  0, 1, 0, 1);
   #linpat.add_color_stop_rgba( 0.80,  0, 0, 1, 1);
   #
   #_draw_node_polygon(cr, None, self.nodes, ox=sx, oy=sy,
   #        gradient=linpat)
    _draw_node_polygon(cr, None, nodes, ox=sx, oy=sy,
            color=color)
    surf.flush()
    pixbuf = Gdk.pixbuf_get_from_surface(surf, 0, 0, w, h)
    layer = lib.layer.PaintingLayer(name='')
    layer.load_surface_from_pixbuf(pixbuf, int(sx), int(sy))
    del surf, cr

    tiles = set()
    layer.mode = mode

    if mode == lib.mypaintlib.CombineDestinationIn:
        tiles.update(target_layer.get_tile_coords())
        dstsurf = target_layer._surface
        srcsurf = layer._surface
        for tx, ty in tiles:
            with dstsurf.tile_request(tx, ty, readonly=False) as dst:
                if srcsurf.tile_request(tx, ty, readonly=False) == None:
                    lib.mypaintlib.tile_clear_rgba16(dst)
                else:
                    layer.composite_tile(dst, True, tx, ty, mipmap_level=0)
    else:
        tiles.update(layer.get_tile_coords())
        dstsurf = target_layer._surface
        for tx, ty in tiles:
            with dstsurf.tile_request(tx, ty, readonly=False) as dst:
                layer.composite_tile(dst, True, tx, ty, mipmap_level=0)


    bbox = tuple(target_layer.get_full_redraw_bbox())
    target_layer.root.layer_content_changed(target_layer, *bbox)
    del layer


## Class defs
class _EditZone_Polyfill(_EditZone_Bezier):
    """Enumeration of what the pointer is on in the ADJUST phase"""
    FILL_ATOP_BUTTON = 201
    ERASE_BUTTON = 202
    ERASE_OUTSIDE_BUTTON = 203

class _Shape:
    """Enumeration of shape. 
    this is same as contents of shape_type_combobox of 
    OptionsPresenter_Polyfill
    """
    Bezier = 0
    Polyline = 1
    Rectangle = 2
    Elipse = 3


class _ButtonInfo(object):
    """ Buttons infomation management class.
    In Polyfill tool, Buttons increased and eventually 
    become difficult to manage.
    """

    button_info = (
                ( 0, 'mypaint-ok-symbolic',
                    _EditZone_Polyfill.ACCEPT_BUTTON ),
                ( 1, 'mypaint-trash-symbolic',
                    _EditZone_Polyfill.REJECT_BUTTON ),
                ( 2, 'mypaint-eraser-symbolic', 
                    _EditZone_Polyfill.ERASE_BUTTON ),
                ( 3, 'mypaint-cut-symbolic', 
                    _EditZone_Polyfill.ERASE_OUTSIDE_BUTTON ),
                ( 4, 'mypaint-add-symbolic',
                    _EditZone_Polyfill.FILL_ATOP_BUTTON ),
            )

    button_zones = {
        _EditZone_Polyfill.ACCEPT_BUTTON:lib.mypaintlib.CombineNormal,
        _EditZone_Polyfill.ERASE_BUTTON:lib.mypaintlib.CombineDestinationOut,
        _EditZone_Polyfill.ERASE_OUTSIDE_BUTTON:lib.mypaintlib.CombineDestinationIn,
        _EditZone_Polyfill.FILL_ATOP_BUTTON:lib.mypaintlib.CombineSourceAtop,
        }

    def __init__(self):
        self._pos = [None] * len(self.button_info)

    def get_mode_from_zone(self, zone):
        if zone in self.button_zones:
            return self.button_zones[zone]
        else:
            return None

    def setup_round_position(self, tdw, pos, count=None):
        """ setup button position,rotating around (x,y)
        
        :param pos:   center position,in display coordinate.
        :param count: maximum button count, if we need 
                      minimum (only ACCEPT/REJECT buttons),
                      use this argument as 2
                      by default,this is None
        """
        x, y = pos
        button_radius = gui.style.FLOATING_BUTTON_RADIUS
        margin = 1.5 * button_radius
        alloc = tdw.get_allocation()
        view_x0, view_y0 = alloc.x, alloc.y
        view_x1, view_y1 = view_x0+alloc.width, view_y0+alloc.height

        palette_radius = 64 #gui.style.BUTTON_PALETTE_RADIUS
        area_radius = palette_radius + margin 

        if x + area_radius > view_x1:
            x = view_x1 - area_radius
        elif x - area_radius < view_x0:
            x = view_x0 + area_radius
        
        if y + area_radius > view_y1:
            y = view_y1 - area_radius
        elif y - area_radius < view_y0:
            y = view_y0 + area_radius

        if count == None:
            self._valid_count = len(self.button_info)
        else:
            self._valid_count = count

        for i in xrange(self._valid_count):
            rad = (math.pi / self._valid_count) * 2.0 * i
            dx = - palette_radius * math.sin(rad)
            dy = palette_radius * math.cos(rad)
            self._pos[i] = (x + dx, y - dy) 


    def set_position(self, *positions):
        """ setup button position 
        :param positions: positions of buttons,according to self.button_info.
        each of this argument can be 'None', when button is hidden.
        NOTE: valid button is limited to count of this variable argument.
        """
        for i, pos in enumerate(positions):
            self._pos[i] = pos
        self._valid_count = len(positions)

    def get_position(self, idx):
        return self._pos[idx]

    def enumurate_buttons(self):
        for i in xrange(self._valid_count):
            yield (self._pos[i], self.button_info[i])



class PolyFill(Command):
    """Polygon-fill on the current layer"""

    display_name = _("Polygon Fill")

    def __init__(self, doc, nodes, color, gradient,
                 bbox, mode, **kwds):
        """
        :param bbox: boundary rectangle,in model coordinate.
        """
        super(PolyFill, self).__init__(doc, **kwds)
        self.nodes = nodes
        self.color = color
        self.gradient = gradient
        self.bbox = bbox
        self.snapshot = None
        self.composite_mode = mode

    def redo(self):
        # Pick a source
        layers = self.doc.layer_stack
        assert self.snapshot is None
        self.snapshot = layers.current.save_snapshot()
        dst_layer = layers.current
        # Fill connected areas of the source into the destination
        _draw_polygon_to_layer(self.doc, dst_layer, self.nodes,
                self.color, self.gradient, self.bbox,
                self.composite_mode)

    def undo(self):
        layers = self.doc.layer_stack
        assert self.snapshot is not None
        layers.current.load_snapshot(self.snapshot)
        self.snapshot = None

class _Mode_Bezier(object):

    def __init__(self):
        pass

    @staticmethod
    def render(self, cr):
        pass

    def button_press_cb(self, tdw, event):
        pass

    def button_release_cb(self, tdw, event):
        pass

    def drag_start_cb(self, tdw, event):
        pass

    def drag_update_cb(self, tdw, event, dx, dy):
        pass

    def drag_stop_cb(self, tdw):
        pass

class _Mode_Polyline(object):

    def __init__(self):
        pass

    @staticmethod
    def render(self, cr):
        pass

    def button_press_cb(self, tdw, event):
        pass

    def button_release_cb(self, tdw, event):
        pass

    def drag_start_cb(self, tdw, event):
        pass

    def drag_update_cb(self, tdw, event, dx, dy):
        pass

    def drag_stop_cb(self, tdw):
        pass

class _Mode_Rectangle(object):

    def __init__(self):
        pass

    @staticmethod
    def render(self, cr):
        pass

    def button_press_cb(self, tdw, event):
        pass

    def button_release_cb(self, tdw, event):
        pass

    def drag_start_cb(self, tdw, event):
        pass

    def drag_update_cb(self, tdw, event, dx, dy):
        pass

    def drag_stop_cb(self, tdw):
        pass

class _Mode_Elipse(object):

    def __init__(self):
        pass

    @staticmethod
    def render(self, cr):
        pass

    def button_press_cb(self, tdw, event):
        pass

    def button_release_cb(self, tdw, event):
        pass

    def drag_start_cb(self, tdw, event):
        pass

    def drag_update_cb(self, tdw, event, dx, dy):
        pass

    def drag_stop_cb(self, tdw):
        pass


class PolyfillMode (BezierMode):

    ## Metadata properties
    ACTION_NAME = "PolyFillMode"

    ## Metadata methods

    @classmethod
    def get_name(cls):
        return _(u"Polygonfill")

    def get_usage(self):
        return _(u"fill up polygon with current foreground color,or gradient")

    @property
    def foreground_color(self):
        if self.doc:
            return self.doc.app.brush_color_manager.get_color()
        else:
            from application import get_app
            return get_app().brush_color_manager.get_color()


    ## Class config vars
    stroke_history = StrokeHistory(6) # stroke history of polyfilltool 

    DEFAULT_POINT_CORNER = True       # default point is corner,not curve

    ## Other class vars
    _OPTIONS_PRESENTER_POLY = None 

    button_info = _ButtonInfo()       # button infomation class

    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):
        super(PolyfillMode, self).__init__(**kwargs)
        self._polygon_preview_fill = False
        self._shape_type = _Shape.Bezier



    ## Update inner states methods
    def _ensure_overlay_for_tdw(self, tdw):
        overlay = self._overlays.get(tdw)
        if not overlay:
            overlay = OverlayPolyfill(self, tdw)
            tdw.display_overlays.append(overlay)
            self._overlays[tdw] = overlay
            if len(self.nodes) > 0:
                self._queue_redraw_curve()
        return overlay

        
    def _update_zone_and_target(self, tdw, x, y, ignore_handle=False):
        """Update the zone and target node under a cursor position"""
        ## FIXME mostly copied from inktool.py
        ## the differences are 'control handle processing' and
        ## 'cursor changing' and, queuing buttons draw 
        ## to follow current_node_index


        self._ensure_overlay_for_tdw(tdw)
        new_zone = _EditZone_Bezier.EMPTY_CANVAS
        if not self.in_drag and len(self.nodes) > 0:
            if self.phase in (_PhaseBezier.MOVE_NODE, 
                    _PhaseBezier.CREATE_PATH):

                new_target_node_index = None
                
                # Test buttons for hits
                hit_dist = gui.style.FLOATING_BUTTON_RADIUS

                if len(self.nodes) > 1:
                    for pos, info in self.button_info.enumurate_buttons():
                        if pos is None:
                            continue
                        btn_x, btn_y = pos
                        d = math.hypot(btn_x - x, btn_y - y)
                        if d <= hit_dist:
                            new_target_node_index = None
                            new_zone = info[2]
                            break

                if (new_zone == _EditZone_Bezier.EMPTY_CANVAS):

                    # Checking Control handles first:
                    # because when you missed setting control handle 
                    # at node creation stage,if node zone detection
                    # is prior to control handle, they are unoperatable.
                    if (self.current_node_index is not None and 
                            ignore_handle == False):
                        c_node = self.nodes[self.current_node_index]
                        self.current_handle_index = None
                        for i in (0,1):
                            handle = c_node.get_control_handle(i)
                            hx, hy = tdw.model_to_display(handle.x, handle.y)
                            d = math.hypot(hx - x, hy - y)
                            if d > hit_dist:
                                continue
                            new_target_node_index = self.current_node_index
                            self.current_handle_index = i
                            new_zone = _EditZone_Bezier.CONTROL_HANDLE
                            break         

                    # Test nodes for a hit, in reverse draw order
                    if new_target_node_index == None:
                        new_target_node_index = self._search_target_node(tdw, x, y)
                        if new_target_node_index != None:
                            new_zone = _EditZone_Bezier.CONTROL_NODE

                    
                # Update the prelit node, and draw changes to it
                if new_target_node_index != self.target_node_index:
                    if self.target_node_index is not None:
                        self._queue_draw_node(self.target_node_index)
                    self.target_node_index = new_target_node_index
                    if self.target_node_index is not None:
                        self._queue_draw_node(self.target_node_index)

                ## Fallthru below


        # Update the zone, and assume any change implies a button state
        # change as well (for now...)
        if self.zone != new_zone:
            self.zone = new_zone
            self._ensure_overlay_for_tdw(tdw)
            if len(self.nodes) > 1:
                self._queue_previous_draw_buttons()

        # Update the "real" inactive cursor too:
        # these codes also a little changed from inktool.
        if not self.in_drag:
            cursor = None
            if self.phase in (_PhaseBezier.INITIAL, _PhaseBezier.CREATE_PATH,
                    _PhaseBezier.MOVE_NODE):
                if self.zone == _EditZone_Bezier.CONTROL_NODE:
                    cursor = self._crosshair_cursor
                elif self.zone in self.button_info.button_zones:
                    cursor = self._crosshair_cursor
                else:
                    cursor = self._arrow_cursor
            if cursor is not self._current_override_cursor:
                tdw.set_override_cursor(cursor)
                self._current_override_cursor = cursor


    def _get_maximum_rect(self, tdw, dx=0, dy=0):
        """ get possible maximum rectangle 
        :param tdw: the target tileddrawwidget.if this is None,
                    all values(includeing dx,dy) recognized as 
                    model coordinate value.
        """
        if len(self.nodes) < 2:
            return (0,0,0,0)
        sx = ex = self.nodes[0].x
        sy = ey = self.nodes[0].y
        for i,cn in enumerate(self.nodes):
            # Get boundary rectangle,to reduce processing segment
            n = (i+1) % len(self.nodes)
            nn = self.nodes[n]
            if tdw:
                cnx, cny = tdw.model_to_display(cn.x, cn.y)
            else:
                cnx, cny = cn

            if i in self.selected_nodes:
                cnx+=dx
                cny+=dy

            if tdw:
                nnx, nny = tdw.model_to_display(nn.x, nn.y)
            else:
                nnx, nny = nn

            if n in self.selected_nodes:
                nnx+=dx
                nny+=dy

            sx = min(min(sx,cnx), nnx)
            ex = max(max(ex,cnx), nnx)
            sy = min(min(sy,cny), nny)
            ey = max(max(ey,cny), nny)

            if tdw:
                cx, cy = tdw.model_to_display(*cn.get_control_handle(1))
            else:
                cx, cy = cn.get_control_handle(1)

            if i in self.selected_nodes:
                cx+=dx
                cy+=dy

            if tdw:
                nx, ny = tdw.model_to_display(*nn.get_control_handle(0))
            else:
                nx, ny = cn.get_control_handle(0)

            if n in self.selected_nodes:
                nx+=dx
                ny+=dy

            sx = min(min(sx, cx), nx)
            ex = max(max(ex, cx), nx)
            sy = min(min(sy, cy), ny)
            ey = max(max(ey, cy), ny)

        return (sx, sy, ex, ey)

    def _reset_all_internal_state(self):
        self._queue_draw_buttons()
        self._queue_redraw_all_nodes()
        self._queue_redraw_curve() 
        self._reset_nodes()
        self._reset_capture_data()
        self._reset_adjust_data()
        self.phase = _PhaseBezier.INITIAL
        self._stroke_from_history = False
        self.forced_button_pos = None

    ## Properties
    def set_shape(self, shape_type):
        self._shape_type = shape_type

    ## Redraws
    

    def redraw_curve_cb(self, erase=False):
        """ Frontend method,to redraw curve from outside this class"""
        pass # do nothing

    def _queue_redraw_curve(self, tdw=None):
        self._stop_task_queue_runner(complete=False)
        for tdw in self._overlays:
            
            if len(self.nodes) < 2:
                continue

            sdx, sdy = self.selection_rect.get_display_offset(tdw)
    
            sx, sy, ex, ey = self._get_maximum_rect(tdw, sdx, sdy)
        
            self._queue_task(
                    self._queue_polygon_area,
                    sx, sy, ex, ey
            )

        self._start_task_queue_runner()

    def _queue_polygon_area(self, sx, sy, ex, ey):
        for tdw in self._overlays:
            tdw.queue_draw_area(sx, sy, ex-sx+1, ey-sy+1)

    def _queue_draw_buttons(self):
        for tdw, overlay in self._overlays.items():
            overlay.update_button_positions()
            for pos, info in self.button_info.enumurate_buttons():
           #for idx, pos in enumerate(overlay.button_pos):
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

    def is_drawn_handle(self, i, hi):
        return True # All handle should be drawn, in Polyfill tool

    def _start_new_capture_phase_polyfill(self, tdw, mode, rollback=False):
        if rollback:
            self._stop_task_queue_runner(complete=False)
            self._reset_all_internal_state()
        else:
            self._stop_task_queue_runner(complete=True)
            self.execute_draw_polygon(mode=mode)


    def leave(self):
        if not self._is_active() and len(self.nodes) > 1:
            # The modechange is not overriding.
            # so commit last pending work
            self._start_new_capture_phase_polyfill(None, rollback=False)

        super(BezierMode, self).leave()

    def checkpoint(self, flush=True, **kwargs):
        """Sync pending changes from (and to) the model
        When this mode is left for another mode (see `leave()`), the
        pending brushwork is committed properly.
    
        """
        if flush:
            super(InkingMode, self).checkpoint(flush=flush, **kwargs) # call super-superclass method
        else:
            # Queue a re-rendering with any new brush data
            # No supercall
            self._stop_task_queue_runner(complete=False)
            self._queue_draw_buttons()
            self._queue_redraw_all_nodes()
            self._queue_redraw_curve()


    ### Event handling
    def scroll_cb(self, tdw, event):
        # to cancelling scroll-wheel pressure modification.
        return super(InkingMode, self).scroll_cb(tdw, event) # simply call super-superclass!

    ## Raw event handling (prelight & zone selection in adjust phase)
    def button_press_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False
        self._update_zone_and_target(tdw, event.x, event.y,
                event.state & Gdk.ModifierType.MOD1_MASK)
        self._update_current_node_index()

        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
        ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK

        if self.phase == _PhaseBezier.INITIAL: 
            self.phase = _PhaseBezier.CREATE_PATH
            # FALLTHRU: *do* start a drag 
        elif self.phase in (_PhaseBezier.CREATE_PATH,):
            # Initial state - everything starts here!
       
            if (self.zone in self.button_info.button_zones): 
                if (event.button == 1 and 
                        event.type == Gdk.EventType.BUTTON_PRESS):

                        # To avoid some of visual glitches,
                        # we need to process button here.
                        if self.zone == _EditZone_Bezier.REJECT_BUTTON:
                            self._start_new_capture_phase_polyfill(
                                tdw, None, rollback=True)
                        else:
                            self._start_new_capture_phase_polyfill(
                                tdw, 
                                self.button_info.get_mode_from_zone(self.zone),
                                rollback=False)
                        
                        self._reset_adjust_data()
                        return False
                    
                    
            elif self.zone == _EditZone_Bezier.CONTROL_NODE:
                # Grabbing a node...
                button = event.button
                if self.phase == _PhaseBezier.CREATE_PATH:

                    # normal move node start
                    self.phase = _PhaseBezier.MOVE_NODE

                    if button == 1 and self.current_node_index != None:
                        if (event.state & Gdk.ModifierType.CONTROL_MASK):
                            # Holding CONTROL key = adding or removing a node.
                            if self.current_node_index in self.selected_nodes:
                                self.selected_nodes.remove(self.current_node_index)
                            else:
                                self.selected_nodes.append(self.current_node_index)
        
                            self._queue_draw_selected_nodes() 
                        else:
                            # no CONTROL Key holded.
                            # If new solo node clicked without holding 
                            # CONTROL key,then reset all selected nodes.
        
                            do_reset = ((event.state & Gdk.ModifierType.MOD1_MASK) != 0)
                            do_reset |= not (self.current_node_index in self.selected_nodes)
        
                            if do_reset:
                                # To avoid old selected nodes still lit.
                                self._queue_draw_selected_nodes() 
                                self._reset_selected_nodes(self.current_node_index)

                # FALLTHRU: *do* start a drag 

            elif self.zone == _EditZone_Bezier.EMPTY_CANVAS:
                
                if self.phase == _PhaseBezier.CREATE_PATH:
                    if (len(self.nodes) > 0): 
                        if shift_state and ctrl_state:
                            self._queue_draw_buttons() 
                            self.forced_button_pos = (event.x, event.y)
                            self.phase = _PhaseBezier.CHANGE_PHASE 
                            self._returning_phase = _PhaseBezier.CREATE_PATH
                        elif shift_state:
                            # selection box dragging start!!
                            if self._returning_phase == None:
                                self._returning_phase = self.phase
                            self.phase = _PhaseBezier.ADJUST_SELECTING
                            self.selection_rect.start(
                                    *tdw.display_to_model(event.x, event.y))
                        elif ctrl_state:
                            mx, my = tdw.display_to_model(event.x, event.y)
                            pressed_segment = self._detect_on_stroke(mx, my)
                            if pressed_segment:
                                # pressed_segment is a tuple which contains
                                # (node index of start of segment, stroke step)

                                # To erase buttons 
                                self._queue_draw_buttons() 

                                self._divide_bezier(*pressed_segment)

                                # queue new node here.
                                self._queue_draw_node(pressed_segment[0] + 1)
                                
                                self.phase = _PhaseBezier.PLACE_NODE
                                return False # Cancel drag event


            elif self.zone == _EditZone_Bezier.CONTROL_HANDLE:
                if self.phase == _PhaseBezier.CREATE_PATH:
                    self.phase = _PhaseBezier.ADJUST_HANDLE


            # FALLTHRU: *do* start a drag 

        elif self.phase == _PhaseBezier.ADJUST_SELECTING:
            # XXX Not sure what to do here.
            pass
        elif self.phase in (_PhaseBezier.ADJUST_HANDLE, _PhaseBezier.INIT_HANDLE):
            pass
        elif self.phase in (_PhaseBezier.MOVE_NODE, _PhaseBezier.CHANGE_PHASE):
            # THIS CANNOT BE HAPPEN...might be an evdev dropout.through it.
            pass
        else:
            raise NotImplementedError("Unrecognized phase %r", self.phase)
        # Update workaround state for evdev dropouts
        self._button_down = event.button

        # Super-Supercall(not supercall) would invoke drag-related callbacks.
        return super(InkingMode, self).button_press_cb(tdw, event) 

    def button_release_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False

        # Here is 'button_release_cb',which called 
        # prior to drag_stop_cb.
        # so, in this method, changing self._phase
        # is very special case. 
        if self.phase == _PhaseBezier.PLACE_NODE:
            self._queue_redraw_curve(tdw) 
            self.phase = _PhaseBezier.CREATE_PATH
            pass

        # Update workaround state for evdev dropouts
        self._button_down = None

        # Super-Supercall(not supercall) would invoke drag_stop_cb signal.
        return super(InkingMode, self).button_release_cb(tdw, event)
        

    ## Drag handling (both capture and adjust phases)
    def drag_start_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        mx, my = tdw.display_to_model(event.x, event.y)

        self._queue_previous_draw_buttons() # To erase button,and avoid glitch

        # Basically,all sections should do fall-through.
        if self.phase == _PhaseBezier.CREATE_PATH:

            if self.zone == _EditZone_Bezier.EMPTY_CANVAS:
                if event.state != 0:
                    # To activate some mode override
                    self._last_event_node = None
                    return super(InkingMode, self).drag_start_cb(tdw, event)
                else:
                    # New node added!
                    node = self._get_event_data(tdw, event)
                    self.nodes.append(node)
                    self._last_event_node = node
                    self.phase = _PhaseBezier.INIT_HANDLE
                    self.current_node_index=len(self.nodes)-1
                    self._reset_selected_nodes(self.current_node_index)
                    # Important: with setting initial control handle 
                    # as the 'next' (= index 1) one,it brings us
                    # inkscape-like node creation.
                    self.current_handle_index = 1 

                    self._queue_draw_node(self.current_node_index)

        elif self.phase == _PhaseBezier.MOVE_NODE:
            if len(self.selected_nodes) > 0:
                # Use selection_rect class as offset-information
                self.selection_rect.start(mx, my)
        
        elif self.phase == _PhaseBezier.ADJUST_SELECTING:
            self.selection_rect.start(mx, my)
            self.selection_rect.is_addition = (event.state & Gdk.ModifierType.CONTROL_MASK)
            self._queue_draw_selection_rect() # to start
        elif self.phase == _PhaseBezier.ADJUST_HANDLE:
            self._last_event_node = self.nodes[self.target_node_index]
            pass
        elif self.phase == _PhaseBezier.CHANGE_PHASE:
            # DO NOT DO ANYTHING.
            pass
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)


    def drag_update_cb(self, tdw, event, dx, dy):
        self._ensure_overlay_for_tdw(tdw)
        mx, my = tdw.display_to_model(event.x, event.y)
        if self.phase == _PhaseBezier.CREATE_PATH:
            pass
            
        elif self.phase in (_PhaseBezier.ADJUST_HANDLE, _PhaseBezier.INIT_HANDLE):
            self._queue_redraw_curve(tdw)  
            node = self._last_event_node
            if self._last_event_node:
                self._queue_draw_node(self.current_node_index)# to erase
                node.set_control_handle(self.current_handle_index,
                        mx, my)

                self._queue_draw_node(self.current_node_index)
            self._queue_redraw_curve(tdw)
                
        elif self.phase == _PhaseBezier.MOVE_NODE:
            if len(self.selected_nodes) > 0:
                self._queue_redraw_curve(tdw)  
                self._queue_draw_selected_nodes()
                self.selection_rect.drag(mx, my)
                self._queue_draw_selected_nodes()
                self._queue_redraw_curve(tdw)
        elif self.phase == _PhaseBezier.ADJUST_SELECTING:
            self._queue_draw_selection_rect() # to erase
            self.selection_rect.drag(mx, my)
            self._queue_draw_selection_rect()
        elif self.phase == _PhaseBezier.CHANGE_PHASE:
            # DO NOT DO ANYTHING.
            pass
        else:
            raise NotImplementedError("Unknown phase %r" % self.phase)

    def drag_stop_cb(self, tdw):
        self._ensure_overlay_for_tdw(tdw)
        if self.phase == _PhaseBezier.CREATE_PATH:
            self._reset_adjust_data()
            if len(self.nodes) > 0:
                self._queue_redraw_curve(tdw)
                self._queue_redraw_all_nodes()
                if len(self.nodes) > 1:
                    self._queue_draw_buttons()
                
            
        elif self.phase in (_PhaseBezier.ADJUST_HANDLE, _PhaseBezier.INIT_HANDLE):
            node = self._last_event_node
      
            # At initialize handle phase, even if the node is not 'curve'
            # Set the handles as symmetry.
            if (self.phase == _PhaseBezier.INIT_HANDLE):
                node.curve = not self.DEFAULT_POINT_CORNER

            self._queue_redraw_all_nodes()
            self._queue_redraw_curve(tdw)
            if len(self.nodes) > 1:
                self._queue_draw_buttons()
                
            self.phase = _PhaseBezier.CREATE_PATH
        elif self.phase == _PhaseBezier.MOVE_NODE:
            dx, dy = self.selection_rect.get_model_offset()

            for idx in self.selected_nodes:
                cn = self.nodes[idx]
                cn.move(cn.x + dx, cn.y + dy)

            self.selection_rect.reset()
            self._dragged_node_start_pos = None
            self._queue_redraw_curve(tdw)
            self._queue_draw_buttons()
            self.phase = _PhaseBezier.CREATE_PATH
        elif self.phase == _PhaseBezier.ADJUST_SELECTING:
            ## Nodes selection phase
            self._queue_draw_selection_rect()

            modified = False
            if not self.selection_rect.is_addition:
                self._reset_selected_nodes()
                modified = True

            for idx,cn in enumerate(self.nodes):
                if self.selection_rect.is_inside(cn.x, cn.y):
                    if not idx in self.selected_nodes:
                        self.selected_nodes.append(idx)
                        modified = True

            if modified:
                self._queue_redraw_all_nodes()

            self._queue_draw_buttons() # buttons erased while selecting
            self.selection_rect.reset()

            # phase returns the last phase 

        elif self.phase == _PhaseBezier.CHANGE_PHASE:
            pass


        # Common processing
        if self.current_node_index != None:
            self.options_presenter.target = (self, self.current_node_index)

        if self._returning_phase != None:
            self.phase = self._returning_phase
            self._returning_phase = None
            self._queue_draw_buttons() 

    ## Interrogating events

    
    ## Interface methods which call from callbacks

    def execute_draw_polygon(self, mode=None, fill=True, fill_atop=False,
            erase_outside=False):

        if self.doc.model.layer_stack.current.get_fillable():
            if mode:
                composite_mode = mode
            else:
                if fill:
                    if fill_atop:
                        composite_mode = lib.mypaintlib.CombineSourceAtop
                    else:
                        composite_mode = lib.mypaintlib.CombineNormal
                else:
                    if erase_outside:
                        composite_mode = lib.mypaintlib.CombineDestinationIn
                    else:
                        composite_mode = lib.mypaintlib.CombineDestinationOut

                    
            bbox = self._get_maximum_rect(None)
            cmd = PolyFill(self.doc.model,
                    self.nodes,
                    self.foreground_color,
                    None,bbox,
                    composite_mode)
            self.doc.model.do(cmd)

            if not self._stroke_from_history:
                self.stroke_history.register(self.nodes)
                self.options_presenter.reset_stroke_history()
        else:
            logger.debug("Polyfilltool: target is not fillable layer.nothing done.")

        self._reset_all_internal_state()

    ## Node editing

    @property
    def options_presenter(self):
        """MVP presenter object for the node editor panel"""
        cls = self.__class__
        if cls._OPTIONS_PRESENTER_POLY is None:
            cls._OPTIONS_PRESENTER_POLY = OptionsPresenter_Polyfill()
        return cls._OPTIONS_PRESENTER_POLY

                                                
    def apply_pressure_from_curve_widget(self):
        """ apply pressure reprenting points
        from StrokeCurveWidget.
        """
        return # do nothing

    ## properties

    @property
    def polygon_preview_fill(self):
        return self._polygon_preview_fill

    @polygon_preview_fill.setter
    def polygon_preview_fill(self, flag):
        self._polygon_preview_fill = flag
        self._queue_redraw_curve()
    


class OverlayPolyfill (OverlayBezier):
    """Overlay for an BezierMode's adjustable points"""


    def __init__(self, mode, tdw):
        super(OverlayPolyfill, self).__init__(mode, tdw)
        self._draw_initial_handle_both = True
        

    def update_button_positions(self):
        mode = self._inkmode
        if mode.forced_button_pos:
            mode.button_info.setup_round_position(
                    self._tdw, mode.forced_button_pos)
        else:
            super(OverlayPolyfill, self).update_button_positions()
            mode.button_info.set_position(
                    self.reject_button_pos, self.accept_button_pos)

    def paint(self, cr):
        """Draw adjustable nodes to the screen"""
        # Control nodes
        mode = self._inkmode
        alloc = self._tdw.get_allocation()
        dx, dy = mode.selection_rect.get_display_offset(self._tdw)

        # drawing path
        _draw_node_polygon(cr, self._tdw, mode.nodes, 
                selected_nodes=mode.selected_nodes, dx=dx, dy=dy, 
                color = mode.foreground_color,
                stroke=True,
                fill = mode.polygon_preview_fill)

        super(OverlayPolyfill, self).paint(cr, draw_buttons=False)
                
        if (not mode.in_drag and len(mode.nodes) > 1):
            self.update_button_positions()
            radius = gui.style.FLOATING_BUTTON_RADIUS

            for pos, info in mode.button_info.enumurate_buttons():
                if pos is None:
                    continue
                x, y = pos
                id, icon_name, zone = info
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

class GradientStore(object):

    # default gradents.
    # a gradient consists from ( 'name', (color sequence), id )
    #
    # 'color sequence' is a sequence,which consists from color step
    # (position, (R, G, B)) or (position, (R, G, B, A))
    # -1 means foreground color, -2 means background color.
    # if you use RGBA format,every color sequence must have alpha value.
    #
    # 'id' is integer, this should be unique, to distinguish cache.

    DEFAULT_GRADIENTS = [ 
            (
                'Foreground to Transparent', 
                (
                    (0.0, (-1, -1, -1, 1)), (1.0, (-1, -1, -1, 0))
                ),
                0,
            ),
            (
                'Foreground to Background', 
                (
                    (0.0, (-1, -1, -1)) , (1.0, (-2, -2, -2))
                ),
                1
            ),
            (
                'Rainbow', 
                (
                    (0.0, (1.0, 0.0, 0.0)) , 
                    (0.25, (1.0, 1.0, 0.0)) , 
                    (0.5, (0.0, 1.0, 0.0)) , 
                    (0.75, (0.0, 1.0, 1.0)) , 
                    (1.0, (0.0, 0.0, 1.0))  
                ),
                2
            ),
        ]

    def __init__(self): 
        self._gradients = Gtk.ListStore(str,object,int)
        for cg in self.DEFAULT_GRADIENTS:
            self._gradients.append(cg)

        # _cairograds is a cache of cairo.pattern
        # key is Gtk.Treeiter
        self._cairograds = {}

    @property
    def liststore(self):
        return self._gradients


    def get_cairo_gradient(self, iter, sx, sy, ex, ey, fg, bg):
        return self.get_cairo_gradient_raw(self._gradients[iter])

    def get_cairo_gradient_raw(self, graddata, sx, sy, ex, ey, fg, bg):
        cg=cairo.LinearGradient(sx, sy, ex, ey)
        for pos, rgba in graddata:
            if len(rgba) == 4:
                r, g, b, a = rgba
            else:
                r, g, b = rgba

            if r == -1:
                r, g, b = fg
            elif r == -2:
                r, g, b = bg

            if len(rgba) == 4:
                cg.add_color_stop_rgba(pos, r, g, b, a)
            else:
                cg.add_color_stop_rgb(pos, r, g, b)

        return cg

    def get_gradient_for_treeview(self, iter):
        if not iter in self._cairograds or self._cairograd[iter] == None:
            cg, variable = self.get_cairo_gradient(
                    iter, 0, 8, 175, 8,
                    (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))# dummy fg/bg
            self._cairograd[iter]=cg
            return cg
        else:
            return self._cairograd[iter]




class GradientRenderer(Gtk.CellRenderer):
    gradient = GObject.property(type=GObject.TYPE_PYOBJECT, default=None)
    id = GObject.property(type=int , default=-1)

    def __init__(self, gradient_store):
        super(GradientRenderer, self).__init__()
        self.gradient_store = gradient_store
        self.cg={}

    def do_set_property(self, pspec, value):
        setattr(self, pspec.name, value)

    def do_get_property(self, pspec):
        return getattr(self, pspec.name)

    def do_render(self, cr, widget, background_area, cell_area, flags):
        """
        :param cell_area: RectangleInt class
        """
        cr.translate(0,0)

        # first colorstep has alpha = it uses alpha value = need background 
        if len(self.gradient[0][1]) > 3:
            self.draw_background(cr, cell_area)

        cr.rectangle(cell_area.x, cell_area.y, 
                cell_area.width, cell_area.height)
        cr.set_source(self.get_gradient(self.id, self.gradient, cell_area))
        cr.fill()
        # selected = (flags & Gtk.CellRendererState.SELECTED) != 0
        # prelit = (flags & Gtk.CellRendererState.PRELIT) != 0

    def get_gradient(self, id, gradient, cell_area):
        if not id in self.cg:
            halftop = cell_area.y + cell_area.height/2
            cg = self.gradient_store.get_cairo_gradient_raw(
                gradient,
                cell_area.x, halftop, 
                cell_area.x + cell_area.width - 1, halftop,
                (1.0, 1.0, 1.0),
                (0.0, 0.0, 0.0)
                )
            self.cg[id] = cg
            return cg
        return self.cg[id]

    def draw_background(self, cr, cell_area):
        h = 0
        tile_size = 8
        idx = 0
        cr.save()
        tilecolor = ( (0.3, 0.3, 0.3) , (0.7, 0.7, 0.7) )
        while h < cell_area.height:
            w = 0

            if h + tile_size > cell_area.height:
                h = cell_area.height - h

            while w < cell_area.width:

                if w + tile_size > cell_area.width:
                    w = cell_area.width - w

                cr.rectangle(cell_area.x + w, cell_area.y + h, 
                        tile_size, tile_size)
                cr.set_source_rgb(*tilecolor[idx%2])
                cr.fill()
                idx+=1
                w += tile_size
            h += tile_size
            idx += 1
        cr.restore()







class OptionsPresenter_Polyfill (OptionsPresenter_Bezier):
    """Presents UI for directly editing point values etc."""

    gradient_preset_store = GradientStore()

    def __init__(self):
        super(OptionsPresenter_Polyfill, self).__init__()

    def _ensure_ui_populated(self):
        if self._options_grid is not None:
            return
        self._updating_ui = True
        builder_xml = os.path.splitext(__file__)[0] + ".glade"
        builder = Gtk.Builder()
        builder.set_translation_domain("mypaint")
        builder.add_from_file(builder_xml)
        builder.connect_signals(self)
        self._options_grid = builder.get_object("options_grid")
        self._point_values_grid = builder.get_object("point_values_grid")
        self._point_values_grid.set_sensitive(True)
        self._opacity_adj = builder.get_object("opacity_adj")
        self._insert_button = builder.get_object("insert_point_button")
        self._insert_button.set_sensitive(False)
        self._delete_button = builder.get_object("delete_point_button")
        self._delete_button.set_sensitive(False)
        self._check_curvepoint = builder.get_object("checkbutton_curvepoint")
        self._check_curvepoint.set_sensitive(False)
        self._fill_polygon_checkbutton = builder.get_object("fill_checkbutton")
        self._fill_polygon_checkbutton.set_sensitive(True)

        self._shape_type_combobox = builder.get_object("shape_type_combobox")
        self._shape_type_combobox.set_sensitive(True)

        # Creating toolbar
        base_grid = builder.get_object("polygon_operation_grid")
        toolbar = gui.widgets.inline_toolbar(
            self._app,
            [
                ("PolygonFill", "mypaint-add-symbolic"),
                ("PolygonErase", "mypaint-remove-symbolic"),
                ("PolygonEraseOutside", "mypaint-up-symbolic"),
                ("PolygonFillAtop", "mypaint-down-symbolic"),
            ]
        )
        style = toolbar.get_style_context()
        style.set_junction_sides(Gtk.JunctionSides.TOP)
        base_grid.attach(toolbar, 0, 0, 2, 1)

        # Creating history combo
        combo = builder.get_object('path_history_combobox')
        combo.set_model(PolyfillMode.stroke_history.liststore)
        cell = Gtk.CellRendererText()
        combo.pack_start(cell,True)
        combo.add_attribute(cell,'text',0)
        self._stroke_history_combo = combo


        # Creating gradient sample
        store = self.gradient_preset_store.liststore

        treeview = Gtk.TreeView()
        treeview.set_size_request(175, 125)
        treeview.set_model(store)
        col = Gtk.TreeViewColumn(_('Name'), cell, text=0)
        treeview.append_column(col)

        cell = GradientRenderer(self.gradient_preset_store)
        col = Gtk.TreeViewColumn(_('Gradient'), cell, gradient=1, id=2)
        # and it is appended to the treeview
        treeview.append_column(col)
        treeview.set_hexpand(True)

        exp = Gtk.Expander()
        exp.set_label(_("Gradient Presets..."))
        exp.set_use_markup(False)
        exp.add(treeview)
        base_grid.attach(exp, 0, 2, 2, 1)
        self._gradientview = treeview
        exp.set_expanded(True)

        # the last line
        self._updating_ui = False

    @property
    def target(self):
        # this is exactly same as OptionsPresenter_Bezier,
        # but we need this to define @target.setter
        return super(OptionsPresenter_Polyfill, self).target

    @target.setter
    def target(self, targ):
        polyfillmode, cn_idx = targ
        polyfillmode_ref = None
        if polyfillmode:
            polyfillmode_ref = weakref.ref(polyfillmode)
        self._target = (polyfillmode_ref, cn_idx)
        # Update the UI
        if self._updating_ui:
            return
        self._updating_ui = True
        try:
            self._ensure_ui_populated()
            if 0 <= cn_idx < len(polyfillmode.nodes):
                cn = polyfillmode.nodes[cn_idx]
                self._check_curvepoint.set_sensitive(
                    self._shape_type_combobox.get_active() != _Shape.Bezier)
                self._check_curvepoint.set_active(cn.curve)
            else:
                self._check_curvepoint.set_sensitive(False)

            self._insert_button.set_sensitive(polyfillmode.can_insert_node(cn_idx))
            self._delete_button.set_sensitive(polyfillmode.can_delete_node(cn_idx))
            self._fill_polygon_checkbutton.set_active(polyfillmode.polygon_preview_fill)
        finally:
            self._updating_ui = False                               


    ## callback handlers

    def _toggle_fill_checkbutton_cb(self, button):
        if self._updating_ui:
            return
        polymode, node_idx = self.target
        if polymode:
            polymode.polygon_preview_fill = button.get_active()

    def shape_type_combobox_changed_cb(self, combo):
        if self._updating_ui:
            return
        polymode, junk = self.target
        if polymode:
            logger.warning("shape_type is not implemented yet")
            polymode.set_shape(combo.get_active())



    ## Other handlers are as implemented in superclass.  
