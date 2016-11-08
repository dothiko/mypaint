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
from gui.beziertool import _Control_Handle, _Node_Bezier, _PhaseBezier
from lib.command import Command
import lib.surface
import lib.mypaintlib
import lib.tiledsurface
from gui.oncanvas import *

## Module constants

POLYFILLMODES = (
    (lib.mypaintlib.CombineNormal, _("Normal")),
    (lib.mypaintlib.CombineDestinationOut, _("Erase")),
    (lib.mypaintlib.CombineDestinationIn, _("Erase Outside")),
    (lib.mypaintlib.CombineSourceAtop, _("Clipped")),
    )

## Enum defs


## Module funcs

def _render_polygon_to_layer(model, target_layer, shape, nodes, 
        color, gradient, bbox, mode=lib.mypaintlib.CombineNormal):
    """
    :param bbox: boundary box rectangle, in model coordinate
    :param mode: polygon drawing mode(enum _POLYDRAWMODE). 
                 * NOT layer composite mode*
    """
    sx, sy, w, h = bbox
    sx = int(sx)
    sy = int(sy)
    w = int(w)
    h = int(h)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    cr = cairo.Context(surf)

   #linpat = cairo.LinearGradient(0, 0, w, h);
   #linpat.add_color_stop_rgba(0.20,  1, 0, 0, 1);
   #linpat.add_color_stop_rgba(0.50,  0, 1, 0, 1);
   #linpat.add_color_stop_rgba( 0.80,  0, 0, 1, 1);
   #
   #_draw_node_polygon(cr, None, self.nodes, ox=sx, oy=sy,
   #        gradient=linpat)
    shape.draw_node_polygon(cr, None, nodes, ox=sx, oy=sy,
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
                with srcsurf.tile_request(tx, ty, readonly=True) as src:
                    if src is lib.tiledsurface.transparent_tile.rgba:
                        lib.mypaintlib.tile_clear_rgba16(dst)
                    else:
                        layer.composite_tile(dst, True, tx, ty, mipmap_level=0)
    else:
        tiles.update(layer.get_tile_coords())
        dstsurf = target_layer._surface
        for tx, ty in tiles:
            with dstsurf.tile_request(tx, ty, readonly=False) as dst:
                layer.composite_tile(dst, True, tx, ty, mipmap_level=0)


    lib.surface.finalize_surface(dstsurf, tiles)

    bbox = tuple(target_layer.get_full_redraw_bbox())
    target_layer.root.layer_content_changed(target_layer, *bbox)
    target_layer.autosave_dirty = True
    del layer


## Class defs
class _EditZone(EditZone_Mixin):
    """Enumeration of what the pointer is on in the ADJUST phase"""
    CONTROL_HANDLE = 104 

    FILL_ATOP_BUTTON = 201
    ERASE_BUTTON = 202
    ERASE_OUTSIDE_BUTTON = 203


class _ButtonInfo(object):
    """ Buttons infomation management class.
    In Polyfill tool, Buttons increased and eventually 
    become difficult to manage.
    """

    button_info = (
                ( 0, 'mypaint-ok-symbolic',
                    _EditZone.ACCEPT_BUTTON ),
                ( 1, 'mypaint-trash-symbolic',
                    _EditZone.REJECT_BUTTON ),
                ( 2, 'mypaint-eraser-symbolic', 
                    _EditZone.ERASE_BUTTON ),
                ( 3, 'mypaint-cut-symbolic', 
                    _EditZone.ERASE_OUTSIDE_BUTTON ),
                ( 4, 'mypaint-add-symbolic',
                    _EditZone.FILL_ATOP_BUTTON ),
            )

    button_zones = {
        _EditZone.ACCEPT_BUTTON:lib.mypaintlib.CombineNormal,
        _EditZone.REJECT_BUTTON:None,
        _EditZone.ERASE_BUTTON:lib.mypaintlib.CombineDestinationOut,
        _EditZone.ERASE_OUTSIDE_BUTTON:lib.mypaintlib.CombineDestinationIn,
        _EditZone.FILL_ATOP_BUTTON:lib.mypaintlib.CombineSourceAtop,
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

    def buttons_iter(self):
        for i in xrange(self._valid_count):
            yield (self._pos[i], self.button_info[i])

    def buttons_iter_draw(self):
        for i in xrange(self._valid_count):
            yield (self.button_info[i][1], self._pos[i], self.button_info[i][2])


class PolyFill(Command):
    """Polygon-fill on the current layer"""

    display_name = _("Polygon Fill")

    def __init__(self, doc, shape, nodes, color, gradient,
                 bbox, mode, **kwds):
        """
        :param bbox: boundary rectangle,in model coordinate.
        """
        super(PolyFill, self).__init__(doc, **kwds)
        self.shape = shape
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
        _render_polygon_to_layer(self.doc, dst_layer, 
                self.shape, self.nodes,
                self.color, self.gradient, self.bbox,
                self.composite_mode)

    def undo(self):
        layers = self.doc.layer_stack
        assert self.snapshot is not None
        layers.current.load_snapshot(self.snapshot)
        self.snapshot = None

## Shape classes
#  By switching these shape classes,
#  Polyfilltool supports various different shape such as 
#  rectangle or ellipse

class _Shape(object):
    CANCEL_EVENT = 1
    CALL_BASECLASS_HANDLER = 2
    CALL_ANCESTER_HANDLER = 3

    TYPE_BEZIER = 0
    TYPE_POLYLINE = 1
    TYPE_RECTANGLE = 2
    TYPE_ELLIPSE = 3

    MARGIN = 3
    accept_handle = False
    has_own_node_painter = False

    def generate_node(self):
        return _Node_Bezier(
                x=0, y=0,
                pressure=0.0,
                xtilt=0.0, ytilt=0.0,
                dtime=1.0
                )
    def get_maximum_rect(self, tdw, mode, dx=0, dy=0):
        raise NotImplementedError

    def update_event(self, event):
        self.shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
        self.ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK
        self.alt_state = event.state & Gdk.ModifierType.MOD1_MASK

    def clear_event(self):
        self.shift_state = False
        self.ctrl_state = False
        self.alt_state = False

    @staticmethod
    def draw_dash(cr, dot):
        cr.set_dash((), 0)
        cr.set_source_rgb(1,1,1)
        cr.stroke_preserve()
        cr.set_source_rgb(0,0,0)
        cr.set_dash((dot, ) )
        cr.stroke()



class _Shape_Bezier(_Shape):

    name = _("Bezier")
    accept_handle = True

    def __init__(self):
        pass

    def draw_node_polygon(self, cr, tdw, nodes, selected_nodes=None, 
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
            if fill:
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

                _Shape.draw_dash(cr, 4)

                if len(nodes) > 2:
                    cr.move_to(x,y)
                    cr.curve_to(x1, y1, x2, y2, x3, y3) 
                    _Shape.draw_dash(cr, 10)


            cr.restore()


    def get_maximum_rect(self, tdw, mode, dx=0, dy=0):
        """ get possible maximum rectangle 
        :param tdw: the target tileddrawwidget.if this is None,
                    all values(includeing dx,dy) recognized as 
                    model coordinate value.
        :rtype tuple: a tuple of (x, y, width, height)
        """
        if len(mode.nodes) < 2:
            return (0,0,0,0)
        margin = _Shape.MARGIN

        def adjust_from_control_handle(tdw, cn, n_index, h_index, 
                sx, sy, ex, ey, dx, dy, margin):
            if tdw:
                cx, cy = tdw.model_to_display(*cn.get_control_handle(h_index))
            else:
                cx, cy = cn.get_control_handle(h_index)

            if n_index in mode.selected_nodes:
                if (mode.current_handle_index == None or
                        mode.current_handle_index == h_index):
                    cx += dx
                    cy += dy

            return (min(sx, cx - margin), min(sy, cy - margin),
                    max(ex, cx + margin), max(ey, cy + margin))




        # Get boundary rectangle of each segment
        # and return the maximum 
        for i,cn in enumerate(mode.nodes):
            if tdw:
                cnx, cny = tdw.model_to_display(cn.x, cn.y)
            else:
                cnx, cny = cn

            if i in mode.selected_nodes:
                cnx+=dx
                cny+=dy

            if i == 0:
                sx = cnx - margin 
                ex = cnx + margin 
                sy = cny - margin 
                ey = cny + margin 
            else:
                sx = min(sx, cnx - margin)
                ex = max(ex, cnx + margin)
                sy = min(sy, cny - margin)
                ey = max(ey, cny + margin)

            sx, sy, ex, ey = adjust_from_control_handle(tdw, cn, i, 0,
                sx, sy, ex, ey, dx, dy, margin)

            sx, sy, ex, ey = adjust_from_control_handle(tdw, cn, i, 1,
                sx, sy, ex, ey, dx, dy, margin)

        return (sx, sy, abs(ex - sx) + 1, abs(ey - sy) + 1)
        

    def button_press_cb(self, mode, tdw, event):

        if mode.phase in (_PhaseBezier.CREATE_PATH,):
            if mode.zone == _EditZone.CONTROL_NODE:
                # Grabbing a node...
                button = event.button
                if mode.phase == _PhaseBezier.CREATE_PATH:

                    # normal move node start
                    mode.phase = _PhaseBezier.MOVE_NODE

                    if button == 1 and mode.current_node_index != None:
                        if self.ctrl_state:
                            # Holding CONTROL key = adding or removing a node.
                            if mode.current_node_index in mode.selected_nodes:
                                mode.selected_nodes.remove(mode.current_node_index)
                            else:
                                mode.selected_nodes.append(mode.current_node_index)
        
                            mode._queue_draw_selected_nodes() 
                        else:
                            # no CONTROL Key holded.
                            # If new solo node clicked without holding 
                            # CONTROL key,then reset all selected nodes.
        
                            do_reset = self.alt_state
                            do_reset |= not (mode.current_node_index in mode.selected_nodes)
        
                            if do_reset:
                                # To avoid old selected nodes still lit.
                                mode._queue_draw_selected_nodes() 
                                mode._reset_selected_nodes(mode.current_node_index)

                # FALLTHRU: *do* start a drag 

            elif mode.zone == _EditZone.EMPTY_CANVAS:
                
                if mode.phase == _PhaseBezier.CREATE_PATH:
                    if (len(mode.nodes) > 0): 
                       #if shift_state and ctrl_state:
                        if self.ctrl_state:
                            mx, my = tdw.display_to_model(event.x, event.y)
                            pressed_segment = mode._detect_on_stroke(mx, my)
                            if pressed_segment:
                                # pressed_segment is a tuple which contains
                                # (node index of start of segment, stroke step)

                                # To erase buttons 
                                mode._queue_draw_buttons() 

                                mode._divide_bezier(*pressed_segment)

                                # queue new node here.
                                mode._queue_draw_node(pressed_segment[0] + 1)
                                
                                mode.phase = _PhaseBezier.PLACE_NODE
                                return _Shape.CANCEL_EVENT # Cancel drag event


            elif mode.zone == _EditZone.CONTROL_HANDLE:
                if mode.phase == _PhaseBezier.CREATE_PATH:
                    mode.phase = _PhaseBezier.ADJUST_HANDLE


            # FALLTHRU: *do* start a drag 

        elif mode.phase in (_PhaseBezier.ADJUST_HANDLE, _PhaseBezier.INIT_HANDLE):
            pass
        elif mode.phase in (_PhaseBezier.MOVE_NODE, _PhaseBezier.CHANGE_PHASE):
            # THIS CANNOT BE HAPPEN...might be an evdev dropout.through it.
            pass
        else:
            raise NotImplementedError("Unrecognized phase %r", mode.phase)




    def button_release_cb(self, mode, tdw, event):

        # Here is 'button_release_cb',which called 
        # prior to drag_stop_cb.
        # so, in this method, changing mode._phase
        # is very special case. 
        if mode.phase == _PhaseBezier.PLACE_NODE:
            mode._queue_redraw_curve(tdw) 
            mode.phase = _PhaseBezier.CREATE_PATH



    def drag_start_cb(self, mode, tdw, event):
        # Basically,all sections should do fall-through.
        mx, my = tdw.display_to_model(event.x, event.y)

        if mode.phase == _PhaseBezier.CREATE_PATH:

            if mode.zone == _EditZone.EMPTY_CANVAS:
                if event.state != 0:
                    # To activate some mode override
                    mode._last_event_node = None
                    return _Shape.CALL_ANCESTER_HANDLER
                else:
                    # New node added!
                    node = mode._get_event_data(tdw, event)
                    mode.nodes.append(node)
                    mode._last_event_node = node
                    mode.phase = _PhaseBezier.INIT_HANDLE
                    mode.current_node_index=len(mode.nodes)-1
                    mode._reset_selected_nodes(mode.current_node_index)
                    # Important: with setting initial control handle 
                    # as the 'next' (= index 1) one,it brings us
                    # inkscape-like node creation.
                    mode.current_handle_index = 1 

                    mode._queue_draw_node(mode.current_node_index)

        elif mode.phase == _PhaseBezier.MOVE_NODE:
            if len(mode.selected_nodes) > 0:
                mode.drag_offset.start(mx, my)
        elif mode.phase == _PhaseBezier.ADJUST_HANDLE:
            mode._last_event_node = mode.nodes[mode.target_node_index]
        elif mode.phase == _PhaseBezier.CHANGE_PHASE:
            # DO NOT DO ANYTHING.
            pass
        else:
            raise NotImplementedError("Unknown phase %r" % mode.phase)


    def drag_update_cb(self, mode, tdw, event, dx, dy):

        mx, my = tdw.display_to_model(event.x, event.y)

        if mode.phase == _PhaseBezier.CREATE_PATH:
            pass
            
        elif mode.phase in (_PhaseBezier.ADJUST_HANDLE, _PhaseBezier.INIT_HANDLE):
            mode._queue_redraw_curve(tdw)  
            node = mode._last_event_node
            if mode._last_event_node:
                mode._queue_draw_node(mode.current_node_index)# to erase
                node.set_control_handle(mode.current_handle_index,
                        mx, my,
                        self.shift_state)

                mode._queue_draw_node(mode.current_node_index)
            mode._queue_redraw_curve(tdw)
                
        elif mode.phase == _PhaseBezier.MOVE_NODE:
            if len(mode.selected_nodes) > 0:
                mode._queue_redraw_curve(tdw)  
                mode._queue_draw_selected_nodes()
                mode.drag_offset.end(mx, my)
                mode._queue_draw_selected_nodes()
                mode._queue_redraw_curve(tdw)
        elif mode.phase == _PhaseBezier.CHANGE_PHASE:
            # DO NOT DO ANYTHING.
            pass
        else:
            raise NotImplementedError("Unknown phase %r" % mode.phase)

    def drag_stop_cb(self, mode, tdw):
        if mode.phase == _PhaseBezier.CREATE_PATH:
            mode._reset_adjust_data()
            if len(mode.nodes) > 0:
                mode._queue_redraw_curve(tdw)
                mode._queue_redraw_all_nodes()
                if len(mode.nodes) > 1:
                    mode._queue_draw_buttons()
                
            
        elif mode.phase in (_PhaseBezier.ADJUST_HANDLE, _PhaseBezier.INIT_HANDLE):
            node = mode._last_event_node
      
            # At initialize handle phase, even if the node is not 'curve'
            # Set the handles as symmetry.
            if (mode.phase == _PhaseBezier.INIT_HANDLE):
                node.curve = not mode.DEFAULT_POINT_CORNER

            mode._queue_redraw_all_nodes()
            mode._queue_redraw_curve(tdw)
            if len(mode.nodes) > 1:
                mode._queue_draw_buttons()
                
            mode.phase = _PhaseBezier.CREATE_PATH
        elif mode.phase == _PhaseBezier.MOVE_NODE:
            dx, dy = mode.drag_offset.get_model_offset()

            for idx in mode.selected_nodes:
                cn = mode.nodes[idx]
                cn.move(cn.x + dx, cn.y + dy)

            mode.drag_offset.reset()
            mode._dragged_node_start_pos = None
            mode._queue_redraw_curve(tdw)
            mode._queue_draw_buttons()
            mode.phase = _PhaseBezier.CREATE_PATH

        elif mode.phase == _PhaseBezier.CHANGE_PHASE:
            pass


class _Shape_Polyline(_Shape_Bezier):

    name = _("Polyline")
    accept_handle = False

    def __init__(self):
        pass

    def draw_node_polygon(self, cr, tdw, nodes, selected_nodes=None, 
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
            if fill:
                if color:
                    cr.set_source_rgb(*color.get_rgb())
                elif gradient:
                    cr.set_source(gradient)

            for i, node in enumerate(nodes):

                if tdw:
                    x, y = tdw.model_to_display(node.x, node.y)
                else:
                    x, y = node

                x-=ox
                y-=oy

                if selected_nodes:

                    if i in selected_nodes:
                        x += dx
                        y += dy


                if i==0:
                    cr.move_to(x, y)
                else:
                    cr.line_to(x, y)



            if fill and len(nodes) > 2 and (gradient or color):
                cr.close_path()
                cr.fill_preserve()

            if stroke:
                _Shape.draw_dash(cr, 4)

                if len(nodes) > 2:
                    cr.move_to(x,y)
                    x, y = tdw.model_to_display(nodes[0].x, nodes[0].y)
                    cr.line_to(x,y)
                    _Shape.draw_dash(cr, 10)

            cr.restore()

    def drag_start_cb(self, mode, tdw, event):
        # Basically,all sections should do fall-through.
        mx, my = tdw.display_to_model(event.x, event.y)

        if mode.phase == _PhaseBezier.CREATE_PATH:

            if mode.zone == _EditZone.EMPTY_CANVAS:
                if event.state != 0:
                    # To activate some mode override
                    mode._last_event_node = None
                    return _Shape.CALL_ANCESTER_HANDLER
                else:
                    # New node added!
                    node = mode._get_event_data(tdw, event)
                    mode.nodes.append(node)
                    mode._last_event_node = node
                    mode.current_node_index = len(mode.nodes)-1
                    mode._reset_selected_nodes(mode.current_node_index)
                    mode._queue_draw_node(mode.current_node_index)
                    mode.phase = _PhaseBezier.MOVE_NODE
                    mode.drag_offset.start(mx, my)
        else:
            return super(_Shape_Polyline, self).drag_start_cb(
                    mode, tdw, event)

class _Shape_Rectangle(_Shape):

    name = _("Rectangle")
    has_own_node_painter = True

    def __init__(self):
        pass

    def draw_node_polygon(self, cr, tdw, nodes, selected_nodes=None, 
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
        if len(nodes) >= 4:
            if selected_nodes and len(selected_nodes) >= 1:
                selidx = selected_nodes[0]
            else:
                selidx = -1

            if tdw:
                sx, sy = tdw.model_to_display(*nodes[0])
                ex, ey = tdw.model_to_display(*nodes[2])
            else:
                sx, sy = nodes[0]
                ex, ey = nodes[2]
                sx -= ox
                ex -= ox
                sy -= oy
                ey -= oy


            cr.save()
            cr.set_line_width(1)
            if fill:
                if color:
                    cr.set_source_rgb(*color.get_rgb())
                elif gradient:
                    cr.set_source(gradient)
                
            if selidx > -1:
                if selidx in (0, 3):
                    sx += dx
                else:
                    ex += dx
            
                if selidx in (0, 1):
                    sy += dy
                else:
                    ey += dy


            cr.move_to(sx,sy)
            cr.line_to(ex,sy)
            cr.line_to(ex,ey)
            cr.line_to(sx,ey)
            cr.line_to(sx,sy)

            if fill and (gradient or color):
                cr.close_path()
                cr.fill_preserve()

            if stroke:
                _Shape.draw_dash(cr, 4)

            cr.restore()

    def paint_nodes(self, cr, tdw, mode, radius):
        if len(mode.nodes) >= 4:
            dx, dy = mode.drag_offset.get_display_offset(tdw)
            sx, sy, ex, ey = self._setup_node_area(tdw, mode, dx, dy)

            for i, x, y in gui.ui_utils.enum_area_point(sx, sy, ex, ey):
                if i == mode.current_node_index:
                    color = gui.style.ACTIVE_ITEM_COLOR
                else:
                    color = gui.style.EDITABLE_ITEM_COLOR

                gui.drawutils.render_round_floating_color_chip(
                    cr=cr, x=x, y=y,
                    color=color,
                    radius=radius)


    def get_maximum_rect(self, tdw, mode, dx=0, dy=0):
        """
        Get maximum rectangle area:

        :param dx: offset of currently selected nodes.
                   if tdw is not none, dx and dy MUST be
                   display coordinate.
        :param dy: offset of currently selected nodes.
        """
        sx, sy, ex, ey = self._setup_node_area(tdw, mode, dx, dy)
        if sx > ex:
            sx, ex = ex, sx
        if sy > ey:
            sy, ey = ey, sy

        if tdw:
            margin = _Shape.MARGIN
            sx -= margin
            sy -= margin
            ex += margin
            ey += margin

        return (sx, sy, abs(ex-sx)+1, abs(ey-sy)+1)

    def set_area(self, mode, sx, sy, ex, ey):
        if ex < sx:
            sx, ex = ex, sx

        if ey < sy:
            sy, ey = ey, sy

        for i, x, y in gui.ui_utils.enum_area_point(sx, sy, ex, ey):
            mode.nodes[i].x = x
            mode.nodes[i].y = y

    def ensure_mode_nodes(self, mode, x, y):
        for i in xrange(4 - len(mode.nodes)):
            mode.nodes.append(self.generate_node())

        for i in xrange(4):
            mode.nodes[i].x = x
            mode.nodes[i].y = y


    def _setup_node_area(self, tdw, mode, dx, dy):
        """
        Setup nodes as rectangle.

        :param dx: offset of currently selected nodes.
                   if tdw is not none, dx and dy MUST be
                   display coordinate.
        :param dy: offset of currently selected nodes.
        """
        sx, sy = mode.nodes[0]
        ex, ey = mode.nodes[2]

        if tdw:
            sx, sy = tdw.model_to_display(sx, sy)
            ex, ey = tdw.model_to_display(ex, ey)

        if mode.current_node_index in (0, 3):
            sx += dx
        else:
            ex += dx

        if mode.current_node_index in (0, 1):
            sy += dy
        else:
            ey += dy

        if ex < sx:
            sx, ex = ex, sx

        if ey < sy:
            sy, ey = ey, sy

        return (sx, sy, ex, ey)


    def queue_redraw_nodes(self, tdw, mode):
        dx, dy = mode.drag_offset.get_display_offset(tdw)
        sx, sy, ex, ey = self._setup_node_area(tdw, mode, dx, dy)

        size = math.ceil(gui.style.DRAGGABLE_POINT_HANDLE_SIZE * 2)

        for i, x, y in gui.ui_utils.enum_area_point(sx, sy, ex, ey):
            x = math.floor(x)
            y = math.floor(y)
            tdw.queue_draw_area(x-size, y-size, size*2+1, size*2+1)

    def button_press_cb(self, mode, tdw, event):
        mx, my = tdw.display_to_model(event.x, event.y)

        if mode.phase in (_PhaseBezier.CREATE_PATH,):
            if mode.zone == _EditZone.CONTROL_NODE:
                # Grabbing a node...
                button = event.button
                # normal move node start
                mode.phase = _PhaseBezier.MOVE_NODE
                mode.selected_nodes = (mode.current_node_index, )

                # FALLTHRU: *do* start a drag 

            elif mode.zone == _EditZone.EMPTY_CANVAS:
                self.ensure_mode_nodes(mode, mx, my)


            # FALLTHRU: *do* start a drag 

        elif mode.phase in (_PhaseBezier.MOVE_NODE, _PhaseBezier.CHANGE_PHASE):
            # THIS CANNOT BE HAPPEN...might be an evdev dropout.through it.
            pass
        else:
            raise NotImplementedError("Unrecognized phase %r", mode.phase)



    def button_release_cb(self, mode, tdw, event):

        # Here is 'button_release_cb',which called 
        # prior to drag_stop_cb.
        # so, in this method, changing mode._phase
        # is very special case. 
        if mode.phase == _PhaseBezier.PLACE_NODE:
            mode._queue_redraw_curve(tdw) 
            mode.phase = _PhaseBezier.CREATE_PATH



    def drag_start_cb(self, mode, tdw, event):
        # Basically,all sections should do fall-through.
        mx, my = tdw.display_to_model(event.x, event.y)

        if mode.phase == _PhaseBezier.CREATE_PATH:

            if mode.zone == _EditZone.EMPTY_CANVAS:
                if event.state != 0:
                    # To activate some mode override
                    mode._last_event_node = None
                    return _Shape.CALL_ANCESTER_HANDLER
                else:
                    # New node added!
                    mode.current_node_index=0
                    mode._reset_selected_nodes(mode.current_node_index)
                    mode.drag_offset.start(mx, my)
                    mode._queue_redraw_curve(tdw)  
                    mode._queue_redraw_all_nodes()

        elif mode.phase == _PhaseBezier.MOVE_NODE:
            if len(mode.selected_nodes) > 0:
                mode.drag_offset.start(mx, my)
        elif mode.phase == _PhaseBezier.CHANGE_PHASE:
            pass
        else:
            raise NotImplementedError("Unknown phase %r" % mode.phase)


    def drag_update_cb(self, mode, tdw, event, dx, dy):

        mx, my = tdw.display_to_model(event.x, event.y)

        if mode.phase == _PhaseBezier.CREATE_PATH:
            self.queue_redraw_nodes(tdw, mode)
            mode._queue_redraw_curve(tdw)  
            mode.drag_offset.end(mx, my)
            self.queue_redraw_nodes(tdw, mode)
            mode._queue_redraw_curve(tdw)  
        elif mode.phase == _PhaseBezier.MOVE_NODE:
            if len(mode.selected_nodes) > 0:
                mode._queue_redraw_curve(tdw)  
                mode._queue_redraw_all_nodes()
                mode.drag_offset.end(mx, my)
                mode._queue_redraw_all_nodes()
                mode._queue_redraw_curve(tdw)
        elif mode.phase == _PhaseBezier.CHANGE_PHASE:
            # DO NOT DO ANYTHING.
            pass
        else:
            raise NotImplementedError("Unknown phase %r" % mode.phase)

    def drag_stop_cb(self, mode, tdw):
        if mode.phase == _PhaseBezier.CREATE_PATH:
            sx, sy = tdw.display_to_model(
                    mode.start_x, mode.start_y)
            dx, dy = mode.drag_offset.get_model_offset()

            ex = sx + dx
            ey = sy + dy

            self.set_area(mode, sx, sy, ex, ey)

            mode._queue_redraw_curve(tdw)
            mode._queue_redraw_all_nodes()
            mode._queue_draw_buttons()
            mode._reset_adjust_data()
            
        elif mode.phase == _PhaseBezier.MOVE_NODE:
            dx, dy = mode.drag_offset.get_model_offset()

            # Move entire rectangle, when 4 nodes selected. 
            if len(mode.selected_nodes) == 4:
                for i in xrange(4):
                    cn = mode.nodes[i]
                    cn.move(cn.x + dx, cn.y + dy)
            else:
                # Otherwise, only one node could move.
                sx, sy = mode.nodes[0]
                ex, ey = mode.nodes[2]

                if mode.current_node_index in (0, 3):
                    sx += dx
                else:
                    ex += dx

                if mode.current_node_index in (0, 1):
                    sy += dy
                else:
                    ey += dy

                self.set_area(mode, sx, sy, ex, ey)
                

            mode.drag_offset.reset()
            mode._queue_redraw_curve(tdw)
            mode._queue_draw_buttons()
            mode._queue_redraw_all_nodes()
            mode.phase = _PhaseBezier.CREATE_PATH
            mode._reset_adjust_data()



class _Shape_Ellipse(_Shape_Rectangle):

    name = _("Ellipse")

    def __init__(self):
        pass

    def draw_node_polygon(self, cr, tdw, nodes, selected_nodes=None, 
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
        if len(nodes) >= 4:
            if selected_nodes and len(selected_nodes) >= 1:
                selidx = selected_nodes[0]
            else:
                selidx = -1

            if tdw:
                sx, sy = tdw.model_to_display(*nodes[0])
                ex, ey = tdw.model_to_display(*nodes[2])
            else:
                sx, sy = nodes[0]
                ex, ey = nodes[2]
                sx -= ox
                ex -= ox
                sy -= oy
                ey -= oy


            cr.save()
            cr.set_line_width(1)
            if fill:
                if color:
                    cr.set_source_rgb(*color.get_rgb())
                elif gradient:
                    cr.set_source(gradient)
                
            if selidx > -1:
                if selidx in (0, 3):
                    sx += dx
                else:
                    ex += dx

                if selidx in (0, 1):
                    sy += dy
                else:
                    ey += dy

            w = abs(ex - sx) + 1
            h = abs(ey - sy) + 1

            if sx > ex:
                sx = ex

            if sy > ey:
                sy = ey

            # XXX This 'emulated ellipse' is not circle actually
            # ... but almost okay?
            # code from:
            # http://stackoverflow.com/questions/14169234/the-relation-of-the-bezier-curve-and-ellipse
            hw = w / 2.0
            hh = h / 2.0
            tw = w * (2.0 / 3.0)
            x = sx + hw
            y = sy + hh

            cr.move_to(x, y - hh);
            cr.curve_to(x + tw, y - hh, x + tw, y + hh, x, y + hh)
            cr.curve_to(x - tw, y + hh, x - tw, y - hh, x, y - hh)

            if fill and len(nodes) > 2 and (gradient or color):
                cr.close_path()
                cr.fill_preserve()

            if stroke:
                _Shape.draw_dash(cr, 4)

            cr.restore()


## The Polyfill Mode Class

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
    _shape_pool = { _Shape.TYPE_BEZIER : _Shape_Bezier() ,
                    _Shape.TYPE_POLYLINE : _Shape_Polyline() ,
                    _Shape.TYPE_RECTANGLE : _Shape_Rectangle() ,
                    _Shape.TYPE_ELLIPSE : _Shape_Ellipse() }
    _shape = None


    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):
        super(PolyfillMode, self).__init__(**kwargs)
        self._polygon_preview_fill = False
        self.options_presenter.target = (self, None)
        self.phase = _PhaseBezier.CREATE_PATH
        if PolyfillMode._shape == None:
            self.shape_type = _Shape.TYPE_BEZIER



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

        self._ensure_overlay_for_tdw(tdw)
        new_zone = _EditZone.EMPTY_CANVAS
        if not self.in_drag and len(self.nodes) > 0:
            if self.phase in (_PhaseBezier.MOVE_NODE, 
                    _PhaseBezier.CREATE_PATH):

                new_target_node_index = None
                
                # Test buttons for hits
                hit_dist = gui.style.FLOATING_BUTTON_RADIUS

                if len(self.nodes) > 1:
                    for pos, info in self.button_info.buttons_iter():
                        if pos is None:
                            continue
                        btn_x, btn_y = pos
                        d = math.hypot(btn_x - x, btn_y - y)
                        if d <= hit_dist:
                            new_target_node_index = None
                            new_zone = info[2]
                            break

                if (new_zone == _EditZone.EMPTY_CANVAS):

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
                            new_zone = _EditZone.CONTROL_HANDLE
                            break         

                    # Test nodes for a hit, in reverse draw order
                    if new_target_node_index == None:
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
                if self.zone == _EditZone.CONTROL_NODE:
                    cursor = self._crosshair_cursor
                elif self.zone in self.button_info.button_zones:
                    cursor = self._crosshair_cursor
                else:
                    cursor = self._arrow_cursor
            if cursor is not self._current_override_cursor:
                tdw.set_override_cursor(cursor)
                self._current_override_cursor = cursor



    def _reset_all_internal_state(self):
        self._queue_draw_buttons()
        self._queue_redraw_all_nodes()
        self._queue_redraw_curve() 
        self._reset_nodes()
        self._reset_capture_data()
        self._reset_adjust_data()
        self._stroke_from_history = False
        self.forced_button_pos = None

    ## Properties
    @property
    def shape_type(self):
        return PolyfillMode._shape_type

    @property
    def shape(self):
        return PolyfillMode._shape

    @shape_type.setter
    def shape_type(self, shape_type):
        PolyfillMode._shape = PolyfillMode._shape_pool[shape_type]
        PolyfillMode._shape_type = shape_type

    ## Redraws
    

    def redraw_curve_cb(self, erase=False):
        """ Frontend method,to redraw curve from outside this class"""
        pass # do nothing for now

    def _queue_redraw_curve(self, tdw=None):

        for tdw in self._overlays:
            
            if len(self.nodes) < 2:
                continue

            sdx, sdy = self.drag_offset.get_display_offset(tdw)
            tdw.queue_draw_area(
                    *self._shape.get_maximum_rect(tdw, self, sdx, sdy))


    def _queue_draw_buttons(self):
        for tdw, overlay in self._overlays.items():
            overlay.update_button_positions()
            for pos, info in self.button_info.buttons_iter():
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

    def _start_new_capture_phase_polyfill(self, mode, rollback=False):
        if rollback:
            self._stop_task_queue_runner(complete=False)
            self._reset_all_internal_state()
        else:
            self._stop_task_queue_runner(complete=True)
            self.execute_draw_polygon(mode=mode)

        self._reset_adjust_data()


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
        shape = self._shape
        shape.update_event(event)

        self._update_zone_and_target(tdw, event.x, event.y,
                shape.alt_state or not shape.accept_handle)
        self._update_current_node_index()



        # common processing for all shape type.
        if self.phase in (_PhaseBezier.CREATE_PATH,):
            # Initial state - everything starts here!
       
            if (self.zone in self.button_info.button_zones): 
                if (event.button == 1 and 
                        event.type == Gdk.EventType.BUTTON_PRESS):

                        # To avoid some of visual glitches,
                        # we need to process button here.
                        if self.zone == _EditZone.REJECT_BUTTON:
                            self.discard_edit()
                        else:
                            self.accept_edit()
                        
                        return False

            elif self.zone == _EditZone.EMPTY_CANVAS:
                
                if self.phase == _PhaseBezier.CREATE_PATH:
                    if (len(self.nodes) > 0): 
                       #if shift_state and ctrl_state:
                        if shape.alt_state:
                            self._queue_draw_buttons() 
                            self.forced_button_pos = (event.x, event.y)
                            self.phase = _PhaseBezier.CHANGE_PHASE 
                            self._returning_phase = _PhaseBezier.CREATE_PATH
                            self._queue_draw_buttons() 
                            return False

        
        ret = shape.button_press_cb(self, tdw, event)
        if ret == _Shape.CANCEL_EVENT:
            return False
        elif ret == _Shape.CALL_BASECLASS_HANDLER:
            return super(PolyFillMode, self).button_press_cb(tdw, event) 
        elif ret == _Shape.CALL_ANCESTER_HANDLER:
            return super(InkingMode, self).button_press_cb(tdw, event) 

        # Update workaround state for evdev dropouts
        self._button_down = event.button

        # Super-Supercall(not supercall) would invoke drag-related callbacks.
        return super(InkingMode, self).button_press_cb(tdw, event) 

    def button_release_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False

        shape = self._shape
        ret = shape.button_release_cb(self, tdw, event)
        if ret == _Shape.CANCEL_EVENT:
            return False
        elif ret == _Shape.CALL_BASECLASS_HANDLER:
            return super(PolyFillMode, self).button_release_cb(tdw, event) 
        elif ret == _Shape.CALL_ANCESTER_HANDLER:
            return super(InkingMode, self).button_release_cb(tdw, event) 

        # Update workaround state for evdev dropouts
        self._button_down = None

        # Super-Supercall(not supercall) would invoke drag_stop_cb signal.
        return super(InkingMode, self).button_release_cb(tdw, event)

    def motion_notify_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False

        shape = self._shape

        self._update_zone_and_target(tdw, event.x, event.y,
                ignore_handle = not shape.accept_handle)

        return super(InkingMode, self).motion_notify_cb(tdw, event)
        

    ## Drag handling (both capture and adjust phases)
    def drag_start_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)

        self._queue_previous_draw_buttons() # To erase button,and avoid glitch
        shape = self._shape

        ret = shape.drag_start_cb(self, tdw, event)
        if ret == _Shape.CANCEL_EVENT:
            return False
        elif ret == _Shape.CALL_BASECLASS_HANDLER:
            return super(PolyFillMode, self).drag_start_cb(tdw, event) 
        elif ret == _Shape.CALL_ANCESTER_HANDLER:
            return super(InkingMode, self).drag_start_cb(tdw, event) 

    def drag_update_cb(self, tdw, event, dx, dy):
        self._ensure_overlay_for_tdw(tdw)
        shape = self._shape

        ret = shape.drag_update_cb(self, tdw, event, dx, dy)
        if ret == _Shape.CANCEL_EVENT:
            return False
        elif ret == _Shape.CALL_BASECLASS_HANDLER:
            return super(PolyFillMode, self).drag_update_cb(tdw, event) 
        elif ret == _Shape.CALL_ANCESTER_HANDLER:
            return super(InkingMode, self).drag_update_cb(tdw, event) 
        

    def drag_stop_cb(self, tdw):
        self._ensure_overlay_for_tdw(tdw)
        shape = self._shape
        try:

            ret = shape.drag_stop_cb(self, tdw)
            if ret == _Shape.CANCEL_EVENT:
                return False
            elif ret == _Shape.CALL_BASECLASS_HANDLER:
                return super(PolyFillMode, self).drag_stop_cb(tdw)
            elif ret == _Shape.CALL_ANCESTER_HANDLER:
                return super(InkingMode, self).drag_stop_cb(tdw)

            # Common processing
            if self.current_node_index != None:
                self.options_presenter.target = (self, self.current_node_index)

            if self._returning_phase != None:
                self.phase = self._returning_phase
                self._returning_phase = None
                self._queue_draw_buttons() 

        finally:
            shape.clear_event()


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

                    
            bbox = self._shape.get_maximum_rect(None, self)
            cmd = PolyFill(self.doc.model,
                    self.shape,
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
    
    ## Generic Oncanvas-editing handler
    def accept_edit(self):
        if (self.phase in (_PhaseBezier.CREATE_PATH, ) and
                len(self.nodes) > 1):
            self._start_new_capture_phase_polyfill(
                self.button_info.get_mode_from_zone(self.zone),
                rollback=False)

    def discard_edit(self):
        if (self.phase in (_PhaseBezier.CREATE_PATH, )):
            self._start_new_capture_phase_polyfill(
                None, rollback=True)


class OverlayPolyfill (OverlayBezier):
    """Overlay for an BezierMode's adjustable points"""


    def __init__(self, mode, tdw):
        super(OverlayPolyfill, self).__init__(mode, tdw)
        self._draw_initial_handle_both = True
        

    def update_button_positions(self):
        mode = self._mode
        if mode.forced_button_pos:
            mode.button_info.setup_round_position(
                    self._tdw, mode.forced_button_pos)
        else:
            super(OverlayPolyfill, self).update_button_positions(
                    not mode.shape.accept_handle)
            # FIXME:: Copy basic 2 buttons position
            # from base class codes.
            # Actually, copied values used when drawing
            # and self._button_pos[] values are ignored.
            # This code is unefficient, needs to be rewritten.
            mode.button_info.set_position(
                    self.get_button_pos(_EditZone.ACCEPT_BUTTON), 
                    self.get_button_pos(_EditZone.REJECT_BUTTON))

    def paint(self, cr):
        """Draw adjustable nodes to the screen"""
        mode = self._mode
        alloc = self._tdw.get_allocation()
        dx, dy = mode.drag_offset.get_display_offset(self._tdw)
        shape = mode.shape

        # drawing path
        shape.draw_node_polygon(cr, self._tdw, mode.nodes, 
                selected_nodes=mode.selected_nodes, dx=dx, dy=dy, 
                color = mode.foreground_color,
                stroke=True,
                fill = mode.polygon_preview_fill)

        # drawing control nodes
        if shape.has_own_node_painter:
            shape.paint_nodes(cr, self._tdw, mode,
                    gui.style.DRAGGABLE_POINT_HANDLE_SIZE)
        else:
            super(OverlayPolyfill, self).paint(cr, draw_buttons=False)
                
        if (not mode.in_drag and len(mode.nodes) > 1):
            self.update_button_positions()
           #radius = gui.style.FLOATING_BUTTON_RADIUS
            self._draw_buttons(cr, mode.button_info.buttons_iter_draw())

           #for pos, info in mode.button_info.buttons_iter():
           #    if pos is None:
           #        continue
           #    x, y = pos
           #    id, icon_name, zone = info
           #    if mode.zone == zone:
           #        color = gui.style.ACTIVE_ITEM_COLOR
           #    else:
           #        color = gui.style.EDITABLE_ITEM_COLOR
           #    icon_pixbuf = self._get_button_pixbuf(icon_name)
           #    gui.drawutils.render_round_floating_button(
           #        cr=cr, x=x, y=y,
           #        color=color,
           #        pixbuf=icon_pixbuf,
           #        radius=radius,
           #    )

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
        return (self._target[0](), self._target[1])

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
                    self._shape_type_combobox.get_active() != _Shape.TYPE_BEZIER)
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
            polymode.shape_type = combo.get_active()



    ## Other handlers are as implemented in superclass.  
