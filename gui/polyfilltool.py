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
            else:
                final_curve=(x1, y1, x2, y2, x3, y3)



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
                cr.curve_to(x1, y1, x2, y2, x3, y3) 
                draw_dashed(8.0)


        cr.restore()

def _draw_polygon_to_layer(model, target_layer,nodes, color, gradient, bbox):
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
    tiles.update(layer.get_tile_coords())
    dstsurf = target_layer._surface
    for tx, ty in tiles:
        with dstsurf.tile_request(tx, ty, readonly=False) as dst:
            layer.composite_tile(dst, True, tx, ty, mipmap_level=0)

    bbox = tuple(target_layer.get_full_redraw_bbox())
    target_layer.root.layer_content_changed(target_layer, *bbox)
    del layer

## Class defs

class PolyFill(Command):
    """Polygon-fill on the current layer"""

    display_name = _("Polygon Fill")

    def __init__(self, doc, nodes, color, gradient,
                 bbox, **kwds):
        """
        :param bbox: boundary rectangle,in model coordinate.
        """
        super(PolyFill, self).__init__(doc, **kwds)
        self.nodes = nodes
        self.color = color
        self.gradient = gradient
        self.bbox = bbox
        self.snapshot = None

    def redo(self):
        # Pick a source
        layers = self.doc.layer_stack
        src_layer = layers.current
        assert self.snapshot is None
        self.snapshot = layers.current.save_snapshot()
        dst_layer = layers.current
        # Fill connected areas of the source into the destination
        _draw_polygon_to_layer(self.doc, dst_layer, self.nodes,
                self.color, self.gradient, self.bbox)

    def undo(self):
        layers = self.doc.layer_stack
        assert self.snapshot is not None
        layers.current.load_snapshot(self.snapshot)
        self.snapshot = None

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


    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):
        super(PolyfillMode, self).__init__(**kwargs)
        self._polygon_preview_fill = False



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
                overlay = self._ensure_overlay_for_tdw(tdw)
                hit_dist = gui.style.FLOATING_BUTTON_RADIUS

                if len(self.nodes) > 1:
                    button_info = [
                        (_EditZone_Bezier.ACCEPT_BUTTON, overlay.accept_button_pos),
                        (_EditZone_Bezier.REJECT_BUTTON, overlay.reject_button_pos),
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
                elif self.zone in (_EditZone_Bezier.ACCEPT_BUTTON,
                        _EditZone_Bezier.REJECT_BUTTON): 
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


    ## Redraws
    

    def redraw_curve_cb(self, erase=False):
        """ Frontend method,to redraw curve from outside this class"""
        pass # do nothing

    def _queue_redraw_curve(self, tdw=None):
        self._stop_task_queue_runner(complete=False)
        for tdw in self._overlays:
            
            if len(self.nodes) < 2:
                continue
        
            self._queue_task(
                    self._queue_polygon_area
            )
        self._start_task_queue_runner()

    def _queue_polygon_area(self):
        for tdw in self._overlays:
            
            if len(self.nodes) < 2:
                continue
    
            sdx, sdy = self.selection_rect.get_display_offset(tdw)
    
            sx, sy, ex, ey = self._get_maximum_rect(tdw, sdx, sdy)
            tdw.queue_draw_area(sx, sy, ex-sx+1, ey-sy+1)

    def _start_new_capture_phase_polyfill(self, tdw, rollback=False):
        if rollback:
            self._stop_task_queue_runner(complete=False)
        else:
            self._stop_task_queue_runner(complete=True)
            bbox = self._get_maximum_rect(None)
            if self.doc.model.layer_stack.current.get_fillable():
                cmd = PolyFill(self.doc.model,
                        self.nodes,
                        self.foreground_color,
                        None,bbox)
                self.doc.model.do(cmd)


            if not self._stroke_from_history:
                self.stroke_history.register(self.nodes)

        self._queue_draw_buttons()
        self._queue_redraw_all_nodes()
        self._queue_redraw_curve() 
        self._reset_nodes()
        self._reset_capture_data()
        self._reset_adjust_data()
        self.phase = _PhaseBezier.INITIAL
       #self._queue_redraw_curve() 

        self._stroke_from_history = False
        self.options_presenter.reset_stroke_history()

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
        if self.phase == _PhaseBezier.INITIAL: 
            self.phase = _PhaseBezier.CREATE_PATH
            # FALLTHRU: *do* start a drag 
        elif self.phase in (_PhaseBezier.CREATE_PATH,):
            # Initial state - everything starts here!
       
            if (self.zone in (_EditZone_Bezier.REJECT_BUTTON, 
                        _EditZone_Bezier.ACCEPT_BUTTON)):
                if (event.button == 1 and 
                        event.type == Gdk.EventType.BUTTON_PRESS):

                        # To avoid some of visual glitches,
                        # we need to process button here.
                        if self.zone == _EditZone_Bezier.REJECT_BUTTON:
                            self._start_new_capture_phase_polyfill(tdw, rollback=True)
                        elif self.zone == _EditZone_Bezier.ACCEPT_BUTTON:
                            self._start_new_capture_phase_polyfill(tdw, rollback=False)
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

                        if (event.state & Gdk.ModifierType.SHIFT_MASK):
                            # selection box dragging start!!
                            if self._returning_phase == None:
                                self._returning_phase = self.phase
                            self.phase = _PhaseBezier.ADJUST_SELECTING
                            self.selection_rect.start(
                                    *tdw.display_to_model(event.x, event.y))
                        elif (event.state & Gdk.ModifierType.CONTROL_MASK):
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
               #if (len(self.nodes) > 1 and node.curve == False):
               #    node.curve = True 
               #    node.curve = False

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

    ## Interrogating events

    

    ## Node editing

    @property
    def options_presenter(self):
        """MVP presenter object for the node editor panel"""
        cls = self.__class__
        if cls._OPTIONS_PRESENTER is None:
            cls._OPTIONS_PRESENTER = OptionsPresenter_Polyfill()
        return cls._OPTIONS_PRESENTER



                                                
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

        super(OverlayPolyfill, self).paint(cr)
                


class OptionsPresenter_Polyfill (OptionsPresenter_Bezier):
    """Presents UI for directly editing point values etc."""

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
        self._check_curvepoint= builder.get_object("checkbutton_curvepoint")
        self._check_curvepoint.set_sensitive(False)
        self._fill_polygon_checkbutton= builder.get_object("fill_checkbutton")
        self._fill_polygon_checkbutton.set_sensitive(True)

        

        combo = builder.get_object('path_history_combobox')
        combo.set_model(PolyfillMode.stroke_history.liststore)
        cell = Gtk.CellRendererText()
        combo.pack_start(cell,True)
        combo.add_attribute(cell,'text',0)
        self._stroke_history_combo = combo

        base_grid = builder.get_object("points_editing_grid")
        self._updating_ui = False

    @property
    def target(self):
        # this is exactly same as OptionsPresenter_Bezier,
        # but we need this to define @target.setter
        return super(OptionsPresenter_Bezier, self).target

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
                self._check_curvepoint.set_sensitive(True)
                self._check_curvepoint.set_active(cn.curve)
            else:
                self._check_curvepoint.set_sensitive(False)

            self._insert_button.set_sensitive(polyfillmode.can_insert_node(cn_idx))
            self._delete_button.set_sensitive(polyfillmode.can_delete_node(cn_idx))
            self._fill_polygon_checkbutton.set_active(polyfillmode.polygon_preview_fill)
        finally:
            self._updating_ui = False                               

    def _toggle_fill_checkbutton_cb(self, button):
        if self._updating_ui:
            return
        polymode, node_idx = self.target
        if polymode:
            polymode.polygon_preview_fill = button.get_active()



    ## Other handlers are as implemented in superclass.  
