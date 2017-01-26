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
from gui.beziertool import _Control_Handle, _Node_Bezier
from lib.command import Command
import lib.surface
import lib.mypaintlib
import lib.tiledsurface
from gui.oncanvas import *
import gui.gradient
from gui.polyfillshape import *
from gui.polyfillshape import _Phase, _EditZone
import gui.dialogs

## Module constants

POLYFILLMODES = (
    (lib.mypaintlib.CombineNormal, _("Normal")),
    (lib.mypaintlib.CombineDestinationOut, _("Erase")),
    (lib.mypaintlib.CombineDestinationIn, _("Erase Outside")),
    (lib.mypaintlib.CombineSourceAtop, _("Clipped")),
    )

## Enum defs

class _ActionButton(ActionButtonMixin):
    FILL_ATOP = 201
    ERASE = 202
    ERASE_OUTSIDE = 203

# Other Enum defs (_EditZone and _Phase) are moved to gui/polyfillshape.py

## Module funcs

def _render_polygon_to_layer(model, target_layer, shape, nodes, 
        color, gradient, bbox, mode=lib.mypaintlib.CombineNormal):
    """
    :param bbox: boundary box rectangle, in model coordinate
    :param mode: polygon drawing mode(enum _POLYDRAWMODE). 
                 * NOT layer composite mode*
    """
    sx, sy, w, h = bbox
    w = int(w)
    h = int(h)

    # convert to adapt library
    sx = int(sx)
    sy = int(sy)

    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    cr = cairo.Context(surf)

    shape.draw_node_polygon(cr, None, nodes, ox=sx, oy=sy,
            color=color, gradient=gradient)
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

# Shape classes defined at polyfillshape.py

class PolyFill(Command):
    """Polygon-fill command, on the current layer"""

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




## The Polyfill Mode Class

class PolyfillMode (OncanvasEditMixin,
        HandleNodeUserMixin):
    """ Polygon fill mode

    This class can handle multiple types of shape,
    such as polygon, curved polygon, rectangle and ellipse.

    In order to deal with such a shape, 
    Polyfillmode holds the dedicated proxy class (defined above)
    therein.
    """

    ## Metadata properties
    ACTION_NAME = "PolyfillMode"

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

    _enable_switch_actions=set(gui.mode.BUTTON_BINDING_ACTIONS).union([
            'RotateViewMode',
            'ZoomViewMode',
            'PanViewMode',
            "SelectionMode",
        ])

    ## Other class vars

   #button_info = _ButtonInfo()       # button infomation class
    _shape_pool = { Shape.TYPE_BEZIER : Shape_Bezier() ,
                    Shape.TYPE_POLYLINE : Shape_Polyline() ,
                    Shape.TYPE_RECTANGLE : Shape_Rectangle() ,
                    Shape.TYPE_ELLIPSE : Shape_Ellipse() }
    _shape = None

    _gradient_ctrl = None

    buttons = {
            _ActionButton.ACCEPT : ('mypaint-ok-symbolic', 
                'accept_button_cb'),
            _ActionButton.REJECT : ('mypaint-trash-symbolic', 
                'reject_button_cb'), 
            _ActionButton.ERASE : ('mypaint-eraser-symbolic', 
                'erase_button_cb'),
            _ActionButton.ERASE_OUTSIDE : ('mypaint-cut-symbolic', 
                'erase_outside_button_cb'),
            _ActionButton.FILL_ATOP : ('mypaint-add-symbolic', 
                'fill_atop_button_cb')
            }


    BUTTON_OPERATIONS = {
        _ActionButton.ACCEPT:lib.mypaintlib.CombineNormal,
        _ActionButton.REJECT:None,
        _ActionButton.ERASE:lib.mypaintlib.CombineDestinationOut,
        _ActionButton.ERASE_OUTSIDE:lib.mypaintlib.CombineDestinationIn,
        _ActionButton.FILL_ATOP:lib.mypaintlib.CombineSourceAtop,
        }

    ## Initialization & lifecycle methods

    def __init__(self, **kwargs):
        super(PolyfillMode, self).__init__(**kwargs)
        self._polygon_preview_fill = False
        self.options_presenter.target = (self, None)
        self.phase = _Phase.ADJUST
        if PolyfillMode._shape == None:
            self.shape_type = Shape.TYPE_BEZIER
        self.forced_button_pos = None


    def _reset_capture_data(self):
        super(PolyfillMode, self)._reset_capture_data()
        self.phase = _Phase.ADJUST
        pass

    def _reset_adjust_data(self):
        super(PolyfillMode, self)._reset_adjust_data()
        self.current_node_handle = None
        self._stroke_from_history = False

    def can_delete_node(self, idx):
        """ differed from InkingMode,
        BezierMode can delete the last node.
        """ 
        return 1 <= idx < len(self.nodes)

    ## Inner states related methods
    def _generate_overlay(self, tdw):
        return OverlayPolyfill(self, tdw)

    def _generate_presenter(self):
        return OptionsPresenter_Polyfill()
        

    def _search_target_node(self, tdw, x, y, margin=12):
        shape = self._shape
        assert shape != None

        if shape.accept_handle:
            return HandleNodeUserMixin._search_target_node(self,
                    tdw, x, y, margin)
        else:
            return NodeUserMixin._search_target_node(self,
                    tdw, x, y, margin)


    def _update_zone_and_target(self, tdw, x, y):

        if self.gradient_ctrl.active:
            new_zone = self.zone
            idx = self.gradient_ctrl.hittest_node(tdw, x, y)
            if idx >= -1:
                new_zone = _EditZone.GRADIENT_BAR
                self._enter_new_zone(tdw, new_zone)
                return

        super(PolyfillMode, self)._update_zone_and_target(tdw, x, y)

    def update_cursor_cb(self, tdw): 
        # Update the "real" inactive cursor too:
        # these codes also a little changed from inktool.
        cursor = None
        if self.phase in (_Phase.ADJUST,
                _Phase.ADJUST_POS):
            if self.zone == _EditZone.CONTROL_NODE:
                cursor = self._crosshair_cursor
            elif self.zone == _EditZone.ACTION_BUTTON:
                cursor = self._crosshair_cursor
            else:
                cursor = self._arrow_cursor

        return cursor

    def is_adjusting_phase(self):
        """To know whether current phase is node adjusting phase.
        this method should be overriden in deriving classes.
        """
        return self.phase in (_Phase.ADJUST,
                              _Phase.ADJUST_POS,
                              _Phase.ADJUST_HANDLE)

    def enter(self, doc, **kwds):
        super(PolyfillMode, self).enter(doc, **kwds)

        # OneshotdragMode might override this mode,
        # so avoid double-registration to view_changed_observers
        # (Although, append() method might reject it...)
        if not self.view_changed_cb in doc.view_changed_observers:
            doc.view_changed_observers.append(self.view_changed_cb)

            # So, this is not Onshotdragmode overriding...
            # we move gradient controller, if it is away
            # from visual range.
            gctl = self.gradient_ctrl
            if gctl.active:
                for tdw in self._overlays:
                    if gctl.start_pos and gctl.end_pos:
                        self.call_gradient_controller_center(tdw, visible_check=True)
        
            color_manager = doc.app.brush_color_manager
            if not self.brush_color_changed_cb in color_manager.color_updated:
                color_manager.color_updated += self.brush_color_changed_cb
                        
    def leave(self, **kwds):
        if self._is_active() == False:
            # Mode leave is not temporary.
           #if self.gradient_ctrl.active:
           #    self.gradient_ctrl.active = False
            for tdw in self._overlays:
                self.gradient_ctrl.queue_redraw(tdw)

            self.doc.view_changed_observers.remove(self.view_changed_cb)
            color_manager = self.doc.app.brush_color_manager
            color_manager.color_updated -= self.brush_color_changed_cb

        super(PolyfillMode, self).leave(**kwds)

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

    @property
    def gradient_ctrl(self):
        cls = self.__class__
        if cls._gradient_ctrl == None:
            cls._gradient_ctrl = gui.gradient.GradientController(self.doc.app)
        return cls._gradient_ctrl

    ## Redraws
    

    def redraw_item_cb(self, erase=False):
        """ Frontend method,to redraw curve from outside this class"""
        pass # do nothing for now

    def _queue_redraw_item(self, tdw=None):

        for tdw in self._overlays:
            
            if len(self.nodes) < 2:
                continue

            sdx, sdy = self.drag_offset.get_display_offset(tdw)
            tdw.queue_draw_area(
                    *self._shape.get_maximum_rect(tdw, self, sdx, sdy))

    def _queue_draw_node(self, i, offsets=None, tdws=None):
        """This method might called from baseclass,
        so we need to call HandleNodeUserMixin method explicitly.
        """
        return self._queue_draw_handle_node(i, offsets, tdws)


    def is_drawn_handle(self, i, hi):
        return self._shape.accept_handle

    ## Drawing Phase related

    def _start_new_capture_phase(self, composite_mode, rollback=False):
        self._queue_draw_buttons()
        self._queue_redraw_all_nodes()
        self._queue_redraw_item()

        if rollback:
            self._stop_task_queue_runner(complete=False)
        else:
            self._stop_task_queue_runner(complete=True)
            self._draw_polygon(composite_mode)

        self._reset_capture_data()
        self._reset_adjust_data()
        self.forced_button_pos = None


    def checkpoint(self, flush=True, **kwargs):
        """Sync pending changes from (and to) the model
        When this mode is left for another mode (see `leave()`), the
        pending brushwork is committed properly.
    
        """
        if flush:
            pass
        else:
            # Queue a re-rendering with any new brush data
            # No supercall
            self._stop_task_queue_runner(complete=False)
            self._queue_draw_buttons()
            self._queue_redraw_all_nodes()
            self._queue_redraw_item()


    ### Event handling

    ## Raw event handling (prelight & zone selection in adjust phase)
    def mode_button_press_cb(self, tdw, event):
        shape = self._shape
        shape.update_event(event)

        if self.phase == _Phase.CAPTURE:
            self.phase = _Phase.ADJUST

        # common processing for all shape type.
        if self.phase == _Phase.ADJUST:
            # Initial state - everything starts here!
       
            if self.zone == _EditZone.EMPTY_CANVAS:
                
                if self.phase == _Phase.ADJUST:
                    if (len(self.nodes) > 0): 
                       #if shift_state and ctrl_state:
                        if shape.alt_state:
                            self._queue_draw_buttons() 
                            self.forced_button_pos = (event.x, event.y)
                            self.phase = _Phase.CHANGE_PHASE 
                            self._returning_phase = _Phase.ADJUST
                            self._queue_draw_buttons() 
                            return 

            elif self.zone == _EditZone.GRADIENT_BAR:
                gctl = self.gradient_ctrl
                assert gctl.active
                self.phase = _Phase.GRADIENT_CTRL
                gctl.button_press_cb(self, tdw, event)
                return False
        
        ret = shape.button_press_cb(self, tdw, event)
        if ret == Shape.CANCEL_EVENT:
            return True

    def mode_button_release_cb(self, tdw, event):
        shape = self._shape

        if self.phase == _Phase.GRADIENT_CTRL:
            gctl = self.gradient_ctrl
            assert gctl.active
            gctl.button_release_cb(self, tdw, event)
            return False

        ret = shape.button_release_cb(self, tdw, event)

        if ret == Shape.CANCEL_EVENT:
            return True

    def motion_notify_cb(self, tdw, event):
        """ Override raw handler, to ignore handle according to
        whether current shape supports it or not.
        """
        self._ensure_overlay_for_tdw(tdw)

        current_layer = tdw.doc._layers.current
        if not (tdw.is_sensitive and current_layer.get_paintable()):
            return False

        shape = self._shape

        self._update_zone_and_target(tdw, event.x, event.y)
               #,ignore_handle = not shape.accept_handle)

        return super(OncanvasEditMixin, self).motion_notify_cb(tdw, event)
        

    ## Drag handling (both capture and adjust phases)
    def node_drag_start_cb(self, tdw, event):

        self._queue_draw_buttons() # To erase button,and avoid glitch
        if self.phase == _Phase.GRADIENT_CTRL:
            gctl = self.gradient_ctrl
            gctl.queue_redraw(tdw)
            assert gctl.active
            gctl.drag_start_cb(self, tdw, event)
            self._queue_draw_buttons()
            return False


        shape = self._shape

        ret = shape.drag_start_cb(self, tdw, event)
        if ret == Shape.CANCEL_EVENT:
            return True

    def node_drag_update_cb(self, tdw, event, dx, dy):
        if self.phase == _Phase.GRADIENT_CTRL:
            gctl = self.gradient_ctrl
            gctl.queue_redraw(tdw)
            assert gctl.active
            gctl.drag_update_cb(self, tdw, event, dx, dy)
            gctl.queue_redraw(tdw)
            return False

        shape = self._shape
        ret = shape.drag_update_cb(self, tdw, event, dx, dy)
        if ret == Shape.CANCEL_EVENT:
            return True
        

    def node_drag_stop_cb(self, tdw):
        if self.phase == _Phase.GRADIENT_CTRL:
            gctl = self.gradient_ctrl
            gctl.queue_redraw(tdw)
            assert gctl.active
            gctl.drag_stop_cb(self, tdw)
            gctl.queue_redraw(tdw)
            self.phase = _Phase.ADJUST
            self._queue_draw_buttons()
            return False

        shape = self._shape
        try:

            ret = shape.drag_stop_cb(self, tdw)
            if ret == Shape.CANCEL_EVENT:
                return False

            # Common processing
            if self.current_node_index != None:
                self.options_presenter.target = (self, self.current_node_index)

        finally:
            shape.clear_event()

    def view_changed_cb(self, doc):
        """Observer handler for view changing.
        """
        if self.gradient_ctrl.active:
            self.gradient_ctrl.invalidate_cairo_gradient()
            for tdw in self._overlays:
                self.gradient_ctrl.queue_redraw(tdw)

    def brush_color_changed_cb(self, manager):
        if self.gradient_ctrl.active:
            if self.gradient_ctrl.follow_brushcolor():
                for tdw in self._overlays:
                    self.gradient_ctrl.queue_redraw(tdw)


    ## Interrogating events
    def _get_event_data(self, tdw, event):
        """ Overriding mixin method.
        
        almost same as inktool,but we needs generate _Node_Bezier object
        not _Node object
        """
        x, y = tdw.display_to_model(event.x, event.y)
        xtilt, ytilt = self._get_event_tilt(tdw, event)
        # Using _Node_Bezier, ignoring pressure & dtime
        return _Node_Bezier(
            x=x, y=y,
            pressure=1.0,
            xtilt=xtilt, ytilt=ytilt,
            dtime=0.0
            )

    
    ## Interface methods which call from callbacks
    def execute_draw_polygon(self, action_name): 
        """Draw polygon interface method"""

        if action_name == "PolygonFill":
            composite_mode = lib.mypaintlib.CombineNormal
        elif action_name == "PolygonFillAtop":
            composite_mode = lib.mypaintlib.CombineSourceAtop
        elif action_name == "PolygonErase":
            composite_mode = lib.mypaintlib.CombineDestinationOut
        elif action_name == "PolygonEraseOutside":
            composite_mode = lib.mypaintlib.CombineDestinationIn
        else:
            return

        self._start_new_capture_phase(composite_mode, rollback=False)

    def _draw_polygon(self, composite_mode):
        """Draw polygon (inner method)
        """

        if self.doc.model.layer_stack.current.get_fillable():
                    
            bbox = self._shape.get_maximum_rect(None, self)
            gradient = None
            if self.gradient_ctrl.active:
                gradient = self.gradient_ctrl.generate_gradient(None,
                        offset_x=-bbox[0], offset_y=-bbox[1])

            cmd = PolyFill(
                    self.doc.model,
                    self.shape,
                    self.nodes,
                    self.foreground_color,
                    gradient,
                    bbox,
                    composite_mode)
            self.doc.model.do(cmd)

            if not self._stroke_from_history:
                self.stroke_history.register(self.nodes)
                self.options_presenter.reset_stroke_history()

        else:
            logger.debug("Polyfilltool: target is not fillable layer.nothing done.")

       #self._reset_all_internal_state()


    ## properties

    @property
    def polygon_preview_fill(self):
        return self._polygon_preview_fill

    @polygon_preview_fill.setter
    def polygon_preview_fill(self, flag):
        self._polygon_preview_fill = flag
        self._queue_redraw_item()
    
    ## Action button handlers
    def _do_action(self, key):
        if (self.phase in (_Phase.ADJUST, _Phase.ACTION) and
                len(self.nodes) > 1):
            self._start_new_capture_phase(
                self.BUTTON_OPERATIONS[_ActionButton.ACCEPT],
                rollback=False)

    def accept_button_cb(self, tdw):
        self._do_action(_ActionButton.ACCEPT)

    def reject_button_cb(self, tdw):
        if (self.phase in (_Phase.ADJUST, _Phase.ACTION)):
            self._start_new_capture_phase(
                None, rollback=True)

    def erase_button_cb(self, tdw):
        self._do_action(_ActionButton.ERASE)

    def erase_outside_button_cb(self, tdw):
        self._do_action(_ActionButton.ERASE_OUTSIDE)

    def fill_atop_button_cb(self, tdw):
        self._do_action(_ActionButton.FILL_ATOP)

    # Gradient Controller related
    def enable_gradient_controller(self, gradient_data, 
                                   tdw=None, toggle=False):
        """
        :param gradient_data: a tuple of (linear_pos, RGBtuple).
                              if this is None, 
                              gradient controller would be disabled.
        :param toggle: toggle controller enable state, if True.
        """
        gctl = self.gradient_ctrl

        # tdw initialization and erasing queue_redraw()
        # should be placed prior to actual gradient controller
        # manipulation codes.
        if tdw == None:
            if len(self._overlays) > 0:
                tdw = self._overlays.keys()[0]

        gctl.queue_redraw(tdw)# to erase current controller

        if gradient_data == None: 
            gctl.active = False
        else:
            if toggle:
                gctl.active = not gctl.active
            else:
                gctl.active = True

        if gctl.active:

            # Re-check, not else.because tdw can be set
            # in above lines.
            # But we might still be missing tdw.
            if tdw:
                if gctl.start_pos == None:
                    ta = tdw.get_allocation()
                    height = ta.height / 2.0
                    x = (ta.width - gctl._radius) / 2.0 
                    sy = height - height / 2.0
                    ey = height + height / 2.0
                    gctl.set_start_pos(tdw, (x, sy))
                    gctl.set_end_pos(tdw, (x, ey))

                gctl.setup_gradient(gradient_data)

        gctl.queue_redraw(tdw)

    def call_gradient_controller_center(self, tdw, visible_check=False):
        """ place gradient controller into
        center of display.
        :param visible_check: if True, move controller only when it is away from
                              visible display area.
        """
        ta = tdw.get_allocation()
        gctl = self.gradient_ctrl
        sx, sy = tdw.model_to_display(*gctl.start_pos)
        ex, ey = tdw.model_to_display(*gctl.end_pos)

        if visible_check:

            if ((ta.x <= sx <= ta.x + ta.width 
                 and ta.y <= sy <= ta.y + ta.height) 
                or
                (ta.x <= ex <= ta.x + ta.width 
                 and ta.y <= ey <= ta.y + ta.height)):
                # the gradient controller is visible:
                # (at least a part of it)
                return

        l, nx, ny = gui.linemode.length_and_normal(
                sx, sy, ex, ey)
        cx = ta.x + ta.width / 2
        cy = ta.y + ta.height / 2

        sx = cx - (nx * l * 0.5)
        sy = cy - (ny * l * 0.5)
        ex = cx + (nx * l * 0.5)
        ey = cy + (ny * l * 0.5)
        gctl.set_start_pos(tdw, (sx, sy))
        gctl.set_end_pos(tdw, (ex, ey))
        gctl.queue_redraw(tdw)



class OverlayPolyfill (OverlayBezier):
    """Overlay for an BezierMode's adjustable points"""

    BUTTON_PALETTE_RADIUS = 64


    def __init__(self, mode, tdw):
        super(OverlayPolyfill, self).__init__(mode, tdw)
        self._draw_initial_handle_both = True
        

    def update_button_positions(self):
        mode = self._mode
        if mode.forced_button_pos:
            pos = gui.ui_utils.setup_round_position(
                    self._tdw, mode.forced_button_pos,
                    len(mode.buttons),
                    self.BUTTON_PALETTE_RADIUS), 
            for i, key in enumerate(mode.buttons):
                id, junk = mode.buttons[i]
                self._button_pos[id] = pos[i]
        else:
            super(OverlayPolyfill, self).update_button_positions(
                    not mode.shape.accept_handle)
            # Disable extended buttons by assigning None.
            # It is not sufficient to capture the Keyerror exception 
            # at enumeration loop. Because, the position has been 
            # recorded when you expand a button even once. 
            self._button_pos[_ActionButton.ERASE] = None
            self._button_pos[_ActionButton.ERASE_OUTSIDE] = None
            self._button_pos[_ActionButton.FILL_ATOP] = None

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
        if hasattr(shape, "paint_nodes"):
            shape.paint_nodes(cr, self._tdw, mode,
                    gui.style.DRAGGABLE_POINT_HANDLE_SIZE)
        else:
            super(OverlayPolyfill, self).paint(cr, draw_buttons=False)
                
        if (not mode.in_drag and len(mode.nodes) > 1):
            self._draw_mode_buttons(cr)

        if mode.gradient_ctrl.active:
            mode.gradient_ctrl.paint(cr, mode, self._tdw)

class GradientStore(object):

    # default gradents.
    # a gradient consists from ( 'name', (color sequence), id )
    #
    # 'color sequence' is a sequence,which consists from color step
    # (position, (R, G, B)) or (position, (R, G, B, Alpha)) or
    # (position, (Special_number, Alpha)).
    # Special number is , -1 means current selected brush color.
    #
    # if you use RGBA format,every color sequence must have alpha value.
    #
    # 'id' is integer, this should be unique, to distinguish cache.
    # 'id' is generated at runtime.
    
    IDX_NAME = 0
    IDX_COLORS = 1
    IDX_ID = 2

    # Gradients data format is defined in gui/gradient.py
    DEFAULT_GRADIENTS = [ 
            (
                'Disabled', 
                None,
            ),
            (
                'Foreground to Transparent', 
                (
                    (0.0, (-1, 1.0)), (1.0, (-1, 0.0))
                ),
            ),
           #(
           #    'Foreground to Background', 
           #    (
           #        (0.0, (-1,)) , (1.0, (-2,))
           #    ),
           #),
            (
                'Rainbow', 
                (
                    (0.0, (1.0, 0.0, 0.0)) , 
                    (0.25, (1.0, 1.0, 0.0)) , 
                    (0.5, (0.0, 1.0, 0.0)) , 
                    (0.75, (0.0, 1.0, 1.0)) , 
                    (1.0, (0.0, 0.0, 1.0))  
                ),
            ),
        ]

    def __init__(self): 
        self._store = Gtk.ListStore(str,object,int)
        self._id_seed = 0
        for name, colors in self.DEFAULT_GRADIENTS:
            self.register_gradient(name, colors)

    def register_gradient(self, name, colors):
        id = self._id_seed 
        self._store.append((name, colors, id))
        self._id_seed = id + 1

    def remove_gradient(self, target_name=None, target_id=None, iter=None):
        """Remove a gradient from store information,
        by either name or id or iter.
        """
        if iter != None:
            self._store.remove(iter)
            return True

        iter = self._store.get_iter_first()
        while iter != None:
            name, colors, id = self._store[iter]
            if target_name:
                if target_name == name:
                    self._store.remove(iter)
                    return True
            elif id:
                if target_id == id:
                    self._store.remove(iter)
                    return True
            iter = self._store.iter_next(iter)

        return False

    @property
    def liststore(self):
        return self._store

    def get_gradient_data(self, iter):
        """Get a sequance of gradient data easily.
        """
        return self._store[iter][self.IDX_COLORS]

    def get_cairo_gradient(self, iter, sx, sy, ex, ey, fg, bg):
        """Get cairo gradient object, not the sequence data.
        This is for Optionspresenter.
        """
        data = self.get_gradient_data(iter)
        if data:
            return self.get_cairo_gradient_raw(
                    data,
                    sx, sy,
                    ex, ey,
                    fg, bg)

    def get_cairo_gradient_raw(self, graddata, sx, sy, ex, ey, fg, bg):
        cg = None
        for i, info in enumerate(graddata):
            pos, color = info
            a = 1.0

            if len(color) == 4:
                r, g, b, a = color
            elif len(color) == 3:
                r, g, b = color
            elif len(color) <= 2:
                # Use foreground/ background color
                # The second value, if exist, it is alpha value.
                if color[0] == -1:
                    r, g, b = fg
                elif color[0] == -2:
                    r, g, b = bg
                else:
                    logger.warning("invalid color in graddata")

                if len(color) == 2:
                    a = color[1]

            if cg == None:
                cg=cairo.LinearGradient(sx, sy, ex, ey)

            cg.add_color_stop_rgba(pos, r, g, b, a)

        return cg



class GradientRenderer(Gtk.CellRenderer):
    colors = GObject.property(type=GObject.TYPE_PYOBJECT, default=None)
    id = GObject.property(type=int , default=-1)

    SAMPLE_WIDTH = 96
    SAMPLE_HEIGHT = 20

    def __init__(self, gradient_store):
        super(GradientRenderer, self).__init__()
        self.gradient_store = gradient_store
        self.cg={}

    def _init_surfaces(self):
        def create_surf_and_ctx():
            surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 
                    self.SAMPLE_WIDTH, self.SAMPLE_HEIGHT)
            cr = cairo.Context(surf)
            return (surf, cr)
    
        bgsurf, cr = create_surf_and_ctx()
        self.draw_background(cr, area)
        self._background = bgsurf

        # gradient simulator, for 'foreground to transparent'
        bgsurf, cr = create_surf_and_ctx()
        self.draw_background(cr, area)
        self._background = bgsurf

    def do_set_property(self, pspec, value):
        setattr(self, pspec.name, value)

    def do_get_property(self, pspec):
        return getattr(self, pspec.name)

    def do_render(self, cr, widget, background_area, cell_area, flags):
        """
        :param cell_area: RectangleInt class
        """
        grad = None
        if self.colors != None:
            # first colorstep has alpha = it uses alpha value = need background 
            if len(self.colors[0][1]) > 3:
                self.draw_background(cr, cell_area)

            grad = self.get_gradient(self.id, self.colors, cell_area)

        # All drawing codes (especially cr.set_source())
        # must placed after calling self.draw_background().
        # Because cr.save()/restore() does not restore cairo source
        # actually.
        cr.save()
        cr.translate(cell_area.x, cell_area.y)
        cr.rectangle(0, 0,
                cell_area.width, cell_area.height)
        if grad == None:
            cr.set_source_rgb(0.5, 0.5, 0.5) # invalid color
        else:
            cr.set_source(grad)
        cr.fill()
        cr.restore()
        # selected = (flags & Gtk.CellRendererState.SELECTED) != 0
        # prelit = (flags & Gtk.CellRendererState.PRELIT) != 0

    def get_gradient(self, id, colors, cell_area):
        if not id in self.cg:
            halftop = cell_area.height/2
            cg = self.gradient_store.get_cairo_gradient_raw(
                colors,
                0, halftop, 
                cell_area.width - 1, halftop,
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

               #cr.rectangle(cell_area.x + w, cell_area.y + h, 
               #        tile_size, tile_size)
                cr.rectangle(w, h, 
                        tile_size, tile_size)
                cr.set_source_rgb(*tilecolor[idx%2])
                cr.fill()
                idx+=1
                w += tile_size
            h += tile_size
            idx += 1
        cr.restore()

    def do_get_preferred_width(self, view_widget):
        return (self.SAMPLE_WIDTH, self.SAMPLE_HEIGHT)

    def refresh_gradient_sample(self, id):
        if id in self.cg:
            del self.cg[id]
            # By deleting cached data,
            # automatically generate new gradient 
            # in next time rendering.


class OptionsPresenter_Polyfill (OptionsPresenter_Bezier):
    """Presents UI for directly editing point values etc."""

    GRADIENT_STORE = GradientStore()

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

        self._enable_ctrl_button = builder.get_object(
                "activate_controller_togglebutton")

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


        # Creating gradient sample and its popup menu
        store = self.GRADIENT_STORE.liststore

        treeview = Gtk.TreeView()
        treeview.set_size_request(175, 125)
        treeview.set_model(store)
        col = Gtk.TreeViewColumn(_('Name'), cell, text=0)
        treeview.append_column(col)

        cell = GradientRenderer(self.GRADIENT_STORE)
        col = Gtk.TreeViewColumn(
                _('Gradient'), 
                cell, 
                colors=GradientStore.IDX_COLORS, 
                id=GradientStore.IDX_ID)
        self._gradient_renderer = cell

        selection = treeview.get_selection()
        selection.connect('changed', self.gradientview_selection_changed_cb)
        self._gradient_selection = selection
        

        popup = builder.get_object("gradient_popupmenu")
        popup.show_all()
        self._popup_gradient = popup
        self._remove_gradient_menu = builder.get_object(
                'remove_gradient_menuitem')
        self._rename_gradient_menu = builder.get_object(
                'rename_gradient_menuitem')
        self._replace_gradient_menu = builder.get_object(
                'replace_gradient_menuitem')

        # button-release event for treeview is to invoke popup menu.
        treeview.connect('button-release-event',
                self.gradientview_button_release_cb)


        # and it is appended to the treeview
        treeview.append_column(col)
        treeview.set_hexpand(True)

        # adding expander for treeview
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
                    self._shape_type_combobox.get_active() != Shape.TYPE_BEZIER)
                self._check_curvepoint.set_active(cn.curve)
            else:
                self._check_curvepoint.set_sensitive(False)

            self._insert_button.set_sensitive(polyfillmode.can_insert_node(cn_idx))
            self._delete_button.set_sensitive(polyfillmode.can_delete_node(cn_idx))
            self._fill_polygon_checkbutton.set_active(polyfillmode.polygon_preview_fill)
        finally:
            self._updating_ui = False                               

    ## Utility method
    def get_current_gradient_data(self, iter=None):
        if iter == None:
            store, iter = self._gradient_selection.get_selected()

        # Re-check iter
        if iter:
            return self.GRADIENT_STORE.get_gradient_data(iter)
        else:
            return None

    def ask_newlayer_name(self):
        return gui.dialogs.ask_for_name(
                    self._gradientview,
                    _("Input name for new gradient"),
                    _("new gradient")
                    )

    def get_color_array_from_controller(self, gradient_ctrl):
        """Utility method, to get color array from
        current controller setting.
        """
        colors = []
        for cn in gradient_ctrl.nodes:
            colors.append(
                            (cn.linear_pos,
                             cn.get_rgba())
                         )
        return colors

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

    def activate_controller_togglebutton_toggled_cb(self, button):
        if self._updating_ui:
            return
        polymode, junk = self.target
        if polymode:
            if button.get_active():
                data = self.get_current_gradient_data()
            else:
                data = None
            polymode.enable_gradient_controller(data)

    def gradientview_selection_changed_cb(self, selection):
        if self._updating_ui:
            return
        polymode, junk = self.target
        store, iter = selection.get_selected()
        if iter and polymode:
            # In this situation, the actual selection
            # change might not be refrected to 
            # self._gradient_selection.It might still show
            # old selection.
            # so, we cannot share this with
            # activate_controller_togglebutton_toggled_cb(),
            # just toggling that button does not work well.
            data = self.GRADIENT_STORE.get_gradient_data(iter)
            polymode.enable_gradient_controller(data)
            
            self._updating_ui = True
            self._enable_ctrl_button.set_active(True)
            self._updating_ui = False

    def gradientview_button_release_cb(self, iconview, event):
        if (self._updating_ui or 
                event.button != Gdk.BUTTON_SECONDARY):
            return
        polymode, junk = self.target
        store, iter = self._gradient_selection.get_selected()
        if iter and polymode:
            # Pre-popup sensitive modify,
            # to avoid deleting 'Disabled' item.
            flag = (store[iter][GradientStore.IDX_ID] != 0)
            self._replace_gradient_menu.set_sensitive(flag)
            self._rename_gradient_menu.set_sensitive(flag)
            self._remove_gradient_menu.set_sensitive(flag)

            self._popup_gradient.popup(None, None, None, None,
                    event.button, event.time)

    def add_gradient_menuitem_activate_cb(self, menuitem):
        polymode, junk = self.target
        if polymode:
            name = self.ask_newlayer_name()
            colors = self.get_color_array_from_controller(
                    polymode.gradient_ctrl)
           #colors = []
           #for cn in polymode.gradient_ctrl.nodes:
           #    colors.append(
           #                    (cn.linear_pos,
           #                     cn.get_rgba())
           #                 )

            self.GRADIENT_STORE.register_gradient(
                    name,
                    colors)

    def remove_gradient_menuitem_activate_cb(self, menuitem):     
        store, iter = self._gradient_selection.get_selected()
        if iter != None:
            self.GRADIENT_STORE.remove_gradient(iter=iter)

    def new_gradient_menuitem_activate_cb(self, menuitem):
        polymode, junk = self.target
        if polymode:
            name = self.ask_newlayer_name()
            brushcolor = self._app.brush_color_manager.get_color().get_rgb()
            colors = ( 
                        (1.0, brushcolor), 
                        (1.0, brushcolor)
                     )
            self.GRADIENT_STORE.register_gradient(
                    name,
                    colors)

    def rename_gradient_menuitem_activate_cb(self, menuitem):
        store, iter = self._gradient_selection.get_selected()
        if iter != None:
            oldname = store[iter][GradientStore.IDX_NAME]
            name = gui.dialogs.ask_for_name(
                        self._gradientview,
                        _("Input new name for exisitng gradient"),
                        oldname
                        )
            if name:
                store[iter][GradientStore.IDX_NAME] = name

    def replace_gradient_menuitem_activate_cb(self, menuitem):
        store, iter = self._gradient_selection.get_selected()
        polymode, junk = self.target
        if polymode and iter != None:
            oldname = store[iter][GradientStore.IDX_NAME]
            name = gui.dialogs.ask_for_name(
                        self._gradientview,
                        _("Input new name for gradient to be replaced"),
                        oldname
                        )
            if name:
                store[iter][GradientStore.IDX_NAME] = name
                colors = self.get_color_array_from_controller(
                        polymode.gradient_ctrl)
                store[iter][GradientStore.IDX_COLORS] = colors
                self._gradient_renderer.refresh_gradient_cache(
                    store[iter][GradientStore.IDX_ID]
                    )

   #def mode_leave_notify_cb(self, mode):
   #    """ Actually, this is not GTK signal.
   #    this is called from PolyfillMode.leave()
   #    """
   #    self._updating_ui = True
   #   #self._enable_ctrl_button.set_active(False)
   #    self._updating_ui = False

   #def mode_enter_notify_cb(self, mode):
   #    """ Actually, this is not GTK signal.
   #    this is called from PolyfillMode.enter()
   #    """
   #    self._updating_ui = True
   #    # To follow actual state of gradient controller
   #    self._enable_ctrl_button.set_active(
   #            mode.gradient_ctrl.active:
   #            )
   #    self._updating_ui = False


    ## Other handlers are as implemented in superclass.  
