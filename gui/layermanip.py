# This file is part of MyPaint.
# Copyright (C) 2014-2016 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or


"""Modes for moving layers around on the canvas"""

## Imports
from __future__ import division, print_function

from gettext import gettext as _
import weakref
import numpy as np

from gi.repository import Gdk
from gi.repository import GLib
from gi.repository import Gtk # XXX for `relative move`

import gui.mode
import lib.command
import gui.cursor
import gui.tileddrawwidget # XXX for `relative move`
import gui.overlays
import gui.linemode as linemode

## Class defs
# XXX for 'overlay move'
class _MoveType:
    SURFACE = 0 # Ordinary move
    OVERLAY = 1 # Using mimmap image of layer.

class _Prefs:
    MOVE_TYPE_PREF = 'LayerMoveMode.move_type'

    DEFAULT_MOVE_TYPE = _MoveType.SURFACE

# XXX for 'overlay move' end

class LayerMoveMode (gui.mode.ScrollableModeMixin,
                     gui.mode.DragMode,
                     gui.mode.OverlayMixin): # for `overlay move`
    """Moving a layer interactively

    MyPaint is tile-based, and tiles must align between layers.
    Therefore moving layers involves copying data around. This is slow
    for very large layers, so the work is broken into chunks and
    processed in the idle phase of the GUI for greater responsiveness.

    """

    ## API properties and informational methods

    ACTION_NAME = 'LayerMoveMode'
    _OPTIONS_PRESENTER = None
    _OVERLAY = None

    pointer_behavior = gui.mode.Behavior.CHANGE_VIEW
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW

    @classmethod
    def get_name(cls):
        return _(u"Move Layer")

    def get_usage(self):
        return _(u"Move the current layer")

    @property
    def active_cursor(self):
        cursor_name = gui.cursor.Name.HAND_CLOSED
        if not self._move_possible:
            cursor_name = gui.cursor.Name.FORBIDDEN_EVERYWHERE
        return self.doc.app.cursors.get_action_cursor(
            self.ACTION_NAME,
            cursor_name,
        )

    @property
    def inactive_cursor(self):
        cursor_name = gui.cursor.Name.HAND_OPEN
        if not self._move_possible:
            cursor_name = gui.cursor.Name.FORBIDDEN_EVERYWHERE
        return self.doc.app.cursors.get_action_cursor(
            self.ACTION_NAME,
            cursor_name,
        )

    permitted_switch_actions = set([
        'RotateViewMode',
        'ZoomViewMode',
        'PanViewMode',
    ] + gui.mode.BUTTON_BINDING_ACTIONS)
    
    # XXX for `relative move`
    def get_options_widget(self):
        cls = self.__class__
        if cls._OPTIONS_PRESENTER is None:
            cls._OPTIONS_PRESENTER = _OptionsPresenter()
        return cls._OPTIONS_PRESENTER    
    # XXX for `relative move` end
    
    ## Initialization

    def __init__(self, **kwds):
        super(LayerMoveMode, self).__init__(**kwds)
        self._cmd = None
        self._drag_update_idler_srcid = None
        self.final_modifiers = 0
        self._move_possible = False
        self._drag_active_tdw = None
        self._drag_active_model = None
        # XXX for `overlay move`
       #self.move_type = _MoveType.SURFACE 
        self.move_type = _MoveType.OVERLAY
        self.pixbuf = None
        self.cur_x = 0
        self.cur_y = 0
        self.cur_bbox = None
        self._overlays = {}  # keyed by tdw
        # XXX for `overlay move` end

    ## Layer stacking API

    def enter(self, doc, **kwds):
        super(LayerMoveMode, self).enter(doc, **kwds)
        self.final_modifiers = self.initial_modifiers
        rootstack = self.doc.model.layer_stack
        rootstack.current_path_updated += self._update_ui
        rootstack.layer_properties_changed += self._update_ui

        self._update_ui()

        opt = self.get_options_widget()
        assert opt is not None
        opt.target = self

    def leave(self, **kwds):
        if self._cmd is not None:
            while self._finalize_move_idler():
                pass
        rootstack = self.doc.model.layer_stack
        rootstack.current_path_updated -= self._update_ui
        rootstack.layer_properties_changed -= self._update_ui

        if not self._is_active():
            self._discard_overlays()

        opt = self.get_options_widget()
        assert opt is not None
        opt.target = None

        return super(LayerMoveMode, self).leave(**kwds)

    def _is_active(self):
        for mode in self.doc.modes:
            if mode is self:
                return True
        return False

    def checkpoint(self, **kwds):
        """Commits any pending work to the command stack"""
        if self._cmd is not None:
            while self._finalize_move_idler():
                pass
        return super(LayerMoveMode, self).checkpoint(**kwds)

    ## Drag-mode API

    def drag_start_cb(self, tdw, event):
        """Drag initialization"""
        if self._move_possible and self._cmd is None:
            model = tdw.doc
            layer_path = model.layer_stack.current_path
            x0, y0 = tdw.display_to_model(self.start_x, self.start_y)
            if self.move_type == _MoveType.SURFACE: # XXX for 'overlay move'
                cmd = lib.command.MoveLayer(model, layer_path, x0, y0)
                self._cmd = cmd
            # XXX for 'overlay move'
            elif self.move_type == _MoveType.OVERLAY:
                self._ensure_overlay_for_tdw(tdw)
                layer = model.layer_stack.current
                self.start_x = x0
                self.start_y = y0
                self.cur_x = 0
                self.cur_y = 0
                self.cur_bbox = layer.get_bbox()
                self.queue_redraw()
            else:
                assert False, "Should not here"
            # XXX for 'overlay move' end

            self._drag_active_tdw = tdw
            self._drag_active_model = model
        return super(LayerMoveMode, self).drag_start_cb(tdw, event)

    def drag_update_cb(self, tdw, event, dx, dy):
        """UI and model updates during a drag"""
        if self._cmd:
            if self.move_type == _MoveType.SURFACE: # XXX for 'overlay move'
                assert tdw is self._drag_active_tdw
                x, y = tdw.display_to_model(event.x, event.y)
                self._cmd.move_to(x, y)
                if self._drag_update_idler_srcid is None:
                    idler = self._drag_update_idler
                    self._drag_update_idler_srcid = GLib.idle_add(idler)
        # XXX for 'overlay move'
        elif self.move_type == _MoveType.OVERLAY:
            self._ensure_overlay_for_tdw(tdw)
            x, y = tdw.display_to_model(event.x, event.y)
            self.queue_redraw()
            self.cur_x = x - self.start_x
            self.cur_y = y - self.start_y
            self.queue_redraw()
        # XXX for 'overlay move' end

        return super(LayerMoveMode, self).drag_update_cb(tdw, event, dx, dy)

    def _drag_update_idler(self):
        """Processes tile moves in chunks as a background idler"""
        # Might have exited, in which case leave() will have cleaned up
        if self._cmd is None:
            self._drag_update_idler_srcid = None
            return False
        # Terminate if asked. Assume the asker will clean up.
        if self._drag_update_idler_srcid is None:
            return False
        # Process some tile moves, and carry on if there's more to do
        if self._cmd.process_move():
            return True
        self._drag_update_idler_srcid = None
        return False

    def drag_stop_cb(self, tdw):
        """UI and model updates at the end of a drag"""
        if self.move_type == _MoveType.SURFACE:
            # Stop the update idler running on its next scheduling
            self._drag_update_idler_srcid = None
            # This will leave a non-cleaned-up move if one is still active,
            # so finalize it in its own idle routine.
            if self._cmd is not None:
                assert tdw is self._drag_active_tdw
                # Arrange for the background work to be done, and look busy
                tdw.set_sensitive(False)

                window = tdw.get_window()
                cursor = Gdk.Cursor.new_for_display(
                    window.get_display(), Gdk.CursorType.WATCH)
                tdw.set_override_cursor(cursor)

                self.final_modifiers = self.current_modifiers()
                GLib.idle_add(self._finalize_move_idler)
            else:
                # Still need cleanup for tracking state, cursors etc.
                self._drag_cleanup()
        # XXX for 'overlay move'
        elif self.move_type == _MoveType.OVERLAY:
            self._ensure_overlay_for_tdw(tdw)
            self.queue_redraw() # This mignt no effect...?
            model = tdw.doc
            layer_path = model.layer_stack.current_path
            cmd = lib.command.MoveLayer(model, layer_path, self.start_x, self.start_y)
            self._cmd = cmd
            self._cmd.move_to(self.start_x+self.cur_x, self.start_y+self.cur_y)
            GLib.idle_add(self._finalize_move_idler)
        # XXX for 'overlay move' end

        return super(LayerMoveMode, self).drag_stop_cb(tdw)
        
    def _finalize_move_idler(self):
        """Finalizes everything in chunks once the drag's finished"""
        if self._cmd is None:
            return False  # something else cleaned up
        while self._cmd.process_move():
            return True
        model = self._drag_active_model
        cmd = self._cmd
        tdw = self._drag_active_tdw
        self._cmd = None
        self._drag_active_tdw = None
        self._drag_active_model = None
        self._update_ui()
        tdw.set_sensitive(True)
        model.do(cmd)
        self._drag_cleanup()
        return False

    ## Helpers
    def _update_ui(self, *_ignored):
        """Updates the cursor, and the internal move-possible flag"""
        layer = self.doc.model.layer_stack.current
        self._move_possible = (
            layer.visible
            and not layer.locked
            and layer.branch_visible
            and not layer.branch_locked
        )
        self.doc.tdw.set_override_cursor(self.inactive_cursor)

    def _drag_cleanup(self):
        """Final cleanup after any drag is complete"""
        if self._drag_active_tdw:
            self._update_ui()  # update may have been deferred
        self._drag_active_tdw = None
        self._drag_active_model = None
        self._cmd = None
        # XXX for 'overlay move'
        # IMPORTANT: place queue_draw here to erase the last overlay drawing.
        if self.move_type == _MoveType.OVERLAY:
            self.queue_redraw()
            self.cur_bbox = None
        # XXX for 'overlay move' end

        if not self.doc:
            return
        if self is self.doc.modes.top:
            if self.initial_modifiers:
                if (self.final_modifiers & self.initial_modifiers) == 0:
                    self.doc.modes.pop()

    # XXX for 'overlay move'
    def get_overlay_for_mode(self, tdw):
        cls = self.__class__
        if cls._OVERLAY is None:
            cls._OVERLAY = _Overlay_Move()
        cls._OVERLAY.set_target(self, tdw)
        return cls._OVERLAY

    def _ensure_overlay_for_tdw(self, tdw):
        overlay = self._overlays.get(tdw)
        if not overlay:
            overlay = self.get_overlay_for_mode(tdw)
            tdw.display_overlays.append(overlay)
            self._overlays[tdw] = overlay
        return overlay

    def _discard_overlays(self):
        for tdw, overlay in self._overlays.items():
            tdw.display_overlays.remove(overlay)
            tdw.queue_draw()
        self._overlays.clear()

    def _generate_border(self, tdw, dx, dy):
        x, y, w, h = self.cur_bbox
        x += dx
        y += dy
        return (tdw.model_to_display(x, y),
                tdw.model_to_display(x+w, y),
                tdw.model_to_display(x+w, y+h),
                tdw.model_to_display(x, y+h))

    def _queue_box(self, tdw, x, y, w, m=2):
        """Queue redraw with square box centered at x,y
        """
        w += m
        tdw.queue_draw_area(x-w, y-w, w*2, w*2)

    def _queue_line(self, tdw, start, end, step=6):
        xs, ys = start
        xe, ye = end
        length, nx, ny = linemode.length_and_normal(xs, ys, xe, ye)
        segment = length / step
        half_seg = segment / 2.0
        for i in range(step):
            cl = (segment * i) + half_seg
            cx, cy = linemode.multiply(nx, ny, cl)
            self._queue_box(tdw, cx+xs, cy+ys, segment)

    def queue_redraw(self):
        if self.cur_bbox is not None:
            # To minimize redrawing area. 
            # If just redraw maximum rectangle,
            # when moving large layer, almost entire canvas would be redrawn.
            # 
            # Target rect is consisted from box and diagonal crossing lines
            # so only update around them.
            offsets = ((0,0), (self.cur_x, self.cur_y))
            for tdw, overlay in self._overlays.items():
                for x, y in offsets:
                    tl, tr, br, bl = self._generate_border(tdw, x, y)
                    self._queue_line(tdw, tl, tr)
                    self._queue_line(tdw, tr, br)
                    self._queue_line(tdw, br, bl)
                    self._queue_line(tdw, bl, tl)

                    self._queue_line(tdw, tl, br)
                    self._queue_line(tdw, tr, bl)
    # XXX for 'overlay move' end

# XXX for 'overlay move'
class _Overlay_Move(gui.overlays.Overlay):

   #def __init__(self, movemode, tdw):
   #    super(_Overlay_Move, self).__init__()
   #    self._mode = weakref.proxy(movemode)
   #    self._tdw = weakref.proxy(tdw)

    def __init__(self):
        super(_Overlay_Move, self).__init__()

    def set_target(self, mode, tdw):
        self._mode = weakref.proxy(mode)
        self._tdw = weakref.proxy(tdw)

    def draw_target_rectangle(self, cr, tdw, x, y, rgba):
        bx, by, bw, bh = self._mode.cur_bbox
        bx += x
        by += y
        bx1, by1 = tdw.model_to_display(bx, by)
        bx2, by2 = tdw.model_to_display(bx+bw, by)
        bx3, by3 = tdw.model_to_display(bx+bw, by+bh)
        bx4, by4 = tdw.model_to_display(bx, by+bh)

        # We cannot see target rect when the canvas is filled by (nearly)same color 
        # with target rect. so `dash` with distinguishable color.
        cr.set_source_rgba(0, 0, 0, 0.7)
        cr.set_line_width(1)
        for i in (None, 1):
            if i is not None:
                cr.set_dash((4.0,),)
            cr.new_path()
            cr.move_to(bx1, by1)
            cr.line_to(bx2, by2)
            cr.line_to(bx3, by3)
            cr.line_to(bx4, by4)
            cr.close_path()
           #if i > 1:
           #    gui.drawutils.render_drop_shadow(cr) # not good looking...
            cr.stroke()

            cr.move_to(bx1, by1)
            cr.line_to(bx3, by3)
            cr.stroke()

            cr.move_to(bx2, by2)
            cr.line_to(bx4, by4)
            cr.stroke()
            cr.set_source_rgba(*rgba)


    def paint(self, cr):
        """Draw brush size to the screen"""
        mode = self._mode
        tdw = self._tdw
        if mode and mode.cur_bbox is not None:
            cr.save()
            a = 1.0

            # Drawing current bbox of layer.
            self.draw_target_rectangle(
                cr, tdw, 0, 0, 
                (0.0, 1.0, 0.0, a)
            )

            # Drawing `moved` bbox of layer.
            self.draw_target_rectangle(
                cr, tdw, mode.cur_x, mode.cur_y,
                (1.0, 0.0, 0.0, a)
            )
            cr.restore()
# XXX for 'overlay move' end

# XXX for `relative move`
class _OptionsPresenter(Gtk.Grid):
    """Configuration widget for the LayerMove tool"""

    _SPACING = 6
    _LABEL_MARGIN_LEFT = 32

    def __init__(self):
        self._cmd = None
        
        # XXX Code duplication: from closefill.py
        Gtk.Grid.__init__(self)
        self._update_ui = True
        self.set_row_spacing(self._SPACING)
        self.set_column_spacing(self._SPACING)
        from application import get_app
        self.app = get_app()
        self._mode_ref = None
        prefs = self.app.preferences
        row = 0

        def generate_label(text, tooltip, row, grid, alignment, margin_left):
            # Generate a label, and return it.
            label = Gtk.Label()
            label.set_markup(text)
            label.set_tooltip_text(tooltip)
            label.set_alignment(*alignment)
            label.set_hexpand(False)
            label.set_margin_start(margin_left)
            grid.attach(label, 0, row, 1, 1)
            return label
            
        def generate_spinbtn(row, grid, extreme_value):
            # Generate a Adjustment and spinbutton
            # and return Adjustment.
            adj = Gtk.Adjustment(
                value=0, 
                lower=-extreme_value,
                upper=extreme_value,
                step_increment=1, page_increment=1,
                page_size=0
            )
            spinbtn = Gtk.SpinButton()
            spinbtn.set_hexpand(True)
            spinbtn.set_adjustment(adj)

            # We need spinbutton focus event callback, to disable/re-enable
            # Keyboard manager for them.
            # Without this, we lose keyboard input focus right after 
            # input only one digit(key). 
            # It is very annoying behavior.
            spinbtn.connect("focus-in-event", self._spin_focus_in_cb)
            spinbtn.connect("focus-out-event", self._spin_focus_out_cb)
            grid.attach(spinbtn, 1, row, 1, 1)
            return adj
            
        frame = Gtk.Frame()
        frame.set_label(_("Relative offset :"))
        frame.set_shadow_type(Gtk.ShadowType.NONE)       
        subgrid = Gtk.Grid()
        subgrid.set_margin_top(self._SPACING)
        subgrid.set_row_spacing(self._SPACING)
        subgrid.set_column_spacing(self._SPACING)
        frame.add(subgrid)
        self.attach(frame, 0 ,row, 1, 1)

        # Minimum/Maximum of relative offset is currently 
        # 2^16, This would be enough in most case.
        extreme_value = 2**16
                
        subrow = 0
        label = generate_label(
            _("X:"),
            _("The relative offset x position of current layer"),
            subrow,
            subgrid,
            (1.0, 0.1),
            self._LABEL_MARGIN_LEFT
        )
        self._offset_x_adj = generate_spinbtn(subrow, subgrid, extreme_value)
        
        subrow += 1
        label = generate_label(
            _("Y:"),
            _("The relative offset y position of current layer"),
            subrow,
            subgrid,
            (1.0, 0.1),
            self._LABEL_MARGIN_LEFT
        )
        adj = Gtk.Adjustment(
            value=0, 
            lower=-extreme_value,
            upper=extreme_value,
            step_increment=1, page_increment=1,
            page_size=0
        )
        self._offset_y_adj = generate_spinbtn(subrow, subgrid, extreme_value)
        
        subrow += 1
        btn = Gtk.Button()
        btn.set_label(_("Move current layer"))
        btn.connect("clicked", self._move_button_clicked_cb)
        subgrid.attach(btn, 0, subrow, 2, 1)    

        # XXX for 'overlay move'
        # Move preview method
        subrow += 1
        text = _("Show only target rect")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            _("Show only target rectangle when dragging, to improve response.")
        )
        t = prefs.get(_Prefs.MOVE_TYPE_PREF, _Prefs.DEFAULT_MOVE_TYPE)
        if t == _MoveType.OVERLAY:
            checkbut.set_active(True)
        checkbut.connect("toggled", self._show_only_rect_toggled_cb)
        self.attach(checkbut, 0, subrow, 2, 1)
        # XXX for 'overlay move' end

        self._update_ui = False

    @property
    def target(self):
        if mode is not None:
            return self._mode_ref()
        else:
            return None

    @target.setter
    def target(self, mode):
        if mode is not None:
            self._mode_ref = weakref.ref(mode)
        else:
            self._mode_ref = None
        
    def _spin_focus_in_cb(self, widget, event):
        if self._update_ui:
            return
        kbm = self.app.kbm
        kbm.enabled = False

    def _spin_focus_out_cb(self, widget, event):
        kbm = self.app.kbm
        kbm.enabled = True        
        
    def _move_button_clicked_cb(self, btn):
        # Exit when the move already ongoing.
        if self._cmd is not None:
            return
            
        app = self.app
        model = app.doc.model
        layer_path = model.layer_stack.current_path
        assert layer_path is not None 
        cmd = lib.command.MoveLayer(model, layer_path, 0, 0)
        cmd.move_to(
            self._offset_x_adj.get_value(),
            self._offset_y_adj.get_value()
        )
        self._cmd = cmd      
        for tdw in gui.tileddrawwidget.TiledDrawWidget.get_visible_tdws():
            tdw.set_sensitive(False)
        GLib.idle_add(self._wait_move_complete)

    # XXX for 'overlay move'
    def _show_only_rect_toggled_cb(self, btn):
        mode = self.target
        if mode is not None:
            if btn.get_active():
                mode.move_type = _MoveType.OVERLAY
            else:
                mode.move_type = _MoveType.SURFACE
            self.app.preferences[_Prefs.MOVE_TYPE_PREF] = mode.move_type
    # XXX for 'overlay move' end
        
    def _wait_move_complete(self):
        """Nearly same as LayerMoveMode._finalize_move_idler 
        """
        cmd = self._cmd
        if cmd is None:
            return False  # something else cleaned up        
        while cmd.process_move():
            return True
        app = self.app
        model = app.doc.model            
        model.do(cmd)
        self._cmd = None
        self._offset_x_adj.set_value(0)
        self._offset_y_adj.set_value(0)        
        for tdw in gui.tileddrawwidget.TiledDrawWidget.get_visible_tdws():
            tdw.set_sensitive(True)        
        return False
# XXX for `relative move` end
