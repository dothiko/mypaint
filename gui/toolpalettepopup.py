# This file is part of MyPaint.
# Copyright (C) 2017 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GLib
import cairo
import math

from lib import helpers
import windowing
import drawutils
import quickchoice
import gui.mode


"""Brush Size change popup."""


## Module constants
ICON_SIZE = 32
ICON_RADIUS = ICON_SIZE / 2
LABEL_SPACE = 10
LABEL_HEIGHT = 30
FONT_SIZE = 16
TIMEOUT_LEAVE = int(0.7 * 1000) # Auto timeout, when once change size.

class _Zone:
    INVALID = 0
    CIRCLE = 1
    BUTTON = 2

## Class definitions

class ToolPalettePopup (windowing.PopupWindow):
    """ToolPalette Popup

    This window is normally popup when hover the cursor
    over a layer in layerlist.
    """

    outside_popup_timeout = 0
    _canvas_size = None
    buttons = None

    def __init__(self, app, prefs_id=quickchoice._DEFAULT_PREFS_ID):
        super(ToolPalettePopup, self).__init__(app)
        # FIXME: This duplicates stuff from the PopupWindow
        self.set_position(Gtk.WindowPosition.MOUSE)
        self.app = app
        self.app.kbm.add_window(self)
        self.set_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                        Gdk.EventMask.BUTTON_RELEASE_MASK |
                        Gdk.EventMask.ENTER_NOTIFY_MASK |
                        Gdk.EventMask.LEAVE_NOTIFY_MASK |
                        Gdk.EventMask.POINTER_MOTION_MASK 
                        )
        self.connect("button-release-event", self.button_release_cb)
        self.connect("button-press-event", self.button_press_cb)
        self.connect("leave-notify-event", self.popup_leave_cb)
        self.connect("enter-notify-event", self.popup_enter_cb)
        self.connect("motion-notify-event", self.motion_cb)

        self.connect("draw", self.draw_cb)


        self._button = None
        self._close_timer_id = None
        self._zone = _Zone.INVALID
        self._sub_radius = None
        self._current_button_index = -1

        # Transparent window supports
        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual != None and screen.is_composited():
            self.set_visual(visual)

        self.set_app_paintable(True)
        self.set_decorated(False)
        self.bgpix = None

        # Initialize buttons
        cls = self.__class__
        if cls.buttons == None:
            buttons = []
            registered = []
            for cs in gui.mode.ModeRegistry.mode_classes:
                action_name = None
                if (issubclass(cs, gui.mode.BrushworkModeMixin)
                        or issubclass(cs, gui.mode.SingleClickMode)):
                    # The class which has cs.ACTION_NAME is
                    # not baseclass/mixin, so validate it.
                    action_name = cs.ACTION_NAME
                else:
                    continue

                # I don't know exactly what is happening,
                # but some `valid` action might be enumrated multiple times.
                # To reject them, use `registered` list.
                if (action_name is not None
                        and not action_name in registered):
                    # XXX Should we just use unmodified(not `Flip`) action name?
                    action = app.find_action("Flip%s" % action_name)
                    icon_name = action.get_icon_name()
                    if icon_name == None:
                        icon_name = 'mypaint-ok-symbolic'
                    pixbuf = gui.drawutils.load_symbolic_icon(
                        icon_name=icon_name,
                        size=gui.style.FLOATING_BUTTON_ICON_SIZE,
                        fg=(0, 0, 0, 1),
                    )
                    buttons.append((pixbuf, cs, action))
                    registered.append(action_name)

            # Sort buttons, to make it unique order.
            # Without this, button order might be changed everytime
            # we start mypaint.
            def _sort_buttons(a, b):
                aname = a[1].ACTION_NAME
                bname = b[1].ACTION_NAME
                if aname < bname:
                    return -1
                elif aname == bname:
                    return 0
                else:
                    return 1

            cls.buttons = sorted(buttons, _sort_buttons)

        # Button setup completed. now we can get self.canvas_size
        # and set size_request.
        
        canvas_size = self.canvas_size
        width = canvas_size
        height = canvas_size + LABEL_HEIGHT + LABEL_SPACE * 2
        self.set_size_request(width, height)

    def update_zone(self, x, y):
        """ Get zone information, to know where user clicked.
        """
        w = self.canvas_size
        x -= w/2
        y -= w/2
        dist = math.hypot(x, y)
        sub_r = self.sub_radius
        r = self.radius

        old_zone = self._zone
        old_btn_idx = self._current_button_index

        zone = _Zone.INVALID

        self._current_button_index = -1

        if dist >= sub_r and dist <= r:
            zone = _Zone.CIRCLE

            for i in xrange(len(self.buttons)):
                dx, dy = self._get_button_pos(i)
                if math.hypot(dx - x, dy - y) < ICON_RADIUS:
                    self._current_button_index = i
                    zone = _Zone.BUTTON
                    break

        self._zone = zone
        if (self._zone != old_zone or
                old_btn_idx != self._current_button_index):
            self.queue_redraw(self)

    def capture_screen(self):
        """Capture background screen.
        This method is used when desktop compositor disabled.
        """
        x, y = self.get_position()
        w, h = self.get_size()
        win = Gdk.get_default_root_window()
        self.bgpix = Gdk.pixbuf_get_from_window(win, x, y, w, h)

    def popup(self):
        self.enter()

    def enter(self):
        # Popup this window, in current position.
        x, y = self.get_position()
        self.move(x, y)
        self.show_all()

        window = self.get_window()
        cursor = Gdk.Cursor.new_for_display(
            window.get_display(), Gdk.CursorType.ARROW)
        window.set_cursor(cursor)
        self._leave_cancel = True

    def leave(self, reason):
        self.hide()
        self._close_timer_id = None
        self._button = None
        return False # To stop timer, as timer callback

    def button_press_cb(self, widget, event):
        self._button = event.button
        self.update_zone(event.x, event.y)
        if event.button == 1:

            if self._zone == _Zone.BUTTON:
                # Nothing to be done here.
                # `Changing tool` should be done
                # at button_release_cb, not here.
                # If we change tool here,
                # we messed up the tool cursor
                # after leave this popup.
                pass
            elif self._zone == _Zone.INVALID:
                self.leave("aborted")
                
        self.queue_redraw(widget)

    def button_release_cb(self, widget, event):
        if self._button != None:
            self.update_zone(event.x, event.y)
            if self._zone == _Zone.BUTTON:
                idx = self._current_button_index
                assert idx >= 0
                assert idx < len(self.buttons)
                junk, cls , action = self.buttons[idx]
                doc = self.app.doc
                doc.mode_flip_action_activated_cb(action)
                self.leave("clicked")
            else:
                if self._close_timer_id:
                    GLib.source_remove(self._close_timer_id)
                self._close_timer_id = GLib.timeout_add(
                        TIMEOUT_LEAVE,
                        self.leave,
                        'timer')
                        
        self._button = None

    def popup_enter_cb(self, widget, event):
        if self._leave_cancel:
            self._leave_cancel = False
            
    def popup_leave_cb(self, widget, event):
        if not self._leave_cancel:
            self.leave('outside')

    def motion_cb(self, widget, event):
        self.update_zone(event.x, event.y)

    def queue_redraw(self, widget):
        a = widget.get_allocation()
        t = widget.queue_draw_area(a.x, a.y, a.width, a.height)       

    def _draw_button(self, cr, btninfo, x, y, active):
        radius = ICON_SIZE / 2
        if active:
            color = gui.style.ACTIVE_ITEM_COLOR
        else:
            color = gui.style.EDITABLE_ITEM_COLOR

        icon_pixbuf, junk, junk = btninfo
        gui.drawutils.render_round_floating_button(
            cr=cr, x=x, y=y,
            color=color,
            pixbuf=icon_pixbuf,
            radius=radius,
        )

    def _draw_label(self, cr, idx): 
        if idx == -1:
            return
        junk, cs, act = self.buttons[idx]
        canvsize = self.canvas_size
        # We can draw much better text with pango
        # such as
        # gui.overlay.ScaleOverlay.paint_frame()
        cr.save()
        cr.set_source_rgba(0.0, 0.0, 0.0, 0.6)
        cr.translate(
            canvsize / 2, 
            canvsize + LABEL_SPACE + LABEL_HEIGHT / 2
        )
        gui.overlays.rounded_box(
            cr,
            -canvsize/2, -LABEL_HEIGHT / 2,
            canvsize,
            LABEL_HEIGHT,
            6
        )
        cr.fill()

        cr.set_source_rgba(1, 1, 1, 1)
        cr.set_font_size(FONT_SIZE)
        txt = cs.get_name()
        x_bear, y_bear, width, height, x_adv, y_adv = cr.text_extents(txt)
        cr.move_to(-width/2, height/2)
        cr.show_text(txt)
        cr.restore()

    def draw_cb(self, widget, cr):
        if not self.is_composited and self.bgpix is None:
            self.capture_screen() # test
            assert(self.bgpix is not None)

        cr.save() # Saving before transparent state
        cr.set_operator(cairo.OPERATOR_SOURCE)
        # Drawing background for transparent.
        if self.bgpix is None:
            cr.set_source_rgba(1.0, 1.0, 1.0, 0.0)
            cr.paint()
        else:
            Gdk.cairo_set_source_pixbuf(cr, self.bgpix, 0, 0)
            cr.paint()

        w = self.canvas_size
        r = self.radius
        sub_r = self.sub_radius

        cr.translate(w/2, w/2)

        # Drawing background of presets
        cr.save()
        cr.set_source_rgba(1.0, 1.0, 1.0, 0.6)
        cr.arc(0, 0, r, 0, math.pi * 2)
        cr.fill()
        # Erase center
        cr.set_source_rgba(1.0, 1.0, 1.0, 0.0)
        cr.arc(0, 0, sub_r, 0, math.pi * 2)
        cr.fill()
        cr.restore()

        cr.restore() # Restoring from transparent state

        cr.save()
        cr.translate(w/2, w/2)
        btncnt = len(self.buttons)
        for i, cbtn in enumerate(self.buttons):
            dx, dy = self._get_button_pos(i)
            self._draw_button(
                cr, cbtn, 
                dx, dy, 
                i == self._current_button_index
            )
        cr.restore()

        self._draw_label(cr, self._current_button_index)
        
        return True

    def advance(self):
        """Currently,nothing to do."""
        pass

    def backward(self):
        """Currently,nothing to do."""
        pass

    @property
    def radius(self):
        return self.canvas_size / 2

    @property
    def sub_radius(self):
        if self._sub_radius == None:
            adj = 1.4
            self._sub_radius = int(self.radius - ICON_SIZE * adj)
        return self._sub_radius


    def _get_button_pos(self, i):
        """Get button position, from origin (i.e. from x,y = 0,0).
        """
        btn_rad = (math.pi / len(self.buttons)) * 2.0 * i
        sub_r = self.sub_radius # Center radius of buttons.
        adj = 1.4
        sub_r += (ICON_SIZE * adj) / 2
        dx = - sub_r * math.sin(btn_rad)
        dy = sub_r * math.cos(btn_rad)
        return (dx, dy)

    def get_button_idx(self, x, y):
        """ Get Button index from the (cursor) position.
        """
        for i in xrange(len(self.buttons)):
            dx, dy = self._get_button_pos(i)
            if math.hypot(dx - x, dy - y) < ICON_RADIUS:
                return i
        return -1

    @property
    def canvas_size(self):
        """CAUTION: Use this after buttons setup!!
        """
        cls = self.__class__
        if cls._canvas_size == None:
            # We can get suitable canvas size from circumference.
            canvas_size = ((len(self.buttons) * (ICON_SIZE * 1.1)) / math.pi) * 2
            cls._canvas_size = canvas_size
        return self._canvas_size

