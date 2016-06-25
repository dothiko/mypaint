# This file is part of MyPaint.
# Copyright (C) 2016 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from gi.repository import Gtk
from gi.repository import Gdk
import cairo

from lib import helpers
import windowing
import drawutils


"""Layer preview popup."""


## Module constants

FONT_SIZE = 10
MARGIN = 5
TITLE_SPACE = FONT_SIZE + (MARGIN * 2)
POPUP_WIDTH = 128
POPUP_HEIGHT = POPUP_WIDTH + TITLE_SPACE

## Class definitions

class PreviewPopup (windowing.PopupWindow):
    """Layer preview popup window.

    This window is normally popup when hover the cursor
    over a layer in layerlist.
    """

    outside_popup_timeout = 0

    def __init__(self, app):
        super(PreviewPopup, self).__init__(app)
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
        self.connect("leave-notify-event", self.leave_cb)
        self.connect("motion-notify-event", self.motion_cb)

        self.connect("draw", self.draw_cb)

        self.set_size_request(POPUP_WIDTH, POPUP_HEIGHT)
        self._cairo_surface = None
        self._pixbuf = None

    @property
    def active(self):
        return self._pixbuf != None


    def enter(self, layer):
        self._target_layer = layer
        assert layer != None
        # popup placement
        x, y = self.get_position()
        self.move(x, y)
        self.show_all()

        window = self.get_window()
        cursor = Gdk.Cursor.new_for_display(
            window.get_display(), Gdk.CursorType.CROSSHAIR)
        window.set_cursor(cursor)
        bbox = layer.get_bbox()
        if bbox.w > 0 and bbox.h > 0:
            self._pixbuf = layer.render_as_pixbuf(*bbox, alpha=True)
            self._layer_name = layer.name

    def leave(self, reason):
        if self.active:
           #del self._cairo_surface
            self._cairo_surface = None
            self._pixbuf = None
        self.hide()

    def button_press_cb(self, widget, event):
        pass

    def button_release_cb(self, widget, event):
        pass

    def leave_cb(self, widget, event):
        self.leave('outside')

    def motion_cb(self, widget, event):
        self.leave('motion')


    def draw_cb(self, widget, cr):
        canvas_w = POPUP_WIDTH
        canvas_h = POPUP_WIDTH

        # Drawing background check
        check_size = canvas_w / 5
        cr.save()
        cr.translate(0.0, TITLE_SPACE)
        drawutils.render_checks(cr, check_size, 
                int(canvas_w / check_size))
        cr.restore()

        if self._pixbuf:

            # Drawing the layer into the center of canvas
            cr.save()
            pw = float(self._pixbuf.get_width())
            ph = float(self._pixbuf.get_height())
            hw = float(canvas_w) / 2.0
            hh = float(canvas_h) / 2.0
            cr.translate(hw, hh + TITLE_SPACE)

            aspect = pw / ph
            if aspect > 1.0:
                ratio = float(canvas_w) / pw
            elif aspect < 1.0:
                ratio = float(canvas_h) / ph

            cr.scale(ratio, ratio)
            px = -pw / 2.0
            py = -ph / 2.0
            Gdk.cairo_set_source_pixbuf(cr, self._pixbuf, px , py)
            cr.rectangle(px, py, pw, ph)
            cr.fill()

            cr.restore()

        # Drawing layer name
        cr.set_font_size(FONT_SIZE)
        cr.set_source_rgb(0.0, 0.0, 0.0)
        cr.rectangle(0, 0, canvas_w, TITLE_SPACE)
        cr.fill()

        cr.rectangle(MARGIN, MARGIN, 
                canvas_w - MARGIN*2, 
                TITLE_SPACE - MARGIN*2)
        cr.clip()

        cr.new_path()
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.move_to(MARGIN, FONT_SIZE + MARGIN);
        cr.show_text(self._layer_name)
        return True
