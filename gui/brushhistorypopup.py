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
from gi.repository import GdkPixbuf
import cairo

from lib import helpers
from lib.observable import event
import windowing
import quickchoice
import history

"""Brush History popup."""

## Class definitions

class BrushHistoryPopup (windowing.PopupWindow):
    """Brush History popup
    """
    
    def __init__(self, app, prefs_id=quickchoice._DEFAULT_PREFS_ID):
        super(BrushHistoryPopup, self).__init__(app)
        # FIXME: This duplicates stuff from the PopupWindow
        #self.set_position(Gtk.WindowPosition.MOUSE)
        self.app = app
        self.app.kbm.add_window(self)
        
        vbox = Gtk.VBox()
        brush_hist_view = history.BrushHistoryView(app)
        vbox.pack_start(brush_hist_view, True, False, 0)
        self.add(vbox)
        
        brush_hist_view.button_clicked += self._button_clicked_cb   
        
        icon_size = history.HISTORY_PREVIEW_SIZE 
        margin = 8      
        self.set_size_request(
            (icon_size + margin) * 5 + margin * 2,
            icon_size + margin * 2
        )

    def _button_clicked_cb(self, history_view):
        self.leave("clicked")
     
    def popup(self):       
        self.enter()

    def enter(self):
        # Popup this window, in current position.
        x, y = self.get_position()
        self.move(x, y)
        self.show_all()
               
    def leave(self, reason):
        self.hide()

    def advance(self):
        """Currently,nothing to do."""
        self.leave("advanced")
        pass

    def backward(self):
        """Currently,nothing to do."""
        self.leave("backward")
        pass

