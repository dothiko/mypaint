# This file is part of MyPaint.
# Copyright (C) 2017 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from gi.repository import Gtk

import widgets
import windowing
import quickchoice
import history
from lib.observable import event
from gettext import gettext as _

"""Brush History popup."""

## Class definitions

class BlendButtonView (Gtk.Grid):
    """A set of clickable images showing the blending modes.
    Based on BrushHistoryView of gui/history.py"""

    def __init__(self, app):
        Gtk.Grid.__init__(self)
        self._app = app
        s = history.HISTORY_PREVIEW_SIZE
        self.set_border_width(widgets.SPACING)
        self.set_hexpand(True)
        self.set_halign(Gtk.Align.FILL)
        self._buttons = []
        
        actions = (
            app.find_action("BlendModeNormal"),        
            app.find_action("BlendModeEraser"),
            app.find_action("BlendModeLockAlpha"),
            app.find_action("BlendModeColorize")        
        )
        self._actions = actions
        
        for i, act in enumerate(actions):
            button = widgets.borderless_button(
                icon_name = act.get_icon_name(),
                size = s
            )
            button.connect("clicked", self._blendbutton_clicked_cb, act)
            button.set_hexpand(True)
            button.set_halign(Gtk.Align.FILL)
            self.attach(button, i, 0, 1, 1)
            self._buttons.append(button)

    def _blendbutton_clicked_cb(self, button, act):
        if not act.get_active():
            act.activate()
            self._app.show_transient_message(
                _("Set brush blending mode as %s." % act.get_label())
            )
        self.button_clicked()
    
    @event
    def button_clicked(self):
        """Event: a color history button was clicked"""

class BrushHistoryPopup (windowing.PopupWindow):
    """Brush History popup, to quick access recent brushes on canvas.
    """
    
    def __init__(self, app, prefs_id=quickchoice._DEFAULT_PREFS_ID):
        super(BrushHistoryPopup, self).__init__(app)
        vbox = Gtk.VBox()
       #vbox = Gtk.Grid()
       #vbox.set_border_width(widgets.SPACING_LOOSE)
       #vbox.set_column_spacing(widgets.SPACING)
       #vbox.set_row_spacing(widgets.SPACING)        
        
        brush_hist_view = history.BrushHistoryView(app)
        vbox.pack_start(brush_hist_view, True, False, 0)
       #vbox.attach(brush_hist_view, 0, 0, 1, 1)
        
        blend_method_view = BlendButtonView(app)
        vbox.pack_end(blend_method_view, True, True, 0)
        #vbox.attach(blend_method_view, 0, 1, 1, 1)
        
                
        self.add(vbox)
        brush_hist_view.button_clicked += self._button_clicked_cb   
        self._hist_view = brush_hist_view        

        blend_method_view.button_clicked += self._button_clicked_cb   
        self._blend_view = blend_method_view 
        
        bm = app.brushmanager
        icon_size = history.HISTORY_PREVIEW_SIZE 
        margin = 8      
        self.set_size_request(
            (icon_size + margin) * len(bm.history) + margin * 2,
            icon_size + margin * 2
        )

    def _button_clicked_cb(self, history_view):
        if history_view is self._hist_view:
            app = self.app
            brushinfo = app.brush
            app.show_transient_message(
                _("Set brush as %s." % brushinfo.get_string_property('parent_brush_name'))
            )
        self.leave("clicked")
     
    def popup(self):
        self.enter()
        self._autoleave_start(True) # XXX for `autoleave`

    def enter(self):
        # Popup this window, in current position.
        x, y = self.get_position()
        self.move(x, y)
        self.show_all()
               
    def leave(self, reason):
        self._autoleave_start(False) # XXX for `autoleave`
        self.hide()

    def advance(self):
        """Currently,nothing to do."""
        self.leave("advanced")
        pass

    def backward(self):
        """Currently,nothing to do."""
        self.leave("backward")
        pass

