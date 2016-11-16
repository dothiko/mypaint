#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Stamp editor"""


## Imports
from __future__ import print_function

import os
import logging
logger = logging.getLogger(__name__)

from lib.gettext import C_
from gettext import gettext as _
import gi
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import Pango
from gi.repository import GLib
from gi.repository import GdkPixbuf
from gi.repository import GObject
from lib import brushsettings

import gui.stamps
from windowing import SubWindow

## Class definitions

class StampStore(object):
    """Stamp Store used from stamp listview"""

    def __init__(self, stamp):
        self._stamp = stamp
        self._store = Gtk.ListStore(object, str, str)
        for idx in stamp.get_valid_tiles():
            surface = stamp.get_current_src(idx)
            size = "%d x %d" % (surface.get_width(), surface.get_height())
            srcdesc = stamp.get_desc(idx)
            self._store.append((surface, size, srcdesc))

    @property
    def liststore(self):
        return self._store

    def get_surface(self, iter):
        return self._store[iter]

class StampRenderer(Gtk.CellRenderer):
    """Stamp renderer used from stamp listview"""

    surface = GObject.property(type=GObject.TYPE_PYOBJECT, default=None)
   #id = GObject.property(type=int , default=-1)
   #desc = GObject.property(type=str, default="")

    def __init__(self, store):
        super(StampRenderer, self).__init__()
        self._store = store

    def do_set_property(self, pspec, value):
        setattr(self, pspec.name, value)

    def do_get_property(self, pspec):
        return getattr(self, pspec.name)

    def do_render(self, cr, widget, background_area, cell_area, flags):
        """
        :param cell_area: RectangleInt class
        """
        cr.save()
        cr.translate(cell_area.x, cell_area.y)
        surf = self.surface

        sw = float(surf.get_width())
        sh = float(surf.get_height())
        vw = float(cell_area.width) 
        vh = float(cell_area.height) 
        hw = vw / 2.0
        hh = vh / 2.0
        cr.translate(hw, hh)

        aspect = vw / vh
        if aspect >= 1.0:
            ratio = vh / sh
        else:
            ratio = vw / sw

        cr.scale(ratio, ratio)
        sx = -sw / 2.0
        sy = -sh / 2.0
        cr.set_source_surface(surf, sx , sy)
        cr.rectangle(sx, sy, sw, sh)

       #cr.rectangle(0, 0,
       #        cell_area.width, cell_area.height)
       #
       #cr.set_source_surface(self.surface)
        cr.fill()
        cr.restore()

   #def do_get_preferred_height(self,view_widget):
   #    print('preferred!')
   #    return (80, 80) 
    def do_get_preferred_width(self,view_widget):
        return (96, 96) 
    
   #def do_get_preferred_height_for_width(self, width):
   #    print('preferred! w')
   #    return (80, 80) 
   #
   #def do_get_preferred_width_for_height(self, width, height):
   #    print('preferred! h')
   #    return (128, 128) 

    def do_get_size(self, view_widget, cell_area):
        if cell_area != None:
            print(cell_area)
        return (0, 0, 128, 80)

class StampEditorWindow (SubWindow):
    """Window containing the stamp editor"""

    ## Class constants

    _UI_DEFINITION_FILE = "stampeditor.glade"
    _LISTVIEW_THUMBNAIL_COLUMN = 0
    _LISTVIEW_DISPLAYNAME_COLUMN = 1

    _LISTVIEW_COLUMN_HEIGHT = 80

    ## Construction

    def __init__(self, app, stamp, stamp_manager):
        self.manager = stamp_manager
        self._stamp = stamp

        SubWindow.__init__(self, app, key_input=True)

        self.set_title(C_(
            "Stamp settings editor: subwindow title",
            "Stamp Settings Editor",
        ))
        self._setting = None
        self._builder = Gtk.Builder()
        self._builder.set_translation_domain("mypaint")
        self._build_ui()
        self.connect_after("show", self._post_show_cb)
        editor = self._builder.get_object("stamp_editor")
        self.add(editor)
       #self._live_update_idle_cb_id = None


    def _build_ui(self):
        """Builds the UI from ``brusheditor.glade``"""

        self._updating_ui = True
        ui_dir = os.path.dirname(os.path.abspath(__file__))
        ui_path = os.path.join(ui_dir, self._UI_DEFINITION_FILE)
        with open(ui_path, 'r') as ui_fp:
            ui_xml = ui_fp.read()

        builder = self._builder
        builder.add_from_string(ui_xml)

        stamp_grid = builder.get_object('stamp_grid')
        sw = builder.get_object('stamp_scrolledwindow')

        self.stamp_icon = builder.get_object('stamp_icon_image')

        self.stamp_name = builder.get_object('stamp_name_label')
        self.stamp_name.set_text(self._stamp.name)

        self.stamp_desc = builder.get_object('stamp_type_label')
        self.stamp_desc.set_text(self._stamp.desc)

        self.stamp_icon = builder.get_object('stamp_icon_image')
        self.stamp_icon.set_from_pixbuf(self._stamp.thumbnail)

        # Building stamp list view.

        # Important: Ensure all tiles surface
        # before calling StampStore constructor.
        self._stamp.validate_all_tiles() 
                                         
        store = StampStore(self._stamp)
        self._stamp_list_store = store

        view = Gtk.TreeView()
        view.set_model(store.liststore)

        stamprender = StampRenderer(store)
        col = Gtk.TreeViewColumn(_('Stamp'), stamprender, surface=0)
        col.set_sizing(Gtk.TreeViewColumnSizing.AUTOSIZE)
        col.set_resizable(True)
        view.append_column(col)

        def generate_text_column(label, idx):
            textrender = Gtk.CellRendererText()
            col = Gtk.TreeViewColumn(label, textrender, text=idx)
            col.set_resizable(True)
            view.append_column(col)

        generate_text_column(_('Size'), 1)
        generate_text_column(_('Source'), 2)

        view.set_hexpand(True)
        view.set_vexpand(True)
        view.set_halign(Gtk.Align.FILL)
        view.set_valign(Gtk.Align.FILL)
        view.set_grid_lines(Gtk.TreeViewGridLines.BOTH)

        view.connect('key-release-event', self.treeview_key_released_cb)

        self._stamp_list = view
        sw.set_vexpand(True)
        sw.set_hexpand(True)
        sw.set_halign(Gtk.Align.FILL)
        sw.set_valign(Gtk.Align.FILL)
        sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)            
        sw.add(view)

        # Connecting signals which written in glade.
        self._builder.connect_signals(self)
            
        # Certain actions must be coordinated via a real app instance
        if not self.app:
            action_buttons = [
                "save_button",
                "delete_button",
               #"close_button"
            ]
            for b_name in action_buttons:
                w = self._builder.get_object(b_name)
                w.set_sensitive(False)

        self._updating_ui = False
    
    def _post_show_cb(self, widget):
        return

    ## Main action buttons

    def save_button_clicked_cb(self, button):
        """ Save the current stamp settings.
        If target stamp is not file-backed type,
        this handler creates (exports to) new file-backed stamp.
        """
        pass

    def load_button_clicked_cb(self, button):
        """ load the stamp settings from a file """
        pass

    def refresh_button_clicked_cb(self, button):
        """ Refresh - reflect the current stamp settings  
        into treeview.
        """
        pass

    def insert_button_clicked_cb(self, button):
        """Insert a picture """
        pass

    def delete_button_clicked_cb(self, button):
        """delete a currently selected stamp picture """
        pass

    def close_button_clicked_cb(self, button):
        print('close button clicked')
        pass

    ## Settings treeview management and change callbacks
    # XXX Copied from gui/brusheditor.py

    def _settings_treeview_selectfunc(self, seln, model, path, is_seld, data):
        """Determines whether settings listview rows can be selected"""
        i = model.get_iter(path)
        is_leaf = model.get_value(i, self._LISTVIEW_IS_SELECTABLE_COLUMN)
        return is_leaf

    def settings_treeview_row_activated_cb(self, view, path, column):
        """Double clicking opens expander rows"""
        model = view.get_model()
        i = model.get_iter(path)
        is_leaf = model.get_value(i, self._LISTVIEW_IS_SELECTABLE_COLUMN)
        if is_leaf or not view.get_visible():
            return
        if view.row_expanded(path):
            view.collapse_row(path)
        else:
            view.expand_row(path, True)

    def settings_treeview_cursor_changed_cb(self, view):
        """User has chosen a different setting using the treeview"""
        sel = view.get_selection()
        if sel is None:
            return
        model, i = sel.get_selected()
        setting = self._setting
        if i is None:
            setting = None
        else:
            cname = model.get_value(i, self._LISTVIEW_CNAME_COLUMN)
            setting = brushsettings.settings_dict.get(cname)
        if setting is not self._setting:
            self._setting = setting
            self._current_setting_changed()

    ## Treeview signal handlers

    def treeview_key_released_cb(self, view, event):
        if event.keyval in (Gdk.KEY_Delete, Gdk.KEY_BackSpace):
            self.delete_button_clicked_cb(view)
        elif event.keyval in (Gdk.KEY_Insert, ):
            self.insert_button_clicked_cb(view)


def _test():
    """Run interactive tests, outside the application."""

    logging.basicConfig()

    # Create dummy stamp.
    print("### TEST ###")
    sm = gui.stamps.StampPresetManager(None)
    TEST_STAMP = [
                { "version" : "1",
                  "name" : "test stamps",
                  "settings" : {
                      "source" : "file",
                      "gtk-thumbnail" : "gtk-paste",
                      "filenames" : [
                                    'pixmaps/mypaint_logo.png',
                                    'pixmaps/mypaint_logo.png',
                                    'pixmaps/layers.png' ,
                                    'pixmaps/mypaint_logo.png',
                                    'pixmaps/plus.png'
                                    ],
                      },
                  "desc" : _("A sequence of test stamps")
                },
                ]
    stamp = sm.create_stamp_from_json(TEST_STAMP[0])
    sm.stamps.append(stamp)
    sm.set_current_index(-1)

    win = StampEditorWindow(None, sm.current, sm)
    win.connect("delete-event", lambda *a: Gtk.main_quit())
    win.show_all()
    Gtk.main()


if __name__ == '__main__':
    _test()



