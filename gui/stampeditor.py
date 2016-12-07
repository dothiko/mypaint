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

from windowing import SubWindow
import gui.stamps

## Class definitions

class StampStoreWrapper(object):
    """Stamp Store wrapper, used from stamp listview"""

    STORE_IDX_SURF=0
    STORE_IDX_SIZE=1
    STORE_IDX_DESC=2
    STORE_IDX_ID=3

    def __init__(self, stamp):
        self._stamp = stamp
        self._store = Gtk.ListStore(object, str, str, int)
        for id, surf in stamp.source_surface_iter():
            # XXX Currently 'desc' is empty(might be obsoluted)
            self._append_single_surf(surf, "", id)

    def _append_single_surf(self, surf, desc, id):
        size = StampStoreWrapper.get_surf_size_str(surf)
        self._store.append((surf, size, desc, id))

    @staticmethod
    def get_surf_size_str(surf):
        return "%d x %d" % (surf.get_width(), surf.get_height())

    @property
    def liststore(self):
        return self._store

    def refresh(self, force_update=False):
        """Refresh ListStore with current stamp pictures
        """
        store = self._store
        stamp = self._stamp
        if force_update:
            store.clear()
            for id, surf in stamp.source_surface_iter():
                self._append_single_surf(surf, "", id)
        else:
            # Refresh current _pictures_store
            for id, surf in stamp.source_surface_iter():
                iter = store.get_iter_first()
                while iter:
                    old_surf= store.get_value(iter, self.STORE_IDX_SURF) 
                    store_id = store.get_value(iter, self.STORE_IDX_ID)
                    if id == store_id:
                        # Already this stamp exist.
                        if old_surf != surf:
                            # Already this stamp exist, but icon changed.
                            # so update it, and exit loop.
                            store.set_value(iter, self.STORE_IDX_SURF, surf)
                            size = StampStoreWrapper.get_surf_size_str(surf)
                            store.set_value(iter, self.STORE_IDX_SIZE, size)

                        # Clear 'id' variable to notify 
                        # 'current icon is already registered in the store'
                        id = None
                        break
                    else:
                        iter = store.iter_next(iter)
                        continue

                if id != None:
                    # current icon is not registered into picture store
                    self._append_single_surf(surf, "" , id)

    def remove(self, iter):
        """Remove a picture from stamp, with iter."""
        store = self._store
        id = store.get_value(iter, self.STORE_IDX_ID)
        store.remove(iter)
        self._stamp.remove(id)

    def append_from_file(self, filename): 
        """Append a picture to stamp, from filename."""
        id = self._stamp.add_file_source(filename)
        self._append_single_surf(self._stamp.get_surface(id), "", id)

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

    _DEFAULT_WIDTH = 512
    _DEFAULT_HEIGHT = 480

    ## Construction

    def __init__(self):
        self.manager = None
        self._stamp = None
        if __name__ != '__main__':
            from application import get_app
            app = get_app()
            self.manager = app.stamp_manager
            self._stamp = app.stamp_manager.get_current()
        else:
            # For testing
            self.manager, self._stamp, app = _test_generate_objects()

        SubWindow.__init__(self, app, key_input=True)
        # self.app attribute is set at superclass constructor.

        self.set_title(C_(
            "Stamp settings editor: subwindow title",
            "Stamp Settings Editor",
        ))
        self._buttons = {}
        self._stamp_list_store_wrapper = None

        self._builder = Gtk.Builder()
        self._builder.set_translation_domain("mypaint")
        self._build_ui()

        self.connect_after("show", self._post_show_cb)
        editor = self._builder.get_object("stamp_editor")
        self.add(editor)

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
        self._stamp.ensure_sources()

        # Building main treeview
        view = Gtk.TreeView()
        self._stamp_list = view

        wrapper = self._refresh_store_wrapper()

        stamprender = StampRenderer(wrapper)
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

        selection = view.get_selection()
        selection.connect('changed', self.treeview_selection_changed_cb)

        sw.set_vexpand(True)
        sw.set_hexpand(True)
        sw.set_halign(Gtk.Align.FILL)
        sw.set_valign(Gtk.Align.FILL)
        sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)            
        sw.add(view)
        view.set_visible(True)
        sw.set_visible(True)

        # Connecting signals which written in glade.
        self._builder.connect_signals(self)
            
        # Building buttons
        action_buttons = [
            "save_button",
            "delete_button",
        ]
        for b_name in action_buttons:
            w = self._builder.get_object(b_name)
            w.set_sensitive(False)
            self._buttons[b_name] = w

        self._buttons['save_button'].set_sensitive(self._stamp.dirty)

        # For unit testing.
        if __name__ == '__main__':
            self.connect('hide', self.close_window_cb)

        self.set_size_request(self._DEFAULT_WIDTH, self._DEFAULT_HEIGHT)
        self._updating_ui = False

    def _refresh_store_wrapper(self):
        old = None
        if self._stamp_list_store_wrapper:
            old = self._stamp_list_store_wrapper

        wrapper = StampStoreWrapper(self._stamp)
        wrapper.liststore.connect("row-deleted", self.stamp_store_deleted_cb)
        wrapper.liststore.connect("row-inserted", self.stamp_store_inserted_cb)
        self._stamp_list_store_wrapper = wrapper

        if old:
            del old

        self._modified = False
        self._stamp_list.set_model(wrapper.liststore)

        return wrapper
    
    def _post_show_cb(self, widget):
        return

    ## Main action buttons

    def export_button_clicked_cb(self, button):
        """ Save the current stamp settings.
        If target stamp is not file-backed type,
        this handler creates (exports to) new file-backed stamp.
        """
        dialog = Gtk.FileChooserDialog(
            _("Export..."),
            self.app.drawWindow,
            Gtk.FileChooserAction.SAVE,
            (
                Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                Gtk.STOCK_SAVE, Gtk.ResponseType.OK,
            ),
        )
        dialog.set_default_response(Gtk.ResponseType.OK)
        dialog.set_do_overwrite_confirmation(True)
        dialog.show_all()

        if self._stamp.filename != '':
            dialog.set_filename(self._stamp.filename)
        else:
            default_filename = self.manager.get_adjusted_path("untitled.mys")
            dialog.set_filename(default_filename)

        try:
            response = dialog.run()
            if response == Gtk.ResponseType.OK:
                filename = dialog.get_filename()
                ext = os.path.splitext(filename)[1]
                if ext == '':
                    filename = u"%s.mys" % filename
                self._stamp.save_to_file(filename)

        finally:
            dialog.destroy()


    def load_button_clicked_cb(self, button):
        """ Load a stamp preset file.
        """
        pass

    def refresh_button_clicked_cb(self, button):
        """ Refresh - reflect the current stamp settings  
        into treeview.
        """
        self._refresh_store_wrapper()
        selection = self._stamp_list.get_selection()
        store, iter = selection.get_selected()
        self._buttons['delete_button'].set_sensitive(iter != None)
        self._buttons['save_button'].set_sensitive(self._stamp.dirty)

    def append_button_clicked_cb(self, button):
        """Insert a  stamp picture from a file """
        fh = self.app.filehandler

        # [TODO] Ora file currently unsupported.
        # because it would be hard to flatten...?
        file_filters = [
            # (name, patterns)
            (_("Picture Files"), ("*.png", "*.jpg", "*.jpeg")),
            (_("PNG (*.png)"), ("*.png",)),
            (_("JPEG (*.jpg; *.jpeg)"), ("*.jpg", "*.jpeg")),
        ]
        dialog = fh.get_open_dialog(file_filters=file_filters)
        response = dialog.run()
        try:
            if response == Gtk.ResponseType.OK:
                fname = dialog.get_filename()
                if os.path.exists(fname):
                    self._stamp_list_store_wrapper.append_from_file(fname)
                    self._refresh_store_wrapper()
                else:
                    logger.warning("file %s disappeared before load." % fname)
            elif response == Gtk.ResponseType.CANCEL:
                pass
        finally:
            dialog.destroy() 

    def delete_button_clicked_cb(self, button):
        """delete a currently selected stamp picture """
        selection = self._stamp_list.get_selection()
        store, iter = selection.get_selected()
        if iter != None:
            self._stamp_list_store_wrapper.remove(iter)

    ## #indow handlers

    def close_window_cb(self, widget):
        """For unit testing.
        """ 
        assert __name__ == '__main__'
        Gtk.main_quit()

    ## Stamp store(model) handlers
    def stamp_store_deleted_cb(self, store, path):
        self._buttons['save_button'].set_sensitive(True)
        self._modified = True

    def stamp_store_inserted_cb(self, store, path, iter):
        self.stamp_store_deleted_cb(store, path)

    ## Treeview signal handlers
    def treeview_selection_changed_cb(self, selection):
       #store, iter = selection.get_selected()
        self._buttons['delete_button'].set_sensitive(True)

    def treeview_key_released_cb(self, view, event):
        if event.keyval in (Gdk.KEY_Delete, Gdk.KEY_BackSpace):
            self.delete_button_clicked_cb(view)
        elif event.keyval in (Gdk.KEY_Insert, ):
            self.insert_button_clicked_cb(view)

def _test_generate_objects():
    """Generate test stamps, for unit test.
    """
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

    # Create Dummy app class, to use gui.filehandling.FileHandler
    # even in unit test.
    import gui.filehandling

    class DummyKbm(object):
        def takeover_action(self, arg):
            pass

    class DummyApp(object):

        def __init__(self):
            self.drawWindow = None
            builder = Gtk.Builder()
            xml="""
            <interface>
              <object class="GtkActionGroup" id="FileActions">
              <child>
              <object class="GtkAction" id="menuitem1">
                <property name="name">menuitem1</property>
                <property name="label" translatable="yes">_File</property>
              </object>
              </child>
              </object>
            </interface>
            """
            builder.add_from_string(xml)
            self.builder = builder
            self.kbm = DummyKbm()
            self.filehandler = gui.filehandling.FileHandler(self)

        def find_action(self, arg):
            return Gtk.RecentChooserMenu()

    app = DummyApp()

    sm = gui.stamps.StampPresetManager(None)
    stamp = sm.create_stamp_from_json(TEST_STAMP[0])
    assert stamp != None
    sm.stamps[0] = stamp
    sm.set_current(stamp)
    return (sm, stamp, app)

def _test():
    """Run interactive unit test, outside the application."""

    logging.basicConfig()

    print("### TEST ###")
    win = StampEditorWindow()
    win.connect("delete-event", lambda *a: Gtk.main_quit())
    win.show()
    Gtk.main()


if __name__ == '__main__':
    _test()



