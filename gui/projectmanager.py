#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Project manager"""


## Imports
from __future__ import print_function

import os
import datetime
import xml.etree.ElementTree as ET
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
import cairo

from windowing import SubWindow
import gui.stamps

## Class definitions
class _Visiblity:
    """Enumeration of visibility state of each version-directory"""
    NORMAL = 0
    IGNORE = 1
    ALWAYS_VISIBLE = 2  

class _Store_index:
    """Enumeration of contents-index at Liststore"""
    SURF=0
    DATE=1
    IGNORESTATUS=2
    DESC=3
    PATH=4
    ID=5

class _Version_id:
    """Enumeration of special version-id """
    CURRENT = -1
    BASE = 0

class VersionStoreWrapper(object):
    """Version Store wrapper, used from version listview"""



    # temporary flag file name.
    # if a file named as this in a version directory,
    # that version would be ignored.
    IGNORE_FILE_NAME = 'ignore-this'

    def __init__(self):

        basestore = Gtk.ListStore(object, str, int, str, str, int)
        self._basestore = basestore

        self._store = basestore.filter_new()
        self._store.set_visible_func(self.visible_func_cb)
        self._id_seed = _Version_id.BASE
        self._show_all = False

    @property
    def liststore(self):
        return self._store

    @property
    def show_all(self):
        return self._show_all

    @show_all.setter
    def show_all(self, flag):
        self._show_all = flag

    def visible_func_cb(self, model, iter, data):
        """ The visible filter callback for Gtk.TreeModelFilter.
        """ 
        retflag = False
        status = model[iter][_Store_index.IGNORESTATUS]
        retflag |= (status != _Visiblity.IGNORE or self._show_all)
        return retflag

    def remove(self, iter, actually_delete=False):
        """Remove a version, with iter.
        
        :param actually_delete: if this is True, remove the version directory
            from filesystem.Otherwise, create 'ignore-this' empty file to
            ignore when walking backup directory at initialize.
        """
        store = self._store
        id = store.get_value(iter, _Store_index.ID)
        path = store.get_value(iter, _Store_index.PATH)
        store.remove(iter)

        # Remove directory...!
        if actually_delete:
            logger.warning("Deleting version directory is not implemented yet.")
        else:
            ignore_flag_file = os.path.join(path, self.IGNORE_FILE_NAME)
            if not os.path.exists(ignore_flag_file):
                with open(ignore_flag_file, 'wt') as ofp:
                    ofp.write('flag file: ignore this directory')

    def get_surface(self, iter):
        return self._store[iter]

    def load_from_directory(self, dirname):
        """Load all versions
        """
        def append_store(fname, datestr, status, desc, dirpath, id=None):
            assert os.path.exists(fname)
            surf = cairo.ImageSurface.create_from_png(fname)
            if id == None:
                id = self._id_seed
                self._id_seed += 1

            self._basestore.append((surf, datestr, status, desc, dirpath, id))

        thumbnail_path = os.path.join(dirname, 'Thumbnails', 'thumbnail.png')
        append_store(thumbnail_path, _("Current version"), 
                _Visiblity.ALWAYS_VISIBLE,
                "", dirname, _Version_id.CURRENT) 

        backupdir = os.path.join(dirname, 'backup')
        if os.path.exists(backupdir):
            for cdir in os.listdir(backupdir):
                try:
                    datebase = [int(x) for x in cdir.split('-')]
                    if len(datebase) == 0:
                        raise ValueError
                except ValueError:
                    logger.error("failed to convert directory base to integer array.")
                    continue

                basedir = os.path.join(backupdir, cdir)
                for cdir in os.listdir(basedir):

                    status = _Visiblity.NORMAL

                    ignore_flag_file = os.path.join(basedir, cdir, self.IGNORE_FILE_NAME)
                    if os.path.exists(ignore_flag_file):
                        status = _Visiblity.IGNORE

                    try:
                        datetail = [int(x) for x in cdir.split('-')]
                        if len(datetail) == 0:
                            raise ValueError
                    except ValueError:
                        logger.error("failed to convert directory tail to integer array.")
                        continue

                    thumbnail_path = os.path.join(basedir, cdir, 'thumbnail.png')
                    datesrc = [int(x) for x in datebase + datetail]
                    dateobj = datetime.datetime(*datesrc)
                    datestr = dateobj.strftime("%c")

                    descpath = os.path.join(basedir, cdir, "desc.txt")
                    desc = ""
                    if os.path.exists(descpath):
                        with open(descpath, 'rt') in ifp:
                            desc = ifp.read()

                    append_store(thumbnail_path, datestr, status, desc, cdir, status) 




class ThumbnailRenderer(Gtk.CellRenderer):
    """renderer used from version thumbnail"""

    surface = GObject.property(type=GObject.TYPE_PYOBJECT, default=None)
   #id = GObject.property(type=int , default=-1)
   #desc = GObject.property(type=str, default="")

    def __init__(self):
        super(ThumbnailRenderer, self).__init__()

    # Gtk Property related:
    # These are something like a idiom.
    # DO NOT EDIT
    def do_set_property(self, pspec, value):
        setattr(self, pspec.name, value)

    def do_get_property(self, pspec):
        return getattr(self, pspec.name)

    # Rendering
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

class IgnorestateRenderer(Gtk.CellRenderer):
    """Renderer to be draw Status bitflags """

    status = GObject.property(type=int, default=0)
    VISIBLE_ICON = None
    INVISIBLE_ICON = None
    ICON_SIZE = 32
    

    def __init__(self):
        super(IgnorestateRenderer, self).__init__()

    # Gtk Property related:
    # These are something like a idiom.
    # DO NOT EDIT
    def do_set_property(self, pspec, value):
        setattr(self, pspec.name, value)

    def do_get_property(self, pspec):
        return getattr(self, pspec.name)

    # Ordinary properties:
    @property
    def visible_icon_surface(self):
        cls = self.__class__
        if cls.VISIBLE_ICON == None:
            pixbuf = gui.drawutils.load_symbolic_icon(
                icon_name="mypaint-object-visible-symbolic",
                size=cls.ICON_SIZE,
                fg=(0, 0, 0, 1),
            )
            cls.VISIBLE_ICON = Gdk.cairo_surface_create_from_pixbuf(pixbuf, 1, None)

        return cls.VISIBLE_ICON

    @property
    def invisible_icon_surface(self):
        cls = self.__class__
        if cls.INVISIBLE_ICON == None:
            pixbuf = gui.drawutils.load_symbolic_icon(
                icon_name="mypaint-object-hidden-symbolic",
                size=cls.ICON_SIZE,
                fg=(0, 0, 0, 1),
            )
            cls.INVISIBLE_ICON = Gdk.cairo_surface_create_from_pixbuf(pixbuf, 1, None)

        return cls.INVISIBLE_ICON

    # Rendering
    def do_render(self, cr, widget, background_area, cell_area, flags):
        """
        :param cell_area: RectangleInt class
        """

        # Draw background as cell style
        cr.set_source_rgba(*self.get_property('cell-background-rgba'))
        cr.rectangle(cell_area.x, cell_area.y, 
                cell_area.width, cell_area.height)
        cr.fill()

        status = self.status
        if status == _Visiblity.IGNORE:
            surf = self.invisible_icon_surface
        elif status == _Visiblity.NORMAL:
            surf = self.visible_icon_surface
        else:
            return # no draw for _Visiblity.ALWAYS_VISIBLE

        cr.save()
        sw = surf.get_width()
        sh = surf.get_height()
        cr.translate(cell_area.x + cell_area.width / 2 - sw / 2, 
                cell_area.y + cell_area.height / 2 - sh / 2) 
        cr.set_source_surface(surf, 0 , 0)
        cr.rectangle(0, 0, sw, sh)
        cr.fill()
        cr.restore()

    def do_get_preferred_width(self,view_widget):
        return (self.ICON_SIZE * 2, self.ICON_SIZE * 2) 

class ProjectManagerWindow (SubWindow):
    """Window for the project manager
    
    Use set_directory() method, to set target directory.
    """

    ## Class constants

    _UI_DEFINITION_FILE = "projectmanager.glade"
    _LISTVIEW_THUMBNAIL_COLUMN = 0
    _LISTVIEW_DISPLAYNAME_COLUMN = 1

    _LISTVIEW_COLUMN_HEIGHT = 80

    _DEFAULT_WIDTH = 512
    _DEFAULT_HEIGHT = 480

    ## Construction

    def __init__(self):
        if __name__ != '__main__':
            from application import get_app
            app = get_app()
        else:
            # For testing
            app = _test_generate_objects()

        SubWindow.__init__(self, app, key_input=True)
        # self.app attribute is set at superclass constructor.

        self.set_title(C_(
            "Project manager window: subwindow title",
            "Project Manager",
        ))
        self._buttons = {}
        self._version_list_store_wrapper = None

        self._builder = Gtk.Builder()
        self._builder.set_translation_domain("mypaint")
        self._build_ui()

        self.connect_after("show", self._post_show_cb)
        editor = self._builder.get_object("project_manager")
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

        project_grid = builder.get_object('project_grid')
        sw = builder.get_object('project_scrolledwindow')

        self.project_icon = builder.get_object('project_icon_image')
        self.project_name = builder.get_object('project_name_label')
        self.directory_label = builder.get_object('directory_label')
        self.showall_checkbutton = builder.get_object('showall_checkbutton')

        # Building project list view.

        # Building main treeview
        view = Gtk.TreeView()
        self._version_list = view

        thumbrender = ThumbnailRenderer()
        col = Gtk.TreeViewColumn(_('Thumbnail'), thumbrender, surface=0)
        col.set_sizing(Gtk.TreeViewColumnSizing.AUTOSIZE)
        col.set_resizable(True)
        view.append_column(col)

        renderer = Gtk.CellRendererText()
        col = Gtk.TreeViewColumn(_('Date'), renderer, text=1)
        col.set_resizable(True)
        view.append_column(col)

        renderer = IgnorestateRenderer()
        col = Gtk.TreeViewColumn(_('Ignore'), renderer, status=2)
        col.set_resizable(True)
        self._status_column = col
        
        if self.showall_checkbutton.get_active():
            view.append_column(self._status_column)

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
        action_buttons = (
            "revert_button",
            "delete_button",
            "ignore_button",
        )
        for b_name in action_buttons:
            w = self._builder.get_object(b_name)
            w.set_sensitive(False)
            self._buttons[b_name] = w

       #self._buttons['revert_button'].set_sensitive(self._stamp.dirty)
       #self._buttons['delete_button'].set_sensitive(self._stamp.dirty)

        # For unit testing.
        if __name__ == '__main__':
            self.connect('hide', self.close_window_cb)

        self.set_size_request(self._DEFAULT_WIDTH, self._DEFAULT_HEIGHT)
        self._updating_ui = False
    

    def set_directory(self, projdir):
        """ Set target directory, which should contain 
        stack.xml and backuped versions directory.
        """
        old = None
        if self._version_list_store_wrapper:
            old = self._version_list_store_wrapper

        wrapper = VersionStoreWrapper()
        wrapper.load_from_directory(projdir)
        wrapper.liststore.connect("row-deleted", self.version_store_deleted_cb)
        wrapper.liststore.connect("row-inserted", self.version_store_inserted_cb)
        self._version_list_store_wrapper = wrapper

        if old:
            del old

        self._modified = False
        self._version_list.set_model(wrapper.liststore)

        xmlpath = os.path.join(projdir, "stack.xml")
        assert os.path.exists(xmlpath)

        projname = os.path.basename(projdir)

        self.project_name.set_text(projname)
        self.directory_label.set_text(projdir)

        return wrapper
    
    def _post_show_cb(self, widget):
        return

    ## Main action buttons

    def revert_button_clicked_cb(self, button):
        """Revert project to currently selected version
        """
        pass

    def delete_button_clicked_cb(self, button):
        """Delete a currently selected version"""
        selection = self._version_list.get_selection()
        store, iter = selection.get_selected()
        if iter != None:
            self._version_list_store_wrapper.remove(iter)

    def ignore_button_clicked_cb(self, button):
        selection = self._version_list.get_selection()
        store, iter = selection.get_selected()
        if iter != None:
            dstidx = _Store_index.IGNORESTATUS
            ignore_flag = 1
            status = store[iter][dstidx]

            if status == _Visiblity.IGNORE: 
                status = _Visiblity.NORMAL
            elif status == _Visiblity.NORMAL:
                status = _Visiblity.IGNORE

            store[iter][dstidx] = status

    ## window handlers

    def close_window_cb(self, widget):
        """For unit testing only.
        """ 
        assert __name__ == '__main__'
        Gtk.main_quit()

    ## Version store(model) handlers
    def version_store_deleted_cb(self, store, path):
        self._modified = True

    def version_store_inserted_cb(self, store, path, iter):
        self.version_store_deleted_cb(store, path)


    ## Treeview signal handlers
    def treeview_selection_changed_cb(self, selection):
        store, iter = selection.get_selected()
        if iter:
            id = store[iter][_Store_index.ID]

            flag = (id != _Version_id.CURRENT)
            self._buttons['delete_button'].set_sensitive(flag)
            self._buttons['revert_button'].set_sensitive(flag)
            self._buttons['ignore_button'].set_sensitive(flag)

    def treeview_key_released_cb(self, view, event):
        if event.keyval in (Gdk.KEY_Delete, Gdk.KEY_BackSpace):
            self.delete_button_clicked_cb(view)

    ## Other widgets handlers
    def showall_checkbutton_toggled_cb(self, widget):
        view = self._version_list
        if len(view.get_columns()) > 2:
            view.remove_column(self._status_column)
        else:
            view.append_column(self._status_column)

        wrapper = self._version_list_store_wrapper
        assert wrapper != None
        wrapper.show_all = widget.get_active()

def _test_generate_objects():
    """Generate test stamps, for unit test.
    """

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
    return app

def _test():
    """Run interactive unit test, outside the application."""

    logging.basicConfig()
    icon_theme = Gtk.IconTheme.get_default()
    icon_theme.append_search_path("./desktop/icons")

    print("### TEST ###")
    win = ProjectManagerWindow()
    # Use your own project-saved directory.
    win.set_directory("/home/dothiko/workarea/2016/gantan/final_image")
    win.connect("delete-event", lambda *a: Gtk.main_quit())
    win.show()
    Gtk.main()


if __name__ == '__main__':
    _test()





