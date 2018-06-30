#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Project manager"""


## Imports
from __future__ import print_function

import os
import time
import xml.etree.ElementTree as ET
import logging
logger = logging.getLogger(__name__)
import glob
import weakref

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
import gui.dialogs
import lib.projectsave

## Class definitions
class _Store_index:
    """Enumeration of contents-index at Liststore"""
    SURF=0
    DATE=1
    VISIBLESTATUS=2
    DESC=3
    VER=4

_THUMBNAIL_WIDTH = 128
_THUMBNAIL_HEIGHT = 90

_ICON_COLOR = (0.4, 0.4, 0.4, 1.0)

_CURRENT_VERSION_NUM = 0

class VersionStoreWrapper(object):
    """Version Store wrapper for Gtk.ListStore, used from version listview"""

    # temporary flag file name.
    # if a file named as this in a version directory,
    # that version would be ignored.
    IGNORE_FILE_NAME = 'ignore-this'

    def __init__(self, checkpt_info, dirname):

        basestore = Gtk.ListStore(object, str, bool, str, int)
        self._basestore = basestore

        self._store = basestore.filter_new()
        self._store.set_visible_func(self.visible_func_cb)
        self._show_all = False
        self.checkpt_info = weakref.proxy(checkpt_info)

        self._load_from_directory(dirname)

    @property
    def liststore(self):
        return self._store

    @property
    def show_all(self):
        return self._show_all

    @show_all.setter
    def show_all(self, flag):
        self._show_all = flag

    @property
    def invalid_thumbnail(self):
        if not hasattr(self, '_invalid_thumbnail'):
            pixbuf = gui.drawutils.load_symbolic_icon(
                icon_name="mypaint-layer-fallback-symbolic",
                size=min(_THUMBNAIL_WIDTH, _THUMBNAIL_HEIGHT),
                fg=_ICON_COLOR,
            )
            surf = Gdk.cairo_surface_create_from_pixbuf(pixbuf, 1, None)
            self._invalid_thumbnail = surf
        return self._invalid_thumbnail

    def visible_func_cb(self, model, iter, data):
        """ The visible filter callback for Gtk.TreeModelFilter.
        """ 
        retflag = False
        status = model[iter][_Store_index.VISIBLESTATUS]
        retflag |= (status or self._show_all)
        return retflag

    def remove(self, iter, actually_delete=False):
        """Remove a version, with iter.
        
        :param actually_delete: if this is True, remove the version directory
            from filesystem.Otherwise, create 'ignore-this' empty file to
            ignore when walking checkpoints directory at initialize.
        """
        store = self._store
        ver_num = store.get_value(iter, _Store_index.VER)
        path = store.get_value(iter, _Store_index.PATH)
        store.remove(iter)

        # Remove directory...!
        if actually_delete:
            logger.warning("Deleting version is not implemented yet.")
        else:
            ignore_flag_file = os.path.join(path, self.IGNORE_FILE_NAME)
            if not os.path.exists(ignore_flag_file):
                with open(ignore_flag_file, 'wt') as ofp:
                    ofp.write('flag file: ignore this directory')

    def get_surface(self, iter):
        return self._store[iter]

    def _load_from_directory(self, dirname):
        """Load all versions
        """
        def append_store(thumbpath, datestr, status, desc, vernum):
            if os.path.exists(thumbpath):
                surf = cairo.ImageSurface.create_from_png(thumbpath)
            else:
                surf = self.invalid_thumbnail
            self._basestore.append((surf, datestr, status, 
                desc, vernum))

        thumbnail_path = os.path.join(dirname, 'Thumbnails', 'thumbnail.png')
        append_store(thumbnail_path, _("Current version"), 
                True,
                "", _CURRENT_VERSION_NUM) 

        vinfo = self.checkpt_info
        checkpointdir = os.path.join(dirname, 'checkpoints')
        if os.path.exists(checkpointdir):
            xmls = glob.glob(os.path.join(checkpointdir,"stack.xml.*"))
            xmls.sort(reverse=True)
            for cxml in xmls:
                stinfo = os.stat(cxml)
                mod_date = time.localtime(stinfo.st_mtime)
                datestr = time.strftime("%c", mod_date)

                # Extract version number as file-extension.
                junk, ver_num = os.path.splitext(cxml)
                try:
                    ver_num = int(ver_num[1:])
                except ValueError:
                    logger.error("%s is not version xml" % cxml)
                    continue

                thumbnail_path = os.path.join(
                    checkpointdir, 
                    'thumbnail.png.%d' % ver_num
                )

                desc = vinfo.get_description(ver_num)
                status = vinfo.get_visible_status(ver_num)
                append_store(thumbnail_path, datestr, status, desc, ver_num) 

class ThumbnailRenderer(Gtk.CellRenderer):
    """renderer used from version thumbnail"""

    surface = GObject.property(type=GObject.TYPE_PYOBJECT, default=None)
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
        return (0, 0, _THUMBNAIL_WIDTH, _THUMBNAIL_HEIGHT)


class IgnorestateRenderer(Gtk.CellRenderer):
    """Renderer to be draw Status bitflags """

    status = GObject.property(type=bool, default=0)
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
                fg=_ICON_COLOR,
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

        visible = self.status
        if not visible:
            surf = self.invisible_icon_surface
        else:
            surf = self.visible_icon_surface

        cr.save()
        sw = surf.get_width()
        sh = surf.get_height()
        cr.translate(cell_area.x + cell_area.width / 2 - sw / 2, 
                cell_area.y + cell_area.height / 2 - sh / 2) 
        cr.set_source_surface(surf, 0 , 0)
        cr.rectangle(0, 0, sw, sh)
        cr.fill()
        cr.restore()

    def do_get_preferred_width(self, view_widget):
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
        self._store_wrapper = None

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
        col = Gtk.TreeViewColumn(_('Thumbnail'), thumbrender, 
                                 surface=_Store_index.SURF)
        col.set_sizing(Gtk.TreeViewColumnSizing.AUTOSIZE)
        col.set_resizable(True)
        view.append_column(col)

        renderer = Gtk.CellRendererText()
        col = Gtk.TreeViewColumn(_('Date'), renderer, 
                                 text=_Store_index.DATE)
        col.set_resizable(True)
        view.append_column(col)

        renderer = Gtk.CellRendererText()
        col = Gtk.TreeViewColumn(_('Description'), renderer, 
                                 text=_Store_index.DESC)
        col.set_resizable(True)
        view.append_column(col)

        renderer = IgnorestateRenderer()
        col = Gtk.TreeViewColumn(_('Ignore'), renderer, 
                                 status=_Store_index.VISIBLESTATUS)
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

        self.connect('hide', self.hide_window_cb)

        self.set_size_request(self._DEFAULT_WIDTH, self._DEFAULT_HEIGHT)
        self._updating_ui = False

    def set_directory(self, projdir, filehandler):
        """ Set target directory, which should contain 
        stack.xml and checkpoints directory.
        This method is startup point of this dialog window.
        """
        if self._store_wrapper:
            del self._store_wrapper
            self._store_wrapper = None

        if filehandler:
            self._filehandler = weakref.ref(filehandler)
        else:
            self._filehandler = None

        self.project_dir = projdir

        if projdir == None:
            self.checkpt_info = None
            return

        checkpt_info = lib.projectsave.Checkpoint(projdir)
        wrapper = VersionStoreWrapper(checkpt_info, projdir)
        wrapper.liststore.connect("row-deleted", 
                                  self.version_store_deleted_cb)
        wrapper.liststore.connect("row-inserted", 
                                  self.version_store_inserted_cb)
        self._store_wrapper = wrapper
        self.checkpt_info = checkpt_info

        self._modified = False
        self._version_list.set_model(wrapper.liststore)

        xmlpath = os.path.join(projdir, "stack.xml")
        assert os.path.exists(xmlpath)

        projname = os.path.basename(projdir)

        self.project_name.set_text(projname)
        self.directory_label.set_text(projdir)

        # This window is modal, 
        # avoiding user to change mypaint(canvas) state.
        self.set_modal(True)

    def _post_show_cb(self, widget):
        return

    @property
    def filehandler(self):
        if self._filehandler:
            return self._filehandler()
        else:
            return None

    ## Main action buttons

    def revert_button_clicked_cb(self, button):
        """Revert project to currently selected version
        """
        selection = self._version_list.get_selection()
        store, iter = selection.get_selected()
        if not iter:
            return
        
        checkpt_info = self.checkpt_info
        target_version = store[iter][_Store_index.VER]

        filehandler = self.filehandler
        if filehandler:
            model = filehandler.doc.model
            assert model.is_project
            assert target_version != model.project_version

            save_before_revert = False

            # Warn to user about unsaved painting work.
            if model.unsaved_painting_time > 0: 
                save_before_revert = gui.dialogs.confirm(
                    self,
                    _("There is unsaved painting work.\n"
                    "Do you create checkpoint of current unsaved document "
                    "before revert?"),
                )
            elif (model.project_version > 0 and 
                    checkpt_info.is_current_document_changed(model.project_version)):
                # When right after document loaded,
                # model.unsaved_painting_time does not work.
                #
                # So use dedicated function 
                # `is_current_document_changed`
                # to detect whether the document is modified or not.

                save_before_revert = gui.dialogs.confirm(
                    self,
                    _("Checkpoint for this modified work is not created yet.\n"
                    "Do you create checkpoint of current unsaved document "
                    "before revert?"),
                )

            if save_before_revert:
                filehandler.set_project_checkpoint_cb(None)
                # Recreate checkpoint info, to update new version information
                checkpt_info = lib.projectsave.Checkpoint(self.project_dir)

                checkpt_info.set_description(
                    checkpt_info.max_version_num,
                    _("Automatically created revision,"
                      "before revert to version %d." % target_version)
                )

                # update checkpoint info attribute
                self.checkpt_info = checkpt_info

            prefs = filehandler.app.preferences
            display_colorspace_setting = prefs["display.colorspace"]
            if target_version != _CURRENT_VERSION_NUM:
                model.load(
                    filehandler.filename,
                    feedback_cb=filehandler.gtk_main_tick,
                    convert_to_srgb=(display_colorspace_setting == "srgb"),
                    target_version=target_version,
                    checkpt_info=self.checkpt_info
                )
            else:
                model.load(
                    filehandler.filename,
                    feedback_cb=filehandler.gtk_main_tick,
                    convert_to_srgb=(display_colorspace_setting == "srgb"),
                )

            # Version number of model(document) is set inside load
            # (actually, it is load_project) method.
                
            # Finalize checkpoint information
            # This updates json file of checkpoint information.
            checkpt_info.finalize()
            
        self.hide()

    def delete_button_clicked_cb(self, button):
        """Delete a currently selected version"""
        selection = self._version_list.get_selection()
        store, iter = selection.get_selected()
        if iter != None:
            self._store_wrapper.remove(iter)

    def ignore_button_clicked_cb(self, button):
        selection = self._version_list.get_selection()
        store, iter = selection.get_selected()
        if iter != None:
            ver_num = store[iter][_Store_index.VER]
            if ver_num > 0:
                dstidx = _Store_index.VISIBLESTATUS
                status = store[iter][dstidx]
                store[iter][dstidx] = not status

    def cancel_button_clicked_cb(self, button):
        self.hide()

    ## window handlers
    def hide_window_cb(self, widget):
        """For unit testing only.
        Actually this callback is not binded.
        """ 
        self.set_modal(False)
        self.set_directory(None, None)

        if __name__ == '__main__':
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
            ver_num = store[iter][_Store_index.VER]
            current_ver = 0
            filehandler = self.filehandler
            if filehandler:
                model = filehandler.doc.model
                current_ver = model.project_version

            flag = (ver_num != current_ver)
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

        wrapper = self._store_wrapper
        assert wrapper != None
        wrapper.show_all = widget.get_active()
        wrapper.liststore.refilter()

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
    win.set_directory("/home/dothiko/workarea/projtest/teste", None)
    win.connect("delete-event", lambda *a: Gtk.main_quit())
    win.show()
    Gtk.main()

if __name__ == '__main__':
    _test()
