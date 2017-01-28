#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Gradient editor"""


## Imports
from __future__ import print_function

import os
import weakref
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
from gui.gradient import *
import gui.drawutils
import lib.color

## Class definitions



class GradientEditorWindow (SubWindow):
    """Window containing the stamp editor"""

    ## Class constants

    _UI_DEFINITION_FILE = "gradienteditor.glade"
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
            "Gradient settings editor: subwindow title",
            "Gradient Settings Editor",
        ))
        self._buttons = {}
        self._gradient_controller = GradientController(app)

        self._builder = Gtk.Builder()
        self._builder.set_translation_domain("mypaint")
        self._build_ui()

        self._bg_surf = None
        self._bg_grad = None
        self.in_drag = False # whether user dragging gradiation controller.
                              # THIS ATTRIBUTE IS NEEDED TO DRAW 
                              # GRADIENT CONTROLLER.
        self._drag_should_start = False

       #self.connect_after("show", self._post_show_cb)
        editor = self._builder.get_object("gradient_editor")
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

        main_grid = builder.get_object('main_grid')

        self.draw_area = builder.get_object("gradient_drawingarea")
        self.draw_area.connect('size-allocate', 
                self.gradient_drawingarea_size_changed_cb)
        self.draw_area.add_events( 
                Gdk.EventMask.BUTTON_PRESS_MASK | 
                Gdk.EventMask.BUTTON_RELEASE_MASK | 
                Gdk.EventMask.BUTTON_MOTION_MASK) 
        self.name_label = builder.get_object('gradient_name_label')
        self.color_button = builder.get_object('colorbutton')
        self.alpha_adj = builder.get_object('alpha_adjustment')

        # Connecting signals which written in glade.
        self._builder.connect_signals(self)


            
        # Building buttons
        action_buttons = [
            "ok_button",
            "cancel_button",
            "rename_button",
            "new_button",
        ]
        for b_name in action_buttons:
            w = self._builder.get_object(b_name)
            w.set_sensitive(True)
            self._buttons[b_name] = w

       #self._buttons['save_button'].set_sensitive(self._stamp.dirty)

        self.connect('hide', self.close_window_cb)

        self._updating_ui = False

    def start(self, name, colors):
        grctrl = self._gradient_controller
        grctrl.setup_gradient(colors)

        self.name_label.set_text(name)
        self._original_name = name

        self._buttons["new_button"].set_sensitive(False)

        self.show()

    def invalidate_gradient(self):
        grctrl = self._gradient_controller
        grctrl.invalidate_cairo_gradient()
       #grctrl.queue_redraw(self)
        self._bg_grad = None
        da = self.draw_area.get_allocation()
        self.draw_area.queue_draw_area(
                0, 0, da.width, da.height)


    def set_color_to_colorbutton(self, gradnode):
        """Utility method, to set gradnode color
        to self.color_button."""
        self._updating_ui = True
        self.color_button.set_color(gradnode.gdk_color)
        self.alpha_adj.set_value(gradnode.alpha)
        self._updating_ui = False


    ## Dummy TDW methods
    # These dummy methods are needed for gradientcontroller.
    def model_to_display(self, x, y):
        return (x, y)

    def display_to_model(self, x, y):
        return (x, y)

    def queue_draw_area(self, x, y, w, h):
        self.draw_area.queue_draw_area(x, y, w, h)

    ## Properties
   #@property
   #def background_surface(self):
   #    if self._bg_surf is None:
   #        da = self.draw_area.get_allocation()
   #        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 
   #                da.width, da.height)
   #        da.x = 0
   #        da.y = 0
   #        cr = cairo.Context(surf)
   #        self.draw_background(cr, da)
   #        self._bg_surf = surf
   #    return self._bg_surf

    @property
    def background_gradient(self):
        if self._bg_grad is None:
            da = self.draw_area.get_allocation()
            sx = 0
            sy = 0
            ex = da.width - 1
            ey = 0
            self._bg_grad = self._gradient_controller.generate_gradient(
                    sx, sy, ex, ey)
        return self._bg_grad

   #def draw_background(self, cr, width, height, tile_size):
   #    y = 0
   #    idx = 0
   #    tilecolor = ( (0.3, 0.3, 0.3) , (0.7, 0.7, 0.7) )
   #
   #    cr.save()
   #    while y < height:
   #        x = 0
   #
   #        while x < width:
   #            cr.rectangle(x, y, 
   #                    tile_size, tile_size)
   #            cr.set_source_rgb(*tilecolor[idx%2])
   #            cr.fill()
   #            idx+=1
   #            x += tile_size
   #        y += tile_size
   #        idx += 1
   #    cr.restore()


    ## Main action buttons

    def ok_button_clicked_cb(self, button):
        """ Load a stamp preset file.
        """
        self.hide()

    def cancel_button_clicked_cb(self, button):
        """ Refresh - reflect the current stamp settings  
        into treeview.
        """
        self.hide()
       #self._buttons['delete_button'].set_sensitive(iter is not None)
       #self._buttons['save_button'].set_sensitive(self._stamp.dirty)
    def new_button_clicked_cb(self, button):
        newname = self.name_label.get_text()
        assert newname != self._original_name
        self.hide()

    def rename_button_clicked_cb(self, button):
        new_name = gui.dialogs.ask_for_name(
                    self,
                    _("Input name for this gradient"),
                    self.name_label.get_text()
                    )

        if new_name == '':
            gui.dialogs.error(self, _("Cannot accept empty name."))
        elif new_name is not None:
            self.name_label.set_text(new_name)
            self._buttons["new_button"].set_sensitive(
                    self._original_name != new_name)

    ## window handlers

    def close_window_cb(self, widget):
        """For unit testing.
        """ 
        # For unit testing.
        if __name__ == '__main__': 
            Gtk.main_quit()

    ## gradient drawarea handlers
    def gradient_drawingarea_button_press_event_cb(self, d_area, event):
        grctrl = self._gradient_controller
        self._pressed_button = event.button
        self.start_x = event.x
        self.start_y = event.y
        self.last_x = self.start_x
        self.last_y = self.start_y
        idx = grctrl.hittest_node(self, event.x, event.y)
        if idx >= 0:
            grctrl.button_press_cb(
                    self,
                    self,
                    event)
            

            curnode = grctrl.nodes[idx]
            self.set_color_to_colorbutton(curnode)

            if idx > 0 and idx < len(grctrl.nodes) - 1:
                # Only intermidiate point should be graggable.
                self._drag_should_start = True
            grctrl.queue_redraw(self)
        return True

    def gradient_drawingarea_button_release_event_cb(self, d_area, event):
        self._pressed_button = None
        grctrl = self._gradient_controller

        if self._drag_should_start:
            # Utilize self._drag_should_start to
            # know whether button event is processed or not
            grctrl.button_release_cb(
                    self,
                    self,
                    event)
        self._drag_should_start = False

        if self.in_drag:
            grctrl.drag_stop_cb(
                    self,
                    self)
            self.invalidate_gradient()
        self.in_drag = False
        return True

    def gradient_drawingarea_motion_notify_event_cb(self, d_area, event):
        grctrl = self._gradient_controller
        if self._drag_should_start:
            self.in_drag = True
            grctrl.drag_start_cb(
                    self,
                    self,
                    event)
            self._drag_should_start = False

        if self.in_drag:
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            grctrl.drag_update_cb(
                    self,
                    self,
                    event,
                    dx, dy
                    )
            self.last_x = event.x
            self.last_y = event.y


        return True

    def gradient_drawingarea_draw_cb(self, d_area, cr):
        d_alloc = d_area.get_allocation()
       #gui.ui_util.draw_background(cr, d_alloc.width, d_alloc.height, 24)
        gui.drawutils.render_checks(cr, 24, 
                d_alloc.width / 24 + 1, d_alloc.height / 24 + 1)
        cr.set_source(self.background_gradient)
        cr.rectangle(0, 0, d_alloc.width, d_alloc.height)
        cr.fill()

        grctrl = self._gradient_controller
        grctrl.paint(cr, self, self)

    def gradient_drawingarea_size_changed_cb(self, d_area, rect):
        grctrl = self._gradient_controller
        assert grctrl is not None
        margin = 8
        new_sx = margin + grctrl._radius
        new_y = (rect.height - grctrl._radius) / 2
        new_ex = (rect.width - margin * 2 - grctrl._radius)

        # In this editor, gradient controller always
        # placed parallel to horizon, and no user interaction
        # enabled for moving/reshaping controller.
        # so, y axis value is always same.

        if (grctrl.start_pos is None
                or grctrl.end_pos is None
                or grctrl.start_pos[0] != new_sx
                or grctrl.end_pos[0] != new_ex):
            # Use self as dummy tdw.
            grctrl.set_start_pos(self, (new_sx, new_y))
            grctrl.set_end_pos(self, (new_ex, new_y))


    ## Other UI widgets handlers
    def currentbrush_checkbutton_toggled_cb(self, button):
        if self._updating_ui:
            return 
        grctrl = self._gradient_controller
        idx = grctrl.current_node_index
        curnode = grctrl.current_node
        new_node = None
        if curnode is not None:
            if button.get_active():
                if isinstance(curnode, GradientInfo):
                    new_node = GradientInfo_Brushcolor(
                            self.app,
                            curnode.linear_pos,
                            curnode.alpha)
            else:
                if isinstance(curnode, GradientInfo_Brushcolor):
                    new_node = GradientInfo(
                            curnode.linear_pos,
                            curnode.color,
                            curnode.alpha)

            if new_node is not None:
                grctrl.nodes[idx] = new_node
                self.set_color_to_colorbutton(new_node)
                self.invalidate_gradient()

    def colorbutton_color_set_cb(self, button):
        if self._updating_ui:
            return 
        grctrl = self._gradient_controller
        col = self.color_button.get_color()
        curnode = grctrl.current_node
        if curnode is not None:
            curnode.set_color(
                    col,
                    self.alpha_adj.get_value())
            self.invalidate_gradient()

    def alpha_adjustment_value_changed_cb(self, button):
        if self._updating_ui:
            return 
        grctrl = self._gradient_controller
        curnode = grctrl.current_node
        if curnode is not None:
            curnode.alpha = self.alpha_adj.get_value()
            self.invalidate_gradient()

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

    class DummyBrushColorManager(object):

        def __init__(self):
            self._color_store = (
                                    lib.color.RGBColor(r=1.0, g=0.0, b=0.0),
                                    lib.color.RGBColor(r=0.0, g=1.0, b=0.0),
                                    lib.color.RGBColor(r=0.0, g=0.0, b=1.0),
                                )

            self._cur_store_idx = 0
            #elf._color = lib.color.RGBColor(r=1.0, g=0.0, b=0.0)

        def get_color(self):
           #self._cur_store_idx += 1
           #self._cur_store_idx %= len(self._color_store)
            return self._color_store[self._cur_store_idx]

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
            self.brush_color_manager = DummyBrushColorManager()

        def find_action(self, arg):
            return Gtk.RecentChooserMenu()

    app = DummyApp()
    return app

def _test():
    """Run interactive unit test, outside the application."""

    logging.basicConfig()

    print("### TEST ###")
    win = GradientEditorWindow()
    win.connect("delete-event", lambda *a: Gtk.main_quit())
    # sample code
    colors = ( 
                (0.0, (1.0, 0.0, 0.0)),
                (0.25, (1.0, 1.0, 0.0)),
                (0.50, (0.0, 1.0, 0.0)),
                (0.75, (0.0, 1.0, 1.0)),
                (1.0, (0.0, 0.0, 1.0))
              )
    win.start('test gradient', colors)
    Gtk.main()


if __name__ == '__main__':
    _test()




