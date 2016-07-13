#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from gettext import gettext as _
from gi.repository import Gtk

from gui.assist import *

class AssistManager(object):
    """ AssistManager, To manage multiple assistants
    such as Stabilizer, Ruler, etc.

    and, with this manager, assistants are blend-mode dependent,
    when you switching to another mode(such as Eraser mode)
    manager remember an assistant which is applied when that blend mode
    enabled, and switch assistants automatically.

    This class is singleton, can be accessed as app.assistmanager
    """

    def __init__(self, app):
        # self._assistants holds singleton instances
        # for each Gtk.Action/ToggleAction name.

        self. app = app
        self._assistants = { "AssistModeStabilizer" : Stabilizer(app),
                             None : None # the default, no assistant enabled.
                }

        self._current = None
        self._current_action_name = None

        self._blend_modes_action={}
        app.brushmodifier.blend_mode_changed += self.blend_mode_changed_cb
        app.brushmanager.brush_selected += self.brush_selected_cb
        self._current_blend = None
        self._internal_update = False
        self._presenter_box = None
        self._empty_box = Gtk.VBox()

    @property
    def current(self):
        return self._current

    def _do_action(self, name, flag):
        if name:
            action = self.app.find_action(name)
            if action:
                self._internal_update = True
                if hasattr(action, "set_active"):
                    action.set_active(flag)
                elif flag:
                    action.activate()
                self._internal_update = False
            else:
                logger.warning('Action %s assigned but not found.' % name)
        else:
            # default action(no assistant)
            pass


    def enable_assistant(self, action_name):
        old = self._current

        assert action_name in self._assistants.keys()
        self._current = self._assistants[action_name]
        self._current_action_name = action_name

        if not self._internal_update:
            self._blend_modes_action[self._current_blend] = action_name

        combo = self._assistant_combo

        if self._current:
            self._current.reset()

            combo_model = combo.get_model()
            for row in combo_model:
                if combo_model.get(row.iter,0)[0] == self._current.name:
                   combo.set_active_iter(row.iter)

            self._activate_presenter(self._current.get_presenter()) 
        else:
            combo.set_active(0)
            self._activate_presenter(None)

        return self._current

    def blend_mode_changed_cb(self, modifier, new_blend):

        if self._current_blend != None:
            old_action_name = self._blend_modes_action.get(self._current_blend, None)
            self._do_action(old_action_name, False)

        new_action_name = self._blend_modes_action.get(new_blend, None)
        self._do_action(new_action_name, True) 
        # From this self._do_action(), Gtk.ToggleAction signalled
        # and self.enable_assistant() would be called from outside this class.

        self._blend_modes_action[new_blend] = new_action_name
        self._current_blend = new_blend

    def brush_selected_cb(self, bm, managed_brush, brushinfo):
        """ Anyway, reset current assistant.
        """
        self._do_action(self._current_action_name, False)

    # Options presenter
    def init_options_presenter_box(self):
        """ Option presenter initialize method.
        This is called from FreehandOptionsWidget.init_specialized_widgets
        of gui/freehand.py 
        """ 
        grid = Gtk.Grid(column_spacing=8, row_spacing=6, 
                hexpand_set=True, hexpand=True, halign=Gtk.Align.FILL)
        grid.margin = 4
        grid.hide()

        label = Gtk.Label()
        label.set_text(_("Assistant:"))
        label.halign = Gtk.Align.START
        grid.attach(label,0,0,1,1)

        combo = Gtk.ComboBoxText(hexpand_set=True, hexpand=True, halign=Gtk.Align.FILL)
        combo.append_text(_("No Assistant")) 
        for cl in self._assistants:
            if cl is not None:
                assistant = self._assistants[cl]
                combo.append_text(assistant.name)
        combo.set_active(0)
        combo.popup_fixed_width = False
        self._assistant_combo = combo
        grid.attach(combo,1,0,1,1)


        # Creating per-assistants option presenter space.
        align = Gtk.Alignment.new(0.5, 0.5, 1.0, 1.0)
        align.set_padding(0, 0, 0, 0)
        align.set_border_width(3)
        align.hexpand = True
        align.halign = Gtk.Align.FILL
        self._assistant_options_bin = align
        grid.attach(align,0,1,2,1)

        self._presenter_grid = grid                             
        grid.show_all()
        return grid

    @property
    def presenter_box(self):
        return self._presenter_grid

    def _activate_presenter(self, widget):
        assert self._presenter_grid != None
        if widget == None:
            widget = self._empty_box

        bin = self._assistant_options_bin
        old_option = bin.get_child()
        if old_option:
            old_option.hide()
            bin.remove(old_option)

        if widget:
            bin.add(widget)
            widget.show()
            bin.show_all()


if __name__ == '__main__':

    pass


