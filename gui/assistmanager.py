#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2016 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from gettext import gettext as _
from gi.repository import Gtk

from gui.assist import *
from gui.ui_utils import *

class AssistManager(object):
    """ AssistManager is a singlton, to manage multiple assistants
    such as Stabilizer, Ruler, etc.

    With this manager, assistants become brush dependent.
    When you switching to another brush,
    the manager remember whether the assistant is enabled for the brush or not
    and enable/disable assistant automatically.

    This class is singleton, and this can be accessed as a property of app, 
    i.e. app.assistmanager.
    """

    def __init__(self, app):
        # self._assistants holds singleton instances
        # for each Gtk.Action/ToggleAction name.

        self. app = app

        # Register the assistants.
        # Key is Gtk.Action name , not the assistants name attribute.
        # The reason to use Gtk.Action name is that
        # assistant name might be translated for i13n. 
        self._assistants = { "AssistModeStabilizer" : Stabilizer(app),
                             "AssistModeParallelRuler" : ParallelRuler(app),
                             "AssistModeFocusRuler" : FocusRuler(app),
                             "AssistModeEasyLiner" : EasyLiner(app),
                             None : None # the default, no assistant enabled.
                }

        self._current = None

        self._blend_modes_action={}
        app.brushmanager.brush_selected += self.brush_selected_cb
        self._current_blend = None
        self._empty_box = Gtk.VBox()
        self._brushlookup = {}
        self._presenter_grid = None
        self._assistant_combo = None

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, assistant):
        force_redraw_overlay() # To clear old overlay contents.

        self._current = assistant

        # When options presenter is not displayed by default,
        # assitant_combo is not generated yet. so generate it now.
        # After this, when presenter_grid property accessed from
        # another line, that method just returns already generated
        # grid widget immidiately.
        if self._assistant_combo is None:
            grid = self.presenter_grid # generate with this
        combo = self._assistant_combo

        binfo = self.app.doc.model.brush.brushinfo
        brush_name = binfo.get_string_property("parent_brush_name")

        if assistant == None:
            combo.set_active(0)
            self._activate_presenter(None)
        else:
            self._current.reset()
            combo_model = combo.get_model()
            for row in combo_model:
                if combo_model.get(row.iter,0)[0] == self._current.name:
                   combo.set_active_iter(row.iter)
            self._activate_presenter(self._current.options_presenter) 

        self._brushlookup[brush_name] = assistant

    @property
    def presenter_grid(self):
        """Getting assistants Option presenter property.
        With this method, Gtk.Alignment(self._assistant_options_bin) has been 
        created, and Options-presenter of each assistant classes are shown
        inside that Gtk.Alignment.

        This should be called from FreehandOptionsWidget.init_specialized_widgets
        of gui/freehand.py 
        """ 
        if self._presenter_grid is not None:
            return self._presenter_grid

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
        combo.connect("changed", self.assistant_combo_changed_cb)
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

    def get_assistant_from_label(self, label):
        """ Get assistant from label, i.e. 'combobox' text == assistant.name
        not 'Gtk.Action' name.
        :rtype tuple: the tuple of (action name, assistant)
        """
        for item in self._assistants.items():
            action_name, assistant = item
            if assistant and assistant.name == label:
                return item

        return (None, None)

    def enable_assistant(self, action_name):
        """ Enable assistant.
        
        :param action_name: the Gtk.Action name of assistant. 
        if this is None, assistant disabled.
        """
        assert action_name in self._assistants.keys()

        # With setting up current assistant through the property
        # "current", internal _brushlookup list also updated.
        self.current = self._assistants[action_name]

    def brush_selected_cb(self, bm, managed_brush, brushinfo):
        """ Anyway, reset current assistant.
        """
        if managed_brush.name in self._brushlookup:
            assistant = self._brushlookup[managed_brush.name]
        else:
            assistant = None
        self.current = assistant

    ## Presenter codes

    @property
    def presenter_box(self):
        return self._presenter_grid

    def _activate_presenter(self, presenter):
        """
        :param presenter: assist.Optionpresenter_* class instance.
        """
        assert self._presenter_grid is not None
        if presenter is None:
            widget = self._empty_box
        else:
            widget = presenter.get_box_widget()

        bin = self._assistant_options_bin
        old_option = bin.get_child()
        if old_option:
            old_option.hide()
            bin.remove(old_option)

        if widget:
            bin.add(widget)
            widget.show()
            bin.show_all()

    ## Presenter widget handlers
    def assistant_combo_changed_cb(self, widget):
        action_name, assistant = \
                self.get_assistant_from_label(widget.get_active_text())
        self.current = assistant

if __name__ == '__main__':

    pass


