#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from gui.assist import *

class AssistManager(object):
    """ AssistManager, To manage multiple assistants
    such as Stabilizer, Ruler, etc.

    and, with this manager, assistants are blend-mode dependent,
    when you switching to another mode(such as Eraser mode)
    manager remember an assistant which is applied when that blend mode
    enabled, and switch assistants automatically.
    """

    def __init__(self, app):
        # self._assistants holds singleton instances
        # for each Gtk.Action/ToggleAction name.

        self. app = app
        self._assistants = { "AssistModeStabilizer" : Stabilizer_Krita(app),
                             None : None # the default, no assistant enabled.
                }

        self._current = None
        self._current_action_name = None

        self._blend_modes_action={}
        app.brushmodifier.blend_mode_changed += self.blend_mode_changed_cb
        app.brushmanager.brush_selected += self.brush_selected_cb
        self._current_blend = None
        self._internal_update = False

    def get_current_assistant(self):
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

        if self._current:
            self._current.reset()
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


if __name__ == '__main__':

    pass


