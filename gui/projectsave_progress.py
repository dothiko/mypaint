#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os

import gi
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GObject
from logging import getLogger

class ProjectSave_progress(object):

    def __init__(self, proc, parent=None):
        self._proc = proc
        self._max_task = float(proc.get_work_count())
        self._retflag = True

        builder_xml = os.path.splitext(__file__)[0] + ".glade"
        builder = Gtk.Builder()
        builder.set_translation_domain("mypaint")
        builder.add_from_file(builder_xml)
        builder.connect_signals(self)

        self._progressbar = builder.get_object("tasks_progressbar")

        self._window = builder.get_object("dialog_window")
        self._window.connect('destroy', self._window_destroy_cb)
        self._window.set_size_request(400, -1)
        if parent:
            self._window.set_parent(parent)
            self._window.set_transient_for(parent)
              

    @property
    def processed_percentage(self):
        return (self._max_task - self._proc.get_work_count()) / self._max_task

    def run(self):
        self._window.show()
        self._source_id = GObject.timeout_add_seconds(0.5, self._timer_cb)
        self._mainloop = GObject.MainLoop()
        self._mainloop.run()
        return self._retflag

    def cancel_clicked_cb(self, action):
        self._retflag = False
        if self._source_id != None:
            GObject.source_remove(self._source_id)
        self._window.close()

    def _window_destroy_cb(self, widget):
        self._mainloop.quit()

    def _timer_cb(self):
        self._proc._process()
        self._progressbar.set_fraction(self.processed_percentage)
        self._progressbar.queue_draw()
        if not self._proc.has_work():
            self._source_id = None
            self._window.close()
            return False
        return True


if __name__ == '__main__':

    import lib.idletask
    import time
    proc = lib.idletask.Processor()

    def dummy_work(idx, dummyarg):
        print('work %d start' % idx)
        time.sleep(0.5)
        print('work %d end' % idx)
        return False


    for i in range(10):
        proc.add_work(
                dummy_work,
                i,None)

    def parent_destroy(widget):
        Gtk.main_quit()

    def parent_click(widget, event):
        p = ProjectSave_progress(proc, parent=widget)
        if p.run() == False:
            print('cancelled!')
        else:
            print('ended!')

    parent_win = Gtk.Window(Gtk.WindowType.TOPLEVEL)
    parent_win.set_title("parent window for test projectsave-progress")
    parent_win.set_resizable(False)
    parent_win.show()
    parent_win.connect("destroy", parent_destroy)
    parent_win.connect("button-release-event", parent_click)
    parent_win.set_size_request(500,500)


    Gtk.main()

    pass


