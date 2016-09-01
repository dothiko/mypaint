# This file is part of MyPaint.
# Copyright (C) 2016 dothiko<dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or


## Imports
import math
import array
from lib.helpers import clamp
import logging
from collections import deque
logger = logging.getLogger(__name__)
from gettext import gettext as _

from gi.repository import Gtk, Gdk

import gui.drawutils
import gui.tileddrawwidget
import gui.style
from gui.linemode import *
from gui.ui_utils import *

## Module settings



## Class defs
#
# All Assistants should be derived from Assistbase.
# And it is needed to be registered at the constructor of 
# Assistmanager class (assistmanager.py).

class Assistbase(object):


    def __init__(self):
        self.reset()

    def reset(self):
        self._prev_button = None
        self._last_button = None

    def enum_samples(self, tdw):
        """ Enum current assisted position, from fetched samples.
        This is the most important method of assistant class.
        This method should return a value with yield.
        """
        pass # By the way, this is melely stub.

    def button_press_cb(self, tdw, x, y, pressure, time, button):
        pass
    def button_release_cb(self, tdw, x, y, pressure, time, button):
        pass

    def fetch(self, tdw, x, y, pressure, time, button):
        """Fetch samples.

        Assistants stores these 'raw' input datas into
        its own storage, or some calculate to generate
        new stroke, and freehand tool enumerate 

        :param tdw: Current TiledDrawWidget  
        :param x,y: current cursor position, in display coordinate.
        :param pressure: current pressure of input device.
        :param time: current time of event issued.
        :param button: currently pressed button, which is same as
            event.button of button_pressed_cb().
        
        """
        pass


    def set_active_cb(self, flag):
        """ activated from Gtk.Action """
        pass

    ## Overlay drawing related

    def queue_draw_area(self, tdw):
        """ Queue draw area for overlay """
        pass

    def draw_overlay(self, cr, tdw):
        """ Drawing overlay """
        pass

    ## Options presentor for assistant
    #  Return Gtk.Box type widget ,to fill freehand option presenter.
    def get_presenter(self):
        pass

class Averager(Assistbase):
    """ Averager Stabilizer class, which fetches 
    gtk.event x/y position as a sample,and return 
    the average of recent samples.
    """
    name = _("Averager")
    STABILIZE_START_MAX = 24

    # Averager ring buffer
    _sampling_max = 32
    _samples_x = array.array('f')
    _samples_y = array.array('f')
    _samples_p = array.array('f')
    _sample_index = 0
    _sample_count = 0
    _current_index = 0

    def __init__(self, app):
        if len(Averager._samples_x) < Averager._sampling_max:
            for i in range(Averager._sampling_max):
                Averager._samples_x.append(0.0) 
                Averager._samples_y.append(0.0) 
                Averager._samples_p.append(0.0) 

        super(Stabilizer, self).__init__()
        self._stabilize_cnt = None
        self.app = app

    def enum_samples(self, tdw):
        if self._sample_count < self._sampling_max:
            raise StopIteration

        rx = 0.0
        ry = 0.0
        rp = self._latest_pressure
        idx = 0

        while idx < self._sampling_max:
            rx += self._get_stabilized_x(idx)
            ry += self._get_stabilized_y(idx)
            idx += 1

        rx /= self._sampling_max 
        ry /= self._sampling_max

        self._prev_rx = rx
        self._prev_ry = ry
        self._prev_rp = rp

        # Heading / Trailing glitch workaround
        if self._last_button == 1:
            if (self._prev_button == None):
                self._stabilize_cnt = 0
                yield (rx, ry, 0.0)
                yield (rx, ry, self._get_initial_pressure(rp))
                raise StopIteration
            elif self._stabilize_cnt < self.STABILIZE_START_MAX:
                rp = self._get_initial_pressure(rp)
        elif self._last_button == None:
            if (self._prev_button != None):
                rp = self._get_stabilized_pressure(idx)
                if rp > 0.0:
                    self._prev_button = 1
                    yield (rx, ry, rp)
                    raise StopIteration
                else:
                    yield (rx, ry, 0.0)
                    raise StopIteration

            rp = 0.0

        yield (rx, ry, rp)
        raise StopIteration

    def _get_initial_pressure(self, rp):
        self._stabilize_cnt += 1
        return rp * float(self._stabilize_cnt) / self.STABILIZE_START_MAX

    def reset(self):
        super(Stabilizer, self).reset()
        Averager._sample_index = 0
        Averager._sample_count = 0
        self._prev_rx = None
        self._prev_ry = None
        self._prev_time = None
        self._prev_rp = None
        self._release_time = None
        self._latest_pressure = None
        

    def _get_stabilized_x(self, idx):
        return self._samples_x[self.get_current_index(idx)]
    
    def _get_stabilized_y(self, idx):
        return self._samples_y[self.get_current_index(idx)]

    def _get_stabilized_pressure(self, idx):
        return self._samples_p[self.get_current_index(idx)]

    def get_current_index(self, offset):
        return (self._current_index + offset) % self._sampling_max

    def fetch(self, tdw, x, y, pressure, time, button):
        """Fetch samples"""
        self._latest_pressure = pressure
        self._prev_button = self._last_button
        self._last_button = button
        
        # To reject extreamly slow and near samples
        if self._prev_time == None or time - self._prev_time > 8:
            px = self._get_stabilized_x(0)
            py = self._get_stabilized_x(0)
            if math.hypot(x - px, y - py) > 4:
                super(self.__class__, self).fetch(x, y, pressure, time)
            self._prev_time = time 

       #Assistbase._samples_x[Assistbase._sample_index] = x 
       #Assistbase._samples_y[Assistbase._sample_index] = y 
       #Assistbase._samples_p[Assistbase._sample_index] = pressure 
       #Assistbase._current_index = Assistbase._sample_index
       #Assistbase._sample_index+=1
       #Assistbase._sample_index%=Assistbase._sampling_max
       #Assistbase._sample_count+=1
            
class Stabilizer(Assistbase): 
    """ Stabilizer class, like Krita's one.  This stablizer actually 'average angle'.  
    Actually stroke stopped at (self._cx, self._cy), and when 'real' pointer
    move to outside the stabilize-range circle, a stroke is drawn.
    The drawn stroke have the direction from (self._cx, self._cy) 
    to (self._rx, self._ry) and length from the ridge of stabilize-range circle to 
    (self._rx, self._ry).
    Thus, the stroke is 'pulled' by motion of pointer.
    """ 

    name = _("Stabilizer")
    DRAW_THRESHOLD = 16 # The minimum length of strokes to check auto-enable.
    FRAME_PERIOD = 16.6666 # one frame is 1/60 = 16.6666...ms, for stabilizer.

    def __init__(self, app):
       #super(Stabilizer, self).__init__()
        self.app = app
        self._rx = None
        self._ry = None
        self._average_previous = True
        self._presenter = None
        self._stabilize_range = 48
        self._current_range = self._stabilize_range
        self._last_time = None
        self._auto_adjust_range = True # Auto stabilizer range adjust flag
        self._prev_range = 0.0
        self._cycle = 0L
        self.reset()

    @property
    def _enabled(self):
        return (self._last_button != None and 
                self._prev_range > 0)

    def enum_samples(self, tdw):

        if self._cycle == 1L:
            # Drawing initial pressure, to avoid heading glitch.
            # That value would be 0.0 in normal.
            # However, When auto-range adjust is enabled,
            # drawing stroke is already undergo, so
            # 'initial pressure' would be the pressure of current stylus input,
            # not 0.0.
            # And,after this cycle, proceed normal stabilized stage.
            self._cycle = 2L 
            yield (self._cx , self._cy , self._initial_pressure)
        else:
            # Normal stabilize stage.
            
            if self._last_button == 1:
                cx = self._cx
                cy = self._cy

                dx = self._rx - cx
                dy = self._ry - cy
                cur_length = math.hypot(dx, dy)

                if cur_length <= self._current_range:
                    raise StopIteration

                if self._average_previous:
                    if self._prev_dx != None:
                        dx = (dx + self._prev_dx) / 2.0
                        dy = (dy + self._prev_dy) / 2.0
                    self._prev_dx = dx
                    self._prev_dy = dy

                move_length = cur_length - self._current_range
                mx = (dx / cur_length) * move_length
                my = (dy / cur_length) * move_length

                self._cx = cx + mx
                self._cy = cy + my

                yield (self._cx , self._cy , self._latest_pressure)
                self._cycle += 1L
            
            elif self._last_button == None:
                if self._prev_button != None:
                    if self._latest_pressure > 0.0:
                        # button is released but
                        # still remained some pressure...
                        # rare case,but possible.
                        yield (self._cx, self._cy, self._latest_pressure)

                    # We need this for avoid trailing glitch
                    yield (self._cx, self._cy, 0.0)
                # Always output 0.0 pressure,
                # as normal freehand tool.
                yield (self._rx, self._ry, 0.0) 


        raise StopIteration

    def reset(self):
        super(Stabilizer, self).reset()
        self._cx = None
        self._cy = None
        self._latest_pressure = None
        self._last_button = None
        self._start_time = None
        self._cycle = 0L
        self._initial_pressure = 0.0

    def fetch(self, tdw, x, y, pressure, time, button):
        """ Fetch samples(i.e. current stylus input datas) 
        into attributes

        Explanation of attributes which are used at here:
        
        _rx,_ry == Raw input of pointer, 
                         very 'current' position of input,
                         stroke should be drawn this point.
        _cx, _cy == Current center of drawing stroke radius.
                    They also represents previous end point of stroke.
        """

        self._prev_button = self._last_button
        self._last_button = button
        self._last_time = time
        self._latest_pressure = pressure
        self._rx = x
        self._ry = y

        if self._last_button == 1:
            self._prev_range = self._current_range
            if (self._prev_button == None):
                self._cx = x
                self._cy = y
                self._cycle = 1L
                self._start_time = time
                self._initial_pressure = 0.0
                self._prev_dx = None
                if self._auto_adjust_range:
                    # In auto disable mode, stabilizer disabled by default.
                    self._drawlength = 0
                    self._current_range = 1
                    self._ox = x
                    self._oy = y
                else:
                    self._current_range = self._stabilize_range
                self._prev_range = 0.0
            elif (self._auto_adjust_range and self._start_time != None):
                ctime = time - self._start_time
                self._drawlength += math.hypot(x - self._ox, y - self._oy) 

                if ctime > self.FRAME_PERIOD:
                    speed = self._drawlength / (ctime / self.FRAME_PERIOD)
                    # When drawing time exceeds the threshold timeperiod, 
                    # then calculate the speed of storke per 'time unit'
                    # (16.6666..ms = one frame in 60fps), in pixel.
                    #
                    # And depend on that speed of stroke,
                    # inflate/deflate current stabilize range.
                    # If speed is within 10.0pixel/frame to
                    # 6.0pixel/frame, current stabilize range is sustained.
                    #
                    # The adjusting and threshold values are not theorical,
                    # it is from my feeling and experience at testing.
                    # so, these values might be user-configurable...

                    if speed > 30.0:
                        self._current_range -= 3.0
                    elif speed >= 10.0:
                        self._current_range -= speed / 10.0
                    elif speed < 1.0:
                        self._current_range += 1.0
                    elif speed <= 6.0:
                        self._current_range += 1.0 / speed

                    if self._current_range > self._stabilize_range:
                        self._current_range = self._stabilize_range
                    elif self._current_range < 0:
                        self._current_range = 0

                    # Update current/previous position in every case.
                    self._ox = x
                    self._oy = y
                    self._drawlength = 0
                    self._start_time = time


        elif self._last_button == None:
            if self._prev_button != None:
                if self._auto_adjust_range:
                    self._drawlength = 0
                    self._start_time = None
                    self._cycle = 0L


        

    ## Overlay drawing related

    def queue_draw_area(self, tdw):
        """ Queue draw area for overlay """
        if self._enabled:
            half_rad = int(self._current_range+ 2)
            full_rad = half_rad * 2
            tdw.queue_draw_area(self._cx - half_rad, self._cy - half_rad,
                    full_rad, full_rad)

    @dashedline_wrapper
    def _draw_dashed_circle(self, cr, info):
        x, y, radius = info
        cr.arc( x, y,
                int(radius),
                0.0,
                2*math.pi)

    def draw_overlay(self, cr, tdw):
        """ Drawing overlay """
        if self._enabled and self._current_range > 0:
            self._draw_dashed_circle(cr, 
                    (self._cx, self._cy, self._current_range))

            # XXX Drawing actual stroke point.
            # This should be same size as current brush radius,
            # but it would be a huge workload when the brush is 
            # extremely large.
            # so, currently this is fixed size, only shows the center
            # point of stroke.
            self._draw_dashed_circle(cr, 
                    (self._cx, self._cy, 2))


    ## Options presenter for assistant
    def get_presenter(self):
        if self._presenter == None:
            self._presenter = Optionpresenter_Stabilizer(self)
        return self._presenter.get_box_widget()


class ParallelRuler(Assistbase): 
    """ Parallel Line Ruler.
    """ 

    name = _("Parallel Ruler")

    MODE_DRAW = 0
    MODE_SET_BASE = 1
    MODE_SET_DEST = 2
    MODE_FINALIZE = 3

    def __init__(self, app):
       #super(ParallelRuler, self).__init__()
        self.app = app
        self.reset(True) # Attributes inited in reset(), with hard reset
        self.cnt=0

    @property
    def _ready(self):
        return (self._vx != None and self._px != None)

    def _update_positions(self, tdw, x, y, initial):
        if self._last_button == 1:
            mx, my = tdw.display_to_model(x, y)
            if self._mode == self.MODE_DRAW:
                if initial:
                    self.cnt = 0
                    self._sx, self._sy = mx, my
                    self._px, self._py = mx, my
            elif self._mode == self.MODE_SET_BASE:
                self._bx, self._by = mx, my
            elif self._mode == self.MODE_SET_DEST:
                self._dx, self._dy = mx, my
                self._vx, self._vy = normal(self._bx, self._by, self._dx, self._dy)

            self._cx, self._cy = mx, my

    def button_press_cb(self, tdw, x, y, pressure, time, button):
        self._last_button = button
        self._latest_pressure = pressure
        self._update_positions(tdw, x, y, True)

    def button_release_cb(self, tdw, x, y, pressure, time, button):
        print('released')
        self._last_button = None
        self._px = None
        self._py = None
        if self._mode == self.MODE_DRAW:
            self._mode = self.MODE_FINALIZE
        elif self._mode == self.MODE_SET_BASE:
            print('step dest')
            self._mode = self.MODE_SET_DEST
        elif self._mode == self.MODE_SET_DEST:
            print('step draw')
            self._mode = self.MODE_DRAW

    def enum_samples(self, tdw):

        if self._mode == self.MODE_DRAW:
            if self._ready: 
                if self._last_button != None:
                    if self.cnt == 0:
                        cx, cy = tdw.model_to_display(self._sx, self._sy)
                        yield (cx , cy , 0.0)

                    length = distance(self._cx , self._cy, self._px, self._py)

                    if length > 0:
                        cx = (length * self._vx) + self._sx
                        cy = (length * self._vy) + self._sy
                        self._sx = cx
                        self._sy = cy
                        cx, cy = tdw.model_to_display(cx, cy)
                        yield (cx , cy , self._latest_pressure)
                        self._px, self._py = self._cx, self._cy
                        self.cnt += 1

        elif self._mode == self.MODE_FINALIZE:
            # Finalizing previous stroke.
            cx, cy = tdw.model_to_display(self._sx, self._sy)
            yield (cx , cy , 0.0)
            self._mode = self.MODE_DRAW
            raise StopIteration

        raise StopIteration


    def reset(self, hard_reset = False):
        super(ParallelRuler, self).reset()
        if hard_reset:
            # _bx, _by stores the base point of ruler.
            self._bx = None
            self._by = None
            # _dx, _dy stores the destination point of ruler.
            self._dx = None
            self._dy = None

            # _vx, _vy stores the identity vector of ruler, which is
            # (_dx, _dy) - (_bx, _by) 
            # Each strokes should be parallel against this vector.
            self._vx = None
            self._vy = None

        # Above values should not be soft-reset().

        # However, these attributes used for GUI,
        # actually the ruler uses pre-calculated vector.


        # _px, _py is 'initially device pressed(started) point'
        self._px = None
        self._py = None

        if self._bx == None:
            self._mode = self.MODE_SET_BASE
        elif self._dx == None:
            self._mode = self.MODE_SET_DEST
        else:
            self._mode = self.MODE_DRAW

    def fetch(self, tdw, x, y, pressure, time, button):
        """ Fetch samples(i.e. current stylus input datas) 
        into attributes
        """

        self._last_time = time
        self._latest_pressure = pressure
        self._update_positions(tdw, x, y, False)
        

    ## Overlay drawing related

    def queue_draw_area(self, tdw):
        """ Queue draw area for overlay """
        if self._ready:
            margin = 4
            bx, by = tdw.model_to_display(self._bx, self._by)
            dx, dy = tdw.model_to_display(self._dx, self._dy)

            if bx > dx:
                bx, dx = dx, bx
            if by > dy:
                by, dy = dy, by

            tdw.queue_draw_area(bx - margin, by - margin, 
                    dx - bx + margin + 1, dy - by + margin + 1)

    def _draw_dashed_line(self, cr, info):
        sx, sy, ex, ey = info
        cr.move_to(sx, sy)
        cr.line_to(ex, ey)

    def draw_overlay(self, cr, tdw):
        """ Drawing overlay """
        if self._ready:
            bx, by = tdw.model_to_display(self._bx, self._by)
            dx, dy = tdw.model_to_display(self._dx, self._dy)
            self._draw_dashed_line(cr, 
                    (bx, by, dx, dy))

            if self._mode == self.MODE_SET_BASE:
                color = gui.style.ACTIVE_ITEM_COLOR
            else:
                color = gui.style.EDITABLE_ITEM_COLOR
            gui.drawutils.render_round_floating_color_chip(cr, bx, by, 
                    color, gui.style.DRAGGABLE_POINT_HANDLE_SIZE)

            if self._mode == self.MODE_SET_DEST:
                color = gui.style.ACTIVE_ITEM_COLOR
            else:
                color = gui.style.EDITABLE_ITEM_COLOR
            gui.drawutils.render_round_floating_color_chip(cr, dx, dy, 
                    color, gui.style.DRAGGABLE_POINT_HANDLE_SIZE)
            


    ## Options presenter for assistant
   #def get_presenter(self):
   #    if self._presenter == None:
   #        self._presenter = Optionpresenter_ParallelRuler(self)
   #    return self._presenter.get_box_widget()

## Option presenters for assistants

class _Presenter_Mixin(object):
    """ Base Mixin of assistants option presenter"""

    def force_redraw_overlay(self, area=None):
        for tdw in gui.tileddrawwidget.TiledDrawWidget.get_visible_tdws():
            if area:
                tdw.queue_draw_area(*area)
            else:
                tdw.queue_draw()

class Optionpresenter_Stabilizer(_Presenter_Mixin):
    """ Optionpresenter for Stabilizer assistant.
    """

    def __init__(self, assistant):
        self.assistant = assistant
        self._updating_ui = True
        grid = Gtk.Grid(column_spacing=6, row_spacing=4)
        grid.set_hexpand_set(True)
        row = 0

        def create_slider(row, label, handler, min_adj, max_adj, value, digits=1):
            labelobj = Gtk.Label(halign=Gtk.Align.START)
            labelobj.set_text(label)
            grid.attach(labelobj, 0, row, 1, 1)

            adj = Gtk.Adjustment(value, min_adj, max_adj)
            adj.connect('value-changed', handler)

            scale = Gtk.HScale(hexpand_set=True, hexpand=True, 
                    halign=Gtk.Align.FILL, adjustment=adj, digits=digits)
            scale.set_value_pos(Gtk.PositionType.RIGHT)
            grid.attach(scale, 1, row, 1, 1)
            return scale


        # Scale slider for range circle setting.
        create_slider(row, _("Range:"), self._range_changed_cb,
                32, 64, assistant._stabilize_range)
        row+=1

        # Checkbox for average direction.
        checkbox = Gtk.CheckButton(_("Average direction"),
            hexpand_set=True, hexpand=True, halign=Gtk.Align.FILL)
        checkbox.set_active(assistant._average_previous)
        checkbox.connect('toggled', self._average_toggled_cb)
        grid.attach(checkbox, 0, row, 2, 1)
        row+=1

        # Checkbox for use auto-disable feature.
        checkbox = Gtk.CheckButton(_("Auto range adjust"),
            hexpand_set=True, hexpand=True, halign=Gtk.Align.FILL)
        checkbox.set_active(assistant._auto_adjust_range) 
        checkbox.connect('toggled', self._auto_adjust_range_toggled_cb)
        grid.attach(checkbox,0,row,2,1)
        row+=1

        grid.show_all()
        self._grid = grid
        self._updating_ui = False

    def get_box_widget(self):
        return self._grid

    # Handlers
    def _average_toggled_cb(self, checkbox):
        if not self._updating_ui:
            self.assistant._average_previous = checkbox.get_active()

    def _range_changed_cb(self, adj):
        if not self._updating_ui:
            self.assistant._stabilize_range = adj.get_value()

    def _auto_adjust_range_toggled_cb(self, checkbox):
        if not self._updating_ui:
            flag = checkbox.get_active()
            self.assistant._auto_adjust_range = flag
