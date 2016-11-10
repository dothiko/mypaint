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


    def __init__(self, app):
        self.app = app
        self._presenter = None

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

        super(Averager, self).__init__(app)
        self._stabilize_cnt = None

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

    # Mode constants.
    #
    # In Stabilizer, self._mode decides how enum_samples()
    # work. 
    # Usually, it enumerate nothing (MODE_INVALID)
    # so no any strokes drawn.
    # When the device is pressed, self._mode is set as 
    # MODE_INIT to initialize stroke at enum_samples().
    # After that, self._mode is set as MODE_DRAW.
    # and enum_samples() yields some modified device positions,
    # to draw strokes.
    # Finally, when the device is released,self._mode is set
    # as MODE_FINALIZE, to avoid trailing stroke glitches.
    # And self._mode returns its own normal state, i.e. MODE_INVALID.
    
    MODE_INVALID = -1
    MODE_INIT = 0
    MODE_DRAW = 1
    MODE_FINALIZE = 2


    def __init__(self, app):
        super(Stabilizer, self).__init__(app)
        print('construct')
        self._rx = 0.0
        self._ry = 0.0
        self._average_previous = True
        self._stabilize_range = 48
        self._current_range = self._stabilize_range
        self._last_time = None
        self._range_switcher = True # Auto stabilizer range adjust flag
        self._cycle = 0L
        self.reset()

    @property
    def _ready(self):
        return (self._mode == self.MODE_DRAW and
                self._current_range > 0)
                
    def button_press_cb(self, tdw, x, y, pressure, time, button):
        self._last_button = button
        self._latest_pressure = pressure
        self._cx = x
        self._cy = y
        self._cycle = 1L
        self._start_time = time
        self._initial_pressure = 0.0
        self._prev_dx = None
        self._mode = self.MODE_INIT
        if self._range_switcher:
            self._drawlength = 0
            self._current_range = 1
            self._ox = x
            self._oy = y
        else:
            self._current_range = self._stabilize_range
        self._prev_range = 0.0

    def button_release_cb(self, tdw, x, y, pressure, time, button):
        self._last_button = None
        self._mode = self.MODE_FINALIZE
        if self._range_switcher:
            self._drawlength = 0
            self._start_time = None
            self._cycle = 0L

    def enum_samples(self, tdw):

        if self._mode == self.MODE_INIT:
            # Drawing initial pressure, to avoid heading glitch.
            # That value would be 0.0 in normal.
            # However, When auto-range adjust is enabled,
            # drawing stroke is already undergo, so
            # 'initial pressure' would be the pressure of current stylus input,
            # not 0.0.
            # And,after this cycle, proceed normal stabilized stage.
            self._mode = self.MODE_DRAW       
            yield (self._cx , self._cy , self._initial_pressure)        
        elif self._mode == self.MODE_DRAW:
            # Normal stabilize stage.
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

        elif self._mode == self.MODE_FINALIZE:
            if self._latest_pressure > 0.0:
                # button is released but
                # still remained some pressure...
                # rare case,but possible.
                yield (self._cx, self._cy, self._latest_pressure)

            # We need this for avoid trailing glitch
            yield (self._cx, self._cy, 0.0)

            self._mode = self.MODE_INVALID
        else:
            # Set empty stroke, as usual.
            yield (self._rx, self._ry, 0.0)

        raise StopIteration
        
    def reset(self):
        super(Stabilizer, self).reset()
        self._cx = 0.0
        self._cy = 0.0
        self._latest_pressure = None
        self._last_button = None
        self._start_time = None
        self._cycle = 0L
        self._initial_pressure = 0.0
        self._mode = self.MODE_INVALID
        self._speed = 0
        self._stop_cnt = 0

    def fetch(self, tdw, x, y, pressure, time, button):
        """ Fetch samples(i.e. current stylus input datas) 
        into attributes.
        This method would be called each time motion_notify_cb is called.

        Explanation of attributes which are used at here:
        
        _rx,_ry == Raw input of pointer, 
                         the 'current' position of input,
                         stroke should be drawn TO (not from) this point.
        _cx, _cy == Current center of drawing stroke radius.
                    These also represent the previous end point of stroke.
        """

        self._last_time = time
        self._latest_pressure = pressure
        self._rx = x
        self._ry = y

        if self._mode == self.MODE_DRAW:
            if (self._range_switcher and self._start_time != None):
                ctime = time - self._start_time
                self._drawlength += math.hypot(x - self._ox, y - self._oy) 

                if ctime > self.FRAME_PERIOD:
                    speed = self._drawlength / (ctime / self.FRAME_PERIOD)
                    # When drawing time exceeds the threshold timeperiod, 
                    # then calculate the speed of storke.
                    # It is stroke length per 'specific time unit = one frame'
                    # (16.6666..ms = one frame in 60fps), in pixel.
                    #
                    # When the speed below the specfic value,
                    # (currently, it is 0.00001)
                    # it is recognized as 'Pointer Stopped'
                    # and the stopping frame count exceeds certain threshold,
                    # then stabilizer range is expanded.

                    if speed <= 0.00001:
                        # if the style holded over 16 frames,
                        # stabilizer range should be maxed out.
                        self._stop_cnt += 1
                        if self._stop_cnt > 24:
                            self._current_range = self._stabilize_range
                        elif self._stop_cnt > 8:
                            self._current_range = self._stabilize_range / 2

                    self._current_range = max(0, min(self._current_range, 
                        self._stabilize_range))
                        
                    self._speed = speed

                    # Update current/previous position in every case.
                    self._ox = x
                    self._oy = y
                    self._drawlength = 0
                    self._start_time = time


    ## Overlay drawing related

    def queue_draw_area(self, tdw):
        """ Queue draw area for overlay """
        if self._ready:
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
        if self._ready:
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

    MODE_INVALID = -1
    MODE_DRAW = 0
    MODE_SET_BASE = 1
    MODE_SET_DEST = 2
    MODE_FINALIZE = 3
    MODE_INIT = 4

    def __init__(self, app):
        super(ParallelRuler, self).__init__(app)
        self.reset(True) # Attributes inited in reset(), with hard reset

    @property
    def _ready(self):
        return self._vx != None

    def _update_positions(self, tdw, x, y):
        if self._last_button == 1:
            mx, my = tdw.display_to_model(x, y)
            if self._mode == self.MODE_INIT:
                self._sx, self._sy = mx, my
                self._px, self._py = mx, my
            elif self._mode == self.MODE_SET_BASE:
                self._bx, self._by = mx, my
            elif self._mode == self.MODE_SET_DEST:
                self._dx, self._dy = mx, my

            self._cx, self._cy = mx, my

    def button_press_cb(self, tdw, x, y, pressure, time, button):
        self._last_button = button
        self._latest_pressure = pressure

        if self._ready:
            # This line need to be placed prior to
            # calling _update_positions(), because
            # it looks whether the self._mode is 
            # self.MODE_INIT or not.
            self._mode = self.MODE_INIT

        self._update_positions(tdw, x, y)

    def button_release_cb(self, tdw, x, y, pressure, time, button):
        self._last_button = None
        self._px = None
        self._py = None
        if self._mode == self.MODE_DRAW:
            self._mode = self.MODE_FINALIZE
        elif self._mode == self.MODE_SET_BASE:
            self._mode = self.MODE_SET_DEST
        elif self._mode == self.MODE_SET_DEST:
            self._mode = self.MODE_INVALID
            self._vx, self._vy = normal(self._bx, self._by, self._dx, self._dy)
            self.queue_draw_area(tdw)
        elif self._mode == self.MODE_INIT:
            # Initialize mode but nothing done
            # = simply return to initial state.
            self._mode = self.MODE_INVALID

    def enum_samples(self, tdw):

        if self._mode == self.MODE_DRAW:
            if self._ready and self._last_button != None:

                length, nx, ny = length_and_normal(self._cx , self._cy, 
                        self._px, self._py)
                direction = cross_product(self._vy, -self. _vx,
                        nx, ny)

                if length > 0:

                    if direction > 0.0:
                        length *= -1.0

                    cx = (length * self._vx) + self._sx
                    cy = (length * self._vy) + self._sy
                    self._sx = cx
                    self._sy = cy
                    cx, cy = tdw.model_to_display(cx, cy)
                    yield (cx , cy , self._latest_pressure)
                    self._px, self._py = self._cx, self._cy

        elif self._mode == self.MODE_INIT:
            if self._ready and self._last_button != None:
                cx, cy = tdw.model_to_display(self._sx, self._sy)
                yield (cx , cy , 0.0)
                self._mode = self.MODE_DRAW
                self.enum_samples(tdw) # Re-enter this method with another mode

        elif self._mode == self.MODE_FINALIZE:
            # Finalizing previous stroke.
            cx, cy = tdw.model_to_display(self._sx, self._sy)
            yield (cx , cy , 0.0)
            self._mode = self.MODE_INVALID
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
            # from (_bx, _by) to (_dx, _dy) 
            # Each strokes should be parallel against this vector.
            self._vx = None
            self._vy = None

        # Above values should not be 'soft' reset().
        # because reset() called each time device pressed.

        # _px, _py is 'initially device pressed(started) point'
        self._px = None
        self._py = None

        if self._bx == None:
            self._mode = self.MODE_SET_BASE
        elif self._dx == None:
            self._mode = self.MODE_SET_DEST
        else:
            self._mode = self.MODE_INVALID

    def fetch(self, tdw, x, y, pressure, time, button):
        """ Fetch samples(i.e. current stylus input datas) 
        into attributes
        """
        self._last_time = time
        self._latest_pressure = pressure
        self._update_positions(tdw, x, y)

    ## Overlay drawing related
    def _queue_chip_area(self, tdw, x, y, margin):
        x, y = tdw.model_to_display(x, y)
        tdw.queue_draw_area(x - margin, y - margin, 
                x + margin, y + margin)
        return (x, y)

    def queue_draw_area(self, tdw):
        """ Queue draw area for overlay """
        margin = gui.style.DRAGGABLE_POINT_HANDLE_SIZE + 2

        if self._dx != None:
            dx, dy = self._queue_chip_area(tdw, self._dx, self._dy, margin)

        if self._bx != None:
            bx, by = self._queue_chip_area(tdw, self._bx, self._by, margin)

        if self._ready:
            margin = 2

            if bx > dx:
                bx, dx = dx, bx
            if by > dy:
                by, dy = dy, by

            tdw.queue_draw_area(bx - margin, by - margin, 
                    dx - bx + margin + 1, dy - by + margin + 1)


    @dashedline_wrapper
    def _draw_dashed_line(self, cr, info):
        sx, sy, ex, ey = info
        cr.move_to(sx, sy)
        cr.line_to(ex, ey)

    def _draw_floating_chip(self,cr, x, y, flag):
        if flag:
            color = gui.style.ACTIVE_ITEM_COLOR
        else:
            color = gui.style.EDITABLE_ITEM_COLOR
        gui.drawutils.render_round_floating_color_chip(cr, x, y, 
                color, gui.style.DRAGGABLE_POINT_HANDLE_SIZE)

    def draw_overlay(self, cr, tdw):
        """ Drawing overlay """

        # Draw control chips.
        if self._dx != None:
            dx, dy = tdw.model_to_display(self._dx, self._dy)

        if self._bx != None:
            bx, by = tdw.model_to_display(self._bx, self._by)


        # As first, Draw ruler segment.
        if self._ready:
            assert self._dx != None and self._bx != None
            self._draw_dashed_line(cr, 
                    (bx, by, dx, dy))

        # Then, draw floating chips to overwrite 
        # dashed line.
        cr.save()
        if self._dx != None:
            self._draw_floating_chip(cr, dx, dy,
                self._mode == self.MODE_SET_DEST)

        if self._bx != None:
            self._draw_floating_chip(cr, bx, by,
                self._mode == self.MODE_SET_BASE)
        cr.restore()

    ## Options presenter for assistant
    def get_presenter(self):
        if self._presenter == None:
            self._presenter = Optionpresenter_ParallelRuler(self)
        return self._presenter.get_box_widget()


class FocusRuler(ParallelRuler): 
    """ Focus(Convergence) Line Ruler.
    """ 

    name = _("Focus Ruler")

    def __init__(self, app):
        super(FocusRuler, self).__init__(app)

    @property
    def _ready(self):
        return self._bx != None

    def _update_positions(self, tdw, x, y):
        if self._last_button == 1:
            mx, my = tdw.display_to_model(x, y)
            if self._mode == self.MODE_INIT:
                self._sx, self._sy = mx, my
                self._px, self._py = mx, my
                # Here is a difference for ParallelRuler.
                # In 'FocusRuler', identity vector should
                # be set here and it should be calculated
                # each time user draw a stroke.
                self._vx, self._vy = normal(self._bx, self._by, 
                        self._px, self._py)
            elif self._mode == self.MODE_SET_BASE:
                self._bx, self._by = mx, my

            self._cx, self._cy = mx, my

    def button_release_cb(self, tdw, x, y, pressure, time, button):
        self._last_button = None
        self._px = None
        self._py = None
        if self._mode == self.MODE_DRAW:
            self._mode = self.MODE_FINALIZE
        elif self._mode == self.MODE_SET_BASE:
            # Here is a difference for ParallelRuler.
            # In 'FocusRuler', when base point is set,
            # then end ruler setting stage immidiately.
            self._mode = self.MODE_INVALID
            self.queue_draw_area(tdw)
        elif self._mode == self.MODE_INIT:
            # Initialize mode but nothing done
            # = simply return to initial state.
            self._mode = self.MODE_INVALID


    def reset(self, hard_reset = False):
        super(FocusRuler, self).reset(hard_reset)

        # Superclass reset() method should set
        # self._mode, but it refers self._dx/self.MODE_SET_DEST
        # which is not used for FocusRuler.
        # so self._mode set again.
        if self._bx == None:
            self._mode = self.MODE_SET_BASE
        else:
            self._mode = self.MODE_INVALID

    def queue_draw_area(self, tdw):
        """ Queue draw area for overlay """
        margin = gui.style.DRAGGABLE_POINT_HANDLE_SIZE + 2
        if self._bx != None:
            bx, by = self._queue_chip_area(tdw, self._bx, self._by, margin)


    def draw_overlay(self, cr, tdw):
        """ Drawing overlay """


        # Draw control chips.
        if self._bx != None:
            bx, by = tdw.model_to_display(self._bx, self._by)
            self._draw_floating_chip(cr, bx, by,
                self._mode == self.MODE_SET_BASE)


    ## Options presenter for assistant

    # For now, Options presenter is same as ParallelRuler.
   #def get_presenter(self):
   #    if self._presenter == None:
   #        self._presenter = Optionpresenter_ParallelRuler(self)
   #    return self._presenter.get_box_widget()


## Option presenters for assistants

class _Presenter_Mixin(object):
    """ Base Mixin of assistants option presenter"""

    def initialize_start(self, colspace=6, rowspace=4):
        grid = Gtk.Grid(column_spacing=colspace, row_spacing=rowspace)
        grid.set_hexpand_set(True)
        self._grid = grid
        self._updating_ui = True
        self._row = 0
        return grid

    def initialize_end(self):
        self._grid.show_all()
        self._updating_ui = False
        del self._row

    def _attach_grid(self, widget, col=0, width=2):
        self._grid.attach(widget, col, self._row, width, 1)
        self._row += 1


    def get_box_widget(self):
        return self._grid

class Optionpresenter_Stabilizer(_Presenter_Mixin):
    """ Optionpresenter for Stabilizer assistant.
    """

    def __init__(self, assistant):
        self.assistant = assistant

        # Start of initialize widgets
        # With this, internal attributes setupped
        # and some widget setup done automatically.
        self.initialize_start()

        def create_slider(label, handler, min_adj, max_adj, value, digits=1):
            labelobj = Gtk.Label(halign=Gtk.Align.START)
            labelobj.set_text(label)
            self._attach_grid(labelobj, col=0, width=1)

            adj = Gtk.Adjustment(value, min_adj, max_adj)
            adj.connect('value-changed', handler)

            scale = Gtk.HScale(hexpand_set=True, hexpand=True, 
                    halign=Gtk.Align.FILL, adjustment=adj, digits=digits)
            scale.set_value_pos(Gtk.PositionType.RIGHT)
            self._attach_grid(scale, col=1, width=1)
            return scale


        # Scale slider for range circle setting.
        create_slider(_("Range:"), self._range_changed_cb,
                32, 64, assistant._stabilize_range)

        # Checkbox for average direction.
        checkbox = Gtk.CheckButton(_("Average direction"),
            hexpand_set=True, hexpand=True, halign=Gtk.Align.FILL)
        checkbox.set_active(assistant._average_previous)
        checkbox.connect('toggled', self._average_toggled_cb)
        self._attach_grid(checkbox)

        # Checkbox for use 'range switcher' feature.
        checkbox = Gtk.CheckButton(_("Range switcher"),
            hexpand_set=True, hexpand=True, halign=Gtk.Align.FILL)
        checkbox.set_active(assistant._range_switcher) 
        checkbox.connect('toggled', self._range_switcher_toggled_cb)
        self._attach_grid(checkbox)

        # End of initialize widgets
        self.initialize_end()

    # Handlers
    def _average_toggled_cb(self, checkbox):
        if not self._updating_ui:
            self.assistant._average_previous = checkbox.get_active()

    def _range_changed_cb(self, adj):
        if not self._updating_ui:
            self.assistant._stabilize_range = adj.get_value()

    def _range_switcher_toggled_cb(self, checkbox):
        if not self._updating_ui:
            flag = checkbox.get_active()
            self.assistant._range_switcher = flag

class Optionpresenter_ParallelRuler(_Presenter_Mixin):
    """ Optionpresenter for ParallelRuler assistant.
    """

    def __init__(self, assistant):
        self.assistant = assistant
        self.initialize_start()

        # button to force reset ruler
        button = Gtk.Button(label = _("Clear ruler")) 
        button.connect('clicked', self._reset_clicked_cb)
        self._attach_grid(button)

        self.initialize_end()

    # Handlers
    def _reset_clicked_cb(self, button):
        if not self._updating_ui:
            # To discard current(old) overlay.
            force_redraw_overlay() 

            self.assistant.reset(hard_reset=True)

