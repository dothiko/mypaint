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
import weakref

from gi.repository import Gtk, Gdk, GLib

import gui.drawutils
import gui.tileddrawwidget
import gui.style
from gui.linemode import *
from gui.ui_utils import *
from gui.rulercontroller import *

## Module settings



## Class defs
#
# All Assistants should be derived from Assistbase.
# And it is needed to be registered at the constructor of 
# Assistmanager class (assistmanager.py).

class Assistbase(object):
    """ Assistant base class.

    Note: Every assistants are singleton, so there is no need
    to use class attribute.
    """


    def __init__(self, app):
        self.app = weakref.proxy(app)
        self._presenter = None
        self._prev_button = None
        self._last_button = None

    @property
    def options_presenter(self):
        if self._presenter is None:
            self._presenter = self.generate_presenter()
        return self._presenter

    @property
    def last_button(self):
        return self._last_button

    @last_button.setter
    def last_button(self, button):
        self._prev_button = self._last_button
        self._last_button = button

    @property
    def prev_button(self):
        return self._prev_button

    def reset(self):
        """ Called when Completely new starting of assitant. 
        Reset all information, includeing GUI related.
        Basically called once when the assistant activated.
        """
        self.initialize()

    def initialize(self):
        """ Called when Initializing assitant.
        Initilizing information, 
        i.e. called each time mouse(stylus) primary button pressed.
        """
        pass

    def enum_samples(self, tdw):
        """ Enum current assisted position, from fetched samples.
        This is the most important method of assistant class.
        This method should return a value with yield.
        """
        pass 

    ## Event handlers
    #  These callback would call another assitant handler
    #  according to situation.
    #  
    #  In most cases, instead of overriding these methods, 
    #  use drag_*_cb() handler.

    def button_press_cb(self, tdw, event):
        """ Base button press notify callback.
        :return : boolean flag, True to cancel entire freehand motion handler.
                  return value is got from drag_start_cb()
        """
        self.last_button = event.button
        self.initialize()
        if not self.drag_start_cb(tdw, event, 
                                  event.get_axis(Gdk.AxisUse.PRESSURE)):
            return False
        else:
            self.last_button = None
            return True

    def button_release_cb(self, tdw, event):
        if self.last_button is not None:
            self.drag_stop_cb(tdw, event)
            self.last_button = None

    def motion_notify_cb(self, tdw, event):
        """ Base motion notify callback.
        :return : boolean flag, True to cancel entire freehand motion handler.
                  return value is got from drag_update_cb()
        """
        if self.last_button is not None:
            self.queue_draw_area(tdw)
            return self.drag_update_cb(tdw, event, 
                                       event.get_axis(Gdk.AxisUse.PRESSURE))

        return False
                                
    # Event handlers for overrideing.
    # Use these handlers to update GUI(Overlay).
    def drag_start_cb(self, tdw, event, pressure):
        """ motion notify callback for assitant, mainly GUI.
        :return : boolean flag, True to cancel entire freehand motion handler.
        """
        return False

    def drag_update_cb(self, tdw, event, pressure):
        """ motion notify callback for assitant, mainly GUI.
        :return : boolean flag, True to cancel entire freehand motion handler.
        """
        return False

    def drag_stop_cb(self, tdw, event):
        pass

    # Other handlers  
    def set_active_cb(self, flag):
        """ Called when activated from Gtk.Action """
        pass

    def fetch(self, x, y, pressure, time):
        """ Fetching(storing) current stylus information, 
        to generate assisted information at enum_samples method.

        This method should be nothing to do with GUI, simply store/calculate
        internal informations.
        """
        pass

    ## Overlay drawing related

    def queue_draw_area(self, tdw):
        """ Queue draw area for overlay """
        pass

    def draw_overlay(self, cr, tdw):
        """ Drawing overlay """
        pass

    ## Options presentor for assistant
    def generate_presenter(self):
        """ Generate additional option presenter widget for assistant.
        This method called from property 'options_presenter'
        and that property handler set generated Gtk.Box(or something
        gtk container widget) into self._presenter attribute.

        :return A Gtk container widget which contains options presenter
                for assistant.
        """
        return None

#class Averager(Assistbase):
#    """ Averager Stabilizer class, which fetches 
#    gtk.event x/y position as a sample,and return 
#    the average of recent samples.
#    """
#    name = _("Averager")
#    STABILIZE_START_MAX = 24
#
#    # Averager ring buffer
#    _sampling_max = 32
#    _samples_x = array.array('f')
#    _samples_y = array.array('f')
#    _samples_p = array.array('f')
#    _sample_index = 0
#    _sample_count = 0
#    _current_index = 0
#
#    def __init__(self, app):
#        if len(Averager._samples_x) < Averager._sampling_max:
#            for i in range(Averager._sampling_max):
#                Averager._samples_x.append(0.0) 
#                Averager._samples_y.append(0.0) 
#                Averager._samples_p.append(0.0) 
#
#        super(Averager, self).__init__(app)
#        self._stabilize_cnt = None
#
#    def enum_samples(self, tdw):
#        if self._sample_count < self._sampling_max:
#            raise StopIteration
#
#        rx = 0.0
#        ry = 0.0
#        rp = self._latest_pressure
#        idx = 0
#
#        while idx < self._sampling_max:
#            rx += self._get_stabilized_x(idx)
#            ry += self._get_stabilized_y(idx)
#            idx += 1
#
#        rx /= self._sampling_max 
#        ry /= self._sampling_max
#
#        self._prev_rx = rx
#        self._prev_ry = ry
#        self._prev_rp = rp
#
#        # Heading / Trailing glitch workaround
#        if self._last_button == 1:
#            if (self._prev_button is None):
#                self._stabilize_cnt = 0
#                yield (rx, ry, 0.0)
#                yield (rx, ry, self._get_initial_pressure(rp))
#                raise StopIteration
#            elif self._stabilize_cnt < self.STABILIZE_START_MAX:
#                rp = self._get_initial_pressure(rp)
#        elif self._last_button is None:
#            if (self._prev_button is not None):
#                rp = self._get_stabilized_pressure(idx)
#                if rp > 0.0:
#                    self._prev_button = 1
#                    yield (rx, ry, rp)
#                    raise StopIteration
#                else:
#                    yield (rx, ry, 0.0)
#                    raise StopIteration
#
#            rp = 0.0
#
#        yield (rx, ry, rp)
#        raise StopIteration
#
#    def _get_initial_pressure(self, rp):
#        self._stabilize_cnt += 1
#        return rp * float(self._stabilize_cnt) / self.STABILIZE_START_MAX
#
#    def initialize(self):
#        Averager._sample_index = 0
#        Averager._sample_count = 0
#        self._prev_rx = None
#        self._prev_ry = None
#        self._prev_time = None
#        self._prev_rp = None
#        self._release_time = None
#        self._latest_pressure = None
#
#    def _get_stabilized_x(self, idx):
#        return self._samples_x[self.get_current_index(idx)]
#    
#    def _get_stabilized_y(self, idx):
#        return self._samples_y[self.get_current_index(idx)]
#
#    def _get_stabilized_pressure(self, idx):
#        return self._samples_p[self.get_current_index(idx)]
#
#    def get_current_index(self, offset):
#        return (self._current_index + offset) % self._sampling_max
#
#    def fetch(self, x, y, pressure, time):
#        """Fetch samples"""
#        self._latest_pressure = pressure
#        self._prev_button = self._last_button
#        self._last_button = button
#        
#        # To reject extreamly slow and near samples
#        if self._prev_time is None or time - self._prev_time > 8:
#            px = self._get_stabilized_x(0)
#            py = self._get_stabilized_x(0)
#            if math.hypot(x - px, y - py) > 4:
#                super(self.__class__, self).fetch(x, y, pressure, time)
#            self._prev_time = time 

            
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

    _AVERAGE_PREF_KEY = "assistant.stabilizer.average_previous"
    _STABILIZE_RANGE_KEY = "assistant.stabilizer.range"
    _RANGE_SWITCHER_KEY = "assistant.stabilizer.range_switcher"
    _FORCE_TAPERING_KEY = "assistant.stabilizer.force_tapering"

    _TAPERING_LENGTH = 32.0 
    _TIMER_PERIOD = 500.0
    _TIMER_PERIOD_2ND = 700.0


    def __init__(self, app):
        super(Stabilizer, self).__init__(app)

        pref = self.app.preferences
        self._average_previous = pref.get(self._AVERAGE_PREF_KEY, True)
        self._stabilize_range = pref.get(self._STABILIZE_RANGE_KEY, 48)
        self._range_switcher = pref.get(self._RANGE_SWITCHER_KEY, True)
        self._force_tapering = pref.get(self._FORCE_TAPERING_KEY, False)

        self.reset()

    @property
    def _ready(self):
        return (self._mode in (self.MODE_DRAW, self.MODE_INIT) and
                self._current_range > 0)

    @property
    def average_previous(self):
        return self._average_previous
    
    @average_previous.setter
    def average_previous(self, flag):
        self._average_previous = flag
        self.app.preferences[self._AVERAGE_PREF_KEY] = flag

    @property
    def stabilize_range(self):
        return self._stabilize_range
    
    @stabilize_range.setter
    def stabilize_range(self, value):
        self._stabilize_range = value
        if self._current_range > value:
            self._current_range = value
        self.app.preferences[self._STABILIZE_RANGE_KEY] = value

    @property
    def range_switcher(self):
        return self._range_switcher
    
    @range_switcher.setter
    def range_switcher(self, flag):
        self._range_switcher = flag
        self.app.preferences[self._RANGE_SWITCHER_KEY] = flag

    @property
    def force_tapering(self):
        return self._force_tapering

    @force_tapering.setter
    def force_tapering(self, flag):
        self._force_tapering = flag
        self.app.preferences[self._FORCE_TAPERING_KEY] = flag
                
    # Signal Handlers

   #def button_press_cb(self, tdw, x, y, pressure, time, button):
    def drag_start_cb(self, tdw, event, pressure):
        self._tdw = tdw
        self._start_time = event.time
        self._last_time = event.time
        self._rx = event.x
        self._ry = event.y

        if self._range_switcher:
            self._start_range_timer(self._TIMER_PERIOD)

   #def button_release_cb(self, tdw, x, y, pressure, time, button):
    def drag_stop_cb(self, tdw, event):
        self._stop_range_timer() # Call this first, before attributes
                                 # invalidated.
        self._mode = self.MODE_FINALIZE

        if self._range_switcher:
            self._drawlength = 0
            self._total_drag_length = 0
            self._current_range = 0
            self._cycle = 0L

        # After stop the timer, invalidate cached tdw.
        self._tdw = None

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

            # cur_length is the distance from
            # center (start) point to current cursor.
            # On the other hand, self._total_drag_length
            # is total motion length of cursor.
            # These 2 variables are totally different.
            if self._ox is not None:
                tx = self._rx - self._ox
                ty = self._ry - self._oy
                self._total_drag_length += math.hypot(tx, ty)
            else:
                self._total_drag_length += cur_length

            self._ox = self._rx
            self._oy = self._ry

            if cur_length <= self._current_range:
                raise StopIteration

            if (self._current_range > 0.0 and 
                    self._average_previous):
                if self._prev_dx is not None:
                    dx = (dx + self._prev_dx) / 2.0
                    dy = (dy + self._prev_dy) / 2.0
                self._prev_dx = dx
                self._prev_dy = dy

            move_length = cur_length - self._current_range
            mx = (dx / cur_length) * move_length
            my = (dy / cur_length) * move_length

            self._actual_drawn_length += move_length
            
            self._cx = cx + mx
            self._cy = cy + my

            if (self._force_tapering and 
                    self._actual_drawn_length < self._TAPERING_LENGTH):
                adj = min(1.0, self._actual_drawn_length / self._TAPERING_LENGTH)
                yield (self._cx , self._cy , self._latest_pressure * adj)
            else:
                yield (self._cx , self._cy , self._latest_pressure)

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
        # We need special initialization for resetting this class
        # So do not supercall
        self._initialize_attrs(self.MODE_INVALID,
                               0.0, 0.0, 0.0)

    def initialize(self):
        self._initialize_attrs(self.MODE_INIT,
                               self._rx, self._ry, 0.0)

    def _initialize_attrs(self, mode, x, y, pressure):
        self._latest_pressure = pressure
        self._cx = x
        self._cy = y
        self._ox = None
        self._initial_pressure = 0.0
        self._prev_dx = None
        self._mode = mode
        self._actual_drawn_length = 0.0
        self._total_drag_length = 0.0
        if self._range_switcher:
            self._drawlength = 0
            self._current_range = 0.0
        else:
            self._current_range = self._stabilize_range
        self._timer_id = None

    def fetch(self, x, y, pressure, time):
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
    def generate_presenter(self):
        """ Because assitant class is used as singleton,
        this method is called only once.
        """
        return Optionpresenter_Stabilizer(self)

    ## Stabilizer Range switcher related
    def _switcher_timer_cb(self):

        # In some case, although timer stopped by 
        # GLib.remove_id(), but handler still invoked.
        # for such case, reject invalid timer loop at here.
        if self.last_button is None:
            return False

        ctime = self._last_time - self._start_time
        drawlength = self._total_drag_length - self._drawlength
        try:
            # Even check whether drawlength > 0.0
            # exception raised, so try-except used.
            speed = drawlength / ctime 
        except ZeroDivisionError:
            speed = 0.0

        # 'cont_timer' is a flag, used for checking stroke speed and
        # also returning value of callback,to continue/stop current timer
        cont_timer = (speed >= 0.003) 

        # When the drawing speed below the specfic value,
        # (currently, it is 0.003 --- i.e. 3px per second)
        # it is recognized as 'Pointer stands still'
        # then stabilizer range is expanded.

        if cont_timer == False:
            #  First of all, current timer  
            #  so current _timer_id should be invalidated.
            self._timer_id = None

            half_range = self._stabilize_range / 2
            if self._current_range < half_range:
                self._mode = self.MODE_INIT
                self._current_range = half_range
                if self._latest_pressure > 0.95:
                    # immidiately enter 2nd stage
                    self._current_range = self._stabilize_range
                elif self._latest_pressure > 0.9:
                    self._start_range_timer(self._TIMER_PERIOD_2ND * 0.5)
                else:
                    self._start_range_timer(self._TIMER_PERIOD_2ND)
            else:
                self._current_range = self._stabilize_range

            assert self._tdw is not None
            self.queue_draw_area(self._tdw)
        else:
            # When right after entering 2nd stage
            # and speed is faster than threshold
            # (i.e. user drawing fast stroke right now),
            # reset the timer period to ordinary one,of 1st stage.
            # With this, we can avoid unintentioal stabilizer range
            # expansion at 2nd stage.
            if (self._current_range > 0 and
                    self._timer_period < self._TIMER_PERIOD):
                self._start_range_timer(self._TIMER_PERIOD)
                cont_timer = False

        self._current_range = clamp(self._current_range,
                0, self._stabilize_range)

        # Update current/previous position in every case.
        self._drawlength = self._total_drag_length
        self._start_time = self._last_time

        return cont_timer

    def _start_range_timer(self, period):
        assert self.last_button != None
        self._stop_range_timer()
        self._timer_id = GLib.timeout_add(period,
                self._switcher_timer_cb)
        self._timer_period = period

    def _stop_range_timer(self):
        if self._timer_id:
            GLib.source_remove(self._timer_id)
            self._timer_id = None
            self._timer_period = 0

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
    MODE_RULER = 5

    def __init__(self, app):
        super(ParallelRuler, self).__init__(app)
        self._ruler = RulerController(app)
        self.reset() 

    def is_ready(self):
        return (self._ruler.is_ready() 
                and self.last_button is not None)

    def _update_positions(self, dx, dy, pressing):
        """ update current positions from pointer position 
        of display coordinate.
        """
        assert self._tdw is not None
        mpos = self._tdw.display_to_model(dx, dy)

        if pressing:
            if self._mode == self.MODE_INIT:
                self._sx, self._sy = mpos
            self._px, self._py = mpos

        self._cx, self._cy = mpos

    def _update_ruler_vector(self):
        sx, sy = self._ruler.start_pos
        ex, ey = self._ruler.end_pos
        self._vx, self._vy = normal(sx, sy, ex, ey)

   #def button_press_cb(self, tdw, x, y, pressure, time, button):
    def drag_start_cb(self, tdw, event, pressure):
        self._tdw = tdw
        self._latest_pressure = pressure
        self.start_x = event.x
        self.start_y = event.y
        self.last_x = event.x
        self.last_y = event.y

        if self._ruler.is_ready():
            node_idx = self._ruler.hittest_node(tdw, event.x, event.y)

            if node_idx is not None:
                # For ruler moving
                self._mode = self.MODE_RULER
                self._ruler.button_press_cb(self, tdw, event)
                self._ruler.drag_start_cb(self, tdw, event)
            else:
                # For drawing.
                self._update_positions(event.x, event.y, True)

        elif self._mode == self.MODE_INVALID:
            self._mode = self.MODE_SET_BASE
            self._ruler.set_start_pos(tdw, (event.x, event.y))
        elif self._mode == self.MODE_SET_DEST:
            self._ruler.set_end_pos(tdw, (event.x, event.y))
        else:
            print('other mode %d' % self._mode)

    def drag_stop_cb(self, tdw, event):
        if self._mode == self.MODE_DRAW:
            self._mode = self.MODE_FINALIZE
        elif self._mode == self.MODE_SET_BASE:
            self._mode = self.MODE_SET_DEST
        elif self._mode == self.MODE_SET_DEST:
            self._mode = self.MODE_INIT
            self._update_ruler_vector()
            self.queue_draw_area(tdw)
        elif self._mode == self.MODE_INIT:
            # Initialize mode but nothing done
            # = simply return to initial state.
            self._mode = self.MODE_INVALID
        elif self._mode == self.MODE_RULER:
            self._ruler.button_release_cb(self, tdw, event)
            self._ruler.drag_stop_cb(self, tdw)
            self._mode = self.MODE_INIT
            self._update_ruler_vector()
            self.queue_draw_area(tdw)

        self._tdw = None

    def drag_update_cb(self, tdw, event, pressure):

        self.queue_draw_area(tdw)

        if self._mode == self.MODE_SET_BASE:
            self._ruler.set_start_pos(tdw, (event.x, event.y))
            self.queue_draw_area(tdw)
            return True
        elif self._mode == self.MODE_SET_DEST:
            self._ruler.set_end_pos(tdw, (event.x, event.y))
            self.queue_draw_area(tdw)
            return True
        elif self._mode == self.MODE_RULER:
            dx = event.x - self.last_x 
            dy = event.y - self.last_y 
            self.last_x = event.x
            self.last_y = event.y
            self._ruler.drag_update_cb(self, tdw, event, dx, dy)

    def motion_notify_cb(self, tdw, event):
        """ Base motion notify callback.
        :return : boolean flag, True to cancel entire freehand motion handler.
                  return value is got from drag_update_cb()
        """
        if self.last_button is None and self._ruler.is_ready():
            self._ruler.update_zone_index(self, tdw, event.x, event.y)
            cursor = self._ruler.update_cursor_cb(tdw)
            if cursor != self._overrided_cursor:
                tdw.set_override_cursor(cursor)
            self._overrided_cursor = cursor

        return super(ParallelRuler, self).motion_notify_cb(tdw, event)

    def enum_samples(self, tdw):

        if self._mode == self.MODE_DRAW:
            if self.is_ready():
                # All position attributes are in model coordinate.

                # _cx, _cy : current position of stylus
                # _px, _py : previous position of stylus.
                # _sx, _sy : start position of 'stroke'. not stylus.
                # _vx, _vy : Identity vector of ruler direction.


                # Calculate and reflect current stroking 
                # length and direction.
                length, nx, ny = length_and_normal(self._cx , self._cy, 
                        self._px, self._py)
                direction = cross_product(self._vy, -self._vx,
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
            if self._ruler.is_ready() and self.last_button is not None:
                cx, cy = tdw.model_to_display(self._sx, self._sy)
                yield (cx , cy , 0.0)
                self._mode = self.MODE_DRAW
                self.enum_samples(tdw) # Re-enter this method with another mode

        elif self._mode == self.MODE_FINALIZE:
            # Finalizing previous stroke.
            cx, cy = tdw.model_to_display(self._sx, self._sy)
            yield (cx , cy , 0.0)
            self._mode = self.MODE_INIT
            raise StopIteration

        raise StopIteration

    def reset(self):
        super(ParallelRuler, self).reset()

        # _vx, _vy stores the identity vector of ruler, which is
        # from (_bx, _by) to (_dx, _dy) 
        # Each strokes should be parallel against this vector.
        # This attributes are set in _update_ruler_vector()
        self._vx = None
        self._vy = None
        self._ruler.reset()

        if self._ruler.is_ready():
            self._mode = self.MODE_INIT
        else:
            self._mode = self.MODE_INVALID

        self._overrided_cursor = None

    def initialize(self):
        self._tdw = None
        # _px, _py is 'initially device pressed(started) point'
        self._px = None
        self._py = None

    def fetch(self, x, y, pressure, time):
        """ Fetch samples(i.e. current stylus input datas) 
        into attributes
        """
        if self._tdw is not None:
            self._last_time = time
            self._latest_pressure = pressure
            self._update_positions(x, y, False)

    ## Overlay drawing related
    def queue_draw_area(self, tdw):
        """ Queue draw area for overlay """
        self._ruler.queue_redraw(tdw)

    def draw_overlay(self, cr, tdw):
        """ Drawing overlay """
        self._ruler.paint(cr, None, tdw)
    
    ## Options presenter for assistant
    def generate_presenter(self):
        return Optionpresenter_ParallelRuler(self)


class FocusRuler(ParallelRuler): 
    """ Focus(Convergence) Line Ruler.
    """ 

    name = _("Focus Ruler")

    def __init__(self, app):
        super(FocusRuler, self).__init__(app)

    @property
    def _ready(self):
        return self._bx is not None

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
        super(FocusRuler, self).reset()
        self._bx = None
        self._by = None

        # Superclass reset() method should set
        # self._mode, but it refers self._dx/self.MODE_SET_DEST
        # which is not used for FocusRuler.
        # so self._mode set again.
        if self._bx is None:
            self._mode = self.MODE_SET_BASE
        else:
            self._mode = self.MODE_INVALID

    def queue_draw_area(self, tdw):
        """ Queue draw area for overlay """
        margin = gui.style.DRAGGABLE_POINT_HANDLE_SIZE + 2
        if self._bx is not None:
            bx, by = self._queue_chip_area(tdw, self._bx, self._by, margin)


    def draw_overlay(self, cr, tdw):
        """ Drawing overlay """


        # Draw control chips.
        if self._bx is not None:
            bx, by = tdw.model_to_display(self._bx, self._by)
            self._draw_floating_chip(cr, bx, by,
                self._mode == self.MODE_SET_BASE)


    ## Options presenter for assistant

    # For now, Options presenter is same as ParallelRuler.

class EasyLiner(Assistbase): 
    """ easyliner class, it is easily draw line with hand.

    This assistant simply follows the average of initial motion of
    cursor.
    """ 

    name = _("EasyLiner")

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

    _STABILIZE_RANGE_KEY = "assistant.easyliner.range"
    _RANGE_SWITCHER_KEY = "assistant.easyliner.range_switcher"
    _FORCE_TAPERING_KEY = "assistant.easyliner.force_tapering"

    _TIMER_PERIOD = 600.0


    def __init__(self, app):
        super(EasyLiner, self).__init__(app)

        pref = self.app.preferences
        self._stabilize_range = pref.get(self._STABILIZE_RANGE_KEY, 48)
        self._range_switcher = pref.get(self._RANGE_SWITCHER_KEY, False)
        self._force_tapering = pref.get(self._FORCE_TAPERING_KEY, False)

        self.reset()

    @property
    def _ready(self):
        return (self._mode in (self.MODE_DRAW, self.MODE_INIT) and
                self._current_range > 0)


    @property
    def stabilize_range(self):
        return self._stabilize_range
    
    @stabilize_range.setter
    def stabilize_range(self, value):
        self._stabilize_range = value
        if self._current_range > value:
            self._current_range = value
       #self.app.preferences[self._STABILIZE_RANGE_KEY] = value

    @property
    def range_switcher(self):
        return self._range_switcher
    
    @range_switcher.setter
    def range_switcher(self, flag):
        self._range_switcher = flag
       #self.app.preferences[self._RANGE_SWITCHER_KEY] = flag

    @property
    def force_tapering(self):
        return self._force_tapering

    @force_tapering.setter
    def force_tapering(self, flag):
        self._force_tapering = flag
   #    self.app.preferences[self._FORCE_TAPERING_KEY] = flag

   #def _update_positions(self, tdw, x, y):
   #    if self._last_button == 1:
   #        mx, my = tdw.display_to_model(x, y)
   #        if self._mode == self.MODE_INIT:
   #            self._sx, self._sy = mx, my
   #            self._px, self._py = mx, my
   #        elif self._mode == self.MODE_SET_BASE:
   #            self._bx, self._by = mx, my
   #        elif self._mode == self.MODE_SET_DEST:
   #            self._dx, self._dy = mx, my
   #
   #        self._cx, self._cy = mx, my
                
    # Signal Handlers

   #def button_press_cb(self, tdw, x, y, pressure, time, button):
    def drag_start_cb(self, tdw, event, pressure):
        self._tdw = tdw

        if self._range_switcher:
            self._start_range_timer(self._TIMER_PERIOD)

   #def button_release_cb(self, tdw, x, y, pressure, time, button):
    def drag_stop_cb(self, tdw, event):
        self._last_button = None
        self._mode = self.MODE_FINALIZE
        if self._range_switcher:
            self._drawlength = 0
            self._start_time = None
            self._cycle = 0L

        self._stop_range_timer()
        # After stop the timer, invalidate cached tdw.
        self._tdw = None

    def enum_samples(self, tdw):

        if self._mode == self.MODE_INIT:
            # Normal stabilize stage.

            cx = self._cx
            cy = self._cy

            dx = self._rx - cx
            dy = self._ry - cy
            cur_length = math.hypot(dx, dy)

            if cur_length <= self._current_range:
                if self._prev_dx is not None:
                    dx = (dx + self._prev_dx) / 2.0
                    dy = (dy + self._prev_dy) / 2.0
                self._prev_dx = dx
                self._prev_dy = dy
                self._nx, self._ny = normal(0.0, 0.0,
                        self._prev_dx, self._prev_dy)
                # Fallthrough
            else:
                self._nx, self._ny = normal(0.0, 0.0,
                        self._prev_dx, self._prev_dy)
                self._mode = self.MODE_DRAW
                # _px, _py is past drawn x & y,
                # _tx, _ty is past raw input x & y
                self._px = self._cx + self._nx * cur_length
                self._py = self._cy + self._ny * cur_length
                self._tx = self._rx
                self._ty = self._ry


                yield (self._cx , self._cy , self._initial_pressure)        
                yield (self._tx , self._ty , self._initial_pressure)        

        elif self._mode == self.MODE_DRAW:

           #dx = self._rx - self._tx
           #dy = self._ry - self._ty
           #cur_length = math.hypot(dx, dy)
            cur_length, jx, jy = length_and_normal(self._tx, self._ty,
                        self._rx, self._ry)


            direction = cross_product(self._ny, -self. _nx,
                   jx, jy)

           #if cur_length > 4:
           #    cur_length = 4
           #direction = cross_product(self._vy, -self. _vx,
           #        nx, ny)
           #
            if direction < 0.0:
                cur_length *= -1.0

            self._px += self._nx * cur_length
            self._py += self._ny * cur_length
            self._actual_drawn_length += cur_length

            if (self._force_tapering and 
                    self._actual_drawn_length < self._TAPERING_LENGTH):
                adj = min(1.0, self._actual_drawn_length / self._TAPERING_LENGTH)
                yield (self._px ,self._py ,self._latest_pressure * adj)
            else:
                yield (self._px ,self._py ,self._latest_pressure)



            self._tx = self._rx
            self._ty = self._ry

        elif self._mode == self.MODE_FINALIZE:

            if self._latest_pressure > 0.0:
                # button is released but
                # still remained some pressure...
                # rare case,but possible.
                yield (self._px, self._py, self._latest_pressure)

            # We need this for avoid trailing glitch
            yield (self._px, self._py, 0.0)

            self._mode = self.MODE_INVALID
        else:
            # Set empty stroke, as usual.
            yield (self._rx, self._ry, 0.0)

        raise StopIteration
        
    def reset(self):
        super(EasyLiner, self).reset()
        self._initialize_attrs(self.MODE_INVALID,
                0.0, 0.0, 0.0,
                None,
                0)

    def _initialize_attrs(self, mode, x, y, pressure, time, button):
        self._last_button = button
        self._latest_pressure = pressure
        self._cx = x
        self._cy = y
        self._rx = x
        self._ry = y
        self._tx = x
        self._ty = y
        self._start_time = time
        self._initial_pressure = 0.0
        self._prev_dx = None
        self._mode = mode
        self._actual_drawn_length = 0.0
       #if tdw:
       #    self._update_positions(tdw, x, y)
        if self._range_switcher:
            self._drawlength = 0
            self._current_range = 0.0
        else:
            self._current_range = self._stabilize_range
        self._timer_id = None

    def fetch(self, x, y, pressure, time):
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

    @dashedline_wrapper
    def _draw_dashed_vector(self, cr, info):
        sx, sy, ex, ey = info
        cr.move_to(sx, sy)
        cr.line_to(ex, ey)

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

            self._draw_dashed_vector(cr, 
                    (self._cx, self._cy, 
                        self._cx + self._nx * self._current_range,
                        self._cy + self._ny * self._current_range)
                    )
                        
                    
    ## Options presenter for assistant
    def generate_presenter(self):
       #if self._presenter is None:
       #    self._presenter = Optionpresenter_Stabilizer(self)
       #return self._presenter.get_box_widget()
        return None

    ## Stabiliezr Range switcher related
    def _switcher_timer_cb(self):
        ctime = self._last_time - self._start_time
        drawlength = self._actual_drawn_length - self._drawlength

        try:
            # Even check whether drawlength > 0.0
            # exception raised, so try-except used.
            speed = drawlength / ctime 
        except ZeroDivisionError:
            speed = 0.0

        # When drawing time exceeds the threshold timeperiod, 
        # then calculate the speed of storke.
        #
        # When the speed below the specfic value,
        # (currently, it is 0.001 --- i.e. 1px per second)
        # it is recognized as 'Pointer Stopped'
        # and the stopping frame count exceeds certain threshold,
        # then stabilizer range is expanded.

        if speed <= 0.001:
            half_range = self._stabilize_range / 2
            if self._current_range < half_range:
                print('timer comes')
                self._mode = self.MODE_INIT
                self._current_range = half_range
                if self._latest_pressure > 0.7:
                    self._start_range_timer(self._TIMER_PERIOD * 0.75)
                else:
                    self._start_range_timer(self._TIMER_PERIOD)
            else:
                self._current_range = self._stabilize_range
                self._stop_range_timer()

            assert self._tdw is not None
            self.queue_draw_area(self._tdw)
        else:
            self._start_range_timer(self._TIMER_PERIOD)

        self._current_range = max(0, min(self._current_range, 
            self._stabilize_range))

        # Update current/previous position in every case.
        self._drawlength = self._actual_drawn_length
        self._start_time = self._last_time

    def _start_range_timer(self, period):
        self._stop_range_timer()
        self._timer_id = GLib.timeout_add(period,
                self._switcher_timer_cb)

    def _stop_range_timer(self):
        if self._timer_id:
            GLib.source_remove(self._timer_id)
            self._timer_id = None


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
                32, 64, assistant.stabilize_range)

        # Checkbox for average direction.
        checkbox = Gtk.CheckButton(_("Average direction"),
            hexpand_set=True, hexpand=True, halign=Gtk.Align.FILL)
        checkbox.set_active(assistant.average_previous)
        checkbox.connect('toggled', self._average_toggled_cb)
        self._attach_grid(checkbox)

        # Checkbox for use 'range switcher' feature.
        checkbox = Gtk.CheckButton(_("Range switcher"),
            hexpand_set=True, hexpand=True, halign=Gtk.Align.FILL)
        checkbox.set_active(assistant.range_switcher) 
        checkbox.connect('toggled', self._range_switcher_toggled_cb)
        self._attach_grid(checkbox)

        # Checkbox for use 'Force start tapering' feature.
        checkbox = Gtk.CheckButton(_("Force start tapering"),
            hexpand_set=True, hexpand=True, halign=Gtk.Align.FILL)
        checkbox.set_active(assistant.force_tapering) 
        checkbox.connect('toggled', self._force_tapering_toggled_cb)
        self._attach_grid(checkbox)

        # End of initialize widgets
        self.initialize_end()

    # Handlers
    def _average_toggled_cb(self, checkbox):
        if not self._updating_ui:
            self.assistant.average_previous = checkbox.get_active()

    def _range_changed_cb(self, adj):
        if not self._updating_ui:
            self.assistant.stabilize_range = adj.get_value()

    def _range_switcher_toggled_cb(self, checkbox):
        if not self._updating_ui:
            self.assistant.range_switcher = checkbox.get_active()

    def _force_tapering_toggled_cb(self, checkbox):
        if not self._updating_ui:
            self.assistant.force_tapering = checkbox.get_active()

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

