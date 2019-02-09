# This file is part of MyPaint.
# Copyright (C) 2017 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""Freehand drawing modes"""

## Imports
from __future__ import division, print_function

import math
from lib.helpers import clamp
import logging
from collections import deque
logger = logging.getLogger(__name__)
import weakref
import struct

from gettext import gettext as _
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GLib

import numpy as np

import gui.mode
from gui.drawutils import spline_4p
from lib import mypaintlib
import lib.helpers
from . import freehand_assisted
from gui.ui_utils import *
from gui.rulercontroller import *
from gui.linemode import *
import lib.pickable as pickable

N = mypaintlib.TILE_SIZE

## Module settings
class _Phase(object):
    INVALID = -1
    DRAW = 0
    SET_BASE = 1
    SET_DEST = 2
    INIT = 4
    RULER = 5
    JUMP = 6 # For context-lasting feature.

## Class defs

class _Prefs(object):
    """Preference key constants"""
    LASTING_PREF_KEY = "assisted.parallelruler.context_lasting"
    DISTANCE_PREF_KEY = "assisted.parallelruler.context_distance"
    
    # Stroke context lasts when within 1 seconds 
    # from pen stylus detached previously.
    DEFAULT_LASTING_PREF = 1 

    # Stroke context lasts when the distance from
    # previously released position to currently pressed
    # is within this range.
    DEFAULT_DISTANCE_PREF = 32 

class StrokeLastableMixin(object):
    """A mixin to share `context-lasting` feature.
    
    CAUTION: don't forget call some mixin methods
    from specific user-class methods.
    """

    def _init_lastable_mixin(self, app, _pref_class):
        """Call this method from enter method
        """
        prefs = app.preferences
        self.context_lasting = prefs.get(_pref_class.LASTING_PREF_KEY,
                                         _pref_class.DEFAULT_LASTING_PREF)
        self.context_distance = prefs.get(_pref_class.DISTANCE_PREF_KEY,
                                          _pref_class.DEFAULT_DISTANCE_PREF)
        self._previous_release_time = -1
        self._previous_release_pos = None

    def _is_context_lasting(self, tdw, event):
        x, y = tdw.display_to_model(event.x, event.y)
        lasting_time = self.context_lasting * 1000
        dist = 65 # Greater than maximum allowed distance
        rpos = self._previous_release_pos
        if rpos is None:
            return False

        rx, ry = rpos
        dist = math.hypot(rx-event.x, ry-event.y)
        return (event.time - self._previous_release_time < lasting_time
                    and dist < self.context_distance)

    def _update_lastable_mixin_info(self, event):
        """Call this method from drag_end_cb
        """
        self._previous_release_time = event.time
        self._previous_release_pos = (event.x, event.y)

class ParallelFreehandMode (freehand_assisted.AssistedFreehandMode,
                            StrokeLastableMixin,
                            pickable.PickableInfoMixin):
    """Freehand drawing mode with parallel ruler.

    """

    ## Class constants & instance defaults
    ACTION_NAME = 'ParallelFreehandMode'

    _initial_cursor = None
    _level_margin = 0.005 # in radian, practical value.
    _level_margin_rough = 0.03 # in radian, practical value.
    # 'rough' margin is to activate 'snap level' button.

    ## Class variables

    # Level vector. This tuple means x and y of identity vector.
    # If the ruler have completely same angle with this,
    # 'level indicator' would be shown.
    _level_vector = (0.0, 1.0)

    # Common ruler object for each ParallelFreehandMode instance.
    _ruler = None

    # XXX for `node pick`
    # Node type id. This is used for ruler pickable strokemap. 
    # This MUST be unique, 32bit unsigned value.
    # And type_id 0 means `generic namedtuple node`,
    # which is used in inktool.
    type_id = 0x00000002
    # XXX for `node pick` end
 
    ## Initialization

    def __init__(self, ignore_modifiers=True, **args):
        # Ignore the additional arg that flip actions feed us

        cls = self.__class__
        if cls._ruler is None:
            cls._ruler = RulerController(self.app)

        # We need to set initial phase here.
        # Otherwise, self.motion_notify_cb and 
        # reset_assist() would raise exception.
        if self._ruler.is_ready():
            self._phase = _Phase.INIT
            self._update_ruler_vector()
        else:
            self._phase = _Phase.INVALID 
        self._overrided_cursor = None
        
        super(ParallelFreehandMode, self).__init__(**args)

    ## Metadata

    @classmethod
    def get_name(cls):
        return _(u"Freehand Parallel")

    def get_usage(self):
        return _(u"Paint free-form brush strokes with parallel ruler")

    ## Properties

   #def is_ready(self):
   #    return (self._ruler.is_ready() 
   #            and self.last_button is not None)

    @property
    def initial_cursor(self):
        cursors = self.app.cursors
        cls = self.__class__
        if cls._initial_cursor == None:
            cls._initial_cursor = cursors.get_action_cursor(
                self.ACTION_NAME,
                gui.cursor.Name.CROSSHAIR_OPEN_PRECISE,
            )
        return cls._initial_cursor

    ## Mode stack & current mode
    def enter(self, doc, **kwds):
        """Enter freehand mode"""
        super(ParallelFreehandMode, self).enter(doc, **kwds)
        self._ensure_overlay_for_tdw(doc.tdw)
        if self._ruler.is_ready():
            self.queue_draw_ui(doc.tdw)

        self._init_lastable_mixin(doc.app, _Prefs)

    ## Input handlers
    def drag_start_cb(self, tdw, event, pressure):
        self._latest_pressure = pressure
        self.start_x = event.x
        self.start_y = event.y
        self.last_x = event.x
        self.last_y = event.y

        if self._ruler.is_ready():
            node_idx = self._ruler.hittest_node(tdw, event.x, event.y)

            if node_idx is not None:
                # For ruler moving
                self._phase = _Phase.RULER
                self._ruler.button_press_cb(self, tdw, event)
                self._ruler.drag_start_cb(self, tdw, event)
            elif self._phase == _Phase.INIT:
                # For drawing.
                x, y = tdw.display_to_model(event.x, event.y)
                if self._is_context_lasting(tdw, event):
                    self._phase = _Phase.JUMP
                    self._update_positions(x, y, False)
                else:
                    self._update_positions(x, y, True)
                    # Queue empty stroke, to eliminate heading stroke glitch.
                    self.queue_motion(tdw, event.time, 
                                      self._px, self._py,
                                      pressure=0.0)

        elif self._phase == _Phase.INVALID:
            self._phase = _Phase.SET_BASE
            self._ruler.set_start_pos(tdw, (event.x, event.y))
        elif self._phase == _Phase.SET_DEST:
            self._ruler.set_end_pos(tdw, (event.x, event.y))
        else:
            print('other mode %d' % self._phase)

    def drag_update_cb(self, tdw, event, pressure):
        """ motion notify callback for assisted freehand drawing
        :return : boolean flag or None, True to CANCEL entire freehand motion 
                  handler and call motion_notify_cb of super-superclass.

        There is no mouse-hover(buttonless) event happen. 
        it can be detected only motion_notify_cb. 
        """
        if self._phase == _Phase.SET_BASE:
            self._ruler.set_start_pos(tdw, (event.x, event.y))
            self.queue_draw_ui(tdw)
            return True
        elif self._phase == _Phase.SET_DEST:
            self._ruler.set_end_pos(tdw, (event.x, event.y))
            self.queue_draw_ui(tdw)
            return True
        elif self._phase == _Phase.RULER:
            dx = event.x - self.last_x 
            dy = event.y - self.last_y 
            self.last_x = event.x
            self.last_y = event.y
            self._ruler.drag_update_cb(self, tdw, event, dx, dy)

            self.queue_draw_ui(tdw)
            return True

    def motion_notify_cb(self, tdw, event, fakepressure=None):
        if self.last_button is None:
            if self._ruler.is_ready():
                self.queue_draw_ui(tdw)
                self._ruler.update_zone_index(self, tdw, event.x, event.y)
                cursor = self._ruler.update_cursor_cb(tdw)
                if cursor != self._overrided_cursor:
                    tdw.set_override_cursor(cursor)
                self._overrided_cursor = cursor
            else:
                if self._phase == _Phase.INVALID:
                    cursor = self.initial_cursor
                    tdw.set_override_cursor(cursor)
                    self._overrided_cursor = cursor

            # XXX To eliminate heading stroke glitches,
            # we need empty `queue_motion`.
            # Without this, You'll see that one when you start drawing
            # stroke.
            # But, if queue motion even in context still lasting,
            # it cause another glitches around `touched` stroke.
            # Therefore, check `context-lasting` and do empty queueing.
            if not self._is_context_lasting(tdw, event):
                x, y = tdw.display_to_model(event.x, event.y)
                self.queue_motion(tdw, event.time, x, y, pressure=0.0)
            return True
        return super(ParallelFreehandMode, self).motion_notify_cb(
                tdw, event, fakepressure)

    def drag_stop_cb(self, tdw, event):
        if self._phase == _Phase.DRAW:
            # To eliminate trailing stroke glitch.
            self.queue_motion(tdw, event.time, 
                              self._sx, self._sy,
                              pressure=0.0)

            self._update_lastable_mixin_info(event)
            self._phase = _Phase.INIT
        elif self._phase == _Phase.SET_BASE:
            self._phase = _Phase.SET_DEST
        elif self._phase == _Phase.SET_DEST:
            self._phase = _Phase.INIT
            self._update_ruler_vector()
            self.queue_draw_ui(tdw)
        elif self._phase == _Phase.INIT:
            # Initialize mode but nothing done
            # = simply return to initial state.
            self._phase = _Phase.INVALID
        elif self._phase == _Phase.RULER:
            self._ruler.button_release_cb(self, tdw, event)
            self._ruler.drag_stop_cb(self, tdw)
            self._phase = _Phase.INIT
            self._update_ruler_vector()
            self.queue_draw_ui(tdw)

    ## Mode options

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        cls = self.__class__
        if cls._OPTIONS_WIDGET is None:
            widget = ParallelOptionsWidget(self)
            cls._OPTIONS_WIDGET = widget
        else:
            cls._OPTIONS_WIDGET.set_mode(self)
        return cls._OPTIONS_WIDGET

                
    def enum_samples(self, tdw):
       #if not self.is_ready():
       #    raise StopIteration
        if not self._ruler.is_ready():
            raise StopIteration

        assert self.last_button is not None

        # Calculate and reflect current stroking 
        # length and direction.
        length, nx, ny = length_and_normal(self._cx , self._cy, 
                self._px, self._py)
        direction = cross_product(self._vy, -self._vx, nx, ny)

        if self._phase == _Phase.DRAW:
            # All position attributes are in model coordinate.

            # _cx, _cy : current position of stylus
            # _px, _py : previous position of stylus.
            # These Positions are actulally only used for getting
            # length and direction.

            # _sx, _sy : current position of 'stroke'. not stylus.
            # _vx, _vy : Identity vector of ruler direction.
            
            if length > 0:
                if direction > 0.0:
                    length *= -1.0
            
                cx = (length * self._vx) + self._sx
                cy = (length * self._vy) + self._sy
                self._sx = cx
                self._sy = cy
                yield (cx , cy , self._latest_pressure)
                self._px, self._py = self._cx, self._cy

        elif self._phase == _Phase.INIT:
            # At here, we need to eliminate heading (a bit curved)
            # slightly visible stroke.
            # To do it, we need a point which is along ruler
            # but oppsite direction point.

            # Make fake `pre-previous` position and yield it.
            # 4.0 is practically enough length for fake position.
            tmp_length = 4.0 

            if length != 0 and direction < 0.0:
                tmp_length *= -1.0

            px = (tmp_length * self._vx) + self._px
            py = (tmp_length * self._vy) + self._py

            yield (px ,py , 0.0)

            # And then yield `previous` position (but actually 
            # current position at this stage) of stylus.
            yield (self._px , self._py , 0.0)

            self._sx = self._px
            self._sy = self._py

            self._phase = _Phase.DRAW

        elif self._phase == _Phase.JUMP:
            yield (self._sx , self._sy , 0.0)
            self._px = self._cx
            self._py = self._cy

            if length > 0:
                if direction > 0.0:
                    length *= -1.0
            
                cx = (length * self._vx) + self._sx
                cy = (length * self._vy) + self._sy
                self._sx = cx
                self._sy = cy
                yield (cx , cy , 0.0)

            self._phase = _Phase.DRAW

        raise StopIteration

    def reset_assist(self):
        super(ParallelFreehandMode, self).reset_assist()

        # _vx, _vy stores the identity vector of ruler, which is
        # from (_bx, _by) to (_dx, _dy) 
        # Each strokes should be parallel against this vector.
        # This attributes are set in _update_ruler_vector()
        self._vx = None
        self._vy = None

        # _px, _py is 'initially device pressed(started) point'
        # And they are updated each enum_samples() as
        # current end of stroke.
        self._px = None
        self._py = None

        self._ruler.reset()
        if self._ruler.is_ready():
            self._phase = _Phase.INIT
        else:
            self._phase = _Phase.INVALID

        self._overrided_cursor = None

    def fetch(self, tdw, x, y, pressure, time):
        """ Fetch samples(i.e. current stylus input datas) 
        into attributes.
        This method would be called each time motion_notify_cb is called.
        """
        if self.last_button is not None:
            self._last_time = time
            self._latest_pressure = pressure
            self._update_positions(x, y, False)

    ## Overlay related

    def _generate_overlay(self, tdw):
        """Called from OverlayMixin. 
        """
        return _Overlay_Parallel(self, tdw)

    def queue_draw_ui(self, tdw):
        """ Queue draw area for overlay """
        if tdw is None:
            for tdw in self._overlays.keys():
                self.queue_draw_ui(tdw)
            return
        self._ruler.queue_redraw(tdw)

    ## Ruler related

    def _update_positions(self, x, y, starting):
        """ update current positions from pointer position 
        of model coordinate.
        """
        if starting:
            self._px, self._py = x, y

        self._cx, self._cy = x, y

    def _update_ruler_vector(self):
        self._vx, self._vy = self._ruler.identity_vector

    ## Level angle related
    def reset_ruler_level(self):
        cls = self.__class__
        cls._level_vector = (0.0, 1.0)

    def set_ruler_as_level(self):
        """Set current ruler as level 
        """
        cls = self.__class__
        cls._level_vector = self._ruler.identity_vector
        self.queue_draw_ui(None)

    def is_level_or_cross(self):
        """Return constant value to tell whether
        the ruler for this class is level (or cross)
        in rough or (mostly) precise.

        :return: RulerController constants.
                 When the ruler is roughly level,
                 return RulerController.ROUGH_LEVEL
                 ruler is precisely level
                 return RulerController.LEVEL
                 otherwise,
                 return RulerController.NOT_LEVEL 
        """
        if self._ruler.is_ready():
            vx, vy = self._ruler.identity_vector
            lx, ly = self._level_vector
            margin = self._level_margin_rough
            
            # crossed vector is (ly, -lx),
            # not just exchanged values.
            if (self._ruler.is_level(lx, ly, margin) 
                    or self._ruler.is_level(ly, -lx, margin)):
                margin = self._level_margin
                if (self._ruler.is_level(lx, ly, margin) 
                        or self._ruler.is_level(ly, -lx, margin)):
                    return RulerController.LEVEL
                else:
                    return RulerController.ROUGH_LEVEL
        return RulerController.NOT_LEVEL

    def snap_ruler_to_level(self):
        if self._ruler.is_ready():
            lx, ly = self._level_vector
            margin = self._level_margin_rough

            ans = self._ruler.is_level(lx, ly, margin)

            if ans != 0:
                pass
            else:
                ans = self._ruler.is_level(ly, lx, margin)
                if ans != 0:
                    lx, ly = ly, -lx
                else:
                    return

            self.queue_draw_ui(None)

            if ans < 0:
                lx *= -1.0
                ly *= -1.0

            self._ruler.snap(lx, ly)
            self.queue_draw_ui(None)

    # XXX for `info pick`
    # Override brushwork_begin to set ruler information into strokemap. 
    def brushwork_begin(self, model, description=None, abrupt=False,
                        layer=None):
        # XXX Almost copy from gui.mode.BrushworkModeMixin.
        # But using active_brushwork property which is added
        # for OncanvasEditMixin
    
        # Commit any previous work for this model
        brushwork = self._BrushworkModeMixin__active_brushwork
        cmd = brushwork.get(model)
        if cmd is not None:
            self.brushwork_commit(model, abrupt=abrupt)
        # New segment of brushwork
        if layer is None:
            layer_path = model.layer_stack.current_path
        else:
            layer_path = None

        # Make unified pickable-information data
        info = pickable.regularize_info(*self._pack_info())

        # The difference from BrushworkModeMixin is
        # using PickableStrokework, instead of Brushwork.
        cmd = lib.command.PickableStrokework(
            model,
            info,
            layer_path=layer_path,
            description=description,
            abrupt_start=(abrupt or self._BrushworkModeMixin__first_begin),
            layer=layer,
        )
        self._BrushworkModeMixin__first_begin = False
        cmd.__last_pos = None
        brushwork[model] = cmd

    ## Use node pick as ruler pick.
    def _apply_info(self, si, offset): 
        ruler = self._ruler
        info = pickable.extract_info(si.get_info())
        sx, sy, ex, ey = self._unpack_info(info)
        if sx != sx: # i.e. sx is nan
            return 

        # Erase previous ruler, if exists.
        if ruler.is_ready():
            self.queue_draw_ui(None)

        if offset != (0, 0):
            dx, dy = offset
            sx += dx 
            ex += dx 
            sy += dy 
            ey += dy 

        ruler.set_start_pos(None, (sx, sy)) 
        ruler.set_end_pos(None, (ex, ey)) 
        # Important: make self._phase same as right after ruler defined.
        # With this, self._px/_py are initialized at button press.
        self._phase = _Phase.INIT
        self._update_ruler_vector()
        self.queue_draw_ui(None)

    def _match_info(self, infotype):
        return infotype == pickable.Infotype.RULER

    def _unpack_info(self, info):
        field_cnt = 4 
        fmt = ">%dd" % field_cnt
        data_length = field_cnt * 8
        return struct.unpack(fmt, info[:data_length])
                     
    def _pack_info(self):
        ruler = self._ruler
        if ruler.end_pos is None:
            n = float('nan')
            sx, sy, ex, ey = n, n, n, n
        else:
            sx, sy = ruler.start_pos
            ex, ey = ruler.end_pos
        return (struct.pack(">4d", sx, sy, ex, ey),
                    pickable.Infotype.RULER)
    # XXX for `info pick` end

class LastableOptionsMixin(object):
    """A mixin to share codes of Optionspresnter
    around `context-lasting` features.
    
    CAUTION: You need to declare two class constant attributes
             * LASTING_PREF_KEY - app.preferences keyname of `context-lasting`
             * DEFAULT_LASTING_PREF - how many seconds context lasts.
             * DISTANCE_PREF_KEY - keyname of `context-lastable distance`
             * DEFAULT_DISTANCE_PREF - how far strokes within same context,in display.
    It can be just use _Prefs constant class as a Mixin.
    """

    def init_sliders(self, app, row):
        pref = app.preferences
        self._create_slider(
            row,
            _("Context lasting:"), 
            self._lasting_changed_cb,
            pref.get(self.LASTING_PREF_KEY, self.DEFAULT_LASTING_PREF),
            0, 
            3.0 # Maximum 3 seconds
        )
        row += 1

        self._create_slider(
            row,
            _("Allowed distance:"), 
            self._context_distance_changed_cb,
            pref.get(self.DISTANCE_PREF_KEY, self.DEFAULT_DISTANCE_PREF),
            16.0, 
            64.0 # Maximum 64 pixels
        )
        row += 1

        return row

    def _lasting_changed_cb(self, adj, data=None):
        if not self._updating_ui:
            mode = self.mode
            if mode:
                value = adj.get_value()
                self.mode.context_lasting = value
                self.app.preferences[self.LASTING_PREF_KEY] = value

    def _context_distance_changed_cb(self, adj, data=None):
        if not self._updating_ui:
            mode = self.mode
            if mode:
                value = adj.get_value()
                self.mode.context_distance = value
                self.app.preferences[self.DISTANCE_PREF_KEY] = value


class ParallelOptionsWidget (freehand_assisted.AssistantOptionsWidget,
                             LastableOptionsMixin,
                             _Prefs):
    """Configuration widget for freehand mode"""


    def __init__(self, mode):
        super(ParallelOptionsWidget, self).__init__(mode)

    def init_specialized_widgets(self, row):
        self._updating_ui = True
        row = super(ParallelOptionsWidget, self).init_specialized_widgets(row)
    
        # Use mixin method to init `context-lasting` sliders
        # Event callbacks are also used from that mixin.
        row = self.init_sliders(self.app, row)

        button = Gtk.Button(label = _("Snap to level")) 
        button.connect('clicked', self._snap_level_clicked_cb)
        self.attach(button, 1, row, 1, 1)
        row += 1

        button = Gtk.Button(label = _("current as level")) 
        button.connect('clicked', self._current_level_clicked_cb)
        self.attach(button, 1, row, 1, 1)
        row += 1

        button = Gtk.Button(label = _("Clear ruler")) 
        button.connect('clicked', self._reset_clicked_cb)
        self.attach(button, 0, row, 2, 1)
        row += 1
        
        self._updating_ui = False
        return row

    def _reset_clicked_cb(self, button):
        if not self._updating_ui:
            # To discard current(old) overlay.
            mode = self.mode
            if mode:
                mode.queue_draw_ui(None) # To erase.
                mode.reset_ruler_level()
                mode.reset_assist()

    def _snap_level_clicked_cb(self, button):
        if not self._updating_ui:
            # To discard current(old) overlay.
            mode = self.mode
            if mode:
                if mode.is_level_or_cross():
                    mode.snap_ruler_to_level()

    def _current_level_clicked_cb(self, button):
        if not self._updating_ui:
            # To discard current(old) overlay.
            mode = self.mode
            if mode:
                mode.set_ruler_as_level()


class _Overlay_Parallel(gui.overlays.Overlay):
    """Overlay for stabilized freehand mode """

    def __init__(self, mode, tdw):
        super(_Overlay_Parallel, self).__init__()
        self._mode_ref = weakref.ref(mode)
        self._tdw_ref = weakref.ref(tdw)

    def paint(self, cr):
        """Draw brush size to the screen"""
        mode = self._mode_ref()
        if mode is not None:
            tdw = self._tdw_ref()
            assert tdw is not None
            ruler = mode._ruler
            if (ruler.is_ready() or 
                    mode._phase in (_Phase.SET_BASE, 
                                    _Phase.SET_DEST)):
                ruler.paint(cr, tdw, mode.is_level_or_cross())
                # ruler level indicator color is defined at
                # gui/rulercontroller.py

