#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2016 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import math

from gui.linearcontroller import *
import gui.style
from gui.linemode import *

class RulerNode(object):

    color = gui.style.EDITABLE_ITEM_COLOR

    def __init__(self, linear_pos):
        self.linear_pos = linear_pos


class RulerController(LinearController):
    """RulerController, to manage & show parallel ruler settings
    in freehand_parallel
    """
    ACTION_NAME = "FreehandMode"
    _level_color = (0.0, 1.0, 0.5)
    _level_color_rough = (0.7, 1.0, 0.9)

    # level status constants
    LEVEL = 1
    ROUGH_LEVEL = 2
    NOT_LEVEL = 0

    @property
    def identity_vector(self):
        sx, sy = self._start_pos
        ex, ey = self._end_pos
        return normal(sx, sy, ex, ey)

    def is_ready(self):
        # Does not use self.nodes in this class.
        return (self._start_pos is not None 
                and self._end_pos is not None)

    def is_level(self, vx, vy, margin):
        """Return interger value to tell whether 
        this ruler is level (or cross) with
        identity vector (vx, vy).

        :return : interger 1 , 0 or -1.
                  If ruler is level with (vx,vy)
                  and has same direction, return 1.
                  If ruler is level but has opposite
                  direction, return -1.
                  Otherwise, return 0.
        """
        if self.is_ready():
            ix, iy = self.identity_vector
            rad = get_radian(ix, iy, vx, vy)
            if rad < margin:
                return 1
            elif abs(rad - math.pi) < margin:
                return -1
        return 0

    def snap(self, vx, vy):
        """Rotate current ruler vector as assigned identity vector.
        """
        assert self.is_ready()
        sx, sy = self._start_pos
        ex, ey = self._end_pos
        length = vector_length(ex-sx, ey-sy)
        self._end_pos = (sx + length * vx,
                         sy + length * vy)

    def set_start_pos(self, tdw, disp_pos):
        super(RulerController, self).set_start_pos(tdw, disp_pos)
        if len(self.nodes) < 1:
            self.nodes.append(RulerNode(0.0))

    def set_end_pos(self, tdw, disp_pos):
        super(RulerController, self).set_end_pos(tdw, disp_pos)
        if len(self.nodes) < 2:
            self.nodes.append(RulerNode(1.0))

    def _shading_contents(self, cr, tdw, level_status):
        """Shade ruler contents.

        :param level_status: RulerController constants.
        """
        cr.save()
        if level_status == self.LEVEL:
            cr.set_source_rgb(*self._level_color) 
        elif level_status == self.ROUGH_LEVEL:
            cr.set_source_rgb(*self._level_color_rough) 
        else:
            cr.set_source_rgb(1.0, 1.0, 1.0) 
        cr.stroke()

        # Drawing simurated gradiation
        if len(self.nodes) > 0:

            # Drawing nodes.
            # if do this above 'simurated gradiation' loop,
            # some lines overdraw node chips.
            # so do that here to overdraw simulated lines.
            for i, pt, cx, cy, ex, ey in self._enum_node_position(tdw):
                self._draw_single_node(cr, ex, ey, i, pt)

        cr.restore()

if __name__ == '__main__':

    pass


