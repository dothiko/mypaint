#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2016 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

from gui.linearcontroller import *
import gui.style

class RulerNode(object):

    color = gui.style.EDITABLE_ITEM_COLOR

    def __init__(self, linear_pos):
        self.linear_pos = linear_pos


class RulerController(LinearController):
    """RulerController, to manage & show parallel ruler settings
    in freehand_parallel
    """
    ACTION_NAME = "FreehandMode"

    def is_ready(self):
        # Does not use self.nodes in this class.
        return (self._start_pos is not None 
                and self._end_pos is not None)

    def set_start_pos(self, tdw, disp_pos):
        super(RulerController, self).set_start_pos(tdw, disp_pos)
        if len(self.nodes) < 1:
            self.nodes.append(RulerNode(0.0))

    def set_end_pos(self, tdw, disp_pos):
        super(RulerController, self).set_end_pos(tdw, disp_pos)
        if len(self.nodes) < 2:
            self.nodes.append(RulerNode(1.0))

    def _shading_contents(self, cr, mode, tdw):
        # base shading, considering for alpha transparency.
        cr.save()
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


