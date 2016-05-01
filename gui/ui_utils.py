#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# This file is part of MyPaint.
# Copyright (C) 2016 by dothiko <a.t.dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import math

import gui.linemode

def get_drag_direction(bx, by, cx, cy, margin=0):
    """ get mouse drag direction,as
    0 = up
    1 = right
    2 = down
    3 = left
    (i.e. clockwise)
    -1 = not moved.

    return value is a tuple, (direction, length)

    :param bx, by: origin of movement
    :param cx, cy: current mouse cursor position
    :param margin: the margin of centering
    """

    if bx == cx and by == cy:
        return (-1, 0)

    # Getting angle against straight vertical identity vector.
    # That straight vector is (0.0 , 1.0)
    length, nx, ny = gui.linemode.length_and_normal(bx, by, cx, cy)
    angle = math.acos(ny)  

    if length < margin:
        return (-1, 0)
    

    # direction 0 = up, 1 = right, 2=down,3 = left
    if angle < math.pi * 0.25:
        direction = 2
    elif angle < math.pi * 0.75:
        direction = 3
    else:
        direction = 0

    if nx > 0.0 and direction == 3:
        direction = 1

    return (direction, length - margin)

def is_inside_triangle(x, y, triangle):
    """ Check the (x,y) is whether inside the triangle or not.
    from stackoverflow 
    http://stackoverflow.com/questions/2049582/how-to-determine-a-point-in-a-2d-triangle

    :param x: x coordinate of point
    :param y: y coordinate of point
    :param triangle: a tuple of triangle, ( (x0,y0) , (x1, y1), (x2,y2) )
    """
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign((x,y) , triangle[0], triangle[1]) < 0.0
    b2 = sign((x,y) , triangle[1], triangle[2]) < 0.0
    b3 = sign((x,y) , triangle[2], triangle[0]) < 0.0
    return b1 == b2 == b3

## Class defs

class DragOffset(object):

    """ To get dragging offset easily, 
    from any line where cannot get tdw.
    (because this is model coordinate!)
    """ 

    def __init__(self):
        self.reset()

    def start(self, x, y):
        """ start position, in model
        """
        self._sx = x
        self._sy = y

    def end(self, x, y):
        self._ex = x
        self._ey = y

    def get_display_offset(self, tdw):
        sx, sy = tdw.model_to_display(self._sx, self._sy)
        ex, ey = tdw.model_to_display(self._ex, self._ey)
        return (ex - sx, ey - sy)

    def get_model_offset(self):
        return (self._ex - self._sx, self._ey - self._sy)

    def reset(self):
        self._sx = 0 
        self._ex = 0 
        self._sy = 0 
        self._ey = 0 

if __name__ == '__main__':

    pass


