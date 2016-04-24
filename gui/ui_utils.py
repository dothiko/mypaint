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
    2 = left
    3 = down
    -1 = not moved.

    return value is a tuple, (direction, length)

    :param bx, by: origin of movement
    :param cx, cy: current mouse cursor position
    :param margin: the margin of centering
    """

    if bx == cx and by == cy:
        return (-1, 0)

    # Getting angle against straight vertical identity vector.
    # That straight vector is (0.0 , 1.0),so it is downward.
    length, nx, ny = gui.linemode.length_and_normal(bx, by, cx, cy)
    angle = math.acos(ny)  

    if length < margin:
        return (-1, 0)
    

    # direction 0 = up, 1 = right, 2 = left 3 = down
    if angle < math.pi * 0.25:
        direction = 3
    elif angle < math.pi * 0.75:
        direction = 2
    else:
        direction = 0

    if nx > 0.0 and direction == 2:
        direction = 1

    return (direction, length - margin)

if __name__ == '__main__':

    pass


