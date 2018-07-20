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

from gi.repository import Gdk

import gui.linemode
import gui.tileddrawwidget

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

def get_outmost_area(tdw, sx, sy, ex, ey, margin=0):
    """ Get outmost AREA(not bbox), mostly to get (rotated)displaying area.  
    :rtype: a tuple of area(sx, sy, ex, ey), IN DISPLAY COORDINATE.

    :param tdw: a TiledDrawWidget, this can be None.
                if None, this method does no any coordinate conversion.
    :param sx, sy: start point(the left-top) point of rectangle, in model.
    :param ex, ey: end point(the right-bottom) point of rectangle, in model.
    :param margin: margin of rectangle, in DISPLAY coordinate.
    """

    points = ( (ex, sy), (ex, ey), (sx, ey) )

    if tdw:
        dsx, dsy = tdw.model_to_display(sx, sy)
    else:
        dsx, dsy = sx, sy

    dex, dey = dsx, dsy

    for x, y in points:
        if tdw:
            x, y = tdw.model_to_display(x, y)
        dsx = min(dsx, x)
        dsy = min(dsy, y)
        dex = max(dex, x)
        dey = max(dey, y)

    return (dsx - margin, dsy - margin, 
            dex + margin, dey + margin)

def enum_area_point(sx, sy, ex, ey):
    """
    Enumerate area points clockwise.
    :param sx, sy: left-top of rectangle
    :param ex, ey: right-bottom of rectangle
    :rtype: a tuple, (index, x, y)
    """
    for i, pt in enumerate(((sx, sy), (ex, sy), (ex, ey), (sx, ey))):
        yield (i, pt[0], pt[1])


def get_scroll_delta(event, step=1):
    """
    Get a delta value from scroll-wheel event.

    :param event: event object in scroll_cb
    :param step: scroll step value.default is 1.

    :rtype: a tuple of delta value, (x_delta, y_delta)

    CAUTION: in some environment,scroll_cb() might called twice at same time.
    for example, roll wheel up, then at first Gdk.ScrollDirection.SMOOTH event
    happen,and then Gdk.ScrollDirection.UP event happen.
    But make matters worse, in some environment, Gdk.ScrollDirection.UP would
    not happen...
    so we need some facility to reject such case.
    """

    if event.direction == Gdk.ScrollDirection.UP:
        return (0, -step)
    elif event.direction == Gdk.ScrollDirection.DOWN:
        return (0, step)
    elif event.direction == Gdk.ScrollDirection.LEFT:
        return (-step, 0)
    elif event.direction == Gdk.ScrollDirection.RIGHT:
        return (step, 0)
    elif event.direction == Gdk.ScrollDirection.SMOOTH:
        return (step * event.delta_x, step * event.delta_y)
    else:
        raise NotImplementedError("Unknown scroll direction %s" % str(event.direction))


def force_redraw_overlay(area=None):
    """ Force all tdws to redraw. 
    This function is very useful when you have no any explicit
    target tdw instances, but need to redraw(clear) current displayed 
    overlay by any means.

    :param area: a tuple of (x, y, width, height),
        or None(i.e. entire overlay is redrawn) 
    """
    for tdw in gui.tileddrawwidget.TiledDrawWidget.get_visible_tdws():
        if area:
            tdw.queue_draw_area(*area)
        else:
            tdw.queue_draw()


def display_to_model_distance(tdw, disp_dist):
    """ Convert distance in display coordinate 
    to model coordinate. 
    """
    bx, by = tdw.display_to_model(0, 0)
    lx, ly = tdw.display_to_model(disp_dist, 0)
    return math.hypot(lx - bx, ly - by)

def model_to_display_area(tdw, x, y, w, h):
    """ Convert model-coodinate area into display one.
    """
    sx = x
    sy = y
    ex = x + w - 1
    ey = y + h - 1
    sx, sy = tdw.model_to_display(sx, sy)
    ex, ey = tdw.model_to_display(ex, ey)

    sx , ex = min(sx, ex), max(sx, ex)
    sy , ey = min(sy, ey), max(sy, ey)
    w = ex - sx + 1
    h = ey - sy + 1
    return (sx, sy, w, h)

def setup_round_position(tdw, pos, count, radius):
    """ setup button position,rotating around (x,y)
    with considering tdw viewing area.
    
    :param pos:   a tuple of center position,in display coordinate.
    :param count: maximum button count
    :param radius: the radius of button circle.

    :return : a list of positions.position is a tuple of (x, y) 
    """
    x, y = pos
    button_radius = gui.style.FLOATING_BUTTON_RADIUS
    margin = 1.5 * button_radius
    alloc = tdw.get_allocation()
    view_x0, view_y0 = alloc.x, alloc.y
    view_x1, view_y1 = view_x0+alloc.width, view_y0+alloc.height
    buttons = []

    palette_radius = 64 #gui.style.BUTTON_PALETTE_RADIUS
    area_radius = radius + margin 

    if x + area_radius > view_x1:
        x = view_x1 - area_radius
    elif x - area_radius < view_x0:
        x = view_x0 + area_radius
    
    if y + area_radius > view_y1:
        y = view_y1 - area_radius
    elif y - area_radius < view_y0:
        y = view_y0 + area_radius

    for i in xrange(count):
        rad = (math.pi / count) * 2.0 * i
        dx = - palette_radius * math.sin(rad)
        dy = palette_radius * math.cos(rad)
        buttons.append(x + dx, y - dy) 

    return buttons

## Decorators

def dashedline_wrapper(callable):
    """
    Dashed line decorator.
    """
    def decorated(self, cr, info, width = 1, dash_space = 10):
        """
        The wrapping decorator, to draw dashed line automatically.

        :param self: The self argument,to wrapping class method. 
            When to wrapping a function, simply set self as None.
        :param cr: The cairo context.
        :param info: to pass some information into the callable.
            such as a tuple of point, or rectangle, etc.
        :param width: dashed line width. 
        :param dash_space: dashed dot spacing.

        The wrapped callable should accept arguments (self, cr, info)
        Inside the callable, only setting some path(s) and return, without draw it. 
        """
        cr.save()
        cr.set_source_rgb(0, 0, 0)
        cr.set_line_width(width)
        callable(self, cr, info)
        cr.stroke_preserve()
        cr.set_dash( (dash_space,) )
        cr.set_source_rgb(1, 1, 1)
        cr.stroke()
        cr.restore()
    return decorated

def queue_circular_area(tdw, x, y, r, min_r=16, margin=2):
    """Queue to draw circular area, to decrease redrawing area.
    :param min_r: minimum radius to just queue rectangular area.
    """
    if r <= min_r:
        r += margin
        tdw.queue_draw_area(
            x-r,
            y-r,
            x+r*2,
            y+r*2
        )
        return

    # We would use `Midpoint circle algorithm` for this.
    # Currently I use very easy codes for this, not using that algorithm.
    
    # Deciding block size
    b = r // 4
    m = margin
    if b < 4.0:
        b = r // 3
        # Queue around outer rim area, divide by 6.  
        xs, xe = (r, r-b)
        w = (r + m) * 2
        h = b + m * 2
        h2 = (r - b + m) * 2
        offsets = (
            (-xs, -xs, w, h), 
            (-xs, xe, w, h), 
            (-xs, -xe, h, h2), 
            (xe, -xe, h, h2)
        )
    else:
        # Queue around outer rim area, divide by 8.  
        xs, xe = (r, r-b)
        w = (r + m) * 2
        h = b + m * 2
        offsets = (
            (-xe, -xs, w, h), 
            (-xe, xe, w, h), 
            (-xs, -xe, h, w), 
            (xe, -xe, h, w)
        )

        # Drawing additional corner areas.
        xe2 = r - (b * 2)
        for ox, oy in ((-xe, -xe), (xe2, -xe), (-xe, xe2), (xe2, xe2)):
            tdw.queue_draw_area(x+ox-m, y+oy-m, h, h)

    for ox, oy, w, h in offsets:
        tdw.queue_draw_area(x+ox-m, y+oy-m, w, h)

def dbg_draw_circular_area(cr, x, y, r, min_r=16, margin=2):
    """Queue to draw circular area.
    :param min_r: minimum radius to just queue rectangular area.
    """

    cr.save()
    cr.set_source_rgb(1, 0, 0)
    if r <= min_r:
        r += margin
        cr.rectangle(
            x-r,
            y-r,
            x+r*2,
            y+r*2
        )
        cr.fill()
        cr.restore()
        return

    # We would use `Midpoint circle algorithm` for this.
    # Currently I use very easy codes for this.
    r += margin
    
    # Deciding block size
    b = r // 8
    if b < 4.0:
        b = r // 6
        # Queue around outer rim area, divide by 6.  
        # top
        tx = x - r 
        w = r * 2
        ty = y - r 
        h = b 
        cr.rectangle(tx, ty, w, h)
        # bottom
        ty = y + r - b 
        cr.rectangle(tx, ty, w, h)
        # left
        w = h
        h = (r - b) * 2
        ty = y - r + b
        cr.rectangle(tx, ty, w, h)
        # right
        tx = x + r - b
        cr.rectangle(tx, ty, w, h)
    else:
        # Queue around outer rim area, divide by 8.  

        # top
        tx = x - r + b
        w = (r - b) * 2
        ty = y - r
        h = b 
        cr.rectangle(tx, ty, w, h)
        # bottom
        ty = y + r - b 
        cr.rectangle(tx, ty, w, h)
        # left
        w = h
        h = (r - b) * 2
        tx = x - r 
        ty = y - r + b
        cr.rectangle(tx, ty, w, h)
        # right
        tx = x + r - b
        cr.rectangle(tx, ty, w, h)

        # Queue additional inner areas.
        w = b 
        tx = x - r + b
        ty = y - r + b
        cr.rectangle(tx, ty, w, w)
        tx = x + r - (b * 2)
        cr.rectangle(tx, ty, w, w)
        ty = y + r - (b * 2)
        cr.rectangle(tx, ty, w, w)
        tx = x - r + b
        cr.rectangle(tx, ty, w, w)

    cr.fill()
    cr.restore()

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
        self._ex = x
        self._ey = y

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


