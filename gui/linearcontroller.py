#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2016 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import weakref
import cairo
import math

from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GdkPixbuf

import gui.drawutils
import gui.style 
import gui.linemode

class _LinearPhase:
    INIT_NODE = 0
    MOVE = 1
    MOVE_NODE = 2
    STAY = 10


class LinearController(object):
    """LinearController, base class for managing & showing
    'linear' shaped oncanvas visualize user interface.
    This is intended to be used in something 'linear' information
    editing, such as linear gradient or parallel ruler.

    ABOUT NODES:
        Contained nodes(self.nodes) object should have 
        some attribute (or property) to keep compatibility.

    linear_pos attribute:
        This attribute represents the linear position 
        of a node. Its range is from 0.0(start) to 1.0(end).

    color attribute:
        This attribute is a lib.color.UIColor object,
        for coloring each nodes through the function
        gui.drawutils.render_round_floating_color_chip()
        when drawing.
        Mainly used for gradient controller.
        If not needed this feature, use gui.style.EDITABLE_ITEM_COLOR
        or something like that as class constant attribute,
        to always provide same color value. 
    """

    # Class constants
    MOVING_CURSOR = 0
    MOVING_CURSOR_NAME = gui.cursor.Name.HAND_OPEN

    MOVING_NODE_CURSOR = 1
    MOVING_NODE_CURSOR_NAME = gui.cursor.Name.CROSSHAIR_CLOSED


    # You will need to set ACTION_NAME name as class attribute.
    # This is used for searching cursor from app. 
    #
    # ACTION_NAME = "PolyfillMode"

    def __init__(self, app):
        self.app = app
        self._current_node_index = None
        self._target_node_index = None

        self._radius = 6.0
        self._phase = _LinearPhase.STAY
        self._dx = self._dy = 0
        self._target_pos = None
        self._overlay_ref = None

        self._cursors = {} # GUI cursor cache.
        self._hit_area_index = None # most recent hit area index.

        # With reset() method, initialized some attributes.
        self.reset()

    def set_start_pos(self, tdw, pos):
        """Set gradient start position, from display coordinate.
        :param tdw: TileDrawWidget. if None, pos is in model.
                    Otherwise, pos is in display.
        :param disp_pos: start position for cairo.LinearGradient.
                          if None, used current polygon
                          center X and minimum Y coordinate.
        """
        if tdw is not None:
            self._start_pos = tdw.display_to_model(*pos)
        else:
            self._start_pos = pos

    def set_end_pos(self, tdw, pos):
        """Set gradient end position, from display coordinate.
        :param disp_pos: end position for cairo.LinearGradient.
                          if None, used current polygon
                          center X and maximum Y coordinate.
        """
        if tdw is not None:
            self._end_pos = tdw.display_to_model(*pos)
        else:
            self._end_pos = pos

    @property
    def start_pos(self):
        """Get start pos, in model coordinate."""
        return self._start_pos

    @property
    def end_pos(self):
        """Get end pos, in model coordinate."""
        return self._end_pos


    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, flag):
        self._active = flag


    def is_ready(self):
        return (len(self.nodes) > 0 
                and self._start_pos is not None 
                and self._end_pos is not None)

    def reset(self):
        self._active = False
        self._start_pos = None
        self._end_pos = None
        self.nodes = []

        # self._contents_update flag indicates
        # is there need for updating 'contents'
        # which is held by this linear controller
        # (i.e. gradient or something)
        self._contents_update = False

    # node related method

    def add_control_point_from_display(self, tdw, x, y):
        """Add control point from display coordinate.
        """ 
        pass

    @property
    def current_node_index(self):
        return self._current_node_index

    @property
    def current_node(self):
        if self._current_node_index is not None:
            return self.nodes[self._current_node_index]

    def set_target_node_index(self, idx):
        self._target_node_index = idx
        self._current_node_index = idx

    def clear_target_node_index(self):
        self._target_node_index = None
        # Still remained self._current_node_index
            
    def hittest_node(self, tdw, x, y):
        """Get the index of a point which located at target position.
        :param x, y: target position.
        :return : index of point, or -1 when pointing gradiant gauge line.
                  otherwise, return None.
        """
        if not self.is_ready():
            return None

        r = self._radius

        for i, pt, jx, jy, cx, cy in self._enum_node_position(tdw):
            # do not use pt.x/pt.y, they are only valid for
            # first and last node.
            dist = math.hypot(cx-x, cy-y)
            if dist <= r*2:
                return i


        if len(self.nodes) >= 2:
            sx, sy = tdw.model_to_display(*self._start_pos)
            ex, ey = tdw.model_to_display(*self._end_pos)
            dist_s = math.hypot(x-sx, y-sy)
            dist_e = math.hypot(x-ex, y-ey)

            vx = ex - sx
            vy = ey - sy

            leng, nx, ny = gui.linemode.length_and_normal(sx, sy, ex, ey)

            if dist_s > leng or dist_e > leng:
                # XXX Check the cursor location exceeding controller or not
                # This check is actually not precise, but almost work.
                return None

            vpx = x - sx
            vpy = y - sy

            dp = gui.linemode.dot_product(nx, ny, vpx, vpy)

            hx = vx * dp / leng - vpx
            hy = vy * dp / leng - vpy

            height = math.hypot(hx, hy)

            if height < r:

                return -1
        return None

    def add_offsets(self, offset_x, offset_y):
        if len(self.nodes) >= 2:
            self._dx += offset_x
            self._dy += offset_y

    def finalize_offsets(self, tdw):
        if len(self.nodes) >= 2 and self.is_ready():
            sx, sy = tdw.model_to_display(*self._start_pos)
            self._start_pos = tdw.display_to_model(sx+self._dx, sy+self._dy)
            ex, ey = tdw.model_to_display(*self._end_pos)
            self._end_pos = tdw.display_to_model(ex+self._dx, ey+self._dy)
            self._dx = 0
            self._dy = 0
        
    def _enum_node_position(self, tdw):
        """Enumrate nodes position with a tuple

        :yield : a tuple of (node_index, control_point object, 
                             half_x, half_y,
                             end_x, end_y)
                 end_x/y is same as the current coordinate of the control point.
                 half_x/y used for painting psuedo gradient.
        """

        if not self.is_ready():
            if self._start_pos is not None:
                sx, sy = tdw.model_to_display(*self._start_pos)
                yield (0, None, 0, 0, sx, sy)
            raise StopIteration

        dx = self._dx
        dy = self._dy

        if self._start_pos is not None:
            sx, sy = tdw.model_to_display(*self._start_pos)

        if self._end_pos is not None:
            ex, ey = tdw.model_to_display(*self._end_pos)
            tl, nx, ny = gui.linemode.length_and_normal(
                    sx, sy,
                    ex, ey)

        bx, by = sx + dx, sy + dy


        if len(self.nodes) > 0:
            ppt = self.nodes[0]
            yield (0, ppt, 0, 0, bx, by)

            for i in xrange(1, len(self.nodes)):
                cpt = self.nodes[i]
                ln = tl * cpt.linear_pos - tl * ppt.linear_pos
                cx = bx + (nx * (ln * 0.5)) 
                cy = by + (ny * (ln * 0.5)) 
                ex = bx + nx * ln 
                ey = by + ny * ln 

                yield (i, cpt, cx, cy, ex, ey)

                bx, by = ex, ey
                ppt = cpt

    # GUI related

    def update_zone_index(self, mode, tdw, x, y):
        """Update the zone and target node under a cursor position
        Different from ordinary 'Mode' object method,
        This _update_zone_and_target does not update cursor.

        :param mode: the parent mode instance. currently not used.
                     reserved for future use.
        """
        idx = self.hittest_node(tdw, x, y)
        self._hit_area_index = idx
        return idx

    def update_cursor_cb(self, tdw):
        """Mainly called from PolyfillMode.update_cursor_cb
        :param tdw: not used currently, for future use.
        """
        cursor = None
        if self._hit_area_index == -1:
            cursor = self._get_cursor(self.MOVING_CURSOR,
                                      self.MOVING_CURSOR_NAME)
        elif self._hit_area_index is not None:
            cursor = self._get_cursor(self.MOVING_NODE_CURSOR,
                                      self.MOVING_NODE_CURSOR_NAME)
        return cursor

    def _get_cursor(self, id, name):
        cdict = self._cursors
        if not id in cdict:
            cursors = self.app.cursors
            cdict[id] = cursors.get_action_cursor(
                    self.ACTION_NAME,
                    name
                    )
        return cdict[id]

    # Paint related  
    def paint(self, cr, tdw, mode=None):
        """Paint this controller in cairo context.
        Used from Overlay class.
        
        :param mode: arbitrary mode variable.
                     This is class specific, just passed to
                     _shading_contents method.
        """
        cr.save()
        radius = self._radius
        dx = self._dx
        dy = self._dy

        if self.is_ready():
            cr.set_line_width(radius)

            # I dont use cr.translate() here
            # because, _enum_node_position() refer to self._dx/_dy
            # not only drawing, but also queue area and hit test.
            # so using cr.translate() seems to be meaningless...
            sx, sy = tdw.model_to_display(*self._start_pos)
            ex, ey = tdw.model_to_display(*self._end_pos)

            cr.move_to(sx + dx, sy + dy)
            cr.line_to(ex + dx, ey + dy)
            gui.drawutils.render_drop_shadow(cr)

            # base shading, considering for alpha transparency.
            self._shading_contents(cr, tdw, mode)

        elif self.start_pos is not None:
            x, y = tdw.model_to_display(*self.start_pos)
            gui.drawutils.render_round_floating_color_chip(
                    cr,
                    x, y,
                    gui.style.ACTIVE_ITEM_COLOR,
                    self._radius)

        cr.restore()

    def _shading_contents(self, cr, tdw, mode):
        """Shading its contents.
        :param mode: class-specific arbitrary mode variable.
                     This might be None when this is unused.
        """            
        pass

    def _draw_single_node(self, cr, x, y, i, pt):

        gui.drawutils.render_round_floating_color_chip(
                cr,
                x, y,
                gui.style.EDITABLE_ITEM_COLOR,
                self._radius)

        if self._target_node_index == i or self._current_node_index == i:
            gui.drawutils.render_round_floating_color_chip(
                    cr,
                    x, y,
                    pt.color,
                    self._radius / 2)


    def queue_single_point(self, tdw, x, y, r):
        # -2 & +4 is shadow size.
        tdw.queue_draw_area(x-r-2, y-r-2, r*2+4, r*2+4)

    def queue_redraw(self, tdw):
        r = self._radius + 2

        dx = self._dx
        dy = self._dy

        if self._target_pos:
            x, y = self._target_pos
            self.queue_single_point(tdw, x, y, r)

        sx=sy=None

        if len(self.nodes) > 0:
            for i, pt, cx, cy, ex, ey in self._enum_node_position(tdw):
                self.queue_single_point(tdw, ex, ey, r) 

                if sx is not None: # for nodes after index 1
                    # queue parts of line
                    tsx = min(sx, ex)
                    tsy = min(sy, ey)
                    tdw.queue_draw_area(
                            tsx - r, tsy - r, 
                            abs(ex - sx)+ r*2 + 1, 
                            abs(ey - sy)+ r*2 + 1)

                sx, sy = ex, ey 


    # signal handlers
    def button_press_cb(self, mode, tdw, event):
        x = event.x
        y = event.y
        idx = self.hittest_node(tdw, x, y)
        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK

        if idx >= 0:
            self.set_target_node_index(idx)
            if shift_state: 
                self.refresh_current_color()
                self._phase = _LinearPhase.STAY
            else:
                self._phase = _LinearPhase.MOVE_NODE
            self.queue_redraw(tdw)
        elif idx == -1:
            if shift_state:
                self.add_control_point_from_display(tdw, x, y)
                self._phase = _LinearPhase.STAY
                self.queue_redraw(tdw)
            else:
                self._phase = _LinearPhase.MOVE
        else:
            self._phase = _LinearPhase.INIT_NODE
            self._target_pos = (x, y)

    def button_release_cb(self, mode, tdw, event):
        pass                                                         

    def drag_start_cb(self, mode, tdw, event):
        self._contents_update = False
        self._dx = 0
        self._dy = 0

    def drag_update_cb(self, mode, tdw, event, dx, dy):
        self.queue_redraw(tdw) # to erase
        x = event.x
        y = event.y
        if self._phase == _LinearPhase.INIT_NODE:
            self._target_pos = (x, y)
        elif self._phase == _LinearPhase.MOVE_NODE:
            idx = self._target_node_index
            if idx == 0:
                self._start_pos = tdw.display_to_model(x, y)
                self._contents_update = True
            elif idx == len(self.nodes)-1:
                self._end_pos = tdw.display_to_model(x, y)
                self._contents_update = True
            else:
                sx, sy = tdw.model_to_display(*self._start_pos)
                ex, ey = tdw.model_to_display(*self._end_pos)
                l, bnx, bny = gui.linemode.length_and_normal(
                        sx, sy,
                        ex, ey)

                cl, cnx, cny = gui.linemode.length_and_normal(
                        sx, sy,
                        x, y)
                ecl, cnx, cny = gui.linemode.length_and_normal(
                        ex, ey,
                        x, y)

                # If cursor-to-start_pos length or cursor-to-end_pos
                # length is larger than entire controller length , 
                # it means the current dragging node exceeding start point.
                # It should not happen.
                if l > 0.0 and cl <= l and ecl <= l:
                    min_lpos = self.nodes[idx-1].linear_pos
                    max_lpos = self.nodes[idx+1].linear_pos
                    cl /= l
                    if cl > min_lpos and cl < max_lpos:
                        self._contents_update = True
                        self.nodes[idx].set_linear_pos(cl)
        elif self._phase == _LinearPhase.MOVE:
            self.add_offsets(dx, dy)
        self.queue_redraw(tdw)

    def drag_stop_cb(self, mode, tdw):
        
        self.queue_redraw(tdw) # to erase
        if self._phase == _LinearPhase.INIT_NODE:
            if self._start_pos is None:
                self.set_start_pos(tdw, self._target_pos)
            elif self._end_pos is None:
                self.set_end_pos(tdw, self._target_pos)

            self.invalidate_cairo_gradient()
            self._target_pos = None
        elif self._phase == _LinearPhase.MOVE_NODE:
            self.clear_target_node_index()
        elif self._phase == _LinearPhase.MOVE:
            self.finalize_offsets(tdw)

        self.queue_redraw(tdw)
        self._phase = _LinearPhase.STAY

    # Action callback
    def delete_current_item(self):
        """ Caution: this method does not update any visuals
        because there is no access method for tdw.
        so, caller must update visual of gradient controller.
        """
        pass


if __name__ == '__main__':

    pass


