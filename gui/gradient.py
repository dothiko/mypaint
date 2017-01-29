#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import weakref
import cairo
import math

from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GdkPixbuf

import gui.drawutils
import gui.style 
import gui.linemode
import lib.color
from gui.polyfillshape import _EditZone

class _GradientPhase:
    INIT_NODE = 0
    MOVE = 1
    MOVE_NODE = 2
    STAY = 10

class GradientInfo(object):
    """ This class contains colors as lib.color.RGBColor object.
    Thats because to keep compatibility to drawutils.py functions.
    With storing lib.color.RGBColor object, we can use that color
    directly when drawing overlay graphics with drawutils.
    """

    def __init__(self, linear_pos, color, alpha=1.0): 
        """
        :param linear_pos: gradient linear position. inside from 0.0 to 1.0.
        :param color:lib.color object, or tuple
        :param alpha:float value from 0.0 to 1.0, as alpha component.
        """
        self._lpos = linear_pos # Gradient linear position.
        self._alpha = alpha  # Place this line here, 
                             # prior to self._color setup
        self._gdk_color = None

        if color:
            self.set_color(color, alpha)

    def set_linear_pos(self, linear_pos):
        self._lpos = linear_pos # Gradient linear position.

    @property
    def linear_pos(self):
        return self._lpos

    @property
    def color(self):
        return self._color

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    def get_rgba(self):
        """get a rgba tuple, converted inner represents.
        This is for Cairo gradient, not for GradientStore
        or coping data into another GradientController.

        Because, this is 'Converted' data, so this might
        lose some special information such as 
        'use current brush color' etc.
        """
        col = self.color # Use property, to enable overriding
        return (col.r, col.g, col.b, self._alpha)

    def get_raw_data(self):
        """get a raw color data, as a tuple.
        For this class, same as get_rgba().
        But other class, such as GradientInfo_Brushcolor,
        it is completely dedicated method.
        """
        return self.get_rgba()

    def set_color(self, color, alpha=1.0):
        """Use lib.color.RGBColor object 
        for compatibility to drawutils.py functions. """

        self._gdk_color = None
        if type(color) == tuple or type(color) == list:
            if len(color) == 3:
                self._color = lib.color.RGBColor(rgb=color)
                self._alpha = alpha
            elif len(color) == 4:
                self._color = lib.color.RGBColor(rgb=color[:3])
                self._alpha = color[3]
            else:
                raise ValueError
        elif isinstance(color, lib.color.RGBColor):
            self._color = color # assume it as color object 
            self._alpha = alpha
        elif isinstance(color, Gdk.Color):
            self._color = lib.color.RGBColor(
                    r = color.red / 65535.0,
                    g = color.green / 65535.0,
                    b = color.blue / 65535.0)
            self._alpha = alpha
        else:
            raise ValueError("only accepts tuple,list or lib.color.RGBColor.")

    @property
    def gdk_color(self):
        if self._gdk_color is None:
            col = self.color
            self._gdk_color = Gdk.Color(col.r * 65535, 
                                        col.g * 65535, 
                                        col.b * 65535)
        return self._gdk_color

class GradientInfo_Brushcolor(GradientInfo):
    """ A gradient color class, to Automatically follow current brush color.
    """

    def __init__(self, app, linear_pos, alpha=1.0):
        self.app = app
        super(GradientInfo_Brushcolor, self).__init__(linear_pos, None, alpha)

    @property
    def color(self):
        return self.app.brush_color_manager.get_color()

    def set_color(self, color, alpha=1.0):
        self._alpha = alpha

    def get_raw_data(self):
        """get a raw color data, as a tuple."""
        return (-1, self._alpha)

class GradientController(object):
    """GradientController, to manage & show gradient oncanvas.
    This cannot be Mode object, because we need to show
    current gradient setting in polyfilltool.
    """

    # Class constants
    MOVING_CURSOR = 0
    MOVING_CURSOR_NAME = gui.cursor.Name.HAND_OPEN

    MOVING_NODE_CURSOR = 1
    MOVING_NODE_CURSOR_NAME = gui.cursor.Name.CROSSHAIR_CLOSED


    # Action name (need to get cursor from self.app)
    #
    # Actually, this is not 'mode' class object.
    # And this object should be used only from PolyfillMode.
    # so, use the name "PolyfillMode"
    ACTION_NAME = "PolyfillMode"

    def __init__(self, app):
        self.app = app
        self._current_node_index = None
        self._target_node_index = None
        self._active = False
        self.nodes = []
        self.invalidate_cairo_gradient()

        self._radius = 6.0
        self._phase = _GradientPhase.STAY
        self._dx = self._dy = 0
        self._start_pos = None
        self._end_pos = None
        self._target_pos = None
        self._overlay_ref = None

        self._follow_brushcolor = False
        self._prev_brushcolor = None

        self._cursors = {} # GUI cursor cache.
        self._hit_area_index = None # most recent hit area index.

    def setup_gradient(self, datas):
        """Setup gradient colors with a sequence.

        This method clears all nodes information and 
        controller positions.

        :param colors: a sequence of linear position and color tuple.
                       (linear_position, (r, g, b, a))

        """
        self.nodes = []
        self._follow_brushcolor = False
        self._prev_brushcolor = None

        for i, data in enumerate(datas):
            pos, color = data
            alpha = 1.0
            if pos is not None:
                curpos = max(min(pos, 1.0), 0.0)
            else:
                curpos = max(min(float(i) / (len(datas)-1), 1.0), 0.0)

            assert color[0] is not None

            if color[0] == -1:
                # Current color gradient.
                self._follow_brushcolor = True
                self._prev_brushcolor = self.get_current_color()
                if len(color) == 2:
                    alpha = color[1]
                self.nodes.append(
                        GradientInfo_Brushcolor(self.app, curpos, alpha)
                        )
            elif color[0] == -2:
                raise NotImplementedError("There is no background color for mypaint!")
            else:
                assert len(color) >= 3
                self.nodes.append(
                        GradientInfo(curpos, color, alpha)
                        )

        self.invalidate_cairo_gradient()

    def set_start_pos(self, tdw, disp_pos):
        """Set gradient start position, from display coordinate.
        :param disp_pos: start position for cairo.LinearGradient.
                          if None, used current polygon
                          center X and minimum Y coordinate.
        """
        self._start_pos = tdw.display_to_model(*disp_pos)

    def set_end_pos(self, tdw, disp_pos):
        """Set gradient end position, from display coordinate.
        :param disp_pos: end position for cairo.LinearGradient.
                          if None, used current polygon
                          center X and maximum Y coordinate.
        """
        self._end_pos = tdw.display_to_model(*disp_pos)

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


    # Color/gradient related

    def get_current_color(self):
        """Utility method.
        For future change, to add alpha component to color tuple.
        (yet implemented)
        Use lib.color.RGBColor object directly
        for compatibility to drawutils.py functions.
        """
        return self.app.brush_color_manager.get_color()

    def refresh_current_color(self):
        """Refresh selected(current) node with current brushcolor"""
        if self._current_node_index is not None:
            pt = self.nodes[self._current_node_index]
            pt.set_color(self.get_current_color())
            self.invalidate_cairo_gradient() 

    def follow_brushcolor(self):
        """This method would be called from brush color change observer
        of PolyfillMode.
        If current gradient has setting to follow the brush color change
        ,do it.
        
        :rtype : boolean
        :return : True when we need update overlay visual.
                  On the other hand, cairo gradient update notification is 
                  done internally,so no need to care it.
                  We need 'tdw' to update overlay, so avoid to write that
                  code here.
        """
        if self._follow_brushcolor:
            nc = self.get_current_color()
            oc = self._prev_brushcolor
            if (oc is None or nc.r != oc.r or nc.g != oc.g or nc.b != oc.b):
                self.invalidate_cairo_gradient() 
                return True

    def get_cairo_gradient(self, tdw):
        """Get cairo gradient.
        This is for Editing gradient.
        """
        if self._cairo_gradient is None:
            if tdw:
                sx, sy = tdw.model_to_display(*self._start_pos)
                ex, ey = tdw.model_to_display(*self._end_pos)
            else:
                sx, sy = self._start_pos
                ex, ey = self._end_pos
            
            self._cairo_gradient = self.generate_gradient(sx, sy, ex, ey)
        return self._cairo_gradient

    def generate_gradient(self, sx, sy, ex, ey):
        """Generate cairo gradient object from
        internal datas.
        Use this for Final output.
        """
        if len(self.nodes) >= 2:
            g = cairo.LinearGradient(
                sx, sy,
                ex, ey
                )
            for pt in self.nodes:
                g.add_color_stop_rgba(
                        pt.linear_pos, 
                        *pt.get_rgba())
            return g

    def invalidate_cairo_gradient(self):
        self._cairo_gradient = None


    # node related method

    def add_intermidiate_point(self, linear_pos, color):
        # search target index from linear_pos
        i=0
        for pt in self.nodes:
            if pt.linear_pos > linear_pos:
                break
            i+=1

        if i < len(self.nodes):
            self.nodes.insert(
                    i,
                    GradientInfo(linear_pos, color)
                    )
        else:
            self.nodes.append(
                    GradientInfo(linear_pos, color)
                    )

        self._current_node_index = i


        self.invalidate_cairo_gradient() 

    def add_gradient_point_from_display(self, tdw, x, y):
        tx, ty = tdw.display_to_model(x, y)
        sx, sy = self._start_pos
        ex, ey = self._end_pos
        lln, nx, ny = gui.linemode.length_and_normal(
                sx, sy,
                ex, ey)

        tln, tnx, tny = gui.linemode.length_and_normal(
                sx, sy,
                tx, ty)

        linear_pos = tln / lln

        self.add_intermidiate_point(
                linear_pos, 
                self.get_current_color())
        
    def remove_gradient_node(self, idx):
        assert idx < len(self.nodes)

        del self.nodes[idx]
        self.set_target_node_index(None)
        self.invalidate_cairo_gradient() 

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

           #h = V * dot_product(Normalize(v), vp) / length - vp

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
           #self.invalidate_cairo_gradient() # To generate new one.

    def finalize_offsets(self, tdw):
        if len(self.nodes) >= 2 and self.is_ready():
            sx, sy = tdw.model_to_display(*self._start_pos)
            self._start_pos = tdw.display_to_model(sx+self._dx, sy+self._dy)
            ex, ey = tdw.model_to_display(*self._end_pos)
            self._end_pos = tdw.display_to_model(ex+self._dx, ey+self._dy)
            self._dx = 0
            self._dy = 0
            self.invalidate_cairo_gradient() # To generate new one.
        
    def _enum_node_position(self, tdw):
        """Enumrate nodes position with a tuple

        :yield : a tuple of (node_index, control_point, center_x, center_y,
                 end_x, end_y)
                 end_x/y is same as the current coordinate of the control point.
        """
        if not self.is_ready():
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
    def paint(self, cr, mode, tdw):
        """Paint this controller in cairo context.
        Used from Overlay class.
        """

        cr.save()
        radius = self._radius
        dx = self._dx
        dy = self._dy

        if len(self.nodes) >= 2 and self.is_ready():
            cr.save()
            cr.set_line_width(radius)

            # I dont use cr.translate() here
            # because, _enum_node_position() refer to self._dx/_dy
            # not only drawing, but also queue area and hit test.
            # so using cr.translate() seems to be meaningless...
            sx, sy = tdw.model_to_display(*self._start_pos)
            ex, ey = tdw.model_to_display(*self._end_pos)
            last_node = self.nodes[-1]

            cr.move_to(sx + dx, sy + dy)
            cr.line_to(ex + dx, ey + dy)
            gui.drawutils.render_drop_shadow(cr)

            # base shading, considering for alpha transparency.
            cr.set_source_rgb(0.5, 0.5, 0.5) 
            if not mode.in_drag:
                cr.stroke_preserve()
                cr.set_source(self.get_cairo_gradient(tdw))
            cr.stroke()

            # Drawing simulated gradient, to save 
            # memory efficiency.
            # otherwise, we need to generate
            # cairo.*Gradient object each time 
            # for drawing controller,by moving control points.

            # Drawing simurated gradiation
            if mode.in_drag:
                for i, pt, cx, cy, ex, ey in self._enum_node_position(tdw):
                    if i > 0:
                        cr.set_source_rgba(*ppt.get_rgba())
                        cr.move_to(sx, sy)
                        cr.line_to(cx, cy)
                        cr.stroke()
                        cr.move_to(cx, cy)
                        cr.set_source_rgba(*pt.get_rgba())
                        cr.line_to(ex, ey)
                        cr.stroke()

                    ppt = pt
                    sx,sy = ex, ey

            # Drawing nodes.
            # if do this above 'simurated gradiation' loop,
            # some lines overdraw node chips.
            # so do that here to overdraw simulated lines.
            for i, pt, cx, cy, ex, ey in self._enum_node_position(tdw):
                self._draw_single_node(cr, ex, ey, i, pt)

            cr.restore()

        if self._target_pos is not None:
            x, y = self._target_pos
            gui.drawutils.render_round_floating_color_chip(
                    cr,
                    x, y,
                    gui.style.ACTIVE_ITEM_COLOR,
                    3.0)

        cr.restore()

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
        # -2 is shadow size.
        tdw.queue_draw_area(x-r-2, y-r-2, r*2+4, r*2+4)

    def queue_redraw(self, tdw):
        r = self._radius + 2
        # Offset dx/dy should be used when moving entire controller.
        # so, it should be used for only the case when len(self.nodes) is >= 2.
        # Although, adding offsets is harmless even in other cases,
        # it is just meaningless. 
        dx = self._dx
        dy = self._dy

        if self._target_pos:
            x, y = self._target_pos
            self.queue_single_point(tdw, x, y, r)

        sx=sy=None
        
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
               #self.remove_gradient_node(idx)
                self.refresh_current_color()
                self._phase = _GradientPhase.STAY
            else:
                self._phase = _GradientPhase.MOVE_NODE
            self.queue_redraw(tdw)
        elif idx == -1:
            if shift_state:
                self.add_gradient_point_from_display(tdw, x, y)
                self._phase = _GradientPhase.STAY
                self.queue_redraw(tdw)
            else:
                self._phase = _GradientPhase.MOVE
        else:
            self._phase = _GradientPhase.INIT_NODE
            self._target_pos = (x, y)

    def button_release_cb(self, mode, tdw, event):
        pass                                                         

    def drag_start_cb(self, mode, tdw, event):
        self._gradient_update = False
        self._dx = 0
        self._dy = 0
        pass

    def drag_update_cb(self, mode, tdw, event, dx, dy):
        self.queue_redraw(tdw) # to erase
        x = event.x
        y = event.y
        if self._phase == _GradientPhase.INIT_NODE:
            self._target_pos = (x, y)
        elif self._phase == _GradientPhase.MOVE_NODE:
            idx = self._target_node_index
            if idx == 0:
                self._start_pos = tdw.display_to_model(x, y)
                self._gradient_update = True
            elif idx == len(self.nodes)-1:
                self._end_pos = tdw.display_to_model(x, y)
                self._gradient_update = True
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
                        self._gradient_update = True
                        self.nodes[idx].set_linear_pos(cl)
        elif self._phase == _GradientPhase.MOVE:
            self.add_offsets(dx, dy)
        self.queue_redraw(tdw)
        pass

    def drag_stop_cb(self, mode, tdw):
        
        self.queue_redraw(tdw) # to erase
        if self._phase == _GradientPhase.INIT_NODE:
            if self._start_pos is None:
                self.set_start_pos(tdw, self._target_pos)
            elif self._end_pos is None:
                self.set_end_pos(tdw, self._target_pos)

            self.invalidate_cairo_gradient()
            self._target_pos = None

        elif self._phase == _GradientPhase.MOVE_NODE:

            if self._gradient_update:
                # Clear cairo gradient cache,
                # to invoke generation of it in paint() method.
                self.invalidate_cairo_gradient()

            self.clear_target_node_index()
        elif self._phase == _GradientPhase.MOVE:
            self.finalize_offsets(tdw)

        self.queue_redraw(tdw)
        self._phase = _GradientPhase.STAY
        pass

    # Action callback
    def delete_current_item(self):
        """ Caution: this method does not update any visuals
        because there is no access method for tdw.
        so, caller must update visual of gradient controller.
        """
        idx = self._current_node_index
        if 0 < idx < len(self.nodes)-1:
            self.remove_gradient_node(idx)


if __name__ == '__main__':

    pass


