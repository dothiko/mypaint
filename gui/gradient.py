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

from gui.linearcontroller import *
from gui.linearcontroller import _LinearPhase

#class _GradientPhase:
#    INIT_NODE = 0
#    MOVE = 1
#    MOVE_NODE = 2
#    STAY = 10

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

class GradientController(LinearController):
    """GradientController, to manage & show gradient oncanvas.
    This cannot be Mode object, because we need to show
    current gradient setting in polyfilltool.
    """

    # Class constants
   #MOVING_CURSOR = 0
   #MOVING_CURSOR_NAME = gui.cursor.Name.HAND_OPEN
   #
   #MOVING_NODE_CURSOR = 1
   #MOVING_NODE_CURSOR_NAME = gui.cursor.Name.CROSSHAIR_CLOSED


    # Action name (need to get cursor from self.app)
    #
    # Actually, this is not 'mode' class object.
    # And this object should be used only from PolyfillMode.
    # so, use the name "PolyfillMode"
    ACTION_NAME = "PolyfillMode"

    def __init__(self, app):
        super(GradientController, self).__init__(app)
        self.invalidate_cairo_gradient()
        self._follow_brushcolor = False
        self._prev_brushcolor = None


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

    def add_control_point_from_display(self, tdw, x, y):
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

            
    def finalize_offsets(self, tdw):
        super(GradientController, self).finalize_offsets(tdw)
        if len(self.nodes) >= 2 and self.is_ready():
            self.invalidate_cairo_gradient() # To generate new one.
        

    def _shading_contents(self, cr, mode, tdw):
        # base shading, considering for alpha transparency.
        cr.save()

        cr.set_source_rgb(0.5, 0.5, 0.5) 
        if not mode.in_drag:
            cr.stroke_preserve()
            cr.set_source(self.get_cairo_gradient(tdw))
        cr.stroke()

        # Drawing simurated gradiation
        if len(self.nodes) >= 2 and mode.in_drag:
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

    ## Event handlers (for overriding)

    def drag_stop_cb(self, mode, tdw):

        if self._phase == _LinearPhase.MOVE_NODE:
            if self._contents_update:
                # Clear cairo gradient cache,
                # to invoke generation of it in paint() method.
                self.invalidate_cairo_gradient()

        super(GradientController, self).drag_stop_cb(mode, tdw)

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


