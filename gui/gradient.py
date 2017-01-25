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

#class _EditZone_Gradient:
#    EMPTY_CANVAS = 0
#    CONTROL_NODE = 1
#    ACTION_BUTTON = 2

class _GradientPhase:
    INIT_NODE = 0
    MOVE = 1
    MOVE_NODE = 2
    STAY = 10

class _GradientInfo(object):

    def __init__(self, linear_pos, color, alpha=1.0): 
        """
        :param linear_pos: gradient linear position. inside from 0.0 to 1.0.
        :param color:lib.color object, or tuple
        :param alpha:float value from 0.0 to 1.0, as alpha component.
        """
        self._lpos = linear_pos # Gradient linear position.
        self._alpha = alpha  # Place this line here, 
                             # prior to self._color setup

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

    def get_rgba(self):
        col = self._color
        return (col.r, col.g, col.b, self._alpha)

    def set_color(self, color, alpha=1.0):
        if type(color) == tuple:
            if len(color) == 3:
                self._color = lib.color.RGBColor(rgb=color)
            elif len(color) == 4:
                self._color = lib.color.RGBColor(rgb=color[:3])
                self._alpha = color[3]
            else:
                raise ValueError
        else:
            self._color = color # assume it as color object 


class GradientController(object):
    """GradientController, to manage & show gradient oncanvas.
    This cannot be Mode object, because we need to show
    current gradient setting in polyfilltool.
    """

    def __init__(self, app):
        self.app = weakref.proxy(app)
        self._current_node = None
        self._target_node = None
        self._active = False
        self.nodes = []
        self._cairo_gradient = None

        self._radius = 6.0
        self._phase = _GradientPhase.STAY
        self._dx = self._dy = 0
        self._start_pos = None
        self._end_pos = None
        self._target_pos = None
        self._overlay_ref = None


   #def setup_gradient(self, colors, pos):
   #    """Setup gradient colors with a sequence.
   #
   #    This method clears all nodes information and 
   #    controller positions.
   #
   #    :param colors: a sequence of color tuple.
   #    :param pos: a sequence of float number, gradient linear positions.
   #                This must be same counts as colors sequence.
   #                if None, linear positions automatically generated.
   #
   #    """
   #    self.nodes = []
   #    if pos:
   #        assert len(pos) == len(colors)
   #        pos.sort()
   #
   #    for i, c in enumerate(colors):
   #        if pos:
   #            curpos = max(min(pos[i], 1.0), 0.0)
   #        else:
   #            curpos = max(min(float(i) / (len(colors)-1), 1.0), 0.0)
   #
   #        self.nodes.append(
   #                _GradientInfo(curpos, c, 1.0)
   #                )
    def setup_gradient(self, datas):
        """Setup gradient colors with a sequence.

        This method clears all nodes information and 
        controller positions.

        :param colors: a sequence of linear position and color tuple.
                       (linear_position, (r, g, b, a))

        """
        self.nodes = []

        for i, data in enumerate(datas):
            pos, color = data
            if pos > -1:
                curpos = max(min(pos, 1.0), 0.0)
            else:
                curpos = max(min(float(i) / (len(datas)-1), 1.0), 0.0)

            if color == -1:
                color = self.get_current_color()
            elif color == -2:
                assert NotImplementedError("There is no background color for mypaint!")
               #color = self.get_current_bgcolor()
            else:
                assert len(color) >= 3

            self.nodes.append(
                    _GradientInfo(curpos, color, 1.0)
                    )

    def set_start_pos(self, tdw, disp_pos):
        """Set gradient start position, in model coordinate.
        :param disp_pos: start position for cairo.LinearGradient.
                          if None, used current polygon
                          center X and minimum Y coordinate.
        """
        self._start_pos = tdw.display_to_model(*disp_pos)

    def set_end_pos(self, tdw, disp_pos):
        """Set gradient end position, in model coordinate.
        :param disp_pos: end position for cairo.LinearGradient.
                          if None, used current polygon
                          center X and maximum Y coordinate.
        """
        self._end_pos = tdw.display_to_model(*disp_pos)

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
                    _GradientInfo(linear_pos, color)
                    )
        else:
            self.nodes.append(
                    _GradientInfo(linear_pos, color)
                    )

        self._current_node = i


        self._cairo_gradient = None 

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
        
    def remove_gradient_point(self, idx):
        assert idx < len(self.nodes)

        del self.nodes[idx]
        self.set_target_node(None)
        self._cairo_gradient = None 

    @property
    def current_point(self):
        return self._current_node

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, flag):
        self._active = flag

    @property
    def is_ready(self):
        return (len(self.nodes) > 0 
                and self._start_pos != None 
                and self._end_pos != None)

    def set_target_node(self, idx):
        self._target_node = idx
        self._current_node = idx

    def clear_target_node(self):
        self._target_node = None
        # Still remained self._current_node

    def get_current_color(self):
        return self.app.brush_color_manager.get_color()

    def refresh_current_color(self):
        if self._current_node != None:
            pt = self.nodes[self._current_node]
            pt.set_color(self.get_current_color())
            self._cairo_gradient = None 

    def get_cairo_gradient(self, tdw):
        """Get cairo gradient.
        This is for Editing gradient.
        """
        if self._cairo_gradient == None:
            self._cairo_gradient = self.generate_gradient(tdw)
        return self._cairo_gradient

    def generate_gradient(self, tdw=None):
        """Generate cairo gradient object from
        internal datas.
        Use this for Final output.
        """
        if len(self.nodes) >= 2:
            if tdw:
                sx, sy = tdw.model_to_display(*self._start_pos)
                ex, ey = tdw.model_to_display(*self._end_pos)
            else:
                sx, sy = self._start_pos
                ex, ey = self._end_pos
            g = cairo.LinearGradient(
                sx, sy,
                ex, ey
                )
            for pt in self.nodes:
                g.add_color_stop_rgba(
                        pt.linear_pos, 
                        *pt.get_rgba())
            return g
            
    def get_hit_point(self, tdw, x, y):
        """Get the index of a point which located at target position.
        :param x, y: target position.
        :return : index of point, or -1 when pointing gradiant gauge line.
                  otherwise, return None.
        """
        if not self.is_ready:
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

           #h = V * dot_product(Normalize(v), vp) / length - vp

            vx = ex - sx
            vy = ey - sy

            leng, nx, ny = gui.linemode.length_and_normal(sx, sy, ex, ey)

            vpx = x - sx
            vpy = y - sy

            dp = gui.linemode.dot_product(nx, ny, vpx, vpy)

            hx = vx * dp / leng - vpx
            hy = vy * dp / leng - vpy

            height = math.hypot(hx, hy)

            if height < r:
                return -1
        return None

    def set_offsets(self, offset_x, offset_y):
        if len(self.nodes) >= 2:
            self._dx = offset_x
            self._dy = offset_y
           #self._cairo_gradient = None # To generate new one.

    def finalize_offsets(self, tdw):
        if len(self.nodes) >= 2 and self.is_ready:
            sx, sy = tdw.model_to_display(*self._start_pos)
            self._start_pos = tdw.display_to_model(sx+self._dx, sy+self._dy)
            ex, ey = tdw.model_to_display(*self._end_pos)
            self._end_pos = tdw.display_to_model(ex+self._dx, ey+self._dy)
            self._dx = 0
            self._dy = 0
            self._cairo_gradient = None # To generate new one.
        

    # GUI related

   #def _update_zone_and_target(self, mode, tdw, x, y):
   #    """Update the zone and target node under a cursor position"""
   #
   #    
   #    new_zone = EditZoneMixin.EMPTY_CANVAS
   #    new_handle_idx = None
   #
   #    if not self.in_drag:
   #       #if self.phase in (PhaseMixin.CAPTURE, PhaseMixin.ADJUST):
   #       #if self.phase == PhaseMixin.ADJUST:
   #        if self.is_actionbutton_ready():
   #            new_target_node_index = None
   #            self.current_button_id = None
   #            # Test buttons for hits
   #            hit_dist = gui.style.FLOATING_BUTTON_RADIUS
   #
   #            for btn_id in self.buttons.keys():
   #                btn_pos = overlay.get_button_pos(btn_id)
   #                if btn_pos is None:
   #                    continue
   #                btn_x, btn_y = btn_pos
   #                d = math.hypot(btn_x - x, btn_y - y)
   #                if d <= hit_dist:
   #                    new_target_node_index = None
   #                    new_zone = EditZoneMixin.ACTION_BUTTON
   #                    self.current_button_id = btn_id
   #                    break
   #
   #        # Test nodes for a hit, in reverse draw order
   #        if new_zone == EditZoneMixin.EMPTY_CANVAS:
   #            new_target_node_index, new_handle_idx = \
   #                    self._search_target_node(tdw, x, y)
   #            if new_target_node_index != None:
   #                new_zone = EditZoneMixin.CONTROL_NODE
   #
   #        # Update the prelit node, and draw changes to it
   #        if new_target_node_index != self.target_node_index:
   #            # Redrawing old target node.
   #            if self.target_node_index is not None:
   #                self._queue_draw_node(self.target_node_index)
   #                self.node_leave_cb(tdw, self.nodes[self.target_node_index]) 
   #
   #            self.target_node_index = new_target_node_index
   #            if self.target_node_index is not None:
   #                self._queue_draw_node(self.target_node_index)
   #                self.node_enter_cb(tdw, self.nodes[self.target_node_index]) 
   #
   #        self.current_node_handle = new_handle_idx
   #
   #    # Update the zone, and assume any change implies a button state
   #    # change as well (for now...)
   #    if self.zone != new_zone:
   #       #if not self.in_drag:
   #        self._enter_zone_cb(new_zone)
   #        self.zone = new_zone
   #        self._ensure_overlay_for_tdw(tdw)
   #        self._queue_draw_buttons()
   #
   #    if not self.in_drag:
   #        self._update_cursor(tdw)

    # Paint related  
    def paint(self, cr, mode, tdw):
        """Paint this controller in cairo context.
        Used from Overlay class.
        """

        cr.save()
        radius = self._radius
        dx = self._dx
        dy = self._dy

        if len(self.nodes) >= 2 and self.is_ready:
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

        if self._target_pos != None:
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

        if self._target_node == i or self._current_node == i:
            gui.drawutils.render_round_floating_color_chip(
                    cr,
                    x, y,
                    pt.color,
                    self._radius / 2)

    # node management

    def _enum_node_position(self, tdw):
        """Enumrate nodes position with a tuple

        :yield : a tuple of (node_index, control_point, center_x, center_y,
                 end_x, end_y)
                 end_x/y is same as the current coordinate of the control point.
        """
        if not self.is_ready:
            raise StopIteration

        dx = self._dx
        dy = self._dy

        if self._start_pos != None:
            sx, sy = tdw.model_to_display(*self._start_pos)
        if self._end_pos != None:
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

            if sx != None: # for nodes after index 1
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
        idx = self.get_hit_point(tdw, x, y)
        shift_state = event.state & Gdk.ModifierType.SHIFT_MASK

        if idx >= 0:
            self.set_target_node(idx)
            if shift_state: 
               #self.remove_gradient_point(idx)
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
        pass

    def drag_update_cb(self, mode, tdw, event, dx, dy):
        self.queue_redraw(tdw) # to erase
        x = event.x
        y = event.y
        if self._phase == _GradientPhase.INIT_NODE:
            self._target_pos = (x, y)
        elif self._phase == _GradientPhase.MOVE_NODE:
            idx = self._target_node
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

                # Use virtual-cross product to detect 
                # whether exceeding 1st point or not.
                # if this value is less than 0,
                # it exceeds the 1st point.
                cp = gui.linemode.cross_product(sx, sy, x, y)

                min_lpos = self.nodes[idx-1].linear_pos
                max_lpos = self.nodes[idx+1].linear_pos

                if l > 0.0:
                    cl /= l
                    if cp > 0 and cl > min_lpos and cl < max_lpos:
                        self._gradient_update = True
                        self.nodes[idx].set_linear_pos(cl)
        elif self._phase == _GradientPhase.MOVE:
            self.set_offsets(dx, dy)
        self.queue_redraw(tdw)
        pass

    def drag_stop_cb(self, mode, tdw):
        
        self.queue_redraw(tdw) # to erase
        if self._phase == _GradientPhase.INIT_NODE:
            if self._start_pos == None:
                self.set_start_pos(tdw, self._target_pos)
            elif self._end_pos == None:
                self.set_end_pos(tdw, self._target_pos)

            self._cairo_gradient = None
            self._target_pos = None

        elif self._phase == _GradientPhase.MOVE_NODE:

            if self._gradient_update:
                # Clear cairo gradient cache,
                # to invoke generation of it in paint() method.
                self._cairo_gradient = None

            self.clear_target_node()
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
        idx = self._current_node
        if 0 < idx < len(self.nodes)-1:
            self.remove_gradient_point(idx)


if __name__ == '__main__':

    pass


