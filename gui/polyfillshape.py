#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
from gettext import gettext as _

from gi.repository import Gdk

from gui.oncanvas import *
import gui.gradient
from gui.beziertool import detect_on_stroke, _Control_Handle, _Node_Bezier
import gui.style

class _EditZone(EditZoneMixin):
    """Enumeration of what the pointer is on in the ADJUST phase"""
    GRADIENT_BAR = 1001

class _Phase(PhaseMixin):
    """Enumeration of the states that an BezierCurveMode can be in"""
    ADJUST_HANDLE = 103     #: change control-handle position
    INIT_HANDLE = 104       #: initialize control handle,right after create a node
    PLACE_NODE = 105        #: place a new node into clicked position on current
                            # stroke,when you click with holding CTRL key
    CALL_BUTTONS = 106      #: show action buttons around the clicked point.
    GRADIENT_CTRL = 107     #: gradient controller

## Shape classes
#  Used from PolyfillMode class (gui/polyfilltool.py)
#  By switching these shape classes,
#  Polyfilltool supports various different shape such as
#  rectangle or ellipse

class Shape(object):
    CANCEL_EVENT = 1
    CALL_BASECLASS_HANDLER = 2
    CALL_ANCESTER_HANDLER = 3

    TYPE_BEZIER = 0
    TYPE_POLYLINE = 1
    TYPE_RECTANGLE = 2
    TYPE_ELLIPSE = 3

    MARGIN = int(math.ceil(gui.style.DROP_SHADOW_BLUR))
    accept_handle = False

    def generate_node(self):
        return _Node_Bezier(
                x=0, y=0,
                pressure=0.0,
                xtilt=0.0, ytilt=0.0,
                dtime=1.0
                )
    def get_maximum_rect(self, tdw, mode, dx=0, dy=0):
        raise NotImplementedError

    def update_event(self, event):
        self.shift_state = event.state & Gdk.ModifierType.SHIFT_MASK
        self.ctrl_state = event.state & Gdk.ModifierType.CONTROL_MASK
        self.alt_state = event.state & Gdk.ModifierType.MOD1_MASK

    def clear_event(self):
        self.shift_state = False
        self.ctrl_state = False
        self.alt_state = False

    @staticmethod
    def draw_dash(cr, dot):
        cr.set_dash((), 0)
        cr.set_source_rgb(1,1,1)
        cr.stroke_preserve()
        cr.set_source_rgb(0,0,0)
        cr.set_dash((dot, ) )
        cr.stroke()


class Shape_Bezier(Shape):

    name = _("Bezier")
    accept_handle = True

    def __init__(self):
        pass

    def draw_node_polygon(self, cr, tdw, nodes, selected_nodes=None,
            color=None, gradient=None,
            dx=0, dy=0, ox=0, oy=0, stroke=False, fill=True):
        """ draw cairo polygon
        :param color: color object of mypaint
        :param gradient: Gradient object. cairo.LinearGradient or something

        if both of color and gradient are null,polygon is not filled.

        :param selected_nodes: list of INDEX of selected nodes
        :param dx,dy: offset position of selected nodes, in MODEL.
        :param ox,oy: polygon origin position
        """
        if len(nodes) > 1:
            cr.save()
            cr.set_line_width(1)
            if fill:
                if gradient:
                    cr.set_source(gradient)
                elif color:
                    cr.set_source_rgb(*color.get_rgb())

            for i, node in enumerate(nodes):#self._get_onscreen_nodes():

                if i==len(nodes)-1 and len(nodes) < 3:
                    break

                if tdw:
                    x, y = tdw.model_to_display(node.x, node.y)
                else:
                    x, y = node

                x-=ox
                y-=oy

                n = (i+1) % len(nodes)

               #if tdw:
               #    x1, y1 = tdw.model_to_display(*node.get_control_handle(1))
               #    x2, y2 = tdw.model_to_display(*nodes[n].get_control_handle(0))
               #    x3, y3 = tdw.model_to_display(nodes[n].x, nodes[n].y)
               #else:
                x1, y1 = node.get_control_handle(1)
                x2, y2 = nodes[n].get_control_handle(0)
                x3, y3 = nodes[n].x, nodes[n].y

                x1-=ox
                x2-=ox
                x3-=ox
                y1-=oy
                y2-=oy
                y3-=oy

                if selected_nodes:

                    if i in selected_nodes:
                        x += dx
                        y += dy
                        x1 += dx
                        y1 += dy

                    if n in selected_nodes:
                        x2 += dx
                        y2 += dy
                        x3 += dx
                        y3 += dy

                if tdw:
                    x1, y1 = tdw.model_to_display(x1, y1)
                    x2, y2 = tdw.model_to_display(x2, y2)
                    x3, y3 = tdw.model_to_display(x3, y3)

                if fill or i < len(nodes)-1:
                    if i==0:
                        cr.move_to(x,y)

                    cr.curve_to(x1, y1, x2, y2, x3, y3)



            if fill and len(nodes) > 2 and (gradient or color):
                cr.close_path()
                cr.fill_preserve()

            if stroke:

                Shape.draw_dash(cr, 4)

                if len(nodes) > 2:
                    cr.move_to(x,y)
                    cr.curve_to(x1, y1, x2, y2, x3, y3)
                    Shape.draw_dash(cr, 10)


            cr.restore()


    def get_maximum_rect(self, tdw, mode, dx=0, dy=0):
        """ get possible maximum rectangle
        :param tdw: the target tileddrawwidget.if this is None,
                    all values(includeing dx,dy) recognized as
                    model coordinate value.
        :rtype tuple: a tuple of (x, y, width, height)
        """
        if len(mode.nodes) < 2:
            return (0,0,0,0)
        margin = Shape.MARGIN

        def adjust_from_control_handle(tdw, cn, n_index, h_index,
                sx, sy, ex, ey, dx, dy, margin):
            if tdw:
                cx, cy = tdw.model_to_display(*cn.get_control_handle(h_index))
            else:
                cx, cy = cn.get_control_handle(h_index)

            if n_index in mode.selected_nodes:
                if (mode.current_node_handle == None or
                        mode.current_node_handle == h_index):
                    cx += dx
                    cy += dy

            return (min(sx, cx - margin), min(sy, cy - margin),
                    max(ex, cx + margin), max(ey, cy + margin))




        # Get boundary rectangle of each segment
        # and return the maximum
        for i,cn in enumerate(mode.nodes):
            if tdw:
                cnx, cny = tdw.model_to_display(cn.x, cn.y)
            else:
                cnx, cny = cn

            if i in mode.selected_nodes:
                cnx+=dx
                cny+=dy

            if i == 0:
                sx = cnx - margin
                ex = cnx + margin
                sy = cny - margin
                ey = cny + margin
            else:
                sx = min(sx, cnx - margin)
                ex = max(ex, cnx + margin)
                sy = min(sy, cny - margin)
                ey = max(ey, cny + margin)

            sx, sy, ex, ey = adjust_from_control_handle(tdw, cn, i, 0,
                sx, sy, ex, ey, dx, dy, margin)

            sx, sy, ex, ey = adjust_from_control_handle(tdw, cn, i, 1,
                sx, sy, ex, ey, dx, dy, margin)

        return (sx, sy, abs(ex - sx) + 1, abs(ey - sy) + 1)


    def button_press_cb(self, mode, tdw, event):


        if mode.phase in (_Phase.ADJUST, _Phase.ADJUST_POS):
            # Remember, the base class (of mode instance) might
            # change Phase automatically into _Phase.ADJUST_POS
            # when user grab the node(or its handle)
            if mode.zone == _EditZone.CONTROL_NODE:
                # Grabbing a node...
                button = event.button

                if button == 1 and mode.current_node_index != None:
                    if self.ctrl_state:
                        # Holding CONTROL key = adding or removing a node.
                        mode.select_node(mode.current_node_index)
                       #if mode.current_node_index in mode.selected_nodes:
                       #    mode.selected_nodes.remove(mode.current_node_index)
                       #else:
                       #    mode.selected_nodes.append(mode.current_node_index)
                       #mode._queue_draw_selected_nodes()
                    else:
                        # no CONTROL Key holded.
                        # If new solo node clicked without holding
                        # CONTROL key,then reset all selected nodes.

                        if mode.current_node_handle != None:
                            mode.phase = _Phase.ADJUST_HANDLE
                            mode._queue_draw_node(tdw,
                                                  mode.current_node_index)
                        else:
                            mode.phase = _Phase.ADJUST_POS
                            do_reset = self.alt_state
                            do_reset |= not (mode.current_node_index in mode.selected_nodes)

                            if do_reset:
                                # To avoid old selected nodes still lit.
                                mode._queue_draw_selected_nodes(tdw)
                               #mode._reset_selected_nodes(mode.current_node_index)
                                mode.select_node(mode.current_node_index, True)

                # FALLTHRU: *do* start a drag

           # TODO deprecated, because there is some problem
           # to dividing stroke as user interface aspect.
           # We want to pick colors from EMPTY_CANVAS,
           # but on_stroke check is high cost for processing power,
           # so it is difficult to distinguish that point is actually
           # empty canvas or on stroke midpoint.
           # We might need other 'User-defined modifier keys'...

           #elif mode.zone == _EditZone.EMPTY_CANVAS:
           #
           #    if mode.phase == _Phase.ADJUST:
           #        if (len(mode.nodes) > 0):
           #            if shift_state and ctrl_state:
           #           #if self.ctrl_state:
           #                mx, my = tdw.display_to_model(event.x, event.y)
           #                pressed_segment = beziertool.detect_on_stroke(mx, my)
           #                if pressed_segment:
           #                    # pressed_segment is a tuple which contains
           #                    # (node index of start of segment, stroke step)
           #
           #                    # To erase buttons
           #                    mode._queue_draw_buttons()
           #
           #                    mode._divide_bezier(*pressed_segment)
           #
           #                    # queue new node here.
           #                    mode._queue_draw_node(pressed_segment[0] + 1)
           #
           #                    mode.phase = _Phase.PLACE_NODE
           #                    return Shape.CANCEL_EVENT # Cancel drag event



            # FALLTHRU: *do* start a drag


    def button_release_cb(self, mode, tdw, event):

        # Here is 'button_release_cb',which called
        # prior to drag_stop_cb.
        # so, in this method, changing mode._phase
        # is very special case.
        if mode.phase == _Phase.PLACE_NODE:
            mode._queue_redraw_item(tdw)
            mode.phase = _Phase.ADJUST



    def drag_start_cb(self, mode, tdw, event):
        # Basically,all sections should do fall-through.
        mx, my = tdw.display_to_model(event.x, event.y)

        if mode.phase == _Phase.ADJUST:

            if mode.zone == _EditZone.EMPTY_CANVAS:
                if event.state != 0:
                    # To activate some mode override
                    mode._last_event_node = None
                    return Shape.CALL_ANCESTER_HANDLER
                else:
                    # New node added!
                    node = mode._get_event_data(tdw, event)
                    mode.nodes.append(node)
                    mode._last_event_node = node
                    mode.phase = _Phase.INIT_HANDLE
                    idx = len(mode.nodes) - 1
                    mode.select_node(idx, exclusive=True)
                    # Important: with setting initial control handle
                    # as the 'next' (= index 1) one,it brings us
                    # inkscape-like node creation.
                    mode.current_node_handle = 1

                    mode.current_node_index=idx
                    mode._queue_draw_node(tdw, idx)

                    # Actually, cancel select node.
                    mode.drag_offset.start(mx, my)

        elif mode.phase == _Phase.ADJUST_POS:
            if len(mode.selected_nodes) > 0:
                mode.drag_offset.start(mx, my)
        elif mode.phase in (_Phase.ADJUST_HANDLE, _Phase.INIT_HANDLE):
           #mode._last_event_node = mode.nodes[mode.target_node_index]
            assert mode.current_node_index != None
            mode._last_event_node = mode.nodes[mode.current_node_index]


    def drag_update_cb(self, mode, tdw, event, dx, dy):

        mx, my = tdw.display_to_model(event.x, event.y)

        if mode.phase == _Phase.ADJUST:
            pass

        elif mode.phase in (_Phase.ADJUST_HANDLE, _Phase.INIT_HANDLE):
            mode._queue_redraw_item(tdw)# to erase item - because it uses overlay.
            node = mode._last_event_node
            if node:
                mode._queue_draw_node(tdw, mode.current_node_index)# to erase
                node.set_control_handle(mode.current_node_handle,
                        mx, my,
                        self.shift_state)
                mode._queue_draw_node(tdw, mode.current_node_index)
            mode._queue_redraw_item(tdw)

        elif mode.phase == _Phase.ADJUST_POS:
            if len(mode.selected_nodes) > 0:
                mode._queue_redraw_item(tdw)
                mode._queue_draw_selected_nodes(tdw)
                mode.drag_offset.end(mx, my)
                mode._queue_draw_selected_nodes(tdw)
                mode._queue_redraw_item(tdw)

    def drag_stop_cb(self, mode, tdw):
        if mode.phase == _Phase.ADJUST:
            mode._queue_redraw_all_nodes()
            mode._reset_adjust_data()
            if len(mode.nodes) > 0:
                mode._queue_redraw_item(tdw)
               #mode._queue_redraw_all_nodes()
                if len(mode.nodes) > 1:
                    mode._queue_draw_buttons()


        elif mode.phase in (_Phase.ADJUST_HANDLE, _Phase.INIT_HANDLE):
            node = mode._last_event_node

            # At initialize handle phase, even if the node is not 'curve'
            # Set the handles as symmetry.
            if (mode.phase == _Phase.INIT_HANDLE):
                node.curve = not mode.DEFAULT_POINT_CORNER

            mode._queue_redraw_all_nodes()
            mode._queue_redraw_item(tdw)
            if len(mode.nodes) > 1:
                mode._queue_draw_buttons()

            mode.phase = _Phase.ADJUST
        elif mode.phase == _Phase.ADJUST_POS:
            dx, dy = mode.drag_offset.get_model_offset()

            for idx in mode.selected_nodes:
                cn = mode.nodes[idx]
                cn.move(cn.x + dx, cn.y + dy)

            mode.drag_offset.reset()
            mode._dragged_node_start_pos = None
            mode._queue_redraw_item(tdw)
            mode._queue_draw_buttons()
            mode.phase = _Phase.ADJUST


class Shape_Polyline(Shape_Bezier):

    name = _("Polyline")
    accept_handle = False

    def __init__(self):
        pass

    def draw_node_polygon(self, cr, tdw, nodes, selected_nodes=None,
            color=None, gradient=None,
            dx=0, dy=0, ox=0, oy=0, stroke=False, fill=True):
        """ draw cairo polygon
        :param color: color object of mypaint
        :param gradient: Gradient object. cairo.LinearGradient or something

        if both of color and gradient are null,polygon is not filled.

        :param selected_nodes: list of INDEX of selected nodes
        :param dx,dy: offset position of selected nodes, in MODEL.
        :param ox,oy: polygon origin position
        """
        if len(nodes) > 1:
            cr.save()
            cr.set_line_width(1)
            if fill:
                if gradient:
                    cr.set_source(gradient)
                elif color:
                    cr.set_source_rgb(*color.get_rgb())

            for i, node in enumerate(nodes):

               #if tdw:
               #    x, y = tdw.model_to_display(node.x, node.y)
               #else:
               #    x, y = node
                x, y = node

                x-=ox
                y-=oy

                if selected_nodes:

                    if i in selected_nodes:
                        x += dx
                        y += dy

                if tdw:
                    x, y = tdw.model_to_display(x, y)

                if i==0:
                    cr.move_to(x, y)
                else:
                    cr.line_to(x, y)



            if fill and len(nodes) > 2 and (gradient or color):
                cr.close_path()
                cr.fill_preserve()

            if stroke:
                Shape.draw_dash(cr, 4)

                if len(nodes) > 2:
                    cr.move_to(x,y)
                    x, y = tdw.model_to_display(nodes[0].x, nodes[0].y)
                    cr.line_to(x,y)
                    Shape.draw_dash(cr, 10)

            cr.restore()

    def drag_start_cb(self, mode, tdw, event):
        # Basically,all sections should do fall-through.
        mx, my = tdw.display_to_model(event.x, event.y)

        if mode.phase == _Phase.ADJUST:

            if mode.zone == _EditZone.EMPTY_CANVAS:
                if event.state != 0:
                    # To activate some mode override
                    mode._last_event_node = None
                    return Shape.CALL_ANCESTER_HANDLER
                else:
                    # New node added!
                    node = mode._get_event_data(tdw, event)
                    mode.nodes.append(node)
                    mode._last_event_node = node
                    mode.current_node_index = len(mode.nodes)-1
                    mode.select_node(mode.current_node_index, True)
                    mode._queue_draw_node(tdw, mode.current_node_index)
                    mode.phase = _Phase.ADJUST_POS
                    mode.drag_offset.start(mx, my)
        else:
            return super(Shape_Polyline, self).drag_start_cb(
                    mode, tdw, event)


class Shape_Rectangle(Shape):
    """Rectangular shape class.
    For this type of shape, always contains 4 nodes
    it is clockwise, 0 = upper-left, 1 = upper-right,
    2 = lower-right, 3 = lower-left.

    TODO : Still this class(and ellipse) does not support
    screen rotation. It is not good, so it should be
    supported in near future.

    To support screen rotation, this class remember the
    initial right-angled vertical vector, and when modifying
    rectangle corners, it follows the vertical vector of
    creation time.
    """

    name = _("Rectangle")

    def __init__(self):
        self.nx = None
        pass

    def draw_node_polygon(self, cr, tdw, nodes, selected_nodes=None,
            color=None, gradient=None,
            dx=0, dy=0, ox=0, oy=0, stroke=False, fill=True):
        """ draw the shape consist from nodes with cairo
        This method used when not only drawing editing preview
        but final drawing of the shape.
        So, We cannot use here 'mode' instance.

        :param color: color object of mypaint
        :param gradient: Gradient object. cairo.LinearGradient or something

        if both of color and gradient are null,polygon is not filled.

        :param selected_nodes: list of INDEX of selected nodes
        :param dx,dy: offset position of selected nodes
        :param ox,oy: polygon origin position
        """
        if len(nodes) >= 4:
            if selected_nodes and len(selected_nodes) >= 1:
                selidx = selected_nodes[0]
            else:
                selidx = -1

            cr.save()
            cr.set_line_width(1)
            if fill:
                if gradient:
                    cr.set_source(gradient)
                elif color:
                    cr.set_source_rgb(*color.get_rgb())

            cnt = 0
            for i, x, y in self._iter_edges_raw(nodes, dx, dy, selidx):
               #if tdw:
               #    x, y = tdw.model_to_display(x, y)
               #else:
               #    x -= ox
               #    y -= oy
                x -= ox
                y -= oy

                if tdw:
                    x, y = tdw.model_to_display(x, y)

                if cnt == 0:
                   cr.move_to(x, y)
                   sx = x
                   sy = y
                else:
                   cr.line_to(x, y)
                cnt+=1
            cr.line_to(sx, sy)

            if fill and (gradient or color):
                cr.close_path()
                cr.fill_preserve()

            if stroke:
                Shape.draw_dash(cr, 4)

            cr.restore()

    def _iter_edges(self, mode):
        """Utility method, to ease call _iter_edges_raw.
        """
        dx, dy = mode.drag_offset.get_model_offset()
        for t in self._iter_edges_raw(mode.nodes, dx, dy,
                mode.current_node_index):
            yield t

    def _iter_edges_raw(self, nodes, dx, dy, cidx):
        """Iter all edges , with offset adjustment.
        This method create to support canvas rotation. 

        :param dx, dy: offsets in MODEL COODINATE
        """
        if cidx is None:
            print("cidx is none")
            for i, cn in enumerate(nodes):
                yield (i, cn.x, cn.y)
            raise StopIteration
        print("cidx is not none")

        cn = nodes[cidx]
        cx = cn.x + dx
        cy = cn.y + dy

        nidx = (cidx + 1) % 4
        nn = nodes[nidx]
        pidx = (cidx - 1) % 4
        pn = nodes[pidx]

        if self.nx is None:
            yield (cidx, cx, cy)
            raise StopIteration

        if cidx in (0, 2):
            nx, ny = self.nx, self.ny
        else:
            nx, ny = self.ny, -self.nx

        # Dot product between
        # Normalized vector A and an arbitrary vector B
        # means 'the length of vector which is
        # project vector B to vector A.
        side_length = dot_product(nx, ny,
                                  pn.x-cx, pn.y-cy)
        px = nx * side_length + cx
        py = ny * side_length + cy
        yield (pidx, px, py)
        yield (cidx, cx, cy)  # after 'previous' node yielded,
                              # current node should be yielded.

        nx, ny = ny, -nx
        top_length = dot_product(nx, ny,
                                 nn.x-cx, nn.y-cy)
        px = nx * top_length + cx
        py = ny * top_length + cy
        yield (nidx, px, py)

        lidx = (cidx + 2) % 4
        ln = nodes[lidx]
        yield (lidx, ln.x, ln.y)

    def paint_nodes(self, cr, tdw, mode, radius):
        """Called from Overlay class
        """
        if len(mode.nodes) >= 4:
           #dx, dy = mode.drag_offset.get_display_offset(tdw)
           #sx, sy, ex, ey = self._setup_node_area(tdw, mode, dx, dy)
           #
           #for i, x, y in gui.ui_utils.enum_area_point(sx, sy, ex, ey):
            for i, x, y in self._iter_edges(mode):
                x, y = tdw.model_to_display(x, y)
                if i == mode.current_node_index:
                    color = gui.style.ACTIVE_ITEM_COLOR
                else:
                    color = gui.style.EDITABLE_ITEM_COLOR

                gui.drawutils.render_round_floating_color_chip(
                    cr=cr, x=x, y=y,
                    color=color,
                    radius=radius)


    def get_maximum_rect(self, tdw, mode, dx=0, dy=0):
        """
        Get maximum rectangle area:

        :param dx: offset of currently selected nodes.
                   if tdw is not none, dx and dy MUST be
                   display coordinate.
        :param dy: offset of currently selected nodes.
        """
        sx = ex = mode.nodes[0].x
        sy = ey = mode.nodes[0].y
       #for cn in mode.nodes[1:]:
       #    sx = min(cn.x, sx)
       #    sy = min(cn.y, sy)
       #    ex = max(cn.x, ex)
       #    ey = max(cn.y, ey)
       #
       #sx, sy, ex, ey = self._setup_node_area(tdw, mode, dx, dy)
       #if sx > ex:
       #    sx, ex = ex, sx
       #if sy > ey:
       #    sy, ey = ey, sy
       #for cn in mode.nodes[1:]:
       #    sx = min(cn.x, sx)
       #    sy = min(cn.y, sy)
       #    ex = max(cn.x, ex)
       #    ey = max(cn.y, ey)

        if tdw:
            sx , sy = tdw.model_to_display(mode.nodes[0].x, 
                                           mode.nodes[0].y)
        else:
            sx , sy = mode.nodes[0].x, mode.nodes[0].y
        ex = sx
        ey = sy

        for i, x, y in self._iter_edges(mode):
            if tdw:
                x, y = tdw.model_to_display(x, y)
            sx = min(x, sx)
            sy = min(y, sy)
            ex = max(x, ex)
            ey = max(y, ey)

        if tdw:
            margin = Shape.MARGIN + 1
            sx -= margin
            sy -= margin
            ex += margin
            ey += margin

        return (sx, sy, abs(ex-sx)+1, abs(ey-sy)+1)

    def set_area(self, mode, tdw, sx, sy, ex, ey):
        if ex < sx:
            sx, ex = ex, sx

        if ey < sy:
            sy, ey = ey, sy

        for i, x, y in gui.ui_utils.enum_area_point(sx, sy, ex, ey):
            x, y = tdw.display_to_model(x, y)
            mode.nodes[i].x = x
            mode.nodes[i].y = y

    def ensure_mode_nodes(self, mode, x, y):
        for i in xrange(4 - len(mode.nodes)):
            mode.nodes.append(self.generate_node())

        for i in xrange(4):
            mode.nodes[i].x = x
            mode.nodes[i].y = y


    def _setup_node_area(self, tdw, mode, dx, dy):
        """
        Setup nodes as rectangle.

        if tdw is valid(not None),
        dx and dy MUST be display coordinate.

        :param dx: offset of currently selected nodes.
        :param dy: offset of currently selected nodes.
        """
        sx, sy = mode.nodes[0]
        ex, ey = mode.nodes[2]

        if tdw:
            sx, sy = tdw.model_to_display(sx, sy)
            ex, ey = tdw.model_to_display(ex, ey)

        if mode.current_node_index in (0, 3):
            sx += dx
        else:
            ex += dx

        if mode.current_node_index in (0, 1):
            sy += dy
        else:
            ey += dy

        if ex < sx:
            sx, ex = ex, sx

        if ey < sy:
            sy, ey = ey, sy

        return (sx, sy, ex, ey)

    def _queue_redraw_all_nodes(self, mode):
        """Specialized version of redrawing nodes
        of this shape class.
        Redraws all nodes on all known view TDWs.

        For this type shape, moving a node would also
        move another side of node.
        i.e. if you move upper-right node vertically,
        the upper-left node would also move.
        And, if move upper-right one horizontally,
        the lower-left node would move.
        """

        radius = mode.NODE_SIZE + 1

       #for tdw in mode._overlays:
       #    offsets = mode.drag_offset.get_display_offset(tdw)
       #    area = self._setup_node_area(tdw, mode, *offsets)
       #    for i, x, y in gui.ui_utils.enum_area_point(*area):
       #        tdw.queue_draw_area(x-radius, y-radius,
       #                            radius*2,
       #                            radius*2)
        for tdw in mode._overlays:
            for i, x, y in self._iter_edges(mode):
                x, y = tdw.model_to_display(x, y)
                tdw.queue_draw_area(x-radius, y-radius,
                                    radius*2,
                                    radius*2)

    def button_press_cb(self, mode, tdw, event):
        mx, my = tdw.display_to_model(event.x, event.y)

        if mode.phase in (_Phase.ADJUST,):
            self.nx = None
            if mode.zone == _EditZone.CONTROL_NODE:
                # Grabbing a node...
                button = event.button
                # normal move node start
                mode.phase = _Phase.ADJUST_POS
                mode.selected_nodes = (mode.current_node_index, )

                # FALLTHRU: *do* start a drag

            elif mode.zone == _EditZone.EMPTY_CANVAS:
                if (len(mode.nodes) == 4):
                    mode.accept_button_cb(tdw)
                self.ensure_mode_nodes(mode, mx, my)

            # FALLTHRU: *do* start a drag

    def button_release_cb(self, mode, tdw, event):

        # Here is 'button_release_cb',which called
        # prior to drag_stop_cb.
        # so, in this method, changing mode._phase
        # is very special case.
        if mode.phase == _Phase.PLACE_NODE:
            mode._queue_redraw_item(tdw)
            mode.phase = _Phase.ADJUST

    def drag_start_cb(self, mode, tdw, event):
        # Basically,all sections should do fall-through.
        mx, my = tdw.display_to_model(event.x, event.y)

        if mode.phase == _Phase.ADJUST:

            if mode.zone == _EditZone.EMPTY_CANVAS:
                if event.state != 0:
                    # To activate some mode override
                    mode._last_event_node = None
                    return Shape.CALL_ANCESTER_HANDLER
                else:
                    # New node added!
                    mode.current_node_index=0
                    mode.select_node(mode.current_node_index, True)
                    mode.drag_offset.start(mx, my)
                    mode._queue_redraw_item(tdw)
                    self._queue_redraw_all_nodes(mode)

        elif mode.phase == _Phase.ADJUST_POS:
            if len(mode.selected_nodes) > 0:
                mode.drag_offset.start(mx, my)


    def drag_update_cb(self, mode, tdw, event, dx, dy):

        mx, my = tdw.display_to_model(event.x, event.y)

        if mode.phase == _Phase.ADJUST:
            self._queue_redraw_all_nodes(mode)
            mode._queue_redraw_item(tdw)
            mode.drag_offset.end(mx, my)

            self.nx, self.ny = normal(
                                mode.nodes[0].x,
                                mode.nodes[0].y,
                                mode.nodes[3].x,
                                mode.nodes[3].y
                               )

            self._queue_redraw_all_nodes(mode)
            mode._queue_redraw_item(tdw)
        elif mode.phase == _Phase.ADJUST_POS:
            if len(mode.selected_nodes) > 0:
                mode._queue_redraw_item(tdw)
                self._queue_redraw_all_nodes(mode)
                mode.drag_offset.end(mx, my)
                self._queue_redraw_all_nodes(mode)
                mode._queue_redraw_item(tdw)

    def drag_stop_cb(self, mode, tdw):
        if mode.phase == _Phase.ADJUST:
            # For this class, ADJUST phase means
            # "Initially set up the shape"
           #sx, sy = tdw.display_to_model(
           #        mode.start_x, mode.start_y)
            sx, sy = mode.start_x, mode.start_y
            dx, dy = mode.drag_offset.get_display_offset(tdw)

            ex = sx + dx
            ey = sy + dy

            self.set_area(mode, tdw, sx, sy, ex, ey)

            self.nx, self.ny = gui.linemode.normal(
                                mode.nodes[0].x,
                                mode.nodes[0].y,
                                mode.nodes[3].x,
                                mode.nodes[3].y)

            mode._queue_redraw_item(tdw)
            self._queue_redraw_all_nodes(mode)
            mode._queue_draw_buttons()
            mode._reset_adjust_data()

        elif mode.phase == _Phase.ADJUST_POS:

            dx, dy = mode.drag_offset.get_model_offset()
            # Move entire rectangle, when 4 nodes selected.
            if len(mode.selected_nodes) == 4:
                for i in xrange(4):
                    cn = mode.nodes[i]
                    cn.move(cn.x + dx, cn.y + dy)
            else:
                # Otherwise, only one node could move.
               #dx, dy = mode.drag_offset.get_display_offset()
               #sn = mode.nodes[0]
               #en = mode.nodes[2]
               #sx, sy = tdw.model_to_display(sn.x, sn.y)
               #ex, ey = tdw.model_to_display(en.x, en.y)
               #
               #if mode.current_node_index in (0, 3):
               #    sx += dx
               #else:
               #    ex += dx
               #
               #if mode.current_node_index in (0, 1):
               #    sy += dy
               #else:
               #    ey += dy
               #
               #self.set_area(mode, tdw, sx, sy, ex, ey)
                cn = mode.nodes[mode.current_node_index]
                cn.x += dx
                cn.y += dy

                nn = mode.nodes[(mode.current_node_index+1)%4]
                pn = mode.nodes[(mode.current_node_index-1)%4]

                if mode.current_node_index in (0, 2):
                    nx, ny = self.nx, self.ny
                else:
                    nx, ny = self.ny, -self.nx

                # Dot product between
                # Normalized vector A and an arbitrary vector B
                # means 'the length of vector which is
                # project vector B to vector A.
                side_length = dot_product(nx, ny,
                                          pn.x-cn.x, pn.y-cn.y)
                pn.x = nx * side_length + cn.x
                pn.y = ny * side_length + cn.y

                nx, ny = ny, -nx
                top_length = dot_product(nx, ny,
                                         nn.x-cn.x, nn.y-cn.y)
                nn.x = nx * top_length + cn.x
                nn.y = ny * top_length + cn.y

               #if mode.current_node_index == 0:
               #    cur = mode.nodes[0]
               #    follow_p = mode.nodes[3]
               #    follow_n = mode.nodes[1]
               #    cur.x += dx
               #    cur.y += dy
               #    follow_p.x += dx
               #    follow_n.y += dy





            mode.drag_offset.reset()
            mode._queue_redraw_item(tdw)
            mode._queue_draw_buttons()
            self._queue_redraw_all_nodes(mode) # self method
            mode.phase = _Phase.ADJUST
            mode._reset_adjust_data()



class Shape_Ellipse(Shape_Rectangle):

    name = _("Ellipse")

    def __init__(self):
        pass

    def draw_node_polygon(self, cr, tdw, nodes, selected_nodes=None,
            color=None, gradient=None,
            dx=0, dy=0, ox=0, oy=0, stroke=False, fill=True):
        """ draw cairo polygon
        :param color: color object of mypaint
        :param gradient: Gradient object. cairo.LinearGradient or something

        if both of color and gradient are null,polygon is not filled.

        :param selected_nodes: list of INDEX of selected nodes
        :param dx,dy: offset position of selected nodes
        :param ox,oy: polygon origin position
        """
        if len(nodes) >= 4:
            if selected_nodes and len(selected_nodes) >= 1:
                selidx = selected_nodes[0]
            else:
                selidx = -1

            if tdw:
                sx, sy = tdw.model_to_display(*nodes[0])
                ex, ey = tdw.model_to_display(*nodes[2])
            else:
                sx, sy = nodes[0]
                ex, ey = nodes[2]
                sx -= ox
                ex -= ox
                sy -= oy
                ey -= oy


            cr.save()
            cr.set_line_width(1)
            if fill:
                if gradient:
                    cr.set_source(gradient)
                elif color:
                    cr.set_source_rgb(*color.get_rgb())

            if selidx > -1:
                if selidx in (0, 3):
                    sx += dx
                else:
                    ex += dx

                if selidx in (0, 1):
                    sy += dy
                else:
                    ey += dy

            w = abs(ex - sx) + 1
            h = abs(ey - sy) + 1

            if sx > ex:
                sx = ex

            if sy > ey:
                sy = ey

            # XXX This 'emulated ellipse' is not circle actually
            # ... but almost okay?
            # code from:
            # http://stackoverflow.com/questions/14169234/the-relation-of-the-bezier-curve-and-ellipse
            hw = math.floor(w / 2.0)
            hh = math.floor(h / 2.0)
            # To avoid arithmetic error, not use w, but hw * 2.0
            tw = (hw * 2.0) * (2.0 / 3.0)
            x = sx + hw
            y = sy + hh

            cr.move_to(x, y - hh);
            cr.curve_to(x + tw, y - hh, x + tw, y + hh, x, y + hh)
            cr.curve_to(x - tw, y + hh, x - tw, y - hh, x, y - hh)

            if fill and len(nodes) > 2 and (gradient or color):
                cr.close_path()
                cr.fill_preserve()

            if stroke:
                Shape.draw_dash(cr, 4)

            cr.restore()

if __name__ == '__main__':

    pass


