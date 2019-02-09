#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of MyPaint.  
# Copyright (C) 2013-2016 by the MyPaint Development Team.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


## Imports

import math
import logging
logger = logging.getLogger(__name__)
import weakref
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

from gi.repository import Gtk, Gdk
from gettext import gettext as _
from gi.repository import GLib

import lib.helpers
import lib.mypaintlib
import lib.surface
import gui.mode
import gui.cursor
import gui.overlays
import gui.ui_utils
import gui.style
import lib.command 
from gui.oncanvas import *
from gui.linemode import *
import gui.widgets 
from gui.curve import CurveWidget

## Module constants
N = int(lib.mypaintlib.TILE_SIZE)
 
## Function defs

# To know whether a point is inside triangle or not.
# https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
def _sign (p1, p2, p3):
    """Get sign from points(nodes).
    """
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)


def _point_in_triangle (x, y, v1, v2, v3):
    b1 = _sign(pt, v1, v2) < 0.0
    b2 = _sign(pt, v2, v3) < 0.0
    b3 = _sign(pt, v3, v1) < 0.0
    return ((b1 == b2) and (b2 == b3))

def _bitblt(dstbuf, dx, dy, w, h, srcbuf, sx, sy):
    """Bitblt numpy tile
    """
    dstbuf[dy:dy+h, dx:dx+w] = srcbuf[sy:sy+h, sx:sx+w]

## Class defs

class _Phase:
    """Enumeration of the states that an AdjustLayerMode can be in"""
    INITIAL = 0
    CREATE_AREA = 1
    ADJUST_AREA = 2
    MOVE_NODE = 3
    MOVE_AREA = 4
    MOVE_RIDGE = 5
    ROTATE_AREA = 6
    ACTION = 7

class _ActionButton(ActionButtonMixin):
    """Enumeration for the action button of AdjustMode
    """
    EDIT=2 # To toggle editing source rectangle again.

class _EditZone(EditZoneMixin):
    """Enumeration for the Oncanvas-item of AdjustMode
    """
    CONTROL_HANDLE = 3 #: Control handle index to manipulate edge of selection, 
                       #  not all mixin user use this constant.
    
class _Operation:
    """Enumeration of the operation that an AdjustLayerMode should do"""
    MOVE = 0 # Actually, cut and copy (in target layer)
    COPY = 1 # Copy area, usually into another layer.
    TRANSFORM = 2 # Transform and copy area, usually into another layer.

class _AreaShape:
    """Enumeration of the shapes that an _Area class can be in"""
    NONE = 0 # i.e. That area is not defined yet.
    RECT = 1
    FREE = 2
    ASPECT_FIXED = 3 # Aspect fixed rectangle

    LABELS = {
        1 : "Rectangle",
        3 : "Keep aspect rectangle",
        2 : "Free Transform"
    }

class _Prefs:
    """Constants of preferences.
    """
    TRANSFORM_METHOD_PREF = 'AdjustMode.transform_method'

    DEFAULT_TRANSFORM_METHOD = _AreaShape.ASPECT_FIXED
    
class _Node:
    """Different from _Node of inktool.py, this is class because
    class _Area has 4 nodes everytime and frequently rewrite it.
    And we will need matrix multiplication, so use numpy.array
    for nodes.
    
    This class always contains values in Model coordinate.
    """
        
    def __init__(self, x=0.0, y=0.0):
        self._v = np.array( (x, y, 1) )
        
    def set_display_coord(self, tdw, sx, sy):
        """Utility method."""
        self.set(*tdw.display_to_model(sx, sy))
    
    def get_display_coord(self, tdw):
        """Utility method."""
        return tdw.model_to_display(self.x, self.y) 
        
    @property
    def x(self):
        return self._v[0]
    @property
    def y(self):
        return self._v[1]

    def set(self, x, y):
        self._v[0] = x
        self._v[1] = y

    def __getitem__(self, axis):
        return self._v[axis]
        
    def multiply(self, mat, ret):
        mat.dot(self._v, ret)
        
   #def __str__(self):
   #    v = self._v
   #    return "vec %.4f, %.4f" % (v[0], v[1])

    def debug_out(self):
        print(self), 
        print(self._v)

class _Area:
    """This class contains model coordinate rectangle
    of target area.
    """
    
    # Node index, in clockwise.
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_RIGHT = 2
    BOTTOM_LEFT = 3
    
    # Node index to share movement, when the Area is rectangular shape.
    _NODE_SHARE_X = (3, 2, 1, 0)
    _NODE_SHARE_Y = (1, 0, 3, 2)
    
    _HIT_MARGIN = 4
        
    # TODO all nodes should be rewritten with one np.array.
    # TODO all mat should be rewritten with np.matrix
    def __init__(self):
        self._nodes = (
            _Node(),
            _Node(),
            _Node(),
            _Node()
        )
        self._tmp_vec = np.array( (0, 0, 1) )
        self._shape = _AreaShape.NONE
        #self._mat = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 0))) # Z never used.
        self._mat = np.matrix(((1, 0, 0), (0, 1, 0), (0, 0, 1))) # Z never used.
        self._movemat = np.matrix(((1, 0, 0), (0, 1, 0), (0, 0, 1))) # Z never used.
        self._rotmat = np.matrix(((1, 0, 0), (0, 1, 0), (0, 0, 1))) # Z never used.
        self._rev_mat = None
        self._rotate = 0.0
        self._selected_idx = None
        self._selected_handle_idx = None

        self._init_offsets(0, 0)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, new_shape):
        self._shape  = new_shape

    @property
    def selected_node_index(self):
        return self._selected_idx 

    @property
    def selected_handle_index(self):
        return self._selected_handle_idx 

    def select_handle_index(self, index):
        self._selected_idx = None
        self._selected_handle_idx = index

    def get_node(self, idx):
        return self._nodes[idx]

        
    def _get_transformed_model(self, mx, my):
        """Convert model coordinate point (sx, sy) into transformed 
        local coodinate.
        
        :param mx, my: a point of model coordinate.
        """
        
        t = self._tmp_vec
        t[0], t[1] = mx, my
        if self._rotate != 0.0:
            t *= self._rotmat
        return (t[0], t[1])
    
   #def _setup_rotate_matrix(self):
   #    # Rotate around origin
   #    r = self._rotate 
   #    if r != 0.0:
   #        ox = self._cx - (self._width / 2)
   #        oy = self._cy - (self._height / 2)
   #        mm = self._movemat
   #        mm[0, 2] = -ox
   #        mm[1, 2] = -oy
   #                
   #        c = math.cos(r)
   #        s = math.sin(r)
   #        rm = self._rotmat
   #        rm[0, 0] = c
   #        rm[0, 1] = -s
   #        rm[0, 2] = ox # Move back nodes from origin.
   #        rm[1, 0] = s
   #        rm[1, 1] = c
   #        rm[1, 2] = oy
   #        
   #        fm = self._mat
   #        np.multiply(mm, rm, fm)       
   #        self._rev_mat = np.linarg.inv(fm)
                
    def _justify(self):
        """Justfying as area shape. 
        This would be called AFTER the selected vector 
        dragged (and mouse button released).
        
        """
        tl, tr, br, bl = self._nodes
        if self.shape in (_AreaShape.RECT, _AreaShape.ASPECT_FIXED):
            # Currently justfy only square shape.
            min_x = min(min(tr.x, br.x), min(tl.x, bl.x))
            min_y = min(min(tr.y, br.y), min(tl.y, bl.y))
            max_x = max(max(tr.x, br.x), max(tl.x, bl.x))
            max_y = max(max(tr.y, br.y), max(tl.y, bl.y))
            tl.set(min_x, min_y)
            tr.set(max_x, min_y)
            br.set(max_x, max_y)
            bl.set(min_x, max_y)
           #print("justified %d, %d, %d, %d" % (min_x, min_y, max_x, max_y))

    def iter_display_node(self, tdw):
        """Get display coordinates of a node.
        Actually we can get display coords from each nodes,
        but it does not contain dragging offset.
        Therefore you must use this method for displaying
        nodes on Overlay during drag.

        [TODO]  This should be iterator when all nodes are one 2d array.
        """
        ox, oy = self._get_offsets()
        n = self._nodes
        hi = self._selected_handle_idx
        si = self._selected_idx

        if hi == 4:
            # Centerpoint handle is dragged!
            for cn in n:
                x, y = cn.get_display_coord(tdw)
                yield (x+ox, y+oy)
        elif hi is not None:
            # Moving handle. this should be prior to
            # other node constraints.

            # Current index is `hi`
            pidx = self.get_iter_starting_index() # Prev index
            nidx = (pidx + 2) % 4  # Next index
            oidx = (pidx + 3) % 4  # Opposite index 

            # As first, yield `previous` oppsite node
            # following clockwise direction.
            yield n[pidx].get_display_coord(tdw)

            ol = vector_length(ox, oy)
            cn = n[hi]
            nn = n[nidx]
            x1, y1 = cn.get_display_coord(tdw)
            x2, y2 = nn.get_display_coord(tdw)
            l, ux, uy = length_and_normal(x1, y1, x2, y2)
            # To know moving direction, utilize cross product with
            # the normal vector of the ridge.
            cp = cross_product(ux, uy, ox, oy)
            if cp > 0:
                ol *= -1

            # We need right-angled normal, so exchange x and y.
            ux *= -1
            yield (x1 + uy * ol,
                   y1 + ux * ol) 

            yield (x2 + uy * ol,
                   y2 + ux * ol) 

            yield n[oidx].get_display_coord(tdw)

        elif si is not None:
            # Current index is si
            pidx = self.get_iter_starting_index() # Prev index
            nidx = (pidx + 2) % 4  # Next index
            oidx = (pidx + 3) % 4  # Opposite index 

            pn = n[pidx]
            cn = n[si]
            nn = n[nidx]
            on = n[oidx]

            x, y = cn.get_display_coord(tdw)
            x1, y1 = on.get_display_coord(tdw)

            shape = self.shape
            if shape in (_AreaShape.RECT, _AreaShape.ASPECT_FIXED):
                if shape == _AreaShape.ASPECT_FIXED:
                    ol, dx, dy = length_and_normal(x1, y1, self._drag_x, self._drag_y)
                    l, ux, uy = length_and_normal(x1, y1, x, y)
                    if l == 0: 
                        # Initial state. Actually there is no any rectangle
                        # defined, so there is no diagnol line.
                        # Just add offset.
                        x += ox
                        y += oy
                    else:
                        r = get_radian(ux, uy, dx, dy)
                        if r > math.pi * 0.5:
                            ol *= -1
                        x = x1+ux*ol
                        y = y1+uy*ol
                else:
                    x += ox
                    y += oy

                px, py = pn.get_display_coord(tdw)
                nx, ny = nn.get_display_coord(tdw)

                if si % 2 == 0:
                    # Share x with previous node,
                    # and Share y with next node.
                    yield (x, py) # Previous
                    yield (x, y)  # Current
                    yield (nx, y) # Next
                    yield (x1, y1) # Opposite
                else:
                    # Share y with previous node,
                    # and Share x with next node.
                    yield (px, y) # Previous
                    yield (x, y)  # Current
                    yield (x, ny) # Next
                    yield (x1, y1) # Oppsite

            elif shape == _AreaShape.FREE:
                yield pn.get_display_coord(tdw)
                yield (x+ox, y+oy)
                yield nn.get_display_coord(tdw)
                yield (x1, y1)
            else:
                assert False # UNDEFINED CASE. DO NOT GO INTO HERE.
        else:
            # No any manipulation done.
            # Just enum nodes and yield it.
            for cn in n:
                yield (cn.get_display_coord(tdw))

    def iter_display_handle(self, tdw):
        """Iterate display coords of `handle controller` 

        `handle controller` is placed at center point of
        each ridge and area.
        That controllers are indexed from top, right, bottom, left, 
        and center of area.

        """
        n = self._nodes
        cpt = []
        px, py = None, None
        for dx, dy in self.iter_display_node(tdw):
            if px is not None:
                l, x, y = length_and_normal(px, py, dx, dy)
                l *= 0.5
                cx = px+(x*l)
                cy = py+(y*l)
                cpt.append( (cx, cy) )
                yield (cx, cy)
            else:
                ox = dx
                oy = dy
            px = dx
            py = dy

        l, x, y = length_and_normal(px, py, ox, oy)
        l *= 0.5
        cx = px+(x*l)
        cy = py+(y*l)
        cpt.append( (cx, cy) )
        yield (cx, cy)


      # cn = n[0]
      # dx1, dy1 = cn.get_display_coord(tdw)
      # cpt = []
      # for i in range(4):
      #     nn = n[(i+1) % 4]
      #     dx2, dy2 = nn.get_display_coord(tdw)
      #     l, x, y = length_and_normal(dx1, dy1, dx2, dy2)
      #     l *= 0.5
      #     cx = dx1+(x*l)
      #     cy = dy1+(y*l)
      #     cpt.append( (cx, cy) )
      #     yield (cx, cy)
      #     cn = nn
      #     dx1, dy1 = dx2, dy2

        # Get centerpoint from cross products
        a, b, c, d = cpt
        ax, ay = a
        cx, cy = c
        avx = cx - ax
        avy = cy - ay
        cp1 = cross_product(avx, avy, b[0]-ax, b[1]-ay)
        cp2 = cross_product(avx, avy, d[0]-ax, d[1]-ay)
        if cp1 != 0 and cp2 != 0:
            ratio = abs((cp1 / cp2) * 0.5)
            l, x, y = length_and_normal(ax, ay, cx, cy)
            yield (ax+x*l*ratio, ay+y*l*ratio)
        else:
            yield (None, None)

    def get_iter_starting_index(self):
        """To get the first node index for iter_display_node method.
        This is also used in `end` method and overlay class.

        In iter_display_node method, it is not sure that the returned 
        (yielded) node is start from _node[0].
        """
        idx = 0
        hi = self._selected_handle_idx
        si = self._selected_idx
        if (hi is not None and hi != 4):
            # Handle index is valid, and not center handle.
            idx = (hi - 1) % 4
        elif si is not None:
            idx = (si - 1) % 4
        return idx

    def get_node_index_from_pos(self, tdw, sx, sy):
        """Get a node `index` from display coordinate.
        This method does not consider offsets.
        
        :param tdw: TiledDrawWidget,to get display coordinate of nodes.
        :param sx, sy: Clicked point in display
        :return: index if hit test passed. otherwise, return None.
        """
        margin = self._HIT_MARGIN
        for i, cn in enumerate(self._nodes):
            mx, my = self._get_transformed_model(cn.x, cn.y)
            #t[0], t[1] = cn.x, cn.y
            tx, ty = tdw.model_to_display(mx, my)
            if abs(sx - tx) <= margin and abs(sy - ty) <= margin:
                return i

    def get_node_from_pos(self, tdw, sx, sy):
        """Get a node from display coordinate.
        This method does not consider offsets.
        
        :param tdw: TiledDrawWidget,to get display coordinate of nodes.
        :param sx, sy: Clicked point in display
        """
        idx = self.get_node_index_from_pos(tdw, sx, sy)
        if idx is not None:
            return self._nodes[idx]
        return None

    def get_handle_index_from_pos(self, tdw, sx, sy):
        margin = self._HIT_MARGIN
       #n = self._nodes
       #nn = n[0]
       #dx1, dy1 = tdw.model_to_display(
       #    *self._get_transformed_model(nn.x, nn.y)
       #)
       #for i in range(len(n)):
       #    nn = n[(i+1)%4]
       #    dx2, dy2 = tdw.model_to_display(
       #        *self._get_transformed_model(nn.x, nn.y)
       #    )
       #    l, x, y = length_and_normal(dx1, dy1, dx2, dy2)
       #    l *= 0.5
       #    hx = dx1 + (x * l) 
       #    hy = dy1 + (y * l) 
       #    if abs(sx - hx) <= margin and abs(sy - hy) <= margin:
       #        return i
       #    dx1, dy1 = dx2, dy2
        for i, pos in enumerate(self.iter_display_handle(tdw)):
            dx, dy = pos
            if dx is not None and abs(sx-dx) <= margin and abs(sy-dy) <= margin:
                return i

    def select_node(self, node):
        n = self._nodes
        assert node in n
        self._selected_idx = n.index(node)
        self._selected_handle_idx = None

    def _init_offsets(self, sx, sy):
        self._start_x = sx
        self._start_y = sy
        self._drag_x = sx
        self._drag_y = sy

    def _get_offsets(self):
        return (self._drag_x-self._start_x, self._drag_y-self._start_y)

    def set_drag_pos(self, tdw, sx, sy): 
        self._drag_x = sx
        self._drag_y = sy
    
   #def refresh_nodes(self, tdw):
   #    """Refresh nodes with currentlt set offsets. This should be called 
   #    during mouse drag ongoing. (drag_update_cb).
   #    
   #    And when button released, `justfy` method would be called.
   #    """
   #    return
   #    ox, oy = self._offset_x, self._offset_y
   #    self._init_offsets() # Clear offset
   #    for i, cn in enumerate(self._nodes):
   #        if cn.selected:
   #            x, y = cn.get_display_coord(tdw)
   #            x += ox
   #            y += oy
   #            
   #            # Convert display coordinate into 
   #            # model coordinate, along with transform matrix(if used)
   #            mx, my = tdw.display_to_model(x, y)
   #            #cn.x, cn.y = self._get_transformed_model(mx, my) # does not work
   #            cn.set(*self._get_transformed_model(mx, my))
   #            
   #            if self._shape == _AreaShape.RECT:                        
   #                # Share coordinate with another node.
   #                x_share_node = self._nodes[self._NODE_SHARE_X[i]]
   #                y_share_node = self._nodes[self._NODE_SHARE_Y[i]]
   #                
   #                x_share_node.x = cn.x
   #                y_share_node.y = cn.y
   #                return
    
    def reset(self):
        self._init_offsets(0, 0) # Clear offset
        self._rotate = 0.0
        self.shape = _AreaShape.NONE
        # Matrics are re-setup when _setup_rotate_matrix called.
        
    def start(self, tdw, sx, sy):
        """Start dragging to create selected area
        """
        if self.shape == _AreaShape.NONE:
            mx, my = tdw.display_to_model(sx, sy)
            n = self._nodes
            for cn in n:
                cn.set(mx, my)
        
            # When creating selected area,
            # Origin is always self._node[0], so the movable node is
            # always diagonal one, i.e. self._node[2]
            self._selected_idx = 2
            self.shape = _AreaShape.RECT

        self._init_offsets(sx, sy)
        
    def end(self, tdw):
        """End dragging and justify(finalize) selected area.

        Offsets already added in refresh_nodes()
        """
        n = self._nodes
        # Applying moving offsets into nodes.
        # But some node positions might affect other nodes.
        # So we cannot apply them in single loop, 
        # cache it once into `lst` and apply it later.
        lst = []


        # Starting node index is exactly same as 
        # in `iter_display_node` method.
        idx = self.get_iter_starting_index()

        for pos in self.iter_display_node(tdw):
            lst.append(pos)
        for pos in lst:
            n[idx].set(*tdw.display_to_model(*pos))
            idx = (idx + 1) % 4

        self._init_offsets(0, 0)
        self._selected_idx = None
        self._justify()

    def get_update_rect(self, tdw):
        """Utility method, to get (x, y, w, h) of current display area.
        That area might be scaled/rotated, so different from
        self._cx/_cy/_width/_height attributes, which are in model coodinate.
        
        also, returning display rect includes control node gui size.

        :return: tuple of maximum bbox with margin, in display coordinate.
        """
       #min_x, min_y = self.get_display_node(tdw, 0)
       #max_x, max_y = min_x, min_y
       #margin = self._HIT_MARGIN + 3 # + 3 for cairo drawing margin
       #for i in range(1, 4):
       #    x, y = self.get_display_node(tdw, i)
       #    min_x = min(x, min_x)
       #    min_y = min(y, min_y)
       #    max_x = max(x, max_x)
       #    max_y = max(y, max_y)

        min_x = None
        margin = self._HIT_MARGIN + 3 # + 3 for cairo drawing margin
        for x, y in self.iter_display_node(tdw):
            if min_x is None:
                min_x = x
                min_y = y
                max_x = x
                max_y = y
            else:
                min_x = min(x, min_x)
                min_y = min(y, min_y)
                max_x = max(x, max_x)
                max_y = max(y, max_y)
        return (min_x - margin, 
                min_y - margin, 
                max_x - min_x + (margin * 2) + 1, 
                max_y - min_y + (margin * 2) + 1)
    
    def is_inside(self, mx, my):
        """Tells whether the point(mx, my) is inside area or not.
        
        :param mx, my: The (clicked) point, in MODEL coordinate.
        """
        n0, n1, n2, n3 = self._nodes
        return (_point_in_triangle(mx, my, n0, n1, n2)
                    or  _point_in_triangle(mx, my, n2, n3, n0))

    def get_min_x(self):
        """Get minimum x, with in model
        """
        return self._get_min(0)
    def get_max_x(self):
        """Get maximum x, with in model
        """
        return self._get_max(0)
    def get_min_y(self):
        """Get minimum y, with in model
        """
        return self._get_min(1)
    def get_max_y(self):
        """Get maximum y, with in model
        """
        return self._get_max(1)

    def _get_min(self, a):
        n = self._nodes
        return min(min(min(n[0][a], n[1][a]), n[2][a]), n[3][a])
    def _get_max(self, a):
        n = self._nodes
        return max(max(max(n[0][a], n[1][a]), n[2][a]), n[3][a])
    
    def get_bbox(self):
        sx = int(self.get_min_x())
        sy = int(self.get_min_y())
        ex = int(self.get_max_x())
        ey = int(self.get_max_y())
        return (sx, sy, ex-sx+1, ey-sy+1)

    def get_center_pos(self, tdw):
        """Get center point of this area.

        :param tdw: The TiledDrawWidget, to get display coord.
                    If this is None, return model coord.
        """
        if self.shape != _AreaShape.NONE:
            n = self._nodes
            cpt = []
            cn = n[0]
            for i in range(len(n)):
                nn = n[(i+1)%4]
                # Get center points of each ridge
                lt, tx, ty =length_and_normal(cn.x, cn.y, nn.x, nn.y)
                cpt.append( (cn.x+(tx*lt*0.5), cn.y+(ty*lt*0.5)) )
                cn = nn
            
                


    def get_nodes_as_tuple(self):
        """Return nodes as tuple.
        With this, we can deep copy of node datas at once.
        """
        a, b, c, d = self._nodes
        return (
            (a[0], a[1]),
            (b[0], b[1]),
            (c[0], c[1]),
            (d[0], d[1])
        )

    def copy_to(self, area):
        for i, cn in enumerate(self._nodes):
            dn = area.get_node(i)
            dn.set(cn.x, cn.y)

    def debug_show_nodes(self, nodes=None, msg=None, offsets=False, tdw=None):
        if nodes is None:
            nodes = self._nodes
        if msg is not None:
            print("%s " % msg),
        if tdw is not None:
           #for i in range(len(nodes)):
            for pos in self.iter_display_node(tdw):
                print("(%d, %d) " % pos),
        else:
            for cn in nodes:
                print(cn),
                print("(%d, %d) " % (cn.x, cn.y)),
        if offsets:
            print(",offsets (%d, %d) " % (self._offset_x, self._offset_y)),
        print("")
   
## Interaction modes for making lines

class AdjustLayerMode(gui.mode.ScrollableModeMixin,
                      gui.mode.DragMode):
    """Oncanvas area selection mode"""

    ## Class constants

    ACTION_NAME = "AdjustLayerMode"
    _OPTIONS_PRESENTER = None
    LINE_WIDTH = 1
    
    _HIT_MARGIN = 4
    
    ## Class configuration.

    permitted_switch_actions = set([
        "PanViewMode",
        "ZoomViewMode",
        "RotateViewMode",
    ])

    pointer_behavior = gui.mode.Behavior.PAINT_CONSTRAINED
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW
    
    _cursor = None
    
    _DEFAULT_CURSOR = None
    _ON_NODE_CURSOR = None
    _ON_RIDGE_CURSOR = None

    ## Action buttons
    buttons = {
        ActionButtonMixin.ACCEPT : ('mypaint-ok-symbolic', 
            'accept_button_cb'), 
        ActionButtonMixin.REJECT : ('mypaint-trash-symbolic', 
            'reject_button_cb'),
        _ActionButton.EDIT : ('mypaint-edit-symbolic', 
            'edit_button_cb') 
    }

    @property
    def active_cursor(self):
        return self._cursor  
    
    @classmethod
    def get_name(cls):
        return _(u"Adjust parts")

    def get_usage(self):
        return _(u"Copy, scale, transform parts of currently editing picture.")

    @property
    def inactive_cursor(self):
        return self._cursor  

    unmodified_persist = True
    permitted_switch_actions = set(
        ['RotateViewMode', 'ZoomViewMode', 'PanViewMode']
        + gui.mode.BUTTON_BINDING_ACTIONS)

    ## Initialization

    def __init__(self, **kwds):
        """Initialize"""
        super(AdjustLayerMode, self).__init__(**kwds)
        self.app = None       
        self._overlays = {}  # keyed by tdw
        self._source = _Area()
        self._target = _Area()
        self.current_button_id = None
        self.target_node_index = None
        self.target_handle_index = None
        self.zone = EditZoneMixin.EMPTY_CANVAS
        self.forced_button_pos = None
        self._use_visible = False
        self.phase = _Phase.INITIAL # This call self.reset()

    def reset(self):
        self._active_area = None
        self._source.reset()
        self._target.reset()
        self._cmd = None
        self._prev_targ_bbox = None
        self._prev_targ_nodes = None
        
    # Selection area and node related.
    @property
    def source_area(self):
        return self._source

    @property
    def target_area(self):
        return self._target
        
    @property
    def active_area(self):
        """Currently active (under operation) area.This might be None.
        """
        return self._active_area

    def _search_target_node(self, tdw, x, y):
        area = self.active_area
        if area is not None:
            return area.get_node_index_from_pos(tdw, x, y)
        return None

    # Phase related    
    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, new_phase):
        self._phase = new_phase
        if new_phase == _Phase.INITIAL:
            self.reset()

    ## Preferences and OptionsPresenter related
    @property
    def transform_method(self):
        app = self.app
        if app is not None:
            return app.preferences.get(
                _Prefs.TRANSFORM_METHOD_PREF, 
                _Prefs.DEFAULT_TRANSFORM_METHOD
            )
        return _Prefs.DEFAULT_TRANSFORM_METHOD
    
    @property
    def use_visible(self):
        app = self.app
        if app is not None:
            return app.preferences.get(
                _Prefs.USE_VISIBLE_PREF, 
                _Prefs.DEFAULT_USE_VISIBLE
            )
        return _Prefs.DEFAULT_USE_VISIBLE
                                         
    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        cls = self.__class__
        if cls._OPTIONS_PRESENTER is None:
            cls._OPTIONS_PRESENTER = _OptionsPresenter()
        return cls._OPTIONS_PRESENTER

    ## Visible GUI related
    def _ensure_cursor(self):
        cls = self.__class__
        if cls._DEFAULT_CURSOR is None:
            action_name = None #"FreehandMode" # [TODO] Dummy
            cursors = self.app.cursors
            cls._DEFAULT_CURSOR = cursors._get_overlay_cursor(
                action_name,
                gui.cursor.Name.CROSSHAIR_OPEN_PRECISE
            )                      
            cls._ON_NODE_CURSOR = cursors._get_overlay_cursor(
                action_name,
                gui.cursor.Name.MOVE_NORTHWEST_OR_SOUTHEAST
            )      
            cls._cursor = cls._DEFAULT_CURSOR                

    def _update_cursor(self, tdw):
        old_cursor = self._cursor

        if self.zone in (_EditZone.CONTROL_NODE, _EditZone.CONTROL_HANDLE):
            self._cursor = self._ON_NODE_CURSOR
        elif self.zone == EditZoneMixin.ACTION_BUTTON:
            self._cursor = self._ON_NODE_CURSOR
        else:
            self._cursor = self._DEFAULT_CURSOR
                    
        if old_cursor != self._cursor:
            tdw.set_override_cursor(self._cursor)

    def _update_zone_and_target(self, tdw, x, y):
        """Update the zone and target node under a cursor position"""
        overlay = self._ensure_overlay_for_tdw(tdw)
        new_zone = EditZoneMixin.EMPTY_CANVAS
 
        if not self.in_drag:
            if self.is_actionbutton_ready():
                new_target_node_index = None
                self.current_button_id = None
                # Test buttons for hits
                hit_dist = gui.style.FLOATING_BUTTON_RADIUS

                for btn_id in self.buttons.keys():
                    btn_pos = overlay.get_button_pos(btn_id)
                    if btn_pos is None:
                        continue
                    btn_x, btn_y = btn_pos
                    d = math.hypot(btn_x - x, btn_y - y)
                    if d <= hit_dist:
                        new_target_node_index = None
                        new_zone = EditZoneMixin.ACTION_BUTTON
                        self.current_button_id = btn_id
                        break

                # Fallthrough

            area = self._active_area
            # Test nodes for a hit, in reverse draw order
            if new_zone == EditZoneMixin.EMPTY_CANVAS:
                new_target_node_index = self._search_target_node(tdw, x, y)
                if new_target_node_index != None:
                    new_zone = EditZoneMixin.CONTROL_NODE

            # Update the prelit node, and draw changes to it
            if new_target_node_index != self.target_node_index:
                # Redrawing old target node.
                if self.target_node_index is not None:
                    self._queue_draw((area, )) # To erase

                self.target_node_index = new_target_node_index
                if self.target_node_index is not None:
                    self._queue_draw((area, )) # To draw

            # In spite of above line, still active node not found...  
            # Cursor might be on `handle`
            self.target_handle_index = None
            if self.target_node_index is None and area is not None:
                new_handle_index = area.get_handle_index_from_pos(tdw, x, y)
                if new_handle_index is not None:
                    self.target_handle_index = new_handle_index
                    new_zone = _EditZone.CONTROL_HANDLE
 
        # Update the zone, and assume any change implies a button state
        # change as well (for now...)
        if self.zone != new_zone:
            self._enter_new_zone(tdw, new_zone)
            self.zone = new_zone

    def _enter_new_zone(self, tdw, new_zone):
        """Entering new zone. This is actually postprocess of 
        _update_zone_and_target().

        This is useful when customize _update_zone_and_target()
        in deriving classes.
        """
        if new_zone == EditZoneMixin.ACTION_BUTTON:
            self._queue_draw_buttons()
        elif new_zone == EditZoneMixin.EMPTY_CANVAS:
            if self.phase != _Phase.INITIAL:
                self._queue_draw_buttons()

        if not self.in_drag:
            self._update_cursor(tdw)
    ## InteractionMode/DragMode implementation

    def enter(self, doc, **kwds):
        """Enter the mode.
        """
        super(AdjustLayerMode, self).enter(doc, **kwds)
        self.app = self.doc.app
        self._ensure_cursor()
        if not self._is_active():
            self._discard_overlays()
        self._drag_update_idler_srcid = None

        opt = self.get_options_widget()
        opt.target = self

        if lib.command.AdjustLayer.get_opencv2() is None:
            self.app.message_dialog(
                _(u"To use this tool, you need to install Python-OpenCV2."),
                Gtk.MessageType.WARNING,
                title = self.get_name()
            )

    def leave(self, **kwds):
        if not self._is_active():
            self._discard_overlays()

        # Processing unfinished command. 
        cmd = self._cmd
        if cmd is not None:
            model = self.doc.model
            model.do(cmd)
            self._cmd = None

        opt = self.get_options_widget()
        opt.target = None
        
       #if self._is_valid():
       #    returning_mode = self.doc.modes.top
       #    if hasattr(returning_mode, 'select_area_cb'):
       #        returning_mode.select_area_cb(self)

        # leave snapshot when undo stack returns this object.
        return super(AdjustLayerMode, self).leave(**kwds)

    def _is_active(self):
        for mode in self.doc.modes:
            if mode is self:
                return True
        return False

    def button_press_cb(self, tdw, event):
        if self.zone == EditZoneMixin.ACTION_BUTTON:
            self.phase = _Phase.ACTION
        else:
            super(AdjustLayerMode, self).button_press_cb(tdw, event)
    
    def button_release_cb(self, tdw, event):
        if self.phase == _Phase.ACTION:
            self._queue_draw_buttons()
            cid = self.current_button_id
            buttons = self.buttons
            assert cid is not None
            assert cid in buttons
            self._call_action_button(tdw, cid)
            # self.phase should be resetted in action button handler.
        else:
            return super(AdjustLayerMode, self).button_release_cb(tdw, event)

    def motion_notify_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        self._update_zone_and_target(tdw, event.x, event.y)

                    
        # Fall through to other behavioral mixins, just in case
        return super(AdjustLayerMode, self).motion_notify_cb(tdw, event)

    def drag_start_cb(self, tdw, event):
        self._ensure_overlay_for_tdw(tdw)
        phase = self._phase
        area = self._active_area
        sx, sy = event.x, event.y
        if phase == _Phase.INITIAL:
            src = self._source
            assert src.shape == _AreaShape.NONE
            src.start(tdw, sx, sy)
            self.phase = _Phase.CREATE_AREA
            self._active_area = src
        elif phase == _Phase.ADJUST_AREA:
            self._queue_draw_buttons() # To erase
            cn = None
            if not area.shape in (_AreaShape.NONE, ):
                cn = area.get_node_from_pos(tdw, sx, sy)
                
           #if cn is None:
           #    if targ.shape != _AreaShape.NONE:
           #        # Already target exists
           #        cn = targ.get_node_from_pos(tdw, sx, sy)
           #        self._active_area = targ
           #    
           #    if cn is None:
           #        if targ.shape == _AreaShape.NONE:
           #            # Target does not exist,or to be redefined.
           #            targ.start(tdw, sx, sy)
           #
           #            # If code reaches here, above conditional block
           #            # is not executed. so, setup _active_area.
           #            self._active_area = targ
           #    else:
           #        self.phase = _Phase.MOVE_NODE
           #        targ.select_node(cn)
                    
            if cn is not None:
                area.select_node(cn)
                area.start(tdw, sx, sy)
                self.phase = _Phase.MOVE_NODE
            else:
                if self.target_handle_index is not None:
                    area.select_handle_index(self.target_handle_index)
                    area.start(tdw, sx, sy)
                    self.phase = _Phase.MOVE_NODE

        self._queue_draw() # to start

        super(AdjustLayerMode, self).drag_start_cb(tdw, event)

    def drag_update_cb(self, tdw, event, dx, dy):
        self._ensure_overlay_for_tdw(tdw)
        self._queue_draw() # to erase
        phase = self._phase
        area = self._active_area
        if phase in (_Phase.CREATE_AREA, _Phase.ADJUST_AREA, _Phase.MOVE_NODE):
            assert area is not None
            area.set_drag_pos(tdw, event.x, event.y)
            # I placed lines of code here 
            # to process ongoing transform in idle function,
            # But it seems to be never processed during drag, even I hold 
            # pen stylus at same position. 
            # So I remove that code.
        self._queue_draw()
        return super(AdjustLayerMode, self).drag_update_cb(tdw, event, dx, dy)

   ## XXX Based on gui/layermanip.py
    def _drag_update_idler(self):
        """Processes tile moves in chunks as a background idler"""
        # Might have exited, in which case leave() will have cleaned up
        cmd = self._cmd
        if cmd is None:
            self._drag_update_idler_srcid = None
            return False
        # Terminate if asked. Assume the asker will clean up.
        if self._drag_update_idler_srcid is None:
            return False
        # Process transform, and carry on if there's more to do
        self._do_transform()
        self._drag_update_idler_srcid = None
        return False

    @gui.widgets.with_wait_cursor
    def _do_transform(self):
        """To show wait-cursor for transform operation with decorator
        """
        cmd = self._cmd
        assert cmd is not None
        targ = self._target
        targ_bbox = targ.get_bbox()
        targ_nodes = targ.get_nodes_as_tuple()
        if (self._prev_targ_bbox != targ_bbox
                or self._prev_targ_nodes != targ_nodes):
            cmd.transform_to(
                targ_bbox,
                targ_nodes,
                update_canvas=True
            )
            self._prev_targ_bbox = targ_bbox
            self._prev_targ_nodes = targ_nodes
            return True
    
    def drag_stop_cb(self, tdw):
        self._ensure_overlay_for_tdw(tdw)
        self._queue_draw()
        phase = self._phase
        area = self._active_area   
        if phase in (_Phase.CREATE_AREA, _Phase.ADJUST_AREA):
            assert area is not None
            area.end(tdw)
            if area == self._source:
                assert area is self._source
               #self._debug_create_temp_buffer()
                model = self.doc.model
                if self._use_visible:
                    layer = model.layer_stack
                    self.finalize_use_visible()
                else:
                    layer = model.layer_stack.current
                self._cmd = lib.command.AdjustLayer(
                    model, layer,
                    area.get_bbox()
                )
                targ = self._target
                area.copy_to(targ)
               #targ.shape = _AreaShape.FREE
                targ.shape = self.transform_method
                self._active_area = targ
            if phase == _Phase.CREATE_AREA:
                self.phase = _Phase.ADJUST_AREA
            self._queue_draw_buttons()

        elif phase == _Phase.MOVE_NODE:
            assert area is not None
            area.end(tdw)
            if area == self._target:
               #self._debug_show_opencv_result()
                # Ensure last modification, with idle function 
                if self._drag_update_idler_srcid is None:
                    idler = self._finalize_transform_idler
                    self._drag_update_idler_srcid = GLib.idle_add(idler)
            # With assigning None to select_handle_index,
            # Both of normal node index and handle index disabled.
            area.select_handle_index(None) 
            self.phase = _Phase.ADJUST_AREA
            self._queue_draw_buttons()
        
        return super(AdjustLayerMode, self).drag_stop_cb(tdw)

    def _finalize_transform_idler(self):
        self._drag_update_idler()
        return False # Ensure this is the last idle processing.

   ## XXX Mostly copied from gui/layermanip.py
   #def _finalize_transform_idler(self):
   #    """Finalizes everything in chunks once the drag's finished"""
   #    if self._cmd is None:
   #        return False  # something else cleaned up
   #    while self._cmd.process_transform():
   #        return True
   #    model = self._drag_active_model
   #    cmd = self._cmd
   #    tdw = self._drag_active_tdw
   #    if tdw is not None:
   #        # self._update_ui() would be called in self._drag_cleanup()
   #        tdw.set_sensitive(True)
   #    model.do(cmd)
   #    self._drag_cleanup()
   #    return False
   #
   ## XXX Mostly copied from gui/layermanip.py
   #def _drag_cleanup(self):
   #    """Final cleanup after any drag is complete"""
   #    if self._drag_active_tdw:
   #        self._update_ui()  # update may have been deferred
   #    self._drag_active_tdw = None
   #    self._drag_active_model = None
   #    self._cmd = None
   #
       #if not self.doc:# If already detached from command stack
       #    return
       #if self is self.doc.modes.top:
       #    if self.initial_modifiers:
       #        if (self.final_modifiers & self.initial_modifiers) == 0:
       #            self.doc.modes.pop()

    ## Overlays
    #  FIXME: mostly copied from gui/inktool.py
    #  Would it be better there is something mixin?
    #  (OverlayedMixin??)
    
    def _ensure_overlay_for_tdw(self, tdw):
        
        overlay = self._overlays.get(tdw)
        if not overlay:
            overlay = _Overlay(self, tdw)
            tdw.display_overlays.append(overlay)
            self._overlays[tdw] = overlay
        return overlay

    def _discard_overlays(self):
        for tdw, overlay in self._overlays.items():
            tdw.display_overlays.remove(overlay)
            tdw.queue_draw()
        self._overlays.clear()

    def _queue_draw(self, areas=None):
        """Redraws selection area"""
        if areas is None:
            areas = (self._source, self._target)
        for tdw, overlay in self._overlays.items():
            for area in areas:
                tdw.queue_draw_area(
                    *area.get_update_rect(tdw))

    ## Action button handlers
    def is_actionbutton_ready(self):
        return self._active_area is not None

    def _queue_draw_buttons(self):
        """Redraws the accept/reject buttons on all known view TDWs"""
        for tdw, overlay in self._overlays.items():
            overlay.update_button_positions()
            for id in self.buttons:
                pos = overlay.get_button_pos(id)
                if pos is None:
                    continue
                r = gui.style.FLOATING_BUTTON_ICON_SIZE
                r += max(
                    gui.style.DROP_SHADOW_X_OFFSET,
                    gui.style.DROP_SHADOW_Y_OFFSET,
                )
                r += gui.style.DROP_SHADOW_BLUR
                x, y = pos
                tdw.queue_draw_area(x-r, y-r, 2*r+1, 2*r+1)

    def _call_action_button(self, tdw, id):
        """ Call action button, from id (i.e. _ActionButton constants)

        Internally, this method get method from string 
        which is defined as class attribute of button structure
        and call it.
        """
        junk, handler_name = self.buttons[id]
        method = getattr(self, handler_name)
        method(tdw)

    def accept_button_cb(self, tdw):
        area = self._active_area
        assert area is not None
        if area is self._target:
            self._queue_draw()
            self._queue_draw_buttons()
            cmd = self._cmd
            assert cmd is not None
            self.doc.model.do(cmd)
            self.phase = _Phase.INITIAL

    def reject_button_cb(self, tdw):
        self._queue_draw()
        self._queue_draw_buttons()
        cmd = self._cmd
        assert cmd is not None
        cmd.restore_snapshot()
        self.phase = _Phase.INITIAL
        self.reset()

    def edit_button_cb(self, tdw):
        area = self._active_area
        self._queue_draw()
        cmd = self._cmd
        assert cmd is not None
        cmd.restore_snapshot()
        self._cmd = None
        if area == self._target:
            self.phase = _Phase.ADJUST_AREA
            self._active_area = self._source
            self._target.reset()

    ## Notifications from optionspresenter
    def notify_tranform_method_changed(self):
        area = self._target
        # When target area is already active, change transform method.
        # Otherwise, it would be applied at next transformation.
        if area.shape != _AreaShape.NONE:
            area.shape = self.transform_method

    def notify_use_visible_changed(self, state):
        if self._cmd is not None:
            # Cancel current transformation with pressing reject button.
            self.reject_button_cb(None) # reject_button_cb does not need tdw actually
        self._use_visible = state

    def finalize_use_visible(self):
        self._use_visible = False
        self.get_options_widget().finalize_use_visible()

    ## Temporary Buffer related
    #  Free transforming will refer source pixels over and over
    #  again.It will make effeciency problem. 
    #  So create large uniformed temporary buffer and blt tiles
    #  into it.

    def _debug_create_temp_buffer(self):
        """Create temporary linear buffer.
        Free transforming will access source pixel
        nearly random. Source tiles would be changed heavily,
        it will make severe performance impact.
        So, make linear buffer.
        """
        src = self._source
        assert src is not None
        x, y, w, h = src.get_bbox()
    
        srcdict = self._snapshot.surface_sshot.tiledict
        btx = x // N
        bty = y // N
        
        # Calc tilewise width and height.
        # We need to add one tile when the each of edge of source area
        # exceed tile border.
        tw = (w // N) + int((x + w) % N != 0) + int(x % N !=0)
        th = (h // N) + int((y + h) % N != 0) + int(y % N !=0)
    
        buf = np.zeros((th*N, tw*N, 3), 'uint8')
        alpha = np.zeros((th*N, tw*N, 1), 'uint8')
        for ty in range(th):
            cty = ty * N
            sty = ty + bty
            for tx in range(tw):
                stx = tx + btx
                if (stx, sty) in srcdict:
                    tile = srcdict[(stx, sty)].rgba
                    ctx = tx * N
                    lib.mypaintlib.opencvutil_convert_tile_to_image(
                        buf, 
                        alpha,
                        tile,
                        ctx, cty
                    )
        self._source_buf = buf
        self._source_alpha = alpha
    
       #self.debug_show_array((buf,), ('source',))

    def debug_output_array(self, buf, tw, th, filename='/tmp/test.png'):
        # XXX for debug
        test = np.zeros((th*N, tw*N, 4), 'uint8')
        test[:,:,:] = buf[:,:,:]
        import scipy.misc 
        scipy.misc.imsave(filename, test)

    def debug_show_array(self, bufs, names=None):
        from matplotlib import pyplot as plt
        for i, buf in enumerate(bufs):
            if names is not None:
                cname = names[i]
            else:
                cname = 'buf %d' % i
            plt.subplot(121+i),plt.imshow(buf),plt.title(cname)
        plt.show()

    ## Update test
    def debug_update_target(self, tdw):
        """Just test, only copying from source to dest.
        """
        model = self.doc.model
        layer = model.layer_stack.current
        s = self._source
        d = self._target
        srcdict = self._snapshot.surface_sshot.tiledict
        dstsurf = layer._surface
        sx, sy, sw, sh = s.get_bbox()
        dx, dy, dw, dh = d.get_bbox()


        tsx = sx // N
        tsy = sy // N
        tdx = dx // N
        tdy = dy // N
        psx = int(sx % N)
        psy = int(sy % N)
        pdx = int(dx % N)
        pdy = int(dy % N)
        bdx = (N - (psx - pdx)) # initial blt destination x. 
                                # Actually, this can be over tile size
                                # But we need that information to adjust
                                # tile blt location.
        bdy = (N - (psy - pdy)) # initial blt destination y

        # Destination tile width & Height. 
        # As a default, it is 2x2 and One source tile
        # blt into that 4 tiles.
        dtw = 2 
        dth = 2

        # Target size is limited with source size.
        tw = sw // N + int((dw % N) > 0)
        th = sh // N + int((dh % N) > 0)

        # Destination tile position is adjusted
        # when src-tile position is not aligned tile border and
        # src-tile pixel position is smaller than destination pixel position.
        if bdx < N:
            tdx -= 1
        elif bdx == N and psx == 0:
            # Also , when source & dest position is aligned to
            # tile-border, just blit one tile.
            dtw -= 1

        if bdy < N:
            tdy -= 1
        elif bdy == N and psy == 0:
            dth -= 1

        # Finally, modulo bdx/bdy with N
        bdx %= N
        bdy %= N

        bw = N - bdx # initial blt width
        bh = N - bdy # initial blt height

       #print("bw,bh(%d,%d), tw,th(%d, %d)" % (bw, bh, tw, th))

        for ty in range(th):
            lsw = sw
            for tx in range(tw):
                csx = tsx + tx
                csy = tsy + ty
                spos = (csx, csy)
                # Utilize srcdict to reject empty tile
                if spos in srcdict:
                    srctile = srcdict[spos]
                    # Copy single source tile into (maximum) 4 destination tiles.
                    bsy = 0
                    pdh = bh
                    pdy = bdy # Temporary destination y
                    for dty in range(dth):
                        cdy = tdy + ty + dty
                        pdx = bdx # Temporary destination x
                        bsx = 0 # Bitblt-source-x
                        pdw = bw # Initialize pdw as left tranfer width.

                        # Clipping transfer areas within destination.
                        # When pdh goes into negative, transfer cancelled.
                        oy = cdy * N
                        csy = bsy + oy
                        cey = csy + pdh
                        if csy < dy:
                            diff = dy - csy
                            bsy += diff
                            pdh -= diff
                        if cey >= dy+dh: # Not `elif` - both condition might be true when smaller than one tile.
                            diff = (dy+dh) - csy - 1
                            pdh = min(diff, pdh)

                        if pdh > 0:
                            for dtx in range(dtw):
                                cdx = tdx + tx + dtx

                                # Clipping transfer areas within destination.
                                # When pdw goes into negative, transfer cancelled.
                                ox = cdx * N
                                csx = bsx + ox 
                                cex = csx + pdw
                                if csx < dx:
                                    diff = dx - csx
                                    bsx += diff
                                    pdw -= diff
                                if cex >= dx+dw:
                                    diff = (dx+dw) - csx - 1
                                    pdw = min(diff, pdw)

                                if pdw > 0:
                                    with dstsurf.tile_request(cdx, cdy, readonly=False) as dsttile:
                                        _bitblt(dsttile, pdx, pdy, pdw, pdh, srctile.rgba, bsx, bsy)
                                # Caution: Do not use pd* variables itself for
                                # re-initialize for next loop.
                                # They might be changed in loop.
                                bsx = bw # Source x become last transferred width.
                                pdw = N - bw
                                pdx = 0 # Destination x become left of next tile column.

                        # Caution: Do not use pd* variables itself for
                        # re-initialize for next loop.
                        bsy = bh
                        pdh = N - bh 
                        pdy = 0 # Destination y become top of next tile line.

        # Update canvas
        redraw_bbox = lib.helpers.Rect(dx, dy, dw, dh)
        dstsurf.notify_observers(*redraw_bbox)

    def _debug_show_opencv_result(self):
        if cv2 is None:
            self.doc.app.message_dialog(
                "OpenCV2 is not ready.(not installed yet?)", 
                Gtk.MessageType.Error
            )
            return
        #img = cv2.imread('sudokusmall.png')
        s = self._source
        sx, sy, sw, sh = s.get_bbox()
       #pos_src = [[sx, sy],[sx+sw, sy],[sx, sy+sh],[sw, sh]]
    
        t = self._target
        dx, dy, dw, dh = t.get_bbox()
        dtx = dx // N # Destination tile position, in tile unit.
        dty = dy // N
        # Make destination width & height with tile-border aligned.
        tw = dw // N + int(((dx + dw) % N) != 0)
        th = dh // N + int(((dy + dh) % N) != 0)
        dw = tw * N
        dh = th * N
    
        # Modifying destination box, as in source rectangle.
       #stx = sx // N # Source tile position, in tile unit.
       #sty = sy // N
        ox = dtx * N  # Source tile position, in pixel(model) unit.
        oy = dty * N
        pos_dst = []
        for i in range(4):
            n = t.get_node(i)
            pos_dst.append((n.x-ox, n.y-oy))
    
        # Then, adjust source position as local of source rectangle.
        # It is tile aligned.
        sx %= N
        sy %= N
    
        pos_src = [[sx, sy],[sx+sw, sy],[sx+sw, sy+sh],[sx, sy+sh]]
        print("original src %s" % str(pos_src))
        print("original dst %s" % str(pos_dst))


class _Overlay (OverlayOncanvasMixin):
    """Overlay for an AdjustLayerMode"""
    
    _NODE_RADIUS = 4
    _HANDLE_RADIUS = 2
    _SOURCE_COLOR = (1, 1, 1)
    _TARGET_COLOR = (0, 1, 1)

    def __init__(self, mode, tdw):
        super(_Overlay, self).__init__(mode, tdw)

    def draw_selection_rect(self, cr):
        tdw = self._tdw
        mode = self._mode
        cr.save()
        cr.set_source_rgb(0, 0, 0)
        cr.set_line_width(mode.LINE_WIDTH)
        area_info = (
            (mode.source_area, self._SOURCE_COLOR),
            (mode.target_area, self._TARGET_COLOR)
        )
        
        # Draw ridge of selected area.
        for area, color in area_info:
            if area.shape == _AreaShape.NONE:
                continue
            cr.save()
            cr.new_path()
            cr.set_dash((), 0)
            ox = None
            for sx, sy in area.iter_display_node(tdw):
                if ox is None:
                    cr.move_to(sx, sy)
                    ox = sx
                else:
                    cr.line_to(sx, sy)

          # sx, sy = area.get_display_node(tdw, 0)
          # cr.move_to(sx, sy)
          # for i in range(1, 4):
          #     sx, sy = area.get_display_node(tdw, i)
          #     cr.line_to(sx, sy)
            cr.close_path()
            cr.stroke_preserve()

            cr.set_source_rgb(*color)
            cr.set_dash( (3.0, ) )
            cr.stroke()
            cr.restore()
        cr.restore()
        
    def draw_target_nodes(self, cr):
        tdw = self._tdw
        mode = self._mode        
        r = self._NODE_RADIUS
        # Draw target rectangle, if needed.
        if mode.phase != _Phase.INITIAL:
            area = mode.active_area
            assert area is not None
            if area.shape != _AreaShape.NONE:
                cr.save()
                si = area.selected_node_index
                i = area.get_iter_starting_index()
                for sx, sy in area.iter_display_node(tdw):
                    self._draw_node(cr, r, sx, sy, si==i)
                    i = (i + 1) % 4
                cr.restore()                

    def draw_target_handles(self, cr):
        tdw = self._tdw
        mode = self._mode        
        r = self._HANDLE_RADIUS
        # Draw target rectangle, if needed.
        if mode.phase != _Phase.INITIAL:
            area = mode.active_area
            assert area is not None
            if area.shape != _AreaShape.NONE:
                cr.save()
                si = area.selected_handle_index
                i = area.get_iter_starting_index()
                for sx, sy in area.iter_display_handle(tdw):
                    if sx is not None:
                        self._draw_node(cr, r, sx, sy, si==i)
                    i = (i + 1) % 4
                cr.restore()                
                
    def update_button_positions(self):
        """Recalculates the positions of the mode's buttons."""
        # XXX Copied from gui/beziertool.py
        # FIXME mostly copied from inktool.Overlay.update_button_positions
        # The difference is for-loop of nodes , to deal with control handles.
        mode = self._mode
        area = mode.active_area
        if area.shape == _AreaShape.NONE:
            self._button_pos[_ActionButton.REJECT] = None
            self._button_pos[_ActionButton.ACCEPT] = None
            self._button_pos[_ActionButton.EDIT] = None
            return False

        button_radius = gui.style.FLOATING_BUTTON_RADIUS
        alloc = self._tdw.get_allocation()
        view_x0, view_y0 = alloc.x, alloc.y
        view_x1, view_y1 = view_x0+alloc.width, view_y0+alloc.height
        
        def constrain_button(cx, cy, radius):
            if cx + radius > view_x1:
                cx = view_x1 - radius
            elif cx - radius < view_x0:
                cx = view_x0 + radius
            
            if cy + radius > view_y1:
                cy = view_y1 - radius
            elif cy - radius < view_y0:
                cy = view_y0 + radius
            return cx, cy

        if mode.forced_button_pos is not None:
            # User deceided button position 
            cx, cy = mode.forced_button_pos
            margin = 1.5 * button_radius
            area_radius = 64 + margin #gui.style.FLOATING_TOOL_RADIUS

            cx, cy = adjust_button_inside(cx, cy, area_radius)

            pos_list = []
            count = 2
            for i in range(count):
                rad = (math.pi / count) * 2.0 * i
                x = - area_radius*math.sin(rad)
                y = area_radius*math.cos(rad)
                pos_list.append( (x + cx, - y + cy) )

            ac_x, ac_y = pos_list[0][0], pos_list[0][1]
            rj_x, rj_y = pos_list[1][0], pos_list[1][1]
        else:
            node = area.get_node(0)
            cx, cy = self._tdw.model_to_display(node.x, node.y)
            margin = 2.0 * button_radius
            ac_x, ac_y = cx + margin, cy - margin
            rj_x, rj_y = cx - margin, cy + margin

        margin = 2.0 * button_radius
        ac_x, ac_y = constrain_button(ac_x, ac_y, margin)
        rj_x, rj_y = constrain_button(rj_x, rj_y, margin)
        self._button_pos[_ActionButton.ACCEPT] = (ac_x, ac_y)
        self._button_pos[_ActionButton.REJECT] = (rj_x, rj_y)

        # XXX Code duplication from exinktool.py
        # `Edit button` placed at the center of the line between accept_button 
        # to reject_button
        l, nx, ny = length_and_normal(ac_x, ac_y, rj_x, rj_y)
        ml = l / 2.0
        bx = nx * ml 
        by = ny * ml 

        mx = -ny * margin + bx + ac_x 
        my = nx * margin + by + ac_y
        mx, my = constrain_button(mx, my, margin)
        self._button_pos[_ActionButton.EDIT] = (mx, my)
        return True

    def paint(self, cr):
        """Draw selection rectangle to the screen"""
        tdw = self._tdw
        mode = self._mode        
        self.draw_selection_rect(cr)
        self.draw_target_nodes(cr)
        self.draw_target_handles(cr)

        if mode.phase == _Phase.ADJUST_AREA:
            self._draw_mode_buttons(cr)


class _OptionsPresenter(Gtk.Grid):
    """Configuration widget for the AdjustMode tool"""

    _SPACING = 6
    _LABEL_MARGIN_LEFT = 8

    def __init__(self):
        self._cmd = None
        
        # XXX Code duplication: from closefill.py
        Gtk.Grid.__init__(self)
        self._update_ui = True
        self.set_row_spacing(self._SPACING)
        self.set_column_spacing(self._SPACING)
        from application import get_app
        self.app = get_app()
        self._mode_ref = None
        prefs = self.app.preferences
        row = 0

        def generate_label(text, tooltip, row, grid, alignment, 
                           margin_left=self._LABEL_MARGIN_LEFT):
            # Generate a label, and return it.
            label = Gtk.Label()
            label.set_markup(text)
            label.set_tooltip_text(tooltip)
            label.set_alignment(*alignment)
            label.set_hexpand(False)
            label.set_margin_start(margin_left)
            grid.attach(label, 0, row, 1, 1)
            return label
            
        def generate_spinbtn(row, grid, extreme_value):
            # Generate a Adjustment and spinbutton
            # and return Adjustment.
            adj = Gtk.Adjustment(
                value=0, 
                lower=-extreme_value,
                upper=extreme_value,
                step_increment=1, page_increment=1,
                page_size=0
            )
            spinbtn = Gtk.SpinButton()
            spinbtn.set_hexpand(True)
            spinbtn.set_adjustment(adj)

            # We need spinbutton focus event callback, to disable/re-enable
            # Keyboard manager for them.
            # Without this, we lose keyboard input focus right after 
            # input only one digit(key). 
            # It is very annoying behavior.
            spinbtn.connect("focus-in-event", self._spin_focus_in_cb)
            spinbtn.connect("focus-out-event", self._spin_focus_out_cb)
            grid.attach(spinbtn, 1, row, 1, 1)
            return adj
            
        label = generate_label(
            _("Transform Method:"),
            _("The transformation method to use"),
            0,
            self,
            (1.0, 0.1)
        )
        vbox = Gtk.VBox()

        label_list = _AreaShape.LABELS
        btndict = {} # Reverse-lookup table from button widget to method-id.
        self._transmethod_buttons = btndict
        radio_base = None
        for k in label_list.keys():
            label = label_list[k]
            radio = Gtk.RadioButton.new_with_label_from_widget(
                radio_base,
                label
            )
            radio.connect("toggled", self._transmethod_toggled_cb, k)
            vbox.pack_start(radio, False, False, 0)
            btndict[radio] = k
            if radio_base is None:
                radio_base = radio

        self.attach(vbox, 1, row, 1, 1)
        method = prefs.get(_Prefs.TRANSFORM_METHOD_PREF ,
                           _Prefs.DEFAULT_TRANSFORM_METHOD)
        active_btn = btndict.get(method, radio_base)
        active_btn.set_active(True)

        row += 1
        text = _("Source as visible")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            _("Use source image as visible contents without background."
              "This would be generate a new layer to draw image.")
        )
        checkbut.connect("toggled", self._use_visible_toggled_cb)
        self._use_visible_btn = checkbut
        self.attach(checkbut, 0, row, 2, 1)

        self._update_ui = False

    @property
    def target(self):
        if self._mode_ref is not None:
            return self._mode_ref()
        else:
            return None

    @target.setter
    def target(self, mode):
        if mode is not None:
            self._mode_ref = weakref.ref(mode)
        else:
            self._mode_ref = None
        
    def _use_visible_toggled_cb(self, btn):
        mode = self.target
        if mode is not None and not self._update_ui:
            mode.notify_use_visible_changed(btn.get_active())

    def finalize_use_visible(self):
        # Like a flood-fill, use_visible is cancelled once executed.
        self._update_ui = True
        self._use_visible_btn.set_active(False)
        self._update_ui = False
        
    # `Transform method` 
    def _transmethod_toggled_cb(self, btn, idx):
        mode = self.target
        if mode is not None and not self._update_ui and btn.get_active():
            buttons = self._transmethod_buttons
            assert btn in buttons
            method_id = buttons[btn]
            self.app.preferences[_Prefs.TRANSFORM_METHOD_PREF] = method_id
            mode.notify_tranform_method_changed()
