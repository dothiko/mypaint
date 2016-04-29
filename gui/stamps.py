#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
from numpy import array
from numpy import isfinite
from lib.helpers import clamp
import logging
logger = logging.getLogger(__name__)
import random
import json
import os
import glob

from gettext import gettext as _
from gi.repository import Gdk, Gtk
from gi.repository import GLib
from gi.repository import GdkPixbuf
import cairo

from gui.linemode import *
#from lib import mypaintlib
import lib

def draw_stamp_to_layer(target_layer, stamp, nodes, bbox):
    """
    :param bbox: boundary box, in model coordinate
    """
    sx, sy, w, h = bbox
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(w), int(h))
    cr = cairo.Context(surf)

    stamp.initialize_draw(cr)
    for cn in nodes:
        stamp.draw(None, cr, cn.x - sx, cn.y - sy,
                cn, True)
    stamp.finalize_draw(cr)

    surf.flush()
    pixbuf = Gdk.pixbuf_get_from_surface(surf, 0, 0, w, h)
    layer = lib.layer.PaintingLayer(name='')
    layer.load_surface_from_pixbuf(pixbuf, int(sx), int(sy))
    del surf, cr

    tiles = set()
    tiles.update(layer.get_tile_coords())
    dstsurf = target_layer._surface
    for tx, ty in tiles:
        with dstsurf.tile_request(tx, ty, readonly=False) as dst:
            layer.composite_tile(dst, True, tx, ty, mipmap_level=0)

    bbox = tuple(target_layer.get_full_redraw_bbox())
    target_layer.root.layer_content_changed(target_layer, *bbox)
    target_layer.autosave_dirty = True
    del layer



def get_barycentric_point(x, y, triangle):
    """
    Get barycentric point of a point(x,y) in a triangle.
    :param triangle: a sequential of 3 points, which compose a triangle.
    """
    p0, p1, p2 = triangle
    v1x = p1[0] - p2[0]
    v1y = p1[1] - p2[1]
    v2x = p0[0] - p2[0]
    v2y = p0[1] - p2[1]

    d = v2y * v1x - v1y * v2x
    if d == 0.0:
        return False

    p2x = x - p2[0] 
    p2y = y - p2[1] 
    p0x = x - p0[0] 
    p0y = y - p0[1] 

    b0 = (p2y * v1x + v1y * -p2x) / d
    b1 = (p0y * -v2x + v2y * -p0x) / d
    return (b0, b1, 1.0 - b0 - b1)

#def is_inside_triangle(x, y, triangle):
#    b0, b1, b2 = get_barycentric_point(x, y, triangle)
#    return (0.0 <= b0 <= 1.0 and 
#            0.0 <= b1 <= 1.0 and 
#            0.0 <= b2 <= 1.0)


def is_inside_triangle(x, y, triangle):
    """ from stackoverflow
    """
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = sign((x,y) , triangle[0], triangle[1]) < 0.0
    b2 = sign((x,y) , triangle[1], triangle[2]) < 0.0
    b3 = sign((x,y) , triangle[2], triangle[0]) < 0.0
    return b1 == b2 == b3



## Class defs

class _Pixbuf_Source(object):
    """ Pixbuf source management class.
    This is used Pixbuf source and Mask source.
    """


    def __init__(self, filenames):
        self._file_idx = 0
        self._tile_w = 0
        self._tile_h = 0
        self._tile_count = -1
        self._tile_idx = 0

        self.set_file_sources(filenames)

    def _load_from_file(self, filename):
        def get_offsets(pixbuf):
            return (-(pixbuf.get_width() / 2),
                    -(pixbuf.get_height() / 2))

        pixbuf = GdkPixbuf.Pixbuf.new_from_file(filename)
        if self._tile_w > 0 and self._tile_h > 0:
            ret = []
            for y in xrange(pixbuf.get_height() / self._tile_h):
                for x in xrange(pixbuf.get_width() / self._tile_w):
                    cpb = pixbuf.new_subpixbuf(x, y, 
                            self._tile_w, self._tile_h)
                    ox, oy = get_offsets(cpb)
                   #ret.append(Gdk.cairo_surface_create_from_pixbuf(cpb, ox, oy))
                   #del cpb
                    ret.append(cpb)
            del pixbuf
            return ret
        else:
            ox, oy = get_offsets(pixbuf)
           #return [Gdk.cairo_surface_create_from_pixbuf(pixbuf, ox, oy), ]
            return [pixbuf, ]

    ## Information methods

    @property
    def tile_count(self):
        if self._tile_count == -1:
            cnt = 0
            for ck in self._pixbufs:
                cnt += len(self._pixbufs)
            self._tile_count = cnt
        return self._tile_count

    def get_current_src(self, tile_idx):
        self._ensure_current_pixbuf()
        return self._pixbufs[self._file_idx][tile_idx]

    def set_tile_division(self, w, h):
        """ Set tile size as dividing factor.
        To cancel tile division, use 0 for each arguments.

        :param w: width of a tile.
        :param h: height of a tile.
        """
        self._tile_w = w
        self._tile_h = h

    def set_file_sources(self, filenames):
        self._source_files = filenames
        self._pixbufs = {}

    ## Source Pixbuf methods

    def _ensure_current_pixbuf(self):
        assert self._file_idx < len(self._source_files)
        if self._file_idx in self._pixbufs:
            pass
        else:
            filename = self._source_files[self._file_idx]
            self._pixbufs[self._file_idx] = self._load_from_file(filename)

    def next(self):
        """
        Proceed to source next pixbuf
        """
        self._tile_idx += 1
        if self._tile_idx >= len(self._pixbufs[self._file_idx]):
            self._file_idx += 1
            self._tile_idx = 0

            if self._file_idx >= len(self._pixbufs):
                self._file_idx = 0

    def clear_all(self):
        """
        Clear all cached pixbuf
        """
        for ck in self._pixbufs:
            for cpb in self._pixbufs[ck]:
                del cpb
            self._pixbufs[ck] = None
        self._pixbufs = {}



class Stamp(object):
    """
    This class holds stamp information and pixbuf/masks 
    and draw it to "cairo surface",
    not mypaint surface.(But this is to ease development 
    and might be changed to improve memory efficiency)

    When 'finalize' action executed (i.e. ACCEPT button pressed)
    the cairo surface converted into GdkPixbuf,
    and that GdkPixbuf converted into Mypaint layer
    and it is merged into current layer.
    """

    def __init__(self, name):
        self._pixbuf_src = None
        self._mask_src = None
        self._prev_src = {}
        self.name = name
        self._default_scale_x = 1.0
        self._default_scale_y = 1.0
        self._default_angle = 0.0
        self._thumbnail = None

    ## Information methods

    def set_default_scale(self, scale_x, scale_y):
        if scale_x == 0.0:
            scale_x = 0.00000001
        if scale_y == 0.0:
            scale_y = 0.00000001
        self._default_scale_x = scale_x
        self._default_scale_y = scale_y

    def set_default_angle(self, angle):
        self._default_angle = angle

    @property
    def default_scale_x(self):
        return self._default_scale_x

    @property
    def default_scale_y(self):
        return self._default_scale_y

    @property
    def default_angle(self):
        return self._default_angle

    @property
    def pixbuf_src(self):
        return self._pixbuf_src

    @property
    def mask_src(self):
        return self._mask_src

    def set_tile_division(self, w, h):
        """ Set tile size as dividing factor.
        To cancel tile division, use 0 for each arguments.

        :param w: width of a tile.
        :param h: height of a tile.
        """
        if self._pixbuf_src:
            self._pixbuf_src.set_tile_division(w, h)
        if self._mask_src:
            self._mask_src.set_tile_division(w, h)

    @property
    def tile_count(self):
        if self._pixbuf_src:
            return self._pixbuf_src.tile_count
        else:
            return 0

    def set_file_sources(self, filenames):
        self._pixbuf_src = _Pixbuf_Source(filenames)

    def set_mask_sources(self, filenames):
        self._mask_src = _Pixbuf_Source(filenames)

    def set_thumbnail(self, pixbuf):
        self._thumbnail = pixbuf

    @property
    def thumbnail(self):
        return self._thumbnail

    ## Drawing methods

    def initialize_draw(self, cr):
        """ Initialize draw calls.
        Call this method prior to draw stamps loop.
        """
        self._prev_src[cr] = None

    def finalize_draw(self, cr):
        """ Finalize draw calls.
        Call this method after draw stamps loop.
        """
        del self._prev_src[cr]

    

    def fetch_pattern(self, cr, tile_idx):
        if self._pixbuf_src:
            stamp_src = self._pixbuf_src.get_current_src(tile_idx)
            if ((not cr in self._prev_src) or 
                    self._prev_src[cr] != stamp_src):
                w = stamp_src.get_width() 
                h = stamp_src.get_height()
                ox = -(w / 2)
                oy = -(h / 2)
                Gdk.cairo_set_source_pixbuf(cr, stamp_src, ox, oy)
               #cr.set_source_surface(stamp_src, ox, oy)
                self._cached_cairo_src = cr.get_source()
                self._prev_src[cr] = stamp_src
            else:
                cr.set_source(self._cached_cairo_src)

    def fetch_mask(self, cr, tile_idx):
        if self._mask_src:
            mask_src = self._mask_src.get_current_src(tile_idx)
            if ((not cr in self._prev_mask) or 
                    self._prev_mask[cr] != stamp_src):
                w = stamp_src.get_width() 
                h = stamp_src.get_height()
                ox = -(w / 2)
                oy = -(h / 2)
               #Gdk.cairo_set_source_pixbuf(cr, stamp_src, ox, oy)
                cr.set_mask_surface(stamp_src, ox, oy)
                self._cached_cairo_src = cr.get_source()
                self._prev_mask[cr] = stamp_src
            else:
                cr.set_source(self._cached_cairo_src)



    def draw(self, tdw, cr, x, y, node, save_context=False):
        """ draw this stamp into cairo surface.
        """
        if save_context:
            cr.save()

        stamp_src = self._pixbuf_src.get_current_src(node.tile_index)

        w = stamp_src.get_width() 
        h = stamp_src.get_height()

        ox = -(w / 2)
        oy = -(h / 2)

        angle = node.angle
        scale_x = node.scale_x
        scale_y = node.scale_y

        cr.translate(x,y)
        if ((tdw and tdw.renderer.rotation != 0.0) or 
                angle != 0.0):
            if tdw:
                angle += tdw.renderer.rotation
            cr.rotate(angle)

        if ((tdw and tdw.renderer.scale != 1.0) or 
                (scale_x != 1.0 and scale_y != 1.0)):
            if tdw:
                scale_x *= tdw.renderer.scale
                scale_y *= tdw.renderer.scale

            if scale_x != 0.0 and scale_y != 0.0:
                cr.scale(scale_x, scale_y)

       #Gdk.cairo_set_source_pixbuf(cr, self._stamp_src, ox, oy)
        self.fetch_pattern(cr, node.tile_index)
        cr.rectangle(ox, oy, w, h) 

        cr.clip()
        cr.paint()
        
        if save_context:
            cr.restore()

    ## Boundary / hit-check methods

    def is_inside(self, mx, my, node):
        """ 
        THIS METHOD IS OBSOLUTED,USE get_handle_index INSTEAD.
        Check whether the point(mx, my) is inside of the stamp
        which placed as node information.
        """
        pos = self.get_boundary_points(node)
        if not is_inside_triangle(mx, my, (pos[0], pos[1], pos[2])):
            return is_inside_triangle(mx, my, (pos[0], pos[2], pos[3]))
        else:
            return True

    def get_handle_index(self, mx, my, node, margin):
        """
        Get control handle index
        :param mx,my: Cursor position in model coordination.
        :param node: Target node,which contains scale/angle information.
        :param margin: i.e. the half of control handle rectangle

        :rtype: integer, whitch is targetted handle index.
                Normally it should be either one of targetted handle index (0 to 3),
                or -1 (cursor is not on this node)
                When 'node is targetted but not on control handle',
                return 4.

        """
        pos = self.get_boundary_points(node)

        for i,cp in enumerate(pos):
            cx, cy = cp
            if (cx - margin <= mx <= cx + margin and
                    cy - margin <= my <= cy + margin):
                return i

        # Returning 4 means 'inside but not on a handle'
        if not is_inside_triangle(mx, my, (pos[0], pos[1], pos[2])):
            if is_inside_triangle(mx, my, (pos[0], pos[2], pos[3])):
                return 4 
        else:
            return 4

        return -1

    def get_boundary_points(self, node, tdw=None, dx=0.0, dy=0.0, no_transform=False):
        """ Get boundary corner points, when this stamp is
        placed/rotated/scaled into the node.

        The return value is a list of 'boundary corner points'.
        The stamp might be rotated, so boundary might not be 'rectangle'. 
        so this method returns a list of 4 corner points.

        Each corner points are model coordinate.

        When parameter 'tdw' assigned, returned corner points are 
        converted to display coordinate.

        :param mx, my: the center point in model coordinate.
        :param tdw: tiledrawwidget, to get display coordinate. 
                    By default this is None.
        :param dx, dy: offsets, in model coordinate.
        :param no_transform: boolean flag, when this is True,
                             rotation and scaling are cancelled.

        :rtype: a list of tuple,[ (pt0.x, pt0.y) ... (pt3.x, pt3.y) ]
        """
        stamp_src = self._pixbuf_src.get_current_src(node.tile_index)

        w = stamp_src.get_width() 
        h = stamp_src.get_height()
        if not no_transform:
            w *= node.scale_x 
            h *= node.scale_y
        sx = - w / 2
        sy = - h / 2
        ex = w+sx
        ey = h+sy
        bx = node.x + dx
        by = node.y + dy

        if node.angle != 0.0 and not no_transform:
            points = [ (sx, sy),
                          (ex, sy),
                          (ex, ey),
                          (sx, ey) ]
            cos_s = math.cos(node.angle)
            sin_s = math.sin(node.angle)
            for i in xrange(4):
                x = points[i][0]
                y = points[i][1]
                tx = (cos_s * x - sin_s * y) + bx
                ty = (sin_s * x + cos_s * y) + by
                points[i] = (tx, ty) 
        else:
            sx += bx
            ex += bx
            sy += by
            ey += by
            points = [ (sx, sy),
                          (ex, sy),
                          (ex, ey),
                          (sx, ey) ]

        if tdw:
            points = [ tdw.model_to_display(x,y) for x,y in points ]

        return points

    def get_bbox(self, tdw, node, dx=0.0, dy=0.0, margin = 0):
        """ Get outmost boundary box, to get displaying area or 
        to do initial collision detection.
        return value is a tuple of rectangle,
        (x, y, width, height)
        """
        pos = self.get_boundary_points(node, dx=dx, dy=dy)
        if tdw:
            sx, sy = tdw.model_to_display(*pos[0])
        else:
            sx, sy = pos[0]
        ex = sx
        ey = sy
        for x, y in pos[1:]:
            if tdw:
                x, y = tdw.model_to_display(x, y)
            sx = min(sx, x)
            sy = min(sy, y)
            ex = max(ex, x)
            ey = max(ey, y)

        return (sx - margin, sy - margin, 
                (ex - sx) + 1 + margin * 2, 
                (ey - sy) + 1 + margin * 2)



class StampPresetManager(object):
    """ Stamp preset manager.
    This class is singleton, owned by Application class 
    as application.stamp_preset_manager

    With this manager class , we can generates Stamp instance
    from its name. 
    """

    # Class constants
    STAMP_DIR_NAME = u'stamps' # Stamp stored dir name, it is under the app.user_data dir.

    def __init__(self, app):
        self._app = app
        self._icon_size = 32
       #self.basedir = os.path.join(app.state_dirs.user_data, self.STAMP_DIR_NAME)

        # XXX mostly copied from gui/application.py _init_icons()
        icon_theme = Gtk.IconTheme.get_default()
        icon_theme.append_search_path(app.state_dirs.app_icons)
        self._default_icon = icon_theme.load_icon('mypaint', 32, 0)

    def _get_adjusted_path(self, filepath):
        if not os.path.isabs(filepath):
            return  os.path.join(self._app.state_dirs.user_data,
                        self.STAMP_DIR_NAME, filepath)
        else:
            return filepath

    def initialize_icon_store(self):
        """ Initialize iconview widget which is used in
        stamptool's OptionPresenter.
        """
        liststore = Gtk.ListStore(GdkPixbuf.Pixbuf, str, object)

        for cf in glob.glob(self._get_adjusted_path("*.mys")):
            stamp = self.load_from_file(cf)
            liststore.append([stamp.thumbnail, stamp.name, stamp])

        return liststore

    def load_thumbnail(self, name):
        if name != None:
            filepath = self._get_adjusted_path(name)
            if os.path.exists(filepath):
                return  GdkPixbuf.Pixbuf.new_from_file_at_size(
                            self._get_adjusted_path(name),
                            self._icon_size, self._icon_size)

        assert self._default_icon != None
        return self._default_icon
        
    def load_from_file(self, filename):
        """ Presets saved as json file, just like as brushes.

        stamp preset .mys file is a json file, which has
        attributes below:

            "comment" : comment of preset
            "name" : name of preset
            "settings" : a dictionary to contain stamp settings.
            "thumbnail" : a thumbnail .jpg/.png filename
            "version" : 1

        stamp setting:
            "source" : "file" - Create stamp from file(s).
                       "clipboard" - Use run-time clipboard image for stamp.
                       "layer" - create stamp from layer image 
                                 where user defined area.
                       "current_visible"  - create stamp from currently visible image
                                            from user defined area.
                       "foreground" - foreground color, with masked.
                                      This source needs mask.
                                      Without mask, the stamp meaninglessly
                                      create filled rectangle.
            "filename" : list of filepath of stamp source .jpg/.png.
                         Multiple filepath means this stamp has
                         tiles in separeted files.
            "scale" : a tuple of default scaling ratio, 
                      (horizontal_ratio, vertical_ratio)
            "angle" : a floating value of default angle,in degree. 
            "mask" : list of filepath of mask source .jpg/.png, this must be
                     8bit grayscale.
                     Multiple filepath means this stamp has
                     multiple masks which corespond to each tile.
            "tile" : a tuple of (width, height),it represents tile size
                     of a picture. 
                     This setting used when only one picture/mask file 
                     assigned.
            "type" : "random" - the next tile index is randomly seleceted.
                     "increment" - the next tile index is automatically incremented.
                     "same" - the next tile index is always 0.
                              user change it manually.(default)

        :rtype: Stump class instance.
        """
        junk, ext = os.path.splitext(filename) 
        assert ext.lower() == '.mys'

        filename = self._get_adjusted_path(filename)

        with open(filename,'r') as ifp:
            print filename
            jo = json.load(ifp)
            settings = jo['settings']

            if jo['version'] == "1":
                source = settings['source']
                if source == 'file':
                    stamp = Stamp(jo['name'])
                    assert 'filenames' in settings
                    stamp.set_file_sources(settings['filenames'])
                   #stamp.load_from_files(files, 
                   #        settings.get('tile', None))
                    

                elif source == 'clipboard':
                    pass
                elif source == 'layer':
                    pass
                elif source == 'current_visible':
                    pass
                elif source == 'foreground':
                    pass
                else:
                    raise NotImplementedError("Unknown source %r" % source)

                # common setting
                if 'scale' in settings:
                    stamp.set_default_scale(*settings['scale'])

                if 'angle' in settings:
                    stamp.set_default_angle(math.radians(settings['angle'] % 360.0))

                if 'mask' in settings:
                   #stamp.load_masks(settings['mask'],
                   #        settings.get('tile', None))
                    stamp.set_mask_sources(settings['mask'])

                stamp.set_tile_division(*settings.get('tile', (0, 0)))

                stamp.set_thumbnail(
                        self.load_thumbnail(settings.get('thumbnail', None)))

            else:
                raise NotImplementedError("Unknown version %r" % jo['version'])

            return stamp

   #def save_stamp_to_file(self, filename, stamp):
   #    json_src = stamp.output_as_json()
   #    self.save_to_file(self, filename)

    def save_to_file(self, filename, json_src):
        filename = self._get_adjusted_path(filename)
        with open(filename, 'w') as ofp:
            json.dump(json_src, ofp)


def _test():
    from application import get_app
    app = get_app()
    m = StampPresetManager()
    print(m.initialize_icon_store())

if __name__ == '__main__':
    _test()



