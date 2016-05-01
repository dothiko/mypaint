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
import lib
from gui.ui_utils import *

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


## Class defs

## Source classes
class _SourceMixin(object):

    def __init__(self):
        pass

    @property
    def tile_count(self):
        return 1

    @property
    def tile_index(self):
        """ for compatibility """
        return 0

    @property
    def current_src(self):
        """ Get current (cached) src pixbuf.
        Most important property.
        """
        pass

    def set_tile_division(self, w, h):
        """ Set tile size as dividing factor.
        To cancel tile division, use 0 for each arguments.

        :param w: width of a tile.
        :param h: height of a tile.
        """
        pass

    @staticmethod
    def get_offsets(pixbuf):
        return (-(pixbuf.get_width() / 2),
                -(pixbuf.get_height() / 2))


class _Pixbuf_Source(_SourceMixin):
    """ Pixbuf source management class.
    This is used as 'Pixbuf source' and 
    also 'Mask source' at Stamp class.
    """


    def __init__(self, filenames):

        self.set_file_sources(filenames)
        # Other attributes initialized at
        # _reset_info() , which called from
        # set_file_sources()

    def _load_from_file(self, filename):
        pixbuf = GdkPixbuf.Pixbuf.new_from_file(filename)
        if self._tile_w > 0 and self._tile_h > 0:
            ret = []
            for y in xrange(pixbuf.get_height() / self._tile_h):
                for x in xrange(pixbuf.get_width() / self._tile_w):
                    cpb = pixbuf.new_subpixbuf(x, y, 
                            self._tile_w, self._tile_h)
                    ox, oy = _StampMixin.get_offsets(cpb)
                   #ret.append(Gdk.cairo_surface_create_from_pixbuf(cpb, 1, None))
                   #del cpb
                    ret.append(cpb)
            del pixbuf
        else:
            ox, oy = _StampMixin.get_offsets(pixbuf)
           #return [Gdk.cairo_surface_create_from_pixbuf(pixbuf, 1, None), ]
            ret = [pixbuf, ]

        if self._tile_segment == None:
            self._tile_segment = len(ret)

        return ret

    ## Information methods
    @property
    def tile_count(self):
        return self.tile_segment * len(self._pixbufs)

    @property
    def tile_segment(self):
        """ 'tile_segment' means the tile counts per one picture.
        Total tile count should be 'tile_segment * picture count'
        """
        return self._tile_segment

    @property
    def tile_index(self):
        return self._linear_tile_idx

    @tile_index.setter
    def tile_index(self, index):
        if self._tile_count > 0 and self._linear_tile_idx != index:
            if index == 0:
                self._file_idx = 0
                self._tile_idx = 0
            else:
                self._file_idx = index / int(self.tile_segment)
                self._tile_idx = index % int(self.tile_segment)
            self._linear_tile_idx = index

        return 0

    @property
    def current_src(self):
        self._ensure_current_pixbuf()
        return self._pixbufs[self._file_idx][self._tile_idx]

    def set_tile_division(self, w, h):
        """ Set tile size as dividing factor.
        To cancel tile division, use 0 for each arguments.

        :param w: width of a tile.
        :param h: height of a tile.
        """
        self._tile_w = w
        self._tile_h = h

    def set_file_sources(self, filenames):
        if type(filenames) == str:
            self._source_files = [filenames, ]
        else:
            self._source_files = filenames
        self._reset_info()

    def _reset_info(self):
        self._pixbufs = {}
        self._tile_segment = None
        self._linear_tile_idx = -1
        self._file_idx = 0
        self._tile_idx = 0
        self._tile_w = 0
        self._tile_h = 0
        self._tile_count = 0
        

    ## Source Pixbuf methods

    def _ensure_current_pixbuf(self):
        assert self._file_idx < len(self._source_files)
        if self._file_idx in self._pixbufs:
            pass
        else:
            filename = self._source_files[self._file_idx]
            self._pixbufs[self._file_idx] = self._load_from_file(filename)


    def clear_all(self):
        """
        Clear all cached pixbuf
        """
        for ck in self._pixbufs:
            for cpb in self._pixbufs[ck]:
                del cpb
            self._pixbufs[ck] = None
        self._pixbufs = {}




## Stamp classes

class _StampMixin(object):
    """ Stamp Mixin, base class of all stamp classes. """

    THUMBNAIL_SIZE = 32

    def __init__(self, name):
        self._pixbuf_src = None
        self._mask_src = None
        self._prev_src = {}
        self._prev_mask= {}
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
        pass

    @property
    def tile_count(self):
        return 1

    def set_file_sources(self, filenames):
        self._pixbuf_src = _Pixbuf_Source(filenames)

    def set_mask_sources(self, filenames):
        self._mask_src = _Pixbuf_Source(filenames)

    def set_thumbnail(self, pixbuf):
        self._thumbnail = pixbuf

    @property
    def thumbnail(self):
        return self._thumbnail

    @property
    def current_src(self):
        """ Get current src pixbuf.
        by default it is from self._pixbuf_src
        (it would be _SourceMixin derived class), 
        we can change this with returning pixbuf object directly.
        """
        if self._pixbuf_src:
            return self._pixbuf_src.current_src

    ## Drawing methods

    def initialize_draw(self, cr):
        """ Initialize draw calls.
        Call this method prior to draw stamps loop.
        """
        self._prev_src[cr] = None
        self._prev_mask[cr] = None

    def finalize_draw(self, cr):
        """ Finalize draw calls.
        Call this method after draw stamps loop.

        This method called from the end of each drawing sequence,
        NOT END OF DRAWING PHASE OF STAMPTOOL! 
        Therefore, source.finalize() MUST not be called here!
        """
        del self._prev_src[cr]
        del self._prev_mask[cr]

    

    def apply_pattern(self, cr, tile_idx):
        """ Apply pixbuf pattern to cairo.
        The pattern might be cached.

        This method should be called from draw() method.
        """
        srcinfo = self._setup_source(cr, tile_idx, 
                self._pixbuf_src, self._prev_src)
        if srcinfo:
            stamp_src, ox, oy = srcinfo
            Gdk.cairo_set_source_pixbuf(cr, stamp_src, ox, oy)
            self._cached_cairo_src = cr.get_source()
            self._prev_src[cr] = stamp_src
        elif self._cached_cairo_src:
            cr.set_source(self._cached_cairo_src)

    def apply_mask(self, cr, tile_idx):
        """ Apply pixbuf mask to cairo.
        The mask(cairo surface) might be cached.

        This method should be called from draw() method.
        """
        if self._mask_src:
            srcinfo = self._setup_source(cr, tile_idx, 
                    self._mask_src, self._prev_mask)
            if srcinfo:
                stamp_src, ox, oy = srcinfo
                mask_surface = Gdk.cairo_surface_create_from_pixbuf(
                        stamp_src, 1, None)
                cr.mask_surface(mask_surface, ox, oy)
                self._cached_cairo_mask = mask_surface
                self._mask_offset = (ox, oy)
                self._prev_mask[cr] = stamp_src
            elif self._cached_cairo_mask:
                ox, oy = self._mask_offset
                cr.mask_surface(mask_surface, ox, oy)


    def _setup_source(self, cr, tile_idx, source, cache_list):
        if source:
            source.tile_index = tile_idx
            stamp_src = source.current_src
            if ((not cr in cache_list) or 
                    cache_list[cr] != stamp_src):
                w = stamp_src.get_width() 
                h = stamp_src.get_height()
                ox = -(w / 2)
                oy = -(h / 2)
                return (stamp_src, ox, oy)


    def draw(self, tdw, cr, x, y, node, save_context=False):
        """ draw this stamp into cairo surface.
        """
        pass

    ## Boundary / hit-check methods

    def is_inside(self, mx, my, node):
        """ 
        THIS METHOD IS OBSOLUTED,USE get_handle_index INSTEAD.
        Check whether the point(mx, my) is inside of the stamp
        which placed as node information.
        """
        pos = self.get_boundary_points(node)
        if pos:
            if not is_inside_triangle(mx, my, (pos[0], pos[1], pos[2])):
                return is_inside_triangle(mx, my, (pos[0], pos[2], pos[3]))
            else:
                return True

        return False

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
        if pos:
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
        stamp_src = self.current_src
        if stamp_src:
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
        if pos:
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

    ## drawing methods

    def draw(self, tdw, cr, x, y, node, save_context=False):
        """ Draw this stamp into cairo surface.
        This implementation is as base class,
        node.tile_index ignored here.
        """
        stamp_src = self.current_src
        if stamp_src:
            
            if save_context:
                cr.save()

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
            self.apply_pattern(cr, node.tile_index)
            self.apply_mask(cr, node.tile_index)

            cr.rectangle(ox, oy, w, h) 

            cr.clip()
            cr.paint()
            
            if save_context:
                cr.restore()

    ## Phase related methods
    #  these methods should be called from 
    #  Stamptool._start_new_capture_phase()

    def initialize_phase(self):
        """ Initializing for start of each drawing phase.
        CAUTION: This method is not called when drawing to
        layer.

        the code flow around this method should be :

        ACCEPT BUTTON Pressed (end of drawing phase)
        |
        +-- draw_stamp_to_layer called (from Command object : asynchronously)
        |
        self.finalize_phase() called
        |
        some stampmode initializing lines called
        |
        self.initialize_phase() called
        """
        pass

    def finalize_phase(self):
        """ This called when stamptool comes to end of
        capture(drawing) phase.
        """
        pass

class _PixbufBackedStampMixin(_StampMixin):
    """ Non-tiled single pixbuf stamp mixin.

    This mixin used for dynamically changed stamps,
    for example,Clipboard stamp.
    """

    def __init__(self, name):
        super(_PixbufBackedStampMixin, self).__init__(name)
        self._src_pixbuf = None

    def set_file_sources(self, filenames):
        pass

    @property
    def current_src(self):
        return self._src_pixbuf

    def _setup_source(self, cr, tile_idx, source, cache_list):
        if self._src_pixbuf:
            stamp_src = self._src_pixbuf
            if ((not cr in cache_list) or 
                    cache_list[cr] != stamp_src):
                w = stamp_src.get_width() 
                h = stamp_src.get_height()
                ox = -(w / 2)
                oy = -(h / 2)
                return (stamp_src, ox, oy)


class Stamp(_StampMixin):
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
        super(Stamp, self).__init__(name)

    ## Information methods

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

    ## Information methods
    def get_boundary_points(self, node, tdw=None, dx=0.0, dy=0.0, no_transform=False):
        if self._pixbuf_src:
            self._pixbuf_src.tile_index = node.tile_index
            super(Stamp, self).get_boundary_points(node, tdw, dx, dy, no_transform)

    ## Drawing methods


    def draw(self, tdw, cr, x, y, node, save_context=False):
        """ draw this stamp into cairo surface.
        """
        self._pixbuf_src.tile_index = node.tile_index
        super(Stamp, self).draw(tdw, cr, x, y, node, save_context)

class ClipboardStamp(_PixbufBackedStampMixin):

    def __init__(self, name, doc):
        super(ClipboardStamp, self).__init__(name)
        self._doc = doc

    def _get_clipboard(self):
        # XXX almost copied from gui.document._get_clipboard()
        # we might need some property interface to get it from
        # document class directly...?
        display = self._doc.tdw.get_display()
        cb = gtk.Clipboard.get_for_display(display, Gdk.SELECTION_CLIPBOARD)
        return cb

    def initialize_phase(self):
        """ Initializing for start of each drawing phase.
        CAUTION: This method is not called when drawing to
        layer.
        """
        self._src_pixbuf = None
        # XXX almost copied from gui.document.paste_cb()
        clipboard = self._get_clipboard()

        targs_avail, targets = clipboard.wait_for_targets()
        if not targs_avail:
            # Nothing on clipboard."
            return

        # Then grab any available image, also synchronously
        pixbuf = clipboard.wait_for_image()
        if not pixbuf:
            return

        self._src_pixbuf = pixbuf

class LayerStamp(_PixbufBackedStampMixin):
    """ A Stamp which sources the current layer.
    """

    def __init__(self, name, rootstack):
        super(LayerStamp, self).__init__(name)
        self._selbox = None

    def set_selected_area(self, selbox):
        """
        set selected area of current layer as stamp source.

        :param selbox: a tuple of selection box (x, y, w, h)
        To disable selection(and disable draw), assign selbox as None
        """
        self._selbox = selbox

    def initialize_phase(self):
        """ Get current (cached) src pixbuf.
        Most important property.
        """
        current = self._rootstack.current
        pixbuf = None
        if current and self._selbox:
            pixbuf = current.render_as_pixbuf(
                    *self._selbox, alpha=True)
        self._src_pixbuf = pixbuf


class VisibleStamp(LayerStamp):
    """ A Stamp which sources the currently visible area.
    """

    def __init__(self, name, rootstack):
        super(VisibleStamp, self).__init__(name)

    def initialize_phase(self):
        """ Get current (cached) src pixbuf.
        Most important property.
        """
        current = self._rootstack
        pixbuf = None
        if current and self._selbox:
            pixbuf = current.render_as_pixbuf(
                    *self._selbox, alpha=True)
        self._src_pixbuf = pixbuf

class ForegroundStamp(_PixbufBackedStampMixin):
    """ Foreground color stamp.
    """

    def __init__(self, name, app):
        super(ForegroundStamp, self).__init__(name)
        self._app = app

    @property
    def foreground_color(self):
        return self._app.brush_color_manager.get_color().get_rgb()

    @property
    def current_src(self):
        if self._mask_src:
            return self._mask_src.current_src

    def set_file_sources(self, filenames):
        """ to disable setting self._pixbuf_src """
        pass


    def draw(self, tdw, cr, x, y, node, save_context=False):
        """ Draw this stamp into cairo surface.
        This implementation is as base class,
        node.tile_index ignored here.
        """
        if save_context:
            cr.save()

        stamp_src = self.current_src
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
        cr.set_source_rgb(*self.foreground_color)
        self.apply_mask(cr, stamp_src)
        cr.rectangle(ox, oy, w, h) 

        cr.clip()
        cr.paint()
        
        if save_context:
            cr.restore()




class PixbufStamp(_PixbufBackedStampMixin):
    """ A Stamp for dynamically changeable pixbuf stamps
    to fix its contents within DrawStamp command.
    This stamp would only use inside program,have no any user
    direct interaction.

    MyPaint 'Command' processing done in asynchronously,
    so Clipboard/Layer stamp might be updated its content(pixbuf)
    when actually draw_stamp_to_layer() called.

    Therefore, we need to pass the pixbuf used when editing
    to this stamp class.
    """

    def __init__(self, name, pixbuf):
        super(PixbufStamp, self).__init__(name)
        self._src_pixbuf = pixbuf



## Preset Manager classes

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
       #self.basedir = os.path.join(app.state_dirs.user_data, self.STAMP_DIR_NAME)

        # XXX mostly copied from gui/application.py _init_icons()
        icon_theme = Gtk.IconTheme.get_default()
        icon_theme.append_search_path(app.state_dirs.app_icons)
        self._default_icon = icon_theme.load_icon('mypaint', 32, 0)

    def _get_adjusted_path(self, filepath):
        """ Return absolute path according to application setting.
        
        This method does not check the modified path exists.
        However, glob.glob() will accept nonexistant path,
        glob.glob() returns empty list for such one.
        """
        if not os.path.isabs(filepath):
            return os.path.join(self._app.state_dirs.user_data,
                        self.STAMP_DIR_NAME, filepath)
        else:
            return filepath

    def initialize_icon_store(self):
        """ Initialize iconview store which is used in
        stamptool's OptionPresenter.
        i.e. This method is actually application-unique 
        stamp preset initialize handler.
        """
        liststore = Gtk.ListStore(GdkPixbuf.Pixbuf, str, object)

        for cs in BUILT_IN_STAMPS:
            stamp = self.create_stamp_from_json(cs)
            liststore.append([stamp.thumbnail, stamp.name, stamp])

        for cf in glob.glob(self._get_adjusted_path("*.mys")):
            stamp = self.load_from_file(cf)
            liststore.append([stamp.thumbnail, stamp.name, stamp])

        return liststore

    def load_thumbnail(self, name):
        if name != None:
            filepath = self._get_adjusted_path(name)
            if os.path.exists(filepath):
                icon_size = Stamp.THUMBNAIL_SIZE
                return  GdkPixbuf.Pixbuf.new_from_file_at_size(
                            self._get_adjusted_path(name),
                            icon_size, icon_size)

        assert self._default_icon != None
        return self._default_icon
        
    def load_from_file(self, filename):
        """ Presets saved as json file, just like as brushes.
        :rtype: Stump class instance.
        """
        junk, ext = os.path.splitext(filename) 
        assert ext.lower() == '.mys'

        filename = self._get_adjusted_path(filename)

        with open(filename,'r') as ifp:
            jo = json.load(ifp)
            return self.create_stamp_from_json(jo)

    def create_stamp_from_json(self, jo):
        """ Create a stamp from json-generated dictionary object.
        :param jo: dictionary object, mostly created with json.load()/loads()
        :rtype: Stump class instance.

        ## The specification of mypaint stamp preset file(.mys)

        Stamp preset .mys file is a json file, 
        which has attributes below:

            "comment" : comment of preset
            "name" : name of preset
            "settings" : a dictionary to contain 'Stamp setting's.
            "thumbnail" : a thumbnail .jpg/.png filename
            "version" : 1

        Stamp setting:
            "source" : "file" - Stamp from files.
                       "tiled-file" - Stamp from a file, divided with tile setting.
                       "clipboard" - Use run-time clipboard image for stamp.
                       "layer" - Stamp from layer image of user defined area.
                       "current_visible"  - Stamp from currently visible image
                                            of user defined area.
                       "foreground" - foreground color rectangle as source.
                                      This type of stamp needs mask setting.
                                      Without mask, the stamp cannot figure
                                      its own size, so nothing can be drawn.

            "filename" : LIST of .jpg/png filepaths of stamp source.
                         Multiple filepaths mean 'this stamp has
                         tiles in separeted files'.

                         Otherwise, it will be a single picture stamp.

            "scale" : A tuple of default scaling ratio, 
                      (horizontal_ratio, vertical_ratio)

            "angle" : A floating value of default angle,in degree. 

            "mask" : List of filepath of mask source .jpg/.png, 
                     These files must be 8bit grayscale.

                     Multiple filepath means this stamp has
                     multiple masks which corespond to each tile.

                     Also,each mask size MUST be same as stamp picture.
                     For foreground type stamp, there is no source stamp picture,
                     so mask defines the shape of stamp. 

            "tile" : A tuple of (width, height),it represents tile size
                     of a picture. 
                     Currently this setting use with 'tiled-file' source only.
                     
            "tile-type" : "random" - The next tile index is random value.
                          "increment" - The next tile index is automatically incremented.
                          "same" - The next tile index is always default index.
                                   user change it manually.This is default tile-type.

            "tile-default-index" : The default index of tile. by default, it is 0.

                     Needless to say, 'tile-*' setting will ignore 
                     when there is no tile setting.

        """

        if jo['version'] == "1":
            settings = jo['settings']
            source = settings['source']
            if source == 'file':
                stamp = Stamp(jo['name'])
                assert 'filenames' in settings
                stamp.set_file_sources(settings['filenames'])
            elif source == 'tiled-file':
                stamp = Stamp(jo['name'])
                assert 'filenames' in settings
                stamp.set_file_sources(settings['filenames'])
                stamp.set_tile_division(*settings.get('tile', (0, 0)))
            elif source == 'clipboard':
                stamp = ClipboardStamp(jo['name'], self._app.doc)
            elif source == 'layer':
                stamp = LayerStamp(jo['name'],self._app.doc.model.layer_stack)
            elif source == 'current_visible':
                stamp = VisibleStamp(jo['name'],self._app.doc.model.layer_stack)
            elif source == 'foreground':
                stamp = ForegroundStamp(jo['name'])
            else:
                raise NotImplementedError("Unknown source %r" % source)

            # common setting
            if 'scale' in settings:
                stamp.set_default_scale(*settings['scale'])

            if 'angle' in settings:
                stamp.set_default_angle(math.radians(settings['angle'] % 360.0))

            if 'mask' in settings:
                stamp.set_mask_sources(settings['mask'])


            if 'gtk-thumbnail' in settings:
                pixbuf = Gtk.IconTheme.get_default().load_icon(
                        settings['gtk-thumbnail'], Stamp.THUMBNAIL_SIZE, 0)
                stamp.set_thumbnail(pixbuf)
            else:
                stamp.set_thumbnail(
                        self.load_thumbnail(settings.get('thumbnail', None)))
            return stamp

        else:
            raise NotImplementedError("Unknown version %r" % jo['version'])

   #def save_stamp_to_file(self, filename, stamp):
   #    json_src = stamp.output_as_json()
   #    self.save_to_file(self, filename)

    def save_to_file(self, filename, json_src):
        filename = self._get_adjusted_path(filename)
        with open(filename, 'w') as ofp:
            json.dump(json_src, ofp)

## Built-in stamps
#  These stamps are built-in, automatically registered at OptionPresenter.

BUILT_IN_STAMPS = [
            { "version" : "1",
              "name" : "clipboard stamp",
              "settings" : {
                  "source" : "clipboard",
                  "gtk-thumbnail" : "gtk-paste"
                  }
            },
            { "version" : "1",
              "name" : "layer stamp",
              "settings" : {
                  "source" : "layer"
                  }
            }
        ]
              
            


def _test():
    from application import get_app
    app = get_app()
    m = StampPresetManager(None)
    print(m.initialize_icon_store())

if __name__ == '__main__':
    _test()



