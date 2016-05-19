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
import weakref

from gettext import gettext as _
from gi.repository import Gdk, Gtk
from gi.repository import GLib
from gi.repository import GdkPixbuf
import cairo

from gui.linemode import *
import lib
from gui.ui_utils import *
from lib.observable import event

## Function defs

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
#
#  These Source classes would be used as 'pixbuf pool' and
#  'setting cache' in Stamp classes.

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

    def get_current_src(self, tile_index):
        """ Get current (cached) src pixbuf.
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


    def __init__(self):
        self._source_files = []
        self._surfs = {}
        self._reset_info()

    ## Information methods
    @property
    def tile_count(self):
        return len(self._source_files)

    def get_current_src(self, tile_index):
        if self._ensure_current_pixbuf(tile_index):
            return self._surfs[tile_index]


    def set_file_sources(self, filenames):
        if type(filenames) == str:
            self._source_files = [filenames, ]
        else:
            self._source_files = filenames
        self._reset_info()

    def set_pixbuf(self, tile_index, pixbuf):
        surf = Gdk.cairo_surface_create_from_pixbuf(pixbuf, 1, None)
        self._surfs[tile_index] = surf

    def set_surface(self, tile_index, surf):
        self._surfs[tile_index] = surf

    def get_surface(self, tile_index):
        return self._surfs.get(tile_index, None)

    def _reset_info(self):
        """
        Reset infomations which should be cleared each drawing call.
        """
        self._cached_cairo_src = {}

    ## Source Pixbuf methods

    def _ensure_current_pixbuf(self, tile_index):
        if self._source_files and len(self._source_files) > 0:
            if not tile_index in self._surfs:
                filename = self._source_files[tile_index]
                junk, ext = os.path.splitext(filename)
                if ext.lower() == '.png':
                    surf = cairo.ImageSurface.create_from_png(filename)
                else:
                    pixbuf = GdkPixbuf.Pixbuf.new_from_file(filename)
                    surf = Gdk.cairo_surface_create_from_pixbuf(
                            pixbuf, 1, None)
                self._surfs[tile_index] = surf
            return True
        return False

    def clear_all(self):
        """
        Clear all cached pixbuf
        """
        self._surfs.clear()


    def apply(self, cr, tile_index):
        """ Apply pixbuf pattern to cairo.
        The pattern might be cached.

        This method should be called from draw() method.
        """
        stamp_src = self.get_current_src(tile_index)
        w = stamp_src.get_width() 
        h = stamp_src.get_height()
        ox = -(w / 2)
        oy = -(h / 2)
        cr.set_source_surface(stamp_src, ox, oy)


class _Dynamic_Source(_Pixbuf_Source):
    """ Dynamic sources, i.e. clipboard or layers 
    
    This class has 'surface_requested' event facility, to ensure generating pixbuf
    in various situation.
    This event should be used from some Stamp class.
    """

    @property
    def tile_count(self):
        """
        CAUTION: 

        Other source-class would hold preloaded surfaces
        and simply return len(self._surfs) for 'tile_count.

        but dynamic STAMP class might setup surfaces dynamically
        and self._surfs might not contain needed surfaces yet
        at some point in time.

        Therefore, the dynamic STAMP class might
        not use this method to know how many tiles are there.
        (Although it is rely on how implemented the dynamic
        stamp class is)
        """
        return len(self._surfs) 

    @event
    def surface_requested(self, tile_index):
        """ Event of pixbuf,invoked when tile_index is requested.
        It is actually access to Source object with
        get_current_src() method.

        event handler would be placed in Stamp Class,
        not inherited source class.

        USAGE EXAMPLE:

        class SampleStamp:

            def __init__(self):
                self._pixbuf_src = Some_Dynamic_Source()
                self._pixbuf_src.surface_requested += self.surf_req_cb

            def surf_req_cb(self, source, tile_index):
                source.set_pixbuf(tile_index, None) # Failed to Load

        NOTE:

        If failed to setup the pixbuf by some reason, 
        set that tile_index key (it would be Zero) to None
        with calling 'source.set_pixbuf(tile_index, None)'.
        """

    def get_current_src(self, tile_index):
        self.surface_requested(tile_index)
        return self._surfs.get(tile_index, None)


class _Tiled_Source(_Pixbuf_Source):
    """ Tiled_Source class generates tiles from a single pixbuf.
    """

    def __init__(self, width, height):
        """
        :param width: width of a tile
        :param height: height of a tile
        """
        self._reset_info()
        self._tile_w = width
        self._tile_h = height

    def _reset_info(self):
        super(_Tiled_Source, self)._reset_info()
        self._tile_count = 0

    @property
    def tile_count(self):
        return self._tile_count

    def _load_from_file(self, filename):
        pixbuf = GdkPixbuf.Pixbuf.new_from_file(filename)
        if self._tile_w > 0 and self._tile_h > 0:
            idx = 0
            for y in xrange(pixbuf.get_height() / self._tile_h):
                for x in xrange(pixbuf.get_width() / self._tile_w):
                    tpb = pixbuf.new_subpixbuf(x, y, 
                            self._tile_w, self._tile_h)
                    ox, oy = _StampMixin.get_offsets(tpb)
                    self.set_pixbuf(idx, tpb)
                    # tpb(temporary pixbuf) converted to cairo-surface 
                    # in set_pixbuf(), so unnescessary.
                    del cpb 

                    idx+=1
            del pixbuf
            self._tile_count = idx

    def _ensure_current_pixbuf(self, tile_index):
        if not 0 in self._surfs: 
            # index 0 should exist in every sort of tile setting.
            # therefore index 0 exist = all tiles loaded in
            # _Tiled_Source.
            self._load_from_file(self._source_filename)

class _Pixbuf_Mask(_Pixbuf_Source):
    """ cairo mask source."""

    def apply(self, cr, tile_index):
        """ Apply pixbuf pattern to cairo.
        The pattern might be cached.

        This method should be called from draw() method.
        """
        stamp_src = self.get_current_src(tile_index)
        w = stamp_src.get_width() 
        h = stamp_src.get_height()
        ox = -(w / 2)
        oy = -(h / 2)
        cr.mask_surface(stamp_src, ox, oy)

class _Tiled_Mask(_Tiled_Source,
                  _Pixbuf_Mask):
    """ tiled cairo mask."""

class _Dynamic_Mask(_Dynamic_Source,
                    _Pixbuf_Mask):
    """ dynamic mask """

## Stamp Mixins
#
#  These stamp classes manage drawing, 
#  calculating boundary box according to node's angle and scaling setting,
#  and pointing detection.
#  These classes utilize Source classes to manage pixbuf/tile/mask.
#
#  Stamp classes are generated by StampPresetManager class
#  and used from Stamptool class.

class _StampMixin(object):
    """ Stamp Mixin, base class of all stamp classes. """

    THUMBNAIL_SIZE = 32

    def _reset_members(self, name):
        self._pixbuf_src = None
        self._mask_src = None
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


    @property
    def tile_count(self):
        if self._pixbuf_src:
            return self._pixbuf_src.tile_count
        else:
            return 0

    def set_file_sources(self, filenames):
        if self._pixbuf_src == None:
            self._pixbuf_src = _Pixbuf_Source()
        self._pixbuf_src.set_file_sources(filenames)

    def set_mask_sources(self, filenames):
        if self._mask_src == None:
            self._mask_src = _Pixbuf_Mask()
        self._mask_src.set_file_sources(filenames)

    def set_thumbnail(self, pixbuf):
        self._thumbnail = pixbuf

    @property
    def thumbnail(self):
        return self._thumbnail


    @property
    def is_support_selection(self):
        return False

    @staticmethod
    def _get_outmost_rect_from_points(pts):
        sx, sy = pts[0]
        ex, ey = sx, sy
        for tx, ty in pts[1:]:
            if tx < sx:
                sx = tx
            elif tx > ex:
                ex = tx

            if ty < sy:
                sy = ty
            elif ty > ey:
                ey = ty
        return (sx, sy, ex, ey)

    ## Drawing methods

    def initialize_draw(self, cr):
        """ Initialize draw calls.
        Call this method prior to draw stamps loop.
        """
        pass

    def finalize_draw(self, cr):
        """ Finalize draw calls.
        Call this method after draw stamps loop.

        This method called from the end of each drawing sequence,
        NOT END OF DRAWING PHASE OF STAMPTOOL! 
        Therefore, source.finalize() MUST not be called here!
        """
        pass
        
    

    def draw(self, tdw, cr, x, y, node, save_context=False):
        """ Draw this stamp into cairo surface.
        This implementation is as base class,
        node.tile_index ignored here.
        """
        stamp_src = self._pixbuf_src.get_current_src(node.tile_index)
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

            self._pixbuf_src.apply(cr, node.tile_index)
            if self._mask_src:
                self.mask_src.apply(cr, node.tile_index)

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
        if self._pixbuf_src:
            stamp_src = self._pixbuf_src.get_current_src(node.tile_index)
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
                    cos_s = math.cos(node.angle)
                    sin_s = math.sin(node.angle)
                    points = []
                    for i, x, y in enum_area_point(sx, sy, ex, ey):
                        tx = (cos_s * x - sin_s * y) + bx
                        ty = (sin_s * x + cos_s * y) + by
                        points.append( (tx, ty) )
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
            area = _StampMixin._get_outmost_rect_from_points(pos)
            return get_outmost_area(tdw, 
                    *area,
                    margin=margin)


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


class _DynamicStampMixin(_StampMixin):
    """ Non-tiled single pixbuf stamp mixin.

    This mixin used for dynamically changed stamps with
    utilizing _Dynamic_Source class and its event functionary.
    """

    def _reset_members(self, name):
        super(_DynamicStampMixin, self)._reset_members(name)
        self._pixbuf_src = _Dynamic_Source()
        self._pixbuf_src.surface_requested += self.surface_requested_cb

    def set_file_sources(self, filenames):
        pass

    def surface_requested_cb(self, source, tile_index):
        """
        Pixbuf requested callback, called from 
        self._pixbuf_src._ensure_current_pixbuf()
        """
        pass


## Stamp Classes
#

class Stamp(_StampMixin):
    """
    This class is standard Stamp class.
    This would have multiple pixbufs as tile,which from multiple picture files.

    When 'finalize' action executed (i.e. ACCEPT button pressed)
    the cairo surface converted into GdkPixbuf,
    and that GdkPixbuf converted into Mypaint layer
    and it is merged into current layer.
    """

    def __init__(self, name):
        self._reset_members(name)
        self._pixbuf_src = _Pixbuf_Source()

    ## Information methods
    @property
    def tile_count(self):
        if self._pixbuf_src:
            return self._pixbuf_src.tile_count
        else:
            return 1

    @property
    def is_ready(self):
        """
        In some stamp class, 
        it might not be ready to stamp a tile
        at some point in time.
        For example, there is no clipboard bitmap
        for ClipboardStamp... 
        Otherwise, this property always return True.
        """
        return True


    ## Information methods

    ## Drawing methods

    


class TiledStamp(_StampMixin):
    """
    Tiled stamp. 
    'tiled' means 'a picture(pixbuf) divided into small parts(tiles)'
    """

    def __init__(self, name, tw, th):
        super(TiledStamp, self).__init__(name)
        self._pixbuf_src = _Tiled_Source(tw, th)
        self._tile_w = tw
        self._tile_h = th

    def set_mask_source(self, filenames):
        self._mask_src = _Tiled_Mask(self._tile_w, self._tile_h)
        self._mask_src.set_filenames(filenames)

class ClipboardStamp(_DynamicStampMixin):

    def __init__(self, name, doc):
        self._reset_members(name)
        self._doc = doc

    @property
    def is_ready(self):
        return self._pixbuf_src.tile_count == 1

    def _get_clipboard(self):
        # XXX almost copied from gui.document._get_clipboard()
        # we might need some property interface to get it from
        # document class directly...?
        display = self._doc.tdw.get_display()
        cb = Gtk.Clipboard.get_for_display(display, Gdk.SELECTION_CLIPBOARD)
        return cb

    def _load_clipboard_image(self): 
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

        self._pixbuf_src.set_pixbuf(0, pixbuf)

    def initialize_phase(self):
        """ Initializing for start of each drawing phase.
        CAUTION: This method is not called when drawing to
        layer.
        """
        self._load_clipboard_image()

    def pixbuf_requested_cb(self, source, tile_index):
        assert tile_index == 0
        if self._pixbuf_src.tile_count == 0:
            self._load_clipboard_image()

class LayerStamp(_DynamicStampMixin):
    """ A Stamp which sources the current layer.

    This class has selection(target) areas, which define
    a rectangular area as the source of stamp. 
    Each area is model coordinate.
    """

    def __init__(self, name, rootstack):
        self._reset_members(name)
        self._sel_areas = []
        self._rootstack = rootstack
        self._area_layers = {}

    @property
    def is_support_selection(self):
        return True
    
    @property
    def is_ready(self):
        return len(self._sel_areas) > 0

    def set_selection_area(self, tile_index, area):
        """ Set (add) selection area to this object.
        when the new area added, the source is always
        'currently selected layer'

        :param tile_index: the index of tile == index of selection area
                           if this is -1, area added.
        :param area: a tuple of (sx, sy, ex, ey) , in model.
        """
        if tile_index == -1:
            # Set last selection area == add selection area
            tile_index = len(self._sel_areas) 
            assert not tile_index in self._sel_areas
            self._sel_areas.append(area)
            self.set_layer_for_area(tile_index,
                self._rootstack.current)
        else:
            self._sel_areas[tile_index] = area
        return tile_index

    def get_selection_area(self, tile_index):
        if tile_index < len(self._sel_areas):
            return self._sel_areas[tile_index]

    def remove_selection_area(self, tile_index):
        if tile_index in self._sel_areas:
            del self._sel_areas[tile_index]
            assert tile_index in self._area_layers
            del self._area_layers[tile_index]


    def enum_visible_selection_areas(self, tdw, indexes=None):
        """
        Enumerate visible selection areas at tdw.
        :param tdw: TiledDrawWidget to display
        :param indexes: sequence of index of self._sel_areas. 
                        an index might be None.
        :rtype: yielding a tuple of (index, (start_x, start_y, end_x, end_y)).
                returned values are display coordinate.
        """

        # XXX mostly same as Overlay_Stamp._get_oncscreen_areas()
        # but that method needs to adjust the points with
        # user dragging.

        alloc = tdw.get_allocation()

        def check_area(area):
            sx, sy, ex, ey = get_outmost_area(tdw, *area, margin=0)
            w = (ex - sx) + 1
            h = (ey - sy) + 1
            
            node_on_screen = (
                sx > alloc.x - w  and
                sy > alloc.y - h and
                sx < alloc.x + alloc.width + w and
                sy < alloc.y + alloc.height + h
            )
            
            if node_on_screen:
                return (sx, sy, ex, ey)

        if indexes == None:
            for i, area in enumerate(self._sel_areas):
                ret = check_area(area)
                if ret:
                    yield (i, ret)
        else:
            for i in indexes:
                if i != None:
                    ret = check_area(self._sel_areas[i])
                    if ret:
                        yield (i, ret)

    @property
    def tile_count(self):
        return len(self._sel_areas)

    def initialize_phase(self):
        """ Initialize when a entire phase stated
        i.e. where self.phase is set to _Phase.CAPTURE.
        """
        self._pixbuf_src.clear_all()

    # Pixbuf Source related  
    def set_layer_for_area(self, tile_index, layer):
        self._area_layers[tile_index] = weakref.ref(layer)

    def get_layer_for_area(self, tile_index):
        ref = self._area_layers.get(tile_index, None)
        if ref:
            return ref()

    def _fetch_single_area(self, layer, idx):
        sx, sy, ex, ey = self._sel_areas[idx]
        return layer.render_as_pixbuf(
                int(sx), int(sy), int(ex-sx)+1, int(ey-sy)+1,
                alpha=True)

    def refresh_surface(self, tile_index, source=None):
        if tile_index < len(self._sel_areas):
            if not tile_index in self._area_layers:
                layer = self._rootstack.current
                self.set_layer_for_area(tile_index, layer)
            else:
                layer = self.get_layer_for_area(tile_index)

            if layer != None:
                if source == None:
                    source = self._pixbuf_src
                source.set_pixbuf(tile_index, 
                        self._fetch_single_area(layer, tile_index))

    def surface_requested_cb(self, source, tile_index):
        """
        Pixbuf requested callback, called from 
        self._pixbuf_src._ensure_current_pixbuf()

        :param source: source object, actually it should be 'self._pixbuf_src'.
        :param tile_index: target tile index, to be used right after this callback.
        """

        # Check surface existence, and then,
        # set 'PIXBUF' into source object. not SURFACE.
        # (and the pixbuf is coverted to surface inside set_pixbuf method)
        if source.get_surface(tile_index) == None:
            self.refresh_surface(tile_index,source=source)



class VisibleStamp(LayerStamp):
    """ A Stamp which sources the currently visible area.
    """

    def __init__(self, name, rootstack):
        self._reset_members(name)
        self._sel_areas = []
        self._rootstack = rootstack

    def set_layer_for_area(self, tile_index, layer):
        pass # Disable this method

    def surface_requested_cb(self, source, tile_index):
        """
        Pixbuf requested callback, called from 
        self._pixbuf_src._ensure_current_pixbuf()
        """
        if source.get_surface(tile_index) == None:
            source.set_pixbuf(tile_index, 
                    self._fetch_single_area(self._rootstack, tile_index))
        return True


class ForegroundStamp(_DynamicStampMixin):
    """ Foreground color stamp.
    """

    def __init__(self, name, app, tw, th):
        self._reset_members(name)
        self._app = app
        self._tile_w = tw
        self._tile_h = th

    @property
    def foreground_color(self):
        return self._app.brush_color_manager.get_color().get_rgb()


    def set_file_sources(self, filenames):
        """ to disable setting self._pixbuf_src """
        pass

    def set_mask_sources(self, filenames):
        super(ForegroundStamp, self).set_mask_sources(filenames)
        self._pixbuf_src = self._mask_src


    def draw(self, tdw, cr, x, y, node, save_context=False):
        """ Draw this stamp into cairo surface.
        This implementation is as base class,
        node.tile_index ignored here.
        """
        if not self._mask_src:
            return 

        if save_context:
            cr.save()

        stamp_src = self._mask_src.get_current_src(node.tile_index)

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

        cr.set_source_rgb(*self.foreground_color)
        cr.rectangle(ox, oy, w, h) 
        cr.clip()
        cr.fill()
        self._mask_src.apply(cr, node.tile_index) # Place this code after fill!

        
        if save_context:
            cr.restore()

class ForegroundLayerStamp(ForegroundStamp,
                           LayerStamp):
    """ Another version of Foreground color stamp.
    In this version,mask is generated from current layer.
    """

    def __init__(self, name, app):
        layerstack = app.doc.model.layer_stack
        # Call LayerStamp constructor
        super(ForegroundStamp, self).__init__(name, layerstack)
        self._app = app


    def _reset_members(self, name):
        super(_DynamicStampMixin, self)._reset_members(name)
        self._mask_src = _Dynamic_Mask()
        self._pixbuf_src = self.mask_src
        self._pixbuf_src.surface_requested += self.surface_requested_cb

class PixbufStamp(_DynamicStampMixin):
    """ 
    A Stamp for dynamically changeable pixbuf stamps
    (Currently, it is only ClipboardStamp)
    to fix its contents within DrawStamp command.
    This stamp would only use inside program,have no any user
    direct interaction.

    Due to MyPaint 'Command' processing done in asynchronously,
    Clipboard stamp might be updated its content(pixbuf)
    when actually draw_stamp_to_layer() called
    i.e. StampCommand has issued.

    Therefore, we need to pass the pixbuf used when editing
    to this stamp class, instead of ClipboardStamp.
    """

    def __init__(self, name, pixbuf_src):
        super(PixbufStamp, self).__init__(name)
        self._pixbuf_src = pixbuf_src


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

        # XXX mostly copied from gui/application.py _init_icons()
        icon_theme = Gtk.IconTheme.get_default()
        if app:
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
            "tooltip" : (Optional) tooltip message of this stamp,for iconview item.

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
                assert 'filenames' in settings
                stamp = Stamp(jo['name'])
                stamp.set_file_sources(settings['filenames'])
            elif source == 'tiled-file':
                assert 'filenames' in settings
                assert 'tile' in settings
                stamp = TiledStamp(jo['name'], *settings.get('tile', (1, 1)))
                stamp.set_file_sources(settings['filenames'])
            elif source == 'clipboard':
                stamp = ClipboardStamp(jo['name'], self._app.doc)
            elif source == 'layer':
                stamp = LayerStamp(jo['name'],self._app.doc.model.layer_stack)
            elif source == 'current-visible':
                stamp = VisibleStamp(jo['name'],self._app.doc.model.layer_stack)
            elif source == 'foreground':
                stamp = ForegroundStamp(jo['name'], self._app, *settings.get('tile', (1, 1)))
                assert 'mask' in settings
            elif source == 'foreground-layermask':
                stamp = ForegroundLayerStamp(jo['name'], self._app)
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

            # TODO tooltip does not supported yet
            return stamp

        else:
            raise NotImplementedError("Unknown version %r" % jo['version'])


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
                  },
              "tooltip" : _("Stamp of Clipboard Image")
            },
            { "version" : "1",
              "name" : "layer stamp",
              "settings" : {
                  "source" : "layer"
                  },
              "tooltip" : _("Stamp of area of Layer.")
            },
            { "version" : "1",
              "name" : "visible stamp",
              "settings" : {
                  "source" : "current-visible"
                  },
              "tooltip" : _("Stamp of area of current visible.")
            },
            { "version" : "1",
              "name" : "layer mask",
              "settings" : {
                  "source" : "foreground-layermask"
                  },
              "tooltip" : _("Stamp of forground color, masked with area of Layer.")
            },
        ]
              
            



if __name__ == '__main__':
    pass



