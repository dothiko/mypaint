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
import lib.surface

## Function defs

def render_stamp_to_layer(target_layer, stamp, nodes, bbox):
    """
    Rendering stamps to layer.

    :param bbox: boundary box, in model coordinate
    """

    # TODO this type of processing might waste too many memory
    # if two stamps are placed too far from each other.
    # so, in such case, we will need generate small surface/layers
    # for each stamps and composite them into the target layer.
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

    lib.surface.finalize_surface(dstsurf, tiles)

    bbox = tuple(target_layer.get_full_redraw_bbox())
    target_layer.root.layer_content_changed(target_layer, *bbox)
    target_layer.autosave_dirty = True
    dstsurf.remove_empty_tiles()
    del layer


## Class defs

## Source Mixins
#
#  These Source Mixins provide some functionality like
#  'surface pool' or 'cache management' for Stamp classes.
#
#  INITIALIZE:
#  Source mixin has _init_source() initializer,
#  this must be called from __init__() of Stamp class
#  as well as _init_stamp(name) of Stamp mixin.

class _SourceMixin(object):

   #def _init_source(self):
   #    pass

   #@property
   #def tile_count(self):
   #    return 1

   #def get_current_src(self, tile_index):
   #    """ Get current (cached) src pixbuf.
   #    """
   #    pass

    @staticmethod
    def get_offsets(pixbuf):
        return (-(pixbuf.get_width() / 2),
                -(pixbuf.get_height() / 2))

    def set_surface_from_pixbuf(self, tile_index, pixbuf):
        surf = Gdk.cairo_surface_create_from_pixbuf(pixbuf,
                1, None)
        self._surfs[tile_index] = surf

    def set_surface(self, tile_index, surf):
        if surf:
            self._surfs[tile_index] = surf
        else:
            del self._surfs[tile_index]

    def get_surface(self, tile_index):
        return self._surfs.get(tile_index, None)

    def get_tileindex_from_rawindex(self, raw_index):
        """
        Get tile index from raw index.
        'raw index' means , the index of self._surfs.keys().
        not tileindex itself.
        """
        if raw_index < len(self._surfs):
            return self._surfs.keys()[raw_index]
        else:
            return -1

    def get_rawindex_from_tileindex(self, tile_index):
        if tile_index in self._surfs:
            return self._surfs.keys().index(tile_index)
        else:
            return -1

    def get_valid_tiles(self):
        return self._surfs.keys()

    def validate_all_tiles(self):
        """ Validate all tiles.
        (= load tiles into surface memory) 
        """
        pass


    def get_desc(self, tile_index):
        """ Returns a string which is a description of a tile.
        For example, in Layerstamp, it should be 
        the name of source layer. """ 
        pass


class _PixbufSourceMixin(_SourceMixin):
    """ Pixbuf source management class.
    This is used as 'Pixbuf source' and 
    also 'Mask source' at Stamp class.
    """


    def _init_source(self):
        self._source_files = {}
        self._surfs = {}
        self._tile_index_base = 0

    ## Information methods
    @property
    def tile_count(self):
        return len(self._source_files)

    def get_current_src(self, tile_index):
        if self._ensure_current_pixbuf(tile_index):
            return self._surfs[tile_index]


    def set_file_sources(self, filenames):
        if type(filenames) == str:
            self._source_files[0] = filenames
        else:
            for i, filename in enumerate(filenames):
                self._source_files[i] = filename
            self._tile_index_base = i

    @property
    def latest_tile_index(self):
       #return len(self._source_files) - 1
        return self._tile_index_base


    ## Source Pixbuf methods

    def _ensure_current_pixbuf(self, tile_index):
        """
        To ensure current pixbuf (and cached surface) loaded. 
        Automatically called from get_current_src() method.

        This facility is completely different at _Dynamic_Source.
        see _Dynamic_source.get_current_src()
        """
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


    def clear_all_cache(self):
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

    def get_desc(self, tile_index):
        return self._source_files[tile_index]

    def validate_all_tiles(self):
        for ck in self._source_files.keys():
            self._ensure_current_pixbuf(ck)

class _DynamicSourceMixin(_PixbufSourceMixin):
    """ Dynamic sources, i.e. clipboard or layers 
    
    This class has 'surface_requested' event facility, to ensure generating pixbuf
    in various situation.
    This event should be used from some Stamp class.
    """

    def _init_source(self):
        self._surfs = {}

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

    def surface_requested(self, tile_index):
        """ Event of pixbuf,invoked when tile_index is requested.
        It is actually access to Source object with
        get_current_src() method.
        """
        pass

    def get_current_src(self, tile_index):
        """
        Differred from _Pixbuf_Source, 
        ensureing cached surface should be executed 
        surface_requested observable event,
        which is implemented at Stamp class.
        """
        self.surface_requested(tile_index)
        return self._surfs.get(tile_index, None)

    def clear_surface(self, tile_index):
        if tile_index in self._surfs:
            del self._surfs[tile_index]


class _TiledSourceMixin(_PixbufSourceMixin):
    """ Tiled_Source class generates tiles from a single pixbuf.
    """

    def _init_source(self, width, height):
        """
        :param width: width of a tile
        :param height: height of a tile
        """
        super(_TiledSourceMixin, self)._init_source()
        self._tile_w = width
        self._tile_h = height


    def _load_from_file(self, filename):
        pixbuf = GdkPixbuf.Pixbuf.new_from_file(filename)
        if self._tile_w > 0 and self._tile_h > 0:
            idx = 0
            for y in xrange(pixbuf.get_height() / self._tile_h):
                for x in xrange(pixbuf.get_width() / self._tile_w):
                    tpb = pixbuf.new_subpixbuf(x, y, 
                            self._tile_w, self._tile_h)
                    ox, oy = _SourceMixin.get_offsets(tpb)
                    self.set_surface_from_pixbuf(idx, tpb)
                    del tpb 
                    idx+=1
            del pixbuf
            self._tile_count = idx

    def _ensure_current_pixbuf(self, tile_index):
        if not 0 in self._surfs: 
            # index 0 should exist in every sort of tile setting.
            # therefore index 0 exist = all tiles loaded in
            # _Tiled_Source.
            self._load_from_file(self._source_filename)

class _PixbufMaskMixin(_PixbufSourceMixin):
    """ cairo mask source."""

    def apply(self, cr, tile_index):
        pass

    def apply_mask(self, cr, tile_index):
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


## Stamp Mixins
#
#  These stamp classes manage drawing, 
#  calculating boundary box according to node's angle and scaling setting,
#  and pointing detection.
#  These classes utilize Source classes to manage pixbuf/tile/mask.
#
#  Stamp classes are generated by StampPresetManager class
#  and used from Stamptool class.
#
#  INITIALIZE:
#  Stamp mixin has _init_stamp(name) initializer,
#  this must be called from __init__() of Stamp class
#  as well as _init_source() of Source mixin.

class _StampMixin(object):
    """ Stamp Mixin, base class of all stamp classes. """

    THUMBNAIL_SIZE = 32

    def _init_stamp(self, name, desc):
        """
        :param name: name of stamp
        :param desc: stamp description, it should be same as tooltip.
        """
        self.name = name
        self.desc = desc
        self._default_scale_x = 1.0
        self._default_scale_y = 1.0
        self._default_angle = 0.0
        self._thumbnail = None
        self._tile_index_seed = 0

    ## Information methods / properties

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


    def set_thumbnail(self, pixbuf):
        self._thumbnail = pixbuf

    @property
    def thumbnail(self):
        return self._thumbnail


    @property
    def is_support_selection(self):
        return False

    @property
    def is_ready(self):
        return True

    @property
    def latest_tile_index(self):
        return self.tile_count - 1



    ## Static methods

    @staticmethod
    def _get_outmost_area_from_points(pts):
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
        stamp_src = self.get_current_src(node.tile_index)
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

            self.apply(cr, node.tile_index)

            cr.rectangle(ox, oy, w, h) 

            cr.clip()
            cr.paint()
            
            if save_context:
                cr.restore()

    ## Boundary / hit-check methods

    def get_handle_index(self, tdw, x, y, node, margin):
        """
        Get control handle index
        :param x,y: Cursor position in display coordinate.
        :param node: Target node,which contains scale/angle information.
        :param margin: i.e. the half of control handle rectangle

        :rtype: integer, whitch is targetted handle index.
                Normally it should be either one of targetted handle index (0 to 3),
                or -1 (cursor is not on this node)
                When 'node is targetted but not on control handle',
                return 4.

        """
        pos = self.get_boundary_points(node, tdw=tdw)
        if pos:
            for i, cp in enumerate(pos):
                cx, cy = cp
                if (cx - margin <= x <= cx + margin and
                        cy - margin <= y <= cy + margin):
                    return i

            # Returning 4 means 'inside but not on a handle'
            if (is_inside_triangle(x, y, (pos[0], pos[1], pos[2])) or
                    is_inside_triangle(x, y, (pos[0], pos[2], pos[3]))):
                return 4 

        return -1

    def get_boundary_points(self, node, tdw=None, 
            dx=0.0, dy=0.0, no_rotate=False, no_scale=False):
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
        stamp_src = self.get_current_src(node.tile_index)
        if stamp_src:
            w = stamp_src.get_width() 
            h = stamp_src.get_height()
            if not no_scale:
                w *= node.scale_x 
                h *= node.scale_y
            sx = - w / 2
            sy = - h / 2
            ex = w+sx
            ey = h+sy
            bx = node.x + dx
            by = node.y + dy

            if node.angle != 0.0 and not no_rotate:
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

        :param tdw: Tiledrawwidget to get display coordinate.
                    if tdw is None, result is in model coordinate.
        """

        # We need outmost position in model coordinate,
        # and convert it after all processing finished.
        pos = self.get_boundary_points(node, tdw=None, dx=dx, dy=dy)
        if pos:
            area = _StampMixin._get_outmost_area_from_points(pos)
            sx, sy, ex, ey = get_outmost_area(tdw, 
                    *area,
                    margin=margin)
            return (sx, sy, abs(ex-sx)+1, abs(ey-sy)+1)
            


    ## Phase related methods
    #  these methods should be called from 
    #  Stamptool._start_new_capture_phase()

    def initialize_phase(self, mode):
        """ Initializing for start of each drawing phase.
        :param mode: the stampmode which use this stamp class.

        CAUTION: This method is not called when drawing to
        layer.

        the code flow around this method should be :

        ACCEPT BUTTON Pressed (end of drawing phase)
        |
        +-- render_stamp_to_layer called (from Command object : asynchronously)
        |
        self.finalize_phase() called
        |
        some stampmode initializing lines called
        |
        self.initialize_phase() called
        """
        pass

    def finalize_phase(self, mode):
        """ This called when stamptool comes to end of
        capture(drawing) phase.
        """
        pass

    ## Enter/Leave callbacks
    #  These callbacks called when stamptool.stamp attribute
    #  has been changed.

    def enter(self, doc):
        """
        Called when stamp has selected.
        """
        pass

    def leave(self, doc):
        """
        Called when stamp has unselected.
        """
        pass
    


## Stamp Classes
#

class Stamp(_PixbufSourceMixin, _StampMixin):
    """
    This class is standard Stamp class.
    This would have multiple pixbufs as tile,which from multiple picture files.

    When 'finalize' action executed (i.e. ACCEPT button pressed)
    the cairo surface converted into GdkPixbuf,
    and that GdkPixbuf converted into Mypaint layer
    and it is merged into current layer.
    """

    def __init__(self, name, desc):
        self._init_source()
        self._init_stamp(name, desc)

    ## Information methods

    ## Information methods

    ## Drawing methods



class TiledStamp(_TiledSourceMixin, _StampMixin):
    """
    Tiled stamp. 
    'tiled' means 'a picture(pixbuf) divided into small parts(tiles)'
    """
    def __init__(self, name, desc, tw, th):
        self._init_source(tw, th)
        self._init_stamp(name, desc)


class ClipboardStamp(_DynamicSourceMixin, _StampMixin):

    def __init__(self, name, desc):
        self._init_source()
        self._init_stamp(name, desc)

    @property
    def is_ready(self):
        return self.tile_count == 1

    def _get_clipboard(self, tdw):
        # XXX almost copied from gui.document._get_clipboard()
        # we might need some property interface to get it from
        # document class directly...?
        display = tdw.get_display()
        cb = Gtk.Clipboard.get_for_display(display, Gdk.SELECTION_CLIPBOARD)
        return cb

    def _load_clipboard_image(self, tdw): 
        # XXX almost copied from gui.document.paste_cb()
        clipboard = self._get_clipboard(tdw)

        targs_avail, targets = clipboard.wait_for_targets()
        if not targs_avail:
            # Nothing on clipboard."
            return

        # Then grab any available image, also synchronously
        pixbuf = clipboard.wait_for_image()
        if not pixbuf:
            return

        self.set_surface_from_pixbuf(0, pixbuf)

    def initialize_phase(self, mode):
        """ Initializing for start of each drawing phase.
        CAUTION: This method is not called when drawing to
        layer.
        """
        self._load_clipboard_image(mode.doc.tdw)

    def surface_requested(self, tile_index):
        assert tile_index == 0
        if self.tile_count == 0:
            self._load_clipboard_image()

    @property
    def latest_tile_index(self):
        """
        Clipboard stamp has always 1 source, so index is 0.
        """
        return 0

class LayerStamp(_DynamicSourceMixin, _StampMixin):
    """ A Stamp which sources the current layer.

    This class has selection(target) areas, which define
    a rectangular area as the source of stamp. 
    Each area is model coordinate.
    """

    def __init__(self, name, desc):
        self._init_source()
        self._init_stamp(name, desc)

    def _init_stamp(self, name, desc):
        super(LayerStamp, self)._init_stamp(name, desc)
        self._sel_areas = {}
        self._tile_index_seed = 0
        self._mode_ref = None

    @property
    def is_support_selection(self):
        return True
    
    @property
    def is_ready(self):
        return len(self._sel_areas) > 0

    def get_selection_area(self, tile_index):
        return self._sel_areas.get(tile_index, (None,None))[0]

    def set_selection_area(self, tile_index, area, layer):
        """ Set (add) selection area to this object.

        :param tile_index: the index of tile == index of selection area
                           if this is -1, area added.
        :param area: a tuple of (sx, sy, ex, ey) , in model.
        :param layer: the layer correspond to tile.
                      this would be model.layer_stack.current
        """
        if tile_index == -1:
            # Set last selection area == add selection area
            tile_index = self._tile_index_seed 
            assert not tile_index in self._sel_areas
            layer_ref = weakref.ref(layer)
            self._tile_index_seed += 1
        else:
            assert tile_index in self._sel_areas
            oldarea, oldlayer = self._sel_areas[tile_index]
            if layer == None:
                layer_ref = oldlayer
            else:
                layer_ref = weakref.ref(layer)

        self._sel_areas[tile_index] = (area, layer_ref)
        return tile_index

    def remove_selection_area(self, tile_index):
        if tile_index in self._sel_areas:
            del self._sel_areas[tile_index]
            self.clear_surface(tile_index)

            # Notify deletion to stamptool mode
            assert self._mode_ref != None
            mode = self._mode_ref()
            if mode:
                mode.stamp_tile_deleted_cb(tile_index)

    def enum_visible_selection_areas(self, tdw, indexes=None, 
            enum_outmost_area=True):
        """
        Enumerate visible selection areas in display coordinate at tdw.
        :param tdw: TiledDrawWidget to convert display. 
        :param indexes: sequence of index of self._sel_areas. 
                        an index might be None.
        :param enum_raw_area: if True, yields outmost area of a selection. 
                              otherwise, yields a selection area.
        :rtype: yielding a tuple of (tile_index, (start_x, start_y, end_x, end_y)).
                they are in display coordinate.
        """

        alloc = tdw.get_allocation()

        def check_area(area):
            sx, sy, ex, ey = get_outmost_area(tdw, *area, margin=0)
            w = (ex - sx) + 1
            h = (ey - sy) + 1
            
            area_on_screen = (
                sx > alloc.x - w  and
                sy > alloc.y - h and
                sx < alloc.x + alloc.width + w and
                sy < alloc.y + alloc.height + h
            )
            
            if area_on_screen:
                return (sx, sy, ex, ey)

        if indexes == None:
            for tile_idx in self._sel_areas:
                area, layer = self._sel_areas[tile_idx]
                ret = check_area(area)
                if ret:
                    if enum_outmost_area:
                        yield (tile_idx, ret)
                    else:
                        sx, sy = tdw.model_to_display(area[0], area[1])
                        ex, ey = tdw.model_to_display(area[2], area[3])
                        yield (tile_idx, 
                                (sx, sy, ex, ey))

        else:
            for i in indexes:
                if i != None:
                    if i in self._sel_areas:
                        area, layer = self._sel_areas[i]
                        ret = check_area(area)
                        if ret:
                            if enum_outmost_area:
                                yield (i, ret)
                            else:
                                sx, sy = tdw.model_to_display(area[0], area[1])
                                ex, ey = tdw.model_to_display(area[2], area[3])
                                yield (i, (sx, sy, ex, ey))


    @property
    def tile_count(self):
        return len(self._sel_areas)

    @property
    def latest_tile_index(self):
        if len(self._sel_areas) > 0:
            return self._sel_areas.keys()[-1]
        else:
            return -1

    # Pixbuf Source related  
    def set_layer_for_area(self, tile_index, layer):
        area, layer = self._sel_areas[tile_index]
        self._sel_areas[tile_index] = (area, weakref.ref(layer))

    def get_layer_for_area(self, tile_index):
        area, layer = self._sel_areas[tile_index]
        return layer()

    def _fetch_single_area(self, layer, area):
        sx, sy, ex, ey = area
        return layer.render_as_pixbuf(
                int(sx), int(sy), int(ex-sx)+1, int(ey-sy)+1,
                alpha=True)

    def refresh_surface(self, tile_index):
        if tile_index in self._sel_areas:
            area, layer = self._sel_areas[tile_index]
            layer = layer() # restore reference
            if layer != None:
                self.set_surface_from_pixbuf(tile_index, 
                        self._fetch_single_area(layer, area))

    def surface_requested(self, tile_index):
        """
        Pixbuf requested callback, called from 
        self._ensure_current_pixbuf()

        :param tile_index: target tile index, to be used right after this callback.
        """

        # Check surface existence, and then,
        # set 'PIXBUF' into source object. not SURFACE.
        if self.get_surface(tile_index) == None:
            self.refresh_surface(tile_index)

    ## Phase related

    def initialize_phase(self, mode):
        """ Initialize when a entire phase stated
        i.e. where self.phase is set to _Phase.CAPTURE.
        """
        self._mode_ref = weakref.ref(mode)

    ## Entering/Leaving stamp callbacks
    def enter(self, doc):
        doc.model.sync_pending_changes += self.sync_pending_changes_cb

    def leave(self, doc):
        doc.model.sync_pending_changes -= self.sync_pending_changes_cb

    def sync_pending_changes_cb(self, model, flush=True, **kwargs):
        self.clear_all_cache()

        rejected_layers = []

        for tile_idx in self._sel_areas.keys():
            area, layer = self._sel_areas[tile_idx]
            layer = layer()
            if layer == None or layer in rejected_layers:
                self.remove_selection_area(tile_idx)
            else:
                lidx = model.layer_stack.deepindex(layer)
                if lidx == None:
                    self.remove_selection_area(tile_idx)
                    rejected_layers.append(layer)

class VisibleStamp(LayerStamp):
    """ A Stamp which sources the currently visible area
    = currently canvas contents, not single layer contents.
    """

    def __init__(self, name, desc):
        self._init_stamp(name, desc)

    def set_selection_area(self, tile_index, area, layer):
        super(VisibleStamp, self).set_selection_area(
                tile_index, area, layer.root)

    def set_layer_for_area(self, tile_index, layer):
        super(VisibleStamp, self).set_layer_for_area(
                tile_index, layer.root)



class ForegroundStamp(_PixbufMaskMixin, _StampMixin):
    """ Foreground color stamp.
    """

    def __init__(self, name, desc):
        self._init_source()
        self._init_stamp(name, desc)

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

        fg_col = tdw.app.brush_color_manager.get_color().get_rgb()

        cr.set_source_rgb(*fg_col)
        cr.rectangle(ox, oy, w, h) 
        cr.clip()
        cr.fill()
        self.apply_mask(cr, node.tile_index) # Place this code after fill!

        
        if save_context:
            cr.restore()

class ForegroundLayerStamp(ForegroundStamp,
                           LayerStamp):
    """ Another version of Foreground color stamp.
    In this version,mask is generated from current layer.
    """

    def __init__(self, name, desc):
        # Call LayerStamp constructor
        # (CAUTION: NOT ForegroundStamp)
        # actually & currently, the both is same
        # but this class largely based on layerstamp
        # except for draw() method.
        LayerStamp.__init__(self, name, desc)

    def draw(self, tdw, cr, x, y, node, save_context=False):
        ForegroundStamp.draw(self, tdw,  cr,
                x, y, node, save_context)

class ProxyStamp(ClipboardStamp):
    """ 
    A Proxy Stamp for dynamically changeable pixbuf stamps
    (Currently, it is only ClipboardStamp)
    to fix its contents within DrawStamp command.
    This stamp would only use inside program,have no any user
    direct interaction.

    Due to MyPaint 'Command' processing done in asynchronously,
    Clipboard stamp might be updated its content(pixbuf)
    when actually render_stamp_to_layer() called
    i.e. StampCommand has issued.

    Therefore, we need to pass the pixbuf used when editing
    to this stamp class, instead of ClipboardStamp.
    """

    def __init__(self, name, desc, surfs):
        super(ProxyStamp, self).__init__(name, desc)
        self._surfs = surfs


## Preset Manager classes

class StampPresetManager(object):
    """ Stamp preset manager.
    This class is singleton, owned by Application class 
    as application.stamp_preset_manager

    With this manager class , we can generates Stamp instance
    from its name. 
    """

    # Class constants
    STAMP_DIR_NAME = u'stamps' # The name of directory 
                               # where user defined Stamps are stored. 
                               # it is under the app.user_data dir.

    def __init__(self, app):
        self._app = app

        # XXX mostly copied from gui/application.py _init_icons()
        icon_theme = Gtk.IconTheme.get_default()
        if app:
            icon_theme.append_search_path(app.state_dirs.app_icons)

        try:
            self._default_icon = icon_theme.load_icon('mypaint', 32, 0)
        except GLib.Error:
            self._default_icon = icon_theme.load_icon('gtk-paste', 32, 0)

        stamplist = []

        for cs in BUILT_IN_STAMPS:
            stamp = self.create_stamp_from_json(cs)
            stamplist.append(stamp)

        for cf in glob.glob(self._get_adjusted_path("*.mys")):
            stamp = self.load_from_file(cf)
            stamplist.append(stamp)

        self.stamps = stamplist
        self._stamp_store = {}
        self._current = None

    @property
    def current(self):
        return self._current
       #if self._current_index is not None:
       #    return self_stamps[self._current_index]

    def set_current_iter(self, iter):
        self._current = self._stamp_store[iter]
        return self._current

    def set_current_index(self, idx):
        self._current = self.stamps[idx]
        return self._current


    def _get_adjusted_path(self, filepath):
        """ Return absolute path according to application setting.
        
        This method does not check the modified path exists.
        However, glob.glob() will accept nonexistant path,
        glob.glob() returns empty list for such one.
        """
        if not os.path.isabs(filepath) and self._app is not None:
            return os.path.join(self._app.state_dirs.user_data,
                        self.STAMP_DIR_NAME, filepath)
        else:
            return filepath


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

    ## Utility Methods.

    def initialize_icon_store(self):
        """ Initialize iconview store which is used in
        stamptool's OptionPresenter.
        i.e. This method is actually application-unique 
        stamp preset initialize handler.
        """
        liststore = Gtk.ListStore(GdkPixbuf.Pixbuf, str, object)

        for stamp in self._stamps:
            iter = liststore.append([stamp.thumbnail, stamp.name, stamp])
            self._stamp_store[iter] = stamp

        return liststore

    def save_to_file(self, stamp, filename):
        """ To save (or export) a stamp to file.
        """
        pass

    def get_stamp_type(self, target):
        """ Get stamp type as string, for json file."""
        if isinstance(Stamp, target):
            return "file"
        elif isinstance(TiledStamp, target):
            return "tiled-file"
        elif isinstance(LayerStamp, target):
            return "layer"
        elif isinstance(ClipboardStamp, target):
            return "clipboard"
        elif isinstance(ForegroundStamp, target):
            return "foreground"
        elif isinstance(VisibleStamp, target):
            return "current-visible"
        elif isinstance(ForegroundLayerStamp, target):
            return "foreground-layermask"
        else:
            raise TypeError("undefined stamp type")

    def create_stamp_from_json(self, jo):
        """ Create a stamp from json-generated dictionary object.
        :param jo: dictionary object, mostly created with json.load()/loads()

        :return: Stump class instance.

        ## The specification of mypaint stamp preset file(.mys)

        Stamp preset .mys file is a json file, 
        which has attributes below:

            "comment" : comment of preset
            "name" : name of preset
            "settings" : a dictionary to contain 'Stamp setting's.
            "thumbnail" : a thumbnail .jpg/.png filename
            "version" : 1
            "desc" : (Optional) description of this stamp,for iconview item.
                     this is also used for tooltip message.

        Stamp setting:
            "source" : "file" - Stamp from files.
                       "tiled-file" - Stamp from a file, divided with tile setting.
                       "clipboard" - Use run-time clipboard image for stamp.
                       "layer" - Stamp from layer image of user defined area.
                       "current-visible"  - Stamp from currently visible image
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

            "icon" : load a picture file as Gtk predefined icon for the stamp.
                     when this is not absolute path, 
                     stampmanager assumes it would be placed at 
                     self.STAMP_DIR_NAME below of _app.state_dirs.user_data

            "gtk-thumbnail" : Use thumbnail as Gtk predefined icon for the stamp.

        """

        if jo['version'] == "1":
            settings = jo['settings']
            name = jo.get('name', 'unnamed stamp')
            desc = jo.get('desc', None)
            source = settings['source']

            if source == 'file':
                assert 'filenames' in settings
                stamp = Stamp(name, desc)
                stamp.set_file_sources(settings['filenames'])
            elif source == 'tiled-file':
                assert 'filenames' in settings
                assert 'tile' in settings
                stamp = TiledStamp(name, desc, 
                        *settings.get('tile', (1, 1)))
                stamp.set_file_sources(settings['filenames'])
            elif source == 'clipboard':
                stamp = ClipboardStamp(name, desc)
            elif source == 'layer':
                stamp = LayerStamp(name, desc)
            elif source == 'current-visible':
                stamp = VisibleStamp(name, desc)
            elif source == 'foreground':
                stamp = ForegroundStamp(name, desc)
                assert 'mask' in settings
                stamp.set_file_sources(settings['mask'])
            elif source == 'foreground-layermask':
                stamp = ForegroundLayerStamp(name, desc)
            else:
                raise NotImplementedError("Unknown source %r" % source)

            # common setting
            if 'scale' in settings:
                stamp.set_default_scale(*settings['scale'])

            if 'angle' in settings:
                stamp.set_default_angle(math.radians(settings['angle'] % 360.0))

            if 'icon' in settings:
                try:
                    pixbuf = self.load_thumbnail(settings['icon'])
                    stamp.set_thumbnail(pixbuf)
                except:
                    logger.error('stamp cannot load icon filename %s' % 
                            icon_fname)

            elif 'gtk-thumbnail' in settings:
                try:
                    pixbuf = Gtk.IconTheme.get_default().load_icon(
                            settings['gtk-thumbnail'], Stamp.THUMBNAIL_SIZE, 0)
                    stamp.set_thumbnail(pixbuf)
                except:
                    logger.error('stamp cannot set gtk icon %s' % 
                            settings['gtk-thumbnail'])
           #else:
           #    stamp.set_thumbnail(
           #            self.load_thumbnail(settings.get('thumbnail', None)))


            return stamp

        else:
            raise NotImplementedError("Unknown version %r" % jo['version'])


    def save_to_file(self, filename, json_src):
        """ Save a json object to 'user-data' path.
        (But if the filename parameter is absolute path,
         use that path, not the user-data path.)
        """
        filename = self._get_adjusted_path(filename)
        with open(filename, 'w') as ofp:
            json.dump(json_src, ofp)

    ## Notification callbacks

    def stamp_deleted_cb(self, stamp):
        """ a stamp deleted from GUI operation.
        """
        if stamp in self.stamps:
            self.stamps.remove(stamp)
            for key, val in  self._stamp_store.iteritems():
                if val == stamp:
                    del self._stamp_store[key]
                    break
        pass

## Built-in stamps
#  These stamps are built-in, automatically registered at OptionPresenter.

BUILT_IN_STAMPS = [
            { "version" : "1",
              "name" : "clipboard stamp",
              "settings" : {
                  "source" : "clipboard",
                  "gtk-thumbnail" : "gtk-paste"
                  },
              "desc" : _("Stamp of Clipboard Image")
            },
            { "version" : "1",
              "name" : "layer stamp",
              "settings" : {
                  "source" : "layer"
                  },
              "desc" : _("Stamp of area of Layer.")
            },
            { "version" : "1",
              "name" : "visible stamp",
              "settings" : {
                  "source" : "current-visible"
                  },
              "desc" : _("Stamp of area of current visible.")
            },
            { "version" : "1",
              "name" : "layer mask",
              "settings" : {
                  "source" : "foreground-layermask"
                  },
              "desc" : _("Stamp of forground color, masked with area of Layer.")
            },
        ]
              
            



if __name__ == '__main__':
    pass



