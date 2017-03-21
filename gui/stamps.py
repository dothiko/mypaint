#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2016 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

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
import collections
import weakref
import sys

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
import lib.pixbuf

## Constant defs

# The file filter used for opening file dialogs.
STAMP_PRESET_FILE_FILTER = [
    (_("Mypaint stamp presets"), ("*.mys",)),
    ]

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

   #stamp.initialize_draw(cr)
    for cn in nodes:
        stamp.draw(None, cr, cn.x - sx, cn.y - sy,
                cn, True)
   #stamp.finalize_draw(cr)

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



def load_clipboard_image(): 
    clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)

    targs_avail, targets = clipboard.wait_for_targets()
    if not targs_avail:
        # Nothing on clipboard."
        return

    # Then grab any available image 
    # if there is no image, return None.
    return clipboard.wait_for_image()

## HOW TO GET layer portion as PIXBUF:

# we could get layer portion as pixbuf with calling
# layer.render_as_pixbuf(x, y, width, height, alpha=True)

# also we could get the current visible canvas with
# layer.root.render_as_pixbuf(x, y, width, height, alpha=True)



## Class defs

class PictureSource:
    """Enumeration of the stamp source"""
    FILE = 0       # File backed
    TILED_FILE = 1 # File backed,but it is devided from a picture.
    CAPTURED = 2   # Captured from Layer or clipboard

    ICON_FILE = 10
    ICON_GENERATED = 11
    ICON_STOCK = 12

class Stamp(object):
    """
    This class is standard Stamp class.
    This would have multiple pixbufs as tile,which from multiple picture files.
 
    When 'finalize' action executed (i.e. ACCEPT button pressed)
    the cairo surface converted into GdkPixbuf,
    and that GdkPixbuf converted into Mypaint layer
    and it is merged into current layer.
    """
    THUMBNAIL_SIZE = 32
    PICTURE_ICON_SIZE = 40
    
    DEFAULT_CAPTURED_SOURCE = (PictureSource.CAPTURED, None)
    DEFAULT_ICON = None

    def __init__(self, manager, name, desc):

        # Some important attributes initialized at clear_sources()
        self.clear_sources()

        self._picture_index_base = -1
        self.name = name
        self.desc = desc
        self._default_scale_x = 1.0
        self._default_scale_y = 1.0
        self._default_angle = 0.0
        self._thumbnail = None
        self._thumbnail_type = None
        self._picture_index_seed = 0
        self._use_mask = False
        self._source_info = None
        self._manager = weakref.proxy(manager)

    ## Information methods / properties

    @property
    def filename(self):
        return self._filename

    @property
    def dirty(self):
        return self._dirty

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


    ## Thumbnail related
    def set_thumbnail(self, pixbuf, 
            thumbnail_type=PictureSource.ICON_FILE,
            icon_name=""):
        self._thumbnail = pixbuf
        self._thumbnail_type = thumbnail_type
        self._thumbnail_source = icon_name

    def set_file_thumbnail(self, filename):
        assert filename != None
        if os.path.exists(filename):
            icon_size = Stamp.THUMBNAIL_SIZE
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_size(
                        filename,
                        icon_size, icon_size)
            self.set_thumbnail(pixbuf,
                    PictureSource.ICON_FILE,
                    filename)
        else:
            logger.warning('stamp preset icon %s does not found' % filename)

    def set_gtk_thumbnail(self, icon_name):
        pixbuf = Gtk.IconTheme.get_default().load_icon(
                icon_name, Stamp.THUMBNAIL_SIZE, 0)
        self.set_thumbnail(pixbuf, PictureSource.ICON_STOCK, icon_name)

    def generate_thumbnail(self, id=-1):
        """Generate thumbnail from contained tiles.

        :param id: The generating source tile id. If this is -1, the
                   first tile should be used.
        """
        if len(self._surfs) > 0:
            if id == -1:
                # id -1 means 'The last picture(surface)'
                # self._surfs is not dict but OrderedDict, 
                # so the last picture is always values()[-1]
                surf = self._surfs.values()[-1]
            else:
                surf = self._surfs[id]
            pixbuf = Gdk.pixbuf_get_from_surface(surf, 0, 0, 
                    surf.get_width(), surf.get_height())
            icon = Stamp._create_icon(pixbuf, Stamp.THUMBNAIL_SIZE)
            self.set_thumbnail(icon,
                    PictureSource.ICON_GENERATED,
                    "")
            self._dirty = True
        else:
            self._thumbnail = None # assigning None means
                                   # 'Use class default icon'


    @property
    def thumbnail(self):
        if self._thumbnail:
            return self._thumbnail
        else:
            if Stamp.DEFAULT_ICON == None:
                # XXX mostly copied from gui/application.py _init_icons()
                icon_theme = Gtk.IconTheme.get_default()
                if self._manager._app:
                    icon_theme.append_search_path(
                            self._manager._app.state_dirs.app_icons)

                try:
                    Stamp.DEFAULT_ICON = icon_theme.load_icon('mypaint', 
                            Stamp.THUMBNAIL_SIZE, 0)
                except GLib.Error:
                    Stamp.DEFAULT_ICON = icon_theme.load_icon('gtk-paste', 
                            Stamp.THUMBNAIL_SIZE, 0)

            return Stamp.DEFAULT_ICON



    ## Mask related
    @property
    def use_mask(self):
        return self._use_mask

    @use_mask.setter
    def use_mask(self, flag):
        self._use_mask = flag

    def validate_picture_id(self, tid):
        """Validate the tid argument as tile index.
        If it does not found in self._surf,
        return the last(in the key of _surf dictionary) tile index.

        Mostly convert id '-1' as the last tile id.
        """
        if not tid in self._surfs:
            assert len(self._surfs) > 0
            return self._surfs.keys()[-1]
        else:
            return tid

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

    def draw(self, tdw, cr, x, y, node, save_context=False):
        """ Draw this stamp into cairo surface.
        This implementation is as base class,
        node.picture_index ignored here.

        :param tdw: The tiledraw widget. 
                    CAUTION: when drawing target is off-canvas, 
                    this is None.
                    
        """
        stamp_src = self.get_current_src(node.picture_index)
        if stamp_src:
            if save_context:
                cr.save()

            w = stamp_src.get_width() 
            h = stamp_src.get_height()

            ox = math.ceil(-(w / 2))
            oy = math.ceil(-(h / 2))

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

            if self._use_mask:
                # Use stamp pixbuf as mask
                fg_col = self._manager._app.brush_color_manager.get_color().get_rgb()
                cr.set_source_rgb(*fg_col)
                cr.rectangle(ox, oy, w, h) 
                cr.clip()
                cr.fill()
                cr.mask_surface(stamp_src, ox, oy)
            else:
                cr.set_source_surface(stamp_src, ox, oy)
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
                Normally it should be either one of targetted handle index 
                within 0 to 3, or None (cursor is not on this node)
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

        return None

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
        stamp_src = self.get_current_src(node.picture_index)
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
            area = self._get_outmost_area_from_points(pos)
            sx, sy, ex, ey = get_outmost_area(tdw, 
                    *area,
                    margin=margin)
            return (sx, sy, abs(ex-sx)+1, abs(ey-sy)+1)
            
    @staticmethod
    def _get_outmost_area_from_points(pts):
        sx, sy = pts[0]
        ex, ey = sx, sy
        for tx, ty in pts[1:]:
            sx = min(tx, sx)
            sy = min(ty, sy)
            ex = max(tx, ex)
            ey = max(ty, ey)
        return (sx, sy, ex, ey)

    ## Phase related methods
    #  these methods should be called from 
    #  Stamptool._start_new_capture_phase()

    def initialize_phase(self, mode):
        """ Initializing for start of each drawing phase.
        i.e. called from StampMode._start_new_capture_phase
        """
        pass

    def finalize_phase(self, mode, rollback):
        """ This called when stamptool comes to end of
        capture(drawing) phase.

        :param mode: the StampMode instance.
        :param rollback: True when rollback(cancel) committed. 
        """
        pass

    ## Enter/Leave callbacks
    #  These callbacks called when stamptool.stamp attribute
    #  has been changed.

    def enter(self, doc):
        """
        Called when stamp has selected.
        """
        self.ensure_sources()

    def leave(self, doc):
        """
        Called when stamp has unselected.
        """
        pass
    


    ## Source information related method

    @property
    def picture_count(self):
        return len(self._surfs)

    def get_current_src(self, tid):
        if tid in self._surfs:
            return self._surfs[tid]

    def fetch_source_info(self, info, blacklists):
        """ fetch source info, to prepare later loading.
    
        This is used for stamp preset, to save memory
        and speed up startup time
        until stamps are actually used.
        """
        self._source_info = info
        self._blacklists = blacklists

    def _decode_file_source(self, info):
        """Decode a late-binding information
        and load a stamp picture into memory.

        To speed up Mypaint startup, Stamp class do nothing
        except for caching source information at statup.
        and when the stamp is actually used by end-user,
        Stamp class loads stamp picture with the method
        'ensure_sources()', which calls this method internally.

        :param info: string of filename, or tuple of 
            (filename, tile width, tile height) 
        """
        if isinstance(info, unicode):
            self.add_file_source(info)
        elif isinstance(info, tuple):
            try:
                fname, tile_h, tile_v = info
                self.add_file_sources_as_tile(fname, 
                        tile_h, tile_v,
                        self._blacklists.get(fname, None))
            except ValueError as e:
                logger.error("an error raised at loading tile stamps")
                sys.stderr.write(str(e))
        else:
            logger.warning("Unknown picture source assigned.")
            sys.stderr.write(str(info))
            

    def set_surface_from_pixbuf(self, id, pixbuf, 
            source=None):
        """ Set (or add) surface from pixbuf.
        This method actually set the dictionary of 
        surfaces, so assigning non existent id
        to this, works as 'add' method, not 'set'.

        :param id: stamp picture id, if this is -1,
                   this works 'add', not 'set'.
        :param pixbuf: pixbuf to add
        :param source: a tuple of source information,
            (_PixtureSource enumeration, optional information)
        :return : return the id of stamp picture.if id
        """
        assert isinstance(source, tuple)

        if id == -1:
            id = self._generate_picture_index()
        surf = Gdk.cairo_surface_create_from_pixbuf(
                pixbuf, 1, None)
        self._surfs[id] = surf
        self._create_picture_icon(id, pixbuf)
        self._picture_sources[id] = source
        self._dirty = True
        return id

    def add_file_source(self, filename):
        pixbuf = GdkPixbuf.Pixbuf.new_from_file(filename)
        id = self.set_surface_from_pixbuf(-1, pixbuf, 
                (PictureSource.FILE, filename))
        del pixbuf
        return id

    
    def add_file_sources_as_tile(self, filename, tile_w, tile_h, blacklist=None):
        pixbuf = GdkPixbuf.Pixbuf.new_from_file(filename)
        if tile_w > 0 and tile_h > 0:
            self._tileinfo[filename] = (tile_w, tile_h)
            idx = 0
            # XXX tile_w is tile width in pixel? or divide count of tile??
            # if divide count of tile, this code is wrong...?
            for y in xrange(pixbuf.get_height() / tile_h):
                for x in xrange(pixbuf.get_width() / tile_w):
                    if blacklist == None or not idx in blacklist:
                        tpb = pixbuf.new_subpixbuf(x, y, 
                                tile_w, tile_h)
                        ox, oy = _SourceMixin.get_offsets(tpb)
                        self.set_surface_from_pixbuf(-1, tpb,
                                (PictureSource.TILED_FILE, (filename, idx)))
                    idx+=1
                    del tpb 
            del pixbuf


    def add_pixbuf_source(self, pixbuf, name):
        """ Set stamp picture from a pixbuf.
        This method used when a tile is imported
        from Clipboard, or region of a layer.

        :param name: the name of source layer.
                     If captured from clipboard, this MUST be None.
        """ 
        idx = self._generate_picture_index()
        self.set_surface_from_pixbuf(idx, pixbuf,
                (PictureSource.CAPTURED, name))

    def remove(self, id):
        """ remove a tile from stamp.
        """
        if id in self._surfs:
            del self._surfs[id]
            assert id in self._picture_icons
            del self._picture_icons[id]

            pictype, info = self._picture_sources[id]
            if pictype == PictureSource.TILED_FILE:
                # To detect 'tile-based stamp picture
                # deleted', leave source information,
                # but change index as None. 
                self._picture_sources[id] = (pictype, (info[0], None))
            else:
                del self._picture_sources[id]
                # [TODO] delete file resources actually, if exists.

            self._dirty = True
        else:
            logger.warning('there is no such id %d in stamp' % id)


    def clear_sources(self):
        """ clear sources (but _picture_index_base uncleared)
        """
        self._surfs = collections.OrderedDict()
        self._picture_icons = {}
        self._picture_sources = {}
        self._tileinfo = {}
        self._blacklists = None
        self._dirty = False
        self._filename = ''

    def source_surface_iter(self):
        for id in self._surfs:
            yield (id, self._surfs[id])

    def get_surface(self, id):
        return self._surfs.get(id, None)

    @staticmethod
    def _create_icon(pixbuf, icon_size):
        """ To share icon generating code. 
        """
        pw = pixbuf.get_width()
        ph = pixbuf.get_height()
        ratio = float(icon_size) / ph
        
        icon = GdkPixbuf.Pixbuf.new(
                GdkPixbuf.Colorspace.RGB,
                pixbuf.get_has_alpha(),
                8,
                icon_size, icon_size)
        pixbuf.scale(icon, 
                0, 0,  # dest x, y
                icon_size, icon_size,
                0, 0,  # Offset x, y
                ratio, ratio,
                GdkPixbuf.InterpType.BILINEAR)

        return icon

    def _create_picture_icon(self, id, pixbuf):
        """Create tile icon, 
        for iconview of stamp tiles Options presenter.
        """
        self._picture_icons[id] = Stamp._create_icon(pixbuf, 
                self.PICTURE_ICON_SIZE)

    def picture_icon_iter(self):
        for id in self._picture_icons:
            yield (id, self._picture_icons[id])

    def get_icon(self, id):
        return self._picture_icons.get(id, None)


    @property
    def latest_picture_index(self):
       #return len(self._source_files) - 1
        return self._picture_index_base

    def _generate_picture_index(self):
        """ generate unique tile index

        Currently, just add 1 for each index.
        """
        self._picture_index_base += 1
        return self._picture_index_base 

    def ensure_sources(self):
        """ Late-binding of source pictures.
        (this is default behavior of stamp presets)
        """
        info = self._source_info
        if info:
            # dirty flag should not be changed by this method.
            # this method only 'set stamp to initial(ready) state'
            saved_flag = self._dirty
            for cf in info:
                self._decode_file_source(cf)
            self._dirty = saved_flag

            # Clear source information, because all stamp pictures already loaded.
            self._source_info = None

    ## Source Pixbuf methods

    def clear_all_cache(self):
        """
        Clear all cached pixbuf
        """
        self._surfs.clear()


    ## Serialize methods

    def save_to_file(self, jsonfilename):
        assert os.path.isabs(jsonfilename)
        basedir, basename = os.path.split(jsonfilename)
        basejsonname = os.path.splitext(basename)[0]

        jsondic = {}
        jsondic['version'] = "1" # Version number must be string.
        jsondic['name'] = self.name
        jsondic['desc'] = self.desc

        if self._thumbnail is not None:
            if self._thumbnail_type == PictureSource.ICON_GENERATED:
                # Save Icon picture, if it is generated one.
                basename = "%s_thumbnail.jpg" % self.name
                filename = os.path.join(basedir, basename) 
                lib.pixbuf.save(self._thumbnail, filename, 'jpeg')
                jsondic['thumbnail'] = basename
            elif self._thumbnail_type == PictureSource.ICON_STOCK:
                assert hasattr(self, '_thumbnail_source')
                jsondic['gtk-thumbnail'] = self._thumbnail_source
            elif self._thumbnail_type == PictureSource.ICON_FILE:
                assert hasattr(self, '_thumbnail_source')
                jsondic['thumbnail'] = self._thumbnail_source


        settings = {}

        # When a stamp saved to file,
        # every stamp should turn into file-backed stamp.
        jsondic['type'] = 'file'

        # Enumerate id from self._surfs, because it is ordereddict,
        # it can reproduce same order in same stamp.
        filenames = []
        tile_blacklists = {}
        for id in self._surfs:
            source_type, info = self._picture_sources[id]
            if source_type == PictureSource.FILE:
                filenames.append(info)
            elif source_type == PictureSource.TILED_FILE:
                sourcename, tileidx = info
                if not sourcename in filenames:
                    filenames.append(sourcename)

                    # Do not forget to add settings
                    # the tiled width and height.
                    assert sourcename in self._tileinfo
                    tileinfo = self._tileinfo[sourcename]
                    filenames.append( (sourcename, tileinfo[0], tileinfo[1]) )

                if tileidx == None:
                    # This picture is a deleted one.
                    blacklist = tile_blacklists.get(sourcename, [])
                    blacklist.append(tileidx)
                    tile_blacklists[sourcename] = blacklist

            elif source_type == PictureSource.CAPTURED:
                surf = self._surfs[id]
                pixbuf = Gdk.pixbuf_get_from_surface(surf, 0, 0, 
                        surf.get_width(), surf.get_height())
                picfilename = os.path.join(basedir, 
                        "%s_stamp_%d.png" % (basejsonname, id))
                lib.pixbuf.save(pixbuf, picfilename, 'png')
                filenames.append(picfilename)
               #self._picture_sources[id] = (PictureSource.FILE, filename)

        settings['filenames'] = filenames
        if len(tile_blacklists):
            settings['tile-blacklists'] = tile_blacklists

        if self.use_mask:
            settings['mask'] = True

        if self._default_scale_x != 1.0:
            settings['scale'] = self._default_scale_x

        if self._default_angle != 0.0:
            settings['angle'] = math.degrees(self._default_angle)

        jsondic['settings'] = settings


        with open(jsonfilename, 'w') as ofp:
            json.dump(jsondic, ofp)

        self._dirty = False
        self._filename = jsonfilename

    @staticmethod
    def load_from_file(filename, manager):
        with open(filename, 'r') as ifp:
            jo = json.load(ifp)
        stamp = Stamp.create_stamp_from_json(jo, manager)
        stamp._filename = filename
        return stamp

    @staticmethod
    def create_stamp_from_json(jo, manager):
        """Load a stamp from json-generated dictionary object.
        THIS IS A STATIC METHOD, no instance needed.

        :param jo: dictionary object, mostly created with json.load()/loads()

        :return: Stump class instance.

        ## The specification of mypaint stamp preset file(.mys)

        Stamp preset .mys file is a json file, 
        which has attributes below:

            "name" : name of preset
            "settings" : a dictionary to contain 'Stamp setting's.
            "thumbnail" : a thumbnail .jpg/.png filename
            "version" : a string, indicates stamp version. currently it is "1"
            "desc" : the description of this stamp,for iconview item.
                     this is also used for tooltip message.
                     this can be empty string.

        Stamp setting:
            "type" : Initial source of the stamp
                "file" - Stamp from files.
                "clipboard" - Use run-time clipboard image for stamp.
                "empty" - newly created empty stamp.

            "filenames" : LIST of .jpg/png filepaths of stamp source.
                         An element of this list can be a tuple, if so,
                         it means "The picture is divided into tiles"
                         so such tuple MUST be
                         (filename, horizontal_divide_count,
                             vertical_divide_count)

            "scale" : A tuple of default scaling ratio, 
                      (horizontal_ratio, vertical_ratio)

            "angle" : A floating value of default angle,in DEGREEs.

            "mask" : A boolean flag to use stamp's alpha pixel as mask
                     for foreground color rectangle.

            *** The below 'tile-*' settings are optional. ***

            "tile-blacklists" : dictionary of filename, and it has list of index,
                               which are 'deleted' - i.e. to be ignored tiles.
                     
            "tile-type" : "random" - The next tile index is random value.
                          "increment" - The next tile index is automatically incremented.
                          "same" - The next tile index is always default index.
                                   user will change it manually every time
                                   he want to use the other one.
                                   This is default tile-type.

            "tile-default-index" : The default index of tile. by default, it is 0.

            "gtk-thumbnail" : Use thumbnail as Gtk predefined icon for the stamp.

        """

        if jo['version'] == "1":
            # Decode basic informations.
            name = jo.get('name', 'unnamed stamp')
            desc = jo.get('desc', '')
            stamp_type = jo.get('type', None)

            if stamp_type in ('file', 'empty'):
                stamp = Stamp(manager, name, desc)
            elif stamp_type == 'clipboard':
                stamp = ClipboardStamp(manager, name, desc)
            else:
                logger.warning("Unknown stamp type %s" % stamp_type)

            if 'thumbnail' in jo:
                try:
                    thumbfile = manager.get_adjusted_path(jo['thumbnail'])
                    stamp.set_file_thumbnail(thumbfile)
                except:
                    logger.error('stamp cannot load icon filename %s' % 
                            icon_fname)
            elif 'gtk-thumbnail' in jo:
                try:
                    stamp.set_gtk_thumbnail(jo['gtk-thumbnail'])
                except:
                    logger.error('stamp cannot set gtk icon %s' % 
                            jo['gtk-thumbnail'])
            else:
                stamp.generate_thumbnail()

            # Decode setting parts.
            settings = jo.get('settings', None)

            if settings:
                if stamp_type == 'file':
                    stamp.fetch_source_info(
                            settings.get('filenames', None),
                            settings.get('tile-blacklists', None)
                            )
                elif stamp_type == 'clipboard':
                    pass
                else:
                    # It would be empty tile.
                    pass 

                if 'scale' in settings:
                    stamp.set_default_scale(*settings['scale'])

                if 'angle' in settings:
                    stamp.set_default_angle(math.radians(settings['angle'] % 360.0))

                if 'mask' in settings and settings['mask'] in (1, True, "1", "True"):
                    stamp.use_mask = True


            stamp._dirty = False
            return stamp

        else:
            raise NotImplementedError("Unknown version %r" % jo['version'])


class ClipboardStamp(Stamp):
    """The derived class to specialize to deal with Clipboard.
    """

    def __init__(self, manager, name, desc):
        super(ClipboardStamp, self).__init__(manager, name, desc)

    def initialize_phase(self, mode):
        """ Initializing for start of each drawing phase.
        i.e. called from StampMode._start_new_capture_phase
        """
        self.ensure_sources()

    def ensure_sources(self):
        """ Late-binding of source pictures.
        (this is default behavior of stamp presets)
        """
        if len(self._surfs) == 0:
            pixbuf = load_clipboard_image()
            if pixbuf:
                self.add_pixbuf_source(pixbuf,'Clipboard')


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


        stamplist = {}
        id = 0
        try:

            for cs in BUILT_IN_STAMPS:
                stamp = Stamp.create_stamp_from_json(cs, self)
                stamplist[id] = stamp
                id+=1

            for cf in glob.glob(self.get_adjusted_path("*.mys")):
                stamp = Stamp.load_from_file(cf, self)
                stamplist[id] = stamp
                id+=1

        except Exception as e:
            import sys
            logger.error("an error raised at creating/loading initial stamps")
            sys.stderr.write(str(e)+'\n')

        self._id_base = id
        self.stamps = stamplist
        self._stamp_store = {}
        self._current = None

    def get_current(self):
        return self._current

   #def set_current_iter(self, iter):
   #    self._current = self._stamp_store[iter]
   #    return self._current

    def set_current(self, new_current):
        self._current = new_current
        return self._current


    def get_adjusted_path(self, filepath):
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
            filepath = self.get_adjusted_path(name)
            if os.path.exists(filepath):
                icon_size = Stamp.THUMBNAIL_SIZE
                return  GdkPixbuf.Pixbuf.new_from_file_at_size(
                            self.get_adjusted_path(name),
                            icon_size, icon_size)

        assert self._default_icon != None
        return self._default_icon
        
    def load_from_file(self, filename):
        """Load a preset which saved as json file, just like as brushes.
        :rtype: Stump class instance.
        """
        junk, ext = os.path.splitext(filename) 
        assert ext.lower() == '.mys'

        filename = self.get_adjusted_path(filename)
        return Stamp.load_from_file(filename, self)


   #def save_to_file(self, stamp):
   #    """Save a stamp as a json object file to 'user-data' path.
   #    This is a utility method, stamp can save itself easily
   #    but this would be handy.
   #
   #    If the filename parameter is absolute path,
   #    use that path, not the user-data path.
   #    """
   #    junk, ext = os.path.splitext(filename) 
   #    assert ext.lower() == '.mys'
   #    filename = self.get_adjusted_path(filename)
   #    stamp.save_to_file(filename)


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
              "gtk-thumbnail" : "gtk-paste",
              "type" : "clipboard",
              "settings" : {
                  },
              "desc" : _("Stamp of Clipboard Image")
            },
            { "version" : "1",
              "name" : "new stamp",
              "gtk-thumbnail" : "gtk-paste",
              "type" : "empty",
              "settings" : {
                  },
              "desc" : _("New empty stamp")
            },
            { "version" : "1",
              "name" : "new colored stamp",
              "gtk-thumbnail" : "gtk-paste",
              "type" : "empty",
              "settings" : {
                  "mask" : "1",
                  },
              "desc" : _("new colored stamp")
            },
        ]
              
            



if __name__ == '__main__':
    pass



