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

## Source Mixins
#
#  These Source Mixins provide some functionality like
#  'surface pool' or 'cache management' for Stamp classes.
#
#  INITIALIZE:
#  Source mixin has _init_source() initializer,
#  this must be called from __init__() of Stamp class
#  as well as _init_stamp(name) of Stamp mixin.

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
    def __init__(self, name, desc):

        self.clear_sources()
        self._tile_index_base = 0
        self.name = name
        self.desc = desc
        self._default_scale_x = 1.0
        self._default_scale_y = 1.0
        self._default_angle = 0.0
        self._thumbnail = None
        self._tile_index_seed = 0
        self._use_mask = False
        self._source_info = None

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


   #@property
   #def is_support_selection(self):
   #    return False

    @property
    def is_ready(self):
        return True

    @property
    def latest_tile_index(self):
        return self.tile_count - 1

    @property
    def use_mask(self):
        return self._use_mask

    @use_mask.setter
    def use_mask(self, flag):
        self._use_mask = flag

    ## Deprecated property

    @property
    def is_support_selection(self):
        return True


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

   #def initialize_draw(self, cr):
   #    """ Initialize draw calls.
   #    Call this method prior to draw stamps loop.
   #    """
   #    pass
   #
   #def finalize_draw(self, cr):
   #    """ Finalize draw calls.
   #    Call this method after draw stamps loop.
   #
   #    This method called from the end of each drawing sequence,
   #    NOT END OF DRAWING PHASE OF STAMPTOOL! 
   #    Therefore, source.finalize() MUST not be called here!
   #    """
   #    pass
        
    

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

            if self._use_mask:
                # Use stamp pixbuf as mask
                fg_col = tdw.app.brush_color_manager.get_color().get_rgb()
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
            if tx < sx:
                sx = tx
            elif tx > ex:
                ex = tx

            if ty < sy:
                sy = ty
            elif ty > ey:
                ey = ty
        return (sx, sy, ex, ey)


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
        self.ensure_sources()

    def leave(self, doc):
        """
        Called when stamp has unselected.
        """
        pass
    


    ## Source information related method

    @property
    def tile_count(self):
        return len(self._surfs)

    def get_current_src(self, tid):
       #if self._ensure_current_pixbuf(tile_index):
       #    return self._surfs[tile_index]
        if tid in self._surfs:
            return self._surfs[tid]

    def fetch_source_info(self, source_type, info):
        """ fetch source info, to prepare later loading.

        This is used for stamp preset, to save memory
        and speed up startup time
        until stamp tool is actually used.
        """
        self._source_info = (source_type, info)



   #def set_file_sources(self, filenames):
   #    if type(filenames) == str:
   #        self._source_files[0] = filenames
   #    else:
   #        for i, filename in enumerate(filenames):
   #            self._source_files[i] = filename
   #        self._tile_index_base = i

    def set_surface_from_pixbuf(self, id, pixbuf):
        if id == -1:
            id = self._generate_tile_index()
        surf = Gdk.cairo_surface_create_from_pixbuf(
                pixbuf, 1, None)
        self._surfs[id] = surf

    def add_file_source(self, filename):
        tile_index = self._generate_tile_index()
        junk, ext = os.path.splitext(filename)
        if ext.lower() == '.png':
            surf = cairo.ImageSurface.create_from_png(filename)
        else:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file(filename)
            surf = Gdk.cairo_surface_create_from_pixbuf(
                    pixbuf, 1, None)
        self._surfs[tile_index] = surf
    
    def set_file_sources_as_tile(self, filename, tileinfo):
        pixbuf = GdkPixbuf.Pixbuf.new_from_file(filename)
        tile_w, tile_h = tileinfo
        if tile_w > 0 and tile_h > 0:
            idx = self._generate_tile_index()
            for y in xrange(pixbuf.get_height() / tile_h):
                for x in xrange(pixbuf.get_width() / tile_w):
                    tpb = pixbuf.new_subpixbuf(x, y, 
                            tile_w, tile_h)
                    ox, oy = _SourceMixin.get_offsets(tpb)
                    self.set_surface_from_pixbuf(idx, tpb)
                    del tpb 
                    idx+=1
            del pixbuf


    def set_pixbuf_sources(self, pixbufs):
        for cpb in pixbufs:
            if cpb:
                idx = self._generate_tile_index()
                self.set_surface_from_pixbuf(idx, cpb)

    def clear_sources(self):
        """ clear sources (but _tile_index_base uncleared)
        """
        self._surfs = {}

    @property
    def latest_tile_index(self):
       #return len(self._source_files) - 1
        return self._tile_index_base

    def _generate_tile_index(self):
        """ generate unique tile index

        Currently, just add 1 for each index.
        """
        self._tile_index_base += 1
        return self._tile_index_base 

    def ensure_sources(self):
        """ Late-binding of source pictures.
        (this is default behavior of stamp presets)
        """
        if self._source_info:
            s_type, info = self._source_info
            if s_type == 'file':
                for cf in info:
                    self.add_file_source(cf)
            elif s_type == 'tiled-file':
                self.set_file_sources_as_tile(*info)
            elif s_type == 'clipboard':
                self.clear_sources()
                self.set_pixbuf_sources(self, (load_clipboard_image(),))
                return # To avoid self._sourceinfo cleared.
            else:
                logger.error('unknown stamp source type %s' % s_type)
            
            self._source_info = None



    ## Source Pixbuf methods

   #def _ensure_current_pixbuf(self, tile_index):
   #    """
   #    To ensure current pixbuf (and cached surface) loaded. 
   #    Automatically called from get_current_src() method.
   #
   #    This facility is completely different at _Dynamic_Source.
   #    see _Dynamic_source.get_current_src()
   #    """
   #    if self._source_files and len(self._source_files) > 0:
   #        if not tile_index in self._surfs:
   #            filename = self._source_files[tile_index]
   #            junk, ext = os.path.splitext(filename)
   #            if ext.lower() == '.png':
   #                surf = cairo.ImageSurface.create_from_png(filename)
   #            else:
   #                pixbuf = GdkPixbuf.Pixbuf.new_from_file(filename)
   #                surf = Gdk.cairo_surface_create_from_pixbuf(
   #                        pixbuf, 1, None)
   #            self._surfs[tile_index] = surf
   #        return True
   #    return False


    def clear_all_cache(self):
        """
        Clear all cached pixbuf
        """
        self._surfs.clear()

   #def get_current_src(self, tile_index):
   #    stamp_src = self.get_current_src(tile_index)
   #    w = stamp_src.get_width() 
   #    h = stamp_src.get_height()
   #    return (stamp_src, -(w / 2), -(h / 2))

   #def get_desc(self, tile_index):
   #    return self._source_files[tile_index]

   #def validate_all_tiles(self):
   #    for ck in self._source_files.keys():
   #        self._ensure_current_pixbuf(ck)


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

        for stamp in self.stamps:
            iter = liststore.append([stamp.thumbnail, stamp.name, stamp])
            print(iter)
            self._stamp_store[iter] = stamp

        return liststore

    def save_to_file(self, stamp, filename):
        """ To save (or export) a stamp to file.
        """
        pass

   #def get_stamp_type(self, target):
   #    """ Get stamp type as string, for json file."""
   #    if isinstance(Stamp, target):
   #        return "file"
   #    elif isinstance(TiledStamp, target):
   #        return "tiled-file"
   #    elif isinstance(LayerStamp, target):
   #        return "layer"
   #    elif isinstance(ClipboardStamp, target):
   #        return "clipboard"
   #    elif isinstance(ForegroundStamp, target):
   #        return "foreground"
   #    elif isinstance(VisibleStamp, target):
   #        return "current-visible"
   #    elif isinstance(ForegroundLayerStamp, target):
   #        return "foreground-layermask"
   #    else:
   #        raise TypeError("undefined stamp type")

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
            "source" : Initial source of the stamp
                "file" - Stamp from files.
                "tiled-file" - Stamp from a file, divided with 'tile' setting.
                 "clipboard" - Use run-time clipboard image for stamp.

            "filename" : LIST of .jpg/png filepaths of stamp source.
                         Multiple filepaths mean 'this stamp has
                         tiles in separeted files'.

                         Otherwise, it will be a single picture stamp.

            "scale" : A tuple of default scaling ratio, 
                      (horizontal_ratio, vertical_ratio)

            "angle" : A floating value of default angle,in degree. 

            "mask" : A boolean flag to use stamp's alpha pixel as mask
                     for foreground color rectangle.

            "tile" : A tuple of (width, height),it represents tile size
                     of a picture. 
                     Currently this setting use with 'tiled-file' source only.
                     
            "tile-type" : "random" - The next tile index is random value.
                          "increment" - The next tile index is automatically incremented.
                          "same" - The next tile index is always default index.
                                   user will change it manually every time
                                   he want to use the other one.
                                   This is default tile-type.

            "tile-default-index" : The default index of tile. by default, it is 0.

                     Needless to say, 'tile-*' setting will ignore 
                     when there is no tile setting.

            "icon" : load a picture file as Gtk predefined icon for the stamp.
                     when this is not absolute path, 
                     stampmanager assumes it would be placed at 
                     self.STAMP_DIR_NAME below of _app.state_dirs.user_data

            "gtk-thumbnail" : Use thumbnail as Gtk predefined icon for the stamp.

        """
        try:

            if jo['version'] == "1":
                settings = jo['settings']
                name = jo.get('name', 'unnamed stamp')
                desc = jo.get('desc', None)
                source = settings.get('source', None)
                fnames = settings.get('filenames', None)

                stamp = Stamp(name, desc)

                if source == 'file' and fnames:
                    stamp.fetch_source_info(source, fnames)
                elif source == 'tiled-file' and fnames:
                    tile_info = settings.get('tile', (1, 1))
                    stamp.fetch_source_info(source,
                            (fnames, tile_info))
                elif source == 'clipboard':
                    stamp.fetch_source_info(source, None)
               #elif source == 'layer':
               #    stamp = LayerStamp(name, desc)
               #elif source == 'current-visible':
               #    stamp = VisibleStamp(name, desc)
               #elif source == 'foreground':
                   #stamp = ForegroundStamp(name, desc)
                   #assert 'mask' in settings
                   #stamp.set_file_sources(settings['mask'])
               #elif source == 'foreground-layermask':
               #    stamp = ForegroundLayerStamp(name, desc)
                else:
                    # It would be empty tile.
                    pass 

                # common setting
                if 'scale' in settings:
                    stamp.set_default_scale(*settings['scale'])

                if 'angle' in settings:
                    stamp.set_default_angle(math.radians(settings['angle'] % 360.0))
                if 'mask' in settings and settings['mask'] in (1, True, "1", "True"):
                    stamp.use_mask = True

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

        except Exception as e:
            import sys
            logger.error("an error raised at loading stamps")
            sys.stderr.write(str(e)+'\n')

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
              "name" : "new stamp",
              "settings" : {
                  "gtk-thumbnail" : "gtk-paste"
                  },
              "desc" : _("New empty stamp")
            },
            { "version" : "1",
              "name" : "new colored stamp",
              "settings" : {
                  "mask" : "1",
                  "gtk-thumbnail" : "gtk-paste"
                  },
              "desc" : _("new colored stamp")
            },
        ]
              
            



if __name__ == '__main__':
    pass



