# This file is part of MyPaint.
# Copyright (C) 2019 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import lib.mypaintlib
import lib.surface

import json

N = lib.mypaintlib.TILE_SIZE 

class _PIXEL:
    """Enumeration for pixel values.
    """
    AREA = lib.mypaintlib.Flagtile.PIXEL_AREA_VALUE
    FILLED = lib.mypaintlib.Flagtile.PIXEL_FILLED_VALUE
    CONTOUR = lib.mypaintlib.Flagtile.PIXEL_CONTOUR_VALUE
    EMPTY = lib.mypaintlib.Flagtile.PIXEL_EMPTY_VALUE
    OUTSIDE = lib.mypaintlib.Flagtile.PIXEL_OUTSIDE_VALUE
    OVERWRAP = lib.mypaintlib.Flagtile.PIXEL_OVERWRAP_VALUE

def _convert_result(layer, ft, color):
    """Draw filled result of flagtile surface into 
    mypaint-colortiles of the layer.
    """
    layer.autosave_dirty = True

    ox = ft.get_origin_x()
    oy = ft.get_origin_y()
    tw = ft.get_width()
    th = ft.get_height()
    if color is not None:
        assert type(color) is tuple
        r, g, b = color
    assert hasattr(layer, "_surface")
    dstsurf = layer._surface
    filled = {}
    for fty in range(th):
        for ftx in range(tw):
            ct = ft.get_tile(ftx, fty, False)
            if ct is not None:
                tx = ftx + ox
                ty = fty + oy
                with dstsurf.tile_request(tx, ty, readonly=False) \
                        as dst_tile:
                    if color is None:
                        ct.convert_to_transparent(
                            dst_tile,
                            _PIXEL.FILLED
                        )
                    else:
                        ct.convert_to_color(
                            dst_tile,
                            r, g, b
                        )
                    filled[(tx, ty)] = dst_tile

    # Actual color-tile bbox might be different from FlagtileSurface
    # Because FlagtileSurface has some reserved empty tiles for dilation.
    # So we should use `filled` dictionary.
    tile_bbox = lib.surface.get_tiles_bbox(filled)
    dstsurf.notify_observers(*tile_bbox)

def close_fill(targ, src, nodes, targ_pos, level, color,
               tolerance=0.1,
               alpha_threshold=0.2,
               dilation=2,
               fill_all_holes=True, 
               debug_info=None# XXX for Debug 
               ):
    """Do `Close and fill`, called from ClosefillMode of lib/command.py

    :param src: The source tiledsurface.
    :param nodes: Sequence of polygon nodes, which defines closed area.
    """
    early_culling_level = 2 # XXX fixed value, early culling for 4x4 pixel level

    # XXX Debug/profiling code
    import time
    ctm = time.time()
    # XXX Debug code end

    # Create flagtilesurface object
    ft = lib.mypaintlib.ClosefillSurface(level, nodes)

    ox = ft.get_origin_x()
    oy = ft.get_origin_y()
    tw = ft.get_width()
    th = ft.get_height()
    
    # XXX Debug code
    if debug_info:
        show_flag = debug_info.get('show_flag', False)
        tile_output = debug_info.get('tile_output', False)
        if tile_output:
            _dbg_tile_output(debug_info, ox, oy, tw, th, src)
    else:
        show_flag = False
    # XXX Debug code End
    
    # Get target color if needed
    targ_r = targ_g = targ_b = targ_a = 0
    if targ_pos is not None:
        px, py = targ_pos
        tx = px // N 
        ty = py // N 
        px = int(px % N)
        py = int(py % N)
        with src.tile_request(tx, ty, readonly=True) as sample:
            targ_r, targ_g, targ_b, targ_a = [int(c) for c in sample[py][px]]

    # Convert contours into flagtile.
    for fty in range(0, th):
        for ftx in range(0, tw):
            with src.tile_request(ftx+ox, fty+oy, readonly=True) as src_tile:
                ct = ft.get_tile(ftx, fty, False)
                if ct is not None:
                    ct.convert_from_color(
                        src_tile,
                        targ_r, targ_g, targ_b, targ_a,
                        tolerance,
                        alpha_threshold,
                        False
                    )

    # XXX Debug/profiling code
   #if show_flag:
   #    self._dbg_show_flag(ft, 'contour,filled',title='post-convert tile')
    #XXX debug code end

    # Build pyramid
    ft.build_progress_seed()

    # Deciding outside of filling area
    ft.decide_outside(level)
      
    # Then, start progressing tiles.
    for i in range(level, 0, -1):
        ft.progress_tile(i, True)
        # For early culling:
        if i == early_culling_level:
            ft.identify_areas(
                i, 0.3, # 0.3(30%) is practical value.
                _PIXEL.AREA,
                _PIXEL.AREA,
                _PIXEL.OUTSIDE
            )

    # XXX Debug/profiling code
    if show_flag:
        _dbg_show_flag(
            ft, 
            'contour,filled,outside,area', 
            title='post-progress'
        )
    #XXX debug code end

    # Finalize fill result.
    ft.identify_areas(
        0, 0.1, # 0.1(10%) is practical value.
        _PIXEL.AREA,
        _PIXEL.FILLED,
        _PIXEL.AREA
    )

    if dilation > 0:
        ft.dilate(_PIXEL.FILLED, dilation)

    # After dilation, fill all holes.
    if fill_all_holes:
        ft.fill_holes()

    ft.draw_antialias()
    
    # Convert flagtiles into that new layer.
    _convert_result(targ, ft, color)
    
    # XXX Debug/profiling code
    print("processing time %s" % str(time.time()-ctm))
    if show_flag:
        _dbg_show_flag(
            ft,
            'result', 
            title='final result'
        )
    #XXX debug code end

def lasso_fill(targ, src, nodes, targ_pos, level, color,
               tolerance=0.1,
               alpha_threshold=0.2,
               dilation=2,
               fill_all_holes=True, 
               tile_output=False,
               debug_info=None# XXX for Debug 
               ):

    # XXX Debug/profiling code
    import time
    ctm = time.time()
    # XXX Debug code end

    # Create flagtilesurface object
    lt = lib.mypaintlib.LassofillSurface(self.nodes)

    ox = lt.get_origin_x()
    oy = lt.get_origin_y()
    tw = lt.get_width()
    th = lt.get_height()
    
    # XXX Debug code
    if debug_info:
        show_flag = debug_info.get('show_flag', False)
        tile_output = debug_info.get('tile_output', False)
        if tile_output:
            _dbg_tile_output(debug_info, ox, oy, tw, th, src)
    else:
        show_flag = False

    # Sampling colors from src color tiles around nodes.
    node_colors = {}
    # previous node position, in integer precision.
    px = None
    py = None

    for cn in self.nodes:
        cx = int(cn.x) 
        cy = int(cn.y)
        tx = cx // N
        ty = cy // N
        if cx != px or cy != py:
            px = cx
            py = cy
            if lt.tile_exists(tx-ox, ty-oy):
                with src.tile_request(tx, ty, readonly=True) as src_tile:
                    cx %= N
                    cy %= N
                    key = tuple([int(c) for c in src_tile[cy][cx]])
                    # Only opaque color should be count.
                    if key[3] != 0:
                        cnt = node_colors.get(key, 0)
                        node_colors[key] = cnt + 1
                

    # get most frequent color
    targ_item = sorted(
        node_colors.items(), 
        key=operator.itemgetter(1)
    )[-1]      
    # We sort the dict by value,
    # but need key(tuple of pixel color), not value(color count). 
    targ_r, targ_g, targ_b, targ_a = targ_item[0] 
    
    # Convert colortiles into flagtiles
    for fty in range(th):
        for ftx in range(tw):
            ct = lt.get_tile(ftx, fty, False)
            if ct is not None:
                with src.tile_request(ftx+ox, fty+oy, readonly=True) as src_tile:
                    # That tile has vaild polygon area already.
                    # If not, tile
                    ct.convert_from_color(
                        src_tile,
                        targ_r, targ_g, targ_b, targ_a,
                        tolerance,
                        alpha_threshold,
                        True
                    )
    
    # Finalize lasso fill, with converting 
    # PIXEL_CONTOUR into PIXEL_OUTSIDE (need to fill contour holes)
    # and remained PIXEL_AREA into PIXEL_FILLED.
    if dilation > 0:
        lt.dilate(_PIXEL.FILLED, dilation)
    lt.convert_pixel(0, _PIXEL.CONTOUR, _PIXEL.OUTSIDE)
    lt.convert_pixel(0, _PIXEL.AREA, _PIXEL.FILLED)

    if fill_all_holes:
        lt.fill_holes()
    lt.draw_antialias()
    
    # Convert flagtiles into that new layer.
    _convert_result(targ, lt, color)

    # XXX Debug/profiling code
    print("processing time %s" % str(time.time()-ctm))
    if show_flag:
        _dbg_show_flag(lt)
    #XXX debug code end

def cut_protrude(layers,  
                 alpha_threshold=0.2,
                 debug_info=None# XXX for Debug 
                 ):
    # XXX Debug/profiling code
    import time
    ctm = time.time()

    if debug_info is not None:
        show_flag = debug_info.get('show_flag', False)
        tile_output = debug_info.get('tile_output', False)
    else:
        show_flag = False

    def dbg_show(title, level=0):
        if show_flag:
            _dbg_show_flag(
                ft, 
                'contour,area,filled,overwrap', 
                title=title,
                level=level
            )
    # XXX Debug code end

    level = 2 # Default progress level as 2(4x4 pixel).

    cl = layers.current
    flagtiles = {}
    assert hasattr(cl, "_surface")
    cursurf = cl._surface
    curtiles = cl.get_tile_coords()
    # XXX Important: We must reject(hide) current layer temporally 
    # from visible contents, before any tile-requests for
    # layerstack object.
    cl.visible = False 

    # Make background invisible and
    # Get all-connected
    assert hasattr(layers, "background_visible")
    layers.background_visible = False
    allsurf = lib.surface.TileRequestWrapper(layers)

    for tx, ty in curtiles:
        flag_tile = None
        with cursurf.tile_request(tx, ty, readonly=True) as src_tile:
            flag_tile = lib.mypaintlib.Flagtile(_PIXEL.EMPTY)
            # As first, convert current layer as 
            flag_tile.convert_from_transparency(
                src_tile,
                alpha_threshold, 
                _PIXEL.AREA
            )
            # Then, overwrite with visible contents 
            # (except for current layer).
            with allsurf.tile_request(tx, ty, readonly=True) as src_tile:
                flag_tile.convert_from_transparency(
                    src_tile,
                    alpha_threshold,
                    _PIXEL.CONTOUR, # Placing pixel, same as _PIXEL.CONTOUR
                    _PIXEL.OVERWRAP # Place _PIXEL.OVERWRAP when there is already non-_PIXEL.EMPTY pixel.
                )
            flagtiles[(tx, ty)] = flag_tile

    ft = lib.mypaintlib.CutprotrudeSurface(flagtiles, level)
    ox = ft.get_origin_x()
    oy = ft.get_origin_y()
    for tx, ty in flagtiles:
        ftx = tx - ox 
        fty = ty - oy 
        ft.borrow_tile(ftx, fty, flagtiles[(tx, ty)])

    dbg_show('pre-remove') # XXX Debug/profiling code

    # Build pyramid
    ft.build_progress_seed()

    dbg_show('post-seed', level=level) # XXX Debug/profiling code

    # We need decide(i.e. convert to PIXEL_FILLED) `not protruding` area, 
    # in top level.
    # Because, in this level, We can separate some protruding areas
    # which are actually connected with accepted-filled area at 
    # pyramid level 0.
    ft.identify_areas(
        level, 0.2,
        _PIXEL_AREA,
        _PIXEL_FILLED,
        _PIXEL_AREA
    )

    dbg_show('post-remove', level=level) # XXX Debug/profiling code

    for i in range(level, 0, -1):
        ft.progress_tile(i, False)

    dbg_show('post-progress') # XXX Debug/profiling code

    # Then, decide filled pixels at final level.
    # Protruding pixels are remained as PIXEL_AREA.
    # That PIXEL_AREA areas are erased from mypaint colortile later.

    ft.identify_areas(
        0, 0.1,
        _PIXEL_AREA,
        _PIXEL_FILLED,
        _PIXEL_AREA
    )

    # size_threshold -1 means `remove even maximum area`
    # Because actual maximum area already ensured.
    # All of opened area should be removed, even if it is largest one.

    # Remove overwrapped and isolated contour area.
    ft.remove_overwrap_contour() 

    dbg_show('finalized') # XXX Debug/profiling code

    # Dilate erasing result(PIXEL_AREA) a bit 
    # and eliminate garbage pixels.
    ft.dilate(_PIXEL.AREA, 1)

    # Just make PIXEL_AREA pixel to 
    # transparent(i.e. erase) at the same position of 
    # Mypaint colortile.
    # Other pixels are left unchanged.
    for tx, ty in curtiles:
        flag_tile = flagtiles[(tx, ty)]
        with cursurf.tile_request(tx, ty, readonly=False) as dst_tile:
            flag_tile.convert_to_transparent(
                dst_tile,
                _PIXEL.AREA 
            )

    # Don't forget to restore visible states of 
    # background and current layer.
    layers.background_visible = True
    cl.visible = True

#--------------------------------------- 
# XXX For debug 

def _dbg_show_flag(ft, method, level=0, title=None):
    # XXX debug method
    
    # From lib/progfilldefine.hpp
    FLAG_WORK = 0x10
    FLAG_AA = 0x80
    npbuf = None
    if title is None:
        title = method
    title = title.decode('UTF-8')

    colors = {
        _PIXEL.FILLED : (255, 160, 0),
        _PIXEL.AREA : (0, 255, 255),
        _PIXEL.CONTOUR : (255, 255, 0),
        _PIXEL.EMPTY : (0, 255, 0), # level color
        _PIXEL.OUTSIDE : (255, 0, 128),
        _PIXEL.OVERWRAP : (255, 128, 128),
        _PIXEL.FILLED | FLAG_WORK: (0, 255, 0),
        FLAG_WORK : (255, 0, 0),
        FLAG_AA : (0, 255, 255)
    }

    def create_npbuf(npbuf, level, pix):
        r, g, b = colors[pix]
        return ft.render_to_numpy(
            npbuf, pix,
            r, g, b,
            level
        )
        
    for m in method.split(","):
        m = m.strip()

        if m == 'filled' or m == 'show_all':
            npbuf = create_npbuf(npbuf, level, _PIXEL.FILLED)
            npbuf = create_npbuf(npbuf, level, _PIXEL.CONTOUR)
        elif m == 'area':
            npbuf = create_npbuf(npbuf, level, _PIXEL.AREA)
        elif m == 'close_area':
            npbuf = create_npbuf(npbuf, 0, _PIXEL.AREA)
        elif m == 'initial_level' or m == 'empty':
            npbuf = create_npbuf(npbuf, level, 0)
        elif m == 'contour':
            npbuf = create_npbuf(npbuf, level, _PIXEL.CONTOUR)
        elif m == 'outside':
            npbuf = create_npbuf(npbuf, level, _PIXEL.OUTSIDE)
        elif m == 'overwrap':
            npbuf = create_npbuf(npbuf, level, _PIXEL.OVERWRAP)
        elif m == 'level':
            npbuf = create_npbuf(npbuf, level, _PIXEL.AREA)
            npbuf = create_npbuf(npbuf, level, _PIXEL.FILLED)
            npbuf = create_npbuf(npbuf, level, _PIXEL.CONTOUR)
            npbuf = create_npbuf(npbuf, 0, _PIXEL.FILLED | FLAG_WORK)
        elif m == 'result':
            npbuf = create_npbuf(npbuf, 0, _PIXEL.OUTSIDE)
            npbuf = create_npbuf(npbuf, 0, _PIXEL.AREA)
            npbuf = create_npbuf(npbuf, 0, _PIXEL.FILLED)
            npbuf = create_npbuf(npbuf, 0, _PIXEL.FILLED | FLAG_WORK)
        elif m == 'walking':
            npbuf = create_npbuf(npbuf, -1, FLAG_WORK)
        elif m == 'alias':
            npbuf = create_npbuf(npbuf, -1, FLAG_AA)
        elif show_all:
            npbuf = create_npbuf(npbuf, 0, _PIXEL.AREA)
            npbuf = create_npbuf(npbuf, 0, _PIXEL.FILLED)
            npbuf = create_npbuf(npbuf, 0, _PIXEL.CONTOUR)
        else:
            print("[ERROR] there is no case for %s in _dbg_show_flag" % m)        

    from PIL import Image, ImageDraw
    newimg = Image.fromarray(npbuf)
    d = ImageDraw.Draw(newimg)
    d.text((0,0),title, (0, 0, 0))
    d.text((1,1),title, (255, 255, 255))        
    newimg.save('/tmp/closefill_check.jpg')
    newimg.show()

def _dbg_tile_output(info, ox, oy, tw, th, src, prefix="tiles"):
    # XXX debug method

    print("# tile writing enabled")
    import os
    import numpy as np
    import json
    import tempfile
    import time
    lt = time.localtime()
    dates = "%02d-%02d_%02d,%02d,%02d_" % lt[1:6],
    prefix = "%s_%s" % (prefix, dates)
    outputdir = tempfile.mkdtemp(prefix=prefix)
   #info = {}
   #if hasattr(self, 'bbox'):
   #    info['bbox'] = self.bbox
   #if hasattr(self, 'nodes'):
   #    info['nodes'] = self.nodes
   #info['tolerance'] = self.tolerance
   #info['level'] = self.pyramid_level
   #info['erase_pixel'] = self.erase_pixel
   #info['fill_all_holes'] = self.fill_all_holes

   #targ_pos = self.targ_color_pos
   #if targ_pos is not None:
   #    info['targ_color_pos'] = self.targ_color_pos

    info['tilesurf_dimension'] = (ox, oy, tw, th) # Add this information

    # Modify bbox when it is Rect instance.
    bbox = info.get('bbox', None)
    import lib.helpers as helpers
    if bbox and isinstance(bbox, helpers.Rect):
        info['bbox'] = (bbox.x, bbox.y, bbox.w, bbox.h)

    for ty in range(oy, th+oy):
        for tx in range(ox, tw+ox):
            with src.tile_request(tx, ty, readonly=True) as src_tile:
                np.save('%s/tile_%d_%d.npy' % (outputdir, tx, ty), src_tile)

    with open("%s/info" % outputdir, 'w') as ofp:
        json.dump(info, ofp)


if __name__ == '__main__':

    pass


