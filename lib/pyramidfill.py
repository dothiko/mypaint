# This file is part of MyPaint.
# Copyright (C) 2019 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

import lib.mypaintlib
import lib.surface
import lib.helpers
import gui.drawutils

import operator 
import numpy as np
import math

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
    RESERVE = lib.mypaintlib.Flagtile.PIXEL_RESERVE_VALUE
    INVALID = lib.mypaintlib.Flagtile.PIXEL_INVALID_VALUE


## Utility Functions

def _convert_result(dstsurf, ft, color, combine_mode, pixel=_PIXEL.FILLED):
    """Draw filled result of flagtile surface into 
    mypaint-colortiles of the layer.

    :param dstsurf: Destination tiledsurface.
    :param combine_mode: CombineMode of lib.mypaintlib.tile_combine.
                         It is defined in lib/pixops.hpp .
                         Also refer to
                         https://www.w3.org/TR/compositing-1/
    """

    ox = ft.get_origin_x()
    oy = ft.get_origin_y()
    tw = ft.get_width()
    th = ft.get_height()

    assert type(color) is tuple
    r, g, b = color

    filled = {}
    work_tile = np.zeros((N, N, 4), "uint16")

    for fty in range(th):
        for ftx in range(tw):
            ct = ft.get_tile(ftx, fty, False)
            if ct is not None and not ct.is_filled_with(_PIXEL.INVALID):
                tx = ftx + ox
                ty = fty + oy
                with dstsurf.tile_request(tx, ty, readonly=False) \
                        as dst_tile:
                    ct.convert_to_color(
                        work_tile,
                        r, g, b,
                        pixel
                    )
                    lib.mypaintlib.tile_combine(
                        combine_mode,
                        work_tile,
                        dst_tile, 
                        True, 1.0,
                    )
                    work_tile[:] = 0
                    filled[(tx, ty)] = dst_tile

    # Actual color-tile bbox might be different from FlagtileSurface
    # Because FlagtileSurface has some reserved empty tiles for dilation.
    # So we should use `filled` dictionary.
    tile_bbox = lib.surface.get_tiles_bbox(filled)
    dstsurf.notify_observers(*tile_bbox)

def _create_FlagtileSurface(nodes, lasso=False, show_debug=False):
    fn = nodes[0]
    min_x, min_y = fn.x, fn.y
    max_x, max_y = min_x, min_y

    for cn in nodes[1:]:
        min_x = min(min_x, cn.x)
        min_y = min(min_y, cn.y)
        max_x = max(max_x, cn.x)
        max_y = max(max_y, cn.y)

    min_x = int(min_x)
    min_y = int(min_y)
    max_x = int(max_x)
    max_y = int(max_y)

    ft = lib.mypaintlib.ClosefillSurface(min_x, min_y, max_x, max_y)
    # XXX We need to convert node position in model-coordinate 
    # to flagtilesurface-coordinate.
    ox = ft.get_origin_x() * N
    oy = ft.get_origin_y() * N

    idx=0
    bn = fn
    for cn in nodes[1:]:
        ft.draw_line(
            int(bn.x)-ox, int(bn.y)-oy, 
            int(cn.x)-ox, int(cn.y)-oy, 
            _PIXEL.AREA
        )
        bn = cn
        idx+=1
    ft.draw_line(
        int(bn.x)-ox, int(bn.y)-oy, 
        int(fn.x)-ox, int(fn.y)-oy, 
        _PIXEL.AREA
    )

    # XXX Debug/profiling code
    if show_debug:
        _dbg_show_flag(ft, 'area', title='just drawn')
    # XXX Debug code end

    ft.decide_area()

    # XXX Debug/profiling code
    if show_debug:
        _dbg_show_flag(ft, 'area', title='surface created')
    # XXX Debug code end

    return ft

## Main Functions.

def close_fill(targ, src, nodes, targ_pos, level, 
               color, combine_mode,
               tolerance=0.1,
               alpha_threshold=0.2,
               dilation=2,
               fill_all_holes=True, 
               debug_info=None# XXX for Debug 
               ):
    """Do `Close and fill`, called from ClosefillMode of lib/command.py

    :param targ: The target tiledsurface.
    :param src: The source tiledsurface.
    :param nodes: Sequence of polygon nodes, which defines closed area.
    """
    early_culling_level = 2 # XXX fixed value, early culling for 4x4 pixel level

    # XXX Debug/profiling code
    import time
    ctm = time.time()
    if debug_info:
        show_flag = debug_info.get('show_flag', False)
        tile_output = debug_info.get('tile_output', False)
    else:
        show_flag = False
        tile_output = False
    # XXX Debug code end

    # Create flagtilesurface object
    ft = _create_FlagtileSurface(nodes, show_debug=show_flag)

    ox = ft.get_origin_x()
    oy = ft.get_origin_y()
    tw = ft.get_width()
    th = ft.get_height()
    
    # XXX Debug code
    if tile_output:
        _dbg_tile_output(debug_info, ox, oy, tw, th, src)
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

    # Build pyramid
    ft.propagate_upward(level)

    # Deciding outside of filling area
    ft.decide_outside(level)
      
    # Then, start progressing tiles.
    for i in range(level, 0, -1):
        ft.propagate_downward(i, True)
        # For early culling:
        if i == early_culling_level:
            ft.identify_areas(
                i, 
                _PIXEL.AREA,
                0.0, # All unrejected areas are remained.
                0.3, # 0.3(30%) is practical value.
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
        0, 
        _PIXEL.AREA,
        -1.0, # All unrejected areas are accepted.
        0.1, # 0.1(10%) is practical value.
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
    _convert_result(targ, ft, color, combine_mode)
    
    # XXX Debug/profiling code
    print("processing time %s" % str(time.time()-ctm))
    if show_flag:
        _dbg_show_flag(
            ft,
            'result', 
            title='final result'
        )
    #XXX debug code end

def lasso_fill(targ, src, nodes, targ_pos, level,
               color, combine_mode,
               tolerance=0.1,
               alpha_threshold=0.2,
               dilation=2,
               fill_all_holes=True, 
               debug_info=None# XXX for Debug 
               ):

    # Create flagtilesurface object
    lt = _create_FlagtileSurface(nodes, lasso=True)

    # XXX Debug code
    import time
    ctm = time.time()
    if debug_info:
        show_flag = debug_info.get('show_flag', False)
        tile_output = debug_info.get('tile_output', False)
        if tile_output:
            _dbg_tile_output(debug_info, ox, oy, tw, th, src)
    else:
        show_flag = False
    if show_flag:
        _dbg_show_flag(
            lt, 
            'contour,area', 
            title='created'
        )
    # XXX Debug code end

    ox = lt.get_origin_x()
    oy = lt.get_origin_y()
    tw = lt.get_width()
    th = lt.get_height()
    

    # Sampling colors from src color tiles around nodes.
    node_colors = {}
    # previous node position, in integer precision.
    px = None
    py = None

    for cn in nodes:
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
                   ## Only opaque color should be count.
                   #if key[3] != 0:
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

    if (targ_a == 0 
            and combine_mode != lib.mypaintlib.CombineDestinationOut):
        alpha_threshold = 0.0 # We write in empty area, so disable alpha threshould.
        combine_mode = lib.mypaintlib.CombineNormal
    
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
                        False
                    )
    
    # Finalize lasso fill, with converting 
    # PIXEL_CONTOUR into PIXEL_OUTSIDE (need to fill contour holes)
    # and remained PIXEL_AREA into PIXEL_FILLED.
    if dilation > 0:
        lt.dilate(_PIXEL.FILLED, dilation)
    lt.convert_pixel(0, _PIXEL.CONTOUR, _PIXEL.OUTSIDE)
    lt.convert_pixel(0, _PIXEL.AREA, _PIXEL.FILLED)

    # XXX Debug code
    if show_flag:
        _dbg_show_flag(
            lt, 
            'contour,filled,outside,area', 
            title='converted'
        )
    # XXX Debug code end

    if fill_all_holes:
        lt.fill_holes()

    lt.draw_antialias()

    # XXX Debug code
    if show_flag:
        _dbg_show_flag(
            lt, 
            'contour,filled,outside,area', 
            title='finalized'
        )
    # XXX Debug code end
    
    # Convert flagtiles into that new layer.
    _convert_result(targ, lt, color, combine_mode)

    # XXX Debug/profiling code
    print("processing time %s" % str(time.time()-ctm))
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

   #show_flag=True

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
    background_orig = layers.background_visible
    layers.background_visible = False
    allsurf = layers.get_tile_accessible_layer_rendering(layers)

    for tx, ty in curtiles:
        flag_tile = None
        with cursurf.tile_request(tx, ty, readonly=True) as src_tile:
            flag_tile = lib.mypaintlib.Flagtile(_PIXEL.EMPTY)
            # As first, convert current layer as 
            flag_tile.convert_from_transparency(
                src_tile,
                alpha_threshold, 
                _PIXEL.AREA,
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

    ft = lib.mypaintlib.CutprotrudeSurface(flagtiles)
    ox = ft.get_origin_x()
    oy = ft.get_origin_y()
    for tx, ty in flagtiles:
        ftx = tx - ox 
        fty = ty - oy 
        ft.borrow_tile(ftx, fty, flagtiles[(tx, ty)])

    dbg_show('pre-remove') # XXX Debug/profiling code

    # Build pyramid
    ft.propagate_upward(level)

    dbg_show('post-seed', level=level) # XXX Debug/profiling code

    # We need identify (i.e. convert to PIXEL_FILLED) `not protruding` area, 
    # in top level.
    # Because, in this level, We can separate some protruding areas
    # which are actually connected with accepted-filled area at 
    # pyramid level 0.
    ft.identify_areas(
        level, 
        _PIXEL.AREA,
        -1.0, # All unrejected areas are accepted.
        0.2, # 0.2(20%) is practical value for this stage.
        _PIXEL.FILLED,
        _PIXEL.AREA,
        -1 # Don't use size-threshold
    )

    dbg_show('post-remove', level=level) # XXX Debug/profiling code

    for i in range(level, 0, -1):
        ft.propagate_downward(i, False)

    dbg_show('post-progress') # XXX Debug/profiling code

    # Then, decide filled pixels at final level again.
    # Protruding pixels are remained as PIXEL_AREA.
    # That PIXEL_AREA areas are erased from mypaint colortile later.

    ft.identify_areas(
        0, 
        _PIXEL.AREA,
        -1.0, # All unrejected areas are accepted.
        0.1, # 0.1(10%) is practical value.
        _PIXEL.FILLED,
        _PIXEL.AREA,
        -1 # Don't use size-threshold
    )
    dbg_show('last identify') # XXX Debug/profiling code

    # Dilate erasing result(PIXEL_AREA) a bit 
    # and eliminate garbage pixels.
    #
    # This should be placed before `remove_overwrap_contour`.
    # Because, when a contour completely surrounded by protruding pixels,
    # (and entire contour would be overwrapped)
    # FlagtileSurface::identify_areas misdetect that contour is
    # completely surrounded by discarding pixels, cannot detect it
    # has large FILLED pixels inside, so entire contour pixel is removed. 
    # I have no idea to fix this misdetection, but dilating before
    # `remove_overwrap_contour` would produce mostly works well. 
    ft.dilate(_PIXEL.AREA, 1)

    # Remove overwrapped and isolated contour area.
    ft.remove_overwrap_contour() 

    dbg_show('finalized') # XXX Debug/profiling code

    _convert_result(
        cursurf, ft, (1.0, 1.0, 1.0), 
        lib.mypaintlib.CombineDestinationOut, # To erase, use DestOut
        pixel=_PIXEL.AREA
    )

    # Don't forget to restore visible states of 
    # background and current layer.
    layers.background_visible = background_orig
    cl.visible = True

def flood_fill(src, x, y, color, bbox, tolerance, dst, **kwargs):
    """Fills connected areas of one surface into another
    With using pyramid-fill functionality.

    :param src: Source surface-like object
    :type src: Anything supporting readonly tile_request()
    :param x: Starting point X coordinate
    :param y: Starting point Y coordinate
    :param color: an RGB color
    :type color: tuple
    :param bbox: Bounding box: limits the fill
    :type bbox: lib.helpers.Rect or equivalent 4-tuple
    :param tolerance: how much filled pixels are permitted to vary
    :type tolerance: float [0.0, 1.0]
    :param dst: Target surface
    :type dst: lib.tiledsurface.MyPaintSurface

    Keyword args:
    :param dilate_size: dilation size
    :type dilate_size: int, maximum MYPAINT_TILE_SIZE/2
    :param pyramid_level: progress-pixel-level. a.k.a gap closing level. 
    :type gap_level: int [0, 6]
    :param do_antialias: Draw psuedo anti-aliasing pixels around filled area.
    :type do_antialias: boolean

    See also `lib.layer.Layer.flood_fill()`.
    """
    # Color to fill with
    fill_r, fill_g, fill_b = color

    # Limits
    tolerance = lib.helpers.clamp(tolerance, 0.0, 1.0)

    # Maximum area to fill: tile and in-tile pixel extents
    bbx, bby, bbw, bbh = bbox
    if bbh <= 0 or bbw <= 0:
        return
    bbbrx = bbx + bbw - 1
    bbbry = bby + bbh - 1
    min_tx = int(bbx // N)
    min_ty = int(bby // N)
    max_tx = int(bbbrx // N)
    max_ty = int(bbbry // N)
    base_min_px = int(bbx % N)
    base_min_py = int(bby % N)
    base_max_px = int(bbbrx % N)
    base_max_py = int(bbbry % N)

    # Tile and pixel addressing for the seed point
    btx, bty = int(x // N), int(y // N)
    bpx, bpy = int(x % N), int(y % N)

    # Sample the pixel color there to obtain the target color
    with src.tile_request(btx, bty, readonly=True) as start:
        targ_r, targ_g, targ_b, targ_a = [int(c) for c in start[bpy][bpx]]
    if targ_a == 0:
        targ_r = 0
        targ_g = 0
        targ_b = 0
        targ_a = 0

    # Setting keyword options here.
    dilation_size = kwargs.get('dilation_size', 0) 
    pyramid_level = kwargs.get('pyramid_level', 0)
    erase_pixel = kwargs.get('erase_pixel', False)
    anti_alias = kwargs.get('anti_alias', True)
    alpha_threshold = kwargs.get('alpha_threshold', 0.2)
    fill_all_holes = kwargs.get('fill_all_holes', False)

    if erase_pixel:
        combine_mode = lib.mypaintlib.CombineDestinationOut
    else:
        combine_mode = lib.mypaintlib.CombineNormal 

    # All values setup end. 
    # Adjust pixel coordinate into progress-coordinate.
    MAX_PROGRESS_LEVEL = 6
    if pyramid_level == 0:
        MN = N

    # Flood-fill loop
    filled = {}

    # XXX DEBUG START
    print("tolerance %.6f" % tolerance)
    print("pyramid_level %s" % str(pyramid_level))
    print("dilation %s" % str(dilation_size))
   #print("px/py %s" % str((min_px, min_py, max_px, max_py, px, py)))
    print("tx/ty limits %s" % str((min_tx, min_ty, max_tx, max_ty)))
    print("targ color : %s" % str((targ_r, targ_g, targ_b, targ_a)))
    print("fill_all_holes : %s" % str(fill_all_holes))

    show_flag = kwargs.get('show_flag', False)
    tile_output = kwargs.get('tile_output', False)
    # XXX DEBUG END

    def do_fill(targ_pixel, fill_pixel, level):
        min_px = base_min_px >> level
        min_py = base_min_py >> level
        max_px = base_max_px >> level
        max_py = base_max_py >> level
        tileq = [
            ((btx, bty),
             [(bpx>>level, bpy>>level)])
        ]
        MN = 1 << (MAX_PROGRESS_LEVEL - level) # Pyramid tile-size

        while len(tileq) > 0:
            (tx, ty), seeds = tileq.pop(0)
            # Bbox-derived limits
            if tx > max_tx or ty > max_ty:
                continue
            if tx < min_tx or ty < min_ty:
                continue
            # Pixel limits within this tile...
            min_x = 0
            min_y = 0
            max_x = MN-1
            max_y = MN-1
            # ... vary at the edges
            if tx == min_tx:
                min_x = min_px
            if ty == min_ty:
                min_y = min_py
            if tx == max_tx:
                max_x = max_px
            if ty == max_ty:
                max_y = max_py
            # Flood-fill one tile
            flag_tile = filled.get((tx, ty), None)
            if flag_tile is None:
                with src.tile_request(tx, ty, readonly=True) as src_tile:
                    flag_tile = lib.mypaintlib.Flagtile(_PIXEL.AREA)
                    flag_tile.convert_from_color(
                        src_tile,
                        targ_r, targ_g, targ_b, targ_a,
                        tolerance,
                        alpha_threshold,
                        False
                    )
                    if level > 0:
                        flag_tile.propagate_upward(level)
                    filled[(tx, ty)] = flag_tile
            if flag_tile is not None:
                overflows = lib.mypaintlib.pyramid_flood_fill(
                    flag_tile, seeds,
                    min_x, min_y, max_x, max_y,
                    level,
                    targ_pixel,
                    fill_pixel
                )
                if overflows is not None:
                    seeds_n, seeds_e, seeds_s, seeds_w = overflows

                    # Enqueue overflows in each cardinal direction
                    if seeds_n and ty > min_ty:
                        tpos = (tx, ty-1)
                        tileq.append((tpos, seeds_n))
                    if seeds_w and tx > min_tx:
                        tpos = (tx-1, ty)
                        tileq.append((tpos, seeds_w))
                    if seeds_s and ty < max_ty:
                        tpos = (tx, ty+1)
                        tileq.append((tpos, seeds_s))
                    if seeds_e and tx < max_tx:
                        tpos = (tx+1, ty)
                        tileq.append((tpos, seeds_e))

    # Practically, level 2 is enough for 1st stage flood-fill.
    base_level = 2 
    if pyramid_level > base_level:
        # 1st stage: 
        # Do flood-fill in `enough low` level pyramid.
        # With this, we can get maximum flood-fillable areas of
        # _PIXEL.RESERVE.
        do_fill(_PIXEL.AREA, _PIXEL.RESERVE, base_level)

        # 2nd stage:
        # Propergate upward all tiles genareted in 1st stage.
        for t in filled.values():
            for i in range(base_level+1, pyramid_level+1):
                t.propagate_upward_single(i)

        # Then, Fill it with _PIXEL.FILLED at assigned (greater) level.
        # This gives you possible outmost gap-closing fill.
        # Some of _PIXEL.RESERVE still remained there, but that 
        # placeholder pixels would be converted at 
        # FlagtileSurface::propagate_downward later.
        do_fill(_PIXEL.RESERVE, _PIXEL.FILLED, pyramid_level)

    else:
        do_fill(_PIXEL.AREA, _PIXEL.FILLED, pyramid_level)

    # FloodfillSurface increace reference counter of `filled` dictionary.
    # `filled` dictionary used to just calculate bbox of surface in 
    # constructor.
    ft = lib.mypaintlib.FloodfillSurface(filled)
    ox = ft.get_origin_x()
    oy = ft.get_origin_y()
    for tx, ty in filled:
        ftx = tx - ox 
        fty = ty - oy 
        ft.borrow_tile(ftx, fty, filled[(tx, ty)])

    if show_flag:
        _dbg_show_flag(
            ft, 
            "area, contour, outside, reserve", 
            base_level-1, 
            title="1st stage"
        )
        _dbg_show_flag(
            ft, 
            "area, filled, contour, outside, reserve", 
            pyramid_level, 
            title="flood filled level"
        )

    # Early rejection of outside pixels.
    # This is important, because some pixel areas might be
    # connected at lower level.
    if pyramid_level > 0:
        ft.identify_areas(
            pyramid_level, 
            _PIXEL.AREA,
            0.0, # All non-rejected areas are left unchanged.
            0.1,
            _PIXEL.AREA,
            _PIXEL.OUTSIDE,
            -1 # With -1, reject even largest area (Because outside areas might be largest)
        )

    # XXX DEBUG START
    if show_flag:
        _dbg_show_flag(
            ft, 
            "area, filled, contour, outside", 
            pyramid_level, 
            title="just identifyed level"
        )

    if tile_output:
        info = {}
        info['color'] = color
        info['bbox'] = bbox
        info['tolerance'] = tolerance
        info['level'] = pyramid_level
        info['targ_color_pos'] = [x, y]
        info['erase_pixel'] = erase_pixel
        info['dilation_size'] = dilation_size
        _dbg_tile_output(
            info,
            ox, oy,
            ft.get_width(),
            ft.get_height(),
            src,
            prefix="flood_tiles"
        )
    # XXX DEBUG END
    
    # Downward-progress pixels.
    for i in range(pyramid_level, 0, -1):
        ft.propagate_downward(i, True)

    if show_flag:
        _dbg_show_flag(ft, "area, outside, filled, contour", 1, title="progressed-level")
        _dbg_show_flag(ft, "area, outside, filled, contour", 0, title="progressed")

    # Finalize pixels.
    # Make undecided pixels into filled one, if condition matched.
    if pyramid_level > 0:
        ft.identify_areas(
            0, 
            _PIXEL.AREA,
            0.0, # Accept areas which touches the PIXEL_FILLED areas only.
            0.1, # Reject opened area greater than 0.1
            _PIXEL.FILLED,
            _PIXEL.AREA
        )

    if show_flag:
        _dbg_show_flag(ft, "area, filled, contour", 0, title="removed")
    
    ft.dilate(_PIXEL.FILLED, dilation_size)

    if fill_all_holes:
        ft.fill_holes()
    if anti_alias:
        ft.draw_antialias()
            
    # XXX DEBUG START
    print("end tile")
    if show_flag:
        _dbg_show_flag(ft, "filled", 0, title="result")
    print("--- finalize end ---")
    # XXX DEBUG END

    _convert_result(
        dst, ft, (fill_r, fill_g, fill_b), 
        combine_mode
    )

    bbox = lib.surface.get_tiles_bbox(filled)
    dst.notify_observers(*bbox)
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
        _PIXEL.RESERVE : (128, 128, 255),
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
        elif m == 'reserve':
            npbuf = create_npbuf(npbuf, level, _PIXEL.RESERVE)
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


