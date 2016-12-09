# This file is part of MyPaint.
# Copyright (C) 2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

"""Graphical rendering helpers (splines, alpha checks, brush preview)

See also: gui.style

"""

## Imports
from __future__ import print_function

import logging
logger = logging.getLogger(__name__)

import math

import numpy as np
import cairo

from lib.helpers import clamp
import gui.style
from lib.color import HCYColor, RGBColor

import gi
from gi.repository import GdkPixbuf
from gi.repository import Gdk
from gi.repository import Gtk

from lib.brush import Brush, BrushInfo
import lib.tiledsurface
from lib.pixbufsurface import render_as_pixbuf


## Module constants

_BRUSH_PREVIEW_POINTS = [
    # px,  py,   press, xtilt, ytilt  # px,  py,   press, xtilt, ytilt
    (0.00, 0.00,  0.00,  0.00, 0.00), (1.00, 0.05,  0.00, -0.06, 0.05),
    (0.10, 0.10,  0.20,  0.10, 0.05), (0.90, 0.15,  0.90, -0.05, 0.05),
    (0.11, 0.30,  0.90,  0.08, 0.05), (0.86, 0.35,  0.90, -0.04, 0.05),
    (0.13, 0.50,  0.90,  0.06, 0.05), (0.84, 0.55,  0.90, -0.03, 0.05),
    (0.17, 0.70,  0.90,  0.04, 0.05), (0.83, 0.75,  0.90, -0.02, 0.05),
    (0.25, 0.90,  0.20,  0.02, 0.00), (0.81, 0.95,  0.00,  0.00, 0.00),
    (0.41, 0.95,  0.00,  0.00, 0.00), (0.80, 1.00,  0.00,  0.00, 0.00),
]


## Drawing functions

def spline_4p(t, p_1, p0, p1, p2):
    """Interpolated point using a Catmull-Rom spline

    :param float t: Time parameter, between 0.0 and 1.0
    :param array p_1: Point p[-1]
    :param array p0: Point p[0]
    :param array p1: Point p[1]
    :param array p2: Point p[2]
    :returns: Interpolated point, between p0 and p1
    :rtype: array

    Used for a succession of points, this function makes smooth curves
    passing through all specified points, other than the first and last.
    For each pair of points, and their immediate predecessor and
    successor points, the `t` parameter should be stepped incrementally
    from 0 (for point p0) to 1 (for point p1).  See also:

    * `spline_iter()`
    * http://en.wikipedia.org/wiki/Cubic_Hermite_spline
    * http://stackoverflow.com/questions/1251438
    """
    return (
        t*((2-t)*t - 1) * p_1 +
        (t*t*(3*t - 5) + 2) * p0 +
        t*((4 - 3*t)*t + 1) * p1 +
        (t-1)*t*t * p2
    ) / 2


def get_diff_spline_4p(t, p_1, p0, p1, p2):
    """Get the differential of spline_4p

    :param float t: Time parameter, between 0.0 and 1.0
    :param array or float p_1: Point p[-1]
    :param array or float p0: Point p[0]
    :param array or float p1: Point p[1]
    :param array or float p2: Point p[2]
    :returns: the differential at Time parameter
    :rtype: array or float (according to type of parameter)
    """
    C = -p_1 + p1
    B = 2*p_1 - 5*p0 + 4*p1 - p2
    A = -p_1 + 3*p0 - 3*p1 + p2
    return (3*A*t**2 + 2*B*t + C) * 0.5

def spline_iter(tuples, double_first=True, double_last=True):
    """Converts an list of control point tuples to interpolatable arrays

    :param list tuples: Sequence of tuples of floats
    :param bool double_first: Repeat 1st point, putting it in the result
    :param bool double_last: Repeat last point, putting it in the result
    :returns: Iterator producing (p-1, p0, p1, p2)

    The resulting sequence of 4-tuples is intended to be fed into
    spline_4p().  The start and end points are therefore normally
    doubled, producing a curve that passes through them, along a vector
    aimed at the second or penultimate point respectively.

    """
    cint = [None, None, None, None]
    if double_first:
        cint[0:3] = cint[1:4]
        cint[3] = np.array(tuples[0])
    for ctrlpt in tuples:
        cint[0:3] = cint[1:4]
        cint[3] = np.array(ctrlpt)
        if not any((a is None) for a in cint):
            yield cint
    if double_last:
        cint[0:3] = cint[1:4]
        cint[3] = np.array(tuples[-1])
        yield cint

def spline_iter_2(tuples, selected, offset, double_first=True, double_last=True):
    """Converts an list of control point tuples to interpolatable arrays

    :param list tuples: Sequence of tuples of floats
    :param list selected: Selected control points index list
    :param list offset: An Offset for selected points,a tuple of (x,y).
    :param bool double_first: Repeat 1st point, putting it in the result
    :param bool double_last: Repeat last point, putting it in the result
    :returns: Iterator producing (p-1, p0, p1, p2)

    The resulting sequence of 4-tuples is intended to be fed into
    spline_4p().  The start and end points are therefore normally
    doubled, producing a curve that passes through them, along a vector
    aimed at the second or penultimate point respectively.

    """
    cint = [None, None, None, None]
    if double_first:
        cint[0:3] = cint[1:4]
        cint[3] = np.array(tuples[0])
    for idx,ctrlpt in enumerate(tuples):
        cint[0:3] = cint[1:4]
        cint[3] = np.array(ctrlpt)
        if idx in selected:
            cint[3][0] += offset[0]
            cint[3][1] += offset[1]
        if not any((a is None) for a in cint):
            yield cint
    if double_last:
        cint[0:3] = cint[1:4]
        cint[3] = np.array(tuples[-1])
        yield cint

def spline_iter_3(tuples, basept, selected, radius, factor, offset_vec,
                  double_first=True, double_last=True):
    """Converts an list of control point tuples to interpolatable arrays


    :param list tuples: Sequence of tuples of floats
    :param object basept: Currently editing node.
    :param list selected: Indexes of selected nodes.
    :param float radius: radius of change affected area, in model coordinate.
    :param list offset: An Offset for selected points,a tuple of (x,y).
    :param bool double_first: Repeat 1st point, putting it in the result
    :param bool double_last: Repeat last point, putting it in the result
    :returns: Iterator producing (p-1, p0, p1, p2)

    The resulting sequence of 4-tuples is intended to be fed into
    spline_4p().  The start and end points are therefore normally
    doubled, producing a curve that passes through them, along a vector
    aimed at the second or penultimate point respectively.

    """
    cint = [None, None, None, None]

    if double_first:
        cint[0:3] = cint[1:4]
        cint[3] = np.array(tuples[0])
    for idx, ctrlpt in enumerate(tuples):
        cint[0:3] = cint[1:4]
        cint[3] = np.array(ctrlpt)   
        if basept:
            if (len(selected) <= 1 or idx in selected):
                new_coord = calc_ranged_offset(basept, ctrlpt,
                                radius, factor,
                                offset_vec)
                if new_coord:
                    cint[3][0] = new_coord[0]
                    cint[3][1] = new_coord[1]
        if not any((a is None) for a in cint):
            yield cint
    if double_last:
        cint[0:3] = cint[1:4]
        cint[3] = np.array(tuples[-1])
        yield cint
        
def calc_ranged_offset(basept, curpt, affect_radius, affect_factor, offset_vec):
    """ Calculate offset value with considering affecting radius and factor.
    
    :param tdw: TileDrawWidget, to get screen coordinate.
    :param basept: base control point
    :param curpt: current control point
    :param affect_radius: editing affect range.
    :param affect_factor: editing factor.
    :param offset_vec: A tuple of offset vector, 
        which is (length, normalized_x, normalized_y).
        
    :returns: The editing affected coordinate of curpt, when it is inside affect_radius.
    :rtype tuple: 
    """ 
    
    if basept and offset_vec:
        dist = math.hypot(curpt.x - basept.x, curpt.y - basept.y)
        if dist <= affect_radius:
            offset_len, nx, ny = offset_vec
            if affect_radius > 0.0:
                nd = dist / affect_radius
                # We need reversed value as length factor, so substruct from 1.0.
                factor = (1.0 - (nd ** affect_factor))
                dist = offset_len * factor
            else:
                dist = offset_len
            return (curpt.x + nx * dist, curpt.y + ny * dist)

    # Fallthrough:
    return (curpt.x,  curpt.y)
                
def _variable_pressure_scribble(w, h, tmult):
    points = _BRUSH_PREVIEW_POINTS
    px, py, press, xtilt, ytilt = points[0]
    yield (10, px*w, py*h, 0.0, xtilt, ytilt)
    event_dtime = 0.005
    point_time = 0.1
    for p_1, p0, p1, p2 in spline_iter(points, True, True):
        dt = 0.0
        while dt < point_time:
            t = dt/point_time
            px, py, press, xtilt, ytilt = spline_4p(t, p_1, p0, p1, p2)
            yield (event_dtime, px*w, py*h, press, xtilt, ytilt)
            dt += event_dtime
    px, py, press, xtilt, ytilt = points[-1]
    yield (10, px*w, py*h, 0.0, xtilt, ytilt)


def render_brush_preview_pixbuf(brushinfo, max_edge_tiles=4):
    """Renders brush preview images

    :param lib.brush.BrushInfo brushinfo: settings to render
    :param int max_edge_tiles: Use at most this many tiles along an edge
    :returns: Preview image, at 128x128 pixels
    :rtype: GdkPixbuf

    This generates the preview image (128px icon) used for brushes which
    don't have saved ones. These include brushes picked from .ORA files
    where the parent_brush_name doesn't correspond to a brush in the
    user's MyPaint brushes - they're used as the default, and for the
    Auto button in the Brush Icon editor.

    Brushstrokes are inherently unpredictable in size, so the allowable
    area is grown until the brush fits or until the rendering becomes
    too big. `max_edge_tiles` limits this growth.
    """
    assert max_edge_tiles >= 1
    brushinfo = brushinfo.clone()  # avoid capturing a ref
    brush = Brush(brushinfo)
    surface = lib.tiledsurface.Surface()
    N = lib.tiledsurface.N
    for size_in_tiles in range(1, max_edge_tiles):
        width = N * size_in_tiles
        height = N * size_in_tiles
        surface.clear()
        fg, spiral = _brush_preview_bg_fg(surface, size_in_tiles, brushinfo)
        brushinfo.set_color_rgb(fg)
        brush.reset()
        # Curve
        shape = _variable_pressure_scribble(width, height, size_in_tiles)
        surface.begin_atomic()
        for dt, x, y, p, xt, yt in shape:
            brush.stroke_to(surface.backend, x, y, p, xt, yt, dt)
        surface.end_atomic()
        # Check rendered size
        tposs = surface.tiledict.keys()

        outside = min({tx for tx, ty in tposs}) < 0
        outside = outside or (min({ty for tx, ty in tposs}) < 0)
        outside = outside or (max({tx for tx, ty in tposs}) >= size_in_tiles)
        outside = outside or (max({ty for tx, ty in tposs}) >= size_in_tiles)

        if not outside:
            break
    # Convert to pixbuf at the right scale
    rect = (0, 0, width, height)
    pixbuf = render_as_pixbuf(surface, *rect, alpha=True)
    if max(width, height) != 128:
        interp = (GdkPixbuf.InterpType.NEAREST if max(width, height) < 128
                  else GdkPixbuf.InterpType.BILINEAR)
        pixbuf = pixbuf.scale_simple(128, 128, interp)
    # Composite over a checquered bg via Cairo: shows erases
    size = gui.style.ALPHA_CHECK_SIZE
    nchecks = int(128 / size)
    cairo_surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 128, 128)
    cr = cairo.Context(cairo_surf)
    render_checks(cr, size, nchecks)
    Gdk.cairo_set_source_pixbuf(cr, pixbuf, 0, 0)
    cr.paint()
    cairo_surf.flush()
    return Gdk.pixbuf_get_from_surface(cairo_surf, 0, 0, 128, 128)


def _brush_preview_bg_fg(surface, size_in_tiles, brushinfo):
    """Render the background for brush previews, return paint color"""
    # The background color represents the overall nature of the brush
    col1 = (0.85, 0.85, 0.80)  # Boring grey, with a hint of paper-yellow
    col2 = (0.80, 0.80, 0.80)  # Grey, but will appear blueish in contrast
    fgcol = (0.05, 0.15, 0.20)  # Hint of color shows off HSV varier brushes
    spiral = False
    N = lib.tiledsurface.N
    fx = [
        (
            "eraser",  # pink=rubber=eraser; red=danger
            (0.8, 0.7, 0.7),  # pink/red tones: pencil eraser/danger
            (0.75, 0.60, 0.60),
            False, fgcol
        ),
        (
            "colorize",
            (0.8, 0.8, 0.8),  # orange on gray
            (0.6, 0.6, 0.6),
            False, (0.6, 0.2, 0.0)
        ),
        (
            "smudge",  # blue=water=wet, with some contrast
            (0.85, 0.85, 0.80),  # same as the regular paper color
            (0.60, 0.60, 0.70),  # bluer (water, wet); more contrast
            True, fgcol
        ),
    ]
    for cname, c1, c2, c_spiral, c_fg, in fx:
        if brushinfo.has_large_base_value(cname):
            col1 = c1
            col2 = c2
            fgcol = c_fg
            spiral = c_spiral
            break

    never_smudger = (brushinfo.has_small_base_value("smudge") and
                     brushinfo.has_only_base_value("smudge"))
    colorizer = brushinfo.has_large_base_value("colorize")

    if never_smudger and not colorizer:
        col2 = col1

    a = 1 << 15
    col1_fix15 = [c*a for c in col1] + [a]
    col2_fix15 = [c*a for c in col2] + [a]
    for ty in range(0, size_in_tiles):
        tx_thres = max(0, size_in_tiles - ty - 1)
        for tx in range(0, size_in_tiles):
            topcol = col1_fix15
            botcol = col1_fix15
            if tx > tx_thres:
                topcol = col2_fix15
            if tx >= tx_thres:
                botcol = col2_fix15
            with surface.tile_request(tx, ty, readonly=False) as dst:
                if topcol == botcol:
                    dst[:] = topcol
                else:
                    for i in range(N):
                        dst[0:N-i, i, ...] = topcol
                        dst[N-i:N, i, ...] = botcol
    return fgcol, spiral


def render_checks(cr, size, nchecks):
    """Render a checquerboard pattern to a cairo surface"""
    cr.set_source_rgb(*gui.style.ALPHA_CHECK_COLOR_1)
    cr.paint()
    cr.set_source_rgb(*gui.style.ALPHA_CHECK_COLOR_2)
    for i in xrange(0, nchecks):
        for j in xrange(0, nchecks):
            if (i+j) % 2 == 0:
                continue
            cr.rectangle(i*size, j*size, size, size)
            cr.fill()


def load_symbolic_icon(icon_name, size, fg=None, success=None,
                       warning=None, error=None, outline=None):
    """More Pythonic wrapper for gtk_icon_info_load_symbolic() etc.

    :param str icon_name: Name of the symbolic icon to render
    :param int size: Pixel size to render at
    :param tuple fg: foreground color (rgba tuple, values in [0..1])
    :param tuple success: success color (rgba tuple, values in [0..1])
    :param tuple warning: warning color (rgba tuple, values in [0..1])
    :param tuple error: error color (rgba tuple, values in [0..1])
    :param tuple outline: outline color (rgba tuple, values in [0..1])
    :returns: The rendered symbolic icon
    :rtype: GdkPixbuf.Pixbuf

    If the outline color is specified, a single-pixel outline is faked
    for the icon. Outlined renderings require a size 2 pixels larger
    than non-outlined if the central icon is to be of the same size.

    The returned value should be cached somewhere.

    """
    theme = Gtk.IconTheme.get_default()
    if outline is not None:
        size -= 2
    info = theme.lookup_icon(icon_name, size, Gtk.IconLookupFlags(0))
    rgba_or_none = lambda tup: (tup is not None) and Gdk.RGBA(*tup) or None
    icon_pixbuf, was_symbolic = info.load_symbolic(
        fg=rgba_or_none(fg),
        success_color=rgba_or_none(success),
        warning_color=rgba_or_none(warning),
        error_color=rgba_or_none(error),
    )
    assert was_symbolic
    if outline is None:
        return icon_pixbuf

    result = GdkPixbuf.Pixbuf.new(
        GdkPixbuf.Colorspace.RGB, True, 8,
        size+2, size+2,
    )
    result.fill(0x00000000)
    outline_rgba = list(outline)
    outline_rgba[3] /= 3.0
    outline_rgba = Gdk.RGBA(*outline_rgba)
    outline_stamp, was_symbolic = info.load_symbolic(
        fg=outline_rgba,
        success_color=outline_rgba,
        warning_color=outline_rgba,
        error_color=outline_rgba,
    )
    w = outline_stamp.get_width()
    h = outline_stamp.get_height()
    assert was_symbolic
    offsets = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ]
    for dx, dy in offsets:
        outline_stamp.composite(
            result,
            dx+1, dy+1, w, h,
            dx+1, dy+1, 1, 1,
            GdkPixbuf.InterpType.NEAREST, 255,
        )
    icon_pixbuf.composite(
        result,
        1, 1, w, h,
        1, 1, 1, 1,
        GdkPixbuf.InterpType.NEAREST, 255,
    )
    return result


def render_round_floating_button(cr, x, y, color, pixbuf, z=2,
                                 radius=gui.style.FLOATING_BUTTON_RADIUS):
    """Draw a round floating button with a standard size.

    :param cairo.Context cr: Context in which to draw.
    :param float x: X coordinate of the center pixel.
    :param float y: Y coordinate of the center pixel.
    :param lib.color.UIColor color: Color for the button base.
    :param GdkPixbuf.Pixbuf pixbuf: Icon to render.
    :param int z: Simulated height of the button above the canvas.
    :param float radius: Button radius, in pixels.

    These are used within certain overlays tightly associated with
    particular interaction modes for manipulating things on the canvas.

    """
    x = round(float(x))
    y = round(float(y))
    render_round_floating_color_chip(cr, x, y, color, radius=radius, z=z)
    cr.save()
    w = pixbuf.get_width()
    h = pixbuf.get_height()
    x -= w/2
    y -= h/2
    Gdk.cairo_set_source_pixbuf(cr, pixbuf, x, y)
    cr.rectangle(x, y, w, h)
    cr.clip()
    cr.paint()
    cr.restore()


def _get_paint_chip_highlight(color):
    """Paint chip highlight edge color"""
    highlight = HCYColor(color=color)
    ky = gui.style.PAINT_CHIP_HIGHLIGHT_HCY_Y_MULT
    kc = gui.style.PAINT_CHIP_HIGHLIGHT_HCY_C_MULT
    highlight.y = clamp(highlight.y * ky, 0, 1)
    highlight.c = clamp(highlight.c * kc, 0, 1)
    return highlight


def _get_paint_chip_shadow(color):
    """Paint chip shadow edge color"""
    shadow = HCYColor(color=color)
    ky = gui.style.PAINT_CHIP_SHADOW_HCY_Y_MULT
    kc = gui.style.PAINT_CHIP_SHADOW_HCY_C_MULT
    shadow.y = clamp(shadow.y * ky, 0, 1)
    shadow.c = clamp(shadow.c * kc, 0, 1)
    return shadow


def render_round_floating_color_chip(cr, x, y, color, radius, z=2, fill=True):
    """Draw a round color chip with a slight drop shadow

    :param cairo.Context cr: Context in which to draw.
    :param float x: X coordinate of the center pixel.
    :param float y: Y coordinate of the center pixel.
    :param lib.color.UIColor color: Color for the chip.
    :param float radius: Circle radius, in pixels.
    :param int z: Simulated height of the object above the canvas.
    :param fill: only draw circle when False

    Currently used for accept/dismiss/delete buttons and control points
    on the painting canvas, in certain modes.

    The button's style is similar to that used for the paint chips in
    the dockable palette panel. As used here with drop shadows to
    indicate that the blob can be interacted with, the style is similar
    to Google's Material Design approach. This style adds a subtle edge
    highlight in a brighter variant of "color", which seems to help
    address adjacent color interactions.

    """
    x = round(float(x))
    y = round(float(y))
    radius = round(radius)

    cr.save()
    cr.set_dash([], 0)
    cr.set_line_width(0)

    base_col = RGBColor(color=color)
    hi_col = _get_paint_chip_highlight(base_col)

    cr.arc(x, y, radius+0, 0, 2*math.pi)
    cr.set_line_width(2)
    render_drop_shadow(cr, z=z)

    cr.set_source_rgb(*base_col.get_rgb())
    if fill:
        cr.fill_preserve()
    #cr.clip_preserve()

    cr.set_source_rgb(*hi_col.get_rgb())
    cr.stroke()
    #cr.close_path()

    cr.restore()


def render_square_floating_color_chip(cr, x, y, color, size, fill):
    """Draw a round color chip with a slight drop shadow

    :param cairo.Context cr: Context in which to draw.
    :param float x: X coordinate of the center pixel.
    :param float y: Y coordinate of the center pixel.
    :param lib.color.UIColor color: Color for the chip.
    :param float size: square size, in pixels. the entire size should be
                       size+1(center)+size
    :param bool fill: fill inside rectangle when this is True.

    Currently used for accept/dismiss/delete buttons and control points
    on the painting canvas, in certain modes.

    The button's style is similar to that used for the paint chips in
    the dockable palette panel. As used here with drop shadows to
    indicate that the blob can be interacted with, the style is similar
    to Google's Material Design approach. This style adds a subtle edge
    highlight in a brighter variant of "color", which seems to help
    address adjacent color interactions.

    """
    x = round(float(x))
    y = round(float(y))
    
    cr.save()
    cr.set_dash([], 0)
    cr.set_line_width(0)

    base_col = RGBColor(color=color)

    cr.set_line_width(1)
    cr.set_source_rgb(*base_col.get_rgb())
    cr.rectangle(x - size, y - size, (size * 2) + 1, (size * 2) + 1)
    

    if fill:
        cr.fill()

    cr.stroke()

    cr.restore()

def render_drop_shadow(cr, z=2, line_width=None):
    """Draws a drop shadow for the current path.

    :param int z: Simulated height of the object above the canvas.
    :param float line_width: Override width of the line to shadow.

    This function assumes that the object will be drawn immediately
    afterwards using the current path, so the current path and transform
    are preserved. The line width will be inferred automatically from
    the current path if it is not specified.

    These shadows are suitable for lines of a single brightish color
    drawn over them. The combined style indicates that the object can be
    moved or clicked.

    """
    if line_width is None:
        line_width = cr.get_line_width()
    path = cr.copy_path()
    cr.save()
    dx = gui.style.DROP_SHADOW_X_OFFSET * z
    dy = gui.style.DROP_SHADOW_Y_OFFSET * z
    cr.translate(dx, dy)
    cr.new_path()
    cr.append_path(path)
    steps = int(math.ceil(gui.style.DROP_SHADOW_BLUR))
    alpha = gui.style.DROP_SHADOW_ALPHA / steps
    for i in reversed(range(steps)):
        cr.set_source_rgba(0.0, 0.0, 0.0, alpha)
        cr.set_line_width(line_width + 2*i)
        cr.stroke_preserve()
        alpha += alpha/2
    cr.translate(-dx, -dy)
    cr.new_path()
    cr.append_path(path)
    cr.restore()


def get_drop_shadow_offsets(line_width, z=2):
    """Get how much extra space is needed to draw the drop shadow.

    :param float line_width: Width of the line to shadow.
    :param int z: Simulated height of the object above the canvas.
    :returns: Offsets: (offs_left, offs_top, offs_right, offs_bottom)
    :rtype: tuple

    The offsets returned can be added to redraw bboxes, and are always
    positive. They reflect how much extra space is required around the
    bounding box for a line of the given width by the shadow rendered by
    render_drop_shadow().

    """
    dx = math.ceil(gui.style.DROP_SHADOW_X_OFFSET * z)
    dy = math.ceil(gui.style.DROP_SHADOW_Y_OFFSET * z)
    max_i = int(math.ceil(gui.style.DROP_SHADOW_BLUR)) - 1
    max_line_width = line_width + 2*max_i
    slack = 1
    return tuple(int(max(0, n)) for n in [
        -dx + max_line_width + slack,
        -dy + max_line_width + slack,
        dx + max_line_width + slack,
        dy + max_line_width + slack,
    ])

def draw_rectangle_follow_canvas(cr, tdw, sx, sy, ex, ey):
    """
    Draw a rectangle, which follows to rotating canvas.
    :param sx, sy: start point(left-top) of rectangle
    :param ex, ey: end point(right-bottom) of rectangle
    """
    if tdw:
        cr.move_to(*tdw.model_to_display(sx, sy))
        cr.line_to(*tdw.model_to_display(ex, sy))
        cr.line_to(*tdw.model_to_display(ex, ey))
        cr.line_to(*tdw.model_to_display(sx, ey))
    else:
        cr.move_to(sx, sy)
        cr.line_to(ex, sy)
        cr.line_to(ex, ey)
        cr.line_to(sx, ey)
    cr.close_path()

def create_rounded_rectangle_path(cr, x, y, w, h, radius):
    """ create a rounded rectangle path.
    YOU'LL NEED TO CLOSE IT AND FILL(or do something).

    :param radius: the radius of corner
    """
    right_angle = math.pi / 2.0
    cr.arc (x + w - radius, y + radius, radius, -right_angle, 0)
    cr.arc (x + w - radius, y + h - radius, radius, 0, right_angle)
    cr.arc (x + radius, y + h - radius, radius, right_angle, right_angle * 2.0)
    cr.arc (x + radius, y + radius, radius, right_angle * 2.0, right_angle * 3.0)

## Bezier curve
def get_bezier(p0, p1, p2, t):
    dt = 1-t
    return (dt**2)*p0 + 2.0*t*dt*p1 + (t**2)*p2                       

def get_cubic_bezier(p0, p1, p2, p3, t):
    """ Get the point of cubic bezier

    :param array or float p0: the starting point 
    :param array or float p1: control point of starting point
    :param array or float p2: control point of ending point
    :param array or float p3: the ending point
    :returns: the cubic bezier point at Time parameter
    :rtype: array or float (according to type of parameter)
    """
    dt = 1-t
    return (dt**3.0)*p0 + 3.0*(dt**2)*t*p1 + 3.0*dt*(t**2)*p2 + (t**3)*p3
            
def get_diff_cubic_bezier(p0, p1, p2, p3, t):
    """ Get the differential of cubic bezier

    :param array or float p0: the starting point 
    :param array or float p1: control point of starting point
    :param array or float p2: control point of ending point
    :param array or float p3: the ending point
    :returns: the differential at Time parameter
    :rtype: array or float (according to type of parameter)
    """
    dt = 1-t
    return 3.0*(t**2*(p3-p2)+2*t*(dt)*(p2-p1)+dt**2*(p1-p0))

def get_bezier_raw(p0, p1, p2, t):
    return ( linear_interpolation(p0, p1 , t), 
             linear_interpolation(p1, p2 , t) )

def get_cubic_bezier_raw(p0, p1, p2, p3, t):
    return ( linear_interpolation(p0, p1 , t), 
             linear_interpolation(p1, p2 , t),
             linear_interpolation(p2, p3 , t))

def get_minmax_bezier(x1, x2, x3, x4):
    a = x4-3*(x3-x2)-x1;
    b = 3*(x3-2*x2+x1);
    c = 3*(x2-x1);
    delta = b*b - 3*a*c;
    left = min(x1, x4);
    right = max(x1, x4);
    if 0 < delta:
        try:
            t0 = (-b+math.sqrt(delta))/(3*a)
        except ZeroDivisionError:
            pass
        else:
            if (0 <= t0 and t0 <= 1):
                tp = 1-t0;
                tx0 = (tp**3)*x1 + 3*(tp**2)*t0*x2 + 3*tp*(t0**2)*x3 + (t0**3)*x4
                if tx0 < left: 
                    left = tx0

        try:
            t1 = (-b-math.sqrt(delta))/(3*a)
        except ZeroDivisionError:
            pass
        else:
            if (0 <= t1 and t1 <= 1): 
                tp = 1-t1
                tx1 = (tp**3)*x1 + 3*(tp**2)*t1*x2 + 3*tp*(t1**2)*x3 + (t1**3)*x4
                if right < tx1: 
                    right = tx1

    return (left, right)

## Linear interpolation
def linear_interpolation(base, dest, step):
    return base + ((dest - base) * step)

def is_inside_segment(pt0, pt1, cx, cy):
    """To check whether the cursor point is
    inside assigned line segment of points or not.

    :param pt0: an array object, of starting point of line segment
    :param pt1: an array object, of ending point of line segment
    :param cx, cy: cursor point
    :return : the boolean flag , True when the cursor is inside the section.
    """
    seclen = math.hypot(pt0[0] - pt1[0] , pt0[1] - pt1[1])
    p0_c = math.hypot(pt0[0] - cx , pt0[1] - cy)
    c_p1 = math.hypot(cx - pt1[0] , cy - pt1[1])

    return p0_c < seclen and c_p1 < seclen
## Test code

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    import sys
    import lib.pixbuf
    for myb_file in sys.argv[1:]:
        if not myb_file.lower().endswith(".myb"):
            logger.warning("Ignored %r: not a .myb file", myb_file)
            continue
        with open(myb_file, 'r') as myb_fp:
            myb_json = myb_fp.read()
        myb_brushinfo = BrushInfo(myb_json)
        myb_pixbuf = render_brush_preview_pixbuf(myb_brushinfo)
        if myb_pixbuf is not None:
            myb_basename = myb_file[:-4]
            png_file = "%s_autopreview.png" % (myb_file,)
            logger.info("Saving to %r...", png_file)
            lib.pixbuf.save(myb_pixbuf, png_file, "png")
