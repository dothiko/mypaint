/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "grabcututil.hpp"
#include "common.hpp"
#include "fix15.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <mypaint-tiled-surface.h>
#include <math.h>

// XXX borrowed from lib/fill.cpp
static inline fix15_t
_grabcutfill_color_match(const fix15_short_t c1_premult[4],
                       const fix15_short_t c2_premult[4],
                       const fix15_t tolerance)
{
    const fix15_short_t c1_a = c1_premult[3];
    fix15_short_t c1[] = {
        fix15_short_clamp(c1_a <= 0 ? 0 : fix15_div(c1_premult[0], c1_a)),
        fix15_short_clamp(c1_a <= 0 ? 0 : fix15_div(c1_premult[1], c1_a)),
        fix15_short_clamp(c1_a <= 0 ? 0 : fix15_div(c1_premult[2], c1_a)),
        fix15_short_clamp(c1_a),
    };
    const fix15_short_t c2_a = c2_premult[3];
    fix15_short_t c2[] = {
        fix15_short_clamp(c2_a <= 0 ? 0 : fix15_div(c2_premult[0], c2_a)),
        fix15_short_clamp(c2_a <= 0 ? 0 : fix15_div(c2_premult[1], c2_a)),
        fix15_short_clamp(c2_a <= 0 ? 0 : fix15_div(c2_premult[2], c2_a)),
        fix15_short_clamp(c2_a),
    };

    // Calculate the raw distance
    fix15_t dist = 0;
    for (int i=0; i<4; ++i) {
        fix15_t n = (c1[i] > c2[i]) ? (c1[i] - c2[i]) : (c2[i] - c1[i]);
        if (n > dist)
            dist = n;
    }
    /*
     * // Alternatively, could use
     * fix15_t sumsqdiffs = 0;
     * for (int i=0; i<4; ++i) {
     *     fix15_t n = (c1[i] > c2[i]) ? (c1[i] - c2[i]) : (c2[i] - c1[i]);
     *     n >>= 2; // quarter, to avoid a fixed maths sqrt() overflow
     *     sumsqdiffs += fix15_mul(n, n);
     * }
     * dist = fix15_sqrt(sumsqdiffs) << 1;  // [0.0 .. 0.5], doubled
     * // but the max()-based metric will a) be more GIMP-like and b) not
     * // lose those two bits of precision.
     */

    // Compare with adjustable tolerance of mismatches.
    static const fix15_t onepointfive = fix15_one + fix15_halve(fix15_one);
    if (tolerance > 0) {
        dist = fix15_div(dist, tolerance);
        if (dist > onepointfive) {  // aa < 0, but avoid underflow
            return 0;
        }
        else {
            fix15_t aa = onepointfive - dist;
            if (aa < fix15_halve(fix15_one))
                return fix15_short_clamp(fix15_double(aa));
            else
                return fix15_one;
        }
    }
    else {
        if (dist > tolerance)
            return 0;
        else
            return fix15_one;
    }
}

/**
* @grabcututil_convert_tile_to_binary
* convert mypaint tile to numpy binary image
*
* @param margin:  The margin pixels count around binary image.
*                 This is needed for cv2.grabCut function to detect
*                 'background'
* @param inverted:When this is zero, flag is set
*                 if tile pixel match the assigned color.
*                 When this is not zero, flag is set
*                 if tile pixel does NOT match.
* @return: Py_None
* @detail
* This function converts specific colored area of mypaint tile
* to binary flag and write it into py_binary.
*/

PyObject*
grabcututil_convert_tile_to_binary(
    PyObject *py_binary,
    PyObject *py_tile,
    int dst_x, int dst_y,
    double targ_r, double targ_g, double targ_b,
    int value,
    int margin, int inverted,
    double alpha_tolerance)
{
    PyArrayObject *tile_arr = (PyArrayObject*)py_tile;
    PyArrayObject *bin_arr = (PyArrayObject*)py_binary;
    int bin_w = PyArray_DIM(bin_arr, 1);
    int bin_h = PyArray_DIM(bin_arr, 0);
    dst_x += margin;
    dst_y += margin;
#ifdef HEAVY_DEBUG
    assert(PyArray_ISCARRAY(tile_arr));
    assert(PyArray_ISCARRAY(bin_arr));
    assert(PyArray_TYPE(tile_arr) == NPY_UINT16);
    assert(PyArray_TYPE(bin_arr) == NPY_UINT8);
    assert(dst_x + MYPAINT_TILE_SIZE < bin_w);
    assert(dst_y + MYPAINT_TILE_SIZE < bin_h);
#endif
    const unsigned int xstride_dst = PyArray_STRIDE(bin_arr, 1);
    unsigned int ystride_dst = PyArray_STRIDE(bin_arr, 0);
    uint8_t *buf_dst = (uint8_t*)PyArray_DATA(bin_arr);
    buf_dst+=(ystride_dst * dst_y + xstride_dst * dst_x);

    const unsigned int xstride_src = PyArray_STRIDE(tile_arr, 1);
    unsigned int ystride_src = PyArray_STRIDE(tile_arr, 0);
    fix15_short_t *buf_src = (fix15_short_t*)PyArray_DATA(tile_arr);

    fix15_short_t refcol[4];
    refcol[0] = fix15_short_clamp(targ_r * (double)fix15_one);
    refcol[1] = fix15_short_clamp(targ_g * (double)fix15_one);
    refcol[2] = fix15_short_clamp(targ_b * (double)fix15_one);
    refcol[3] = fix15_one;
    bool pixel_hit;
    fix15_t tolerance = (fix15_t)(0.1 * (double)fix15_one);
    const fix15_short_t alpha_t_short = 
        (fix15_short_t)((double)fix15_one * (1.0 - alpha_tolerance));
    

    ystride_dst -= xstride_dst * MYPAINT_TILE_SIZE;
    ystride_src -= xstride_src * MYPAINT_TILE_SIZE;

    for(int y = 0;
            y < MYPAINT_TILE_SIZE;
            ++y)
    {
        for(int x = 0;
                x < MYPAINT_TILE_SIZE;
                ++x)
        {
            
            if (buf_src[3] >= alpha_t_short)
            {
                pixel_hit =  _grabcutfill_color_match(refcol,
                                buf_src,
                                tolerance
                             ) != 0;

                pixel_hit ^= (inverted != 0);

                if (pixel_hit) {
                    *buf_dst = value;
                }

            }
            buf_dst++;
            buf_src+=4;
        }
        buf_dst += ystride_dst;
        buf_src += ystride_src;
    }

    Py_RETURN_NONE;
}


/**
* @grabcututil_convert_tile_to_image
* convert mypaint tile to part of opencv numpy image (not binary)
*
* @param bg_r, bg_g, bg_b:  Alpha pixel replacing background color.
* @param margin:  The margin pixels count around binary image.
* @return: Py_None
* @detail
* This function converts specific colored area of mypaint tile
* to binary flag and write it into py_binary.
*/

PyObject*
grabcututil_convert_tile_to_image(
    PyObject *py_cvimg,
    PyObject *py_tile,
    int dst_x, int dst_y,
    double bg_r, double bg_g, double bg_b,
    int margin)
{
    PyArrayObject *tile_arr = (PyArrayObject*)py_tile;
    PyArrayObject *cv_arr = (PyArrayObject*)py_cvimg;
    int bin_w = PyArray_DIM(cv_arr, 1);
    int bin_h = PyArray_DIM(cv_arr, 0);
    dst_x += margin;
    dst_y += margin;
#ifdef HEAVY_DEBUG
    assert(PyArray_ISCARRAY(tile_arr));
    assert(PyArray_ISCARRAY(cv_arr));
    assert(PyArray_TYPE(tile_arr) == NPY_UINT16);
    assert(PyArray_TYPE(cv_arr) == NPY_UINT8);
    assert(dst_x + MYPAINT_TILE_SIZE < bin_w);
    assert(dst_y + MYPAINT_TILE_SIZE < bin_h);
#endif
    const unsigned int xstride_dst = PyArray_STRIDE(cv_arr, 1);
    unsigned int ystride_dst = PyArray_STRIDE(cv_arr, 0);
    uint8_t *buf_dst = (uint8_t*)PyArray_DATA(cv_arr);
    buf_dst+=(ystride_dst * dst_y + xstride_dst * dst_x);

    const unsigned int xstride_src = PyArray_STRIDE(tile_arr, 1);
    unsigned int ystride_src = PyArray_STRIDE(tile_arr, 0);
    fix15_short_t *buf_src = (fix15_short_t*)PyArray_DATA(tile_arr);

    fix15_short_t br = (fix15_short_clamp)(bg_r * (double)fix15_one);
    fix15_short_t bg = (fix15_short_clamp)(bg_g * (double)fix15_one);
    fix15_short_t bb = (fix15_short_clamp)(bg_b * (double)fix15_one);
    br = (br * 255) / (1<<15);
    bg = (bg * 255) / (1<<15);
    bb = (bb * 255) / (1<<15);

    ystride_dst -= xstride_dst * MYPAINT_TILE_SIZE;
    ystride_src -= xstride_src * MYPAINT_TILE_SIZE;

    // XXX I want to use lib/pixops.cpp, but there is no
    // conversion function from RGBA of uint16_t to RGB uint8_t...
    // so, copied some codes from tile_convert_rgbu16_to_rgbu8.

    for(int y = 0;
            y < MYPAINT_TILE_SIZE;
            ++y)
    {
        for(int x = 0;
                x < MYPAINT_TILE_SIZE;
                ++x)
        {
            if (buf_src[3] != 0) {
                uint32_t r, g, b;
                r = *buf_src++;
                g = *buf_src++;
                b = *buf_src++;
                buf_src++; // alpha unused

                *buf_dst++ = (r * 255) / (1<<15);
                *buf_dst++ = (g * 255) / (1<<15);
                *buf_dst++ = (b * 255) / (1<<15);
            }
            else {
                buf_src+=4; // pixel unused

                *buf_dst++ = br;
                *buf_dst++ = bg;
                *buf_dst++ = bb;
            }

        }
        buf_dst += ystride_dst;
        buf_src += ystride_src;
    }

    Py_RETURN_NONE;
}

/**
* @grabcututil_convert_binary_to_tile
* convert a Opencv binary image array into mypaint tile.
*
* @param py_tile: mypaint numpy tile object.
* @param py_binary: Opencv 8bit binary array.
* @param targ_value: target value of binary array.
* @return: When at least one pixel is written, return PyTrue.
*          Otherwise, PyFalse.
* @detail
* This function assume the tile should be zero-initialized.
* Binary array is actually 8bit array, not bitwise array.
*/
PyObject*
grabcututil_convert_binary_to_tile(
    PyObject *py_tile,
    PyObject *py_binary,
    int x_src, int y_src,
    double fill_r, double fill_g, double fill_b,
    int targ_value,
    int margin)
{
    PyArrayObject *tile_arr = (PyArrayObject*)py_tile;
    PyArrayObject *bin_arr = (PyArrayObject*)py_binary;
    int bin_w = PyArray_DIM(bin_arr, 1);
    int bin_h = PyArray_DIM(bin_arr, 0);
    x_src += margin;
    y_src += margin;
#ifdef HEAVY_DEBUG
    assert(PyArray_ISCARRAY(tile_arr));
    assert(PyArray_ISCARRAY(bin_arr));
    assert(PyArray_TYPE(tile_arr) == NPY_UINT16);
    assert(PyArray_TYPE(bin_arr) == NPY_UINT8);
    assert(x_src + MYPAINT_TILE_SIZE < bin_w);
    assert(y_src + MYPAINT_TILE_SIZE < bin_h);
#endif
    const unsigned int xstride_src = PyArray_STRIDE(bin_arr, 1);
    unsigned int ystride_src = PyArray_STRIDE(bin_arr, 0);
    uint8_t *buf_src = (uint8_t*)PyArray_DATA(bin_arr);
    buf_src+=(ystride_src * y_src + xstride_src * x_src);

    const unsigned int xstride_dst = PyArray_STRIDE(tile_arr, 1);
    unsigned int ystride_dst = PyArray_STRIDE(tile_arr, 0);
    fix15_short_t *buf_dst = (fix15_short_t*)PyArray_DATA(tile_arr);

    ystride_dst -= xstride_dst * MYPAINT_TILE_SIZE;
    ystride_src -= xstride_src * MYPAINT_TILE_SIZE;

    fix15_short_t fr = (fix15_short_clamp)(fill_r * (double)fix15_one);
    fix15_short_t fg = (fix15_short_clamp)(fill_g * (double)fix15_one);
    fix15_short_t fb = (fix15_short_clamp)(fill_b * (double)fix15_one);

    unsigned int cnt = 0;

    for(int y = 0;
            y < MYPAINT_TILE_SIZE;
            ++y)
    {
        for(int x = 0;
                x < MYPAINT_TILE_SIZE;
                ++x)
        {
            if (*buf_src == targ_value) {
                *buf_dst++ = fr;
                *buf_dst++ = fg;
                *buf_dst++ = fb;
                *buf_dst++ = fix15_one;
                cnt++;
            }
            else {
                buf_dst+=4;
            }
            buf_src++;
            /*
            else {
                buf_dst+=3; // pixel unused
                *buf_dst++ = 0;
            }
            */
        }
        buf_dst += ystride_dst;
        buf_src += ystride_src;
    }

    if (cnt > 0)
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}
/**
* @grabcututil_setup_cvimg
* setup the margin-border of opencv array image.
*
* @param desc_of_parameter
* @return desc_of_return
* @detail
* opencv recognize as the image border is filled with
* background pixel. so, we need fill it.
*/
PyObject*
grabcututil_setup_cvimg(
    PyObject *py_cvimg,
    double bg_r, double bg_g, double bg_b,
    int margin)
{
    PyArrayObject *cv_arr = (PyArrayObject*)py_cvimg;
    int dst_w = PyArray_DIM(cv_arr, 1);
    int dst_h = PyArray_DIM(cv_arr, 0);
#ifdef HEAVY_DEBUG
    assert(PyArray_ISCARRAY(cv_arr));
    assert(PyArray_TYPE(cv_arr) == NPY_UINT8);
#endif
    const unsigned int xstride_dst = PyArray_STRIDE(cv_arr, 1);
    const unsigned int ystride_dst = PyArray_STRIDE(cv_arr, 0);
    unsigned int ystride_tmp;
    uint8_t *buf_dst_base = (uint8_t*)PyArray_DATA(cv_arr);

    uint32_t br = (bg_r * (double)fix15_one);
    uint32_t bg = (bg_g * (double)fix15_one);
    uint32_t bb = (bg_b * (double)fix15_one);
    br = (br * 255) / (1<<15);
    bg = (bg * 255) / (1<<15);
    bb = (bb * 255) / (1<<15);


    uint8_t *buf_dst = buf_dst_base;
    // initialize top and bottom
    ystride_tmp = ystride_dst - (xstride_dst * dst_w);
    for(int i=0; i < 2; i++)
    {
        for(int y = 0;
                y < margin;
                ++y)
        {
            for(int x = 0;
                    x < dst_w;
                    ++x)
            {
                *buf_dst++ = br;
                *buf_dst++ = bg;
                *buf_dst++ = bb;
            }
            buf_dst += ystride_tmp;
        }
        buf_dst = buf_dst_base + ystride_dst * (dst_h - margin);
    }

    // initialize right and left
    buf_dst = buf_dst_base + ystride_dst * margin;
    ystride_tmp = ystride_dst - (xstride_dst * margin);
    for(int i=0; i < 2; i++)
    {
        for(int y = margin;
                y < dst_h - margin;
                ++y)
        {
            for(int x = 0;
                    x < margin;
                    ++x)
            {
                *buf_dst++ = br;
                *buf_dst++ = bg;
                *buf_dst++ = bb;
            }
            buf_dst += ystride_tmp;
        }
        buf_dst = buf_dst_base
                    + (ystride_dst * margin)
                    + (xstride_dst * (dst_w - margin));
    }

    Py_RETURN_NONE;
}

/**
*
* @grabcututil_finalize_cvmask
* Finalize Opencv mask. 
*
* @param py_cvmask: Opencv mask array
* @param py_cvimg : Opencv image array, this should be lineart.
* @param targ_r, targ_g, targ_b: the background color(transparent part)
*                                of py_cvimg.
* @param remove_lineart: if not 0, remove opaque area of 
*                        lineart(py_cvimg).
* @return None
* @detail
* Remove lineart contour, which would be recognized
* as foreground pixels. 
* If these pixels remained, it would be annoying glitch 
* pixel around entire lineart.
* Also, convert mask value 'Probable foreground' to 
* 'surely foreground'.
* Both lineart and final result must same dimension.
*/
PyObject*
grabcututil_finalize_cvmask(
    PyObject *py_cvmask,
    PyObject *py_cvimg,
    double targ_r, double targ_g, double targ_b,
    int remove_lineart)
{
    PyArrayObject *img_arr = (PyArrayObject*)py_cvimg;
    PyArrayObject *mask_arr = (PyArrayObject*)py_cvmask;
    int dst_w = PyArray_DIM(mask_arr, 1);
    int dst_h = PyArray_DIM(mask_arr, 0);
#ifdef HEAVY_DEBUG
    int img_w = PyArray_DIM(img_arr, 1);
    int img_h = PyArray_DIM(img_arr, 0);
    assert(img_w == dst_w);
    assert(img_h == dst_h);
    assert(PyArray_ISCARRAY(img_arr));
    assert(PyArray_ISCARRAY(mask_arr));
    assert(PyArray_TYPE(img_arr) == NPY_UINT8);
    assert(PyArray_TYPE(mask_arr) == NPY_UINT8);
#endif
    const unsigned int xstride_src = PyArray_STRIDE(img_arr, 1);
    unsigned int ystride_src = PyArray_STRIDE(img_arr, 0);
    uint8_t *buf_src = (uint8_t*)PyArray_DATA(img_arr);

    const unsigned int xstride_dst = PyArray_STRIDE(mask_arr, 1);
    unsigned int ystride_dst = PyArray_STRIDE(mask_arr, 0);
    uint8_t *buf_dst = (uint8_t*)PyArray_DATA(mask_arr);

    uint32_t tr = (targ_r * (double)fix15_one);
    uint32_t tg = (targ_g * (double)fix15_one);
    uint32_t tb = (targ_b * (double)fix15_one);
    tr = (tr * 255) / (1<<15);
    tg = (tg * 255) / (1<<15);
    tb = (tb * 255) / (1<<15);

    ystride_dst -= xstride_dst * dst_w;
    ystride_src -= xstride_src * dst_w;

    for(int y = 0;
            y < dst_h;
            ++y)
    {
        for(int x = 0;
                x < dst_w;
                ++x)
        {
            if (remove_lineart != 0
                    && (buf_src[0] != tr
                        || buf_src[1] != tg
                        || buf_src[2] != tb)) {
                *buf_dst=0;
            }
            else {
               switch(*buf_dst) {
                   case 2:
                    // probable background
                    *buf_dst=0;
                    break;
                   case 3:
                    // probable foreground
                    *buf_dst=1;
                    break;
               }
            }
            
            ++buf_dst;
            buf_src+=3;
        }
        buf_dst += ystride_dst;
        buf_src += ystride_src;
    }

    Py_RETURN_NONE;
}
