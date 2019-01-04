/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "opencv_util.hpp"
#include "common.hpp"
#include "fix15.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <mypaint-tiled-surface.h>
#include <math.h>


typedef struct {
    unsigned int xstride;
    unsigned int ystride;
    uint8_t *buf;
} Numpybufinfo;

void
setup_numpy_buffer(
    Numpybufinfo *info, 
    PyArrayObject *arr, 
    int dst_x, int dst_y, int offset, int pixsize) 
{
    info->xstride = PyArray_STRIDE(arr, 1);
    info->ystride = PyArray_STRIDE(arr, 0);
    info->buf = (uint8_t*)PyArray_DATA(arr);
    if (dst_x !=0 || dst_y != 0)
        info->buf += (info->ystride * dst_y + info->xstride * dst_x);
    info->ystride -= (info->xstride * offset);
    info->ystride /= pixsize;
} 

/**
* @opencvutil_convert_tile_to_image
* Convert a mypaint tile to part of opencv numpy `BGR` image and
* part of alpha (i.e. grayscale 8bit) image simultaneously.
*
* @param py_cvimg : 8bit * 3 numpy buffer of destination opencv BGR image.
* @param py_cvalpha : 8bit numpy buffer of destination opencv alpha image.
*                     This MUST have same dimensions with py_cvimg.
* @param py_tile : 16bit * 4 numpy buffer of mypaint tile.
*
* @return: Py_None
* @detail
* This function converts entire colored pixels of mypaint tile
* to numpy BGR(3bytes/pixel) buffer for opencv.
* Also, separately assigned grayscale(one byte/pixel) numpy buffer receives
* alpha channel pixels of that tile.
*/

PyObject*
opencvutil_convert_tile_to_image(
    PyObject *py_cvimg,
    PyObject *py_cvalpha,
    PyObject *py_tile,
    int dst_x, int dst_y)
{
    PyArrayObject *tile_arr = (PyArrayObject*)py_tile;
    PyArrayObject *img_arr = (PyArrayObject*)py_cvimg;
    PyArrayObject *alpha_arr = (PyArrayObject*)py_cvalpha;
    int bin_w = PyArray_DIM(img_arr, 1);
    int bin_h = PyArray_DIM(img_arr, 0);
#ifdef HEAVY_DEBUG
    int alpha_w = PyArray_DIM(alpha_arr, 1);
    int alpha_h = PyArray_DIM(alpha_arr, 0);
    assert(PyArray_ISCARRAY(tile_arr));
    assert(PyArray_ISCARRAY(img_arr));
    assert(PyArray_ISCARRAY(alpha_arr));
    assert(PyArray_TYPE(tile_arr) == NPY_UINT16);
    assert(PyArray_TYPE(img_arr) == NPY_UINT8);
    assert(PyArray_TYPE(alpha_arr) == NPY_UINT8);
    assert(dst_x + MYPAINT_TILE_SIZE <= bin_w);
    assert(dst_y + MYPAINT_TILE_SIZE <= bin_h);
    assert(bin_w == alpha_w);
    assert(bin_h == alpha_h);
#endif

    Numpybufinfo info_img;  
    setup_numpy_buffer(
        &info_img, img_arr, 
        dst_x, dst_y, MYPAINT_TILE_SIZE, sizeof(uint8_t)
    );

    Numpybufinfo info_alpha;  
    setup_numpy_buffer(
        &info_alpha, alpha_arr, 
        dst_x, dst_y, MYPAINT_TILE_SIZE, sizeof(uint8_t)
    );
                     
    Numpybufinfo info_tile;  
    setup_numpy_buffer(
        &info_tile, tile_arr, 
        0, 0, MYPAINT_TILE_SIZE, sizeof(fix15_short_t) 
    );

    uint8_t *buf_dst = info_img.buf;
    fix15_short_t *buf_src = (fix15_short_t*)info_tile.buf;

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
            // Color conversion
            uint32_t r, g, b, a;
            r = *buf_src++;
            g = *buf_src++;
            b = *buf_src++;
            a = *buf_src++;
            *buf_dst++ = (uint8_t)((b * 255) / (1<<15));
            *buf_dst++ = (uint8_t)((g * 255) / (1<<15));
            *buf_dst++ = (uint8_t)((r * 255) / (1<<15));

            // Alpha conversion
            *info_alpha.buf++ = (uint8_t)((a * 255) / (1<<15));
        }
        buf_dst += info_img.ystride;
        info_alpha.buf += info_alpha.ystride;
        buf_src += info_tile.ystride;
    }

    Py_RETURN_NONE;
}

/**
* @opencvutil_convert_image_to_tile
* Convert part of opencv numpy `BGR` image into a mypaint tile.
* Also, alpha (i.e. grayscale 8bit) image is converted from same position, 
* simultaneously.
*
* @param py_cvimg : 8bit * 3 numpy buffer of destination opencv BGR image.
* @param py_cvalpha : 8bit numpy buffer of destination opencv alpha image.
*                     This MUST have same dimensions with py_cvimg.
* @param py_tile : 16bit * 4 numpy buffer of mypaint tile.
* @param py_cvmask : 8bit numpy buffer of destination opencv polygon mask image.
*                    This MUST have same dimensions with py_cvimg.
*
* @return: Py_None
* @detail
* This function converts a part of opencv image array into a mypaint tile.
* The entire pixels of that tile would be written.
* Alpha channel of that tile is converted from  separately assigned 
* grayscale(one byte/pixel) numpy buffer.
*/

PyObject*
opencvutil_convert_image_to_tile(
    PyObject *py_cvimg,
    PyObject *py_cvalpha,
    PyObject *py_tile,
    PyObject *py_cvmask,
    int src_x, int src_y)
{
    PyArrayObject *tile_arr = (PyArrayObject*)py_tile;
    PyArrayObject *img_arr = (PyArrayObject*)py_cvimg;
    PyArrayObject *alpha_arr = (PyArrayObject*)py_cvalpha;
    PyArrayObject *mask_arr = (PyArrayObject*)py_cvmask;
    int bin_w = PyArray_DIM(img_arr, 1);
    int bin_h = PyArray_DIM(img_arr, 0);
#ifdef HEAVY_DEBUG
    int alpha_w = PyArray_DIM(alpha_arr, 1);
    int alpha_h = PyArray_DIM(alpha_arr, 0);
    int mask_w = PyArray_DIM(mask_arr, 1);
    int mask_h = PyArray_DIM(mask_arr, 0);
    assert(PyArray_ISCARRAY(tile_arr));
    assert(PyArray_ISCARRAY(img_arr));
    assert(PyArray_ISCARRAY(alpha_arr));
    assert(PyArray_TYPE(tile_arr) == NPY_UINT16);
    assert(PyArray_TYPE(img_arr) == NPY_UINT8);
    assert(PyArray_TYPE(alpha_arr) == NPY_UINT8);
    assert(src_x + MYPAINT_TILE_SIZE <= bin_w);
    assert(src_y + MYPAINT_TILE_SIZE <= bin_h);
    assert(bin_w == alpha_w);
    assert(bin_h == alpha_h);
    assert(bin_w == mask_w);
    assert(bin_h == mask_h);
#endif

    Numpybufinfo info_img;  
    setup_numpy_buffer(
        &info_img, img_arr, 
        src_x, src_y, MYPAINT_TILE_SIZE, sizeof(uint8_t)
    );

    Numpybufinfo info_alpha;  
    setup_numpy_buffer(
        &info_alpha, alpha_arr, 
        src_x, src_y, MYPAINT_TILE_SIZE, sizeof(uint8_t)
    );

    Numpybufinfo info_mask;  
    setup_numpy_buffer(
        &info_mask, mask_arr, 
        src_x, src_y, MYPAINT_TILE_SIZE, sizeof(uint8_t)
    );
                     
    Numpybufinfo info_tile;  
    setup_numpy_buffer(
        &info_tile, tile_arr, 
        0, 0, MYPAINT_TILE_SIZE, sizeof(fix15_short_t) 
    );

    uint8_t *buf_src = info_img.buf;
    fix15_short_t *buf_dst = (fix15_short_t*)info_tile.buf;

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
            if (*info_mask.buf++ > 0) {
                // XXX basically copied from lib/pixops.cpp
                uint32_t r, g, b, a;
                // rgba are 8bit/channel datas.
                // and, OpenCV should have `BGR` data...
                b = *buf_src++;
                g = *buf_src++;
                r = *buf_src++;
                a = *info_alpha.buf++;

                // convert to fixed point (with rounding)
                r = (r * (1<<15) + 255/2) / 255;
                g = (g * (1<<15) + 255/2) / 255;
                b = (b * (1<<15) + 255/2) / 255;
                a = (a * (1<<15) + 255/2) / 255;

                // premultiply alpha (with rounding), save back
                /*
                *buf_dst++ = (r * a + (1<<15)/2) / (1<<15);
                *buf_dst++ = (g * a + (1<<15)/2) / (1<<15);
                *buf_dst++ = (b * a + (1<<15)/2) / (1<<15);
                */
                *buf_dst++ = r;
                *buf_dst++ = g;
                *buf_dst++ = b;
                *buf_dst++ = a;
            }
            else {
                buf_src+=3;
                info_alpha.buf++;
                buf_dst+=4;
            }
        }
        buf_src += info_img.ystride;
        info_alpha.buf += info_alpha.ystride;
        info_mask.buf += info_mask.ystride;
        buf_dst += info_tile.ystride; 
    }

    Py_RETURN_NONE;
}

/**
* @opencvutil_is_empty_area
* Tells whether the area from (x,y) to 
* (x+MYPAINT_TILE_SIZE-1, y+MYPAINT_TILE_SIZE-1) is empty or not.
* 
* @param py_cvalpha : 8bit numpy buffer of destination opencv alpha image.
* @param sx, sy : Target area position. This function searches 
*                 any valid(i.e. greater than 0) alpha pixels
*                 from (sx,sy) to (sx+MYPAINT_TILE_SIZE-1, sy+MYPAINT_TILE_SIZE-1).
*
* @return: Return Py_True when that area is vacant. otherwise, return Py_False.
*/
PyObject*
opencvutil_is_empty_area(
    PyObject *py_cvalpha,
    int sx, int sy
)
{
    PyArrayObject *alpha_arr = (PyArrayObject*)py_cvalpha;
    int alpha_w = PyArray_DIM(alpha_arr, 1);
    int alpha_h = PyArray_DIM(alpha_arr, 0);
#ifdef HEAVY_DEBUG
    assert(PyArray_ISCARRAY(alpha_arr));
    assert(PyArray_TYPE(alpha_arr) == NPY_UINT8);
    assert(sx + MYPAINT_TILE_SIZE <= alpha_w);
    assert(sy + MYPAINT_TILE_SIZE <= alpha_h);
#endif

    Numpybufinfo info_alpha;  
    setup_numpy_buffer(
        &info_alpha, alpha_arr, 
        sx, sy, MYPAINT_TILE_SIZE, sizeof(uint8_t)
    );
                     
    uint8_t *buf_src = info_alpha.buf;
    uint8_t total_pixel = 0;

    for(int y = 0;
            y < MYPAINT_TILE_SIZE;
            ++y)
    {
        for(int x = 0;
                x < MYPAINT_TILE_SIZE;
                ++x)
        {
            total_pixel |= *buf_src++;
        }
        buf_src += info_alpha.ystride;
    }

    if (total_pixel == 0)
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

