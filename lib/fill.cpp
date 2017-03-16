/* This file is part of MyPaint.
 * Copyright (C) 2013-2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "fill.hpp"

#include "common.hpp"
#include "fix15.hpp"

#include "gapclose.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <glib.h>
#include <mypaint-tiled-surface.h>

// Pixel access helper for arrays in the tile format.

static inline fix15_short_t*
_floodfill_getpixel(PyArrayObject *array,
                    const unsigned int x,
                    const unsigned int y)
{
    const unsigned int xstride = PyArray_STRIDE(array, 1);
    const unsigned int ystride = PyArray_STRIDE(array, 0);
    return (fix15_short_t*)(PyArray_BYTES(array)
                            + (y * ystride)
                            + (x * xstride));
}


// modified local-function version of floodfill_color_match.
// this refers some additional parameters, state_flag.
//
// Similarity metric used by flood fill.  Result is a fix15_t in the range
// [0.0, 1.0], with zero meaning no match close enough.  Similar algorithm to
// the GIMP's pixel_difference(): a threshold is used for a "similar enough"
// determination.

static inline fix15_t
_floodfill_color_match(const fix15_short_t c1_premult[4],
                       const fix15_short_t c2_premult[4],
                       const fix15_t tolerance,
                       const int state_flag,
                       const int target_flag)
{
    // To share original _floodfill_color_match function with dilate.cpp, 
    // original code moved into fill.hpp as renamed 'floodfill_color_match()'
    // and this _floodfill_color_match is a little changed
    // to support 'overflow prevention' functionality for floodfill.
    fix15_t retvalue = floodfill_color_match(c1_premult, c2_premult, tolerance); 

    // When floodfill search pixel touches dilated contour state pixel,
    // it is treated same as color match failed case.
    if(retvalue > 0
       && (state_flag & target_flag) != 0) {
            // Normal fill guard: reject when scan touch dilated contour.
            return 0;
    }
    return retvalue;

}


// True if the fill should update the dst pixel of a src/dst pixel pair.

static inline bool
_floodfill_should_fill(const fix15_short_t src_col[4], // premult RGB+A
                       const fix15_short_t dst_col[4], // premult RGB+A
                       const fix15_short_t targ_col[4],  // premult RGB+A
                       const fix15_t tolerance,  // prescaled to range
                       const int state_flag,
                       const int target_flag)
{
    if (dst_col[3] != 0) {
        return false;   // already filled
    }
    return _floodfill_color_match(src_col, targ_col, 
                                  tolerance, state_flag, target_flag) > 0;
}

// A point in the fill queue

typedef struct {
    unsigned int x;
    unsigned int y;
} _floodfill_point;


// Flood fill implementation
PyObject *
tile_flood_fill (PyObject *src, /* readonly HxWx4 array of uint16 */
                 PyObject *dst, /* output HxWx4 array of uint16 */
                 PyObject *seeds, /* List of 2-tuples */
                 int targ_r, int targ_g, int targ_b, int targ_a, //premult
                 double fill_r, double fill_g, double fill_b,
                 int min_x, int min_y, int max_x, int max_y,
                 double tol,  /* [0..1] */
                 PyObject *state,  /* state pixel tile of uint8.*/
                 int target_flag)    
{
    // Scale the fractional tolerance arg
    const fix15_t tolerance = (fix15_t)(  MIN(1.0, MAX(0.0, tol))
                                        * fix15_one);

    // Fill colour args are floats [0.0 .. 1.0], non-premultiplied by alpha.
    // The targ_ colour components are 15-bit scaled ints in the range
    // [0 .. 1<<15], and are premultiplied by targ_a which has the same range.
    const fix15_short_t targ[4] = {
            fix15_short_clamp(targ_r), fix15_short_clamp(targ_g),
            fix15_short_clamp(targ_b), fix15_short_clamp(targ_a)
        };
    PyArrayObject *src_arr = ((PyArrayObject *)src);
    PyArrayObject *dst_arr = ((PyArrayObject *)dst);
    if(state == Py_None)
       state = NULL;

    // Dimensions are [y][x][component]
#ifdef HEAVY_DEBUG
    assert(PyArray_Check(src));
    assert(PyArray_Check(dst));
    assert(PyArray_DIM(src_arr, 0) == MYPAINT_TILE_SIZE);
    assert(PyArray_DIM(dst_arr, 0) == MYPAINT_TILE_SIZE);
    assert(PyArray_DIM(src_arr, 1) == MYPAINT_TILE_SIZE);
    assert(PyArray_DIM(dst_arr, 1) == MYPAINT_TILE_SIZE);
    assert(PyArray_DIM(src_arr, 2) == 4);
    assert(PyArray_DIM(dst_arr, 2) == 4);
    assert(PyArray_TYPE(src_arr) == NPY_UINT16);
    assert(PyArray_TYPE(dst_arr) == NPY_UINT16);
    assert(PyArray_ISCARRAY(src_arr));
    assert(PyArray_ISCARRAY(dst_arr));
    assert(PySequence_Check(seeds));
#endif
    if (min_x < 0) min_x = 0;
    if (min_y < 0) min_y = 0;
    if (max_x > MYPAINT_TILE_SIZE-1) max_x = MYPAINT_TILE_SIZE-1;
    if (max_y > MYPAINT_TILE_SIZE-1) max_y = MYPAINT_TILE_SIZE-1;
    if (min_x > max_x || min_y > max_y) {
        return Py_BuildValue("[()()()()]");
    }

    // Populate a working queue with seeds
    int x = 0;
    int y = 0;

    GQueue *queue = g_queue_new();   /* Of tuples, to be exhausted */
    for (int i=0; i<PySequence_Size(seeds); ++i) {
        PyObject *seed_tup = PySequence_GetItem(seeds, i);
#ifdef HEAVY_DEBUG
        assert(PySequence_Size(seed_tup) == 2);
#endif
        if (! PyArg_ParseTuple(seed_tup, "ii", &x, &y)) {
            continue;
        }
        Py_DECREF(seed_tup);
        x = MAX(0, MIN(x, MYPAINT_TILE_SIZE-1));
        y = MAX(0, MIN(y, MYPAINT_TILE_SIZE-1));
        const fix15_short_t *src_pixel = _floodfill_getpixel(src_arr, x, y);
        const fix15_short_t *dst_pixel = _floodfill_getpixel(dst_arr, x, y);
        char state_flag = gapclose_get_state_flag(state, x, y);
        if (_floodfill_should_fill(src_pixel, dst_pixel, targ, 
                                   tolerance, state_flag, target_flag)) {
            _floodfill_point *seed_pt = (_floodfill_point*)
                                          malloc(sizeof(_floodfill_point));
            seed_pt->x = x;
            seed_pt->y = y;
            g_queue_push_tail(queue, seed_pt);
        }
    }

    PyObject *result_n = PyList_New(0);
    PyObject *result_e = PyList_New(0);
    PyObject *result_s = PyList_New(0);
    PyObject *result_w = PyList_New(0);


    while (! g_queue_is_empty(queue)) {
        _floodfill_point *pos = (_floodfill_point*) g_queue_pop_head(queue);
        int x0 = pos->x;
        int y = pos->y;
        free(pos);
        
        // Find easternmost and westernmost points of the same colour
        // Westwards loop includes (x,y), eastwards ignores it.
        static const int x_delta[] = {-1, 1};
        static const int x_offset[] = {0, 1};
        for (int i=0; i<2; ++i)
        {
            bool look_above = true;
            bool look_below = true;
            for ( int x = x0 + x_offset[i] ;
                  x >= min_x && x <= max_x ;
                  x += x_delta[i] )
            {
                fix15_short_t *src_pixel = _floodfill_getpixel(src_arr, x, y);
                fix15_short_t *dst_pixel = _floodfill_getpixel(dst_arr, x, y);
                int state_flag = gapclose_get_state_flag(state, x, y);

                if (x != x0) { // Test was already done for queued pixels
                    if (! _floodfill_should_fill(src_pixel, dst_pixel,
                                                 targ, tolerance, 
                                                 state_flag, target_flag)) {
                        break;
                    }
                }
                    
                // Also halt if we're outside the bbox range
                if (x < min_x || y < min_y || x > max_x || y > max_y) {
                    break;
                }
                // Fill this pixel, and continue iterating in this direction
                fix15_t alpha = fix15_one;
                if (tolerance > 0) {
                    alpha = _floodfill_color_match(targ, src_pixel,
                                                   tolerance, state_flag,
                                                   target_flag);
                    // Since we use the output array to mark where we've been
                    // during the fill, we can't store an alpha of zero.
                    if (alpha == 0) {
                        alpha = 0x0001;
                    }
                }
                dst_pixel[0] = fix15_short_clamp(fill_r * alpha);
                dst_pixel[1] = fix15_short_clamp(fill_g * alpha);
                dst_pixel[2] = fix15_short_clamp(fill_b * alpha);
                dst_pixel[3] = alpha;
                // In addition, enqueue the pixels above and below.
                // Scanline algorithm here to avoid some pointless queue faff.
                if (y > 0) {
                    fix15_short_t *src_pixel_above = _floodfill_getpixel(
                                                       src_arr, x, y-1
                                                     );
                    fix15_short_t *dst_pixel_above = _floodfill_getpixel(
                                                       dst_arr, x, y-1
                                                     );
                    int state_flag_above = gapclose_get_state_flag(
                                            state, x, y-1);

                    bool match_above = _floodfill_should_fill(
                                         src_pixel_above, dst_pixel_above,
                                         targ, tolerance, state_flag_above,
                                         target_flag);
                    if (match_above) {
                        if (look_above) {
                            // Enqueue the pixel to the north
                            _floodfill_point *p = (_floodfill_point *) malloc(
                                                    sizeof(_floodfill_point)
                                                  );
                            p->x = x;
                            p->y = y-1;
                            g_queue_push_tail(queue, p);
                            look_above = false;
                        }
                    }
                    else { // !match_above
                        look_above = true;
                    }
                }
                else {
                    // Overflow onto the tile to the North.
                    // Scanlining not possible here: pixel is over the border.
                    PyObject *s = Py_BuildValue("ii", 
                                                x, MYPAINT_TILE_SIZE-1); 
                    PyList_Append(result_n, s);
                    Py_DECREF(s);
#ifdef HEAVY_DEBUG
                    assert(s->ob_refcnt == 1);
#endif
                }
                if (y < MYPAINT_TILE_SIZE - 1) {
                    fix15_short_t *src_pixel_below = _floodfill_getpixel(
                                                       src_arr, x, y+1
                                                     );
                    fix15_short_t *dst_pixel_below = _floodfill_getpixel(
                                                       dst_arr, x, y+1
                                                     );
                    int state_flag_below = gapclose_get_state_flag(
                                            state, x, y+1);

                    bool match_below = _floodfill_should_fill(
                                         src_pixel_below, dst_pixel_below,
                                         targ, tolerance, state_flag_below,
                                         target_flag);
                    if (match_below) {
                        if (look_below) {
                            // Enqueue the pixel to the South
                            _floodfill_point *p = (_floodfill_point *) malloc(
                                                    sizeof(_floodfill_point)
                                                  );
                            p->x = x;
                            p->y = y+1;
                            g_queue_push_tail(queue, p);
                            look_below = false;
                        }
                    }
                    else { //!match_below
                        look_below = true;
                    }
                }
                else {
                    // Overflow onto the tile to the South
                    // Scanlining not possible here: pixel is over the border.
                    PyObject *s = Py_BuildValue("ii", x, 0); 
                    PyList_Append(result_s, s);
                    Py_DECREF(s);
#ifdef HEAVY_DEBUG
                    assert(s->ob_refcnt == 1);
#endif
                }
                // If the fill is now at the west or east extreme, we have
                // overflowed there too.  Seed West and East tiles.
                if (x == 0) {
                    PyObject *s = Py_BuildValue("ii", 
                                                MYPAINT_TILE_SIZE-1, y); 
                    PyList_Append(result_w, s);
                    Py_DECREF(s);
#ifdef HEAVY_DEBUG
                    assert(s->ob_refcnt == 1);
#endif
                }
                else if (x == MYPAINT_TILE_SIZE-1) {
                    PyObject *s = Py_BuildValue("ii", 0, y);
                    PyList_Append(result_e, s);
                    Py_DECREF(s);
#ifdef HEAVY_DEBUG
                    assert(s->ob_refcnt == 1);
#endif
                }
            }
        }
    }

    // Clean up working state, and return where the fill has overflowed
    // into neighbouring tiles.
    g_queue_free(queue);
    PyObject *result = Py_BuildValue("[OOOO]", result_n, result_e,
                                               result_s, result_w);
    Py_DECREF(result_n);
    Py_DECREF(result_e);
    Py_DECREF(result_s);
    Py_DECREF(result_w);
#ifdef HEAVY_DEBUG
    assert(result_n->ob_refcnt == 1);
    assert(result_e->ob_refcnt == 1);
    assert(result_s->ob_refcnt == 1);
    assert(result_w->ob_refcnt == 1);
    assert(result->ob_refcnt == 1);
#endif
    return result;
}
