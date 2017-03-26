/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "dilation.hpp"
#include "common.hpp"
#include "fix15.hpp"
#include "fill.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <glib.h>
#include <mypaint-tiled-surface.h>
#include <math.h>

#define CENTER_TILE_INDEX 4
#define ABOVE_TILE_INDEX 1
#define BELOW_TILE_INDEX 7
#define LEFT_TILE_INDEX 3
#define RIGHT_TILE_INDEX 5
#define MAX_CACHE_COUNT 9

#define GET_TILE_INDEX(x, y)  (((x) + 1) + (((y) + 1) * 3))

#define GET_TX_FROM_INDEX(tx, idx) ((tx) + (((idx) % 3) - 1))
#define GET_TY_FROM_INDEX(ty, idx) ((ty) + (((idx) / 3) - 1))

// Maximum operation size. i.e. 'gap-radius'
#define MAX_OPERATION_SIZE ((MYPAINT_TILE_SIZE / 2) - 1)

#define CAPSULE_NAME "MYP_DT"

//// Utility functions.

inline char *get_tile_pixel(PyArrayObject *tile, const int x, const int y)
{
    // XXX almost copyed from fill.cpp::_floodfill_getpixel()
    const unsigned int xstride = PyArray_STRIDE(tile, 1);
    const unsigned int ystride = PyArray_STRIDE(tile, 0);
    return (PyArray_BYTES((PyArrayObject*)tile)
            + (y * ystride)
            + (x * xstride));
}

//// Structs.

typedef struct {
    unsigned int offset;
    unsigned int length;
} _kernel_line;


/**
 * @ Tilecache
 *
 *
 * @detail
 * Base Template class for tile cache operation and basic pixel operation
 * over entire tile cache.
 *
 * This class has 3x3 tile cache, to ensure morphology operation is exactly
 * done even at the border of the tile.
 * dilating operation would affect surrounding tiles over its border,
 * also, eroding operation would refer surrounding tiles.
 */

template <class T, class USER_PARAM_TYPE>
class Tilecache
{
    protected:
        // Cache array. They are initialized at
        // init_cached_tiles() method, so no need to
        // zerofill them at constructor.
        PyObject *m_cache_tiles[MAX_CACHE_COUNT];
        PyObject *m_cache_keys[MAX_CACHE_COUNT];

        // Parameters for on-demand cache tile generation.
        // These values are passed to Numpy-API.
        int m_NPY_TYPE;
        int m_dimension_cnt;

        // Current center tile location, in tiledict.
        int m_tx;
        int m_ty;

        // current target tiledict.
        PyObject *m_targ_dict;

        // kernel line elements.
        _kernel_line *m_kern;
        int m_kernel_size;

        // --- Tilecache management methods

        PyObject *_generate_cache_key(int tx, int ty)
        {
            return Py_BuildValue("ii", tx, ty);
        }

        PyObject *_generate_tile()
        {
            npy_intp dims[] = {MYPAINT_TILE_SIZE, MYPAINT_TILE_SIZE, m_dimension_cnt};
            return PyArray_ZEROS(3, dims, m_NPY_TYPE, 0);
        }

        PyObject *_get_cached_tile_from_index(int index, bool tile_request)
        {
            if (m_cache_tiles[index] == NULL) {
                if (tile_request == true) {
                    PyObject *dst_tile = _generate_tile();
                    m_cache_tiles[index] = dst_tile;
#ifdef HEAVY_DEBUG
                    assert(m_cache_keys[index] != NULL);
#endif
                    return dst_tile;
                }
            }
            return m_cache_tiles[index];
        }

        // Virtual handler method: used from _search_kernel
        virtual bool is_match_pixel(const int cx, const int cy,
                                    USER_PARAM_TYPE user_param)
        {
            return false;
        }

        // Utility method: check pixel existance inside circle-shaped kernel
        // for morphology operation.
        //
        // For morphology operation, we can use same looping logic to search pixels
        // but wanted result is different for each operation.
        // For dilation, we need
        // 'whether is there any valid pixel inside the kernel or not '.
        // On the other hand, for erosion, we need
        // 'whether is the entire kernel filled with valid pixels or not'.
        // So use 'target_result' parameter to share codes between
        // these different operations.
        inline bool _search_kernel(const int cx, const int cy,
                                   USER_PARAM_TYPE user_param)
        {
#ifdef HEAVY_DEBUG
            assert(m_kern != NULL);
#endif

            int sx = cx - m_kernel_size;
            int sy = cy - m_kernel_size;
            for (int y = 0;y <= m_kernel_size * 2; ++y) {
                for (int x=0 ;
                         x < m_kern[y].length ;
                         ++x) {
                    if (is_match_pixel( sx + x + m_kern[y].offset,
                                        sy + y,
                                        user_param)) {
                        return true;
                    }
                }
            }
            return false;
        }

    public:

        Tilecache(int npy_type, int dimension_cnt)
            : m_targ_dict(NULL), m_kern(NULL)
        {
            // Tile type and dimension are used in _generate_tile(),
            // when a new tile requested.
            // So only setup them, in constructor.
            m_NPY_TYPE = npy_type;
            m_dimension_cnt = dimension_cnt;
        }
        virtual ~Tilecache()
        {
            if (m_kern != NULL)
                delete [] m_kern;
        }

        // Initialize/finalize cache tile
        // Call this and initialize cache before dilate/erode called.
        void init_cached_tiles(PyObject *py_tiledict, int tx, int ty)
        {
            m_tx = tx;
            m_ty = ty;
            m_targ_dict = py_tiledict;

            // fill the cache array with current dilated dict.
            for (int y=-1; y <= 1; y++) {
                for (int x=-1; x <= 1; x++) {
                    int idx = GET_TILE_INDEX(x, y);
                    PyObject *key = _generate_cache_key(tx+x, ty+y);
                    PyObject *dst_tile = PyDict_GetItem(py_tiledict, key);
                    if (dst_tile) {
                        Py_INCREF(dst_tile);
                        // With PyDict_GetItem(), tile is just borrowed.
                        // so incref here.
                        // On the other hand, key object is always tile_requestd
                        // above line, so does not need incref.
#ifdef HEAVY_DEBUG
                        assert(dst_tile->ob_refcnt >= 2);
#endif
                    }
                    // if tile does not exist, line below means
                    // 'initialize with NULL'
                    m_cache_tiles[idx] = dst_tile;
                    m_cache_keys[idx] = key;
                }
            }
        }

        // Call this and finalize cache before exiting interface function.
        unsigned int finalize_cached_tiles()
        {
            unsigned int updated_cnt = 0;
            PyObject *key;

            for (int idx = 0; idx < MAX_CACHE_COUNT; idx++){

                key = m_cache_keys[idx];
#ifdef HEAVY_DEBUG
                assert(key != NULL);
#endif
                // m_cache_tiles might not have actual tile.be careful.
                if (m_cache_tiles[idx] != NULL) {
                    PyObject *dst_tile = (PyObject*)m_cache_tiles[idx];
                    if (PyDict_SetItem(m_targ_dict, key, dst_tile) == 0) {
                        updated_cnt++;
                    }
                    Py_DECREF(dst_tile);
                    m_cache_tiles[idx] = NULL;
#ifdef HEAVY_DEBUG
                    assert(dst_tile->ob_refcnt >= 1);
#endif
                }

#ifdef HEAVY_DEBUG
                assert(key->ob_refcnt >= 1);
#endif
                Py_DECREF(key);

            }

            m_targ_dict = NULL;
            return updated_cnt;
        }

        // --- Pixel manipulating methods

        inline T* get_pixel(const int cache_index,
                            const unsigned int x,
                            const unsigned int y,
                            bool tile_request)
        {
            PyArrayObject *tile = (PyArrayObject*)_get_cached_tile_from_index(
                                                    cache_index, tile_request);
            if (tile)
                return (T*)get_tile_pixel(tile, x, y);

            return NULL;
        }

        // Get pixel pointer over cached tiles linearly.
        // the origin is center tile(index 4) (0,0),
        // from minimum (-64, -64) to maximum (128, 128).
        //
        // -64 <-    x    -> 128
        // -64    |0|1|2|
        //   y    |3|4|5|   the top-left corner of index4 tile is origin (0,0)
        // 128    |6|7|8|
        //
        // we can access pixels linearly within this range,
        // without indicating tile cache index or tiledict key(tx, ty).
        //
        // this method might generate new tile with 
        // _get_cached_tile_from_index().
        T* get_cached_pixel(int cx, int cy, bool generate)
        {
            int tile_index = CENTER_TILE_INDEX;

            // 'cache coordinate' is converted to tile local one.
            // and split it RELATIVE tiledict key offset(tx, ty).
            if (cx < 0) {
                tile_index--;
                cx += MYPAINT_TILE_SIZE;
            }
            else if (cx >= MYPAINT_TILE_SIZE) {
                tile_index++;
                cx -= MYPAINT_TILE_SIZE;
            }

            if (cy < 0) {
                tile_index -= 3;
                cy += MYPAINT_TILE_SIZE;
            }
            else if (cy >= MYPAINT_TILE_SIZE) {
                tile_index += 3;
                cy -= MYPAINT_TILE_SIZE;
            }

            return get_pixel(tile_index, cx, cy, generate);
        }

        // setting kernel size means 'kernel raster information setup'.
        void set_kernel_size(int size)
        {
#ifdef HEAVY_DEBUG
            assert(size > 0);
            assert(size <= MAX_OPERATION_SIZE);
#endif
            int cw;
            double rad;

            if (m_kern != NULL)
                delete[] m_kern;

            m_kernel_size = size;
            int max_line = size * 2 + 1;
            m_kern = new _kernel_line[max_line];

            for (int dy = 0;dy < size; dy++) {
                rad = asin((double)(size-dy) / (double)size);
                cw = (int)(cos(rad) * (double)size);
                m_kern[dy].offset = size - cw;
                m_kern[dy].length = cw * 2 + 1;
                m_kern[max_line-dy-1].offset = size - cw;
                m_kern[max_line-dy-1].length = cw * 2 + 1;
            }
            // the last center line
            //cw = (int)(cos(asin(0.0)) * (double)size);
            m_kern[size].offset = 0; //size - cw;
            m_kern[size].length = max_line;
        }
};


/**
* @class _Dilation_color
* @brief Morphology operation class for fix15_short_t color pixels.
*
* Dilate from a filled color tile to another new tile.
*
*/

class _Dilation_color : public Tilecache<fix15_short_t, PyArrayObject*> {

    protected:

        fix15_short_t m_fill_r;
        fix15_short_t m_fill_g;
        fix15_short_t m_fill_b;

        virtual bool is_match_pixel(const int cx, const int cy,
                                    PyArrayObject *target_tile)
        {
            // this class only refer the currently targetted tile,
            // so coordinates which are negative or exceeding tile size
            // should be just ignored.
            if (cx < 0 || cy < 0
                || cx >= MYPAINT_TILE_SIZE
                || cy >= MYPAINT_TILE_SIZE) {
               return false;
            }

            fix15_short_t *cur_pixel =
                (fix15_short_t*)get_tile_pixel(
                                    target_tile,
                                    cx, cy);
            return (cur_pixel[3] != 0);
        }

    public:
        _Dilation_color(const double fill_r,
                        const double fill_g,
                        const double fill_b)
            : Tilecache(NPY_UINT16, 4)
        {
            double alpha=(double)fix15_one;
            m_fill_r = (fix15_short_clamp)(fill_r * alpha);
            m_fill_g = (fix15_short_clamp)(fill_g * alpha);
            m_fill_b = (fix15_short_clamp)(fill_b * alpha);
        }

        void dilate(PyObject *py_filled_tile)
        {
            for (int cy=-m_kernel_size;
                     cy < MYPAINT_TILE_SIZE+m_kernel_size;
                     ++cy) {
                for (int cx=-m_kernel_size;
                         cx < MYPAINT_TILE_SIZE+m_kernel_size;
                         ++cx) {

                    fix15_short_t *dst_pixel = get_cached_pixel(cx, cy, false);

                    if ((dst_pixel == NULL || dst_pixel[3] == 0)
                            && _search_kernel(cx, cy,
                                    (PyArrayObject*)py_filled_tile)) {
                        if (dst_pixel == NULL)
                            dst_pixel = get_cached_pixel(cx, cy, true);

                        dst_pixel[0] = m_fill_r;
                        dst_pixel[1] = m_fill_g;
                        dst_pixel[2] = m_fill_b;
                        dst_pixel[3] = (fix15_short_t)fix15_one;
                    }
                }
            }
        }
};


//// Interface functions
/**
* @dilation_init
* initialize dilation
*
* @fill_r, fill_g, fill_b : filling RGB color
* @dilation_size : size of dilation
* @return PyCapsule dilation context object.
* @detail
* This function initialize the _Dilation_color class object.
* call this function before using dilation_process_tile().
*/

PyObject*
dilation_init(
    const double fill_r, const double fill_g, const double fill_b,
    const int dilation_size)
{
#ifdef HEAVY_DEBUG
    assert(dilation_size > 0);
    assert(dilation_size <= MAX_OPERATION_SIZE);
#endif

    _Dilation_color *d = new _Dilation_color(
                            fill_r,
                            fill_g,
                            fill_b);

    d->set_kernel_size(dilation_size);

    PyObject *cap = PyCapsule_New(
                        (void *)d,
                        CAPSULE_NAME,
                        NULL);// does not use python-gc
    Py_INCREF(cap);
    return cap;
}



/* @gapclose_dilate_filled_tile
 *
 * @py_ctx : A PyCapsule object, which contains dilation context.
 * @py_dilated: a Python dictinary, which stores 'dilated color tiles'
 * @py_pre_filled_tile: a Numpy.array object of color tile.
 * @tx, ty: the position of py_pre_filled_tile, in tile coordinate.
 * returns: updated tile counts.
 *
 * @detail
 * Dilate flood-filled tile, and store newly generated tiles into
 * py_dilated dictionary.
 * Before calling this function, we need to call dilation_init once
 * and get py_ctx.
 */

// dilate color filled tile. postprocess of flood_fill.
PyObject*
dilation_process_tile(
    PyObject *py_ctx,
    PyObject *py_dilated,
    PyObject *py_filled_tile,
    const int tx, const int ty)
{
#ifdef HEAVY_DEBUG
    assert(py_ctx != NULL);
    assert(py_dilated != NULL);
    assert(py_filled_tile != NULL);
    assert(PyCapsule_IsValid(py_ctx, CAPSULE_NAME));
#endif
    _Dilation_color *d = (_Dilation_color*)PyCapsule_GetPointer(
                            py_ctx, CAPSULE_NAME);

    // XXX each time we need initialize dilation class.
    d->init_cached_tiles(py_dilated, tx, ty);
    d->dilate(py_filled_tile);
    d->finalize_cached_tiles();

    Py_RETURN_NONE;
}

/**
* @dilation_finalize
* finalize dilation operation.
*
* @py_ctx : A PyCapsule object, which contains dilation context.
* @return : PyNone.
* @detail
* This function delete the dilation object immidiately.
* We need call this function once after tile dilation end.
* With this,the memory(instance) of py_ctx is released.
*/

PyObject*
dilation_finalize(PyObject *py_ctx)
{
#ifdef HEAVY_DEBUG
    assert(py_ctx != NULL);
    assert(PyCapsule_IsValid(py_ctx, CAPSULE_NAME));
#endif
    _Dilation_color *d = (_Dilation_color*)PyCapsule_GetPointer(
                            py_ctx, CAPSULE_NAME);
    delete d;
    Py_DECREF(py_ctx);
#ifdef HEAVY_DEBUG
    // context receiving Python variable added refcnt +1, so 2.
    assert(py_ctx->ob_refcnt == 2);
#endif
    Py_RETURN_NONE;
}

