/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "fill_dilate.hpp"
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
#define MAX_CACHE_COUNT 9

#define GET_TILE_INDEX(x, y)  (((x) + 1) + (((y) + 1) * 3))

#define GET_TX_FROM_INDEX(tx, idx) ((tx) + (((idx) % 3) - 1))
#define GET_TY_FROM_INDEX(ty, idx) ((ty) + (((idx) / 3) - 1))

// the _Tile wrapper object surface-cache attribute name.
#define ATTR_NAME_RGBA "rgba"

// Maximum operation size. i.e. 'gap-radius'
#define MAX_OPERATION_SIZE ((MYPAINT_TILE_SIZE / 2) - 1)

inline char *get_tile_pixel(PyArrayObject *tile, const int x, const int y)
{
    // XXX almost copyed from fill.cpp::_floodfill_getpixel()
    const unsigned int xstride = PyArray_STRIDE(tile, 1);
    const unsigned int ystride = PyArray_STRIDE(tile, 0);
    return (PyArray_BYTES((PyArrayObject*)tile)
            + (y * ystride)
            + (x * xstride));
}
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
 *
 * This class is created as template class, for future expansion.
 */

template <class T, class USER_ARG_TYPE>
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

        // --- Tilecache management methods

        PyObject *_generate_cache_key(int tx, int ty)
        {
            PyObject *pytx = PyInt_FromLong(tx);
            PyObject *pyty = PyInt_FromLong(ty);
            PyObject *key = PyTuple_Pack(2, pytx, pyty);
            Py_DECREF(pytx);
            Py_DECREF(pyty);
            return key;
        }

        PyObject *_generate_tile()
        {
            npy_intp dims[] = {MYPAINT_TILE_SIZE, MYPAINT_TILE_SIZE, m_dimension_cnt};
            return PyArray_ZEROS(3, dims, m_NPY_TYPE, 0);
        }

        // Get target(to be put dilated pixels) tile from relative offsets.
        // otx, oty have relative values ranging from -1 to 1,
        // They are relative offsets from center tile. 
        PyObject *_get_cached_tile(int otx, 
                                   int oty, 
                                   bool generate=true) 
        {
            return _get_cached_tile_from_index(
                    GET_TILE_INDEX(otx, oty), 
                    generate
                    );
        }

        PyObject *_get_cached_tile_from_index(int index, bool generate)
        {
            if (m_cache_tiles[index] == NULL) {
                if (generate == true) {
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

        PyObject *_get_cached_key(int index)
        {
            return m_cache_keys[index];
        }

        // Virtual handler method: used from _search_kernel
        virtual bool is_match_pixel(const int cx, const int cy, 
                                    USER_ARG_TYPE user_arg) = 0;

        // Utility method: check pixel existance inside circle-shaped kernel
        // for morphology operation.
        //
        // For morphology operation, we can use same looping logic 
        // to search pixels, but requested result is different for each 
        // operation.
        // For dilation, we need 
        // 'whether is there any valid pixel inside the kernel or not '.
        // On the other hand, for erosion, we need 
        // 'whether is the entire kernel filled with valid pixels or not'.
        // So use 'target_result' parameter to share codes between
        // these different operations.
        inline bool _search_kernel(const int cx, const int cy, 
                                   const int size,
                                   USER_ARG_TYPE user_arg,
                                   const bool target_result)
        {
            // dilate kernel in this class is always square.
            int cw;
            double rad;
            bool result;

            for (int dy = 0; dy <= size; dy++) {
                rad = asin((double)dy / (double)size);
                cw = (int)(cos(rad) * (double)size);
                for (int dx = 0; dx <= cw; dx++) {
                    result = is_match_pixel(cx + dx, cy - dy, user_arg);
                    if (result == target_result) 
                        return true;

                    if (dx != 0 || dy != 0) {
                        result = is_match_pixel(cx + dx, cy + dy, user_arg);
                        if (result == target_result) 
                            return true;

                        result = is_match_pixel(cx - dx, cy - dy, user_arg);
                        if (result == target_result) 
                            return true;

                        result = is_match_pixel(cx - dx, cy + dy, user_arg);
                        if (result == target_result) 
                            return true;
                    }
                }
            }

            return false;
        }

    public:

        Tilecache(const int npy_type, const int dimension_cnt)
        {
            m_NPY_TYPE = npy_type;
            m_dimension_cnt = dimension_cnt;
        }
        virtual ~Tilecache()
        {
        }
        
        // Initialize/finalize cache tile
        // Call this and initialize cache before dilate/erode called.
        void init_cached_tiles(PyObject *py_tiledict, 
                               const int tx, const int ty)
        {
            m_tx = tx;
            m_ty = ty;

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
                        // On the other hand, key object is always generated
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
        unsigned int finalize_cached_tiles(PyObject *py_tiledict)
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
                    if (PyDict_SetItem(py_tiledict, key, dst_tile) == 0) {
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

            return updated_cnt;
        }

        // --- Pixel manipulating methods

        T* get_pixel(const int cache_index,
                     const unsigned int x, 
                     const unsigned int y,
                     const bool generate)
        {
            PyArrayObject *tile = (PyArrayObject*)_get_cached_tile_from_index(
                                                    cache_index, generate);
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
        // this method might generate new tile with _get_cached_tile(). 
        T* get_cached_pixel(const int cx, const int cy, const bool generate)
        {
            int tile_index = CENTER_TILE_INDEX;
            int px = cx; 
            int py = cy;
            
            // 'cache coordinate' is converted to tile local one.
            // and split it RELATIVE tiledict key offset(tx, ty).
            if (cx < 0) {
                tile_index--;
                px += MYPAINT_TILE_SIZE;
            }
            else if (cx >= MYPAINT_TILE_SIZE) {
                tile_index++;
                px -= MYPAINT_TILE_SIZE;
            }

            if (cy < 0) {
                tile_index -= 3;
                py += MYPAINT_TILE_SIZE;
            }
            else if (cy >= MYPAINT_TILE_SIZE) {
                tile_index += 3;
                py -= MYPAINT_TILE_SIZE;
            }
            
            return get_pixel(tile_index, px, py, generate);
        }


        
};




// fix15 dilater class
// This is for ordinary dilating of color pixel tile.
//
class _Dilater_fix15 : public Tilecache<fix15_short_t, PyArrayObject*> {

    protected:

        virtual bool is_match_pixel(const int cx, const int cy, 
                                    PyArrayObject *targ_tile) 
        {
            // This class only refer the currently targetted tile,
            // So coordinates which are negative or exceeding tile size
            // should be just ignored.
            if (cx < 0 || cy < 0 
                || cx >= MYPAINT_TILE_SIZE 
                || cy >= MYPAINT_TILE_SIZE) {
               return false;
            } 

            fix15_short_t *cur_pixel = 
                (fix15_short_t*)get_tile_pixel(
                                    targ_tile,
                                    cx, cy);
            return (cur_pixel[3] != 0);
        }
        
        inline void _put_pixel(const int cx, const int cy, 
                               const fix15_short_t *pixel)
        {
            fix15_short_t *dst_pixel = get_cached_pixel(cx, cy, true);

            if (dst_pixel[3] == 0) { // rejecting already dilated pixel 
                dst_pixel[0] = pixel[0];
                dst_pixel[1] = pixel[1];
                dst_pixel[2] = pixel[2];
                dst_pixel[3] = (fix15_short_t)fix15_one;// Ignore alpha value
            }     
        }

    public:

        _Dilater_fix15() : Tilecache(NPY_UINT16, 4)
        {
        }

        int dilate(PyObject *py_filled_tile, // the filled src tile. 
                   const int tx, const int ty,  // position of py_filled_tile
                   const fix15_short_t *fill_pixel,
                   const int size
                   )
        {

            int dilated_cnt = 0;
            for (int cy=-size; cy < MYPAINT_TILE_SIZE+size; cy++) {
                for (int cx=-size; cx < MYPAINT_TILE_SIZE+size; cx++) {
                    // target pixel is unused.
                    if (_search_kernel(cx, cy, size, 
                            (PyArrayObject*)py_filled_tile, true)) {
                        _put_pixel(cx, cy, fill_pixel);
                        dilated_cnt++;
                    }
                }
            }
            
            return dilated_cnt;
        }

};

// - Python Interface functions.
//

/** tiledilate_dilate_tile:
 *
 * @py_dilated: a Python dictinary, which stores 'dilated color tiles'
 * @py_pre_filled_tile: a Numpy.array object of color tile.
 * @tx, ty: the position of py_pre_filled_tile, in tile coordinate.
 * @fill_r, fill_g, fill_b: premult filling pixel colors.
 * @dilation_size: the dilation size of filled area, in pixel.
 * returns: updated tile counts.
 *
 * dilate flood-filled tile, and store newly generated tiles into 
 * py_dilated dictionary.
 * Dilating operation might generate multiple new tiles 
 * from one flood-filled tile.
 * Such new tiles are stored into py_dilate, and composited other 
 * flood-filled dilation images over and over again.
 * And when the floodfill loop ended,every dilated tiles are composited 
 * to flood-filled tiles.
 * (This composite processing is done in python script)
 * Thus, we can get dilated floodfill image.
 */

PyObject*
tiledilate_dilate_tile(
    PyObject *py_dilated, 
    PyObject *py_filled_tile, 
    const int tx, const int ty,  
    const double fill_r, const double fill_g, const double fill_b, 
    const int dilation_size)    
{
    // Morphology object defined as static, to ease programming. 
    static _Dilater_fix15 d;

#ifdef HEAVY_DEBUG            
    assert(py_dilated != NULL);
    assert(py_filled_tile != NULL);
    assert(py_dilated != Py_None);
    assert(py_filled_tile != Py_None);
    assert(dilation_size <= MAX_OPERATION_SIZE);
#endif

    
    // Actually alpha value is not used currently.
    // for future use.
    double alpha=(double)fix15_one;
    fix15_short_t fill_pixel[3] = {(fix15_short_clamp)(fill_r * alpha),
                                   (fix15_short_clamp)(fill_g * alpha),
                                   (fix15_short_clamp)(fill_b * alpha)};
    
    d.init_cached_tiles(py_dilated, tx, ty);
    d.dilate(py_filled_tile, tx, ty, fill_pixel, dilation_size);
    d.finalize_cached_tiles(py_dilated);

    Py_RETURN_NONE;
}

