/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "gapclose.hpp"
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

// the _Tile wrapper object surface-cache attribute name.
#define ATTR_NAME_RGBA "rgba"

// Maximum operation size. i.e. 'gap-radius'
#define MAX_OPERATION_SIZE ((MYPAINT_TILE_SIZE / 2) - 1)


//// Utility functions

inline char *get_tile_pixel(PyArrayObject *tile, const int x, const int y)
{
    // XXX almost copyed from fill.cpp::_floodfill_getpixel()
    const unsigned int xstride = PyArray_STRIDE(tile, 1);
    const unsigned int ystride = PyArray_STRIDE(tile, 0);
    return (PyArray_BYTES((PyArrayObject*)tile)
            + (y * ystride)
            + (x * xstride));
}

/** gapclose_get_state_flag:
 *
 * @sts_tile : a numpy array of state flag tile.
 * @x, y : position of pixel. it should be 0= < x < MYPAINT_TILE_SIZE
 * returns: an integer value. if sts_tile is invalid, returns 0 
 *
 * Get state flag value of specified position.
 *
 * Actually that state flag tile is array of STATE_PIXEL type
 * (currently it is char), but this returns integer value
 * to share this function with python.
 *
 * This function is NULL safe & Py_None safe.
 */
int gapclose_get_state_flag(PyObject *sts_tile, const int x, const int y)
{
    if(sts_tile && sts_tile != Py_None) {
        return (int)*(get_tile_pixel((PyArrayObject*)sts_tile,x, y));
    }
    return 0;
}

/** gapclose_set_state_flag:
 *
 * @sts_tile : a numpy array of state flag tile.
 * @x, y : position of pixel. it should be 0= < x < MYPAINT_TILE_SIZE
 * @ flag: the flag value to do 'logical OR' with specified pixel.
 * returns: No return value.
 *
 * Set state flag value of specified position.
 * This function do logical OR, so cannot entirely write change
 * the flag.(nealy write once)
 * But it is enough for gap-closing flood-fill state tile. 
 *
 * This function is NULL safe & Py_None safe.
 */
void gapclose_set_state_flag(PyObject *sts_tile, 
                            const int x, const int y, 
                            const char flag)
{
    if(sts_tile && sts_tile != Py_None) {
        char *flagptr = get_tile_pixel((PyArrayObject*)sts_tile, x, y);
        *flagptr |= flag;
    }
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
        // Parameter otx, oty have relative values ranging from -1 to 1,
        // they are relative offsets from center tile.
        PyObject *_get_cached_tile(int otx, int oty, bool generate=true) 
        {
            return _get_cached_tile_from_index(GET_TILE_INDEX(otx, oty), generate);
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
                                    USER_PARAM_TYPE user_param) = 0;

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
                                   const int size,
                                   USER_PARAM_TYPE user_param,
                                   bool target_result)
        {
            int cw;
            double rad;
            bool result;

            for (int dy = 0;dy <= size; dy++) {
                rad = asin((double)(dy-1) / (double)size);
                cw = (int)(cos(rad) * (double)size);
                for (int dx = 0; dx <= cw; dx++) {
                    result = is_match_pixel(cx + dx, cy + dy, user_param);
                    if (result == target_result) 
                        return true;

                    if (dx > 0 || dy > 0) {
                        result = is_match_pixel(cx + dx, cy - dy, user_param);
                        if (result == target_result) 
                            return true;

                        result = is_match_pixel(cx - dx, cy - dy, user_param);
                        if (result == target_result) 
                            return true;

                        result = is_match_pixel(cx - dx, cy + dy, user_param);
                        if (result == target_result) 
                            return true;
                    }
                }
            }
            return false;
        }

    public:

        Tilecache(int npy_type, int dimension_cnt)
        {
            // Tile type and dimension are used in _generate_tile(),
            // when a new tile requested.
            // So only setup them, in constructor.
            m_NPY_TYPE = npy_type;
            m_dimension_cnt = dimension_cnt;
        }
        virtual ~Tilecache()
        {
        }
        
        // Initialize/finalize cache tile
        // Call this and initialize cache before dilate/erode called.
        void init_cached_tiles(PyObject *py_tiledict, int tx, int ty)
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
                     bool generate)
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


        
};


// Morphology operation class for fix15_short_t pixels.
//
// CAUTION:
//   Cached tile is different for each morphology operation.
//   When doing dilation, we set write TARGET tiles into tile-cache.
//   and need only one tile for read-only reference source(flood-filled) tile.
//
//   When doing erosion, we set read-only SOURCE tiles into tile-cache.
//   and need only one tile for write target.
//
class _Dilation_fix15 : public Tilecache<fix15_short_t, PyArrayObject*> {

    protected:

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
        
        inline void _put_pixel(int cx, int cy, const fix15_short_t *pixel)
        {
            fix15_short_t *dst_pixel = get_cached_pixel(cx, cy, true);

            if (dst_pixel[3] == 0) { // rejecting already dilated pixel 
                dst_pixel[0] = pixel[0];
                dst_pixel[1] = pixel[1];
                dst_pixel[2] = pixel[2];
                dst_pixel[3] = (fix15_short_t)fix15_one; 
            }
        }

    public:
        _Dilation_fix15() : Tilecache(NPY_UINT16, 4)
        {
        }

        int dilate(PyObject *py_filled_tile, // the filled src tile. 
                   int tx, int ty,  // the position of py_filled_tile
                   const fix15_short_t *fill_pixel,
                   int size)
        {
            int dilated_cnt = 0;

            for (int cy=-size; cy < MYPAINT_TILE_SIZE+size; cy++) {
                for (int cx=-size; cx < MYPAINT_TILE_SIZE+size; cx++) {
                    // target pixel is unused.
                    if (_search_kernel(cx, cy, size, 
                                       (PyArrayObject*)py_filled_tile, 
                                       true)) {
                        _put_pixel(cx, cy, fill_pixel);
                        dilated_cnt++;
                    }
                }
            }
            
            return dilated_cnt;
        }
};


// _GapCloser : Specialized class for detecting contour, to fill-gap.
// This is for STATE_PIXEL *type,flag_based erode/dilate operation.
//
// in new flood_fill(), we use tiles of 8bit bitflag to figure out 
// original contour, not-filled pixels area(gapclose_EXIST_FLAG) , 
// and dilated area from original contour(gapclose_DILATED_FLAG). 

// STATE_PIXEL type might changed from char. for future expansion.
#define STATE_PIXEL char
#define STATE_PIXEL_NUMPY NPY_UINT8
// when changing STATE_PIXEL type, DO NOT FORGET TO UPDATE
// NUMPY ARRAY TYPE IN CONSTRUCTOR!!

class _GapCloser: public Tilecache<STATE_PIXEL, STATE_PIXEL> {

    protected:
        // pixel state information flag are now defined as 
        // preprocessor constants, to easily share it with other modules.

        inline void _put_flag(const int cx, const int cy, 
                              const STATE_PIXEL flag)
        {
            STATE_PIXEL *dst_pixel = (STATE_PIXEL*)get_cached_pixel(
                                                        cx, cy, true);
            *dst_pixel |= flag;
        }

        virtual bool is_match_pixel(const int cx, const int cy, 
                                    STATE_PIXEL target_pixel) 
        {
            STATE_PIXEL *dst_pixel = (STATE_PIXEL*)get_cached_pixel(
                                                        cx, cy, false);
            return ! (dst_pixel == NULL 
                      || (*dst_pixel & target_pixel) == 0); 
        }

        //// special tile information methods
        //
        // Special informations recorded into a tile with
        // setting INFO_FLAG bitflag to paticular pixel.
        // And they have various means,according to its location. 
        //
        // These methods are accessible from outside this class
        // without any instance, now.
        
        // tile state information flags.
        // This flag is only set to paticular location of tile pixels.
        static const STATE_PIXEL TILE_INFO_FLAG = 0x80;
        static const STATE_PIXEL DILATED_TILE_FLAG = 0x01;
        static const STATE_PIXEL VALID_TILE_FLAG = 0x08;  // valid(exist) tile.

        // Maximum count of Tile info flags.
        // DO NOT FORGET TO UPDATE THIS, when adding above tile flag constant!
        // VALID_TILE_FLAG is not included to this number.
        // currently this is 1, so seems to be meaningless, 
        // but leave for future expanison.
        static const int TILE_INFO_MAX = 1; 

        STATE_PIXEL _get_tile_info(PyArrayObject *tile)
        {
            STATE_PIXEL retflag = 0;
            STATE_PIXEL flag = 1;
            STATE_PIXEL *pixel;

#ifdef HEAVY_DEBUG
            assert(tile != NULL);
#endif
            for(int i=0; i < TILE_INFO_MAX; i++) {
                pixel = get_tile_pixel(tile, 0, i);
                if (*pixel & TILE_INFO_FLAG)
                    retflag |= flag;
                flag = flag << 1;
            }
            return retflag | VALID_TILE_FLAG;
        }

        // Utility method.
        // This can easily follow cached tile when it internally generated
        // inside some method.
        STATE_PIXEL _get_tile_info(int index)
        {
#ifdef HEAVY_DEBUG
            assert(index >= 0);
            assert(index < MAX_CACHE_COUNT);
#endif
            if (m_cache_tiles[index] != NULL)
                return _get_tile_info((PyArrayObject*)m_cache_tiles[index]);
            return 0;
        }

        void _set_tile_info(PyArrayObject *tile, STATE_PIXEL flag)
        {
#ifdef HEAVY_DEBUG
            assert(tile != NULL);
#endif
            STATE_PIXEL *pixel;
            for(int i=0; i < TILE_INFO_MAX && flag != 0; i++) {
                pixel = get_tile_pixel(tile, 0, i);
                if (flag & 0x01)
                    *pixel |= TILE_INFO_FLAG;
                flag = flag >> 1;
            }
        }

        // Utility method.
        // This can easily follow cached tile when it internally generated
        // inside some method.
        void _set_tile_info(int index, STATE_PIXEL flag)
        {
#ifdef HEAVY_DEBUG
            assert(index >= 0);
            assert(index < MAX_CACHE_COUNT);
#endif
            if (m_cache_tiles[index] != NULL)
                _set_tile_info((PyArrayObject*)m_cache_tiles[index], flag);
        }

        // Dilate entire center tile and some area of surrounding 8 tiles, 
        // to ensure center state tile can get complete dilation.
        void _dilate_contour(int gap_radius) 
        {         
#ifdef HEAVY_DEBUG
            assert(gap_radius <= MAX_OPERATION_SIZE);
#endif
            // Tile infomation flag check has done before this method called.

            for (int y=-gap_radius ;
                 y < MYPAINT_TILE_SIZE+gap_radius ;
                 y++) 
            {
                for (int x=-gap_radius ; 
                     x < MYPAINT_TILE_SIZE+gap_radius ; 
                     x++) 
                {
                    STATE_PIXEL *pixel = get_cached_pixel(x, y, false);
                    if (pixel != NULL   
                        && (*pixel & gapclose_DILATED_FLAG) != 0) {
                        // pixel exists, but it already dilated.
                        // = does nothing
                        //
                        // otherwise, pixel is not dilated
                        // or, pixel does not exist(==NULL)
                        // we might place pixel.
                    }
                    else if ( _search_kernel(x, y, 
                                gap_radius, gapclose_EXIST_FLAG, true)) {
                        // If pixel is NULL, the tile that pixel should be contained 
                        // is also NULL.
                        // so generate it, by passing true to get_cached_pixel().
                        if (pixel == NULL)
                            pixel = get_cached_pixel(x, y, true);
            
                        *pixel |= gapclose_DILATED_FLAG;
                    }
                }
            }

            _set_tile_info(CENTER_TILE_INDEX, DILATED_TILE_FLAG);
        }
        
        // Convert(and initialize) color pixel tile into 8bit state tile.
        // state tile updated with 'EXISTED' flag, but not cleared.
        // So exisitng state tile can keep dilated/eroded flags,
        // which is set another call of this function.
        PyObject *_convert_state_tile(PyObject *py_state_dict, // 8bit state dict
                                      PyArrayObject *py_surf_tile, // fix15 color tile
                                      const fix15_short_t *targ_pixel,
                                      const fix15_t tolerance,
                                      PyObject *key) // a tuple of tile location
        {
            

            PyObject *state_tile = PyDict_GetItem(py_state_dict, key);

            if (state_tile == NULL) {
                state_tile = _generate_tile();
                PyDict_SetItem(py_state_dict, key, state_tile);
                // No need to decref tile & key here, 
                // it should be done at _finalize_cached_tiles()
            }

            for (int py = 0; py < MYPAINT_TILE_SIZE; py++) {
                for (int px = 0; px < MYPAINT_TILE_SIZE; px++) {
                    fix15_short_t *cur_pixel = 
                        (fix15_short_t*)get_tile_pixel(py_surf_tile, px, py);
                    if (floodfill_color_match(cur_pixel, 
                                              targ_pixel, 
                                              tolerance) == 0) {
                        STATE_PIXEL *state_pixel = 
                            (STATE_PIXEL*)get_tile_pixel(
                                    (PyArrayObject*)state_tile, px, py);
                        *state_pixel |= gapclose_EXIST_FLAG;
                    }
                }
            }
            
            return state_tile;
        }

        // Before call this method, init_cached_tiles() must be already called.
        // This method converts color tiles around (tx, ty) into state flag tiles.
        // also, this setup function sets newly generated state tile into cache.
        void _setup_state_tiles(PyObject *py_state_dict, // 8bit state tiles dict
                                PyObject *py_surfdict, // source surface tiles dict
                                const fix15_short_t *targ_pixel,
                                const fix15_t tolerance)
        {
            // py_surfdict is surface tile dictionary, it is not cached in this class.
            // this class have cache array of 'state tiles', not surface one.
            // so,extract source color tiles with python API.
#ifdef HEAVY_DEBUG            
            assert(py_surfdict != NULL);
            assert(py_surfdict != Py_None);
#endif

            for (int i=0; i < 9; i++) {

                if (m_cache_tiles[i] == NULL) {
                    PyObject *key = _get_cached_key(i);
                    PyObject *surf_tile = PyDict_GetItem(py_surfdict, key);
                    if (surf_tile != NULL) {
                        Py_INCREF(surf_tile);

                        // The surf_tile might not be tile, but _Tile wrapper object.
                        // If so, extract cached tile from it.
                        // Ensuring color tiles of _Tile wrapper object 
                        // is done at tiledsurface.py
                        int is_tile_obj = PyObject_HasAttrString(
                                            surf_tile, ATTR_NAME_RGBA);
                        if (is_tile_obj != 0) {
                            Py_DECREF(surf_tile);// This surf_tile is _Tile

                            // PyObject_GetAttrString creates new reference.
                            // so no need to INCREF.
                            surf_tile = PyObject_GetAttrString(surf_tile,
                                                               ATTR_NAME_RGBA);
#ifdef HEAVY_DEBUG            
                            assert(surf_tile->ob_refcnt >= 2);
#endif
                        }

                        PyObject *state_tile = 
                            _convert_state_tile(py_state_dict, 
                                                (PyArrayObject*)surf_tile, 
                                                targ_pixel, 
                                                tolerance,
                                                key);
                        Py_DECREF(surf_tile);
#ifdef HEAVY_DEBUG            
                        assert(surf_tile->ob_refcnt >= 1);
#endif
                        m_cache_tiles[i] = state_tile;
                    }
                }
            }

        }

    public:

        _GapCloser() : Tilecache(STATE_PIXEL_NUMPY, 1)
        {
        }

        void close_gap(PyObject *py_state_dict, // 8bit state tiles dict
                       PyObject *py_surfdict, // source surface tiles dict
                       int tx, int ty, // current tile location
                       const fix15_short_t *targ_pixel,// target pixel for conversion
                                                      // from color tile to state tile.
                       fix15_t tolerance,
                       int gap_radius)
        {
#ifdef HEAVY_DEBUG
            assert(gap_radius <= MAX_OPERATION_SIZE);
#endif

            init_cached_tiles(py_state_dict, tx, ty); 

            STATE_PIXEL tile_info = _get_tile_info(CENTER_TILE_INDEX);
            if ((tile_info & DILATED_TILE_FLAG) == 0) {

                _setup_state_tiles(py_state_dict, 
                                   py_surfdict, 
                                   targ_pixel,
                                   tolerance);
            
                // Filling gap with dilated contour
                // (contour = not flood-fill targeted pixel area).
                _dilate_contour(gap_radius);

            }

            finalize_cached_tiles(py_state_dict); 
        }
        
};

//// Python Interface functions.
//

/** gapclose_close_gap:
 *
 * @py_state_dict: a Python dictinary, which stores 'state flag tiles'
 * @py_pre_filled_tile: a Numpy.array object of color tile.
 *                      This is the source of flag tile.
 * @tx, ty: the position of py_pre_filled_tile, in tile coordinate.
 * @targ_r, targ_g, targ_b, targ_a: premult target pixel color
 * @tol: tolerance,[0.0 - 1.0] same as tile_flood_fill().
 * @gap_radius: the filling gap radius.
 * returns: Nothing. returning PyNone always.
 *
 * extract contour into state tiles, 
 * and that state tile is stored into py_state_dict, with key of (tx, ty).
 * And later, this state tile used in flood_fill function, to detect
 * ignorable gaps.
 */

PyObject *
gapclose_close_gap(
    PyObject *py_state_dict, 
    PyObject *py_surfdict, 
    const int tx, const int ty,  
    const int targ_r, const int targ_g, const int targ_b, const int targ_a, 
    const double tol,   
    const int gap_radius) 
{
#ifdef HEAVY_DEBUG            
    assert(py_state_dict != NULL);
    assert(py_surfdict != NULL);
    assert(0.0 <= tol);
    assert(tol <= 1.0);
    assert(gap_radius <= MAX_OPERATION_SIZE);
#endif
    // actually, this function is wrapper.
    
    // XXX Morphology Operation object defined as static. 
    // Because this function called each time before a tile flood-filled,
    // I think constructing/destructing cost cannot be ignored.
    // Otherwise, we can use something start/end wrapper function and
    // use PyCapsule object...
    static _GapCloser m;


    fix15_short_t targ_pixel[4] = {(fix15_short_t)targ_r,
                                   (fix15_short_t)targ_g,
                                   (fix15_short_t)targ_b,
                                   (fix15_short_t)targ_a};

    // XXX Copied from fill.cpp tile_flood_fill()
    const fix15_t tolerance = (fix15_t)(  MIN(1.0, MAX(0.0, tol))
                                        * fix15_one);

    m.close_gap(py_state_dict, py_surfdict,
                tx, ty, 
                (const fix15_short_t*)targ_pixel,
                tolerance,
                gap_radius);

    Py_RETURN_NONE;// DO NOT FORGET THIS!
      
}

/** gapclose_dilate_filled_tile:
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
 * Usually, dilating operation would mutiple new tiles from one 
 * flood-filled tile.
 * They are stored into py_dilate, and composited other flood-filled 
 * dilation images over and over again.
 * And when the floodfill loop ended,every dilated tiles are composited 
 * to flood-filled tiles.
 * (This composite processing is done in python script)
 * Thus, we can get dilated floodfill image.
 */

// dilate color filled tile. postprocess of flood_fill.
PyObject*
gapclose_dilate_filled_tile(
    PyObject *py_dilated, 
    PyObject *py_filled_tile, 
    const int tx, const int ty,  
    const double fill_r, const double fill_g, const double fill_b, 
    const int dilation_size)    
{
    // Morphology object defined as static. 
    // Because this function called each time before a tile flood-filled,
    // so constructing/destructing cost would not be ignored.
    static _Dilation_fix15 d;

#ifdef HEAVY_DEBUG            
    assert(py_dilated != NULL);
    assert(py_filled_tile != NULL);
    assert(dilation_size <= MAX_OPERATION_SIZE);
#endif

    
    // Actually alpha value is not used currently.
    // for future use.
    double alpha=(double)fix15_one;
    fix15_short_t fill_pixel[3] = {(fix15_short_clamp)(fill_r * alpha),
                                   (fix15_short_clamp)(fill_g * alpha),
                                   (fix15_short_clamp)(fill_b * alpha)};
    
    // _Dilation_fix15 class is specialized dilating filled pixels,
    // and uses 'current pixel' for dilation, not fixed/assigned pixel.
    // Therefore this class does not need internal pixel member,
    // dummy pixel is passed to setup_morphology_params().

    d.init_cached_tiles(py_dilated, tx, ty);
    d.dilate(py_filled_tile, tx, ty, fill_pixel, dilation_size);
    d.finalize_cached_tiles(py_dilated);

    Py_RETURN_NONE;
}

/** gapclose_search_start_point:
 *
 * @py_flag_tile : a numpy array of state flag tile.
 * @min_x, min_y, max_x, max_y : search limitation
 * returns: 
 * When new starting point found, it is tuple of (x, y) i.e. length is 2. 
 * if it is not found, return seed sequence, i.e. length is 4.
 * elements of seed sequence might be empty, if no further searching required.
 *
 * This function searches valid stating point of flood-fill.
 * And called when starting point of flood-fill is inside 'contour'.
 * This function is for not only starting flood-fill, but also keeping
 * more correct shape of gap-closing flood-fill pixels, because the
 * gap-closing filled area would be dilated later.
 *
 */

// A point in the search queue
typedef struct {
    unsigned int x;
    unsigned int y;
    int search_cnt;
} _search_point;

PyObject*
gapclose_search_start_point(
    PyObject *py_flag_tile, // the flag tile.
    PyObject *seeds,
    int min_x, int min_y, 
    int max_x, int max_y)
{
    // XXX same algorithm with fill.cpp::tile_flood_fill
    // and copied many codes from it.

#ifdef HEAVY_DEBUG
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
    int search_cnt = 0;

    GQueue *queue = g_queue_new();   /* Of tuples, to be exhausted */
    for (int i=0; i<PySequence_Size(seeds); ++i) {
        PyObject *seed_tup = PySequence_GetItem(seeds, i);
#ifdef HEAVY_DEBUG
        assert(PySequence_Size(seed_tup) == 3);
#endif
        if (! PyArg_ParseTuple(seed_tup, "iii", 
                               &x, &y, &search_cnt)) {
            continue;
        }
        Py_DECREF(seed_tup);

        if (search_cnt > 0) {
            x = MAX(0, MIN(x, MYPAINT_TILE_SIZE-1));
            y = MAX(0, MIN(y, MYPAINT_TILE_SIZE-1));
            STATE_PIXEL state_flag = gapclose_get_state_flag(
                                        py_flag_tile, x, y);
            if ((state_flag & gapclose_DILATED_FLAG) != 0) {
                _search_point *seed_pt = (_search_point*)
                                           malloc(sizeof(_search_point));
                seed_pt->x = x;
                seed_pt->y = y;
                seed_pt->search_cnt = search_cnt;
                g_queue_push_tail(queue, seed_pt);
            }
        }
    }

    PyObject *result_n = PyList_New(0);
    PyObject *result_e = PyList_New(0);
    PyObject *result_s = PyList_New(0);
    PyObject *result_w = PyList_New(0);

    int found_px = -1;
    int found_py = -1;

    while (! g_queue_is_empty(queue) && found_px==-1) {
        _search_point *pos = (_search_point*) g_queue_pop_head(queue);
        int x0 = pos->x;
        int y = pos->y;
        int search_cnt_base = pos->search_cnt;
        free(pos);

        // Find easternmost and westernmost points of the same colour
        // Westwards loop includes (x,y), eastwards ignores it.
        static const int x_delta[] = {-1, 1};
        static const int x_offset[] = {0, 1};
        for (int i=0; 
             i<2 && found_px==-1 && search_cnt_base>0 ; 
             ++i)
        {
            bool look_above = true;
            bool look_below = true;
            int search_cnt = search_cnt_base;
            for ( int x = x0 + x_offset[i] ;
                  x >= min_x && x <= max_x && search_cnt>0 ;
                  x += x_delta[i] )
            {
                if (x != x0) { // Test was already done for queued pixels
                    int state_flag = gapclose_get_state_flag(
                                        py_flag_tile, x, y);

                    if ((state_flag & gapclose_EXIST_FLAG) != 0) {
                        break;
                    }
                    else if ((state_flag & gapclose_DILATED_FLAG) == 0) {
                        // We reach "normal" pixel!
                        found_px = x;
                        found_py = y;
                        break;
                    } 
                    else {
                        search_cnt--;
                        if (search_cnt <= 0) 
                            break;
                    }
                }
                    
                // Also halt if we're outside the bbox range
                if (x < min_x || y < min_y || x > max_x || y > max_y) {
                    break;
                }
                
                // In addition, enqueue the pixels above and below.
                // Scanline algorithm here to avoid some pointless queue faff.
                if (y > 0) {
                    int state_flag_above = gapclose_get_state_flag(
                                            py_flag_tile, x, y-1);
                    if ((state_flag_above & gapclose_EXIST_FLAG) == 0) {
                        if ((state_flag_above & gapclose_DILATED_FLAG) != 0) {
                            if (look_above) {
                                // Enqueue the pixel to the north
                                _search_point *p = (_search_point *) malloc(
                                                        sizeof(_search_point)
                                                      );
                                p->x = x;
                                p->y = y-1;
                                p->search_cnt = search_cnt-1;
                                g_queue_push_tail(queue, p);
                                look_above = false;
                            }
                        }
                        else {
                            found_px = x;
                            found_py = y-1;
                            break;
                        }
                    }
                    else { // !match_above, but found!!
                        look_above = true;
                    }
                }
                else {
                    // Overflow onto the tile to the North.
                    // Scanlining not possible here: pixel is over the border.
                    PyObject *s = Py_BuildValue("iii", 
                                                x, MYPAINT_TILE_SIZE-1, 
                                                search_cnt-1);
                    PyList_Append(result_n, s);
                    Py_DECREF(s);
#ifdef HEAVY_DEBUG
                    assert(s->ob_refcnt == 1);
#endif
                }
                if (y < MYPAINT_TILE_SIZE - 1) {
                    int state_flag_below = gapclose_get_state_flag(
                                            py_flag_tile, x, y+1);
                    if ((state_flag_below & gapclose_EXIST_FLAG) == 0)
                    { 
                        if ((state_flag_below & gapclose_DILATED_FLAG) != 0) {
                            if (look_below) {
                                // Enqueue the pixel to the South
                                _search_point *p = (_search_point *) malloc(
                                                        sizeof(_search_point)
                                                      );
                                p->x = x;
                                p->y = y+1;
                                p->search_cnt = search_cnt-1;
                                g_queue_push_tail(queue, p);
                                look_below = false;
                            }
                        }
                        else {
                            found_px = x;
                            found_py = y+1;
                            break;
                        }
                    }
                    else { //!match_below
                        look_below = true;
                    }
                }
                else {
                    // Overflow onto the tile to the South
                    // Scanlining not possible here: pixel is over the border.
                    PyObject *s = Py_BuildValue("iii", 
                                                x, 0, 
                                                search_cnt-1);
                    PyList_Append(result_s, s);
                    Py_DECREF(s);
#ifdef HEAVY_DEBUG
                    assert(s->ob_refcnt == 1);
#endif
                }
                // If the fill is now at the west or east extreme, we have
                // overflowed there too.  Seed West and East tiles.
                if (x == 0) {
                    PyObject *s = Py_BuildValue("iii", 
                                                MYPAINT_TILE_SIZE-1, y, 
                                                search_cnt-1);
                    PyList_Append(result_w, s);
                    Py_DECREF(s);
#ifdef HEAVY_DEBUG
                    assert(s->ob_refcnt == 1);
#endif
                }
                else if (x == MYPAINT_TILE_SIZE-1) {
                    PyObject *s = Py_BuildValue("iii", 
                                                0, y,
                                                search_cnt-1);
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
    
    // Different from tile_flood_fill(), this function might exit loop
    // before all queues are consumed. so free them all here.
    while (! g_queue_is_empty(queue)) {
        _search_point *pos = (_search_point*) g_queue_pop_head(queue);
        free(pos);
    }
    g_queue_free(queue);

    PyObject *result; 
    if ( found_px != -1 ) {
        result = Py_BuildValue("[ii]", found_px, found_py);
        Py_DECREF(result_n);
        Py_DECREF(result_e);
        Py_DECREF(result_s);
        Py_DECREF(result_w);
    } 
    else {
        result = Py_BuildValue("[OOOO]", result_n, result_e,
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
    }
    return result;
}

