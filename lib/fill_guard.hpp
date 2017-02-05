/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef DILATE_HPP
#define DILATE_HPP

#include <Python.h>

// This Module is for implementing 'Dilating filled area' and 
// 'overflow prevention' functionality for flood_fill (fill.cpp)
//
// This module has the class MorphologyBase, and its deriving class
// _Dilater_fix15 (for dilating filled area)
// _Morphology_contour (for detecting contour, to prevent overflow)
// in .cpp file.

// I'm going to use numpy PyArray Object here, but I met some 
// problems here:
//
// 1. When I define NO_IMPORT_ARRAY, then I get importerror from
//    Python interpreter.
// 2. When I removed NO_IMPORT_ARRAY, then I get segmentation fault
//    at calling PyArray_ZEROS().
//
// Anyway, I confirmed SWIG wrapper .cxx surely call import_array().
//
// so I rewrote entire this file, not to use PyArrayObject*.
// Instead of it, I use PyObject*.
// and cast it in dilate.cpp
//
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#define NO_IMPORT_ARRAY
//#include <numpy/arrayobject.h>

#include <mypaint-tiled-surface.h>

// Constant macro define, to share constants between C++ module.
// I heard that these constants should be 'static const' member
// in C++, but local class member cannot access from other file modules.
//  

#define ERODED_FLAG 0x04


// Base class for Morphology operation and tile cache management.
// We need tile cache , to speed up accessing tile pixels with
// linear coordinate.

class MorphologyBase
{
    protected:
        static PyObject* s_cache_tiles[9];
        static PyObject* s_cache_keys[9];

        // parameters for on-demand cache tile generation, 
        // assign to Numpy-API.
        int m_NPY_TYPE;
        int m_dimension_cnt;

        int m_tx; // current center tile location, in tiledict.
        int m_ty; 

        // - Tilecache management methods

        PyObject* _generate_cache_key(int tx, int ty);

        PyObject* _generate_tile();

        // Get target(to be put dilated pixels) tile from relative offsets.
        PyObject* _get_cached_tile(int otx, // otx, oty have relative values ranging from -1 to 1,
                                   int oty, // they are relative offsets from center tile.
                                   bool generate=true); 
        PyObject* _get_cached_tile_from_index(int index, bool generate);

        PyObject* _get_cached_key(int index);

        // initialize/finalize cache tile
        void _init_cached_tiles(PyObject* py_tiledict, int tx, int ty);
        unsigned int _finalize_cached_tiles(PyObject* py_tiledict);

        // - Morphology operation methods & members
        
        unsigned int m_size;       // size of morphology.
        const char* m_source_pixel;// source pixel, morphology operation methods
                                   // put this pixel into target tile. 
                                  
        // These values initialized with _setup_morphology_params
        // and used for pixel manipulation of morphology methods.
        // These values are mostly unchanged during put_*_*_kernel method,
        // So I think passing them as parameter is inefficient.
        

        char* _get_pixel(PyObject* array,
                         const unsigned int x, 
                         const unsigned int y);

        // get pixel pointer over cached tiles linearly.
        // the origin is center tile (0,0),
        // from minimum (-64, -64) to maximum (128, 128).
        // we can access pixels linearly within this range, 
        // without indicating tile cache index or tiledict key(tx, ty). 
        //
        // this method might generate new tile with _get_cached_tile(). 
        inline char* _get_cached_pixel(int cx, int cy)
        {
            int otx = 0;
            int oty = 0;
            
            // 'cache coordinate' is converted to tile local one.
            // and split it RELATIVE tiledict key offset(otx, oty).
            if (cx < 0) {
                otx = -1;
                cx += MYPAINT_TILE_SIZE;
            }
            else if (cx >= MYPAINT_TILE_SIZE) {
                otx = 1;
                cx -= MYPAINT_TILE_SIZE;
            }

            if (cy < 0) {
                oty = -1;
                cy += MYPAINT_TILE_SIZE;
            }
            else if (cy >= MYPAINT_TILE_SIZE) {
                oty = 1;
                cy -= MYPAINT_TILE_SIZE;
            }
            
            PyObject* dst_tile = _get_cached_tile(otx, oty);
            return _get_pixel(dst_tile, cx, cy); 
        }

        virtual void _put_pixel(char* dst_pixel, const char* src_pixel)
        {
            *dst_pixel = *src_pixel;
        }
        virtual void _put_pixel(PyObject* dst_tile, int x, int y,
                               const char* pixel)
        {// Utility method.
            _put_pixel(_get_pixel(dst_tile, x, y), pixel);
        }

        // XXX Pure virtual functions - child class must implement them.
        // but ... they are slow. 
        virtual bool _is_dilate_target_pixel(const char* pixel) = 0;
        virtual bool _is_erode_target_pixel(const char* pixel) = 0;
        
        
        // CAUTION: 'put_*_*_kernel()' and related methods need for 
        // its parameter as entire 'cache matrix coordinate' = (cx, cy),
        // NOT tile-local coordinate !! 
        // 'cache matrix coordinate' is a coordinate system with (x=0, y=0) of 
        // s_cache_tiles[CENTER_TILE_INDEX] as origin.
        //           
        // -64 <-    x    -> 128
        // -64    |0|1|2| 
        //   y    |3|4|5|
        // 128    |6|7|8|
        //    
        // For example, the index 0 tile of (x=10, y=20) is, (cx=-54, cy=-44)
        // On the other hand,index 5 tile of (x=10, y=20) is (cx=74, cy=20)
        // See the implement of _get_cached_pixel()
        //
        // With this coordinate system, we can handle each tile pixels with linear
        // coordinate, not divided one.

        inline void _put_dilate_pixel(int cx, int cy)
        {
            char* dst_pixel = _get_cached_pixel(cx, cy);
            _put_pixel(dst_pixel, m_source_pixel);
        }

        // - morphology methods

        inline void _put_dilate_pixel_batch(int cx, int cy, int dx, int dy) 
        {
            _put_dilate_pixel(cx-dx, cy-dy);
            _put_dilate_pixel(cx+dx, cy-dy);
            _put_dilate_pixel(cx-dx, cy+dy);
            _put_dilate_pixel(cx+dx, cy+dy);
        }



        inline void _put_dilate_diamond_kernel(int cx, int cy) 
        {
            for (int dy = 0; dy < m_size; dy++) {
                for (int dx = 0; dx < m_size-dy; dx++) {
                    if (dx != 0 || dy != 0) {
                        // The fitst(center) pixel should be already there.   
                        _put_dilate_pixel_batch(cx, cy, dx, dy);
                    }
                }
            }
        }

        inline void _put_dilate_square_kernel(int cx, int cy) 
        {
            for (int dy = 0; dy < m_size; dy++) {
                for (int dx = 0; dx < m_size; dx++) {
                    if (dx != 0 || dy != 0) {
                        // The first(center) pixel shold be already there.
                        _put_dilate_pixel_batch(cx, cy, dx, dy);
                    }
                }
            }
        }

        inline int _is_erodable(int cx, int cy, int dx, int dy)
        {
            if (!_is_erode_target_pixel(_get_cached_pixel(cx-dx, cy-dy)))
                return 0;
            else if (!_is_erode_target_pixel(_get_cached_pixel(cx+dx, cy-dy)))
                return 0;
            else if (!_is_erode_target_pixel(_get_cached_pixel(cx-dx, cy+dy)))
                return 0;
            else if (!_is_erode_target_pixel(_get_cached_pixel(cx+dx, cy+dy)))
                return 0;
            return 1;
        }

        inline int _put_erode_diamond_kernel(PyObject* target_tile,
                                             int cx, int cy) 
        {
            for (int dy = 0; dy < m_size; dy++) {
                for (int dx = 0; dx < m_size - dy; dx++) {
                    if (_is_erodable(cx, cy, dx, dy) == 0)
                        return 0;
                }
            }

            char* dst_pixel = _get_pixel(target_tile, cx, cy);
            _put_pixel(dst_pixel, m_source_pixel);
            return 1;
        }

        inline int _put_erode_square_kernel(PyObject* target_tile,
                                            int cx, int cy) 
        {

            for (int dy = 0; dy < m_size; dy++) {
                for (int dx = 0; dx < m_size; dx++) {
                    if (_is_erodable(cx, cy, dx, dy) == 0)
                        return 0;
                }
            }

            char* dst_pixel = _get_pixel(target_tile, cx, cy);
            _put_pixel(dst_pixel, m_source_pixel);
            return 1;
        }

    public:

        static const int SQUARE_KERNEL = 0;
        static const int DIAMOND_KERNEL = 1;

        MorphologyBase(int npy_type, int dimension_cnt);
        ~MorphologyBase();

        // call this setup method before morphology operation starts.
        void setup_morphology_params(int size, const char* source_pixel);

        // morphology operation interface methods.
        unsigned int dilate(PyObject* py_dilated, // the tiledict for dilated tiles.
                            PyObject* py_filled_tile, // the filled src tile. 
                            int tx, int ty,  // the position of py_filled_tile
                            int kernel_type  // 0 for square kernel, 1 for diamond kernel
                            );

        unsigned int erode(PyObject* py_dilated, // the tiledict for dilated tiles.
                           PyObject* py_target_tile, // the tile to be drawn eroded pixel. 
                           int tx, int ty,  // the position of py_filled_tile
                           int kernel_type   // 0 for square kernel, 1 for diamond kernel
                           );

};



// # Interface functions

// setup state tile, to detect fillable gap.
PyObject* detect_contour(PyObject* py_statedict, // the tiledict for dilated tiles.
                         PyObject* py_surfdict, //  source surface tile dict.
                         int tx, int ty,  // the position of py_filled_tile
                         int targ_r, int targ_g, int targ_b, int targ_a, //premult target pixel color
                         double tol,
                         int gap_size    // fillable gap size.
                         );

// dilate filled tile
unsigned int dilate_filled_tile(PyObject* py_dilated, // the tiledict for dilated tiles.
                                PyObject* py_filled_tile, // the filled src tile. 
                                int tx, int ty,  // the position of py_filled_tile
                                int grow_size,    // growing size from center pixel.
                                int kernel_type  // 0 for square kernel, 1 for diamond kernel
                               );

#endif

