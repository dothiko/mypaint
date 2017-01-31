/* This file is part of MyPaint.
 * Copyright (C) 2017 by Dothiko <dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "fill.hpp"
#include "fill_grow.hpp"

#include "common.hpp"
#include "fix15.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <glib.h>
#include <mypaint-tiled-surface.h>

#define SQUARE_KERNEL 0
#define DIAMOND_KERNEL 1

static PyArrayObject* s_cache_tiles[9];// numpy tile cache, to avoid frequent python object generation.
static PyObject* s_cache_keys[9]; // key-tuple cache, for dilated tiles dictionary.

static PyObject* _get_dilated_key(int tx, int ty)
{
    PyObject* txo = PyInt_FromLong(tx);
    PyObject* tyo = PyInt_FromLong(ty);
    PyObject* key = PyTuple_Pack(2, txo, tyo);
    Py_DECREF(txo);
    Py_DECREF(tyo);
    return key;
}

// Get target(to be put dilated pixels) tile from offsets.

static inline PyArrayObject* _get_dilated_tile(int tx, // tx, ty have values ranging from -1 to 1,
                                        int ty) // it is offsets from center tile.
{
    int idx = (tx + 1) + (ty + 1) * 3;
    if(s_cache_tiles[idx] == NULL)
    {
        npy_intp dims[] = {MYPAINT_TILE_SIZE, MYPAINT_TILE_SIZE, 4};
        PyArrayObject* dst_tile = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_UINT16, 0);
        s_cache_tiles[idx] = dst_tile;
        return dst_tile;
    }
    return s_cache_tiles[idx];
}


static void _init_dilated_tiles(PyObject* py_dilated, int tx, int ty){
    // fill the cache array with current dilated dict.
    for(int y=-1; y <= 1; y++)
    {
        for(int x=-1; x <= 1; x++)
        {
            int idx = (x + 1) + (y + 1) * 3;
            PyObject* key = _get_dilated_key(tx+x, ty+y);
            PyArrayObject* dst_tile = (PyArrayObject*)PyDict_GetItem(py_dilated, key);
            if(dst_tile)
            {
                Py_INCREF(dst_tile); 
                // In this case, tile is just borrowed. so incref.
                // Otherwise, some tile would be generated in this module.
                // Such 'new tile' would be 'increfed' when doing PyDict_SetItem() at
                // _finalize_dilated_tiles().
                // On the other hand, A  Borrowed tile, it is already in dilated 
                // dictionary,it would not get increfed even PyDict_SetItem().
                // Therefore we need incref it here.
            }
            s_cache_tiles[idx] = dst_tile; // if tile does not exist, this means 'initialize with NULL'
            s_cache_keys[idx] = key;
        }
    }
}

static unsigned int _finalize_dilated_tiles(PyObject* py_dilated){

    unsigned int updated_cnt = 0;
    PyObject* key;

    for(int idx = 0; idx < 9; idx++){

        key = s_cache_keys[idx];

        // s_cache_tiles might not have actual tile.be careful.
        if(s_cache_tiles[idx] != NULL){
            PyObject* dst_tile = (PyObject*)s_cache_tiles[idx];
            if (PyDict_SetItem(py_dilated, key, dst_tile) != 0){
                // XXX FAILED TO SET ITEM!!! ... what should we do here?
            }else{
                updated_cnt++;
            }
            Py_DECREF(dst_tile);
            s_cache_tiles[idx] = NULL;
        }

        // Different from s_cache_tiles,
        // Each s_cache_keys elements should be valid python object.
        // so every 'key' MUST BE DONE Py_DECREF.
        Py_DECREF(key);
        s_cache_keys[idx] = NULL;
    }

    return updated_cnt;
}

// XXX Copied from lib/fill.cpp (or, entire this file should be added in fill.cpp?)
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


static inline void _put_dilate_pixel(int x, int y, const fix15_short_t* pixel)
{
    int tx = 0;
    int ty = 0;

    if (x < 0){
        tx = -1;
        x += MYPAINT_TILE_SIZE;
    }else if(x >= MYPAINT_TILE_SIZE){
        tx = 1;
        x -= MYPAINT_TILE_SIZE;
    }

    if (y < 0){
        ty = -1;
        y += MYPAINT_TILE_SIZE;
    }else if(y >= MYPAINT_TILE_SIZE){
        ty = 1;
        y -= MYPAINT_TILE_SIZE;
    }
    
    PyArrayObject* dst_tile = _get_dilated_tile(tx, ty);
    fix15_short_t* dst_pixel = _floodfill_getpixel(dst_tile, x, y);
    
    if (dst_pixel[3] == 0){
    // if zero, it is not written yet.
         dst_pixel[0] = pixel[0];
         dst_pixel[1] = pixel[1];
         dst_pixel[2] = pixel[2];
         dst_pixel[3] = pixel[3];
    }

}


static inline void _put_dilate_diamond_kernel(int x, int y, int grow_size, 
                                              const fix15_short_t* pixel)
{
    for(int dy = 0; dy < grow_size; dy++){
        for(int dx = 0; dx < grow_size - dy; dx++){
            if (dx != 0 || dy != 0) // The fitst(center) pixel should be already there.   
            {
                _put_dilate_pixel(x-dx, y-dy, pixel);
                _put_dilate_pixel(x+dx, y-dy, pixel);
                _put_dilate_pixel(x-dx, y+dy, pixel);
                _put_dilate_pixel(x+dx, y+dy, pixel);
            }
        }
    }
}

static inline void _put_dilate_square_kernel(int x, int y, int grow_size, 
                                             const fix15_short_t* pixel)
{
    for(int dy = 0; dy < grow_size; dy++){
        for(int dx = 0; dx < grow_size; dx++){
            if (dx != 0 || dy != 0) // The first(center) pixel shold be already there.
            {
                _put_dilate_pixel(x-dx, y-dy, pixel);
                _put_dilate_pixel(x+dx, y-dy, pixel);
                _put_dilate_pixel(x-dx, y+dy, pixel);
                _put_dilate_pixel(x+dx, y+dy, pixel);
            }
        }
    }
}
      

// Main function, dilating filled tile implementation
//
// return value is count of updated tiles.
unsigned int dilate_filled_tile(PyObject* py_dilated, // the tiledict for dilated tiles.
                                PyObject* py_filled_tile, // the filled src tile. 
                                int tx, int ty,  // the position of py_filled_tile
                                int grow_size,    // growing size from center pixel.
                                int kernel_type  // 0 for square kernel, 1 for diamond kernel
                               )
{
    
    
    // to failsafe, limit grow size within MYPAINT_TILE_SIZE
    grow_size %= MYPAINT_TILE_SIZE;

    // Initialize 3x3 tile cache.
    // This cache is to avoid generating tuple object
    // for each time we refer to neighbor tiles.
     _init_dilated_tiles(py_dilated, tx, ty); 

    
    for(int y=0; y < MYPAINT_TILE_SIZE; y++)
    {
        for(int x=0; x < MYPAINT_TILE_SIZE; x++)
        {
            fix15_short_t *src_pixel = _floodfill_getpixel((PyArrayObject*)py_filled_tile, x, y);
            if(src_pixel[3] != 0)
            {
                switch(kernel_type)
                {
                    case DIAMOND_KERNEL:
                        _put_dilate_diamond_kernel(x, y, 
                                                   grow_size, 
                                                   src_pixel);
                        break;

                    case SQUARE_KERNEL:
                    default:
                        _put_dilate_square_kernel(x, y, 
                                                  grow_size, 
                                                  src_pixel);
                        break;
                }
            }
        }
    }
    

    // finally, make tiles cache into python directory object.
    // and dereference them.
    return _finalize_dilated_tiles(py_dilated);
}

