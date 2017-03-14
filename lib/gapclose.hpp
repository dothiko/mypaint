/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef GAPCLOSE_HPP
#define GAPCLOSE_HPP

#include <Python.h>

// This Module is for implementing 'Dilating filled area' and 
// 'overflow prevention' functionality for flood_fill (fill.cpp)
// This module uses 'morphology.hpp' internally.

#include <mypaint-tiled-surface.h>

// These constants are also referred from fill.cpp
// So, they are not static const class member, preprocessor constant.

#define gapclose_EXIST_FLAG   0x01
#define gapclose_DILATED_FLAG 0x02 // This means the pixel is just dilated pixel,
                                   // not sure original source pixel.
                                   // it might be empty pixel in source tile.
                          
//// Utility functions for c++ modules

// get flag of state tile pixel. 
// this function is NULL-safe, if tile is NULL, just return 0. 
int gapclose_get_state_flag(PyObject *sts_tile, const int x, const int y);

// set flag of state tile pixel.
// this function is NULL-safe, if tile is NULL, nothing done. 
// 'set' means bitwise OR operation.
void gapclose_set_state_flag(PyObject *sts_tile, 
                     const int x, const int y, 
                     const char flag);

//// Interface functions

// setup gap-closing state flag tile 
PyObject *
gapclose_close_gap(
    PyObject *py_state_dict, // the tiledict for state flag tiles.
    PyObject *py_surfdict, //  source surface tile dict.
    const int tx, const int ty,  // the position of py_filled_tile
    const int targ_r, const int targ_g, const int targ_b, const int targ_a, 
    const double tol,   // pixel tolerance of filled area.
    const int gap_size  // overflow-preventing closable gap size.
    );


// dilate filled tile
PyObject *
gapclose_dilate_filled_tile(
    PyObject *py_dilated, // the tiledict for dilated color tiles.
    PyObject *py_filled_tile, // the filled color src tile. 
    const int tx, const int ty,  // the position of py_filled_tile
    const double fill_r, const double fill_g, const double fill_b, 
    const int dilation_size // dilating pixel radius.
    );

PyObject *
gapclose_search_start_point(
    PyObject *py_flag_tile, // the flag tiles.
    PyObject *seeds,
    int min_x, int min_y, 
    int max_x, int max_y);

#endif

