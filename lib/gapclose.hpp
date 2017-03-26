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

                          

//// Interface functions


// dilate filled tile
PyObject *
gapclose_dilate_filled_tile(
    PyObject *py_dilated, // the tiledict for dilated color tiles.
    PyObject *py_filled_tile, // the filled color src tile. 
    const int tx, const int ty,  // the position of py_filled_tile
    const double fill_r, const double fill_g, const double fill_b, 
    const int dilation_size // dilating pixel radius.
    );

#endif

