/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef FILL_DILATE_HPP
#define FILL_DILATE_HPP

#include <Python.h>

// # Interface functions

// dilate filled tile
PyObject *tiledilate_dilate_tile(
    PyObject *py_dilated, // the tiledict for dilated tiles.
    PyObject *py_filled_tile, // the filled src tile. 
    const int tx, const int ty,  
    const double fill_r, const double fill_g, const double fill_b, 
    const int dilation_size    
    );

#endif

