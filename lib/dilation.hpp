/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef DILATION_HPP
#define DILATION_HPP

#include <Python.h>

// This Module is for implementing 'Dilating filled area' 

//// Interface functions

// Initialize dilation. 
// Call this before calling dilation_process_tile.
PyObject*
dilation_init(
    const double fill_r, const double fill_g, const double fill_b, 
    const int dilation_size);    

// Dilate filled tile.
PyObject *
dilation_process_tile(
    PyObject *py_ctx, // dilation context, returned from dilation_init().
    PyObject *py_dilated, // the tiledict for dilated color tiles.
    PyObject *py_filled_tile, // the filled color src tile. 
    const int tx, const int ty  // the position of py_filled_tile
    );

// Finalize dilation.
// Call this after dilating operation completed.
PyObject*
dilation_finalize(PyObject *py_ctx);

#endif

