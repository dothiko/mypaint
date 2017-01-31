/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko <dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef FILL_GROW_HPP
#define FILL_GROW_HPP

#include <Python.h>

// dialate flood-filled tiles as the rectangular kernel.
// dialated pixels should be written in 'target_tile', 
// which is separated from flood-filled layer.
//
// Returns a count of written pixel(s). Therefore the return value is
// greater than zero, it means 'target_tile written'
// The caller python function refer this value, and generate or recycle
// a tile for next loop when target_tile has been modified.

unsigned int dilate_filled_tile(PyObject* py_dilated, // the tiledict for dilated tiles.
                                PyObject* py_filled_tile, // the filled src tile. 
                                int tx, int ty,  // the position of py_filled_tile
                                int grow_size,    // growing size from center pixel.
                                int kernel_type
                               );

#endif //__HAVE_FILL_GROW_HPP

