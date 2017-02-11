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
// This module uses 'morphology.hpp' internally.

#include <mypaint-tiled-surface.h>

// This constant is to refer from fill.cpp
//
#define SHOULD_FILL_FLAG 0x0024 // both of ERODED_MASK | SKELTON_RESULT_MASK.
                                // But, when skelton infomation generated,
                                // the ERODED_MASK has been removed from the
                                // center tile with specialized finishing 
                                // kernel functor(SkeltonFinishKernel).

// # Interface functions

// setup state tile, to detect fillable gap.
PyObject* detect_contour(PyObject* py_statedict, // the tiledict for dilated tiles.
                         PyObject* py_surfdict, //  source surface tile dict.
                         int tx, int ty,  // the position of py_filled_tile
                         int targ_r, int targ_g, int targ_b, int targ_a, //premult target pixel color
                         double tol,   // pixel tolerance of filled area.
                         int gap_size, // overflow-preventing closable gap size.
                         int do_skelton // use skelton morphology(slow)
                         );

// dilate filled tile
PyObject* dilate_filled_tile(PyObject* py_dilated, // the tiledict for dilated tiles.
                             PyObject* py_filled_tile, // the filled src tile. 
                             int tx, int ty,  // the position of py_filled_tile
                             int grow_size,    // growing size from center pixel.
                             int kernel_type  // 0 for square kernel, 1 for diamond kernel
                            );
// XXX TEST CODES
PyObject* test_skelton(PyObject* py_statedict, // the tiledict for dilated tiles.
                       int tx, int ty,  // the position of py_filled_tile
                       int gap_size  // overflow-preventing closable gap size.
                       );

#endif

