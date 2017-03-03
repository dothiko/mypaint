/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef FILL_GUARD_HPP
#define FILL_GUARD_HPP

#include <Python.h>

// This Module is for implementing 'Dilating filled area' and 
// 'overflow prevention' functionality for flood_fill (fill.cpp)
// This module uses 'morphology.hpp' internally.

#include <mypaint-tiled-surface.h>

// This constant is to refer from fill.cpp
//

#define INNER_CONTOUR_FLAG 0x04
                            // To detect whether initially pressed position
                            // is inside eroded contour.
                            // If so, we need to limit floodfill target
                            // to eroded and targeted pixel area.

// # Interface functions

// setup state tile, to detect fillable gap.
PyObject* fill_gap(PyObject* py_statedict, // the tiledict for dilated tiles.
                   PyObject* py_surfdict, //  source surface tile dict.
                   int tx, int ty,  // the position of py_filled_tile
                   int targ_r, int targ_g, int targ_b, int targ_a, 
                   double tol,   // pixel tolerance of filled area.
                   int gap_size  // overflow-preventing closable gap size.
                   );

// dilate filled tile
PyObject* dilate_filled_tile(PyObject* py_dilated, // the tiledict for dilated tiles.
                             PyObject* py_filled_tile, // the filled src tile. 
                             int tx, int ty,  // the position of py_filled_tile
                             double fill_r, double fill_g, double fill_b, 
                             int grow_size    // growing size from center pixel.
                            );
#ifdef HEAVY_DEBUG
// XXX TEST CODES
PyObject*
test(PyObject* py_statedict, // the tiledict for status tiles.
     const int tx, const int ty,
     const int dilate_size);   // growing size from center pixel.
#endif

#endif

