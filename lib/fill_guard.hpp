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

// These constants are also referred from fill.cpp
// So, they are not static const member, preprocessor constant.

#define EXIST_FLAG 0x01
#define DILATED_FLAG 0x02 // This means the pixel is just dilated pixel,
                          // not sure original source pixel.
                          // it might be empty pixel in source tile.
#define ERODED_FLAG 0x04
#define BORDER_FLAG 0x10 // Border flag, to detect touching flood-filled
                         // area.

//// Utility functions for c++ modules

// get flag of status tile pixel. 
// this function is NULL-safe, if tile is NULL, just return 0. 
int get_status_flag(PyObject* sts_tile, const int x, const int y);

// set flag of status tile pixel.
// this function is NULL-safe, if tile is NULL, nothing done. 
// 'set' means bitwise OR operation.
void set_status_flag(PyObject* sts_tile, 
                     const int x, const int y, 
                     const char flag);

//// Interface functions

// setup state tile, to detect fillable gap.
PyObject* 
close_gap(PyObject* py_status_dict, // the tiledict for status tiles.
          PyObject* py_surfdict, //  source surface tile dict.
          const int tx, const int ty,  // the position of py_filled_tile
          const int targ_r, const int targ_g, const int targ_b, const int targ_a, 
          const double tol,   // pixel tolerance of filled area.
          const int gap_size  // overflow-preventing closable gap size.
          );


// dilate filled tile
PyObject* 
dilate_filled_tile(PyObject* py_dilated, // the tiledict for dilated tiles.
                   PyObject* py_filled_tile, // the filled src tile. 
                   const int tx, const int ty,  // the position of py_filled_tile in tile dict.
                   const double fill_r, const double fill_g, const double fill_b, 
                   const int dilation_size // dilating pixel radius.
                   );

// erode filled tile
PyObject* 
erode_filled_tile(PyObject* py_srctile_dict, // the tiledict for SOURCE pixel tiles.
                  PyObject* py_targ_tile, // the filled TARGET tile.  
                  const int tx, const int ty,  // the position of py_filled_tile in tile dict.
                  const double fill_r, const double fill_g, const double fill_b, 
                  const int erosion_size // eroding pixel radius.
                  );

PyObject*
contour_fill(PyObject* py_status_dict, // the tiledict for status tiles.
             const int tx, const int ty,  // the position of py_filled_tile
             PyObject* py_pixel_tile, // the TARGET pixel tile.  
             const int targ_r, const int targ_g, const int targ_b, const int targ_a);

PyObject*
ensure_dilate_tile(PyObject* py_status_dict,  // entire status tile dictionary
                   const int tx, const int ty,// the tile position of center tile
                   PyObject* py_targ_tile,     // the TARGET status tile)  
                   const int dilation_size
                   ); 

#ifdef HEAVY_DEBUG
// XXX TEST CODES
PyObject*
test(PyObject* py_status_dict, // the tiledict for status tiles.
     const int tx, const int ty,
     const int dilate_size);   // growing size from center pixel.
#endif

#endif

