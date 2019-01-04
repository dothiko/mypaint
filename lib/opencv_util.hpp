/* This file is part of MyPaint.
 * Copyright (C) 2018 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef OPENCV_UTIL_HPP
#define OPENCV_UTIL_HPP

#include <Python.h>

// This Module is for implementing 'Dilating filled area' 

//// Interface functions

PyObject*
opencvutil_convert_tile_to_image(
    PyObject *py_cvimg,
    PyObject *py_cvalpha,
    PyObject *py_tile,
    int dst_x, int dst_y
);

PyObject*
opencvutil_convert_image_to_tile(
    PyObject *py_cvimg,
    PyObject *py_cvalpha,
    PyObject *py_tile,
    PyObject *py_cvmask,
    int src_x, int src_y
);

PyObject*
opencvutil_is_empty_area(
    PyObject *py_cvalpha,
    int sx, int sy
);

#endif


