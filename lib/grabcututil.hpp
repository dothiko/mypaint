/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef GRABCUTUTIL_HPP
#define GRABCUTUTIL_HPP

#include <Python.h>

// This Module is for implementing 'Dilating filled area' 

//// Interface functions

PyObject*
grabcututil_convert_tile_to_binary(
    PyObject *py_binary,
    PyObject *py_tile,
    int dst_x, int dst_y,
    double r, double g, double b,
    int value,
    int margin, int inverted,
    double alpha_tolerance
);

PyObject*
grabcututil_convert_tile_to_image(
    PyObject *py_cvimg,
    PyObject *py_tile,
    int dst_x, int dst_y,
    double bg_r, double bg_g, double bg_b,
    int margin
);


PyObject*
grabcututil_convert_binary_to_tile(
    PyObject *py_tile,
    PyObject *py_binary,
    int src_x, int src_y,
    double fill_r, double fill_g, double fill_b,
    int targ_value,
    int margin
);

PyObject*
grabcututil_setup_cvimg(
    PyObject *py_cvimg,
    double bg_r, double bg_g, double bg_b,
    int margin
);

PyObject*
grabcututil_finalize_cvmask(
    PyObject *py_cvmask,
    PyObject *py_cvimg,
    double targ_r, double targ_g, double targ_b,
    int remove_lineart
);

#endif


