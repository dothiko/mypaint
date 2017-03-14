/* This file is part of MyPaint.
 * Copyright (C) 2013-2014 by Andrew Chadwick <a.t.chadwick@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef FILL_HPP
#define FILL_HPP

#include <Python.h>
#include "fix15.hpp"

// Flood-fills one tile starting at a sequence of seed positions, and returns
// where overflows happened as seed lists in the cardinal directions.
//
// Returns a tuple of four lists (N, E, S, W) of overflow coordinate pairs
// [(x1, y1), ...] denoting to which pixels of the next tile in the identified
// direction the fill has overflowed. These coordinates can be fed back in to
// tile_flood_fill() for the tile identified as seeds.
PyObject *
tile_flood_fill  (PyObject *src,     // readonly HxWx4 array of uint16
                  PyObject *dst,     // output HxWx4 array of uint16
                  PyObject *seeds,   // List of 2-tuples
                  int targ_r, int targ_g, int targ_b, int targ_a, //premult
                  double fill_r, double fill_g, double fill_b,
                  int min_x, int min_y, int max_x, int max_y,
                  double tolerance,  // [0..1]
                  PyObject *status   // Status tile, HxWx1 array of char
                  );       


// XXX To use _floodfill_color_match function from outside fill.cpp,
// copied entirely, with renamed.
// and ignored at mypaintlib.i with %ignore directive, to hide from python.
// 
// Similarity metric used by flood fill.  Result is a fix15_t in the range
// [0.0, 1.0], with zero meaning no match close enough.  Similar algorithm to
// the GIMP's pixel_difference(): a threshold is used for a "similar enough"
// determination.
inline fix15_t
floodfill_color_match(const fix15_short_t c1_premult[4],
                      const fix15_short_t c2_premult[4],
                      const fix15_t tolerance)
{
    const fix15_short_t c1_a = c1_premult[3];
    fix15_short_t c1[] = {
        fix15_short_clamp(c1_a <= 0 ? 0 : fix15_div(c1_premult[0], c1_a)),
        fix15_short_clamp(c1_a <= 0 ? 0 : fix15_div(c1_premult[1], c1_a)),
        fix15_short_clamp(c1_a <= 0 ? 0 : fix15_div(c1_premult[2], c1_a)),
        fix15_short_clamp(c1_a),
    };
    const fix15_short_t c2_a = c2_premult[3];
    fix15_short_t c2[] = {
        fix15_short_clamp(c2_a <= 0 ? 0 : fix15_div(c2_premult[0], c2_a)),
        fix15_short_clamp(c2_a <= 0 ? 0 : fix15_div(c2_premult[1], c2_a)),
        fix15_short_clamp(c2_a <= 0 ? 0 : fix15_div(c2_premult[2], c2_a)),
        fix15_short_clamp(c2_a),
    };

    // Calculate the raw distance
    fix15_t dist = 0;
    for (int i=0; i<4; ++i) {
        fix15_t n = (c1[i] > c2[i]) ? (c1[i] - c2[i]) : (c2[i] - c1[i]);
        if (n > dist)
            dist = n;
    }
    /*
     * // Alternatively, could use
     * fix15_t sumsqdiffs = 0;
     * for (int i=0; i<4; ++i) {
     *     fix15_t n = (c1[i] > c2[i]) ? (c1[i] - c2[i]) : (c2[i] - c1[i]);
     *     n >>= 2; // quarter, to avoid a fixed maths sqrt() overflow
     *     sumsqdiffs += fix15_mul(n, n);
     * }
     * dist = fix15_sqrt(sumsqdiffs) << 1;  // [0.0 .. 0.5], doubled
     * // but the max()-based metric will a) be more GIMP-like and b) not
     * // lose those two bits of precision.
     */

    // Compare with adjustable tolerance of mismatches.
    static const fix15_t onepointfive = fix15_one + fix15_halve(fix15_one);
    if (tolerance > 0) {
        dist = fix15_div(dist, tolerance);
        if (dist > onepointfive) {  // aa < 0, but avoid underflow
            return 0;
        }
        else {
            fix15_t aa = onepointfive - dist;
            if (aa < fix15_halve(fix15_one))
                return fix15_short_clamp(fix15_double(aa));
            else
                return fix15_one;
        }
    }
    else {
        if (dist > tolerance)
            return 0;
        else
            return fix15_one;
    }
}


#endif //__HAVE_FILL_HPP

