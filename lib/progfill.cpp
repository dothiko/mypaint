/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "progfill.hpp"
#include "progfillworkers.hpp"
#include "common.hpp"
#include "fix15.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <mypaint-tiled-surface.h>
#include <math.h>
#include <glib.h>

//// Struct definition

//-----------------------------------------------------------------------------
//// Function definition


// XXX borrowed from `_floodfill_color_match` of lib/fill.cpp. almost same.
static inline fix15_t
_closefill_color_match(const fix15_short_t c1_premult[4],
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
     * // but the MAX()-based metric will a) be more GIMP-like and b) not
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

#ifdef HEAVY_DEBUG
// To share same check code
void 
assert_tile(PyArrayObject* array) {
    assert(PyArray_Check(array));
    // We need to check/assert mypaint color tile with this function,
    // so we MUST use TILE_SIZE here, explicitly.
    assert(PyArray_DIM(array, 0) == TILE_SIZE);
    assert(PyArray_DIM(array, 1) == TILE_SIZE);
    assert(PyArray_DIM(array, 2) == 4);
    assert(PyArray_TYPE(array) == NPY_UINT16);
    assert(PyArray_ISCARRAY(array));
}
#endif

//-----------------------------------------------------------------------------
//// Class definition
//
// FYC, Base worker classes are defined at lib/profilldefine.hpp
// And most of derived worker classes are defined at lib/progfillworkers.hpp

//--------------------------------------
/// KernelWorker 

KernelWorker::KernelWorker(FlagtileSurface *surf, const int level)
    : TileWorker(surf, level)
{
    // C++ virtual function call cannot work properly
    //
    // in constructor.It just call its own method,
    // not virtual(derived) one.
    // Therefore, we need call `set_target_level` of
    // this class here.
    set_target_level(level);
} 

// We can enumerate kernel window(surrounding) 4 pixels with for-loop
// by these offset, in the order of top, right, bottom, left (clockwise)
// AntialiasKernel depends the order of this offsets.
// DO NOT CHANGE THIS ORDER.
const int KernelWorker::xoffset[] = { 0, 1, 0, -1};
const int KernelWorker::yoffset[] = {-1, 0, 1,  0};

void 
KernelWorker::set_target_level(const int level) 
{
#ifdef HEAVY_DEBUG
    assert(level >= 0); 
    assert(level <= MAX_PROGRESS); 
#endif
    m_level = level; 
    m_max_x = m_surf->get_pixel_max_x(level);
    m_max_y = m_surf->get_pixel_max_y(level);
}

/**
* @start
* starting handler of kernel worker.
*
* @param targ: the currently targeted Flagtile object.This might be NULL.
* @return false when kernel processing for a tile is cancelled.
* @detail 
* The method called before the kernel worker is operated
* (especially in FlagtileSurface::filter method).
* If this `start` method return false, the tile processing cancelled and
* proceed next tile.
*/
bool 
KernelWorker::start(Flagtile* targ) 
{
    if(targ != NULL) {
        m_processed = 0;
        return true;
    }
    return false;
}

// The method called after the kernel worker operation completed.
void 
KernelWorker::end(Flagtile* targ) 
{
    if (m_processed > 0)
    {
        targ->set_stat(Flagtile::DIRTY);
        m_processed = 0;
    }
}

uint8_t 
KernelWorker::get_kernel_pixel(const int level,
                               const int sx, const int sy,
                               const int idx) 
{
    m_px = sx + xoffset[idx];
    m_py = sy + yoffset[idx];
    if (m_px > 0 && m_px < m_max_x
            && m_py > 0 && m_py < m_max_y) {
        return m_surf->get_pixel(level, m_px, m_py);
    }
    return 0;
}

void
KernelWorker::finalize() 
{
    // Remove working flag from entire surface 
    // And clear dirty state flags here.
    for(int i=0; i<m_surf->get_width() * m_surf->get_height(); i++) {
        Flagtile *t = m_surf->get_tile(i, false);
        if (t != NULL && (t->get_stat() & Flagtile::DIRTY)) {
            t->clear_bitwise_flag(FLAG_WORK);
            t->unset_stat(Flagtile::DIRTY);
        }
    }
}

//--------------------------------------
//// WalkingKernel class

// Wrapper method to get pixel with direction.
uint8_t 
WalkingKernel::_get_pixel_with_direction(const int x, const int y, 
                                          const int direction) 
{
    return m_surf->get_pixel(
        m_level,
        x + xoffset[direction],
        y + yoffset[direction]
    );
}

// Check whether the right side pixel of current position / direction
// is match to proceed.
bool 
WalkingKernel::_is_match_pixel(const uint8_t pixel) 
{
    return ((pixel & PIXEL_MASK) == PIXEL_FILLED);
}

// Rotate to right.
// This is used when we missed wall at right-hand. 
void 
WalkingKernel::_rotate_right() 
{
    // We need update current direction
    // before call rotation handler.
    m_cur_dir = (m_cur_dir + 1) & 3;
    _on_rotate_cb(true);   
}

// Rotate to left. 
// This is used when we face `wall`
void 
WalkingKernel::_rotate_left() 
{
    // We need update current direction
    // before call rotation handler
    m_cur_dir = (m_cur_dir - 1) & 3;          
    m_left_rotate_cnt++;
    _on_rotate_cb(false); 
}

bool 
WalkingKernel::_proceed() 
{
    uint8_t pix;

    // see front.
    pix = _get_front_pixel(); 

    if (_is_match_pixel(pix)) {
        // Face to wall.
        // Now, we must turn left.
        // With this turn, this kernel draws antialias line or
        // initialize internal status of this object.
        
        _rotate_left();
        if (m_left_rotate_cnt >= 4)
            return false; // Exit from infnite loop of 1px hole!
    } 
    else {
        // Then, proceed.

        int nx = m_x + xoffset[m_cur_dir];
        int ny = m_y + yoffset[m_cur_dir];
        // Refreshing clockwise counter.
        // Algorithm from
        // https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
        m_clockwise_cnt += (nx - m_x) * (ny + m_y);

        m_x = nx;
        m_y = ny;
        m_step++;
        m_left_rotate_cnt = 0;

        if (m_x == m_ox && m_y == m_oy) {
            return false; // Walking end!!
        }
    }
    return true;
}

// Walk single step.
// when end walking, return false.
bool 
WalkingKernel::_walk_single() 
{
    if (!_is_match_pixel(_get_hand_pixel())) {
        // Right hand of kernel misses the wall.
        // Couldn't proceed.
        _rotate_right();
    }
    return _proceed();
}

void 
WalkingKernel::_walk(const int sx, const int sy, const int direction) 
{
#ifdef HEAVY_DEBUG
    unsigned int cnt = 0;
#endif
    m_ox = sx;
    m_oy = sy;
    m_x = sx;
    m_y = sy;
    m_step = 0;
    m_left_rotate_cnt = 0;
    m_clockwise_cnt = 0;
    
    m_cur_dir = direction;

    while (_walk_single()) {
#ifdef HEAVY_DEBUG
        cnt++;
        // `Walking over 100 million pixel` cannot happen.
        // It would be infinite loop bug.
        assert(cnt < 100000000);
#endif
    }
}

// Caution: Child class of this kernel should accepts
// NULL tile.
bool 
WalkingKernel::start(Flagtile* targ) 
{
    // `step` method of this class accepts
    // NULL tile. so we cannot use `start`
    // method of KernelWorker base class.
    m_processed = 0; 
    if ( targ != NULL 
        && (targ->get_stat() & Flagtile::FILLED) != 0) {
            return false;
    }
    return true;
}
    
//--------------------------------------
//// Flagtile 

const int 
Flagtile::m_buf_offsets[MAX_PROGRESS+1] = {
    // progress level 0: 64x64(TILE_SIZE * TILE_SIZE), start from 0.
    0,
    PROGRESS_BUF_SIZE(0),
    Flagtile::m_buf_offsets[1] + PROGRESS_BUF_SIZE(1),
    Flagtile::m_buf_offsets[2] + PROGRESS_BUF_SIZE(2),
    Flagtile::m_buf_offsets[3] + PROGRESS_BUF_SIZE(3),
    Flagtile::m_buf_offsets[4] + PROGRESS_BUF_SIZE(4),
    Flagtile::m_buf_offsets[5] + PROGRESS_BUF_SIZE(5)
};

Flagtile::Flagtile(const int initial_value) 
    : m_statflag(0) 
{
    m_buf = new uint8_t[BUF_SIZE];
    fill(initial_value);
}

Flagtile::~Flagtile() 
{
    if (m_buf != NULL)
        delete[] m_buf;
}

void 
Flagtile::_build_progress_level(const int targ_level) 
{
#ifdef HEAVY_DEBUG
    assert(targ_level >= 1);
#endif
    int c_size = PROGRESS_TILE_SIZE(targ_level);
    int b_level = targ_level - 1;
    int total_area = 0;

    for (int y=0; y < c_size; y++) {
        for (int x=0; x < c_size; x++) {
            int contour_cnt = 0;
            int area_cnt = 0;

            // The progress level beneath is always
            // double sized of current level.
            int bbx = x << 1;
            int bby = y << 1;
            for (int by=0; by < 2; by++) {
                for (int bx=0; bx < 2; bx++) {
                    uint8_t pix = get(b_level, bbx+bx, bby+by);
                    switch(pix & PIXEL_MASK) {
                        case PIXEL_CONTOUR:
                            contour_cnt++;
                            break;
                        case PIXEL_AREA:
                            area_cnt++;
                            break;
                    }
                }
            }

            uint8_t new_pixel = 0;

            if (area_cnt > 0) {
                // Set flag which means `there should be some area pixels.`
                new_pixel = PIXEL_AREA;
            }

            if (contour_cnt > 0) {
                new_pixel = PIXEL_CONTOUR;
            }

            if (new_pixel != 0) {
                replace(
                    targ_level,
                    x, y,
                    new_pixel
                );
            }

            total_area += area_cnt;
        }
    }// Tile processing end
}

void
Flagtile::build_progress_seed(const int max_level) 
{
#ifdef HEAVY_DEBUG
    assert(max_level >= 1);
    assert(max_level <= MAX_PROGRESS);
#endif
    if ((get_stat() & FILLED_AREA) || (get_stat() & FILLED) || (get_stat() & EMPTY))
        return; // There is nothing to do for already filled or empty tile.
        
    for(int i=1;i <= max_level; i++) {
        _build_progress_level(i);
    }
}

/**
* @convert_from_color
* convert color-tile into flagtile.
*
* @param alpha_threshold : With a certain brush preset, freehand tool 
*                          would draw almost invisible strokes even 
*                          no stylus pressure is applied.
*                          It is difficult to reject such pixels with
*                          `tolerance` option.
*                          So, in addition to `tolerance`, use this parameter.
*                          This parameter would be disabled(i.e. only
*                          tolerance is used to produce PIXEL_CONTOUR)
*                          when equal to 0.0.
*                          practically, this value is enough around 0.03
*
* @param limit_within_opaque: boolean, if true, generate contour pixel
*                             even transparent pixel.
*                             Mainly used from LassofillSurface.
* @detail 
* convert color-tile into flagtile, to decide contour pixel.
*/

void
Flagtile::convert_from_color(PyObject *py_src_tile,
                             const int targ_r, 
                             const int targ_g, 
                             const int targ_b, 
                             const int targ_a, 
                             const double tolerance,
                             const double alpha_threshold,
                             const bool limit_within_opaque)
{
    PyArrayObject *array = (PyArrayObject*)py_src_tile;
#ifdef HEAVY_DEBUG
    assert_tile(array);
#endif
    
    const unsigned int xstride = PyArray_STRIDE(array, 1) / sizeof(fix15_short_t);
    const unsigned int ystride = PyArray_STRIDE(array, 0) / sizeof(fix15_short_t);
    const unsigned int yoffset = ystride - xstride * TILE_SIZE;
    fix15_short_t *cptr = (fix15_short_t*)PyArray_BYTES(array);
    uint8_t* tptr = get_ptr();

    const fix15_short_t targ[4] = {
        fix15_short_clamp(targ_r), 
        fix15_short_clamp(targ_g),
        fix15_short_clamp(targ_b), 
        fix15_short_clamp(targ_a)
    };

    fix15_t f15_tolerance = (fix15_t)(tolerance * (double)fix15_one);
    fix15_t f15_threshold = (fix15_t)(alpha_threshold * (double)fix15_one);
    int contour_cnt = 0;

    // Converting color tile into progress level 0 pixel.
    for(int y=0; y<TILE_SIZE; y++) {
        for(int x=0; x<TILE_SIZE; x++) {
            uint8_t pix = *tptr & PIXEL_MASK;
            fix15_t alpha = (fix15_t)cptr[3];
            if (!limit_within_opaque || alpha > 0) {
                // Under certain circumstance,
                // the pixel area would be 
                if (pix == PIXEL_AREA && alpha >= f15_threshold) {
                    fix15_t match = _closefill_color_match(
                        targ, cptr,
                        f15_tolerance
                    );
                    
                    if (match == 0) { 
                        *tptr = PIXEL_CONTOUR;
                        contour_cnt++;
                    }
                }
            }
            else {
                // Code reaches here when
                // limit_within_opaque is true and current pixel
                // is completely transparent.
                *tptr = 0;
            }
            cptr += xstride;
            tptr++;
        }
        cptr += yoffset;
    }

    if(contour_cnt > 0) {
        set_stat(HAS_CONTOUR);
    }
}

/**
* @convert_to_color
* convert Flagtile to mypaint colortile.
*
* @param py_targ_tile Mypaint colortile(numpy array of uint8)
* @param tx,ty  Position of source flag tile.
* @param r,g,b Target pixel color, to be converted from flag.
*              They are floating point value, from 0.0 to 1.0.
* @detail 
* Convert Flagtile pixel into Mypaint color tile.
*/
void 
Flagtile::convert_to_color(PyObject *py_targ_tile,
                           const double r, const double g, const double b)
{
    PyArrayObject *array = (PyArrayObject*)py_targ_tile;
#ifdef HEAVY_DEBUG
    assert_tile(array);
#endif

    if ((get_stat() & Flagtile::FILLED_AREA) 
            || (get_stat() & Flagtile::EMPTY)) {
        return;
    }

    const unsigned int xstride = PyArray_STRIDE(array, 1) / sizeof(fix15_short_t);
    const unsigned int ystride = PyArray_STRIDE(array, 0) / sizeof(fix15_short_t);
    const unsigned int yoffset = ystride - xstride * TILE_SIZE;
    fix15_short_t *cptr = (fix15_short_t*)PyArray_BYTES(array);
    uint8_t *sptr = m_buf;

    // Premult color pixel array.
    fix15_short_t cols[4];
    cols[0] = fix15_short_clamp(r * fix15_one);
    cols[1] = fix15_short_clamp(g * fix15_one);
    cols[2] = fix15_short_clamp(b * fix15_one);
    cols[3] = fix15_one;
    
    // Completely filled tile can be filled with memcpy.
    if (get_stat() & Flagtile::FILLED) {
        fix15_short_t *optr = cptr;
        // Build one line
        for(int x=0; x<TILE_SIZE; x++) {
            cptr[0] = cols[0];
            cptr[1] = cols[1];
            cptr[2] = cols[2];
            cptr[3] = cols[3];
            cptr += xstride;
        }
        cptr += yoffset;
        // Just copy that completed line.
        for(int y=1; y<TILE_SIZE; y++) {
            memcpy(cptr, optr, sizeof(fix15_short_t) * 4 * TILE_SIZE);
            cptr += ystride;
        }
        return;
    }   

    // Converting progress level 0 pixel into color tile.
    for(int y=0; y<TILE_SIZE; y++) {
        for(int x=0; x<TILE_SIZE; x++) {
            uint8_t pix = *sptr;
            // At first, We need check anti-aliasing flag of pixel.
            // Anti-aliasing pixel is not compatible
            // with another PIXEL_* values.
            if ((pix & FLAG_AA) != 0) {
                double alpha = _get_aa_double_value(pix & AA_MASK);
                fix15_short_t cur_alpha = fix15_short_clamp(
                    (alpha * fix15_one) + cptr[3]
                );
                cptr[0] = fix15_short_clamp(r * cur_alpha);
                cptr[1] = fix15_short_clamp(g * cur_alpha);
                cptr[2] = fix15_short_clamp(b * cur_alpha);
                cptr[3] = cur_alpha;
                
            }
            else if ((pix & PIXEL_MASK) == PIXEL_FILLED) {
                cptr[0] = cols[0];
                cptr[1] = cols[1];
                cptr[2] = cols[2];
                cptr[3] = cols[3];
            }
            cptr += xstride;
            sptr++;
        }
        cptr += yoffset;
    }
}

void 
Flagtile::convert_to_transparent(PyObject *py_targ_tile)
{
    PyArrayObject *array = (PyArrayObject*)py_targ_tile;
#ifdef HEAVY_DEBUG
    assert_tile(array);
#endif
     
    if ((get_stat() & Flagtile::FILLED_AREA) 
            || (get_stat() & Flagtile::EMPTY)) {
        return;
    }

    const unsigned int xstride = PyArray_STRIDE(array, 1) / sizeof(fix15_short_t);
    const unsigned int ystride = PyArray_STRIDE(array, 0) / sizeof(fix15_short_t);
    const unsigned int yoffset = ystride - xstride * TILE_SIZE;
    fix15_short_t *cptr = (fix15_short_t*)PyArray_BYTES(array);
    uint8_t *sptr = m_buf;

    // Completely filled tile can be cleared with memset.
    if (get_stat() & Flagtile::FILLED) {
        for(int y=0; y<TILE_SIZE; y++) {
            memset(cptr, 0, sizeof(fix15_short_t) * 4 * TILE_SIZE);
            cptr += ystride;
        }
        return;
    }
    
    // Converting progress level 0 pixel into color tile.
    for(int y=0; y<TILE_SIZE; y++) {
        for(int x=0; x<TILE_SIZE; x++) {
            uint8_t pix = *sptr;
            // We need check anti-aliasing pixel first.
            // anti-aliasing pixel is not compatible
            // with another PIXEL_* values.
            // Some of them accidentally misdetected.
            if ((pix & FLAG_AA) != 0) {
                pix &= AA_MASK;
                double cur_alpha = _get_aa_double_value(pix);
                double targ_alpha = (double)cptr[3];
                // Reverse premult color as original float color(0.0 - 1.0)
                double targ_r = (double)cptr[0] / targ_alpha; 
                double targ_g = (double)cptr[1] / targ_alpha; 
                double targ_b = (double)cptr[2] / targ_alpha; 

                targ_alpha /= fix15_one;
                fix15_short_t final_alpha = (cur_alpha * targ_alpha) * fix15_one;
                // Then, make `premult` them again with new alpha value.
                cptr[0] = fix15_short_clamp(targ_r * final_alpha);
                cptr[1] = fix15_short_clamp(targ_g * final_alpha);
                cptr[2] = fix15_short_clamp(targ_b * final_alpha);
                cptr[3] = (fix15_short_t)final_alpha;
            }
            else if ((pix & PIXEL_MASK) == PIXEL_FILLED) {
                cptr[0] = 0;
                cptr[1] = 0;
                cptr[2] = 0;
                cptr[3] = 0;
            }
            cptr += xstride;
            sptr++;
        }
        cptr += yoffset;
    }
}

void 
Flagtile::fill(const uint8_t val) 
{
    memset(m_buf, val, BUF_SIZE);
    switch(val) {
        case PIXEL_FILLED:
            set_stat(FILLED);
            break;
        case PIXEL_AREA:
            set_stat(FILLED_AREA);
            break;
        case 0:
            set_stat(EMPTY);
            break;
    }
}

// Caution: You MUST call set_stat with only one stat flag.
// Do not comibine it as parameter.
void 
Flagtile::set_stat(const int new_stat) 
{ 
    m_statflag |= new_stat; 

    // These flags are exclusive.
    if((new_stat & HAS_PIXEL) || (new_stat & HAS_CONTOUR)) {
        m_statflag &= ~(FILLED_AREA | EMPTY);    
    }
    else if(new_stat & FILLED) {
        m_statflag |= HAS_PIXEL; 
        m_statflag &= ~(FILLED_AREA | HAS_CONTOUR | EMPTY);
    }
    else if(new_stat & FILLED_AREA) {
        m_statflag &= ~(FILLED | HAS_PIXEL | HAS_CONTOUR | EMPTY); 
    }
    else if(new_stat & EMPTY) {
        m_statflag &= ~(FILLED | HAS_PIXEL | HAS_CONTOUR | FILLED_AREA); 
    }

}

void 
Flagtile::unset_stat(const int new_stat) 
{
    m_statflag &= ~new_stat; 

    if(new_stat & HAS_PIXEL) {
        m_statflag &= ~FILLED; 
    }
}

//--------------------------------------
//// FlagtileSurface class.
//   psuedo surface object.
//   This manages Flagtile objects. 

// Default constructor for derive class. actually, do not use.
FlagtileSurface::FlagtileSurface(const int start_level, 
                                 const uint8_t initial_tile_val) 
    : m_tiles(NULL), m_level(start_level), 
      m_initial_tile_val(initial_tile_val)
{
}

FlagtileSurface::~FlagtileSurface() 
{
    for(int i=0; i<m_width*m_height; i++) {
        Flagtile *ct = m_tiles[i];
        delete ct;
    }
    delete[] m_tiles;

#ifdef HEAVY_DEBUG
    printf("FlagtileSurface destructor called.\n");
#endif
}

//// Internal methods

void
FlagtileSurface::_generate_tileptr_buf(const int ox, const int oy,
                                       const int w, const int h)
{
#ifdef HEAVY_DEBUG
    assert(m_tiles == NULL);
#endif
    // Make Surface size larger than requested 
    // by 1 tile for each direction.
    // Also, make origin 1 tile smaller.
    // Because, Dilation might exceed surface border.

    // CAUTION: m_width and m_height are in tile-unit. not in pixel.

    m_ox = ox - 1;
    m_oy = oy - 1;
    m_width = w + 2;
    m_height = h + 2;

    m_tiles = new Flagtile*[m_width * m_height];
    memset(
        m_tiles, 
        0, 
        m_width * m_height * sizeof(Flagtile*)
    );
}


void 
FlagtileSurface::_convert_flag(const uint8_t targ_flag, 
                               const uint8_t flag,
                               const bool look_dirty)
{
    Flagtile *t;

    for(int ty=0; ty<m_height; ty++) {
        for(int tx=0; tx<m_width; tx++) {

            t = get_tile(tx, ty, false);
            if (t == NULL 
                    || (look_dirty 
                        && !(t->get_stat() & Flagtile::DIRTY))) {
                continue;
            }

            uint8_t *ptr = t->get_ptr();
            for (int i=0; 
                     i < TILE_SIZE*TILE_SIZE; 
                     i++) {
                if (*ptr == targ_flag)
                    *ptr = flag;
                ptr++;
            }

            if(look_dirty)
                t->unset_stat(Flagtile::DIRTY);
        }
    }
}

/**
* @flood_fill
* Do `Flood fill` from
*
* @param level: target progress level to doing flood-fill
* @param px, py: flood-fill starting point, in progress level coordinate.
* @detail 
* Call DrawWorker.step for each flood-fill point.
* Algorithm is copied/conveted from fill.cpp, but in this class
* we do not need to access `real` color pixels of mypaint surface.
* So we can complete flood-fill operation within this class,
* no need to return python seed tuples.
*/

void 
FlagtileSurface::flood_fill(const int sx, const int sy,
                            FillWorker* w)                                
{
    const int level = w->get_target_level();
    const int tile_size = PROGRESS_TILE_SIZE(level);
    int px = sx % tile_size;
    int py = sy % tile_size;
    int otx = sx / tile_size;
    int oty = sy / tile_size;
    
#ifdef HEAVY_DEBUG
    assert(px >= 0);
    assert(py >= 0);
    assert(px < tile_size);
    assert(py < tile_size);
    assert(otx >= 0);
    assert(oty >= 0);
    assert(otx < m_width);
    assert(oty < m_height);
#endif
    Flagtile *ot = get_tile(otx, oty, true);
    uint8_t pix = ot->get(level, px, py);
    
    // Initial seed pixel check.
    if (!w->start(ot) || !w->match(pix))
        return;

    // Populate a working queue with seeds
    GQueue *queue = g_queue_new();   /* Of tuples, to be exhausted */
    _progfill_point *seed_pt = (_progfill_point*)
                                  malloc(sizeof(_progfill_point));
    seed_pt->x = sx;
    seed_pt->y = sy;
    g_queue_push_tail(queue, seed_pt);

    static const int x_delta[] = {-1, 1};
    static const int x_offset[] = {0, 1};
    const int max_x = tile_size * m_width;
    const int max_y = tile_size * m_height;

    while (! g_queue_is_empty(queue)) {
        _progfill_point *pos = (_progfill_point*) g_queue_pop_head(queue);
        int x0 = pos->x;
        int y = pos->y;
        free(pos);
        
        // Find easternmost and westernmost points of the same colour
        // Westwards loop includes (x,y), eastwards ignores it.
        for (int i=0; i<2; ++i)
        {
            bool look_above = true;
            bool look_below = true;
            for ( int x = x0 + x_offset[i] ;
                  x >= 0 && x < max_x ;
                  x += x_delta[i] )
            {
                // halt if we're outside the bbox range
                if (x < 0|| y < 0 || x >= max_x || y >= max_y) {
                    break;
                }
                
                int tx = x / tile_size;
                int ty = y / tile_size;                    
                int px = x % tile_size;
                int py = y % tile_size;
                Flagtile *t;
                
                if (otx == tx && oty == ty) {
                    t = ot;
                }
                else {
                    t = get_tile(tx, ty, true);
                    ot = t;
                    otx = tx;
                    oty = ty;
                }              
                                
                if (x != x0) { // Test was already done for queued pixels
                    pix = t->get(level, px, py);
                    if (!w->match(pix))
                    {
                        break;
                    }
                }
                // Fill this pixel, and continue iterating in this direction
                w->step(t, px, py, x, y);
                
                // In addition, enqueue the pixels above and below.
                // Scanline algorithm here to avoid some pointless queue faff.
                if (y > 0) {
                    pix = get_pixel(level, x, y-1);
                    if(w->match(pix)) {
                        if (look_above) {
                            // Enqueue the pixel to the north
                            _progfill_point *p = (_progfill_point *) malloc(
                                                    sizeof(_progfill_point)
                                                  );
                            p->x = x;
                            p->y = y-1;
                            g_queue_push_tail(queue, p);
                            look_above = false;
                        }
                    }
                    else { // !match_above
                        look_above = true;
                    }
                }

                if (y < max_y - 1) {
                    pix = get_pixel(level, x, y+1);
                    if(w->match(pix)) {
                        if (look_below) {
                            // Enqueue the pixel to the South
                            _progfill_point *p = (_progfill_point *) malloc(
                                                    sizeof(_progfill_point)
                                                  );
                            p->x = x;
                            p->y = y+1;
                            g_queue_push_tail(queue, p);
                            look_below = false;
                        }
                    }
                    else { //!match_below
                        look_below = true;
                    }
                }
                
            }
        }
    }

    // Clean up working state.
    g_queue_free(queue);
    
    //w->end(tile);
    
    // Return where the fill has overflowed
    // into neighbouring tiles through param `queue_border`.
}

/**
* @filter_tiles
* Internal method of morphology operation.
*
* @param level: the target progress level. we need this to decide tile size.
* @return desc_of_return
* @detail detailed_desc
*/

void 
FlagtileSurface::filter_tiles(KernelWorker *w)
{
    Flagtile *t;
    int tile_size = PROGRESS_TILE_SIZE(w->get_target_level());

    // When targ_flag and new_flag are same,
    // it cause chain reaction within operation and
    // almost entire tile would be rewritten incorrectly.
    // To avoid this, we need to place temporary
    // working flag(FLAG_DILATED) and convert it later.

    for(int ty=0; ty<m_height; ty++) {
        for(int tx=0; tx<m_width; tx++) {
            t = get_tile(tx, ty, false);
            if (w->start(t)) {
                int bx = tx * tile_size;
                int by = ty * tile_size;

                // iterate tile pixel.
                for (int y=0; y<tile_size; y++) {
                    for (int x=0; x<tile_size; x++) {
                        w->step(t, x, y, bx+x, by+y);
                    }
                }
                // Get tile again, because there might be
                // new tile generated by some kernelworker
                // which accepts NULL tile.
                if (t == NULL)
                    t = get_tile(tx, ty, false);
                
                // iterate tile pixel end.
                // As a default, the `end` method 
                // of KernelWorker would set 
                // DIRTY stat flag for the tile.
                w->end(t);
            }
        }
    }
    
    w->finalize();
}

/// progress related.

/**
* @build_progress_seed
* The interfacing function of building(finalizing) progress seeds.
*
* @detail 
* To call internal building method for each progress level.
* Call this method after all of targeting color tiles are converted
* into FlagtileSurface.
*
* Before this method called, polygon rendering(with FLAG_AREA)
* should be completed.
*/
void 
FlagtileSurface::build_progress_seed() 
{
#ifdef HEAVY_DEBUG
    assert(m_level >= 1);
    assert(m_level <= MAX_PROGRESS);
#endif

    for(int ty=0; ty<m_height; ty++) {
        for(int tx=0; tx<m_width; tx++) {

            Flagtile *t = get_tile(tx, ty, false);
            if (t == NULL) {
                continue;
            }
            t->build_progress_seed(m_level);
        }
    }// Surface processing end
}

/// Progressive fill related.
/**
* @_progress_tiles
* The interface method of doing progressive reshape of filling area.
*
* @param reject_targ_level: The level where early pixel area rejection
*                           is executed. This value should be lower than
*                           starting progress level(m_level).
*                           We can use `gap-closing` effect of progress
*                           level pixel for pixel area rejection.
*                           To disable this feature, assign -1 to this param.
* @detail 
* With this method, we can reshape target filling area progressively.
* This method itself just call internal method `_progress_tiles`.
* And actual pixel manipulation(deciding the filling area) is done in
* ProgressWorker class.
*/
void 
FlagtileSurface::progress_tiles(const int reject_targ_level,
                                const int perimeter) 
{
    if (m_level > 0) {
        ProgressKernel k(this); 
        for (int i=m_level; i>0; i--) {
            if (i == reject_targ_level) {
                // CountPerimeterKernel just marks such needless
                // pixels, not remove it yet.
                CountPerimeterKernel ck(this, i, perimeter, false, false);
                filter_tiles((KernelWorker*)&ck);
                
                // RemoveGarbageKernel actually remove
                // that marked areas.
                RemoveGarbageKernel rk(this, i);
                filter_tiles((KernelWorker*)&rk);    
            }
            k.set_target_level(i);            
            filter_tiles((KernelWorker*)&k);
        }
    }
}

/**
* @finalize
* Finalize painting target area.
*
* @param threshold: The threshold value of filled area perimeter.
*                   If an area which perimeter is less than this value,
*                   that area is removed(filled with PIXEL_AREA)
* @detail 
* This is a default finalize method.
* progress_tiles method should be called before this method.
* LassofillSurface has dedicated version of finalize.
*/
/*
void 
FlagtileSurface::finalize(const int threshold,
                          const int dilation_size,
                          const bool antialias,
                          const bool fill_all_holes)
{
#ifdef HEAVY_DEBUG
    assert(dilation_size >= 0);
    assert(dilation_size < MYPAINT_TILE_SIZE);
#endif
    
    // Remove annoying small glich-like pixel area.
    // `Progressive fill` would mistakenly fill some
    // concaved areas around jagged contour edges as gap.
    // Theorically, such area cannot be produced when
    // gap-closing-level is less or equal to 1.
    if (threshold > 0 && m_level > 1) {
        // CountPerimeterKernel just marks such needless
        // pixels, not remove it yet.
        CountPerimeterKernel ck(this, 0, threshold, true, fill_all_holes);
        filter_tiles((KernelWorker*)&ck);

        // RemoveGarbageKernel actually remove
        // that marked areas.
        RemoveGarbageKernel rk(this, 0);
        filter_tiles((KernelWorker*)&rk);
    }
    else if (fill_all_holes) {
        // No area removal. Just fill hole.
        CountPerimeterKernel ck(this, 0, 0, true, fill_all_holes);
        filter_tiles((KernelWorker*)&ck);
    }

    // Dilate it. This should be prior to anti-aliasing.
    if (dilation_size > 0) {
        DilateKernel dk(this);
        for(int i=0; i<dilation_size; i++) {
            filter_tiles((KernelWorker*)&dk);
        }
    }

    // At last, do Psuedo Antialias for result pixels.
    if (antialias) {
        AntialiasKernel ak(this);
        filter_tiles((KernelWorker*)&ak);
    }
     
}
*/

/**
* @remove_small_areas
* Remove small areas, and fill all small holes if needed.
*
* @param threshold: The threshold value of filled area perimeter.
*                   If an area which perimeter is less than this value,
*                   that area is removed(filled with PIXEL_AREA)
* 
* @detail 
* This method is to reject small needless areas by counting its perimeter, 
* and fill it if it is `hole` (i.e. counter-clockwised area).
* 
*/
void
FlagtileSurface::remove_small_areas(const int threshold, const bool fill_all_holes)
{
    // Remove annoying small glich-like pixel area.
    // `Progressive fill` would mistakenly fill some
    // concaved areas around jagged contour edges as gap.
    // Theorically, such area cannot be produced when
    // gap-closing-level is less or equal to 1.
    if (threshold > 0 && m_level > 1) {
        // CountPerimeterKernel just marks such needless
        // pixels, not remove it yet.
        CountPerimeterKernel ck(this, 0, threshold, true, fill_all_holes);
        filter_tiles((KernelWorker*)&ck);

        // RemoveGarbageKernel actually remove
        // that marked areas.
        RemoveGarbageKernel rk(this, 0);
        filter_tiles((KernelWorker*)&rk);
    }
    else if (fill_all_holes) {
        // No area removal. Just fill hole.
        CountPerimeterKernel ck(this, 0, 0, true, fill_all_holes);
        filter_tiles((KernelWorker*)&ck);
    }    
}

void
FlagtileSurface::dilate(const int dilation_size)
{
    if (dilation_size > 0) {
        DilateKernel dk(this);
        for(int i=0; i<dilation_size; i++) {
            filter_tiles((KernelWorker*)&dk);
        }
    }
}

/**
* @draw_antialias
* Draw antialias lines around the final result pixels.
*
* @detail 
* This method should be called the last of fill process.
* 
*/
void
FlagtileSurface::draw_antialias()
{
    AntialiasKernel ak(this);
    filter_tiles((KernelWorker*)&ak);
}

// XXX for DEBUG
void
FlagtileSurface::dbg_progress_single_level(const int level, const int perimeter) 
{
    ProgressKernel k(this); 
    k.set_target_level(level);
    filter_tiles((KernelWorker*)&k);
    
    if (perimeter > 0) {
        // CountPerimeterKernel just marks such needless
        // pixels, not remove it yet.
        CountPerimeterKernel ck(this, level, perimeter, false, false);
        filter_tiles((KernelWorker*)&ck);
        
        // RemoveGarbageKernel actually remove
        // that marked areas.
        RemoveGarbageKernel rk(this, level);
        filter_tiles((KernelWorker*)&rk);    
    }
}

//--------------------------------------
// FloodfillSurface class

FloodfillSurface::FloodfillSurface(PyObject* tiledict, 
                                   const int start_level) 
    : FlagtileSurface(start_level, PIXEL_AREA)
{
#ifdef HEAVY_DEBUG
    assert(PyDict_Check(tiledict));
    assert(m_level <= MAX_PROGRESS);
    assert(PROGRESS_TILE_SIZE(0) == MYPAINT_TILE_SIZE);
#endif
    // Build from tiledict.
    // That tiledict would be a dictionary which has 
    // key as tuple (x, y).

    PyObject *keys = PyDict_Keys(tiledict);// New reference
    int length = PyObject_Length(keys);
    int min_x;
    int min_y;
    int max_x;
    int max_y;

    // Get surface dimension and generate it.
    for (int i=0; i < length; i++ ){
        PyObject *ck = PyList_GetItem(keys, i);
#ifdef HEAVY_DEBUG
    assert(PyTuple_Check(ck));
    assert(ck != NULL);
#endif
        int cx = (int)PyInt_AsLong(PyTuple_GetItem(ck, 0));
        int cy = (int)PyInt_AsLong(PyTuple_GetItem(ck, 1));
        if(i == 0) {
            min_x = cx;
            max_x = cx;
            min_y = cy;
            max_y = cy;
        }
        else {
            min_x = MIN(cx, min_x);
            min_y = MIN(cy, min_y);
            max_x = MAX(cx, max_x);
            max_y = MAX(cy, max_y);
        }
    }

    _generate_tileptr_buf(
        min_x, min_y,
        (max_x - min_x) + 1,
        (max_y - min_y) + 1
    );

    // Incref the source dictionary
    // to ensure the flagtile exists
    // at destructor.
    m_src_dict = tiledict;
    Py_INCREF(m_src_dict);
#ifdef HEAVY_DEBUG
    assert(m_src_dict->ob_refcnt >= 2);
#endif
}

FloodfillSurface::~FloodfillSurface() 
{
    for(int i=0; i<m_width*m_height; i++) {
        Flagtile *ct = m_tiles[i];
        // Borrowed tile MUST NOT be deleted.
        if (ct != NULL
                && (ct->get_stat() & Flagtile::BORROWED)!=0)
            m_tiles[i] = NULL; // So, just `hide` it.
    }
    // m_tiles array would be deleted at parent destructor.

    // Do not forget to decref source dictionary.
    Py_DECREF(m_src_dict);

#ifdef HEAVY_DEBUG
    printf("Floodfill destructor called.\n");
#endif
}

void
FloodfillSurface::borrow_tile(const int tx, const int ty, Flagtile* tile)
{
    int idx = _get_tile_index(tx, ty);
    m_tiles[idx] = tile;
    tile->set_stat(Flagtile::BORROWED);
}

//--------------------------------------
// Close and fill
/**
* @Constructor
*
* @param ox, oy: Origin of tiles, in mypaint tiles. not pixels. 
* @param w, h: Surface width and height, in mypaint tiles. not pixels.
*/
ClosefillSurface::ClosefillSurface(const int start_level,
                                   PyObject* node_list) 
    : FlagtileSurface(start_level, 0), m_nodes(NULL) 
{
#ifdef HEAVY_DEBUG
    assert(m_level <= MAX_PROGRESS);
    assert(PROGRESS_TILE_SIZE(0) == MYPAINT_TILE_SIZE);
    assert(node_list != NULL);
#endif
    _init_nodes(node_list);
    _scanline_fill();
}

ClosefillSurface::~ClosefillSurface(){
#ifdef HEAVY_DEBUG
    assert(m_nodes != NULL);
#endif
    delete[] m_nodes;
}

void
ClosefillSurface::_init_nodes(PyObject* node_list) 
{
#ifdef HEAVY_DEBUG
    assert(PyList_Check(node_list));
    assert(m_nodes == NULL);
#endif
    int max_count = PyObject_Length(node_list);


#if PY_VERSION_HEX >= 0x03000000
    PyObject* attr_x = PyUnicode_FromString("x"); 
    PyObject* attr_y = PyUnicode_FromString("y"); 
#else
    PyObject* attr_x = PyString_FromString("x");
    PyObject* attr_y = PyString_FromString("y");
#endif

    // Getting maximum dimension of area , to generate
    // internal buffers.
    PyObject* pynode = PyList_GetItem(node_list, 0);
    int x = (int)PyFloat_AsDouble(PyObject_GetAttr(pynode, attr_x));
    int y = (int)PyFloat_AsDouble(PyObject_GetAttr(pynode, attr_y));
    int px = !x;
    int py = !y;
    int min_x = x;
    int min_y = y;
    int max_x = x;
    int max_y = y;
    int actual_cnt = 0;

    for(int i=1; i < max_count; i++) {
        pynode = PyList_GetItem(node_list, i);
        x = (int)PyFloat_AsDouble(PyObject_GetAttr(pynode, attr_x));
        y = (int)PyFloat_AsDouble(PyObject_GetAttr(pynode, attr_y));
        if(px != x || py != y) {
            min_x = MIN(x, min_x);
            min_y = MIN(y, min_y);
            max_x = MAX(x, max_x);
            max_y = MAX(y, max_y);
            px = x;
            py = y;
            actual_cnt++;
        }
    }

    min_x = (min_x / TILE_SIZE) * TILE_SIZE;
    min_y = (min_y / TILE_SIZE) * TILE_SIZE;
    max_x = ((max_x / TILE_SIZE) + 1) * TILE_SIZE;
    max_y = ((max_y / TILE_SIZE) + 1) * TILE_SIZE;

    _generate_tileptr_buf(
        min_x / TILE_SIZE, 
        min_y / TILE_SIZE, 
        (max_x - min_x) / TILE_SIZE, 
        (max_y - min_y) / TILE_SIZE 
    );

    // After _generate_tileptr_buf,
    // we can use m_ox, m_oy member.
    int opx = m_ox * PROGRESS_TILE_SIZE(0);
    int opy = m_oy * PROGRESS_TILE_SIZE(0);

    m_nodes = new _progfill_point[actual_cnt];
    m_node_cnt = actual_cnt;
    m_cur_node = 0;
    
    // Enumerate nodes again and
    // Fetching & Drawing polygon edge.
    DrawLineWorker dw(this, PIXEL_AREA);
    pynode = PyList_GetItem(node_list, 0);
    // x and y attributes of node are 
    // in original mypaint layer(surface) coordinate.
    // To convert internal FlagtileSurface coordinate,
    // We need substract origin from them.
    x = (int)PyFloat_AsDouble(PyObject_GetAttr(pynode, attr_x)) - opx;
    y = (int)PyFloat_AsDouble(PyObject_GetAttr(pynode, attr_y)) - opy;
    px = !x;
    py = !y;

    _move_to(x, y);
    for(int i=1; i < m_node_cnt; i++) {
        pynode = PyList_GetItem(node_list, i);
        x = (int)PyFloat_AsDouble(PyObject_GetAttr(pynode, attr_x)) - opx;
        y = (int)PyFloat_AsDouble(PyObject_GetAttr(pynode, attr_y)) - opy;
        if(px != x || py != y) {
            _line_to((DrawWorker*)&dw, x, y, false);
            px = x;
            py = y;
        }
    }
    _close_line((DrawWorker*)&dw);

    Py_DECREF(attr_x);
    Py_DECREF(attr_y);
}

/**
* @_walk_line
* walk virtual flag line into the surface.
*
* @param sx, sy, ex, ey : the start and end point of line segment.
* @param f: Pixel worker to do something at line pixels.
* @detail 
* Drawing virtual flag line into the surface.
* If there is any flag in pixel (i.e. intersected)
*/
void 
ClosefillSurface::_walk_line(int sx, int sy, 
                             int ex, int ey,
                             DrawWorker *f)
{
    // draw virtual line of bitwise flag, 
    // by using Bresenham algorithm.
    int dx = abs(ex - sx) + 1;
    int dy = abs(ey - sy) + 1;
    int x, y, error, xs=1, ys=1;
    if (sx > ex) {
        xs = -1;
    }
    if (sy > ey) {
        ys = -1;
    }

    if (dx < dy) {
        // steep angled line
        x = sx;
        y = sy;
        error = dy;
        for (int cy=0; cy < dy; cy++){
            f->step(x, y);
            error -= dx;
            if (error < 0) {
                x+=xs;
                error = error + dy;
            }
            y+=ys;
        }
    }
    else {
        x = sx;
        y = sy;
        error = dx;
        for (int cx=0; cx < dx; cx++){
            f->step(x, y);
            error -= dy;
            if (error < 0) {
                y+=ys;
                error = error + dx;
            }
            x += xs;
        }
    }

    // Ensure the exact last pixel should be drawn
    f->step(ex, ey);
}

/**
* @_walk_polygon
* walk closed-area polygon edge.
*
* @detail 
* This method walks around polygon edge, 
* in PROGRESS LEVEL 0, with assigned DrawWorker.
* So the worker MUST convert coordinates
* for its own progress level.
*/
void 
ClosefillSurface::_walk_polygon(DrawWorker* w) 
{

    for (int i=0; i < m_node_cnt-1; i++) {
        _walk_line(
            m_nodes[i].x, m_nodes[i].y,
            m_nodes[i+1].x, m_nodes[i+1].y,
            w
        );
    }

    _walk_line(
        m_nodes[m_node_cnt-1].x, m_nodes[m_node_cnt-1].y, 
        m_nodes[0].x, m_nodes[0].y,
        w
    );
}

/**
* @_move_to
* initialize line drawing.
*
* @param sx, sy start point of lines.
* @detail 
* We must call this method at the start of
* line drawing.
*
*/
void 
ClosefillSurface::_move_to(const int sx, const int sy) 
{
    m_nodes[0].x = sx;
    m_nodes[0].y = sy;
    m_cur_node = 1;
}

/**
* @_line_to
* draw virtual flag line into the surface.
*
* @param ex, ey the end point of current line segment.
* @param reversed reverse line flags, to deal with intersection.
* @return true when intersection(overwrite) detected.
* @detail 
*
*/
void 
ClosefillSurface::_line_to(DrawWorker* w, 
                           const int ex, const int ey, 
                           const bool closing) 
{
#ifdef HEAVY_DEBUG
    assert(m_cur_node > 0);
    if (closing)
        assert(m_cur_node <= m_node_cnt);
    else
        assert(m_cur_node < m_node_cnt);
#endif
    
    int sx = m_nodes[m_cur_node-1].x;
    int sy = m_nodes[m_cur_node-1].y;

    _walk_line(sx, sy,
               ex, ey, 
               w);

    if (!closing) {
        m_nodes[m_cur_node].x = ex;
        m_nodes[m_cur_node].y = ey;
        m_cur_node++;
    }
}

/**
* @_close_line
* draw last segment of virtual flag line and close the
* area.
*
* @detail 
* We must call this method at the end of
* line drawing.
*/
void 
ClosefillSurface::_close_line(DrawWorker* w) 
{
    _line_to(
        w,
        m_nodes[0].x, 
        m_nodes[0].y,
        true
    );
}

/**
* @_is_inside_polygon
* tells whether the points are inside polygon or not.
*
* @param x1 start x coordinate of horizontal line.
* @param x2 end x coordinate of horizontal line.
* @param y  y coordinate of horizontal line.
* @detail 
* This codes from `Even-odd rule` of wikipedia
* https://en.wikipedia.org/wiki/Even%E2%80%93odd_rule 
*/
bool 
ClosefillSurface::_is_inside_polygon(const int x1, const int x2, const int y)
{
  int j = m_node_cnt - 1;
  bool c1 = false;
  bool c2 = false;
  for (int i=0; i < m_node_cnt; i++) {
      int sx = m_nodes[i].x;
      int sy = m_nodes[i].y;
      int ex = m_nodes[j].x;
      int ey = m_nodes[j].y;
      
      if (_is_inside_polygon_point(x1, y, sx, sy, ex, ey))
            c1 = !c1;

      if (_is_inside_polygon_point(x2, y, sx, sy, ex, ey))
            c2 = !c2;

      j = i;
  }
  return (c1 && c2);
}

/**
* @_scanline_fill
* Fill already drawn polygon edge along horizontal scanline.
*
* @detail 
* Call this method after m_nodes array initialized
* and polygon edges drawn.
*/
void 
ClosefillSurface::_scanline_fill() 
{
    uint8_t pix;
    static const int SKIPPING = 0;// Initial state of scanline.
    static const int START = 1;
    static const int DRAWING = 2;
    int state;
    int start_x;
    int ptx = -1;
    int pty = -1;
    
    // scanline fill operation MUST be done at progress level 0.
    // not m_level.
    int tile_size = PROGRESS_TILE_SIZE(0); 

    for(int y=0; y<get_pixel_max_y(0); y++) {
        state = SKIPPING;
        for(int x=0; x<get_pixel_max_x(0); x++) {
            pix = get_pixel(0, x, y);
            switch(state) {
                case SKIPPING:
                    if (pix == PIXEL_AREA) {
                        state = START;
                    }
                    break;
                case START:
                    if (pix == PIXEL_EMPTY) {
                        state = DRAWING;
                        start_x = x;
                    }
                    break;
                case DRAWING:
                    if (pix == PIXEL_AREA) {
                        // Due to bresenham precision error, we might get
                        // 1 px wrong value to assigned position.
                        if (_is_inside_polygon(start_x+1, x-2, y)) {
                            int tx;
                            int ty = y / tile_size;
                            for(int px=start_x; px<x; px++) {
                                replace_pixel(0, px, y, PIXEL_AREA);
                                
                                // Update tile status, if needed.
                                tx = px / tile_size;
                                if (ptx != tx || pty != ty) {
                                    Flagtile *tile = get_tile(tx, ty, false);
#ifdef HEAVY_DEBUG
                                    assert(tile != NULL);
#endif
                                    tile->unset_stat(Flagtile::EMPTY);
                                    ptx = tx;
                                    pty = ty;
                                }                                
                            }
                        }
                        // Reset state.
                        // From now on, we reject outside polygon pixels
                        // with _is_inside_polygon method,
                        // instead of using SKIPPING state.
                        // Because there might be complicatedly crossed polygon. 
                        // SKIPPING state might skip pixels in such case mistakenly.
                        state = START;
                    }
                    break;
                default:
                    // Should not come here.
                    assert(false);
                    break;
            }
        }
    }
}

/**
* @decide_area
* Decide outside area
*
* @detail 
* Fill outside pixels with PIXEL_OUTSIDE and 
* fill inside-contour pixels with PIXEL_FILLED 
* at current progress level.
* With this, we decide outside/inside of current
* closed area roughly, and, as a side effect, 
* closing gap of contour by progress pixel size.
*
* After this method called, we would call progress_tiles method
* and gradually progress(reshape) filled pixels.
*/
void 
ClosefillSurface::decide_area() 
{
    // Walk around outer rim of closing area polygon,
    // and trigger flood-fill of PIXEL_OUTSIDE 
    // when vacant area pixel found.
    // With this, we fill progress pixel of outside contours with
    // PIXEL_OUTSIDE.
    DecideTriggerWorker w(this, m_level);   // Worker for trigger flood-fill
    _walk_polygon(&w);

    // Then, convert remained vacant PIXEL_AREA into PIXEL_FILLED.
    // With this, we fill the inside of contour with PIXEL_FILLED.
    ConvertKernel k(this, m_level, PIXEL_AREA, PIXEL_FILLED);
    filter_tiles((KernelWorker*)&k);
}

//--------------------------------------
// Lasso fill

LassofillSurface::LassofillSurface(PyObject* node_list) 
    :  ClosefillSurface(0, node_list)
{
}

LassofillSurface::~LassofillSurface() 
{
#ifdef HEAVY_DEBUG
    printf("LassofillSurface destructor called.\n");
#endif
}

/**
* @finalize
* overrided Finalize method.
*
* @param threshold: The threshold value of filled area perimeter.
*                   If an area which perimeter is less than this value,
*                   that area is removed(filled with PIXEL_AREA)
* @detail 
* This is a dedicated version of finalize of Lassofill.
*/
/*
void 
LassofillSurface::finalize(const int dilation_size,
                           const bool antialias,
                           const bool fill_all_holes)
{
#ifdef HEAVY_DEBUG
    assert(dilation_size >= 0);
    assert(dilation_size < MYPAINT_TILE_SIZE);
#endif
    
    if (fill_all_holes) {
        // Use CountPerimeterKernel with threshold 0,
        // This means `disable small area rejection`
        // So this kernel only do `fill_all_holes` functionality.
        CountPerimeterKernel ck(this, 0, 0, true, fill_all_holes);
        filter_tiles((KernelWorker*)&ck);
    }
    
    // Before converting pixels, dilate it.
    // But, for Lassofill dilation is something 
    // complicated.
    // We needs not dilating PIXEL_FILLED
    // but eroding PIXEL_CONTOUR pixels. 
    // Otherwise, dilated pixels would exceed the 
    // lasso area.
    if (dilation_size > 0) {        
        for(int i=0; i<dilation_size; i++) {
            ErodeContourKernel ek(this);
            filter_tiles((KernelWorker*)&ek);
        }
    }
    
    // for this surface, just convert PIXEL_AREA
    // to PIXEL_FILLED completes fill the area.
    ConvertKernel ck(this, 0, PIXEL_CONTOUR, PIXEL_OUTSIDE);
    filter_tiles((KernelWorker*)&ck);
    ck.setup(0, PIXEL_AREA, PIXEL_FILLED);
    filter_tiles((KernelWorker*)&ck);
    
    // Doing Psuedo Antialias for result pixels
    if (antialias) {
        AntialiasKernel ak(this);
        filter_tiles((KernelWorker*)&ak);
    }
     
}
*/

/**
* @convert_result_area
* convert PIXEL_AREA as final result(PIXEL_FILLED)
*
* @detail 
* Just convert valid pixel area as filled area.
* This method should be called next to `dilate` method 
* and prior to `draw_antialias` method,
*/
void
LassofillSurface::convert_result_area()
{
    ConvertKernel ck(this, 0, PIXEL_CONTOUR, PIXEL_OUTSIDE);
    filter_tiles((KernelWorker*)&ck);
    ck.setup(0, PIXEL_AREA, PIXEL_FILLED);
    filter_tiles((KernelWorker*)&ck);
}

//-----------------------------------------------------------------------------
// Functions

/**
* @progfill_flood_fill
* Doing `progress-fill` into Flagtile.
*
* @param tile : A flag tile object. 
*               This object should be already build_progress called
*               before using this function.
* @param seeds: A list of tuple, that tuple is (x, y) of floodfill
*               seed points in PROGRESS coordinate.
* @param min_x, min_y, max_x, max_y : Surface border within tile, in PROGRESS coordinate.
* @param level : The target progress level.
* @return desc_of_return
* @detail
* Used for floodfill tool, to implement gap-closing functionality.
* This is not exactly same as close-and-fill.
*
*/
PyObject *
progfill_flood_fill (Flagtile *tile, /* target flagtile object */
                     PyObject *seeds, /* List of 2-tuples */
                     int min_x, int min_y, int max_x, int max_y,
                     int level)
{
    
    // XXX Code duplication most parts from fill.cpp

#ifdef HEAVY_DEBUG
    assert(tile != NULL);
    assert(PySequence_Check(seeds));
#endif

    // Exit when the tile already filled up.
    if (tile->get_stat() & Flagtile::FILLED)
        Py_RETURN_NONE;

    const int tile_size = PROGRESS_TILE_SIZE(level);

    if (min_x < 0) min_x = 0;
    if (min_y < 0) min_y = 0;
    if (max_x > tile_size-1) max_x = tile_size-1;
    if (max_y > tile_size-1) max_y = tile_size-1;
    if (min_x > max_x || min_y > max_y) {
        return Py_BuildValue("[()()()()]");
    }
    // Populate a working queue with seeds
    int x = 0;
    int y = 0;
    GQueue *queue = g_queue_new();   /* Of tuples, to be exhausted */
    for (int i=0; i<PySequence_Size(seeds); ++i) {
        PyObject *seed_tup = PySequence_GetItem(seeds, i);
#ifdef HEAVY_DEBUG
        assert(PySequence_Size(seed_tup) == 2);
#endif
        if (! PyArg_ParseTuple(seed_tup, "ii", &x, &y)) {
            continue;
        }
        Py_DECREF(seed_tup);
        x = MAX(0, MIN(x, tile_size-1));
        y = MAX(0, MIN(y, tile_size-1));
        uint8_t pix = tile->get(level, x, y);
        if (pix == PIXEL_AREA) {
            _progfill_point *seed_pt = (_progfill_point*)
                                          malloc(sizeof(_progfill_point));
            seed_pt->x = x;
            seed_pt->y = y;
            g_queue_push_tail(queue, seed_pt);
        }
    }

    PyObject *result_n = PyList_New(0);
    PyObject *result_e = PyList_New(0);
    PyObject *result_s = PyList_New(0);
    PyObject *result_w = PyList_New(0);


    // Instantly fill up when the tile is empty(filled by PIXEL_AREA).
    if (tile->get_stat() & Flagtile::FILLED_AREA) {
        
        tile->fill(PIXEL_FILLED);
        
        _progfill_point *pos = (_progfill_point*) g_queue_pop_head(queue);
        PyObject* result = result_w;
        int dx = tile_size - 1;

        for(int x=0; x<=tile_size-1; x+=(tile_size-1)) {
            for(int y=0; y<=tile_size-1; y++) {
                if (pos->x != x && pos->y != y) {
                    PyObject *s = Py_BuildValue("ii", dx, y);
                    PyList_Append(result, s);
                    Py_DECREF(s);
#ifdef HEAVY_DEBUG
                    assert(s->ob_refcnt == 1);
#endif            
                }
            }    
            result = result_e;
            dx = 0;
        }
        
        result = result_n;
        int dy = tile_size - 1;
        for(int y=0; y<=tile_size-1; y+=(tile_size-1)) {
            for(int x=0; x<=tile_size-1; x++) {
                if (pos->x != x && pos->y != y) {
                    PyObject *s = Py_BuildValue("ii", x, dy);
                    PyList_Append(result, s);
                    Py_DECREF(s);
#ifdef HEAVY_DEBUG
                    assert(s->ob_refcnt == 1);
#endif            
                }
            }    
            result = result_s;    
            dy = 0;
        }    
        
        free(pos);
    }
    else {
        // Ordinary flood-fill
        int filled_cnt = 0;
        int contour_cnt = 0;
                
        while (! g_queue_is_empty(queue)) {
            _progfill_point *pos = (_progfill_point*) g_queue_pop_head(queue);
            int x0 = pos->x;
            int y = pos->y;
            free(pos);
            // Find easternmost and westernmost points of the same colour
            // Westwards loop includes (x,y), eastwards ignores it.
            static const int x_delta[] = {-1, 1};
            static const int x_offset[] = {0, 1};
            for (int i=0; i<2; ++i)
            {
                bool look_above = true;
                bool look_below = true;
                for ( int x = x0 + x_offset[i] ;
                      x >= min_x && x <= max_x ;
                      x += x_delta[i] )
                {
                    uint8_t pix = tile->get(level, x, y);
                    if (x != x0) { // Test was already done for queued pixels
                        if (pix != PIXEL_AREA)
                        {
                            if(pix == PIXEL_CONTOUR)
                                contour_cnt++;
                            break;
                        }
                    }
                    // Also halt if we're outside the bbox range
                    if (x < min_x || y < min_y || x > max_x || y > max_y) {
                        break;
                    }
                    // Fill this pixel, and continue iterating in this direction
                    tile->replace(level, x, y, PIXEL_FILLED);
                    filled_cnt++;
                    // In addition, enqueue the pixels above and below.
                    // Scanline algorithm here to avoid some pointless queue faff.
                    if (y > 0) {
                        uint8_t pix_above = tile->get(level, x, y-1);
                        if (pix_above == PIXEL_AREA) {
                            if (look_above) {
                                // Enqueue the pixel to the north
                                _progfill_point *p = (_progfill_point *) malloc(
                                                        sizeof(_progfill_point)
                                                      );
                                p->x = x;
                                p->y = y-1;
                                g_queue_push_tail(queue, p);
                                look_above = false;
                            }
                        }
                        else { // !match_above
                            look_above = true;
                        }
                    }
                    else {
                        // Overflow onto the tile to the North.
                        // Scanlining not possible here: pixel is over the border.
                        PyObject *s = Py_BuildValue("ii", x, tile_size-1);
                        PyList_Append(result_n, s);
                        Py_DECREF(s);
    #ifdef HEAVY_DEBUG
                        assert(s->ob_refcnt == 1);
    #endif
                    }
                    if (y < tile_size - 1) {
                        uint8_t pix_below = tile->get(level, x, y+1);
                        if (pix_below == PIXEL_AREA) {
                            if (look_below) {
                                // Enqueue the pixel to the South
                                _progfill_point *p = (_progfill_point *) malloc(
                                                        sizeof(_progfill_point)
                                                      );
                                p->x = x;
                                p->y = y+1;
                                g_queue_push_tail(queue, p);
                                look_below = false;
                            }
                        }
                        else { //!match_below
                            look_below = true;
                        }
                    }
                    else {
                        // Overflow onto the tile to the South
                        // Scanlining not possible here: pixel is over the border.
                        PyObject *s = Py_BuildValue("ii", x, 0);
                        PyList_Append(result_s, s);
                        Py_DECREF(s);
    #ifdef HEAVY_DEBUG
                        assert(s->ob_refcnt == 1);
    #endif
                    }
                    // If the fill is now at the west or east extreme, we have
                    // overflowed there too.  Seed West and East tiles.
                    if (x == 0) {
                        PyObject *s = Py_BuildValue("ii", tile_size-1, y);
                        PyList_Append(result_w, s);
                        Py_DECREF(s);
    #ifdef HEAVY_DEBUG
                        assert(s->ob_refcnt == 1);
    #endif
                    }
                    else if (x == tile_size-1) {
                        PyObject *s = Py_BuildValue("ii", 0, y);
                        PyList_Append(result_e, s);
                        Py_DECREF(s);
    #ifdef HEAVY_DEBUG
                        assert(s->ob_refcnt == 1);
    #endif
                    }
                }
            }
        }
        
            // Refresh filled status.
            // If there is at least one filled pixel or contour,
            // It should be set HAS_PIXEL stat flag.
            if(filled_cnt > 0
                    || contour_cnt > 0)
                tile->set_stat(Flagtile::HAS_PIXEL);
    }
    // Clean up working state, and return where the fill has overflowed
    // into neighbouring tiles.
    g_queue_free(queue);
    PyObject *result = Py_BuildValue("[OOOO]", result_n, result_e,
                                               result_s, result_w);
    Py_DECREF(result_n);
    Py_DECREF(result_e);
    Py_DECREF(result_s);
    Py_DECREF(result_w);
#ifdef HEAVY_DEBUG
    assert(result_n->ob_refcnt == 1);
    assert(result_e->ob_refcnt == 1);
    assert(result_s->ob_refcnt == 1);
    assert(result_w->ob_refcnt == 1);
    assert(result->ob_refcnt == 1);
#endif

    return result;
}

//-----------------------------------------------------------------------------
/// XXX for debug

// XXX render_to_numpy exists even HEAVY_DEBUG is not defined. 
// this should be removed in release.
PyObject*
FlagtileSurface::render_to_numpy(PyObject *npbuf,
                                 int tflag,
                                 int tr, int tg, int tb)
{
    int w = get_width(); 
    int h = get_height();

    PyArrayObject *array;
    if (npbuf != Py_None) {
        array = (PyArrayObject*)npbuf;
        Py_INCREF(array);
    }     
    else {
        npy_intp dims[] = {
            h * TILE_SIZE, 
            w * TILE_SIZE, 
            3
        };
        array = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_UINT8, 0);
    }

    const unsigned int xstride = PyArray_STRIDE(array, 1);
    const unsigned int ystride = PyArray_STRIDE(array, 0);
    uint8_t *tptr;
    uint8_t *lptr;
    uint8_t *baseptr = (uint8_t*)PyArray_BYTES(array);

    int px;
    int py=0;

    for(int ty=0; ty<h; ty++) {
        px = 0;
        for(int tx=0; tx<w; tx++) {
            Flagtile *t = get_tile(tx, ty, false);
            if ( t != NULL) {
                lptr = baseptr + (ystride * py + xstride * px);
                for (int y=0; y < TILE_SIZE; y++) {
                    tptr = lptr;
                    for (int x=0; x < TILE_SIZE; x++) {
                        if (t->get(0, x, y) == tflag) {
                            tptr[0] = (uint8_t)tr;
                            tptr[1] = (uint8_t)tg;
                            tptr[2] = (uint8_t)tb;
                        }
                        tptr+=xstride;
                    }
                    lptr+=ystride;
                }
            }
            px+=TILE_SIZE;
        }
        py+=TILE_SIZE;
    }
    return (PyObject*)array;
}
#ifdef HEAVY_DEBUG
PyObject*
progfill_render_to_numpy(FlagtileSurface *surf,
                         PyObject *npbuf,
                         int tflag,
                         int level,
                         int tr, int tg, int tb)
{
    int w = surf->get_width(); 
    int h = surf->get_height();

    PyArrayObject *array;
    if (npbuf != Py_None) {
        array = (PyArrayObject*)npbuf;
        Py_INCREF(array);
    }     
    else {
        npy_intp dims[] = {
            h * TILE_SIZE, 
            w * TILE_SIZE, 
            3
        };
        array = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_UINT8, 0);
    }

    const unsigned int xstride = PyArray_STRIDE(array, 1);
    const unsigned int ystride = PyArray_STRIDE(array, 0);
    uint8_t *tptr;
    uint8_t *lptr;
    uint8_t *baseptr = (uint8_t*)PyArray_BYTES(array);

    int px;
    int py=0;

    if (level > 0) {
        // Target level is greater than 0.
        // We need to draw progress chunkey pixels.
        int step = 1 << level;
        for(int ty=0; ty<h; ty++) {
            px = 0;
            for(int tx=0; tx<w; tx++) {
                Flagtile *t = surf->get_tile(tx, ty, false);
                if ( t != NULL) {
                    lptr = baseptr + (ystride * py + xstride * px);
                    for (int y=0; y < TILE_SIZE; y+=step) {
                        tptr = lptr;
                        for (int x=0; x < TILE_SIZE; x+=step) {
                            int mx = x >> level;
                            int my = y >> level;

                            if (t->get(level, mx, my) == tflag) {
                                // Fill a progress chunk.
                                uint8_t *byptr = tptr;
                                for(int cy=0; cy < step; cy ++) {
                                    uint8_t *bxptr = byptr;
                                    for(int cx=0; cx < step; cx ++) {
                                        bxptr[0] = (uint8_t)tr;
                                        bxptr[1] = (uint8_t)tg;
                                        bxptr[2] = (uint8_t)tb;
                                        bxptr += xstride;
                                    }
                                    byptr += ystride;
                                }
                            }
                            tptr+=(xstride * step);
                        }
                        lptr+=(ystride * step);
                    }
                }
                px += TILE_SIZE;
            }
            py += TILE_SIZE;
        }
    }
    else if(level == -1) {
        // XXX Antialiasing debugging test: remove later!
        // The pixel appearance number array.
        int pixel_counts[MAX_AA_LEVEL];
        memset(pixel_counts, 0, MAX_AA_LEVEL * sizeof(int));

        for(int ty=0; ty<h; ty++) {
            px = 0;
            for(int tx=0; tx<w; tx++) {
                Flagtile *t = surf->get_tile(tx, ty, false);
                if ( t != NULL) {
                    lptr = baseptr + (ystride * py + xstride * px);
                    for (int y=0; y < TILE_SIZE; y++) {
                        tptr = lptr;
                        for (int x=0; x < TILE_SIZE; x++) {
                            uint8_t pix = t->get(0, x, y);
                            if (pix & FLAG_AA) {
                                pix &= AA_MASK;
                                double alpha = (double)pix / MAX_AA_LEVEL;
                                alpha *= 0.9; // Adjust alpha(looks too opaque)
                                // Add 10 every color channel
                                // to distinguish with non-AA pixel area.
                                tptr[0] = (uint8_t)(tr * alpha) + 10;
                                tptr[1] = (uint8_t)(tg * alpha) + 10;
                                tptr[2] = (uint8_t)(tb * alpha) + 10;

                                // Increase pixel appearance count.
                                pixel_counts[pix]++;
                            }
                            else if ((pix & PIXEL_MASK) == PIXEL_FILLED) {
                                tptr[0] = (uint8_t)tr;
                                tptr[1] = (uint8_t)tg;
                                tptr[2] = (uint8_t)tb;
                            }
                            tptr+=xstride;
                        }
                        lptr+=ystride;
                    }
                }
                px += TILE_SIZE;
            }
            py += TILE_SIZE;
        }
    }
    else {
        // Otherwise, we draw ordinary non-scaled pixel.
        for(int ty=0; ty<h; ty++) {
            px = 0;
            for(int tx=0; tx<w; tx++) {
                Flagtile *t = surf->get_tile(tx, ty, false);
                if ( t != NULL) {
                    lptr = baseptr + (ystride * py + xstride * px);
                    for (int y=0; y < TILE_SIZE; y++) {
                        tptr = lptr;
                        for (int x=0; x < TILE_SIZE; x++) {
                            if (t->get(level, x, y) == tflag) {
                                tptr[0] = (uint8_t)tr;
                                tptr[1] = (uint8_t)tg;
                                tptr[2] = (uint8_t)tb;
                            }
                            tptr+=xstride;
                        }
                        lptr+=ystride;
                    }
                }
                px += TILE_SIZE;
            }
            py += TILE_SIZE;
        }
    }
    return (PyObject*)array;
}
#endif
