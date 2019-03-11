/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include "pyramidfill.hpp"
#include "pyramidworkers.hpp"
#include "common.hpp"
#include "fix15.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <mypaint-tiled-surface.h>
#include <math.h>
#include <glib.h>

#define TILE_SIZE MYPAINT_TILE_SIZE

//// Struct definition

//-----------------------------------------------------------------------------
//// Function definition


// XXX borrowed from `_floodfill_color_match` of lib/fill.cpp. almost same.
static inline fix15_t
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
// And most of derived worker classes are defined at lib/pyramidworkers.hpp

//----------------------------------------------------------------------------
/// KernelWorker

//// Shared empty tile related.
Flagtile *KernelWorker::m_shared_empty = NULL;

Flagtile*
KernelWorker::get_shared_empty()
{
    if (m_shared_empty == NULL)
        m_shared_empty = new Flagtile(PIXEL_EMPTY);
    return m_shared_empty;
}

bool
KernelWorker::sync_shared_empty(const int level)
{
    if (!m_shared_empty->is_filled_with(level, PIXEL_EMPTY)) {
        m_shared_empty = NULL;
        return true;
    }
    return false;
}

// Call this from FlagtileSurface destructor.
void
KernelWorker::free_shared_empty()
{
    delete m_shared_empty;
    m_shared_empty = NULL;
}

void
KernelWorker::set_target_level(const int level)
{
#ifdef HEAVY_DEBUG
    assert(level >= 0);
    assert(level <= MAX_PYRAMID);
#endif
    m_level = level;
}

// We can enumerate kernel window(surrounding) 4 pixels with for-loop
// by these offset, in the order of
//
// TOP, RIGHT, BOTTOM, LEFT (i.e. clockwise).
//
// Derived from WalkingKernel(especially AntialiasWalker) depends
// the order of this offsets.
// DO NOT CHANGE THIS ORDER.
const int KernelWorker::xoffset[] = { 0, 1, 0, -1};
const int KernelWorker::yoffset[] = {-1, 0, 1,  0};

// Wrapper method to get pixel with direction.
uint8_t
KernelWorker::get_pixel_with_direction(const int sx, const int sy,
                                       const int direction)
{
    return m_surf->get_pixel(
        m_level,
        sx + xoffset[direction],
        sy + yoffset[direction]
    );
}

//--------------------------------------
//// WalkingKernel class

// Rotate to right.
// This is used when we missed wall at right-hand.
void
WalkingKernel::rotate_left()
{
    // We need update current direction
    // before call rotation handler.
    m_cur_dir = get_hand_dir(m_cur_dir);
    on_rotate_cb(false);
}

// Rotate to left.
// This is used when we face `wall`
void
WalkingKernel::rotate_right()
{
    // We need update current direction
    // before call rotation handler
    m_cur_dir = get_reversed_hand_dir(m_cur_dir);
    m_right_rotate_cnt++;
    on_rotate_cb(true);
}

bool
WalkingKernel::forward()
{
    uint8_t pix;

    // see front.
    pix = get_front_pixel();

    if (is_wall_pixel(pix)) {
        // Face to wall.
        // Now, we must turn left.
        // With this turn, this kernel draws antialias line or
        // initialize internal status of this object.

        rotate_right();
        if (m_right_rotate_cnt >= 4)
            return false; // Exit from infnite loop of 1px hole!
    }
    else {
        // Then, forward.

        int nx = m_x + xoffset[m_cur_dir];
        int ny = m_y + yoffset[m_cur_dir];
        // Refreshing clockwise counter.
        // Algorithm from
        // https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
        // XXX CAUTION: mypaint uses inverted Cartesian coordinate system,
        // so the result is also inverted.
        // when m_clockwise_cnt is negative or zero, that area should be clockwise.
        m_clockwise_cnt += (nx - m_x) * (ny + m_y);

        m_x = nx;
        m_y = ny;
        m_step++;
        m_right_rotate_cnt = 0;

        if (m_x == m_ox && m_y == m_oy) {
            return false; // Walking end!!
        }

        on_new_pixel();
    }
    return true;
}

// Walk single step.
// when end walking, return false.
bool
WalkingKernel::proceed()
{
    if (!is_wall_pixel(get_hand_pixel())) {
        // Right hand of kernel misses the wall.
        // Couldn't forward.
        rotate_left();
    }
    return forward();
}

void
WalkingKernel::walk(const int sx, const int sy, const int direction)
{
#ifdef HEAVY_DEBUG
    unsigned int cnt = 0;
#endif
    m_ox = sx;
    m_oy = sy;
    m_x = sx;
    m_y = sy;
    m_step = 1;
    m_right_rotate_cnt = 0;
    m_clockwise_cnt = 0;

    m_cur_dir = direction;

    //  walk into staring point pixel.
    on_new_pixel();

    while (proceed()) {
#ifdef HEAVY_DEBUG
        cnt++;
        // `Walking over 100 million pixel` cannot happen.
        // It would be infinite loop bug.
        assert(cnt < 100000000);
#endif
    }
}

//--------------------------------------
//// Flagtile

const int
Flagtile::m_buf_offsets[MAX_PYRAMID+1] = {
    // pyramid level 0: 64x64(TILE_SIZE * TILE_SIZE), start from 0.
    0,
    PYRAMID_BUF_SIZE(0),
    Flagtile::m_buf_offsets[1] + PYRAMID_BUF_SIZE(1),
    Flagtile::m_buf_offsets[2] + PYRAMID_BUF_SIZE(2),
    Flagtile::m_buf_offsets[3] + PYRAMID_BUF_SIZE(3),
    Flagtile::m_buf_offsets[4] + PYRAMID_BUF_SIZE(4)
};

Flagtile::Flagtile(const int initial_value)
    : m_npbuf(NULL),
      m_statflag(0)
{
    m_buf = new uint8_t[FLAGTILE_BUF_SIZE];
    fill(initial_value);
}

Flagtile::~Flagtile()
{
    if (m_npbuf != NULL) 
        Py_DECREF(m_npbuf);
    
    delete[] m_buf;
}

void
Flagtile::propagate_upward_single(const int targ_level)
{
#ifdef HEAVY_DEBUG
    assert(targ_level >= 1);
#endif
    int c_size = PYRAMID_TILE_SIZE(targ_level);
    int b_level = targ_level - 1;

    // XXX To maximize parallel processing effeciency,
    // not using omp here.
    // It is done at FlagtileSurface::propagate_upward.
    for (int y=0; y < c_size; y++) {
        for (int x=0; x < c_size; x++) {
            // The pyramid level beneath is always
            // double sized of current level.
            int bbx = x << 1;
            int bby = y << 1;
            uint8_t new_pixel = PIXEL_EMPTY;

            // process chunk of pixels, like `pooling`
            for (int by=0; by < 2; by++) {
                for (int bx=0; bx < 2; bx++) {
                    uint8_t pix = get(b_level, bbx+bx, bby+by) & PIXEL_MASK;
                    switch(pix) {
                        case PIXEL_CONTOUR:
                        case PIXEL_OVERWRAP:
                            // Prior to every other pixels.
                            new_pixel = PIXEL_CONTOUR;
                            goto exit_pixel_loop;
                        case PIXEL_AREA:
                            new_pixel = PIXEL_AREA;
                            break;
                        case PIXEL_RESERVE:
                            if (new_pixel == PIXEL_EMPTY)
                                new_pixel = PIXEL_RESERVE;
                            break;
                    }
                }
            }

        exit_pixel_loop:
            if (new_pixel != PIXEL_EMPTY) {
                replace(
                    targ_level,
                    x, y,
                    new_pixel
                );
            }
        }
    }// Tile processing end
}

void
Flagtile::propagate_upward(const int max_level)
{
#ifdef HEAVY_DEBUG
    assert(max_level >= 1);
    assert(max_level <= MAX_PYRAMID);
#endif
    if (is_filled_with(0, PIXEL_INVALID))
        return; // There is nothing to do for empty tile.

    // In this stage, when the all of level-0 pixels are filled with same value
    // , but it might not be filled above level 0.
    // So, ensure the filled pixels such case.
    if (is_filled_with(0, PIXEL_AREA)) {
        if (get(max_level, 0, 0) != PIXEL_AREA)
            fill(PIXEL_AREA);
    }
    else if (is_filled_with(0, PIXEL_CONTOUR)) {
        if (get(max_level, 0, 0) != PIXEL_CONTOUR)
            fill(PIXEL_CONTOUR);
    }
    else if (is_filled_with(0, PIXEL_OUTSIDE)) {
        if (get(max_level, 0, 0) != PIXEL_OUTSIDE)
            fill(PIXEL_OUTSIDE);
    }
    else {
        for(int i=1;i <= max_level; i++) {
            propagate_upward_single(i);
        }
    }
}

/**
* @convert_from_color
* convert color-tile into flagtile.
*
* @param tolerance Color-space tolerance of fillable area.
* @param alpha_threshold Fillable pixel threshould, in alpha component.
* @param limit_within_opaque  If true, generate contour pixel even transparent pixel.
*
* Convert color-tile into flagtile, to decide contour pixel.
* With a certain brush preset, freehand tool would draw almost invisible strokes
* even no stylus pressure is applied.
* It is difficult to reject such pixels with `tolerance` option.
* So, in addition to `tolerance`, use this parameter.
* Practically, alpha_threshold value is enough around 0.03.
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
    fix15_short_t *bptr = (fix15_short_t*)PyArray_BYTES(array);

    const fix15_short_t targ[4] = {
        fix15_short_clamp(targ_r),
        fix15_short_clamp(targ_g),
        fix15_short_clamp(targ_b),
        fix15_short_clamp(targ_a)
    };

    fix15_t f15_tolerance = (fix15_t)(tolerance * (double)fix15_one);
    fix15_t f15_threshold = (fix15_t)(alpha_threshold * (double)fix15_one);

    #pragma omp parallel for
    for(int y=0; y<TILE_SIZE; y++) {
        fix15_short_t *cptr = bptr + y * ystride;

        for(int x=0; x<TILE_SIZE; x++) {
            uint8_t pix = get(0, x, y);
            fix15_t alpha = (fix15_t)cptr[3];
            if (!limit_within_opaque || alpha > 0) {
                // Only some `basic pixel` (not PIXEL_EMPTY,
                // such as PIXEL_AREA or PIXEL_OUTSIDE)
                // should be convertable.
                if (pix != PIXEL_EMPTY && alpha >= f15_threshold) {
                    fix15_t match = floodfill_color_match(
                        targ, cptr,
                        f15_tolerance
                    );

                    if (match == 0) {
                        replace(0, x, y, PIXEL_CONTOUR);
                    }
                }
            }
            /*
            else {
                // Code reaches here when
                // limit_within_opaque is true and current pixel
                // is completely transparent.
                if (pix != PIXEL_EMPTY) {
                    replace(0, x, y, PIXEL_EMPTY);
                }
            }*/
            cptr += xstride;
        }
    }
}

/**
* @convert_to_color
* convert Flagtile to mypaint colortile.
*
* @param py_targ_tile Mypaint colortile(numpy array of uint8)
* @param r,g,b  Target pixel color, to be converted from flag.
* @param pixel  Value of Flagtile pixel to be converted into mypaint tile.
*
* Convert Flagtile pixel into Mypaint color tile.
* Parameter r,g,b is pixel color. They are floating point value,
* from 0.0 to 1.0.
* This method would also render anti-aliasing pixels into mypaint colortile.
*
* For this method, array of py_targ_tile should be empty(Zero-filled).
* Actual pixel composition against target layer surface is done by
* lib.mypaintlib.combine_tile later.
*/
void
Flagtile::convert_to_color(PyObject *py_targ_tile,
                           const double r, const double g, const double b,
                           const int pixel)
{
    PyArrayObject *array = (PyArrayObject*)py_targ_tile;
#ifdef HEAVY_DEBUG
    assert_tile(array);
#endif

    const unsigned int xstride = PyArray_STRIDE(array, 1) / sizeof(fix15_short_t);
    const unsigned int ystride = PyArray_STRIDE(array, 0) / sizeof(fix15_short_t);
    fix15_short_t *cptr = (fix15_short_t*)PyArray_BYTES(array);

    // Premult color pixel array.
    fix15_short_t cols[4];
    cols[0] = fix15_short_clamp(r * fix15_one);
    cols[1] = fix15_short_clamp(g * fix15_one);
    cols[2] = fix15_short_clamp(b * fix15_one);
    cols[3] = fix15_one;

    // Completely filled tile can be filled with memcpy.
    // In this stage, we can know it from level 0 pixel count.
    if (is_filled_with(0, pixel)) {
        fix15_short_t *dptr = cptr;
        // Build one line
        for(int x=0; x<TILE_SIZE; x++) {
            dptr[0] = cols[0];
            dptr[1] = cols[1];
            dptr[2] = cols[2];
            dptr[3] = cols[3];
            dptr += xstride;
        }
        // Just copy that completed line.
        int y=1;
        while(y < TILE_SIZE) {
            dptr = cptr + y * ystride;
            memcpy(dptr, cptr, sizeof(fix15_short_t) * 4 * TILE_SIZE * y);
            y <<= 1;
        }
        return;
    }

    // Converting pyramid level 0 pixel into color tile.
    #pragma omp parallel for
    for(int y=0; y<TILE_SIZE; y++) {
        fix15_short_t *dptr = cptr + y * ystride;

        for(int x=0; x<TILE_SIZE; x++) {
            uint8_t pix = get(0, x, y);
            // At first, We need to check anti-aliasing flag of pixel.
            // Anti-aliasing pixel is not compatible
            // with another PIXEL_* values.
            if ((pix & FLAG_AA) != 0) {
                double alpha = get_aa_double_value(pix & AA_MASK);
                fix15_short_t cur_alpha = fix15_short_clamp((alpha * fix15_one));
                dptr[0] = fix15_short_clamp(r * cur_alpha);
                dptr[1] = fix15_short_clamp(g * cur_alpha);
                dptr[2] = fix15_short_clamp(b * cur_alpha);
                dptr[3] = cur_alpha;
            }
            else if ((pix & PIXEL_MASK) == pixel) {
                dptr[0] = cols[0];
                dptr[1] = cols[1];
                dptr[2] = cols[2];
                dptr[3] = cols[3];
            }
            dptr += xstride;
        }
    }
}

/**
* @convert_from_transparency
* convert mypaint colortile to Flagtile, only using alpha pixel value.
*
* @param alpha_threshold Alpha component threshold of target pixel.
* @param pixel_value Replacing pixel value, where colortile alpha pixel exceeds alpha_threshold.
* @param overwrap_value  If a pixel has already non-PIXEL_EMPTY value, place this value.
*
* Convert(replace) Mypaint colortile pixels into Flagtile pixels.
* A pixel which has larger alpha value than alpha_threshold is converted to
* pixel_value.
* Also, if there is already non-PIXEL_EMPTY(initial value of Flagtile), that pixel
* To disable overwrap_value, just assign same value with pixel_value.
*/
void
Flagtile::convert_from_transparency(PyObject *py_src_tile,
                                    const double alpha_threshold,
                                    const int pixel_value,
                                    const int overwrap_value)
{
    PyArrayObject *array = (PyArrayObject*)py_src_tile;
#ifdef HEAVY_DEBUG
    assert_tile(array);
    assert(pixel_value <= PIXEL_MASK);
    assert(overwrap_value <= PIXEL_MASK);
#endif

    const unsigned int xstride = PyArray_STRIDE(array, 1) / sizeof(fix15_short_t);
    const unsigned int ystride = PyArray_STRIDE(array, 0) / sizeof(fix15_short_t);
    fix15_short_t *cptr_base = (fix15_short_t*)PyArray_BYTES(array);
    fix15_short_t threshold = (fix15_short_t)(alpha_threshold * fix15_one);

    // Converting pyramid level 0 pixel into color tile.
    #pragma omp parallel for
    for(int y=0; y<TILE_SIZE; y++) {
        fix15_short_t *cptr = cptr_base + y * ystride;
        for(int x=0; x<TILE_SIZE; x++) {
            if (cptr[3] > threshold) {
                if (pixel_value != overwrap_value
                        && get(0, x, y) != PIXEL_EMPTY)
                    replace(0, x, y, (uint8_t)overwrap_value);
                else
                    replace(0, x, y, (uint8_t)pixel_value);
            }
            cptr += xstride;
        }
    }
}


void
Flagtile::fill(const uint8_t val)
{
#ifdef HEAVY_DEBUG
    assert((val & PIXEL_MASK) == val); // There is no any flag for pixel,
                                       // and also it is not special pixel value.
    assert(val <= PIXEL_MASK);
#endif
    // Fill only equal and above of assigned level.

    memset(m_buf, val, FLAGTILE_BUF_SIZE);
    memset(m_pixcnt, 0, sizeof(uint16_t) * PIXEL_MAX * (MAX_PYRAMID + 1));
    
    for(int l=0; l <= MAX_PYRAMID; l++) {
        int tile_size = PYRAMID_TILE_SIZE(l);
        m_pixcnt[l][val & PIXEL_MASK] = tile_size * tile_size;
    }
}

/**
* @to_nparray
* Get internal m_buf buffer as numpy.array
*
* @param readonly Swear that requested object is used as read-only.
* @return numpy array object.
*
* You can get and manipulate m_buf buffer pixel, via numpy API.
* That numpy array is just `view`, and you must discard(or just
* stop using) view-object before this Flagtile instance is released.
*
* Also, you can `swear` that view is used as readonly object.
* Although, actually you are still enable to change numpy-buffer flag and
* write change its contents, do not do it. 
*
* Use this method carefully.
*/
PyObject*
Flagtile::lock(const bool readonly)
{
#ifdef HEAVY_DEBUG
    assert(m_npbuf == NULL); 
#endif
    npy_intp dim = FLAGTILE_BUF_SIZE;
    m_npbuf = PyArray_SimpleNewFromData(1, &dim, NPY_UINT8, m_buf);
    Py_INCREF(m_npbuf);
    
    if (readonly) {
        PyArray_CLEARFLAGS((PyArrayObject*)m_npbuf, NPY_ARRAY_WRITEABLE);
    }
    else {
        m_statflag |= COUNT_DIRTY;
    }
    return m_npbuf;
}

/**
* @unlock
* Return the right of use numpy.array, which came from lock method.
*
* @param nparray The numpy array object, which is generated by lock method.
* @return True when that array object is released.
*
* You MUST call this method if you get a numpy.array object with lock method.
* Use this method carefully.
*/
void
Flagtile::unlock(PyObject *nparray)
{
    // Assert strictly, even not in HEAVY_DEBUG.
    assert(nparray == m_npbuf); 

    if (!PyArray_ISWRITEABLE((PyArrayObject*)m_npbuf)) {
        // View object might be changed to writable one. 
        // Force counting flag.
        m_statflag |= COUNT_DIRTY;
    }

    Py_DECREF(m_npbuf);
    m_npbuf = NULL;
}

//--------------------------------------
//// FlagtileSurface class.
//   psuedo surface object.
//   This manages Flagtile objects.

// Default constructor for derive class. actually, do not use.
FlagtileSurface::FlagtileSurface()
    : m_tiles(NULL)
{
#ifdef HEAVY_DEBUG
    // These are actually `static assert`. 
    // We MUST exactly compare them with 
    // MYPAINT_TILE_SIZE, not other macros such as `N` or `TILE_SIZE`.
    assert((1 << TILE_LOG) == MYPAINT_TILE_SIZE); // for Flagtile
    assert(sizeof(uint64_t) == (MYPAINT_TILE_SIZE/8)); // for EdgeQueue
#endif
}

FlagtileSurface::~FlagtileSurface()
{
    for(int i=0; i<m_width*m_height; i++) {
        delete m_tiles[i];
    }
    delete[] m_tiles;

    KernelWorker::free_shared_empty();
#ifdef HEAVY_DEBUG
    printf("FlagtileSurface destructor called.\n");
#endif
}

//// Internal methods
/**
* @generate_tileptr_buf
* Generate(allocate) tile pointer array buffer.
* Very important method.
*
* @details
* All parameters are in tile-unit.
* ox and oy are the origin point of this surface, in mypaint document(model).
* They are used to converting node positions of mypaint coordinate into
* Flagtilesurface local coodinate.
*
* m_width and m_height are minimum dimension of this surface.
*/
void
FlagtileSurface::generate_tileptr_buf(const int ox, const int oy,
                                      const int w, const int h)
{
#ifdef HEAVY_DEBUG
    assert(m_tiles == NULL);
    assert(w >= 1);
    assert(h >= 1);
#endif
    // Make Surface size larger than requested
    // by 1 tile for each direction.
    // Also, make origin 1 tile smaller.
    // Because, Dilation might exceed surface border.
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

/**
* @flood_fill
* Do `Flood fill` from
*
* @param level Target pyramid level to doing flood-fill
* @param sx,sy Flood-fill starting point, in pyramid level coordinate.
*
* Call step method of worker for each flood-fill point.
* Algorithm is copied/conveted from fill.cpp, but in this class
* we do not need to access `real` color pixels of mypaint surface.
* So we can complete flood-fill operation only in this class,
* without returning python seed tuples.
*/
void
FlagtileSurface::flood_fill(const int sx, const int sy, Filler* w)
{
    const int level = w->get_target_level();
    int tile_size = PYRAMID_TILE_SIZE(level);
    int tx = sx / tile_size;
    int ty = sy / tile_size;
    bool request = (w->match(PIXEL_EMPTY));
    if (tx < 0 && ty < 0 && tx >= get_width() && ty >= get_height()) {
        return;
    }

    GQueue *queue = g_queue_new();
    BorderQueue *q = new BorderQueue(
        tx, ty, sx%tile_size, sy%tile_size
    );
    g_queue_push_tail(queue, q);

    while (! g_queue_is_empty(queue)) {
        q = (BorderQueue*)g_queue_pop_head(queue);
        tx = q->get_tx();
        ty = q->get_ty();
#ifdef HEAVY_DEBUG
        assert(q != NULL);
        assert(tx >= 0 && ty >= 0 && tx < get_width() && ty < get_height()); 
#endif
        Flagtile *t = get_tile(tx, ty, request);
        if (t != NULL) {
            if (flagtile_flood_fill(t, q, w)) {
                for(int i=0; i < 4; i++) {
                    if (! q->is_empty_queue(i)) {
                        int ntx = BorderQueue::adjust_tx(tx, i);
                        int nty = BorderQueue::adjust_ty(ty, i);
                        if(ntx >= 0 && nty >= 0 && ntx < get_width() && nty < get_height()) { 
                            BorderQueue *nq = new BorderQueue(i, q, ntx, nty);
                            g_queue_push_tail(queue, nq);
                        }
                    }
                }
            }
        }
        delete q;
    }
    g_queue_free(queue);
}

/**
* @filter_tiles
* Internal method of tile pixel iteration.
*
* @param level The target pyramid level.
*
* This internal method is to iterate all pixels to worker classes.
*/
void
FlagtileSurface::filter_tiles(KernelWorker *w)
{
    int level = w->get_target_level();
    int tile_size = PYRAMID_TILE_SIZE(level);

    // This loop would trigger worker-object, which might
    // also refer/write pixels all over the FlagtileSurface.
    // Furthermore, This method utilize shared-empty tile
    // for workers can access even NULL tile area, generate
    // new tile on demand seamlessly, without requesting it.
    //
    // Therefore, we cannot use OpemMP here.
    for(int ty=0; ty<m_height; ty++) {
        for(int tx=0; tx<m_width; tx++) {
            // Get tile.
            Flagtile *t = get_tile(tx, ty, false);
            bool use_shared = (t==NULL);

            if (use_shared)
                t = KernelWorker::get_shared_empty();

            int bx = tx * tile_size;
            int by = ty * tile_size;
            if (w->start(t, bx, by)) {
                // iterate tile pixel.
                for (int y=0; y<tile_size; y++) {
                    for (int x=0; x<tile_size; x++) {
                        w->step(t, x, y, bx+x, by+y);
                    }
                }
                // iterate tile pixel end.
                // As a default, the `end` method
                // of KernelWorker would set
                // DIRTY stat flag for the tile.
                w->end(t);
            }

            if (use_shared) {
                if(KernelWorker::sync_shared_empty(level)) {
                    m_tiles[get_tile_index(tx, ty)] = t;
                }
            }
        }
    }
}

#ifdef _OPENMP
/**
* @filter_tiles_mp
* Internal method of tile pixel iteration, OpenMP enabled version.
*
* @param level The target pyramid level.
*
* This internal method is to iterate all pixels to worker classes.
*
* Different from original filter_tiles, this cannot use
* shared-empty tile. (i.e. NULL empty tile should not be targetted.)
* Also, worker should not change global pixel.
* So very limited worker can use this method.
* Use carefully.
*/
void
FlagtileSurface::filter_tiles_mp(KernelWorker *w)
{
    int tile_size = PYRAMID_TILE_SIZE(w->get_target_level());

    #pragma omp parallel for
    for(int ty=0; ty<m_height; ty++) {
        for(int tx=0; tx<m_width; tx++) {
            // Get tile.
            Flagtile *t = get_tile(tx, ty, false);

            int bx = tx * tile_size;
            int by = ty * tile_size;
            if (t != NULL && w->start(t, bx, by)) {
                // iterate tile pixel.
                for (int y=0; y<tile_size; y++) {
                    for (int x=0; x<tile_size; x++) {
                        w->step(t, x, y, bx+x, by+y);
                    }
                }
                // iterate tile pixel end.
                // As a default, the `end` method
                // of KernelWorker would set
                // DIRTY stat flag for the tile.
                w->end(t);
            }
        }
    }
}
#endif

/// propagate related.

/**
* @propagate_upward
* The function to build(finalize) propagating pixel seeds.
*
* As first, we would convert color tiles into each Flagtiles
* at pyramid-level 0.
* And register such tiles into FlagtileSurface,
* Then, call this method to `propagate` pixel upward and create
* pyramid level.
* `propagate` of this functionality is simular to so called
* `max pooling`.
*
* Before this method called, polygon rendering(with FLAG_AREA)
* should be completed.
*/
void
FlagtileSurface::propagate_upward(const int max_level)
{
#ifdef HEAVY_DEBUG
    assert(max_level <= MAX_PYRAMID);
#endif

    #pragma omp parallel for
    for(int ty=0; ty<m_height; ty++) {
        for(int tx=0; tx<m_width; tx++) {

            Flagtile *t = get_tile(tx, ty, false);
            if (t == NULL) {
                continue;
            }
            t->propagate_upward(max_level);
        }
    }
}

/**
* @propagate_downwards
* The interface method of doing downward-propagation of
* decided pixels.
*
* @detail
* With propagation of this method, we can reshape target filling area
* progressively into downward pyramid-level.
* Actual pixel manipulation(deciding the filling area) is done in
* PropagateKernel class.
*/
void
FlagtileSurface::propagate_downward(const int level, const bool expand_outside)
{
#ifdef HEAVY_DEBUG
    assert(level > 0);
    assert(level <= MAX_PYRAMID);
#endif
    PropagateKernel k(this, expand_outside);
    k.set_target_level(level);

#ifdef _OPENMP
    filter_tiles_mp((KernelWorker*)&k);  // This worker is parallelizable.
#else
    filter_tiles((KernelWorker*)&k);
#endif

}

/**
* @convert_pixel
* Utility method to convert pixels from python
*
*/
void
FlagtileSurface::convert_pixel(const int level,
                               const int targ_pixel,
                               const int new_pixel)
{
#ifdef HEAVY_DEBUG
    assert(targ_pixel != new_pixel);
#endif
    ConvertKernel ck(this);
    ck.set_target_level(level);
    ck.set_target_pixel(targ_pixel, new_pixel);
#ifdef _OPENMP
    if (targ_pixel != PIXEL_EMPTY) {
        filter_tiles_mp((KernelWorker*)&ck);  // This worker is parallelizable for non-empty pixel.
        return;
    }
#endif
    filter_tiles((KernelWorker*)&ck);
}

//// Callbacks used from identify_areas
//
// A callback for `g_queue_foreach` function,
// used in FlagtileSurface::identify_areas.
// This callback is to get the largest pixel area perimeter.
void __foreach_largest_cb(gpointer data, gpointer user_data)
{
    perimeter_info *info = (perimeter_info*)data;
    int *max_length = (int*)user_data;

    if (info->length > *max_length) {
        *max_length = info->length;
    }
}

/**
* @identify_areas
* Identify pixel areas by how many edge pixels are touching `outside` pixels.
*
* @param targ_pixel  The target pixel value. CountPerimeterWalker walks inside this pixel.
* @param accept_threshold  The threshold value of surrounding `closing` pixels ratio.
* @param reject_threshold  The threshold value of surrounding `opened` pixels ratio.
* @param accepted_pixel Filling pixel value of accepted area.
* @param rejected_pixel Filling(Erasing) pixel value of rejected area.
* @param size_threshold  Optional parameter,to reject area from its size, even if it is not `opened`.
*
* This method is to identify pixel areas whether it is useless(to be rejected)
* or usable(to be accepted).
*
* That decision is made by how `opened` or `closed` an area is.
* `opened` means `the area is neighbored too many un-closed pixels`.
* For example, a PIXEL_FILLED area which surrounded by too many PIXEL_OUTSIDE pixels,
* that area would be rejected.
* On the other hand, `closed` means `the area is neighbored enough closed pixels`.
* For example, a PIXEL_FILLED area which almost surrounded by PIXEL_CONTOUR pixels,
* that area would be accepted.
*
* accept_threshold and reject_threshold are from 0.0 to 1.0.
* If `accept_threshold` is less than 0.0, it means `accept all of not rejected area`.
* If `reject_threshold` is 1.0, it means `never reject`
* There can be a area which is both of `not rejected` and `not accepted`.
* Such area would be left unchanged.
* An area which has larger or equal perimeter than size_threshold would be always accepted,
* even if how it is `opened`.
*/
void
FlagtileSurface::identify_areas(const int level,
                                const int targ_pixel,
                                const double accept_threshold,
                                const double reject_threshold,
                                const int accepted_pixel,
                                const int rejected_pixel,
                                int size_threshold) // Optional, and not const. this might be rewritten.
{
#ifdef HEAVY_DEBUG
    assert(level <= MAX_PYRAMID);
    assert(level >= 0);
#endif
    GQueue *queue = g_queue_new();
    CountPerimeterWalker pk(this, queue);
    pk.set_target_level(level);
    pk.set_target_pixel(targ_pixel);
    filter_tiles(&pk);

    // Get maximum perimeter, to decide largest main area.
    if (size_threshold == 0) {
        int max_length = 0;
        g_queue_foreach(queue, __foreach_largest_cb, &max_length);
        size_threshold = max_length; // Rewrite size_threshold param. so not use const.
    }

    FloodFiller ra;
    ClearflagWalker cf(this);

    ra.set_target_level(level);
    cf.set_target_level(level);

    //printf("---- area start \n");
    while(g_queue_get_length(queue) > 0) {
        perimeter_info *info = (perimeter_info*)g_queue_pop_head(queue);

        if (info->clockwise==false) {
            // info->clockwise == false: i.e. it is not area, it's hole.
            // Just erase walking flags for now.
            cf.set_target_pixel(PIXEL_MASK & get_pixel(level, info->sx, info->sy));
            cf.walk(info->sx, info->sy, cf.get_hand_dir(info->direction));
        }
        else {
            bool erase_flag = false;
            //printf("area %p (%d,%d) accept %f, reject %f\n", info, info->sx, info->sy, info->accept_ratio, info->reject_ratio);

            if (info->length == 1 && level == 0) {
                // Just a dot. Erase it.
                replace_pixel(level, info->sx, info->sy, rejected_pixel);
            }
            else if (info->reject_ratio > reject_threshold
                        || info->length < size_threshold) {
                // This area should be rejected.
                //printf("area %p to be reject\n", info);
                if (targ_pixel == rejected_pixel) {
                    // But, already rejected for some reason.
                    erase_flag = true;
                    //printf("but only cleared.\n");
                }
                else {
                    ra.set_target_pixel(targ_pixel, rejected_pixel);
                    flood_fill(info->sx, info->sy, &ra);
                }
            }
            else if (info->accept_ratio > accept_threshold) {
                // Assign this area as `to be accepted` area.
                if (targ_pixel == accepted_pixel) {
                    erase_flag = true;
                }
                else {
                    ra.set_target_pixel(targ_pixel, accepted_pixel);
                    flood_fill(info->sx, info->sy, &ra);
                }
            }

            if (erase_flag) {
                // Same pixels assinged for target and reject/accept pixel, otherwise,
                // `Neither rejected nor accepted`.
                // So just walk to erase flag, leave it unchanged.
                cf.set_target_pixel(targ_pixel);
                cf.walk(info->sx, info->sy, info->direction);
            }
        }
        delete info;
    }
    g_queue_free(queue);
}

/**
* @fill_holes
* Fill all small holes if needed.
*
* This method is to reject small needless areas by counting its perimeter,
* and fill it if it is `hole` (i.e. counter-clockwised area).
*/
void
FlagtileSurface::fill_holes()
{
    GQueue *queue = g_queue_new();
    CountPerimeterWalker pk(this, queue);
    pk.set_target_level(0);
    pk.set_target_pixel(PIXEL_FILLED);
    filter_tiles(&pk);

    ClearflagWalker cf(this);
    cf.set_target_pixel(PIXEL_FILLED);
    HoleFiller fh;
    fh.set_target_level(0);

    while(g_queue_get_length(queue) > 0) {
        perimeter_info *info = (perimeter_info*)g_queue_pop_head(queue);

        if (!info->clockwise && info->reject_ratio == 0.0) {
            // Walking always starts from PIXEL_FILL pixel, 
            // and left of that pixel is other than it (i.e. fill target).
            int fx = info->sx-1;
            int fy = info->sy;
#ifdef HEAVY_DEBUG
            assert((get_pixel(fx, fy, false) & PIXEL_MASK) != PIXEL_FILLED);
#endif
            flood_fill(fx, fy, (Filler*)&fh);
        }
        else {
            cf.walk(info->sx, info->sy, cf.get_hand_dir(info->direction));
        }
        delete info;
    }
    g_queue_free(queue);
}

void
FlagtileSurface::dilate(const int pixel, const int dilation_size)
{
    if (dilation_size > 0) {
        DilateKernel dk(this, pixel);
        for(int i=0; i<dilation_size; i++) {
            filter_tiles((KernelWorker*)&dk);
        }
    }
}

/**
* @draw_antialias
* Draw antialias lines around the final result pixels.
*
* This method should be called the last of fill process.
*/
void
FlagtileSurface::draw_antialias()
{
    AntialiasWalker ak(this);
    filter_tiles((KernelWorker*)&ak);
}

//--------------------------------------
// FloodfillSurface class

FloodfillSurface::FloodfillSurface()
    : FlagtileSurface(),
      m_src_dict(NULL)
{
}

FloodfillSurface::~FloodfillSurface()
{
    if (m_tiles != NULL) {
        for(int i=0; i<m_width*m_height; i++) {
            Flagtile *ct = m_tiles[i];
            // Borrowed tile MUST NOT be deleted.
            if (ct != NULL
                    && (ct->get_stat() & Flagtile::BORROWED) != 0)
                m_tiles[i] = NULL; // So, just `hide` it.
        }
        // m_tiles array would be deleted at parent destructor.
    }

    if (m_src_dict != NULL)
        Py_DECREF(m_src_dict);

#ifdef HEAVY_DEBUG
    printf("Floodfill destructor called.\n");
#endif
}

void
FloodfillSurface::refer_tiledict(PyObject* tiledict)
{
#ifdef HEAVY_DEBUG
    assert(PyDict_Check(tiledict));
    assert(PYRAMID_TILE_SIZE(0) == MYPAINT_TILE_SIZE);
#endif
    // Build from tiledict, which is created in python code,
    // by calling pyramid_flood_fill function.
    // That tiledict would be a dictionary which has
    // key as tuple (x, y).

    PyObject *keys = PyDict_Keys(tiledict);// New reference
    int length = PyObject_Length(keys);
    int min_x = 0;
    int min_y = 0;
    int max_x = 0;
    int max_y = 0;

    // Get surface dimension and generate it.
    for (int i=0; i < length; i++ ) {
        PyObject *ck = PyList_GetItem(keys, i);
#ifdef HEAVY_DEBUG
    assert(PyTuple_Check(ck));
    assert(ck != NULL);
#endif
#if PY_MAJOR_VERSION >= 3
        int cx = (int)PyLong_AsLong(PyTuple_GetItem(ck, 0));
        int cy = (int)PyLong_AsLong(PyTuple_GetItem(ck, 1));
#else
        int cx = (int)PyInt_AsLong(PyTuple_GetItem(ck, 0));
        int cy = (int)PyInt_AsLong(PyTuple_GetItem(ck, 1));
#endif
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

    generate_tileptr_buf(
        min_x, min_y,
        (max_x - min_x) + 1,
        (max_y - min_y) + 1
    );

    // We can borrow tiles from dict here,
    // but it takes time and effort.
    // For now, I took easist way.

    // By the way, incref the source dictionary
    // to ensure flagtiles exist during this class is used.
    m_src_dict = tiledict;
    Py_INCREF(m_src_dict);
#ifdef HEAVY_DEBUG
    assert(m_src_dict->ob_refcnt >= 2);
#endif
}

void
FloodfillSurface::borrow_tile(const int tx, const int ty, Flagtile *tile)
{
    int idx = get_tile_index(tx, ty);
    m_tiles[idx] = tile;
    tile->set_borrowed();
}

//--------------------------------------
// Close and fill
/**
* @Constructor
*
* @param ox,oy Origin of tiles, in mypaint tiles. not pixels.
* @param w,h Surface width and height, in mypaint tiles. not pixels.
*/
ClosefillSurface::ClosefillSurface(const int min_x,
                                   const int min_y,
                                   const int max_x,
                                   const int max_y)
    : FlagtileSurface()
{
#ifdef HEAVY_DEBUG
    assert(max_x >= min_x);
    assert(max_y >= min_y);
#endif
    generate_tileptr_buf(
        min_x / TILE_SIZE,
        min_y / TILE_SIZE,
        ((max_x - min_x) / TILE_SIZE) + 1,
        ((max_y - min_y) / TILE_SIZE) + 1
    );
}

ClosefillSurface::~ClosefillSurface(){
#ifdef HEAVY_DEBUG
    printf("ClosefillSurface destructor called.\n");
#endif
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
ClosefillSurface::draw_line(const int sx, const int sy,
                            const int ex, const int ey,
                            const int pixel)
{
    // draw virtual line of bitwise flag,
    // by using Bresenham algorithm.
    int dx = abs(ex - sx);
    int dy = abs(ey - sy);
    int x=sx, y=sy, error, xs=1, ys=1;
    if (sx > ex) {
        xs = -1;
    }
    if (sy > ey) {
        ys = -1;
    }

    if (dx < dy) {
        // steep angled line
        error = dy >> 1;
        while (y != ey) {
            replace_pixel(0, x, y, pixel);
            error -= dx;
            if (error < 0) {
                x += xs;
                error += dy;
            }
            y += ys;
        }
    }
    else {
        error = dx >> 1;
        while (x != ex) {
            replace_pixel(0, x, y, pixel);
            error -= dy;
            if (error < 0) {
                y+=ys;
                error += dx;
            }
            x += xs;
        }
    }

    // Ensure the exact last pixel should be drawn
    if (x != ex || y != ey)
        replace_pixel(0, ex, ey, pixel);
}

/**
* @decide_area
* Decide fillable area
*
* @detail
* This method draws fundamental PIXEL_AREA pixels at pyramid-level 0.
*
* To use this method, you must previously draw closed polygon
* of PIXEL_AREA lines by using draw_line method.
*
* This method is a variant of identify_areas method.
*/
void
ClosefillSurface::decide_area()
{
    GQueue *queue = g_queue_new();
    CountPerimeterWalker pk(this, queue);
    pk.set_target_level(0);
    pk.set_target_pixel(PIXEL_EMPTY);
    filter_tiles(&pk);

    FloodFiller fw;
    fw.set_target_level(0);
    fw.set_target_pixel(PIXEL_EMPTY, PIXEL_AREA);

    ClearflagWalker cf(this);
    cf.set_target_level(0);
    cf.set_target_pixel(PIXEL_EMPTY);

    while(g_queue_get_length(queue) > 0) {
        perimeter_info *info = (perimeter_info*)g_queue_pop_head(queue);

        if (info->clockwise==true) {
            flood_fill(info->sx, info->sy, &fw);
        }
        else {
            cf.walk(info->sx, info->sy, info->direction);
        }
        delete info;
    }
    g_queue_free(queue);
}

/**
* @decide_outside
* Decide outside area
*
* @detail
* Fill outside pixels with PIXEL_OUTSIDE and
* fill inside-contour pixels with PIXEL_FILLED
* at current pyramid level.
* With this, we decide outside/inside of current
* closed area roughly, and, as a side effect,
* closing gap of contour by pyramid-level-pixel size.
*
* After this method called, we would call propagate_downwards method
* and gradually progress(reshape) filled pixels.
*/
void
ClosefillSurface::decide_outside(const int level)
{
    // XXX Actually, this is a bit different version of
    // FlagtileSurface::identify_areas
    GQueue *queue = g_queue_new();
    CountPerimeterWalker pk(this, queue);
    pk.set_target_level(level);
    pk.set_target_pixel(PIXEL_AREA);
    filter_tiles(&pk);

    FloodFiller ra;
    ClearflagWalker cf(this);
    ra.set_target_level(level);
    cf.set_target_level(level);

    while(g_queue_get_length(queue) > 0) {
        perimeter_info *info = (perimeter_info*)g_queue_pop_head(queue);

        if (info->clockwise==true && info->reject_ratio > 0.0) {
            // This should be opened outside area.
            ra.set_target_pixel(PIXEL_AREA, PIXEL_OUTSIDE);
            flood_fill(info->sx, info->sy, &ra);
        }
        else {
            // This is completely closed area, or just a hole.
            // So just clear walked flag.
            cf.set_target_pixel(PIXEL_AREA);
            cf.walk(info->sx, info->sy, info->direction);
        }
        delete info;
    }
    g_queue_free(queue);
}

//--------------------------------------
// Cutprotrude
CutprotrudeSurface::CutprotrudeSurface()
    :  FloodfillSurface()
{
}

CutprotrudeSurface::~CutprotrudeSurface()
{
#ifdef HEAVY_DEBUG
    printf("CutprotrudeSurface destructor called.\n");
#endif
}

/**
* @remove_overwrap_contour
* Remove isolated PIXEL_OVERWRAP areas.
*
* @detail
* This method is simular to FlagtileSurface::identify_areas,
* but just remove isolated (i.e. completely surrounded by invalid pixels)
* PIXEL_OVERWRAP pixel area.
*/
void
CutprotrudeSurface::remove_overwrap_contour()
{
    GQueue *queue = g_queue_new();
    CountOverwrapWalker pk(this, queue);
    pk.set_target_level(0);
    pk.set_target_pixel(PIXEL_OVERWRAP);
    filter_tiles(&pk);

    FloodFiller ra;
    ra.set_target_level(0);

    while(g_queue_get_length(queue) > 0) {
        perimeter_info *info = (perimeter_info*)g_queue_pop_head(queue);
        // NOTE: We cannot reject `hole` area at here because
        // That area might be multiple areas which has a composition
        // of diagonally connected. Such areas are needed to be detected
        // separately - i.e. we need new search for `hole` pixels.

        if (info->clockwise == true) {
            if (info->length == 1) {
                // Just a dot. Erase it.
                replace_pixel(0, info->sx, info->sy, PIXEL_AREA);
            }
            else if (info->reject_ratio == 1.0) {
                // Completely surrounded by `open` pixels.
                // Convert it into PIXEL_AREA, it would become `removing`
                // mark for mypaint colortile.
                ra.set_target_pixel(PIXEL_OVERWRAP, PIXEL_AREA);
                flood_fill(info->sx, info->sy, &ra);
            }
            // We would not need erasing FLAG_WORK, because that pixels
            // just ignored
        }
        delete info;
    }
    g_queue_free(queue);
}

//-----------------------------------------------------------------------------
// Functions

/**
* @flagtile_flood_fill
* Flagtile flood-fill core function
*
* @param level The target pyramid level.
* @param tile A flag tile object.
* @param q  BorderQueue object, contains queueing information.
* @param w  Filler object, do execute pixel match/filling operation.
*
* This function is to share codes with every flood-fill operation of flagtile.
*/
bool
flagtile_flood_fill(Flagtile *t, BorderQueue *q, Filler *w)
{
#ifdef HEAVY_DEBUG
    assert(t != NULL);
    assert(q != NULL);
    assert(w != NULL);
#endif
    const int level = w->get_target_level();
    const int tile_size = PYRAMID_TILE_SIZE(level);

    // Some filler-worker might match multiple type of pixel.
    // For such filler, checking fillable pixel count would not work.
    if (w->get_target_pixel() != PIXEL_INVALID) {
        if (t->get_pixel_count(level, w->get_target_pixel()) == 0) {
            q->clear_border_seed();
            return false;
        }
        else if (t->is_filled_with(level, w->get_target_pixel())) {
            t->fill(w->get_fill_pixel());
         
            // With this, we can seed every direction
            // except for incoming seed.
            q->invert_all_border(); 
            return true;
        }
    }

    GQueue *queue = q->get_working_state(level, t, w);
    if (queue == NULL) {
        q->clear_border_seed();
        return false;
    }

    static const int x_delta[] = {-1, 1};
    static const int x_offset[] = {0, 1};
    //int min_x = 0;
    //int min_y = 0;
    int max_x = tile_size;
    int max_y = tile_size;
    /*
    if (min_px >= 0)
        min_x = min_px;
    if (min_py >= 0)
        min_y = min_py;

    if (max_px >= 0)
        max_x = max_px;
    if (max_py >= 0)
        max_y = max_py;
#ifdef HEAVY_DEBUG
    assert(max_x >= min_x);
    assert(max_y >= min_y);
#endif
    */

    uint8_t pix;

    while (! g_queue_is_empty(queue)) {
        floodfill_point *pos = (floodfill_point*) g_queue_pop_head(queue);
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
                if (x < 0|| y < 0|| x >= max_x || y >= max_y) {
                    break;
                }

                // Test was already done for queued pixels ,
                // However, for initial (border) seed, that test is done against
                // `pre-filled` pixel. 
                // Such pixels might be changed by other(previously popped) queue.
                // So, always do match-check. 
                pix = t->get(level, x, y);
                if (!w->match(pix)) {
                    break;
                }
                
                // Operate this pixel, and continue iterating in this direction
                w->step(t, x, y);

                // In addition, enqueue the pixels above and below.
                // Scanline algorithm here to avoid some pointless queue faff.
                if (y > 0) {
                    pix = t->get(level, x, y-1);
                    if(w->match(pix)) {
                        if (look_above) {
                            // Enqueue the pixel to the north
                            floodfill_point *p = (floodfill_point *) malloc(
                                sizeof(floodfill_point)
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
                    q->queue_border(BorderQueue::SPILL_NORTH, x);
                }

                if (y < max_y - 1) {
                    pix = t->get(level, x, y+1);
                    if(w->match(pix)) {
                        if (look_below) {
                            // Enqueue the pixel to the South
                            floodfill_point *p = (floodfill_point *) malloc(
                                sizeof(floodfill_point)
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
                    q->queue_border(BorderQueue::SPILL_SOUTH, x);
                }

                if (x == 0) {
                    q->queue_border(BorderQueue::SPILL_WEST, y);
                }
                else if (x == tile_size-1) {
                    q->queue_border(BorderQueue::SPILL_EAST, y);
                }
            }
        }
    }

    // Clean up working state.
    g_queue_free(queue);
    return true;
}
