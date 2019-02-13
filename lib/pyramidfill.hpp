/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef PYRAMIDFILL_HPP
#define PYRAMIDFILL_HPP

#include <Python.h>
#include "pyramiddefine.hpp"

/* HOW IT WORKS : `pyramid-fill` 
 *
 * This Module is for implementing python interface of 
 * 'Pyramid-fill' 
 *
 * I don't know how this should be called, so I named this as
 * Pyramid-fill. You can see images with googling 'pyramid mipmap'.
 * (Also, I want to avoid name-conflict/confusion around already existing
 * `surface-mipmap` of Mypaint.)
 *
 * Pyramid-fill is simular to mipmap, but not getting average,
 * get maximum pixel value (i.e. PIXEL_CONTOUR).
 * This is almost same as `max-pooling`
 * 
 * And when pyramid creation completed, you will see
 * very large chunky pixels in the higher pyramid-level.
 * These chunky pixel acutually close gaps and avoid
 * spill out the flood-fill operation.
 *
 * Then we gradually progress pixels to downward of pyramid-level. 
 * It gradually propergates `decided`(PIXEL_FILLED/PIXEL_OUTSIDE) 
 * pixel value around neighbored `undecided`(PIXEL_AREA) pixels.
 * This operation also block holes of pixel contour.
 *
 * When propagation reached to level 0, identify undecided pixel area. 
 * If it is neighbored too many `outside/invalid` pixels, reject them. 
 * Otherwise, accept such pixel area as `decided` area.
 *
 * At last, we convert decided pixels as Mypaint colortiles,
 * and combine them to target layer.
 * 
 * Finally, You'll get gap-closed filled pixels around there.
 */


/* Flagtile class, to contain pixel flag information. 
 * 
 * This class holds pixels in like a pyramid shape.
 * Actually, it is the same concept as `mipmap`, 
 * but I named it `Pyramid` to avoid confusion with already existing `surface mipmap`.
 *
 * TODO: Currently, the flag buffer is allocated with C++ `new` function.
 *       This can be numpy buffer, to modefy the contents easily from python.
 */
class Flagtile 
{
protected:
    
    uint8_t *m_buf; // XXX Should this be numpy, not raw C++ memory...?
    
    // Pixel counts. This stores how many pixels per pixel value in this tile.
    uint16_t m_pixcnt[PIXEL_MASK+1];

    // buffer offsets of progress levels.
    static const int m_buf_offsets[MAX_PYRAMID+1];
    
    // Status bit flag for each tile.
    // It is Dirty flag, etc.
    // 32bit length would be (too) enough.
    int32_t m_statflag;

    // Get antialias value(0.0 - 1.0) from Antialias pixel of flagtile.
    inline double get_aa_double_value(const uint8_t pix)
    {
        // 0.9 is practical factor, not to become alpha value too opaque.
        return ((((double)(pix)) / MAX_AA_LEVEL) * 0.9);
    }

public:
    Flagtile(const int initial_value);

    virtual ~Flagtile(); 

    inline uint8_t get(const int level, const int x, const int y) 
    {
        return *BUF_PTR(level, x, y);
    }

    inline void replace(const int level, int x, int y, uint8_t val) 
    {
        uint8_t oldpix = *BUF_PTR(level, x, y) & PIXEL_MASK;
        if (level == 0 
                && (val & PIXEL_MASK) != oldpix) {
            m_pixcnt[oldpix]--;
            m_pixcnt[val & PIXEL_MASK]++;
        }

        if((val & FLAG_MASK) != 0)
            set_dirty();
        
        *BUF_PTR(level, x, y) = val;
    }

    void clear_bitwise_flag(const int level, const uint8_t flag) 
    {
        uint8_t *cp = BUF_PTR(level, 0, 0);
        for(int i=0; i < PYRAMID_BUF_SIZE(level); i++) {
            *cp &= (~flag);
            cp++;
        }
    }

    // Fill entire progress level pixels, with assigned value.
    void fill(const uint8_t val);

    void convert_from_color(PyObject *py_src_tile,
                            const int targ_r, 
                            const int targ_g, 
                            const int targ_b, 
                            const int targ_a, 
                            const double tolerance,
                            const double alpha_threshold,
                            const bool limit_within_opaque);

    void convert_to_color(PyObject *py_targ_tile,
                          const double r, 
                          const double g, 
                          const double b,
                          const int pixel);

    void convert_from_transparency(PyObject *py_targ_tile,
                                   const double alpha_threshold,
                                   const int pixel_value,
                                   const int overwrap_value);

    inline int get_stat() { return m_statflag; }

    inline bool is_filled_with(const int pix) 
    { 
        uint16_t cnt;
        if (pix == PIXEL_INVALID) 
            cnt = m_pixcnt[PIXEL_EMPTY] + m_pixcnt[PIXEL_OUTSIDE];
        else 
            cnt = m_pixcnt[pix & PIXEL_MASK];
        return cnt == (MYPAINT_TILE_SIZE * MYPAINT_TILE_SIZE);
    }
    
    inline int get_pixel_count(const int pix) {
        if (pix == PIXEL_INVALID) 
            return m_pixcnt[PIXEL_EMPTY] + m_pixcnt[PIXEL_OUTSIDE];
        else
            return m_pixcnt[pix & PIXEL_MASK];
    }

    inline void set_dirty() {m_statflag |= DIRTY;}
    inline void clear_dirty() {m_statflag &= (~DIRTY);}
    inline void set_borrowed() {m_statflag |= BORROWED;}

    // Propagate upward until targ_level
    void propagate_upward(const int targ_level);
    // Propagete upward single level
    void propagate_upward_single(const int targ_level);

    //// Tile Status flags.
    // Actually, it is int32_t. but, for SWIG, we need 
    // %include stdint.i 
    // at SWIG interface file.
    // So, currently use int.
    
    // DIRTY means this tile is written something.
    // This flag would be used in
    // FlagtileSurface::_filter method and 
    // some Kernelworkers finalize_worker method.
    static const int DIRTY = 0x00000001;
    
    // This tile is a borrowed one from python dictionary.
    // i.e. this tile should not be deleted. just replace with NULL.
    static const int BORROWED = 0x00000002;

    // To Expose PIXEL_ values for python
    // without `contaminating` original mypaint namespace.
    static const int PIXEL_FILLED_VALUE = PIXEL_FILLED;
    static const int PIXEL_AREA_VALUE = PIXEL_AREA;
    static const int PIXEL_EMPTY_VALUE = PIXEL_EMPTY;
    static const int PIXEL_CONTOUR_VALUE = PIXEL_CONTOUR;
    static const int PIXEL_OUTSIDE_VALUE = PIXEL_OUTSIDE;
    static const int PIXEL_OVERWRAP_VALUE = PIXEL_OVERWRAP;
    static const int PIXEL_INVALID_VALUE = PIXEL_INVALID;
    static const int PIXEL_RESERVE_VALUE = PIXEL_RESERVE;

    static const int MAX_PYRAMID_LEVEL = MAX_PYRAMID;
};

/* Flagtile psuedo surface object.
 * This is the base class of fill operation classes.
 * This class is exposed to python, but cannot be used.
 */
class FlagtileSurface 
{
protected:// Use protected, this is base-class.
    // The array of pointer of Flagtiles.
    Flagtile** m_tiles;

    // Shared buffer, for dummy empty tile. 
    uint8_t *m_empty_shared_buf;
    uint16_t *m_empty_shared_cnt;

    // Origin x,y (tile coordinate)
    int m_ox; 
    int m_oy;
    // Pixel based Origin 
    // (i.e. possible minimum x/y in pixel coordinate)
    // This origin is based on original(i.e. progress level 0)
    // piexl coordinate.
    int m_width;
    int m_height;
    
    void generate_tileptr_buf(const int ox, const int oy, 
                               const int w, const int h);

    void init_nodes(const int max_count);

    // Offsets used for pixel search kernel.
    inline int get_tile_index(const int tx, const int ty) 
    {
        return (ty * m_width + tx);
    }

    // Set tile forcefully. Use carefully.
    // Mainly used from filter_tile method.
    inline void set_tile(const int tx, const int ty, Flagtile* t) 
    {
        int idx = get_tile_index(tx, ty);
#ifdef HEAVY_DEBUG
        assert(idx < (m_width * m_height));
        assert(m_tiles[idx] == NULL);
#endif
        m_tiles[idx] = t;
    }

    // Replace pixel when that pixel exactly same with targ_flag.
    void convert_flag(const uint8_t targ_flag, 
                       const uint8_t flag,
                       const bool look_dirty=false);
    // Hide default constructor from python interface. 
    FlagtileSurface();
public:

    virtual ~FlagtileSurface();

    //// Getter methods

    
    // Get origin and dimension - in TILE unit.
    inline int get_origin_x() { return m_ox; }
    inline int get_origin_y() { return m_oy; }
    inline int get_width() { return m_width; }
    inline int get_height() { return m_height; }

    // progress tile have different dimension.
    // So, maximum size of surface or pixel coordinate
    // is different from original(progress level 0).
    inline int get_pixel_max_x(const int level) 
    {
        return m_width * PYRAMID_TILE_SIZE(level);
    }

    inline int get_pixel_max_y(const int level) 
    {
        return m_height * PYRAMID_TILE_SIZE(level);
    }

    inline Flagtile* get_tile(const int tx, const int ty, 
                              const bool request=false)
    {
        return get_tile(get_tile_index(tx, ty), request);
    }

    inline Flagtile* get_tile_from_pixel(const int level, 
                                         const int sx, const int sy, 
                                         const bool request)
    { 
        int tile_size = PYRAMID_TILE_SIZE(level);
        int raw_tx = sx / tile_size;
        int raw_ty = sy / tile_size;

        // sx and sy should start from (0,0) 
        // in every case.
        // Flagtilesurface reserves 1 tile around
        // of original tiles, and every pixel operation
        // MUST not exceed 64 pixel away(from original tiles).
        // Thus, when sx or sy is lower than zero, it is empty tile.
        if(raw_tx >= m_width || sx < 0 
                || raw_ty >= m_height || sy < 0) {
            return NULL;
        }
        
        // above raw_tx/ty is zero-based, 
        // adjusted by origin already. 
        // so do not use get_tile_index, 
        return get_tile(raw_ty * m_width + raw_tx, request);
    }
    
    inline Flagtile* get_tile(const int idx, const bool request=false)
    {
#ifdef HEAVY_DEBUG
        assert(idx < (m_width * m_height));
#endif
        Flagtile* ct = m_tiles[idx];
        if (ct == NULL) {
            if (request) {
                ct = new Flagtile(PIXEL_EMPTY);
                m_tiles[idx] = ct;
            }
        }
        return ct;
    }

    // Check existence of a tile, 
    // without generating/discarding a wrapper object.
    inline bool tile_exists(const int tx, const int ty) 
    {
        return get_tile(tx, ty, false) != NULL;
    }

    inline uint8_t get_pixel(const int level, 
                             const int sx, const int sy) 
    {
        Flagtile *ct = get_tile_from_pixel(level, sx, sy, false);
        
        // If sx or sy exceeding the border of surface,
        // get_tile_from_pixel returns NULL, instead of m_empty_tile.
        if(ct == NULL) {
            return PIXEL_EMPTY;
        }

        const int tile_size = PYRAMID_TILE_SIZE(level);
        return ct->get(level,
                       POSITIVE_MOD(sx, tile_size), 
                       POSITIVE_MOD(sy, tile_size));
    }

    inline void replace_pixel(const int level, 
                              const int sx, const int sy, 
                              const uint8_t val) 
    {
        Flagtile *ct = get_tile_from_pixel(level, sx, sy, true);
        
#ifdef HEAVY_DEBUG
assert(ct != NULL);
#endif
        const int tile_size = PYRAMID_TILE_SIZE(level);
        ct->replace(level,
                    POSITIVE_MOD(sx, tile_size), 
                    POSITIVE_MOD(sy, tile_size), 
                    val);
    }

    //// Propagate methods.

    // Propagate base pixels upward level.
    void propagate_upward(const int max_level);
    
    // Propagate `Decided` pixels toward level 0.
    void propagate_downward(const int level, const bool expand_outside);

    // flood_fill method. 
    // This should(or, could) not be called from Python codes.
    // but might be called from other C++ worker classes.
    // So this must be public method.
    // To use flood-fill with pyramid-gap-closing feature
    // from python, use pyramidfloodfill function of tiledsurface.py.
    // XXX We might use C++ friend keyword for this...
    void flood_fill(const int sx, const int sy, FillWorker *w);
    
    // Also, filter method would be called from some worker classes.
    // Not for python.
    void filter_tiles(KernelWorker *k);
#ifdef _OPENMP
    // Parallelized version of filter_tiles. very limited usage.
    void filter_tiles_mp(KernelWorker *w);
#endif
    //
    // Utility Methods
    void convert_pixel(const int level, const int targ_pixel, const int new_pixel); 

    // Identify pixels (and reject or accept) 
    // by how many pixels touches `outside` pixels.
    void identify_areas(const int level, 
                        const int targ_pixel, 
                        const double accept_threshold, 
                        const double reject_threshold, 
                        const int accepted_pixel, 
                        const int rejected_pixel,
                        int size_threshold=0); 

    void dilate(const int pixel, const int dilation_size);
    
    // Finalize related methods.
    void fill_holes();
    void draw_antialias();
    
    // XXX for Debug (might be used even not in HEAVY_DEBUG)
    PyObject*
    render_to_numpy(PyObject *npbuf,
                    int tflag,
                    int tr, int tg, int tb,
                    int level);
};


/* FloodfillSurface for flood-fill.
 */
class FloodfillSurface : public FlagtileSurface 
{
private:
    // m_src_dict holds pointer of python dictinary 
    // from python code, used for Py_INCREF / Py_DECREF
    // to it.
    //
    // Under certain situation, source dictonary
    // which contains Flagtile objects
    // might be freed before this object is freed.
    // And, FloodfillSurface borrow some flagtiles
    // from that dictionary.
    // So we need INCREF the dictionary at constructor, 
    // and DECREF it at destructor.
    PyObject* m_src_dict;

public:
    FloodfillSurface(PyObject *tiledict);
    virtual ~FloodfillSurface();

    void borrow_tile(const int tx, const int ty, Flagtile *tile);
};

/* ClosefillSurface for close and fill.
 */
class ClosefillSurface : public FlagtileSurface 
{
public:
    ClosefillSurface(const int min_x, const int min_y,
                     const int max_x, const int max_y);
    virtual ~ClosefillSurface();
    
    void draw_line(const int sx, const int sy, 
                   const int ex, const int ey,
                   const int pixel);

    // Decide outside and inside pixels.
    void decide_outside(const int level);

    // Decide fillable area.
    void decide_area();
};

/* CutprotrudeSurface for `Cut protruding pixels` feature.
 * This class inherits from FloodfillSurface, because this borrows
 * python dictionary of Flagtiles previously generated/converted
 * like FloodfillSurface.
 */
class CutprotrudeSurface : public FloodfillSurface
{
public:
    CutprotrudeSurface(PyObject *tiledict); 
    virtual ~CutprotrudeSurface();

    void remove_overwrap_contour();
};

//// functions

/* floodfill of Pyramid_fill version. 
 * This would be used from lib/tiledsurface.py
 */
PyObject *
pyramid_flood_fill(Flagtile *tile, /* output HxWx4 array of uint16 */
                   PyObject *seeds, /* List of 2-tuples */
                   int min_x, int min_y, int max_x, int max_y,
                   int level,
                   int targ_pixel,
                   int fill_pixel);

#endif
