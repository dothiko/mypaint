/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef PROGFILL_HPP
#define PROGFILL_HPP

#include <Python.h>
#include "progfilldefine.hpp"
#include "fix15.hpp"
#include <glib.h>

// This Module is for implementing 'progressive fill' 

/* Flagtile class, to contain flag information. 
 * 
 * This class holds pixels in like a pyramid shape.
 * Actually, though it is the same concept as `mipmap`, 
 * I named it `Progress` to avoid confusion with existing surface mipmaps.
 */
class Flagtile 
{
protected:
    uint8_t *m_buf;

    // The total buffer size,
    static const int BUF_SIZE = PROGRESS_BUF_SIZE(0) + 
                                PROGRESS_BUF_SIZE(1) + 
                                PROGRESS_BUF_SIZE(2) + 
                                PROGRESS_BUF_SIZE(3) + 
                                PROGRESS_BUF_SIZE(4) + 
                                PROGRESS_BUF_SIZE(5) +
                                PROGRESS_BUF_SIZE(6); 

    // buffer offsets of progress levels.
    static const int m_buf_offsets[MAX_PROGRESS+1];
    
    // Status bit flag for each tile.
    // It is Dirty flag, etc.
    // 32bit length would be (too) enough.
    int32_t m_statflag;

    // Build single progress level
    void _build_progress_level(const int targ_level);

    // Get antialias value(0.0 - 1.0) from Antialias pixel of flagtile.
    inline double _get_aa_double_value(const uint8_t pix)
    {
        // 0.9 is practical factor, not to become alpha value too opaque.
        return ((((double)(pix)) / MAX_AA_LEVEL) * 0.9);
    }

public:
    Flagtile(const int initial_value);

    virtual ~Flagtile(); 

    // For completely seqencial access.
    inline uint8_t *get_ptr() { return m_buf;}

    inline void combine(const int level, int x, int y, uint8_t val) 
    {
        *_BUF_PTR(level, x, y) |= val;
    }

    inline void remove(const int level, int x, int y, uint8_t val) 
    {
        *_BUF_PTR(level, x, y) &= (~val);
    }

    inline uint8_t get(const int level, const int x, const int y) 
    {
        return *_BUF_PTR(level, x, y);
    }

    // Different from `combine`, `put` method changes only pixel value. 
    inline void put(const int level, int x, int y, uint8_t val) 
    {
        uint8_t *ptr = _BUF_PTR(level, x, y);
        *ptr &= FLAG_MASK;
        *ptr |= (PIXEL_MASK & val);
    }

    inline void replace(const int level, int x, int y, uint8_t val) 
    {
        *_BUF_PTR(level, x, y) = val;
    }

    void clear_bitwise_flag(const uint8_t flag) 
    {
        uint8_t *cp = m_buf;
        for(int i=0; i < BUF_SIZE; i++) {
            *cp &= (~flag);
            cp++;
        }
    }

    void clear_bitwise_flag(const int level, const uint8_t flag) 
    {
        uint8_t *cp = _BUF_PTR(level, 0, 0);
        for(int i=0; i < PROGRESS_BUF_SIZE(level); i++) {
            *cp &= (~flag);
            cp++;
        }
    }

    void convert_flag(const uint8_t targ_flag, const uint8_t new_flag) 
    {
        uint8_t *cp = m_buf;
        for(int i=0; i < BUF_SIZE; i++) {
            if (*cp == targ_flag)
                *cp = new_flag;
            cp++;
        }
    }

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
                          const double b);

    void convert_to_transparent(PyObject *py_targ_tile);

    inline int get_stat(){ return (int)m_statflag; }
    // set/unset stat flag with automatically removing conflicting flag/
    // adding conpanion flag.
    void set_stat(const int new_stat);
    void unset_stat(const int new_stat);

    static inline int get_length(){return TILE_SIZE * TILE_SIZE;}

    // Progress methods
    void build_progress_seed(const int start_level);

    //// Tile Status flags.
    
    // DIRTY means this tile is written something.
    // This flag would be used in
    // FlagtileSurface::_filter method and 
    // some Kernelworkers finalize_worker method.
    static const int DIRTY = 0x00000001;
    
    // This tile is a borrowed one from python dictionary.
    // i.e. this tile should not be deleted. just replace with NULL.
    static const int BORROWED = 0x00000002;
    
    /// Status flags of below are exclusive.
    /// We can set only each one of them for status flag.
    
    // This tile has completely filled with PIXEL_FILLED,
    // without any contour.
    static const int FILLED = 0x00000100;
    
    // This tile has some contour pixel.
    // This flag must not coexist with FILLED flag.
    static const int HAS_CONTOUR = 0x00000200;

    // This tile is not filled entirely but has some valid pixel.
    static const int HAS_PIXEL = 0x00000400;

    // This tile has completely filled with PIXEL_AREA.
    static const int FILLED_AREA = 0x00000800;
    
    // This tile is empty(i.e. filled with 0)
    static const int EMPTY = 0x00001000;
};

/* Flagtile psuedo surface object.
 * This is the base class of fill operation classes.
 * This class is exposed to python, but cannot be used.
 */
class FlagtileSurface 
{
protected:
    // The array of pointer of Flagtiles.
    Flagtile** m_tiles;

    // Origin x,y (tile coordinate)
    int m_ox; 
    int m_oy;
    // Pixel based Origin 
    // (i.e. possible minimum x/y in pixel coordinate)
    // This origin is based on original(i.e. progress level 0)
    // piexl coordinate.
    int m_width;
    int m_height;
    
    // Start progress level
    const int m_level;

    // Initial tile values
    // This is different between each Surface classes.
    const uint8_t m_initial_tile_val;

    void _generate_tileptr_buf(const int ox, const int oy, 
                               const int w, const int h);

    void _init_nodes(const int max_count);

    // Offsets used for pixel search kernel.
    inline int _get_tile_index(const int tx, const int ty) 
    {
        return (ty * m_width + tx);
    }

    // Replace pixel when that pixel exactly same with targ_flag.
    void _convert_flag(const uint8_t targ_flag, 
                       const uint8_t flag,
                       const bool look_dirty=false);

    // Hided default constructor.
    // for only initialize derive class. 
    FlagtileSurface(const int start_level, 
                    const uint8_t initial_tile_val);

public:

    virtual ~FlagtileSurface();

    //// Getter methods
    inline int get_origin_x() { return m_ox; }
    inline int get_origin_y() { return m_oy; }
    inline int get_width() { return m_width; }
    inline int get_height() { return m_height; }

    // progress tile have different dimension.
    // So, maximum size of surface or pixel coordinate
    // is different from original(progress level 0).
    inline int get_pixel_max_x(const int level) 
    {
        return m_width * PROGRESS_TILE_SIZE(level);
    }

    inline int get_pixel_max_y(const int level) 
    {
        return m_height * PROGRESS_TILE_SIZE(level);
    }

    inline int get_target_level(){ return m_level;}

    inline Flagtile* get_tile(const int tx, const int ty, 
                              const bool request=false) 
    {
        return get_tile(_get_tile_index(tx, ty), request);
    }

    inline Flagtile* get_tile_from_pixel(const int sx, const int sy, 
                                         const bool request, 
                                         const int level)
    {
        int tile_size = PROGRESS_TILE_SIZE(level);
        int raw_tx = sx / tile_size;
        int raw_ty = sy / tile_size;

        if(raw_tx >= m_width || raw_tx < 0 
                || raw_ty >= m_height || raw_ty < 0)
            return NULL;
        
        // above raw_tx/ty is zero-based, 
        // adjusted by origin already. 
        // so do not use _get_tile_index, 
        return get_tile(raw_ty * m_width + raw_tx, request);
    }
    
    inline Flagtile* get_tile(const int idx, const bool request=false) 
    {
#ifdef HEAVY_DEBUG
// XXX DEBUG
if (idx >= (m_width * m_height)) {
    printf("Exceeding tile limit!! idx %d\n" , idx);
    return NULL;
}
assert(idx < (m_width * m_height));
#endif
        Flagtile* ct = m_tiles[idx];
        if (request && ct == NULL) {
            ct = new Flagtile(m_initial_tile_val);
            m_tiles[idx] = ct;
        }
        return ct;
    }
    
    // Check existence of a tile, 
    // without generating/discarding a wrapper object.
    inline bool tile_exists(const int tx, const int ty) 
    {
        return get_tile(_get_tile_index(tx, ty), false) != NULL;
    }

    inline void combine_pixel(const int level, 
                              const int sx, const int sy, 
                              const uint8_t val) 
    {
        Flagtile *ct = get_tile_from_pixel(sx, sy, true, level);
        
#ifdef HEAVY_DEBUG
assert(ct != NULL);
#endif
        const int tile_size = PROGRESS_TILE_SIZE(level);
        ct->combine(level, positive_mod(sx, tile_size), positive_mod(sy, tile_size), 
                    val);
    }

    inline void put_pixel(const int level, 
                          const int sx, const int sy, 
                          const uint8_t val) 
    {
        Flagtile *ct = get_tile_from_pixel(sx, sy, true, level);
        
#ifdef HEAVY_DEBUG
assert(ct != NULL);
#endif
        const int tile_size = PROGRESS_TILE_SIZE(level);
        ct->put(level,
                positive_mod(sx, tile_size), 
                positive_mod(sy, tile_size), 
                val);
    }

    inline uint8_t get_pixel(const int level, 
                             const int sx, const int sy) 
    {
        Flagtile *ct = get_tile_from_pixel(sx, sy, false, level);
        if (ct == NULL)
            return 0; 

        const int tile_size = PROGRESS_TILE_SIZE(level);
        return ct->get(level,
                       positive_mod(sx, tile_size), 
                       positive_mod(sy, tile_size));
        
    }

    inline void replace_pixel(const int level, 
                              const int sx, const int sy, 
                              const uint8_t val) 
    {
        Flagtile *ct = get_tile_from_pixel(sx, sy, true, level);
        
#ifdef HEAVY_DEBUG
assert(ct != NULL);
#endif
        const int tile_size = PROGRESS_TILE_SIZE(level);
        ct->replace(level,
                    positive_mod(sx, tile_size), 
                    positive_mod(sy, tile_size), 
                    val);
    }


    // Progress methods.
    void build_progress_seed();
    // Actually progress pixels based on progress seeds.
    void progress_tiles(const int reject_targ_level,
                        const int perimeter);
    void dbg_progress_single_level(const int level,
                                   const int perimeter); // XXX for DEBUG

    // flood_fill method. 
    // This should not be called from Python codes.
    // but might be called from other C++ worker classes.
    // So this must be public method.
    // From python, use progfloodfill function of tiledsurface.py.
    // XXX We might use C++ friend keyword for this...
    void flood_fill(int sx, int sy, 
                    FillWorker *w);
    
    // Also, filter method would be called from some worker classes.
    // Make it public.
    void filter_tiles(KernelWorker *k);
    
    // Finalize all processing. 
    /*
    void finalize(const int threshold,
                  const int dilation_size,
                  const bool antialias,
                  const bool fill_all_holes);
    */     
    // Finalize related methods.
    void remove_small_areas(const int threshold, const bool fill_all_holes);
    void dilate(const int dilation_size);
    void draw_antialias();
    
    // XXX for Debug (might be used even not in HEAVY_DEBUG)
    PyObject*
    render_to_numpy(PyObject *npbuf,
                    int tflag,
                    int tr, int tg, int tb);
};

/* FloodfillSurface for flood-fill.
 */
class FloodfillSurface : public FlagtileSurface 
{
protected:
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

    FloodfillSurface(PyObject* tiledict, const int start_level);
    virtual ~FloodfillSurface();

    void borrow_tile(const int tx, const int ty, Flagtile* tile);
};

/* ClosefillSurface for close and fill.
 */
class ClosefillSurface : public FlagtileSurface 
{
protected:

    //// Line drawing & polygon
    _progfill_point *m_nodes;
    int m_cur_node;
    int m_node_cnt;
    
    void _walk_line(int sx, int sy, 
                    int ex, int ey,
                    DrawWorker *f);
    
    // Walk among the eitire nodes.
    void _walk_polygon(DrawWorker *w);

    bool _is_inside_polygon(const int x1, const int x2, const int y);

    inline bool _is_inside_polygon_point(const int x, const int y,
                                         const int sx, const int sy,      
                                         const int ex, const int ey)
    {
      return (((sy > y) != (ey > y)) 
                && (x < (ex - sx) * (y - sy) / (ey - sy) + sx));
    } 
    
    //// Edge line(Polygon) drawing
    // _move_to ,_line_to and _close_line are used in 
    // _init_node method.
    // Not only drawing outerrim flags, also register
    // nodes to internal buffer.
    // That polygon edge buffer uses later with
    // _walk_polygon method.
    void _move_to(const int sx, const int sy);
    void _line_to(DrawWorker* w, 
                  const int ex, const int ey, 
                  const bool closing=false);
    void _close_line(DrawWorker* w);

    void _init_nodes(PyObject* node_list);
    
    // Scanline fill the polygon.
    void _scanline_fill();

public:

    ClosefillSurface(const int start_level, PyObject* node_list);
    virtual ~ClosefillSurface();
    
    // Decide outside and inside pixels.
    void decide_area();
};

/* LassofillSurface for Lasso fill.
 *
 * In progfill.cpp, `Lasso fill` is done as
 * `Mask filled polygon with most-appeared
 * color area, and replace it with foreground color`.
 */
class LassofillSurface : public ClosefillSurface 
{
protected:
public:
    /**
    * @Constructor
    *
    * @param ox, oy: Origin of tiles, in mypaint tiles. not pixels. 
    * @param w, h: Surface width and height, in mypaint tiles. not pixels.
    */
    LassofillSurface(PyObject* node_list);
    virtual ~LassofillSurface();
    /*
    void finalize(const int dilation_size, 
                  const bool antialias,
                  const bool fill_all_holes);
    */
    void convert_result_area();
};

//// functions

/* floodfill of Progressive_fill version. 
 * This would be used from lib/tiledsurface.py
 */
PyObject *
progfill_flood_fill(Flagtile *tile, /* output HxWx4 array of uint16 */
                    PyObject *seeds, /* List of 2-tuples */
                    int min_x, int min_y, int max_x, int max_y,
                    int level);

#ifdef HEAVY_DEBUG
PyObject*
progfill_render_to_numpy(FlagtileSurface *surf,
                         PyObject *npbuf,
                         int tflag, // Ignored when level >= 0
                         int level, // Ignored when level < 0
                         int tr, int tg, int tb);
#endif

#endif
