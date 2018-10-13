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

// This Module is for implementing python interface of 
// 'progressive fill' 

/* Flagtile class, to contain flag information. 
 * 
 * This class holds pixels in like a pyramid shape.
 * Actually, though it is the same concept as `mipmap`, 
 * I named it `Progress` to avoid confusion with existing surface mipmaps.
 */
class Flagtile 
{
private:
    uint8_t *m_buf;

    // Pixel counts.
    int m_filledcnt;
    int m_areacnt;
    int m_contourcnt;

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
    void build_progress_level(const int targ_level);

    // Get antialias value(0.0 - 1.0) from Antialias pixel of flagtile.
    inline double get_aa_double_value(const uint8_t pix)
    {
        // 0.9 is practical factor, not to become alpha value too opaque.
        return ((((double)(pix)) / MAX_AA_LEVEL) * 0.9);
    }

public:
    Flagtile(const int initial_value);

    virtual ~Flagtile(); 

    // For completely seqencial access.
    inline uint8_t *get_ptr() { return m_buf;}

    inline uint8_t get(const int level, const int x, const int y) 
    {
        return *BUF_PTR(level, x, y);
    }

    inline void replace(const int level, int x, int y, uint8_t val) 
    {
        uint8_t oldpix = *BUF_PTR(level, x, y) & PIXEL_MASK;
        if (level == 0 
                && (val & PIXEL_MASK) != oldpix) {
            switch (val & PIXEL_MASK) {
                case PIXEL_FILLED:
                    m_filledcnt++;
                    break;
                case PIXEL_AREA:
                    m_areacnt++;
                    break;
                case PIXEL_CONTOUR:
                    m_contourcnt++;
                    break;
            }

            switch (oldpix) {
                case PIXEL_FILLED:
                    m_filledcnt--;
                    break;
                case PIXEL_AREA:
                    m_areacnt--;
                    break;
                case PIXEL_CONTOUR:
                    m_contourcnt--;
                    break;
            }

        }

        if((val & FLAG_MASK) != 0)
            set_dirty();
        
        *BUF_PTR(level, x, y) = val;
    }

    void clear_bitwise_flag(const int level, const uint8_t flag) 
    {
        uint8_t *cp = BUF_PTR(level, 0, 0);
        for(int i=0; i < PROGRESS_BUF_SIZE(level); i++) {
            *cp &= (~flag);
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

    inline int get_stat()
    {
        if (m_filledcnt == TILE_SIZE*TILE_SIZE)
            return (m_statflag | FILLED | HAS_PIXEL);
        else if (m_areacnt == TILE_SIZE*TILE_SIZE)
            return (m_statflag | FILLED_AREA | HAS_AREA);
        else if (m_contourcnt == TILE_SIZE*TILE_SIZE)
            return (m_statflag | FILLED_CONTOUR | HAS_CONTOUR);
        else if (m_filledcnt == 0 && m_areacnt == 0 && m_contourcnt == 0)
            return (m_statflag | EMPTY);
        
        int32_t retflag = m_statflag;

        if (m_filledcnt > 0)
            retflag |= HAS_PIXEL;
        
        if (m_contourcnt > 0)
            retflag |= HAS_CONTOUR;

        if (m_areacnt > 0)
            retflag |= HAS_AREA;

        return retflag;
    }

    inline bool is_filled_with(const uint8_t pix) 
    { 
        switch(pix) {
            case PIXEL_AREA:
                return (m_statflag & FILLED_AREA);
            case PIXEL_FILLED:
                return (m_statflag & FILLED);
            case PIXEL_CONTOUR:
                return (m_statflag & FILLED_CONTOUR);
            case PIXEL_EMPTY:
            case PIXEL_OUTSIDE:
                return (m_statflag & EMPTY);
            default:
                return false;
        }
    }
    
    inline int get_pixel_count(const uint8_t pix) {
        switch(pix) {
            case PIXEL_AREA:
                return m_areacnt;
            case PIXEL_FILLED:
                return m_filledcnt;
            case PIXEL_CONTOUR:
                return m_contourcnt;
            case PIXEL_EMPTY:
            case PIXEL_OUTSIDE:
                return (TILE_SIZE*TILE_SIZE) 
                            - (m_areacnt+m_filledcnt+m_contourcnt);
            default:
                return 0;
        }
    }

    inline void set_dirty() {m_statflag |= DIRTY;}
    inline void clear_dirty() {m_statflag &= (~DIRTY);}
    inline void set_borrowed() {m_statflag |= BORROWED;}

    // Progress methods
    void build_progress_seed(const int start_level);

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
    
    /// Status flags of below are exclusive.
    /// We can set only each one of them for status flag.
    
    // This tile is not filled entirely but has some valid pixel.
    static const int HAS_PIXEL = 0x00000100;
    //
    // This tile has completely filled with PIXEL_FILLED,
    // without any contour.
    // When this flag is set, statflag should also have HAS_PIXEL.
    static const int FILLED = 0x00000200;
    
    // This tile has some contour pixel.
    static const int HAS_CONTOUR = 0x00000400;
    // Rarely but possible, a tile has only contour pixel.
    // When this flag is set, statflag should also have HAS_CONTOUR.
    static const int FILLED_CONTOUR = 0x00000800;

    // This tile has some vacant pixel.
    static const int HAS_AREA = 0x00001000;

    // This tile has completely filled with PIXEL_AREA.
    // When this flag is set, statflag should also have HAS_AREA.
    static const int FILLED_AREA = 0x00002000;
    
    // This tile is empty(i.e. filled with 0)
    static const int EMPTY = 0x00004000;
    
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

    // Origin x,y (tile coordinate)
    int m_ox; 
    int m_oy;
    // Pixel based Origin 
    // (i.e. possible minimum x/y in pixel coordinate)
    // This origin is based on original(i.e. progress level 0)
    // piexl coordinate.
    int m_width;
    int m_height;
    
    // Starting progress level
    const int m_level;

    // Initial tile values
    // This is different between each Surface classes.
    const uint8_t m_initial_tile_val;

    void generate_tileptr_buf(const int ox, const int oy, 
                               const int w, const int h);

    void init_nodes(const int max_count);

    // Offsets used for pixel search kernel.
    inline int get_tile_index(const int tx, const int ty) 
    {
        return (ty * m_width + tx);
    }

    // Replace pixel when that pixel exactly same with targ_flag.
    void convert_flag(const uint8_t targ_flag, 
                       const uint8_t flag,
                       const bool look_dirty=false);

    // Hided default constructor.
    // This is only for derive class. 
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
        return get_tile(get_tile_index(tx, ty), request);
    }

    inline Flagtile* get_tile_from_pixel(const int level, 
                                         const int sx, const int sy, 
                                         const bool request) 
    { 
        int tile_size = PROGRESS_TILE_SIZE(level);
        int raw_tx = sx / tile_size;
        int raw_ty = sy / tile_size;

        // sx and sy should start from (0,0) 
        // in every case.
        // Flagtilesurface reserves 1 tile around
        // of original tiles, and every pixel operation
        // MUST not exceed 64 pixel away(from original tiles).
        // Thus, when sx or sy is lower than zero, it is empty tile.
        if(raw_tx >= m_width || sx < 0 
                || raw_ty >= m_height || sy < 0)
            return NULL;
        
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
        return get_tile(get_tile_index(tx, ty), false) != NULL;
    }

    inline uint8_t get_pixel(const int level, 
                             const int sx, const int sy) 
    {
        Flagtile *ct = get_tile_from_pixel(level, sx, sy, false);
        if (ct == NULL)
            return 0; 

        const int tile_size = PROGRESS_TILE_SIZE(level);
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
        const int tile_size = PROGRESS_TILE_SIZE(level);
        ct->replace(level,
                    POSITIVE_MOD(sx, tile_size), 
                    POSITIVE_MOD(sy, tile_size), 
                    val);
    }


    // Progress methods.
    void build_progress_seed();
    // Actually progress pixels based on progress seeds.
    void progress_tiles();

    // flood_fill method. 
    // This should not be called from Python codes.
    // but might be called from other C++ worker classes.
    // So this must be public method.
    // From python, use progfloodfill function of tiledsurface.py.
    // XXX We might use C++ friend keyword for this...
    void flood_fill(const int sx, const int sy, FillWorker *w);
    
    // Also, filter method would be called from some worker classes.
    // Make it public.
    void filter_tiles(KernelWorker *k);
    
    // Finalize related methods.
    void remove_small_areas();
    void dilate(const int dilation_size);
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

    FloodfillSurface(PyObject* tiledict, const int start_level);
    virtual ~FloodfillSurface();

    void borrow_tile(const int tx, const int ty, Flagtile* tile);
};

/* ClosefillSurface for close and fill.
 */
class ClosefillSurface : public FlagtileSurface 
{
private:

    //// Line drawing & polygon
    progfill_point *m_nodes;
    int m_cur_node;
    int m_node_cnt;
    
    void walk_line(int sx, int sy, 
                    int ex, int ey,
                    DrawWorker *f);
    
    // Walk among the eitire nodes.
    void walk_polygon(DrawWorker *w);

    bool is_inside_polygon(const int x1, const int x2, const int y);

    inline bool is_inside_polygon_point(const int x, const int y,
                                         const int sx, const int sy,      
                                         const int ex, const int ey) 
    {
      return (((sy > y) != (ey > y)) 
                && (x < (ex - sx) * (y - sy) / (ey - sy) + sx));
    } 
    
    //// Edge line(Polygon) drawing
    // move_to ,_line_to and close_line are used in 
    // init_node method.
    // Not only drawing outerrim flags, also register
    // nodes to internal buffer.
    // That polygon edge buffer uses later with
    // walk_polygon method.
    void move_to(const int sx, const int sy);
    void line_to(DrawWorker* w, 
                  const int ex, const int ey, 
                  const bool closing=false);
    void close_line(DrawWorker* w);

    void init_nodes(PyObject* node_list);
    
    // Scanline fill the polygon.
    void scanline_fill();

public:

    ClosefillSurface(const int start_level, PyObject* node_list);
    virtual ~ClosefillSurface();
    
    // Decide outside and inside pixels.
    void decide_area();

    // Dedicated version of progress_tiles
    void progress_tiles();
};

/* LassofillSurface for Lasso fill.
 *
 * In progfill.cpp, `Lasso fill` is done as
 * `Mask filled polygon with most-appeared
 * color area, and replace it with foreground color`.
 */
class LassofillSurface : public ClosefillSurface 
{
public:
    /**
    * @Constructor
    *
    * @param ox, oy: Origin of tiles, in mypaint tiles. not pixels. 
    * @param w, h: Surface width and height, in mypaint tiles. not pixels.
    */
    LassofillSurface(PyObject* node_list);
    virtual ~LassofillSurface();
 
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

#endif
