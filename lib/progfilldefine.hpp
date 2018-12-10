/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */


/* This file is to hide base classes from python interface.
 * Also, all classes in this file are base class, so all
 * uses protected keyword as inner method/members.
 */

#ifndef PROGFILLDEFINE_HPP
#define PROGFILLDEFINE_HPP

/* This File is for utility class definition for
 * close-and-fill functionality.
 *
 */

#include <Python.h>
#include <mypaint-tiled-surface.h>

// Force positive modulo against every number.
// C/C++ would produce platform-dependent result
// with modulo against negative number.
// This macro is used in progfill.cpp/hpp
#define POSITIVE_MOD(a, b) (((a) % (b) + (b)) % (b))

#define MAX_PROGRESS 6

// Convert a progress level 0(source or final result) coordinate
// into each progress level (greater than 0) coordinate.
#define PROGRESS_PIXEL_COORD(a, l) ((a) >> (l)) 

// XXX CAUTION: PROGRESS_TILE_SIZE(0) MUST BE SAME AS MYPAINT_TILE_SIZE.
#define PROGRESS_TILE_SIZE(l) (1 << (MAX_PROGRESS - (l)))
#define PROGRESS_BUF_SIZE(l) (PROGRESS_TILE_SIZE(l) * PROGRESS_TILE_SIZE(l))

#define TILE_SIZE MYPAINT_TILE_SIZE

// Maximum Anti-Aliasing transparency level.
#define MAX_AA_LEVEL 127

// For Flagtile class. to get progress-level ptr from
#define BUF_PTR(l, x, y) (m_buf + m_buf_offsets[(l)] + ((y) * PROGRESS_TILE_SIZE((l)) + (x))) 

// PIXEL_ Constants.
enum PixelFlags {
    // PIXEL_ values are not bitwise flag. They are just number.
    // The vacant pixel is 0.
    // PIXEL_AREA means 'The pixel is fillable, but not filled(yet)'
    //
    PIXEL_MASK = 0x0F,
    PIXEL_EMPTY = 0x00,   // PIXEL_EMPTY should lower than PIXEL_OUTSIDE
    PIXEL_OUTSIDE = 0x01, // Thus, we can know total outside pixel as <= PIXEL_OUTSIDE
    PIXEL_AREA = 0x02,
    PIXEL_FILLED = 0x03,
    PIXEL_CONTOUR = 0x04, // PIXEL_CONTOUR is one of a filled pixel.
                          // This should be larger than PIXEL_FILLED
                          // to ease finding `filled(or unchangeable)` pixel.

    // PIXEL_EMPTY, PIXEL_AREA, PIXEL_FILLED and PIXEL_CONTOUR are redefined as
    // static const of Flagtile class, and you can access them from python
    // as lib.mypaintlib.Flagtile.PIXEL_*.
    // Other enum constants are hidden from python.
                         
    // FLAG_ values are bitwise flag. 
    FLAG_MASK = 0xF0,
    
    // FLAG_WORK flag is temporary flag, used for filter operation. 
    // This flag should be most significant bit, for final antialiasing.
    FLAG_WORK = 0x10, 
    
    // FLAG_AA used for final anti-aliasing.
    // CAUTION: With this flag set, the pixel contains 128 Level of antialias
    // alpha seed value. 
    // Therefore, all of PIXEL_ constants are invalid for such pixel.
    // And, FLAG_AA should be most significant bit (== 0x80 for uint8_t).
    FLAG_AA = 0x80,
    AA_MASK = 0x7F 
};

// Dummy definition of Flagtile / FlagtileSurface.
class Flagtile;
class FlagtileSurface;

/**
* @class PixelWorker
* @brief Base class of worker for drawing / searching pixel.
*
* Abstruct Worker class of walking line.
* This is to share same logic between drawing line
* and searching unfilled target area.
*/
class PixelWorker 
{
protected:
    FlagtileSurface* m_surf;
    int m_level;

public:
    PixelWorker(FlagtileSurface* surf) 
        : m_surf(surf), m_level(0) 
    {
        // Calling virtual method `set_target_level` 
        // at here (i.e. constructor) is meaningless.
        // Derived virtual function cannot be called from
        // constructor by C++ design.
        // So I use initialization list.
    }
    virtual ~PixelWorker(){}

    inline int get_target_level() { return m_level;}

    virtual void set_target_level(const int level) 
    {
#ifdef HEAVY_DEBUG
        assert(level >= 0); 
        assert(level <= MAX_PROGRESS); 
#endif
        m_level = level; 
    }
};

/**
* @class DrawWorker
* @brief Base class of worker for drawing / searching pixel.
*
* Abstruct Worker class of walking line or some basic pixel operation.
* This worker is used for pixel operations for all over the surface.
*/
class DrawWorker : public PixelWorker 
{
protected:
public:
    DrawWorker(FlagtileSurface* surf) 
        : PixelWorker(surf) { }

    /**
    * @step
    * processing current pixel
    *
    * @param x,y Surface coordinate of current pixel
    * @detail 
    * If pixel operation is not done(failed) by some reason,
    * return false. Otherwise, return true.
    */    
    virtual bool step(const int x, const int y) = 0;
};

/**
* @class TileWorker
* @brief Abstruct class of tile-based pixel operation.
*
*/
class TileWorker : public PixelWorker 
{
protected:
    
    // process only outerrim ridges of a tile.
    // use for a tile which is filled some specific value.
    void process_only_ridge(Flagtile *targ, const int sx, const int sy)
    {
        int ridge = PROGRESS_TILE_SIZE(m_level);

        for(int y=0;y < ridge;y+=ridge-1){
            for(int x=0;x < ridge;x++){
                step(targ, x, y, sx+x, sy+y);
            }
        }

        // Corner pixels are already processed at above loop.
        for(int x=0;x < ridge;x+=ridge-1){
            for(int y=1;y < ridge-1;y++){
                step(targ, x, y, sx+x, sy+y);
            }
        }
    }

    inline uint8_t get_neighbor_pixel(const int level,
                                      const int sx, const int sy, 
                                      const int direction); 

public:
    TileWorker(FlagtileSurface* surf) 
        : PixelWorker(surf) { }
    
    // `start` called at the starting point of tile processing.
    // All processing cancelled when this return false.
    virtual bool start(Flagtile *tile, const int sx, const int sy) 
    {
        return true; // As a default, always return true.
    }
   
    /**
    * @step
    * processing current pixel
    *
    * @param x,y tile-local progress coordinate of current pixel
    * @param sx,sy surface-global progress coordinate of current pixel
    * @detail 
    * Workers should implement this method.
    */    
    virtual void step(Flagtile* tile, 
                      const int x, const int y,
                      const int sx, const int sy) { }
    
    // Called when a tile processing end.
    virtual void end( Flagtile* tile) { }
    
    // Called when entire tiles processing end.
    virtual void finalize() { }
    
    // Offsets to refer neighboring pixels. 
    // This is public. Some class might refer them.
    static const int xoffset[];
    static const int yoffset[];
};

/**
* @class FillWorker
* @brief Dedicated pixelworker for flood-fill operation
*
*/
class FillWorker : public TileWorker
{
protected:
public:
    FillWorker(FlagtileSurface* surf)
        : TileWorker(surf) { }
    
    // To check whether a pixel to be processed or not. 
    // `match` and `step` methods are almost same, it seems to be done
    // at once.
    // but some methods such as FlagtileSurface::_tile_flood_till will needs
    // to only look(check) the pixel, without process it,
    // so they are separated.
    virtual bool match(const uint8_t pix) = 0;
};

/**
* @class KernelWorker
* @brief Base class of Filter Kernel workers. Used in FlagtileSurface::filter method.
* 
* Abstruct class of pixel filter worker.
*/
class KernelWorker : public TileWorker 
{
protected:
    // Cache of m_surf information
    int m_max_x;
    int m_max_y;

public:
    // Defined at lib/progfill.cpp
    KernelWorker(FlagtileSurface *surf)
        : TileWorker(surf) { }

    virtual void set_target_level(const int level);

    virtual bool start(Flagtile *tile, const int sx, const int sy);
    virtual void end(Flagtile *tile);
    
    virtual void finalize();
};

// Walking kernel base class.
// This class implements a function which walk around 
// specific(mostly PIXEL_FILLED) area ridge.
// Walking argorithm is wall-follower(right-hand rules)
class WalkingKernel: public KernelWorker
{
protected:
    // Direction constants.
    // These MUST be same as the order of KernelWorker::xoffset/yoffset.
    static const int DIR_TOP = 0;

    // Current position of the kernel.
    int m_x;
    int m_y;
    
    // Origin point. 
    // when m_x and m_y reaches here, walking end.
    int m_ox;
    int m_oy;

    // Current forwarding direction. Initially, DIR_TOP.
    // And `current hand direction` would be
    // (m_cur_dir + 1) % 4 == (m_cur_dir + 1) & 3
    int m_cur_dir;

    // Walking step count.
    // Some kernel reset this for its purpose.
    // Usually this represents the perimeter of walked area.
    int m_step;

    // To detect 1px infinite loop. if this is greater or equal to 4, 
    // Walking kernel enters infinite rotating state in 1px hole.
    int m_right_rotate_cnt;

    // To detect walking is closewise or counter-clockwise.
    long m_clockwise_cnt;


    // Wrapper method to get pixel with direction.
    virtual uint8_t get_pixel_with_direction(const int x, const int y, 
                                              const int direction);

    inline uint8_t get_front_pixel() 
    {
        return get_pixel_with_direction(m_x, m_y, m_cur_dir);
    }

    inline uint8_t get_hand_pixel() 
    {
        return get_pixel_with_direction(m_x, m_y, get_hand_dir(m_cur_dir));
    }

    //// Walking related.
    
    // Rotate to right.
    // This is used when we missed wall at right-hand. 
    void rotate_right();
    
    // Rotate to left. 
    // This is used when we face `wall`. called from proceed().
    void rotate_left();

    // Go forward. 
    // Called from proceed().
    bool forward();

    // Rotate or Proceed (walk) single step, around the target area.
    // when reaches end of walking, return false.
    bool proceed();

    
    //// Walking callbacks / virtual methods
    
    // Rotation callback. 
    // If parameter `right` is true, kernel turns right. 
    // otherwise turns left.
    virtual void on_rotate_cb(const bool right) { }

    // `Entering new pixel` callback.
    // This called when forward() method go (forward) into new pixel.
    // Current pixel is ensured as `forwardable` target pixel.
    virtual void on_new_pixel() { }

    // Check whether the right side pixel of current position / direction
    // is match to forward.
    virtual bool is_wall_pixel(const uint8_t pixel);

    
public:
    WalkingKernel(FlagtileSurface *surf)
        : KernelWorker(surf) { } 

    // start/end should be implemented in child class.
    virtual bool start(Flagtile* targ, const int sx, const int sy) 
    {
        return false;
    }

    virtual void end(Flagtile* targ) { }

    // Tell whether the walking is clockwise or not.
    // We can use this method only after the walking finished.
    inline bool is_clockwise() {
        return m_clockwise_cnt <= 0;
    }

    // These methods are used when calling walk method from outside.
    inline int get_hand_dir(const int dir) { return (dir + 3) & 3; }
    inline int get_reversed_hand_dir(const int dir) { return (dir + 1) & 3; }
    void walk(const int sx, const int sy, const int direction);
};

// XXX Currently almost same as floodfill_point of fill.cpp,
// But progfill_point member might be changed in future.
typedef struct {
    int x;
    int y;
} progfill_point;

// progfill_tilepoint used for flood-fill operation of FlagtileSurface.
typedef struct {
    int x; // Pixel location of tile 
    int y;
    int tx; // Tile location
    int ty;
} progfill_tilepoint;

#endif
