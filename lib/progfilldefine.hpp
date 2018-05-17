/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
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
// This macro is used in closefill.cpp/hpp
#define positive_mod(a, b) (((a) % (b) + (b)) % (b))

#define MAX_PROGRESS 6

// Convert a progress level 0(source or final result) coordinate
// into each progress level (greater than 0) coordinate.
#define PROGRESS_PIXEL_COORD(a, l) ((a) >> (l)) 

// XXX CAUTION: PROGRESS_TILE_SIZE(0) MUST BE SAME AS MYPAINT_TILE_SIZE.
#define PROGRESS_TILE_SIZE(l) (1 << (MAX_PROGRESS - (l)))
#define PROGRESS_BUF_SIZE(l) (PROGRESS_TILE_SIZE(l) * PROGRESS_TILE_SIZE(l))

#define TILE_SIZE MYPAINT_TILE_SIZE

#define MAX_AA_LEVEL 127

// For Flagtile class. to get progress-level ptr from
#define _BUF_PTR(l, x, y) (m_buf + m_buf_offsets[(l)] + ((y) * PROGRESS_TILE_SIZE((l)) + (x))) 

// PIXEL_ Constants.
enum PixelFlags {
    // PIXEL_ values are not bitwise flag. They are just number.
    // The vacant pixel is 0.
    // PIXEL_AREA means 'The pixel is fillable, but not filled(yet)'
    PIXEL_MASK = 0x0F,
    PIXEL_EMPTY = 0x00,
    PIXEL_INVALID = 0x01, // Invalid pixel, to notify the FillWorker would
                          // generate multiple type of pixel.
    PIXEL_AREA = 0x02,
    PIXEL_OUTSIDE = 0x03,  
    PIXEL_REMOVE = 0x04, // The ridge of an area to be removed.
    PIXEL_FILLED = 0x05,
    PIXEL_CONTOUR, // PIXEL_CONTOUR is one of a filled pixel.
                   // This should be larger than PIXEL_FILLED
                   // to ease finding `filled` pixel.
                   // So, PIXEL_CONTOUR should be largest value
                   // of PIXEL_ constants, and defined next to PIXEL_FILLED.
                         
    // FLAG_ values are bitwise flag. 
    FLAG_MASK = 0xF0,
    
    // FLAG_WORK flag is temporary flag, used for filter operation. 
    // This flag should be most significant bit, for final antialiasing.
    FLAG_WORK = 0x10, 
    
    // FLAG_DECIDED means `This pixel is decided to convert to final color pixel`
    FLAG_DECIDED = 0x20,

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

class PixelWorker {
protected:
    FlagtileSurface* m_surf;
    int m_level;

public:
    PixelWorker(FlagtileSurface* surf, const int level) 
        : m_surf(surf), m_level(level) 
    {
        // Calling virtual method `set_target_level` 
        // at here (i.e. constructor) is meaningless.
        // It cannot call derived virtual one by C++
        // design.
        // So I use initialization list.
    }
    virtual ~PixelWorker(){}

    inline int get_target_level() { return m_level;}

    virtual void set_target_level(const int level) {
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
class DrawWorker : public PixelWorker {
protected:
public:
    DrawWorker(FlagtileSurface* surf, const int level) 
        : PixelWorker(surf, level) 
    {
        // Calling virtual method `set_target_level` 
        // at here (i.e. constructor) is meaningless.
        // It cannot call derived virtual one by C++
        // design.
        // So I use initialization list.
    }

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
class TileWorker : public PixelWorker {
protected:
    int m_processed; // Processed pixel count. update this in child class.
    
public:
    TileWorker(FlagtileSurface* surf, const int level) 
        : PixelWorker(surf, level) ,
          m_processed(0)
    {
    }
    
    inline const int get_processed_count() {
        return m_processed;
    }  
               
    // `start` called at the starting point of tile processing.
    // All processing cancelled when this return false.
    virtual bool start(Flagtile *tile) {
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
                      const int sx, const int sy) = 0;
    
    // Called when a tile processing end.
    virtual void end( Flagtile* tile){}
    
    // Called when entire tiles processing end.
    virtual void finalize(){}
};

/**
* @class FillWorker
* @brief Dedicated pixelworker for flood-fill operation
*
*/
class FillWorker : public TileWorker{
protected:
public:
    FillWorker(FlagtileSurface* surf, const int level) 
        : TileWorker(surf, level) 
    {
        // Calling virtual method `set_target_level` 
        // at here (i.e. constructor) is meaningless.
        // It cannot call derived virtual one by C++
        // design.
        // So I use initialization list.
    }
    
    // To check whether a pixel to be processed or not. 
    // `match` and `step` methods are almost same, it seems to be done
    // at once.
    // but some methods such as FlagtileSurface::_tile_flood_till will needs
    // to only look(check) the pixel, without process it,
    // so they are separated.
    virtual bool match(const uint8_t pix) = 0;
    
    // To tell which pixel this worker would draw.
    // If it is not sure,(i.e. Fill different pixel depending on some situation) 
    // return PIXEL_INVALID.
    virtual uint8_t get_fill_pixel() {
        return PIXEL_INVALID;
    } 
};

/**
* @class KernelWorker
* @brief Base class of Filter Kernel workers. Used in FlagtileSurface::filter method.
* 
* Abstruct class of pixel filter worker.
*/
class KernelWorker : public TileWorker {
protected:
    // Cache of m_surf information
    int m_max_x;
    int m_max_y;

    static const int xoffset[];
    static const int yoffset[];

    // Current filter-kernel pixel position.
    // These can be refreshed with
    // get_kernel_pixel method.
    int m_px;
    int m_py;

public:
    // Defined at lib/progfill.cpp
    KernelWorker(FlagtileSurface *surf, const int level);


    virtual void set_target_level(const int level);

    virtual bool start(Flagtile *tile);
    virtual void end(Flagtile *tile);
    
    /**
    * @finalize
    * Called when the entire pixel operation has end.
    *
    * @detail 
    * This method called when the entire pixel operation has end.
    * It seems that such codes should be written in destructor,
    * but virtual destructor would be called parent class one,
    * it cannot be cancelled easily. so this method is created.
    */ 
    virtual void finalize();

    /**
    * @get_kernel_pixel
    * refresh internal members of kernel pixel position.
    *
    * @param sx, sy: The center pixel position, in surface coodinate.
    *                Most importantly, this position unit is in current ongoing
    *                `progress-level`.
    *                You can get that level with
    *                KernelWorker::get_target_level method. or just accessing
    *                m_level member.
    * @param idx: The index value of kernel pixel.
    *             0=left, 1=right, 2=top, 3=bottom.
    * @return: pixel value of current kernel.
    *          If that kernel position exceeds surface border,
    *          return 0.
    * @detail 
    * Utility method to update valid filter-kernel(4 surrounding pixel)
    * coordinates within for-loop.
    * That pixel position is stored in member variables, m_px and m_py.
    * If that position is off the FlagtileSurface (invalid position),
    * this method returns false.
    * So continue processing only when this method return true.
    */
    uint8_t get_kernel_pixel(const int level,
                             const int sx, const int sy,
                             const int idx);
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
    int m_left_rotate_cnt;

    // To detect walking is closewise or counter-clockwise.
    long m_clockwise_cnt;

    inline int _get_hand_dir(const int dir) { return (dir + 1) & 3; }

    // Wrapper method to get pixel with direction.
    virtual uint8_t _get_pixel_with_direction(const int x, const int y, 
                                              const int direction);

    inline uint8_t _get_front_pixel() {
        return _get_pixel_with_direction(m_x, m_y, m_cur_dir);
    }

    inline uint8_t _get_hand_pixel() {
        return _get_pixel_with_direction(m_x, m_y, _get_hand_dir(m_cur_dir));
    }

    //// Walking related.
    
    // Rotation callback. 
    // if `right` is true, kernel turns right. otherwise turns left.
    virtual void _on_rotate_cb(const bool right) = 0;

    // Check whether the right side pixel of current position / direction
    // is match to forward.
    virtual bool _is_match_pixel(const uint8_t pixel);

    // Rotate to right.
    // This is used when we missed wall at right-hand. 
    void _rotate_right();
    
    // Rotate to left. 
    // This is used when we face `wall`. called from _proceed().
    void _rotate_left();

    // Go forward. 
    // Called from _proceed().
    bool _forward();

    // Rotate or Proceed (walk) single step, around the target area.
    // when reaches end of walking, return false.
    bool _proceed();

    void _walk(const int sx, const int sy, const int direction);
    
public:
    WalkingKernel(FlagtileSurface *surf, const int level) 
        : KernelWorker(surf, level)
    {} 

    // Caution: Child class of this kernel should accepts
    // NULL tile.
    virtual bool start(Flagtile* targ);

    // To disable default end method.
    virtual void end(Flagtile* targ) {}

    // Tell whether the walking is clockwise or not.
    // We can use this method only after the walking.
    inline bool is_clockwise() {
        return m_clockwise_cnt < 0;
    }

};

// XXX Currently almost same as _floodfill_point of fill.cpp,
// But _progfill_point member might be changed in future.
typedef struct {
    int x;
    int y;
} _progfill_point;

// _progfill_tilepoint used for flood-fill operation of FlagtileSurface.
typedef struct {
    int x; // Pixel location of tile 
    int y;
    int tx; // Tile location
    int ty;
} _progfill_tilepoint;


#endif
