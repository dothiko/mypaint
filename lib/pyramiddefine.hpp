/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

// This file is to avoid polluting mypaintlib namespace, because these
// worker classes cannot be used from python.
// Also, all classes in this file are base class, so all
// uses protected keyword as inner method/members.

#ifndef PYRAMIDDEFINE_HPP
#define PYRAMIDDEFINE_HPP

/* This File is for utility class definition for
 * close-and-fill functionality.
 *
 */

#include <Python.h>
#include <mypaint-tiled-surface.h>

// Force positive modulo against even negative number.
// C/C++ would produce platform-dependent result
// with modulo against negative number.
// This macro is used in pyramid.cpp/hpp
#define POSITIVE_MOD(a, b) (((a) % (b) + (b)) % (b))

// logarithmic value of MYPAINT_TILE_SIZE. i.e. 2**6 = 64 pixel.
#define TILE_LOG 6

// MAX pyramid level. 5 means maximum 2**5 == 32 pixel.
#define MAX_PYRAMID ((TILE_LOG)-1)

// Convert a progress level 0(source or final result) coordinate
// into each progress level (greater than 0) coordinate.
#define PYRAMID_PIXEL_COORD(a, l) ((a) >> (l))

// XXX CAUTION: PYRAMID_TILE_SIZE(0) MUST BE SAME AS MYPAINT_TILE_SIZE.
#define PYRAMID_TILE_SIZE(l) (1 << (TILE_LOG - (l)))
#define PYRAMID_BUF_SIZE(l) (PYRAMID_TILE_SIZE(l) * PYRAMID_TILE_SIZE(l))

#define FLAGTILE_BUF_SIZE  (PYRAMID_BUF_SIZE(0) + \
                            PYRAMID_BUF_SIZE(1) + \
                            PYRAMID_BUF_SIZE(2) + \
                            PYRAMID_BUF_SIZE(3) + \
                            PYRAMID_BUF_SIZE(4) + \
                            PYRAMID_BUF_SIZE(5))

// Maximum Anti-Aliasing transparency level.
#define MAX_AA_LEVEL 127

// For Flagtile class. to get progress-level ptr from
#define BUF_PTR(l, x, y) (m_buf + m_buf_offsets[(l)] + ((y) * PYRAMID_TILE_SIZE((l)) + (x)))

// For Flagtile class. pixel-counter buffer size, in bytes.
#define FLAGTILE_CNT_SIZE ((PIXEL_MAX+1) * (MAX_PYRAMID+1) * sizeof(uint16_t))

// For Flagtile class. to get pixel-counter ptr from
#define CNT_PTR(l, p) (m_pixcnt + ((l) * (PIXEL_MAX+1)) + ((p) & PIXEL_MASK))

// XXX PIXEL_ Constants.
//
// PIXEL_ values are not bitwise flag. They are just number.
// The vacant pixel is 0 = PIXEL_EMPTY.

#define PIXEL_MASK 0x07
// PIXEL_* value are from 0 to maximum 7.

#define PIXEL_EMPTY 0x00
// PIXEL_EMPTY should lower than PIXEL_OUTSIDE,
// and both of them are `invalid pixel`.
// When using FlagtileSurface.get_pixel method against
// NULL-tile position, PIXEL_EMPTY would be returned.
#define PIXEL_OUTSIDE 0x01
// Thus we can know invalid pixel as less than or equal PIXEL_OUTSIDE
// Also, there is a special pixel value `PIXEL_INVALID`.
// It is for Flagtile::is_filled_with method.

#define PIXEL_AREA 0x02
// PIXEL_AREA means `The pixel is fillable, but not filled(yet)`

#define PIXEL_FILLED 0x03
// PIXEL_FILLED means `The pixel is filled.`

#define PIXEL_CONTOUR 0x04
// PIXEL_CONTOUR means `The border of pixels`, typically lineart contour.
// This should be larger than PIXEL_FILLED
// to ease finding `unchangeable` pixel.

#define PIXEL_OVERWRAP 0x05
// Use to eliminate overwrapped pixels for `cut protruding`
// feature. Without this value, we cannot detect overwrapped
// parts under the contour pixels.

#define PIXEL_RESERVE 0x06
// A placeholder pixel, which is Used `2pass flood-fill`
// of pyramid-flood-fill.
// Larger pyramid level pixel (gap-closing pixel) would
// easily stuck flood-fill search queue and flood-fill
// operation end up incompletely.
// So we get `entire fillable` area at lower pyramid level
// as PIXEL_RESERVE, then fill it with PIXEL_FILLED at higher
// pyramid level.
//
// XXX NOTE: This pixel value should NOT be propergated to
// lower level in ProgressKernel::step.

#define PIXEL_MAX 0x07 
// Actual maximum pixel value.
// PIXEL_* values of greater than this value is `imaginary`,
// used for special parameter of some functions,
// never placed in tile.

#define PIXEL_INVALID 0x08
// Special pixel value. This actually does not exist as pixel.
// This means `PIXEL_EMPTY or PIXEL_OUTSIDE` and use for
// knowing a tile has any of valid pixels(i.e, AREA, FILLED,
// or CONTOUR) or not.
// This is for Flagtile::is_filled_with method.

// XXX NOTE:
// PIXEL_EMPTY, PIXEL_AREA, PIXEL_FILLED and PIXEL_CONTOUR are redefined as
// static const of Flagtile class, and you can access them from python
// as lib.mypaintlib.Flagtile.PIXEL_*.
// Other enum constants are hidden from python.

// FLAG_ values are bitwise flag.
#define FLAG_MASK 0xF0

// FLAG_WORK flag is temporary flag, used for filter operation.
// This flag should be most significant bit, for final antialiasing.
#define FLAG_WORK 0x10

// FLAG_AA used for anti-aliasing.
// XXX CAUTION: With this flag set, the pixel contains 128 Level
// of antialias alpha value.
// Therefore, all of PIXEL_ and other FLAG_ constants are invalid
// for such `anti-aliased` pixel.
// And, FLAG_AA MUST be most significant bit (== 0x80 for uint8_t).
#define FLAG_AA 0x80
#define AA_MASK 0x7F

// Offset Direction constants, to refer KernelWorker::xoffset/yoffset.
#define OFFSET_TOP 0
#define OFFSET_RIGHT 1
#define OFFSET_BOTTOM 2
#define OFFSET_LEFT 3

// Forward declaration of Flagtile / FlagtileSurface.
// Actually they are defined at pyramidfill.hpp
// and exposed to python.
class Flagtile;
class FlagtileSurface;

/**
* @class BaseWorker
* @brief Base class of worker for drawing / searching pixel.
*
* Abstruct Worker class of walking line.
* This is to share same logic between drawing line
* and searching unfilled target area.
*/
class BaseWorker
{
protected:
    int m_level;

public:
    BaseWorker()
        : m_level(0)
    {
        // Calling virtual method `set_target_level`
        // at here (i.e. constructor) is meaningless.
        // Derived virtual function cannot be called from
        // constructor by C++ design.
        // So I use initialization list.
    }
    virtual ~BaseWorker(){}

    inline int get_target_level() { return m_level;}

    virtual void set_target_level(const int level)
    {
#ifdef HEAVY_DEBUG
        assert(level >= 0);
        assert(level <= MAX_PYRAMID);
#endif
        m_level = level;
    }

    inline bool in_border(const int x, const int y) 
    {
        return (x == 0 || y ==0
                    || x == PYRAMID_TILE_SIZE(m_level)-1
                    || y == PYRAMID_TILE_SIZE(m_level)-1);
    }
};

/**
* @class Filler
* @brief Dedicated pixelworker for flood-fill operation
*
* Actually this worker is a bit different from Baseworker.
* This worker does not use m_surf.
*/
class Filler : public BaseWorker
{
protected:
public:
    Filler() :
        BaseWorker() {}  

    // To check whether a pixel to be processed or not.
    // `match` and `step` methods are almost same, it seems to be done
    // at once.
    // but some methods such as FlagtileSurface::_tile_flood_till will needs
    // to only look(check) the pixel, without process it,
    // so they are separated.
    virtual bool match(const uint8_t pix) = 0;

    virtual void step(Flagtile *t, const int x, const int y) = 0;

    // Just a stub.
    // Some derived class might not be able to decrible target pixel as
    // a single pixel.
    // Because it can be so complicated.
    virtual uint8_t get_fill_pixel(){ return PIXEL_INVALID; }
    virtual uint8_t get_target_pixel(){ return PIXEL_INVALID; }
};

/**
* @class KernelWorker
* @brief Base class of Filter Kernel workers. Used in FlagtileSurface::filter method.
*
* Abstruct class of pixel filter kernel.
* Kernelworker has ability to access neighbouring 4-direction pixels of
* specific location.
*/
class KernelWorker : public BaseWorker
{
protected:
    FlagtileSurface* m_surf;
    static Flagtile *m_shared_empty;
    
    // Utility method to access neighbor pixels.
    // You can use offset constant(macro) such as OFFSET_UP/RIGHT/DOWN/LEFT
    // This is virtual, because needs to be overridden in AntialiasWalker.
    virtual uint8_t get_pixel_with_direction(const int sx, const int sy,
                                             const int direction);

public:
    // Defined at lib/pyramid.cpp
    KernelWorker(FlagtileSurface *surf)
        : BaseWorker(),
          m_surf(surf)
    { }

    virtual void set_target_level(const int level);

    // `start` called at the starting point of tile processing.
    // All processing cancelled when this return false.
    virtual bool start(Flagtile * const targ, const int sx, const int sy) = 0;

    virtual void step(Flagtile * const tile,
                      const int x, const int y,
                      const int sx, const int sy) = 0;
    
    // process only outerrim ridges of a tile.
    // use for a tile which is filled some specific value.
    void process_only_ridge(Flagtile *targ, const int sx, const int sy)
    {
        int ridge = PYRAMID_TILE_SIZE(m_level);

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

    // Called when a tile processing end.
    virtual void end(Flagtile * const targ){}
    
    // Shared-empty tile related.
    static inline bool is_shared_empty(Flagtile* t){return t==m_shared_empty;}
    static Flagtile *get_shared_empty();
    static bool sync_shared_empty(const int level);
    static void free_shared_empty();

    // Offsets to refer neighboring pixels.
    // This is public. Some class might refer them.
    static const int xoffset[];
    static const int yoffset[];
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
    // walker enters into 1px hole, so just quit walking.
    int m_right_rotate_cnt;

    // To detect walking is closewise or counter-clockwise.
    long m_clockwise_cnt;

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
    // void type Callbacks are optional.

    // Rotation callback.
    // If parameter `right` is true, kernel turns right.
    // otherwise turns left.
    virtual void on_rotate_cb(const bool right){}

    // `Entering new pixel` callback.
    // This called when forward() method go (forward) into new pixel.
    // Current pixel is ensured as `forwardable` target pixel.
    virtual void on_new_pixel(){}

    // Check whether the right side pixel of current position / direction
    // is match to forward.
    virtual bool is_wall_pixel(const uint8_t pixel) = 0;

public:
    WalkingKernel(FlagtileSurface *surf)
        : KernelWorker(surf) { }

    // start/end should be implemented in child class.

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

// To decrease memory usage, use short. 
// (actually, enough with `char`...)
typedef struct {
    short x;
    short y;
} floodfill_point;
#endif