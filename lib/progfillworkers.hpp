/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef PROGFILLWORKERS_HPP
#define PROGFILLWORKERS_HPP

// This file is to hide worker class from SWIG/Python interface.

#include <glib.h>
#include "progfill.hpp"

// Base worker/kernel classes defined at progfilldefine.hpp

//-----------------------------------------------------------------------------
//// Kernel classes
// Read notes around KernelWorker base class at progfilldefine.hpp

// Dilation kernel.
class DilateKernel : public KernelWorker
{
protected:
public:
    DilateKernel(FlagtileSurface *surf)
        : KernelWorker(surf, 0) 
    {}

    virtual bool start(Flagtile *targ, const int sx, const int sy) {
        // We cannot use parent class method for DilateKernel.
        // Because this kernel accepts NULL tile.
        if (targ == NULL) {
            _process_only_ridge(targ, sx, sy);
            end(targ);
            return false;
        }
        else {
            if (targ->get_stat() & Flagtile::FILLED) {
                // There is no space to be dilated.
                return false;
            }
        }
        return true;
    }

    virtual void step(Flagtile *targ, 
                      const int x, const int y,
                      const int sx, const int sy) {
        uint8_t pix;
        if (targ == NULL) {
            pix = PIXEL_EMPTY;
        }
        else {
            pix = targ->get(0, x, y);
        }

        if (pix != PIXEL_FILLED && (pix & FLAG_WORK) == 0) {
            for(int i=0; i<4; i++) {
                uint8_t kpix = _get_neighbor_pixel(0, i, sx, sy);
                if (kpix == PIXEL_FILLED) {
                    if (targ == NULL) {
                        // We need to generate(or get) a new tile.
                        int tile_size = PROGRESS_TILE_SIZE(0);
                        targ = m_surf->get_tile(sx/tile_size, sy/tile_size, 
                                                true);
                    }
                    targ->replace(0, x, y, PIXEL_FILLED | FLAG_WORK);
                    return;
                }
            }
        }
    }
};

// Eroding contour pixel kernel, this is used from
// LassofillSurface. 
// With eroding contour, that surface does
// actually dilate the final result area. 
class ErodeContourKernel : public DilateKernel
{
protected:
public:
    ErodeContourKernel(FlagtileSurface *surf)
        : DilateKernel(surf) {
    }

    // Different from DilateKernel, erosion might occur
    // even completely filled tile.
    // Outer-rim of a filled tile might be neighboring vacant pixel.
    virtual bool start(Flagtile *targ, const int sx, const int sy) {
        if (KernelWorker::start(targ, sx, sy)) {
            if ((targ->get_stat() & Flagtile::FILLED) == 0
                    && (targ->get_stat() & Flagtile::FILLED_AREA) == 0) {
                return true;
            }
        }
        return false;
    }

    virtual void step(Flagtile *targ, 
                      const int x, const int y,
                      const int sx, const int sy) {
        uint8_t pix = targ->get(0, x, y);
        if ((pix & PIXEL_MASK) == PIXEL_CONTOUR) {
            for(int i=0; i<4; i++) {
                uint8_t kpix = _get_neighbor_pixel(0, i, sx, sy);
                // Chech whether kpix is exactly PIXEL_AREA
                // (no working flag set)
                // To avoid chain reaction.
                if (kpix == PIXEL_AREA) {
                    targ->replace(0, x, y, PIXEL_AREA | FLAG_WORK);
                    return; // Exit without adding filled count.
                }
            }
        }
    }
};

// Pixel converter kernel
// Also, we need to use this kernel before progress
// to mark the tile has a at least one valid pixel or not.
class ConvertKernel : public KernelWorker
{
protected:
    uint8_t m_targ_pixel;
    uint8_t m_new_pixel;

public:
    ConvertKernel(FlagtileSurface *surf, const int level, 
                  const uint8_t targ_pixel, const uint8_t new_pixel)
        : KernelWorker(surf, level), 
          m_targ_pixel(targ_pixel),
          m_new_pixel(new_pixel) {
    }

    // Another contsructor. Used for class member.
    ConvertKernel(FlagtileSurface *surf) 
        : KernelWorker(surf, 0), 
          m_targ_pixel(0),
          m_new_pixel(0) {
    }

    // To avoid repeatedly generate convert-kernel object,
    // Some class reuse only one kernel object over a series of processes.
    // For such case, we need this (re)setup method.
    void setup(const int level, 
               const uint8_t targ_pixel, 
               const uint8_t new_pixel) {
        set_target_level(level);
        m_targ_pixel = targ_pixel;
        m_new_pixel = new_pixel;
    }

    virtual bool start(Flagtile *targ, const int sx, const int sy) {
        if (targ == NULL && m_targ_pixel == PIXEL_EMPTY) {
            targ = m_surf->get_tile_from_pixel(0, sx, sy, true);
            targ->fill(m_new_pixel);
            return false;
        }
        else if (KernelWorker::start(targ, sx, sy)) {
            if (targ->is_filled_with(m_targ_pixel)) {
                targ->fill(m_new_pixel);
                return false;
            }
            return true;
        }
        return false;
    }

    virtual void step(Flagtile *targ, 
                      const int x, const int y,
                      const int sx, const int sy) {
        uint8_t pix = targ->get(m_level, x, y);
        if (pix == m_targ_pixel) {
            targ->replace(m_level, x, y, m_new_pixel);
            pix = m_new_pixel;
        }
    }
};

// Progressive fill Kernel.
// This Kernelworker is very important, to generate
// filled pixel area with virtually gap-closing.
class ProgressKernel: public KernelWorker
{
protected:
public:
    // This class does not use initial level parameter.
    // Because it would be changed frequently.
    // It would be set at loop, with `set_target_level` method.
    ProgressKernel(FlagtileSurface *surf) 
        : KernelWorker(surf, 0) {}

    // Disable default finalize. 
    // This kernel does not use FLAG_WORK.
    virtual void finalize() {} 
    
    virtual bool start(Flagtile *targ, const int sx, const int sy) {
        if (KernelWorker::start(targ, sx, sy)) {
            // Target tile 
            int stat = targ->get_stat();
            if ((stat & Flagtile::FILLED) || (stat & Flagtile::EMPTY)) {
                return false;
            }
            else if (stat & Flagtile::FILLED_AREA) {
                uint8_t above = targ->get(m_level, 0, 0);
                // FILLED_AREA means `filled with PIXEL_AREA 
                // (all pixels are same and undecided) at level-0`
                // But, at progressing stage, the highest pixels
                // already decided.
                // so, fill and completely decide it now (if needed).
                switch(above) {
                    case PIXEL_OUTSIDE:
                    case PIXEL_FILLED:
                        targ->fill(above);
                        break;
                    default:
                        _process_only_ridge(targ, sx, sy);
                        break;
                }
                end(targ);
                return false;
            }
            return true;
        }
        return false;
    }

    virtual void step(Flagtile *targ, 
                      const int x, const int y,
                      const int sx, const int sy) {
        // `pixel is greater than PIXEL_FILLED` 
        // means `PIXEL_FILLED or PIXEL_CONTOUR`
        uint8_t above = targ->get(m_level, x, y);
        int bx = x << 1;
        int by = y << 1;
        int beneath_level = m_level - 1;

        if (above == PIXEL_OUTSIDE || above == PIXEL_FILLED) {
            // Inherit above pixel 
            for(int py=by; py < by+2; py++) {
                for(int px=bx; px < bx+2; px++) {
                    targ->replace(beneath_level, px, py, above);
                }
            }
        }
        else {
            uint8_t top = _get_neighbor_pixel(m_level, 0, sx, sy);
            uint8_t right = _get_neighbor_pixel(m_level, 1, sx, sy);
            uint8_t bottom = _get_neighbor_pixel(m_level, 2, sx, sy);
            uint8_t left = _get_neighbor_pixel(m_level, 3, sx, sy);

            // The param x and y is current(above) progress coordinate.
            // The coordinate of level beneath should be double of them.
            uint8_t pix_v, pix_h;

            pix_v = top;
            for(int py=by; py < by+2; py++) {
                pix_h = left;
                for(int px=bx; px < bx+2; px++) {
                    uint8_t pix = targ->get(beneath_level, px, py);
                    if (pix == PIXEL_AREA) {
                        if (pix_h <= PIXEL_OUTSIDE || pix_v <= PIXEL_OUTSIDE) { 
                            targ->replace(beneath_level, px, py, PIXEL_OUTSIDE);
                        }
                        else if (pix_h == PIXEL_FILLED || pix_v == PIXEL_FILLED) {
                            targ->replace(beneath_level, px, py, PIXEL_FILLED);
                        }
                    }
                    pix_h = right;
                }
                pix_v = bottom;
            }
        }
    }   
};


// Antialias kernel.
// This class does (psuedo)antialias by walking around the filled area ridge
// and draw gradient lines around there. 
//
// NOTE: In this kernel, m_step will be consumed each time AA-line drawn
// (i.e. rotation occur).
class AntialiasKernel: public WalkingKernel
{
protected:
    // Anti-aliasing line start position
    int m_sx;
    int m_sy;

    // Current anti-aliasing line direction.
    int m_line_dir;

    // Actual walking started flag.
    // This kernel cannot start walking from
    // intermidiate of a ridge.
    bool m_walking_started;
    
    // Dedicated wrapper method to get pixel with direction.
    virtual uint8_t _get_pixel_with_direction(const int x, const int y, 
                                              const int direction) {
        uint8_t pix = m_surf->get_pixel(m_level,
                                        x + xoffset[direction],
                                        y + yoffset[direction]);

        // We need to distinguish AA pixel and others
        // Because Anti-aliasing gradient seed pixel 
        // is completely different from ordinary pixel. 
        // That AA pixel might be conflict with other PIXEL_ constants,
        // and might be misdetected as it.
        if ((pix & FLAG_AA) != 0)
            return 0;
        else
            return pix;
    }

    //// Drawing related.
    
    // Get anti-aliasing value (0.0 to 1.0) from a pixel point.
    double _get_aa_value(
        int x, int y, const bool is_start, const bool face_wall) {
        if (is_start) {
            // For staring pixel check, we need the `back` pixel
            // to determine starting point value.
            int back_dir = (m_line_dir + 2) & 3;
            x += xoffset[back_dir];
            y += yoffset[back_dir];
        }
        else if (face_wall) {
            // Currently kernel is face to wall. This means also 
            // `Kernel (would) turns left after AA line drawn.`
            // Otherwise, kernel will turns right.
            x += xoffset[m_line_dir];
            y += yoffset[m_line_dir];
        }
        else {
            return 0;
        }

        if ((m_surf->get_pixel(0, x, y) & PIXEL_MASK) == PIXEL_FILLED) {
            int leftdir = (m_line_dir + 3) & 3;
            uint8_t side_pixel = _get_pixel_with_direction(x, y, leftdir);
            if ((side_pixel & PIXEL_MASK) == PIXEL_FILLED) {
                // The target pixel is `cliff`
                // It is not suitable as antialias point.
                return 0;
            }
            // Currently, kernel is placed just 1px `step`.
            // It would be a typical endpoint of AA line.
            return 1.0;
        }
        else {
            return 0;
        }
    }

    void _draw_antialias(const int ex, const int ey, const bool right) {
        if (m_step > 0) { 
            if ((m_sx == ex) && (m_sy == ey)) {
                // This might be 1px anti-aliasing. 
                // But there might be already drawn AA pixel.
                if (!(m_surf->get_pixel(0, ex, ey) & FLAG_AA)) {
                    m_surf->replace_pixel(0, ex, ey, MAX_AA_LEVEL>>1);
                }
                return;
            }
       
            // `not right` means `left turn`.
            double start_value = _get_aa_value(m_sx, m_sy, true, !right);
            double end_value = _get_aa_value(ex, ey, false, !right);

            // Gradient from 0 to 0 is meaningless. 
            if (start_value == 0 && end_value == 0) 
                return;
            // Otherwise, gradient from 1 to 1 is `symetric gradient`.

            // drawing antialias line.
            int cx = m_sx;
            int cy = m_sy;
            double value;
            double value_step;
            int pixel_step = m_step;

            if (start_value == end_value) // A symetric gradient.
                value_step = (1.0 / (m_step+1)) * 2;
            else
                value_step = 1.0 / (m_step+1);

            if (start_value == 1.0)
                value_step = -value_step;
                
            if (!right)
                pixel_step++;

            // Set initial value.
            // Anti-aliasing pixel never starts from extream(0.0 or 1.0) value.
            value = start_value + value_step;

            // Drawing anti-alias grayscale seed value.
            // This is `seed value`, not actual grayscale value.
            // That seed value would be used as a factor of
            // calculating actual alpha value, when converting
            // Flagtile into mypaint color tile.
            for(int i=0; i < pixel_step; i++) {
                uint8_t pix = (uint8_t)(value * MAX_AA_LEVEL) | FLAG_AA;
                m_surf->replace_pixel(0, cx, cy, pix);
                value += value_step;
                if (value < 0) {
                    value = 0;
                    // Invert sign.
                    // This make anti-alasing gradient
                    // as bi-directional.
                    value_step = -value_step; 
                }
                cx += xoffset[m_line_dir]; 
                cy += yoffset[m_line_dir]; 
            }
        }
    }

    //// Walking related.

    // Initialize origin or draw antialiasing line.
    virtual void _on_rotate_cb(const bool right) {
        if (m_walking_started ) {
            if (m_step > 0) {
               _draw_antialias(m_x, m_y, right);
            }
        } 
        else {
            // Adjust origin point
            m_ox = m_x;
            m_oy = m_y;
            m_walking_started = true;
        }

        if (right) {
            // This is for after right rotation.
            // And, actual starting point is one pixel forward.
            // NOTE: Therefore, we need to update(rotate) m_curdir
            // before call this method.
            m_sx = m_x + xoffset[m_cur_dir];
            m_sy = m_y + yoffset[m_cur_dir];

            // Also, do not include current right-rotated position.
            m_step = -1;
        } 
        else {
            m_sx = m_x;
            m_sy = m_y;
            m_step = 0;
        }
        m_line_dir = m_cur_dir;
    }
 
    virtual bool _is_wall_pixel(const uint8_t pixel)  {
        return ((pixel & FLAG_AA)==0 && pixel == PIXEL_FILLED);
    }

public:
    AntialiasKernel(FlagtileSurface *surf) 
        : WalkingKernel(surf, 0) {
    }

    // Disable finalize method.
    //
    // Default finalize method recognize FLAG_* pixels as dirty
    // and erase all flags.
    // And, Antialiasing is the last procedure of progress-fill,
    // so there is no need to clear (dirty) flags.
    virtual void finalize() {}

    virtual bool start(Flagtile *targ, const int sx, const int sy) {
        // This kernel accepts NULL tile.
        if(targ == NULL) {
            return true;
        }
        else {
            if(targ->get_stat() & Flagtile::FILLED) {
                return false;
            }
            return true;
        }
    }
    
    // Caution: You should call this method with
    // `progress level` coordinate.
    virtual void step(Flagtile *targ, 
                      const int x, const int y,
                      const int sx, const int sy) {
        uint8_t pix;
        uint8_t pix_right;

        if (targ == NULL) {  
            // Only check the right edge of a tile.
            if (x < PROGRESS_TILE_SIZE(m_level)-1)
                return; 

            pix = 0;
            pix_right = m_surf->get_pixel(m_level, sx+1, sy);
        }
        else {
            pix = targ->get(m_level, x, y);
            if (x < PROGRESS_TILE_SIZE(m_level)-1)
                pix_right = targ->get(m_level, x+1, y);
            else
                pix_right = m_surf->get_pixel(m_level, sx+1, sy);
        }
        
        if ((pix & FLAG_AA) == 0 && (pix_right & FLAG_AA) == 0
                && (pix & PIXEL_MASK) != PIXEL_FILLED
                && (pix_right & PIXEL_MASK) == PIXEL_FILLED) { 
            // This class searches target pixel from left to right for each line.
            // And we use right-handed rule to walk area ridge.
            // Therefore, we would face top at the start point always.

            // IMPORTANT: initialize class unique members before walk!
            m_walking_started = false;
            m_line_dir = 0; // 0 means "Currently face top"
            m_sx = m_x;
            m_sy = m_y;
            _walk(sx, sy, 0);
        }
    }
};


// Removing Small filled area.
// This class is used in RemoveGarbageKernel.
class RemoveAreaWorker : public FillWorker 
{
protected:
public:
    RemoveAreaWorker(FlagtileSurface *surf,
                     const int targ_level) 
        : FillWorker(surf, targ_level) {
    }
    
    virtual bool start(Flagtile *tile) {
        if (tile->get_stat() & Flagtile::FILLED_AREA)
            return false;
        return true;
    }

    virtual bool match(const uint8_t pix) {
        return (pix & PIXEL_MASK) == PIXEL_FILLED;
    }

    virtual void step(Flagtile *tile,
                      const int x, const int y,
                      const int sx, const int sy) {
        tile->replace(m_level, x, y, PIXEL_AREA);
    }
};

// Stores perimeter information.
// To execute some operation later.
// (For example, erase pixel flag)
typedef struct {
    int sx;
    int sy;
    int length;
    int direction;
    int encount;
    bool clockwise;
} _perimeter_info;

// Count perimeter of filled area.
// This class just walk and count perimeter, and stores that position, length and 
// `encount` values into GQueue object.
// Walked regions are marked with FLAG_DECIDED flag, and it is erased later.
class CountPerimeterKernel: public WalkingKernel
{
protected:
    int m_encount; // A counter, incremented when walker `touch` outside/vacant pixels. 
    GQueue* m_queue; // Borrowed queue, set at start method.
    
    // tells whether the pixel value is wall pixel or not.
    virtual bool _is_wall_pixel(const uint8_t pixel) {
        switch(pixel & PIXEL_MASK) {
            case PIXEL_FILLED:
                return false;

            default: // Both of PIXEL_OUTSIDE, PIXEL_EMPTY
                m_encount++;
                // Fall Through.
            case PIXEL_AREA:
            case PIXEL_CONTOUR:
                // Above pixels are not 
                // `encount` pixel.
                return true;
        }
    }

    virtual void _on_rotate_cb(const bool right) {
        // If rotating right, it is `vacant pixel` of perimeter.
        // so decliment perimeter(i.e. m_step).
        if (right)
            m_step--;
    }

    virtual void _on_new_pixel() {
        // Mark this pixel as `already walked`.
        m_surf->replace_pixel(m_level, m_x, m_y, 
                              PIXEL_FILLED | FLAG_WORK);
    }

    void _push_queue(const int sx, const int sy, const int direction) {
        _perimeter_info *info = new _perimeter_info();
        info->sx = sx;
        info->sy = sy;
        info->length = m_step;
        info->direction = direction;
        info->encount = m_encount;
        info->clockwise = is_clockwise();
        g_queue_push_tail(m_queue, info);
    }
    
public:
    CountPerimeterKernel(FlagtileSurface *surf,  
                         const int targ_level,
                         GQueue* queue)
        :   WalkingKernel(surf, targ_level),
            m_queue(queue) 
    {}

    virtual bool start(Flagtile *targ, const int sx, const int sy) {
#ifdef HEAVY_DEBUG
            assert(m_queue != NULL);
#endif
        if (targ != NULL) {
            if (targ->get_stat() & Flagtile::FILLED) {
                _process_only_ridge(targ, sx, sy);
                end(targ);
                return false;
            }
            else if ((targ->get_stat() & Flagtile::EMPTY) 
                        || (targ->get_stat() & Flagtile::FILLED_CONTOUR)
                        || (targ->get_stat() & Flagtile::FILLED_AREA)) {
                // We'll walk INSIDE PIXEL_FILLED area, so filled tile
                // with other pixel value cannot be walked.
                return false;
            }
            return true;
        }
        return false;
    }

    virtual void step(Flagtile *targ, 
                      const int x, const int y,
                      const int sx, const int sy) {
        // Use raw pixel, DO not remove FLAG_DECIDED.
        // (some area previously walked and it might have been marked
        // with FLAG_DECIDED)
        uint8_t pixel = targ->get(m_level, x, y); 

        if (pixel == PIXEL_FILLED) {
            for(int i=0; i<4; i++) {
                uint8_t pix_n = _get_neighbor_pixel(m_level, i, sx, sy);
                m_encount = 0;// IMPORTANT: initialize this here. 
                              // This might be increased in _is_wall_pixel.

                if (_is_wall_pixel(pix_n)) {
                    // Start from current kernel pixel.
                    _walk(sx, sy, _get_reversed_hand_dir(i));
                    _push_queue(sx, sy, i);
                    return;
                }
            }
        }
    }
};

// Clear flag around target pixels.
// This class just utilize WalkingKernel functionality.
class ClearflagWalker: public WalkingKernel
{
protected:
    const uint8_t m_targ_pix;

    virtual bool _is_wall_pixel(const uint8_t pixel) {
        return (pixel & PIXEL_MASK) != m_targ_pix;//(pixel & FLAG_MASK) == 0;
    }

    virtual void _on_new_pixel() {
        // Replace PIXEL_FILLED... i.e. `clear flag`
        uint8_t pix = m_surf->get_pixel(0, m_x, m_y);
        m_surf->replace_pixel(0, m_x, m_y, (pix & PIXEL_MASK));
    }
    
public:
    ClearflagWalker(FlagtileSurface *surf, const uint8_t targ_pix)
        :   WalkingKernel(surf, 0),
            m_targ_pix(targ_pix) {
    }

    void walk_from(const int sx, const int sy, const int direction, const bool outside) {
        if (outside)
            _walk(sx, sy, _get_hand_dir(direction));
        else
            _walk(sx, sy, _get_reversed_hand_dir(direction));
    }
};

// FillHoleWorker
// to fill all small holes inside a large area.
class FillHoleWorker : public FillWorker 
{
protected:
public:
    FillHoleWorker(FlagtileSurface *surf)
        : FillWorker(surf, 0) {
    }
    
    virtual bool match(const uint8_t pix) {
        return (pix & PIXEL_MASK) == PIXEL_AREA;
    }

    virtual void step(Flagtile *tile,
                      const int x, const int y,
                      const int sx, const int sy) {
        tile->replace(0, x, y, PIXEL_FILLED);
    }   
};

/* FillHoleKernel: 
// Searches 'holes' -- counter-clockwise perimeter area 
// and fill them all.
//
// Actual fill operation is done by FillHoleWorker with flood-fill algorithm.
*/
class FillHoleKernel: public CountPerimeterKernel
{
protected:
    virtual bool _is_wall_pixel(const uint8_t pixel) {
        switch(pixel & PIXEL_MASK) {
            case PIXEL_FILLED:
                return true;

            default: // Both of PIXEL_EMPTY, PIXEL_OUTSIDE
                m_encount++;
                // Fallthrough:
            case PIXEL_CONTOUR: 
            case PIXEL_AREA: 
                // PIXEL_AREA and PIXEL_CONTOUR does not increment m_encount.
                return false;
        }
    }

    virtual void _on_new_pixel() {
        // Mark current(new) pixel as `walked`
        m_surf->replace_pixel(0, m_x, m_y, PIXEL_AREA | FLAG_WORK);
    }
    
public:
    /**
    * @fn FillHoleKernel
    * Constructor of FillHoleKernel.
    */    
    FillHoleKernel(FlagtileSurface *surf, GQueue *queue) 
        :   CountPerimeterKernel(surf, 0, queue) 
    {}

    virtual bool start(Flagtile *targ, const int sx, const int sy) {
        // Not baseclass `CountPerimeterKernel::start`,
        // call base-baseclass `KernelWorker::start`.
        // baseclass::start method is not compatible for this class.
        if (KernelWorker::start(targ, sx, sy)) {
            if ((targ->get_stat() & Flagtile::FILLED)) {
                return false;
            }
            return true;
        }
        return false;
    }

    virtual void step(Flagtile *targ, 
                      const int x, const int y,
                      const int sx, const int sy) {
        uint8_t pixel = targ->get(0, x, y); 
        if (pixel == PIXEL_AREA) {
            for(int i=0; i<4; i++) {
                uint8_t kpix = _get_neighbor_pixel(0, i, sx, sy);
                // When kpix is EXACTLY filled pixel... 
                // (A ridge of already detected area would be a combination of
                // PIXEL_FILLED | FLAG_DECIDED, so skip it.)
                if (kpix == PIXEL_FILLED) {
                    m_encount = 0;
                    // We walk into `hole`, so proceed reversed direction.
                    _walk(sx, sy, _get_reversed_hand_dir(i));
                    if (m_step > 0) {
                        // After walking, we might fill current `hole` area.
                        // That is processd later (at FlagtileSurface::fill_holes()) 
                        // just store it into queue for now.
                        _push_queue(sx, sy, i);
                        return;
                    }
                }
            }
        }
    }
};
// Simple drawing line worker
// Only draw pixel into progress level 0.
class DrawLineWorker: public DrawWorker 
{
protected:
    const uint8_t m_flag;
public:
    DrawLineWorker(FlagtileSurface *surf, const uint8_t flag) 
        : DrawWorker(surf, 0), m_flag(flag) 
    {
    }
    
    virtual bool step(const int x, const int y) { 
        m_surf->replace_pixel(0, x, y, m_flag);
        return true;
    }
};

// DecideOutsideWorker 
// for flood-fill operation 
class DecideOutsideWorker : public FillWorker 
{
protected:
public:
    DecideOutsideWorker(FlagtileSurface *surf, const int level) 
        : FillWorker(surf, level) 
    {}

    virtual bool match(const uint8_t pix) {
        return pix == PIXEL_AREA;
    }

    virtual void step(Flagtile *tile,
                      const int x, const int y,
                      const int sx, const int sy) {
        tile->replace(m_level, x, y, PIXEL_OUTSIDE);
    }   
};

// Trigger flood-fill worker.
// Actually this worker itself does no any pixel-writing operation.
// This worker is called from _walk_polygon(and _walk_line) 
// of FlagtileSurface and do flood-fill when condition met.
//
// Caution:
// _walk_polygon/_walk_line method uses progress level 0 coodinate,
// but the this worker might needs another progress level coodinate.
// So we need to convert them in `match` and `step`  method.
class DecideTriggerWorker: public DrawWorker 
{
protected:
    DecideOutsideWorker m_worker;
    
public:
    /**
    * @fn DecideTriggerWorker
    * Constructor of DecideTriggerWorker.
    *
    * @param worker_level: The target progress level. 
    *                      This is only used for m_worker,
    *                      This class itself uses fixed progress level = 0.
    */
    DecideTriggerWorker(FlagtileSurface *surf, 
                        const int worker_level) 
        :  DrawWorker(surf, 0), 
           m_worker(surf, worker_level) 
    {}

    virtual bool step(const int sx, const int sy) {
        // _walk_polygon/_walk_line method uses progress level 0 coodinate,
        // so sx and sy are not target level coordinate, 
        // we need to convert them.
        int worker_level = m_worker.get_target_level();
        int wsx = PROGRESS_PIXEL_COORD(sx, worker_level);
        int wsy = PROGRESS_PIXEL_COORD(sy, worker_level);
        
        if (m_surf->get_pixel(worker_level, wsx, wsy) == PIXEL_AREA) {
            m_surf->flood_fill(wsx, wsy, &m_worker);
            return true;
        }
        return false;
    }
};
#endif
