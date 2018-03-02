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

// This file is for hiding worker class from SWIG/Python interface.

#include "progfill.hpp"

// Base worker/kernel classes defined at progfilldefine.hpp

//-----------------------------------------------------------------------------
//// Kernel classes
// Read notes around KernelWorker base class at progfilldefine.hpp

// Dilation kernel.
class DilateKernel : public KernelWorker
{
protected:
    int m_filled;

public:
    DilateKernel(FlagtileSurface *surf)
        : KernelWorker(surf, 0), m_filled(0) {}

    virtual bool start(Flagtile *targ) 
    {
        m_filled = 0;
        m_processed = 0;
        // We cannot use parent class method for DilateKernel.
        // Because this kernel accepts NULL tile.
        if (targ != NULL) {
            if (targ->get_stat() & Flagtile::FILLED) {
                // There is no space to be dilated.
                return false;
            }
            return true;
        }
        return true; // Accept NULL tile.
    }

    // Utility method: to ensure update some members.
    inline void replace_pixel(Flagtile *targ, 
                              const int x, const int y, 
                              const uint8_t pix)
    {
        targ->replace(0, x, y, pix);
        m_processed++;
        if ((pix & PIXEL_MASK) == PIXEL_FILLED)
            m_filled++;
    }

    inline void replace_pixel(const int sx, const int sy, 
                              const uint8_t pix)
    {
        m_surf->replace_pixel(0, sx, sy, pix);
        m_processed++;
        if ((pix & PIXEL_MASK) == PIXEL_FILLED)
            m_filled++;
    }

    // Utility method: To know whether the tile-local coordinate is
    // inside a tile or not.
    inline bool is_inside_tile(const int level, const int x, const int y) {
        int tile_size = PROGRESS_TILE_SIZE(level);
        // Search only edge of the tile.
        return (x > 0 && x < tile_size-1
                && y > 0 && y < tile_size-1);
    }

    virtual void step(Flagtile *targ, 
                      const int x, const int y,
                      const int sx, const int sy)
    {
        if (targ == NULL) {
            // For completely empty tile,
            // This kernel should search only edge of the tile
            if (is_inside_tile(0, x, y)) {
                return;
            }
        }
        else {
            if ((targ->get(0, x, y) & PIXEL_MASK) == PIXEL_FILLED) { 
                m_filled++;
                return;
            }
        }

        // Code reaches here when current pixel is not PIXEL_FILLED.
        for(int i=0; i<4; i++) {
            uint8_t kpix = get_kernel_pixel(0, sx, sy, i);
            if ((kpix & PIXEL_MASK) == PIXEL_FILLED && (kpix & FLAG_WORK) == 0) {
                if (targ == NULL) {
                    // Generating a new tile.
                    int tile_size = PROGRESS_TILE_SIZE(0);
                    targ = m_surf->get_tile(
                        sx / tile_size, 
                        sy / tile_size, 
                        true
                    );
                }
                replace_pixel(targ, x, y, PIXEL_FILLED | FLAG_WORK);
                return;
            }
        }
    }
    
    virtual void end(Flagtile *targ) 
    {
        if (targ != NULL) {
            KernelWorker::end(targ);
#ifdef HEAVY_DEBUG
            assert(m_filled <= PROGRESS_BUF_SIZE(0));
#endif
            if (m_filled == PROGRESS_BUF_SIZE(0)) {
                targ->set_stat(Flagtile::FILLED);
            }
            else {
                if (m_filled > 0) 
                    targ->set_stat(Flagtile::HAS_PIXEL);
                else
                    targ->unset_stat(Flagtile::HAS_PIXEL);
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
    int m_areacnt;

public:
    ErodeContourKernel(FlagtileSurface *surf)
        : DilateKernel(surf) {}

    // Different from DilateKernel, erosion might occur
    // even completely filled tile.
    // Outer-rim of a filled tile might be neighboring vacant pixel.
    virtual bool start(Flagtile *targ) 
    {
        m_areacnt = 0;
        if (KernelWorker::start(targ)) {
            if ((targ->get_stat() & Flagtile::FILLED) == 0
                    && (targ->get_stat() & Flagtile::FILLED_AREA) == 0) {
                return true;
            }
        }
        return false;
    }

    virtual void step(Flagtile *targ, 
                      const int x, const int y,
                      const int sx, const int sy)
    {
        uint8_t pix = targ->get(0, x, y);
        if ((pix & PIXEL_MASK) == PIXEL_CONTOUR) {
            for(int i=0; i<4; i++) {
                uint8_t kpix = get_kernel_pixel(0, sx, sy, i);
                // Chech whether kpix is exactly PIXEL_AREA
                // (no working flag set)
                // To avoid chain reaction.
                if (kpix == PIXEL_AREA) {
                    replace_pixel(targ, x, y, PIXEL_AREA | FLAG_WORK);
                    m_areacnt++;
                    return; // Exit without adding filled count.
                }
            }
        }
        else if ((pix & PIXEL_MASK) == PIXEL_AREA) {
            m_areacnt++;
        }
    }

    virtual void end(Flagtile *targ) {
        // This kernel does not see PIXEL_FILLED count
        // Because this is used from LassofillSurface.
        // That surface should not have no FILLED pixel
        // when using this kernel.
        if (m_areacnt == PROGRESS_BUF_SIZE(0)) {
            targ->set_stat(Flagtile::FILLED_AREA);
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
    int m_filled;

public:
    ConvertKernel(FlagtileSurface *surf, const int level, 
                  const uint8_t targ_pixel, const uint8_t new_pixel)
        : KernelWorker(surf, level), 
          m_targ_pixel(targ_pixel),
          m_new_pixel(new_pixel),
          m_filled(0) {}

    // Another contsructor. Used for class member.
    ConvertKernel(FlagtileSurface *surf) 
        : KernelWorker(surf, 0), 
          m_targ_pixel(0),
          m_new_pixel(0),
          m_filled(0) {} 

    // To avoid repeatedly generate convert-kernel object,
    // Some class reuse only one kernel object over a series of processes.
    // For such case, we need this (re)setup method.
    void setup(const int level, 
               const uint8_t targ_pixel, 
               const uint8_t new_pixel)
    {
        set_target_level(level);
        m_targ_pixel = targ_pixel;
        m_new_pixel = new_pixel;
        m_filled = 0;
    }

    virtual bool start(Flagtile *targ) 
    {
        if (KernelWorker::start(targ)) {
            if (m_targ_pixel == PIXEL_AREA
                    && (targ->get_stat() & Flagtile::FILLED_AREA) != 0) {
                // At here, completely area tile should be filled up!
                // statflag would be adjusted in fill method.
                targ->fill(m_new_pixel);
                return false;
            }
            m_filled = 0;
            return true;
        }
        return false;
    }

    virtual void step(Flagtile *targ, 
                      const int x, const int y,
                      const int sx, const int sy)
    {
        uint8_t pix = targ->get(m_level, x, y);
        if (pix == m_targ_pixel) {
            targ->replace(m_level, x, y, m_new_pixel);
            pix = m_new_pixel;
            m_processed++;
        }

        if ((pix & PIXEL_MASK) == PIXEL_FILLED)
            m_filled++;
    }

    virtual void end(Flagtile *targ) 
    {
        KernelWorker::end(targ);
#ifdef HEAVY_DEBUG
        assert(m_filled <= PROGRESS_BUF_SIZE(m_level));
#endif
        if (m_level == 0) {
            if (m_filled == PROGRESS_BUF_SIZE(m_level)) {
                targ->set_stat(Flagtile::FILLED);
            }
            else {
                targ->unset_stat(Flagtile::FILLED);
            }

        }

        // If the tile has at least one valid pixel,
        // mark it to stat flag.
        // This flag should be referred from ProgressKernel.
        if (m_filled > 0) {
            targ->set_stat(Flagtile::HAS_PIXEL);
        }
        else {
            targ->unset_stat(Flagtile::HAS_PIXEL);
        }
    }
};

// Progressive fill Kernel.
// This Kernelworker is very important, to generate
// filled pixel area with virtually gap-closing.
class ProgressKernel: public KernelWorker
{
protected:
    int m_filled;
    
    inline uint8_t _get_neighbor_pixel(const int direction, 
                                       const int sx, const int sy) 
    {
        return m_surf->get_pixel(
            m_level, 
            sx+xoffset[direction], 
            sy+yoffset[direction]
        ) & PIXEL_MASK;
    }

    // Process only 4 corner pixel for FILLED_AREA tile.
    void _step_corner(Flagtile *targ, 
                      const int x, const int y,
                      const int sx, const int sy)
    {
        int corner = PROGRESS_TILE_SIZE(m_level) - 1;
        int bx = x << 1;
        int by = y << 1;
        int direction;
                        
        if (x == 0 && y == 0) {
            direction = 3;// left,top pixel
            /*
            uint8_t top = m_surf->get_pixel(m_level, sx, sy-1) & PIXEL_MASK;
            uint8_t left = m_surf->get_pixel(m_level, sx-1, sy) & PIXEL_MASK;
            
            if (top >= PIXEL_FILLED && left >= PIXEL_FILLED) {
                targ->put(beneath_level, bx, by, PIXEL_FILLED);
                m_filled++;
            }
            */
        }
        else if (x == corner && y == 0) {
            direction = 0;// top, right pixel
            bx++;
            /*
            uint8_t top = m_surf->get_pixel(m_level, sx, sy-1) & PIXEL_MASK;
            uint8_t right = m_surf->get_pixel(m_level, sx+1, sy) & PIXEL_MASK;            
            
            if (top >= PIXEL_FILLED && right >= PIXEL_FILLED) {
                targ->put(beneath_level, bx+1, by, PIXEL_FILLED);
                m_filled++;
            }
            */
        }
        else if (x == corner && y == corner) {
            direction = 1; // right, bottom pixel
            bx++;
            by++;
            /*
            uint8_t bottom = m_surf->get_pixel(m_level, sx, sy+1) & PIXEL_MASK;            
            uint8_t right = m_surf->get_pixel(m_level, sx+1, sy) & PIXEL_MASK;                 
            
            if (bottom >= PIXEL_FILLED && right >= PIXEL_FILLED) {
                targ->put(beneath_level, bx+1, by+1, PIXEL_FILLED);
                m_filled++;
            } 
            */           
        }
        else if (x == 0 && y == corner) {
            direction = 2; // bottom, left pixel
            by++;
            /*
            uint8_t bottom = m_surf->get_pixel(m_level, sx, sy+1) & PIXEL_MASK;           
            uint8_t left = m_surf->get_pixel(m_level, sx-1, sy) & PIXEL_MASK;
            
            if (bottom >= PIXEL_FILLED && left >= PIXEL_FILLED) {
                targ->put(beneath_level, bx, by+1, PIXEL_FILLED);
                m_filled++;
            } 
            */           
        }
        else {
            return;
        }
            
        uint8_t pix1 = _get_neighbor_pixel(direction, sx, sy);
        uint8_t pix2 = _get_neighbor_pixel((direction+1)&3, sx, sy);
        
        if (pix1 >= PIXEL_FILLED && pix2 >= PIXEL_FILLED) {
            int beneath_level = m_level - 1;
            targ->put(beneath_level, bx, by, PIXEL_FILLED);
            m_filled++;
        } 
    }
    
public:
    // This class does not use initial level parameter.
    // Because it would be changed frequently.
    // It would be set at loop, with `set_target_level` method.
    ProgressKernel(FlagtileSurface *surf) 
        : KernelWorker(surf, 0) {}

    // Disable default finalize. 
    // This kernel does not use FLAG_WORK.
    virtual void finalize() {} 
    
    virtual bool start(Flagtile *targ) 
    {
        m_filled = 0;
        if (KernelWorker::start(targ)) {
            // Target tile 
            int stat = targ->get_stat();
            if ((stat & Flagtile::FILLED) || (stat & Flagtile::EMPTY)) {
                return false;
            }
            return true;
        }
        return false;
    }

    virtual void step(Flagtile *targ, 
                      const int x, const int y,
                      const int sx, const int sy)
    {
        if (targ->get_stat() & Flagtile::FILLED_AREA) {
            _step_corner(targ, x, y, sx, sy);
        }
        else {
            // `pixel is greater than PIXEL_FILLED` 
            // means `PIXEL_FILLED or PIXEL_CONTOUR`
            if ((targ->get(m_level, x, y) & PIXEL_MASK) >= PIXEL_FILLED) {
                uint8_t top = _get_neighbor_pixel(0, sx, sy);
                uint8_t right = _get_neighbor_pixel(1, sx, sy);
                uint8_t bottom = _get_neighbor_pixel(2, sx, sy);
                uint8_t left = _get_neighbor_pixel(3, sx, sy);

                // The param x and y is current(above) progress coordinate.
                // The coordinate of level beneath should be double of them.
                int beneath_level = m_level - 1;
                int bx = x << 1;
                int by = y << 1;
             
                // Deciding the left-top pixel of beneath progress level.
                uint8_t pix = targ->get(beneath_level, bx, by);
                if (pix == PIXEL_AREA && (top >= PIXEL_FILLED && left >= PIXEL_FILLED)) {
                    targ->put(beneath_level, bx, by, PIXEL_FILLED);
                    m_filled++;
                }

                // The right-top pixel
                int cx = bx + 1;
                pix = targ->get(beneath_level, cx, by);
                if (pix == PIXEL_AREA && (top >= PIXEL_FILLED && right >= PIXEL_FILLED)) {
                    targ->put(beneath_level, cx, by, PIXEL_FILLED);
                    m_filled++;
                }
                
                // The right-bottom pixel
                int cy = by + 1;                
                pix = targ->get(beneath_level, cx, cy);
                if (pix == PIXEL_AREA && (bottom >= PIXEL_FILLED && right >= PIXEL_FILLED)) {
                    targ->put(beneath_level, cx, cy, PIXEL_FILLED);
                    m_filled++;
                }
                
                // The left-bottom pixel
                pix = targ->get(beneath_level, bx, cy);
                if (pix == PIXEL_AREA && (bottom >= PIXEL_FILLED && left >= PIXEL_FILLED)) {
                    targ->put(beneath_level, bx, cy, PIXEL_FILLED);
                    m_filled++;
                }               
            }
        }
    }   

    virtual void end(Flagtile *targ) 
    {
        KernelWorker::end(targ);
        int c_level = m_level - 1;
        if (c_level == 0) {
#ifdef HEAVY_DEBUG
            assert(m_filled <= PROGRESS_BUF_SIZE(c_level));
#endif
            if (targ->get_stat() & Flagtile::FILLED_AREA) {
                // FILLED_AREA tile might be affected from
                // neiboring tiles.
                if (m_filled > 0)
                    targ->unset_stat(Flagtile::FILLED_AREA);
            }
            else {
                if (m_filled == PROGRESS_BUF_SIZE(c_level)) {
                    targ->set_stat(Flagtile::FILLED);
                }
                else {
                    targ->unset_stat(Flagtile::FILLED);
                }
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
    
    bool m_dbg_exit;
    // Dedicated wrapper method to get pixel with direction.
    virtual uint8_t _get_pixel_with_direction(const int x, const int y, 
                                              const int direction) 
    {
        uint8_t pix = m_surf->get_pixel(
            m_level,
            x + xoffset[direction],
            y + yoffset[direction]
        );

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
        int x, int y, const bool is_start, const bool face_wall) 
    {
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

    void _draw_antialias(const int ex, const int ey, const bool right) 
    {
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
    virtual void _on_rotate_cb(const bool right) 
    {
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

public:
    AntialiasKernel(FlagtileSurface *surf) 
        : WalkingKernel(surf, 0) {
        m_dbg_exit=false;}

    // Disable finalize method.
    virtual void finalize() {}

    virtual bool start(Flagtile *targ) 
    {
        if (WalkingKernel::start(targ)) {
            return true;
        }
        return false;
    }
    
    // Caution: You should call this method with
    // `progress level` coordinate.
    virtual void step(Flagtile *targ, 
                      const int x, const int y,
                      const int sx, const int sy)
    {
        uint8_t pix;
        uint8_t pix_right;

        if (targ == NULL) {  
                  //|| (targ->get_stat() & Flagtile::HAS_PIXEL) == 0) {

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
        
        if ((pix & FLAG_AA) == 0 
                && (pix & PIXEL_MASK) != PIXEL_FILLED
                && (pix_right & (PIXEL_MASK | FLAG_AA)) == PIXEL_FILLED) { 
            // This class searches target pixel from left to right for each line.
            // And we use right-handed rule to walk area ridge.
            // Therefore, we would face top at the start point always.

            // IMPORTANT: initialize class unique members here!
            m_walking_started = false;

            _walk(sx, sy, 0);
            m_processed++;
        }
    }
};

// FillHoleWorker
// to fill all small holes inside a large area.
class FillHoleWorker : public FillWorker 
{
protected:
public:
    FillHoleWorker(FlagtileSurface *surf, const int level) 
        : FillWorker(surf, level) {}

    virtual uint8_t get_fill_pixel() {
        return PIXEL_FILLED;
    } 
        
    virtual bool start(Flagtile *tile) {
        m_processed = 0;
        if (tile->get_stat() & Flagtile::EMPTY)
            return false;
        return true;
    }

    virtual bool match(const uint8_t pix) 
    {
        return ((pix & PIXEL_MASK) != PIXEL_FILLED && pix != 0);
    }

    virtual void step(Flagtile *tile,
                      const int x, const int y,
                      const int sx, const int sy)
    {
        tile->replace(
            m_level,
            x, y,
            PIXEL_FILLED
        );
        m_processed++;
    }   
    
    virtual void end(Flagtile *tile) 
    {
        // Update tile status.
        if(m_processed > 0) {
            tile->unset_stat(Flagtile::FILLED_AREA);
        }
    }
};

// Search entire Flagtilesurface and 
// count perimeter of PIXEL_FILLED area.
// If perimeter of that area is smaller than threshold,
// mark that area to be removed, 
// and that areas are removed later, with RemoveGarbageKernel.
// If threshold is Zero, Area removal is not done. 
// This used from LassofillSurface, to do `fill all holes`.
class CountPerimeterKernel: public WalkingKernel
{
protected:
    const int m_threshold;
    uint8_t m_replacing_pixel;
    FillHoleWorker m_fillworker;
    const bool m_fill_all_holes;

    inline int _get_left_dir(const int dir) { return (dir + 3) & 3; }

    // Utilize _is_match_pixel for another purpose.
    // At 1st pass, this kernel would set FLAG_DECIDED flag to current wall pixel.
    // With this, we can distinguish the filled pixel is already 
    // examined or not.
    // And 2nd pass(small area removal), this kernel replace wall pixel
    // as PIXEL_REMOVE, to mark that area should be removed.
    virtual bool _is_match_pixel(const uint8_t pixel) 
    {
        switch(pixel & PIXEL_MASK) {
            case PIXEL_FILLED:
                {
                    int direction = _get_hand_dir(m_cur_dir);
                    m_surf->replace_pixel(
                        m_level,
                        m_x + xoffset[direction],
                        m_y + yoffset[direction],
                        m_replacing_pixel
                    );
                }
                return true;

            case PIXEL_REMOVE:
                return true;

            default:
                return false;
        }
    }

    virtual void _on_rotate_cb(const bool right) 
    {
        if (right)
            m_step--;
    }
    
    // CountPerimeterKernel would target neighboring(or surrounded by) PIXEL_AREA 
    // and vacant and PIXEL_CONTOUR pixel. 
    virtual bool _is_target_pixel(const uint8_t pixel) {
        return  (pixel == PIXEL_AREA 
                    || pixel == 0
                    || pixel == PIXEL_CONTOUR);
    }

public:
    /**
    * @fn CountPerimeterKernel
    * Constructor of CountPerimeterKernel.
    *
    * @param threshold: The threshold of perimeter. 
    *                   If an area has smaller perimeter than this
    *                   threshold, it would be removed.
    *                   If this is 0, area removal is disabled.
    * @param fill_all_holes: When this flag is true, and if there is a hole
    *                        (i.e. counter-clockwize area), it will be filled
    *                        with flood-fill algorithm.
    * 
    * This Kernel detects a PIXEL_FILLED pixel which is neibored with
    * PIXEL_AREA or PIXEL_EMPTY (or PIXEL_CONTOUR, when finalizing flag is 
    * true).
    * If such pixel detected, kernel walks around that pixel area and
    * counts its perimeter. When that area has smaller perimeter than
    * threshold, it would be removed.
    *
    * Also, this kernel detects a hole inside filled pixel area.
    * If fill_all_holes flag is true, this kernel fill up it by flood-fill.
    *
    * Note: fill_all_holes does not consider how large the perimeter of hole.
    * It would be filled up even it is extreamly large one. 
    * To avoid this, turn off the fill_all_holes option from OptionsPresenter.
    */    
    CountPerimeterKernel(FlagtileSurface *surf, 
                         const int targ_level,
                         const int threshold,
                         const bool fill_all_holes) 
        :   WalkingKernel(surf, targ_level), 
            m_threshold(threshold),
            m_fillworker(surf, targ_level),
            m_fill_all_holes(fill_all_holes)  
    {}

    virtual bool start(Flagtile *targ) 
    {
        if (KernelWorker::start(targ)) {
            if ((targ->get_stat() & Flagtile::FILLED) != 0 ) {
                   // || (targ->get_stat() & Flagtile::HAS_PIXEL) == 0) {
                   // TODO HAS_PIXEL stat flag does not work.
                return false;
            }
            return true;
        }
        return false;
    }

    virtual void step(Flagtile *targ, 
                      const int x, const int y,
                      const int sx, const int sy)
    {
        uint8_t pixel = targ->get(m_level, x, y) & PIXEL_MASK;
        if (pixel == PIXEL_AREA 
                || pixel == 0) {
            for(int i=0; i<4; i++) {
                uint8_t kpix = get_kernel_pixel(m_level, sx, sy, i);
                
                // When kpix is EXACTLY filled pixel... 
                // (A ridge of already detected area would be a combination of
                // PIXEL_FILLED | FLAG_DECIDED, so skip it.)
                if (kpix == PIXEL_FILLED) {
                    // Start from current kernel pixel.
                    // We'll proceed to `right` of that pixel,
                    // so _get_hand_dir(i) is the initial direction.
                    m_replacing_pixel = PIXEL_FILLED | FLAG_DECIDED;
                    _walk(sx, sy, _get_left_dir(i));
                    if (m_step > 0) {
                        // After walking, we might remove current pixel area.
                        // But if it is counter-clockwise, it would be a hole
                        // in large pixel area. so ignore it.
                        if (is_clockwise()) {
                            if (m_step < m_threshold) {
                                // There might `diagonally connected areas`.
                                // Such areas cannot be removed with our flood-fill.
                                // Therefore,walk the current area again and 
                                // replace PIXEL_REMOVE to rim pixels.
                                // Actual removal of the area should be 
                                // executed later by another worker(RemoveAreaWorker).
                                m_replacing_pixel = PIXEL_REMOVE;
                                _walk(sx, sy, _get_left_dir(i));
                            }
                        }
                        else if (m_fill_all_holes) {
                            m_surf->flood_fill(sx, sy, &m_fillworker);
                        }
                        return;
                    }
                }
            }
        }
    }
};

// Removing Small filled area, which would be surrounded(or filled) with
// PIXEL_REMOVE. 
// This class is used in RemoveGarbageKernel.
class RemoveAreaWorker : public FillWorker 
{
protected:
public:
    RemoveAreaWorker(FlagtileSurface *surf,
                     const int targ_level) 
        : FillWorker(surf, targ_level) {
    }
    
    virtual uint8_t get_fill_pixel() {
        return PIXEL_AREA;
    } 
    
    virtual bool start(Flagtile *tile) {
        if (tile->get_stat() & Flagtile::FILLED_AREA)
            return false;
        return true;
    }

    virtual bool match(const uint8_t pix)
    {
        return (((pix & PIXEL_MASK)== PIXEL_FILLED) || ((pix & PIXEL_MASK) == PIXEL_REMOVE));
    }

    virtual void step(Flagtile *tile,
                      const int x, const int y,
                      const int sx, const int sy)
    {
        tile->replace(
            m_level,
            x, y,
            PIXEL_AREA
        );
    }
    
    virtual void end(Flagtile *tile) {
    }
};

// For post processing of CountPerimeterKernel.
// Some pixel areas would be connected `diagonally`, and flood-fill algorithm
// cannot fill up such area (treat as another divided area), 
// but walking algorithm detect such areas as connected one.
//
// To deal with this problem, CountPerimeterKernel marks areas to be removed
// with PIXEL_REMOVE. and This kernel seaches that pixel 
// and remove it (fill it with PIXEL_AREA) all.
class RemoveGarbageKernel: public KernelWorker
{
protected:
    RemoveAreaWorker m_remove_worker;

public:
    RemoveGarbageKernel(FlagtileSurface *surf,
                        const int targ_level)
        : KernelWorker(surf, targ_level), 
          m_remove_worker(surf, targ_level) 
    {}

    virtual bool start(Flagtile *targ) 
    {
        // We cannot use parent class method for DilateKernel.
        // Because this kernel accepts NULL tile.
        if (targ != NULL) {
            return (targ->get_stat() & Flagtile::FILLED) == 0;
                   // && (targ->get_stat() & Flagtile::HAS_PIXEL) != 0);
                   // TODO HAS_PIXEL DOES NOT WORK.
        }
        return false; 
    }

    virtual void step(Flagtile *targ, 
                      const int x, const int y,
                      const int sx, const int sy)
    {
        uint8_t pix = targ->get(m_level, x, y);
        if ((pix & PIXEL_MASK) == PIXEL_REMOVE) { 
            // Processing kernel.
            for(int i=0; i<4; i++) {
                uint8_t kpix = get_kernel_pixel(m_level, sx, sy, i) & PIXEL_MASK;
                if (kpix == PIXEL_FILLED || kpix == PIXEL_REMOVE){
                    m_surf->flood_fill(
                        sx, sy,
                        (FillWorker*)&m_remove_worker
                    );
                    return;
                }
            }
            // If code reaches here,
            // It means the pixel is isolated 1x1 pixel.
            targ->replace(m_level, x, y, PIXEL_AREA);
        }
    }
    
    // Disable these methods.
    virtual void end(Flagtile *targ) {} 
    virtual void finalize() {} 
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
    
    virtual bool step(const int x, const int y) 
    { 
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

    virtual uint8_t get_fill_pixel() {
        return PIXEL_OUTSIDE;
    } 
    
    virtual bool start(Flagtile *tile) {
        m_processed = 0;
        if (tile->get_stat() & Flagtile::EMPTY)
            return false;
        return true;
    }

    virtual bool match(const uint8_t pix) 
    {
        return pix == PIXEL_AREA;
    }

    virtual void step(Flagtile *tile,
                      const int x, const int y,
                      const int sx, const int sy)
    {
        tile->replace(
            m_level,
            x, y,
            PIXEL_OUTSIDE
        );
        m_processed++;
    }   
    
    virtual void end(Flagtile *tile) 
    {
        // Update tile status.
        if(m_processed > 0) {
            tile->unset_stat(Flagtile::FILLED_AREA);
        }
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

    virtual bool step(const int sx, const int sy) 
    {
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
