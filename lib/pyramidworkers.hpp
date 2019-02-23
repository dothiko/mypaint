/* This file is part of MyPaint.
 * Copyright (C) 2017 by dothiko<dothiko@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#ifndef PYRAMIDWORKERS_HPP
#define PYRAMIDWORKERS_HPP

// This file is to avoid polluting mypaintlib namespace, because these
// worker classes cannot be used from python.

#include <glib.h>
#include "pyramidfill.hpp"

// Base worker/kernel classes defined at pyramiddefine.hpp

//-----------------------------------------------------------------------------
//// Kernel classes
// Read notes around KernelWorker base class at pyramiddefine.hpp

// Dilation kernel.
class DilateKernel : public KernelWorker
{
protected:
    const uint8_t m_targ_pix;
    int m_processed;

public:
    DilateKernel(FlagtileSurface *surf, const uint8_t targ_pixel)
        : KernelWorker(surf),
          m_targ_pix(targ_pixel),
          m_processed(0) { }

    virtual bool start(Flagtile * const targ, const int sx, const int sy)
    {
#ifdef HEAVY_DEBUG
        assert(m_level == 0);// This kernel only work against level 0.
#endif
        if (targ->is_filled_with(m_level, m_targ_pix)) {
            // There is no room to be dilated.
            return false;
        }

        // At first, process only ridge.
        // With this, we can process 
        process_only_ridge(targ, sx, sy);

        if (targ->is_filled_with(m_level, PIXEL_INVALID)) {
            end(targ);
            return false;
        }

        if (targ->get_pixel_count(m_level, m_targ_pix) > 0) {
            return true;
        }
        return false;
    }

    virtual void step(Flagtile * const targ,
                      const int x, const int y,
                      const int sx, const int sy)
    {
        uint8_t pix = targ->get(0, x, y) & PIXEL_MASK;
        if (pix != m_targ_pix) {
            for(int i=0; i<4; i++) {
                uint8_t kpix;
                if (in_border(x, y))
                    kpix = get_pixel_with_direction(sx, sy, i);
                else 
                    kpix = targ->get(0, x+xoffset[i], y+yoffset[i]);

                if (kpix == m_targ_pix) {
                    m_processed++;
                    targ->replace(0, x, y, m_targ_pix | FLAG_WORK);
                    return;
                }
            }
        }
    }

    virtual void end(Flagtile * const targ)
    {
        KernelWorker::end(targ);
        if (m_processed > 0) {
            targ->clear_bitwise_flag(0, FLAG_WORK);
            m_processed = 0;
        }
    }
};

// Pixel converter kernel
class ConvertKernel : public KernelWorker
{
protected:
    uint8_t m_targ_pix;
    uint8_t m_new_pixel;

public:
    ConvertKernel(FlagtileSurface *surf)
        : KernelWorker(surf),
          m_targ_pix(0),
          m_new_pixel(0) { }

    void set_target_pixel(const uint8_t targ_pixel, const uint8_t new_pixel)
    {
        m_targ_pix = targ_pixel;
        m_new_pixel = new_pixel;
    }

    virtual bool start(Flagtile * const targ, const int sx, const int sy)
    {
        if (targ->is_filled_with(m_level, m_targ_pix)) {
            targ->fill(m_new_pixel);
            return false;
        }
        return true;
    }

    virtual void step(Flagtile * const targ,
                      const int x, const int y,
                      const int sx, const int sy)
    {
        uint8_t pix = targ->get(m_level, x, y);
        if (pix == m_targ_pix) {
            targ->replace(m_level, x, y, m_new_pixel);
            pix = m_new_pixel;
        }
    }
};

// Propagate 'decided' pixels downward.
// It seems that this can be integrated at Flagtile class,
// but `propagating downward` needs to refer `surface-global` neighbor tile pixels.
// It cannot be done at Flagtile.
// But this worker just refer global pixels, never write to them.
// Also this never target empty tile.
// So this worker is parallelizable, can use at FlagtileSurface::fillter_tiles_mp.
class PropagateKernel: public KernelWorker
{
protected:
    const bool m_outside_expandable;// Flag to PIXEL_OUTSIDE areas can expandable into PIXEL_AREA
                                    // This should be true for ClosedAreaFill
                                    // But to be false for CutProtrudingPixels.
                                    // CutProtrudingPixels needs to keep the shape of PIXEL_AREA.
public:
    // This class does not use initial level parameter.
    // Because it would be changed frequently.
    // It would be set at loop, with `set_target_level` method.
    PropagateKernel(FlagtileSurface *surf, const bool outside_expandable=true)
        : KernelWorker(surf), m_outside_expandable(outside_expandable) { }

    virtual bool start(Flagtile * const targ, const int sx, const int sy)
    {
        assert(targ != NULL);
        // Target tile pixel check.
        // If target-tile pixel is already `decided`,
        // There is nothing to do for that tile.
        // Or,  completely undecided (filled with `PIXEL_AREA`)
        // through level 0, that tile can be decided easily.
        if(targ->is_filled_with(0, PIXEL_FILLED)
                || targ->is_filled_with(0, PIXEL_INVALID)) {
            return false;
        }
        else if(targ->is_filled_with(0, PIXEL_AREA)) {
            uint8_t above = targ->get(m_level, 0, 0);
            // is_filled_with(0, PIXEL_AREA) means
            // `All pixels in this tile are uniformly same,
            // and not yet decided, at level-0`
            // But, it is only at level-0.
            // In higher(maximum) pyramid level, pixels would be
            // already decided, when this worker method is called.
            //
            // In such case, we might easily filled up this tile.
            //
            switch(above) {
                case PIXEL_OUTSIDE:
                case PIXEL_FILLED:
                    targ->fill(above);
                    break;
#ifdef HEAVY_DEBUG
                case PIXEL_CONTOUR:
                    // Fill undecided area with PIXEL_CONTOUR
                    // must not happen.
                    assert(false);
                    break;
#endif
                default:
                    // No any pixel deciding flood-fill operation reached.
                    // So, just process ridge.
                    // This might change filled state of tile
                    // because PIXEL_OUTSIDE of neighbor tile might
                    // erode PIXEL_AREA pixels.
                    process_only_ridge(targ, sx, sy);
                    break;
            }
            end(targ);
            return false;
        }
        return true;
    }

    virtual void step(Flagtile * const targ,
                      const int x, const int y,
                      const int sx, const int sy)
    {
        uint8_t above = targ->get(m_level, x, y);
        int bx = x << 1;
        int by = y << 1;
        int beneath_level = m_level - 1;

        switch(above) {
            case PIXEL_OUTSIDE:
            case PIXEL_FILLED:
            case PIXEL_EMPTY:

                // Inherit above pixel, Because they cannot be generated
                // `build pyramid seed` stage.
                // When there is PIXEL_CONTOUR or PIXEL_AREA, it indicates
                // there MIGHT be some of them at level-0 pixel.
                // But there is either of OUTSIDE,FILLED, or EMPTY,
                // It means it is sure that there is only such pixels over
                // level-0.
                for(int py=by; py < by+2; py++) {
                    for(int px=bx; px < bx+2; px++) {
                        targ->replace(beneath_level, px, py, above);
                    }
                }
                break;

            default:
                // Other `undecided` pixels.
                {
                    uint8_t top = get_pixel_with_direction(sx, sy, 0);
                    uint8_t right = get_pixel_with_direction(sx, sy, 1);
                    uint8_t bottom = get_pixel_with_direction(sx, sy, 2);
                    uint8_t left = get_pixel_with_direction(sx, sy, 3);

                    // The param x and y is current(above) pyramid coordinate.
                    // The coordinate of level beneath should be double of them.
                    uint8_t pix_v, pix_h;

                    pix_v = top;
                    for(int py=by; py < by+2; py++) {
                        pix_h = left;
                        for(int px=bx; px < bx+2; px++) {
                            uint8_t pix = targ->get(beneath_level, px, py) & PIXEL_MASK;
                            if (pix == PIXEL_AREA || pix == PIXEL_RESERVE) {
                                if (pix_h <= PIXEL_OUTSIDE || pix_v <= PIXEL_OUTSIDE) {
                                    // For `Cut protruding` feature,
                                    // m_outside_expandable is false.
                                    if (m_outside_expandable)
                                        targ->replace(beneath_level, px, py, PIXEL_OUTSIDE);
                                }
                                else if (pix == PIXEL_RESERVE) {
                                    // This pixel is temporary placeholder which is used for
                                    // pyramid_flood_fill.
                                    // So there is no OUTSIDE pixels around, replace this with
                                    // PIXEL_AREA.
                                    targ->replace(beneath_level, px, py, PIXEL_AREA);
                                }
                                /*
                                else if (pix_h == PIXEL_FILLED || pix_v == PIXEL_FILLED) {
                                    targ->replace(beneath_level, px, py, PIXEL_FILLED);
                                }
                                */
                            }
                            pix_h = right;
                        }
                        pix_v = bottom;
                    }
                }
                break;
        }
    }
};


// Antialias Walker.
// This class does (psuedo)antialias by walking around the filled area ridge
// and draw gradient lines around there.
//
// NOTE: In this walker, m_step will be consumed each time AA-line drawn
// (i.e. rotation occur).
class AntialiasWalker: public WalkingKernel
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
    virtual uint8_t get_pixel_with_direction(const int x, const int y,
                                             const int direction)
    {
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

    // Get anti-aliasing value (0.0 or 1.0) of start/endpoint.
    // This method is called when aa-line is drawn,
    // i.e. at the startpoint and endpoint of aa-line.
    double get_aa_value(
        int x, int y, const bool is_start, const bool face_wall)
    {
        // Adjust position, according to situation,
        // to know whether the point neighbored some wall pixel.
        if (is_start) {
            // For staring pixel check, we need the `back` pixel
            // to determine starting point value.
            int back_dir = (m_line_dir + 2) & 3;
            x += xoffset[back_dir];
            y += yoffset[back_dir];
        }
        else if (face_wall) {
            // Currently kernel is face into wall.
            // This means
            // `Kernel will turns right after AA line drawn.`
            x += xoffset[m_line_dir];
            y += yoffset[m_line_dir];
        }
        else {
            // Otherwise, fade in or out.
            return 0;
        }

        if ((m_surf->get_pixel(0, x, y) & PIXEL_MASK) == PIXEL_FILLED) {
            int rightdir = (m_line_dir + 1) & 3;
            uint8_t side_pixel = get_pixel_with_direction(x, y, rightdir);
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

    void draw_antialias(const int ex, const int ey, const bool right)
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
            double start_value = get_aa_value(m_sx, m_sy, true, right);
            double end_value = get_aa_value(ex, ey, false, right);

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

            if (right)
                pixel_step++;

            // Set initial value.
            // Anti-aliasing pixel never starts from extreme(0.0 or 1.0) value.
            // Because, the outside pixels of AA gradient has already extreme value.
            value = start_value + value_step;

            // Drawing seed value of anti-alias gradient.
            // (This is `seed value`, not actual gradient value.)
            // That seed value would be used as a factor of
            // calculating actual alpha value, at when converting
            // Flagtile into mypaint color tile.
            for(int i=0; i < pixel_step; i++) {
                uint8_t pix = (uint8_t)(value * MAX_AA_LEVEL) | FLAG_AA;
                m_surf->replace_pixel(0, cx, cy, pix);
                value += value_step;
                if (value < 0) {
                    value = 0;
                    // Invert sign (at intermidiate point).
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
    virtual void on_rotate_cb(const bool right)
    {
        if (m_walking_started ) {
            if (m_step > 0) {
               draw_antialias(m_x, m_y, right);
            }
        }
        else {
            // Adjust origin point
            m_ox = m_x;
            m_oy = m_y;
            m_walking_started = true;
        }

        if (!right) {
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

    virtual bool is_wall_pixel(const uint8_t pixel)
    {
        return ((pixel & FLAG_AA) == 0 && (pixel & PIXEL_MASK) == PIXEL_FILLED);
    }

    virtual void on_new_pixel()
    {
        // This needed to avoid misdetecting area which is already Anti-Aliased.
        m_surf->replace_pixel(0, m_x, m_y, FLAG_AA);
    }

public:
    AntialiasWalker(FlagtileSurface *surf)
        : WalkingKernel(surf) { }

    virtual bool start(Flagtile * const targ, const int sx, const int sy)
    {
#ifdef HEAVY_DEBUG
        assert(m_level == 0);
#endif
        if (targ->is_filled_with(m_level, PIXEL_FILLED)
                || targ->is_filled_with(m_level, PIXEL_CONTOUR)) {
            return false;
        }
        else if (targ->is_filled_with(m_level, PIXEL_AREA)
                || targ->is_filled_with(m_level, PIXEL_INVALID)) {
            process_only_ridge(targ, sx, sy);
            end(targ);
            return false;
        }
        return true;
    }

    // Caution: You should call this method with
    // `pyramid level` coordinate.
    virtual void step(Flagtile * const targ,
                      const int x, const int y,
                      const int sx, const int sy)
    {
        uint8_t pix;
        uint8_t pix_right;

        pix = targ->get(m_level, x, y);
        // Chech whether The right side pixel still inside tile border.
        // Because calling FlagtileSurface::get_pixel is rather higher
        // processing cost than Flagtile::get.
        if (x < PYRAMID_TILE_SIZE(0)-1) {
            pix_right = targ->get(0, x+1, y);
        }
        else
            pix_right = m_surf->get_pixel(0, sx+1, sy);


        if ((pix & FLAG_AA) == 0 && (pix_right & FLAG_AA) == 0
                && (pix & PIXEL_MASK) != PIXEL_FILLED
                && (pix_right & PIXEL_MASK) == PIXEL_FILLED) {
            // This class searches target pixel from left to right for each line.
            // And we use right-handed rule to walk area ridge.
            // Therefore, we would face top at the start point always.

            // IMPORTANT: initialize class unique members before walk!
            m_walking_started = false;
            m_line_dir = 2; // 2 means "Currently face below" -
                            // to proceed outside the area (counter-clockwise).
            m_sx = m_x;
            m_sy = m_y;
            walk(sx, sy, 0);
        }
    }
};



// Stores perimeter information.
// To execute some operation later.
// (For example, erase pixel flag)
typedef struct {
    int sx;
    int sy;
    int length;
    double reject_ratio; // Indicates how open this area is.
    double accept_ratio; // Indicates how connected this area is.
    int direction;
    bool clockwise;
} perimeter_info;

// Count perimeter of filled area.
// This class just walk and count perimeter, and stores that position, length and
// `encount` values into GQueue object.
// Walked regions are marked with FLAG_DECIDED flag, and it is erased later.
class CountPerimeterWalker: public WalkingKernel
{
protected:
    uint8_t m_targ_pix; // Target pixel value which we walking within.
    int m_reject; // A reject counter, incremented when walker touch `outside pixels`
    int m_accept; // A accept counter, incremented when walker touch `filled pixels`
    int m_total_surround; // Total surronding pixel count. Used for removing `too opened` area.
    GQueue* m_queue; // Borrowed queue, set at start method.

    // Check whether wall pixel is `opened` one.
    // `opened` means `It is a wall pixel but not PIXEL_CONTOUR`
    // This method should be called ONLT AT ONCE PER
    // entering a new pixel or right-rotation.
    virtual void check_opened_pixel(const uint8_t pixel) {
        switch(pixel & PIXEL_MASK) {
            case PIXEL_EMPTY:
            case PIXEL_OUTSIDE:
                m_reject++;
                break;
            case PIXEL_FILLED:
                m_accept++;
            default:
                break;
        }

        m_total_surround++;
    }

    // tells whether the pixel value is wall pixel or not.
    virtual bool is_wall_pixel(const uint8_t pixel)
    {
        // Return True when pixel is NOT the target.
        return !((pixel & PIXEL_MASK) == m_targ_pix);
    }

    virtual void on_rotate_cb(const bool right)
    {
        // If rotating left, it is `vacant pixel` of perimeter.
        // so decliment perimeter(i.e. m_step).
        if (!right)
            m_step--;
        else {
            check_opened_pixel(get_hand_pixel());
        }
    }

    virtual void on_new_pixel()
    {
        // Mark this pixel as `already walked`.
        m_surf->replace_pixel(m_level, m_x, m_y,
                              m_targ_pix | FLAG_WORK);

        check_opened_pixel(get_hand_pixel());
    }

    void push_queue(const int sx, const int sy, const int direction)
    {
        perimeter_info *info = new perimeter_info();
        info->sx = sx;
        info->sy = sy;
        info->length = m_step;
        info->direction = direction;
        info->reject_ratio = (double)m_reject / (double)m_total_surround;
        info->accept_ratio = (double)m_accept / (double)m_total_surround;
        info->clockwise = is_clockwise();
        g_queue_push_tail(m_queue, info);
    }

public:
    CountPerimeterWalker(FlagtileSurface *surf,
                         GQueue* queue)
        :   WalkingKernel(surf),
            m_queue(queue) { }

    void set_target_pixel(uint8_t pixel) { m_targ_pix = pixel; }

    virtual bool start(Flagtile * const targ, const int sx, const int sy)
    {
#ifdef HEAVY_DEBUG
            assert(m_queue != NULL);
#endif
        if (targ->is_filled_with(m_level, m_targ_pix)) {
            process_only_ridge(targ, sx, sy);
            end(targ);
            return false;
        }
        return (targ->get_pixel_count(m_level, m_targ_pix) > 0);
    }

    virtual void step(Flagtile * const targ,
                      const int x, const int y,
                      const int sx, const int sy)
    {
        // Use raw pixel, DO not remove FLAG_WORK.
        // (some area might be walked previously,
        // and it might have been marked with FLAG_WORK)
        uint8_t pixel = targ->get(m_level, x, y);

        if (pixel == m_targ_pix) {
            // Look just only left pixel. it is enough for ALL situation.
            // Because this search is done from left to right of each
            // scanline.
            uint8_t pix_l = get_pixel_with_direction(sx, sy, OFFSET_LEFT);
            if (is_wall_pixel(pix_l)) {
                m_reject = 0;
                m_accept = 0;
                m_total_surround = 0;

                // Walking always starts from the pixel
                // which is surrounded by at least two wall pixels.
                // But, we always check `top` pixel of starting point at
                // initial call of `on_new_pixel` from `walk` method.
                // So we need just check `left` pixel at here.
                check_opened_pixel(pix_l);

                // Start from current kernel pixel.
                walk(sx, sy, OFFSET_RIGHT);
                // This kernel just walk and queue(GQueue) the result.
                // That queue would be processed (or might be discarded)
                // later.
                push_queue(sx, sy, OFFSET_RIGHT);
            }
        }
    }
};

// Count `overwrapped` contour kernel.
// This is lightly modified version of CountPerimeterWalker.
class CountOverwrapWalker: public CountPerimeterWalker
{
protected:

    // Check whether wall pixel is `opened` one.
    // `opened` means `It is a wall pixel but not PIXEL_CONTOUR`
    // This method should be called ONLT AT ONCE PER
    // entering a new pixel or right-rotation.
    virtual void check_opened_pixel(const uint8_t pixel) {
        switch(pixel & PIXEL_MASK) {
            case PIXEL_FILLED:
                m_accept++;
                break;
            default:
                m_reject++;
                break;
        }
        m_total_surround++;
    }

public:
    CountOverwrapWalker(FlagtileSurface *surf,
                        GQueue* queue)
        :   CountPerimeterWalker(surf, queue) { }
};

// Clear flag around target pixels.
// This class just utilize WalkingKernel functionality.
class ClearflagWalker: public WalkingKernel
{
protected:
    uint8_t m_targ_pix;

    virtual bool is_wall_pixel(const uint8_t pixel)
    {
        return (pixel & PIXEL_MASK) != m_targ_pix;
    }

    virtual void on_new_pixel()
    {
        // Replace PIXEL_FILLED... i.e. `clear flag`
        uint8_t pix = m_surf->get_pixel(m_level, m_x, m_y);
        m_surf->replace_pixel(m_level, m_x, m_y, (pix & PIXEL_MASK));
    }

public:
    ClearflagWalker(FlagtileSurface *surf)
        :   WalkingKernel(surf) {}

    // This kernel does not start and step.
    // just walk from specific point.
    virtual bool start(Flagtile * const targ, const int sx, const int sy)
    { return false; }

    virtual void step(Flagtile * const targ,
                      const int x, const int y,
                      const int sx, const int sy) {}

    void set_target_pixel(uint8_t targ_pix)
    {
        m_targ_pix = targ_pix;
    }
};

// HoleFiller
// to fill all small holes inside a large area.
class HoleFiller : public Filler
{
public:
    HoleFiller()
        : Filler() {}

    virtual bool match(const uint8_t pix)
    {
        return ((pix & PIXEL_MASK) == PIXEL_AREA
                    || (pix & PIXEL_MASK) == PIXEL_CONTOUR
                    || pix == (PIXEL_FILLED | FLAG_WORK));
    }

    virtual void step(Flagtile *t, const int x, const int y) 
    {
        t->replace(m_level, x, y, PIXEL_FILLED);
    }

    virtual uint8_t get_fill_pixel(){ return PIXEL_FILLED; }
    virtual uint8_t get_target_pixel(){ return PIXEL_INVALID; }
};
#endif
