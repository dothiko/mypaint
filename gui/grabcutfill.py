# This file is part of MyPaint.
# Copyright (C) 2016 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or

"""OpenCV grabcut fill tool"""

## Imports
from __future__ import division, print_function
import math
import weakref
import numpy as np
import numpy.lib.stride_tricks as nptrick

import gi
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
from gettext import gettext as _
import cairo

import gui.mode
import gui.cursor
import lib.mypaintlib
import lib.surface
import lib.pixbufsurface
import gui.freehand
import lib.helpers
from gui.additional_mixin import *

""" This is a Experimental tool for graphcut fill.
Currently using python-opencv, OpenCV2.grabCut function for graphcut.
Also, Python-Imaging is used to convert between pixbuf and
opencv2 numpy.array.
At least PIL can be replaced easily with our original C++ module.

Problem is, energy funcion of OpenCV2.grabCut is mainly for photographic
image, not for comic/anime like lineart.
cv2.grabCut is usable for painting ambiguous lineart too,
but we'll need much more color hints compared to GMIC or LazyBrush.

And, graphcut operation is so heavy (not only OpenCV2), Mypaint become
unresponsive while that function is running.

"""

## Enums

class PreviewMethod:
    """Enumeration of preview method"""

    NO_PREVIEW = 0  #: not preview. generate final output.
    QUARTER = 1  #: 1/4 size preview
    EXACT = 2 #: exactly same as final output

## Function defs
def model_to_display_area(tdw, x, y, w, h):
    """ Convert model-coodinate area into display one.
    """
    sx = x
    sy = y
    ex = x + w - 1
    ey = y + h - 1
    sx, sy = tdw.model_to_display(sx, sy)
    ex, ey = tdw.model_to_display(ex, ey)

    sx , ex = min(sx, ex), max(sx, ex)
    sy , ey = min(sy, ey), max(sy, ey)
    w = ex - sx + 1
    h = ey - sy + 1
    return (sx, sy, w, h)

def grabcutfill(sample_layer, lineart_layer,
                fg_color, dilation_size,
                preview, remove_lineart,
                iter_cnt=3):
    """grabcutfill

    This function use grabCut function of Opencv2, to
    fill colors from hint strokes, especially lineart.

    :param sample_layer: color hint sample layer.
    :param lineart_layer: the foreground lineart layer.
    :param fg_color: foreground color tuple.
                     Hints which has other color are
                     treated as BG color hint.
    :param dilation_size: dilation size for final output region.
                          This option is useful when used with
                          remove_lineart option.
    :param preview: PreviewMethod value.
    :param remove_lineart: In some case, lineart itself
                           recognized as foreground.
                           It produces unintentional color
                           pixels around lineart.
                           If remove_lineart is true,
                           opaque pixels of lineart removed from
                           filled area.
    :return: when failed, None.
             when success for execute, it is a tuple of
             (a pixbuf of layer source,
                x position of layer,
                y position of layer)

             when success for preview, it is a tuple of
            (a cairo surface,
                x position of lineart layer,
                y position of lineart layer,
                width of lineart layer,
                height of lineart layer)

            all dimension values are in model coordinate.

    """

    # Import cv2 here,to delay import until this function used.
    # Otherwise, this customized Mypaint cannot run
    # when Python-OpenCV2 is not installed in system
    # even never use this function.
    import cv2
    # Also, I cannot convert pixbuf to numpy array.
    # I googled but nothing usable codes hit...
    # Thus, use Python-Imaging Library for glue code.
    from PIL import Image

    # TODO Currently this code is just experiment one,
    # so ignore about processing speed, memory/processing effeciency
    # etc.


    # In these converter functions,
    # there is no need to convert BGR to RGB
    # Because we never display OpenCV images.
    def convert_layer_to_cvimg(layer, bbox, scaling_filter=Image.NONE):
        if bbox is None:
            bbox = layer.get_bbox()
        pixbuf = layer.render_as_pixbuf(*bbox, alpha=True)

        stride = pixbuf.get_rowstride()
        width = pixbuf.get_width()
        height = pixbuf.get_height()
        pim = Image.frombytes("RGBA",(width,height),pixbuf.get_pixels() )
        if preview == PreviewMethod.QUARTER:
            pim = pim.resize((int(width/4), int(height/4)), scaling_filter)
        pim = pim.convert("RGB")
        return np.asarray(pim)

    def convert_cvimg_to_pixbuf(cvimg, alpha=None):
        # TODO currently, in some (rare) case,
        # adding numpy alpha array generate invalid pixbuf
        # and it cause coredump later.
        # So such alpha-adding functionality removed, 
        # currently regard pure black color pixel as 'transparent'.
        
        have_alpha = alpha is not None

       #if have_alpha:
       #    cvimg = np.concatenate((cvimg, alpha), axis = 2)
       #    pim = Image.fromarray(cvimg, 'RGBA')
       #    pixel_cnt = 4
       #else:
        pim = Image.fromarray(cvimg, 'RGB')
        have_alpha = False
        pixel_cnt = 3

        t_buf=pim.tobytes()
        width, height = pim.size

        cvpixbuf = GdkPixbuf.Pixbuf.new_from_data(
            t_buf,
            GdkPixbuf.Colorspace.RGB,
            have_alpha,
            8,
            width,
            height,
            width*pixel_cnt)

        if not have_alpha:
            # We need add alpha value into pixbuf, to convert it to layer.
            # At here just use complete black as alpha mask.
            # It is not good, only for experimental purpose.
            cvpixbuf = cvpixbuf.add_alpha(True, 0, 0, 0)
        return cvpixbuf

    # Original shape (prefixed with 'o') might not be same as
    # current target image(img_art). that image might be preview one.
    # Also, target image would be added margin pixel border in future.
    lineart_bbox = lineart_layer.get_bbox()
    ox, oy, ow, oh = lineart_bbox
    if ow == 0 or oh == 0:
        return None

    sample_bbox = sample_layer.get_bbox()
    entire_bbox = lib.helpers.Rect()
    entire_bbox.expandToIncludeRect(lineart_bbox)
    entire_bbox.expandToIncludeRect(sample_bbox)
    ex, ey, ew, eh = entire_bbox

    img_art = convert_layer_to_cvimg(lineart_layer, 
                                     entire_bbox,
                                     Image.BICUBIC)
    img_sample = convert_layer_to_cvimg(sample_layer, 
                                        entire_bbox,
                                        Image.NONE)

    layer_sources = []


    w = img_art.shape[1]
    h = img_art.shape[0]

    # For your convinience, in mask array for cv2.grabCut,
    # mask pixel value is, 0 for BG, 1 for FG,
    # 2 for Probably BG and  3 for Probably FG.
    #
    # The 'mask' array created as zero-filled one, but
    # actually initialized by cv2.grabCut with GC_INIT_WITH_RECT
    # as filled by value of 2(Probable Background)
   #mask = np.zeros(img_art.shape[:2], dtype = np.uint8) # This works for opencv,
                                                         # but this array cannot
                                                         # concatenate as alpha value.
    mask = np.zeros((h,w,1), dtype = np.uint8)

    # TODO this is only for testing.
    # we must convert cvimg with surrounding pixel, to ensure
    # blank area around it.
    margin = 1
    rect = (margin, margin, w-margin*2, h-margin*2)

    # Then, detect contours and write it to mask.
    # This is actually labelling, but there is no labelling API
    # until OpenCV3.
    img_gray = cv2.cvtColor(img_sample, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 32, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

    # XXX To get user-assigned sample color,
    # use the center point(moment) of contour currently.
    # It does not consider the contour shape is convex
    # or concave.
    fg_cnt = []
    bg_cnt = []

    for cnt in contours:
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        color = tuple(img_sample[cy, cx])
        if color == fg_color:
            fg_cnt.append(cnt)
        elif color == (0, 0, 0):
            # currently ignore complete black contour.
            pass
        else:
            bg_cnt.append(cnt)

    print('processing start')
    bgdmodel = np.zeros((1,65),np.float64)
    fgdmodel = np.zeros((1,65),np.float64)

    cv2.grabCut(
        img_art, mask, rect,
        bgdmodel, fgdmodel, 1,
        cv2.GC_INIT_WITH_RECT
    )

    cv2.drawContours(mask,
                     bg_cnt,
                     -1,
                     0,
                     -1)

    cv2.drawContours(mask,
                     fg_cnt,
                     -1,
                     1,
                     -1)

    cv2.grabCut(
        img_art, mask, rect,
        bgdmodel, fgdmodel, iter_cnt,
        cv2.GC_INIT_WITH_MASK
       #cv2.GC_EVAL # <- does not work with this?
    )

    mask2 = np.where((mask==1) + (mask==3),1,0).astype('uint8')
    if remove_lineart:
        alpha = np.where(img_art!=(0,0,0), 0, 1).astype('uint8')
        junk, alpha = np.dsplit(alpha, [2])
        np.putmask(mask2, alpha==0, 0)

    if dilation_size > 0:
        dilation_size = int(dilation_size)
        kernshape = (dilation_size, dilation_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           kernshape)
        mask2 = cv2.dilate(mask2, kernel, iterations=1)

    img_fill = np.zeros((h, w, 3), np.uint8)
    img_fill[:] = fg_color
    np.putmask(mask2, mask2==1, 255)
    img_fill = cv2.bitwise_and(img_fill, img_fill ,mask=mask2)

    if preview == PreviewMethod.NO_PREVIEW:
        layer_source = convert_cvimg_to_pixbuf(img_fill)

        print('execute ended!')
        return (layer_source, ex, ey)
    else:
        # make entire alpha value to half,
        # because opaque filled area (might) hide anything below.
        np.putmask(mask2, mask2==255, 128)
        mask2 = mask2.reshape((h, w, 1))

        img_result = cv2.cvtColor(img_fill, cv2.COLOR_BGR2RGB)
        img_result = np.concatenate((img_result, mask2), axis=2)
        stride = cairo.ImageSurface.format_stride_for_width (cairo.FORMAT_ARGB32, w)
        surf = cairo.ImageSurface.create_for_data(
            img_result,
            cairo.FORMAT_ARGB32,
            w, h,
            stride
        )
        print('preview ended!')
        return (surf, ex, ey, ew, eh)

## Class defs
class GrabcutFillMode (gui.freehand.FreehandMode,
                       OverlayMixin):

    """Mode for flood-filling with the sample of current layer"""

    # TODO currently Based on flood-fill, but might be changed to
    # base on freehand. because we would need change color samples
    # with brush stroke, immidiately & seamlessly.

    ## Class constants

    ACTION_NAME = "GrabcutFillMode"
    permitted_switch_actions = set([
        'RotateViewMode', 'ZoomViewMode', 'PanViewMode',
        'ColorPickMode', 'ShowPopupMenu',
        ])

    _OPTIONS_WIDGET = None
    _CURSOR_FILL_PERMITTED = gui.cursor.Name.CROSSHAIR_OPEN_PRECISE
    _CURSOR_FILL_FORBIDDEN = gui.cursor.Name.ARROW_FORBIDDEN

    ## Instance vars (and defaults)

    pointer_behavior = gui.mode.Behavior.PAINT_CONSTRAINED
    scroll_behavior = gui.mode.Behavior.CHANGE_VIEW

    _current_cursor = _CURSOR_FILL_PERMITTED

    ## Method defs

    def enter(self, doc, **kwds):
        super(GrabcutFillMode, self).enter(doc, **kwds)
        self.app = doc.app
        rootstack = self.doc.model.layer_stack
        rootstack.current_path_updated += self._update_ui
        rootstack.layer_properties_changed += self._update_ui
        # We need ensure overlay here,
        # because User might do preview immidiately
        # without any action on canvas.
        self._ensure_overlay_for_tdw(doc.tdw)
        self._update_ui()

    def leave(self, **kwds):
        rootstack = self.doc.model.layer_stack
        rootstack.current_path_updated -= self._update_ui
        rootstack.layer_properties_changed -= self._update_ui
        return super(GrabcutFillMode, self).leave(**kwds)

    @classmethod
    def get_name(cls):
        return _(u'Grabcut Fill')

    def get_usage(self):
        return _(u"Fill areas with color samples of current layer")

    def __init__(self, ignore_modifiers=False, **kwds):
        super(GrabcutFillMode, self).__init__(**kwds)
        self._preview_info = None

    def _update_ui(self, *_ignored):
        """Updates the UI from the model"""
        model = self.doc.model

        # Determine which layer will receive the fill based on the options
        opts = self.get_options_widget()
        opts.target_mode = self
        target_layer = model.layer_stack.current

    ## Event handlers

    def motion_notify_cb(self, tdw, event, fakepressure=None):
        self._ensure_overlay_for_tdw(tdw)
        super(GrabcutFillMode, self).motion_notify_cb(
            tdw, event, fakepressure)

    ## Mode options

    def get_options_widget(self):
        """Get the (class singleton) options widget"""
        cls = self.__class__
        if cls._OPTIONS_WIDGET is None:
            widget = GrabFillOptionsWidget()
            cls._OPTIONS_WIDGET = widget
        return cls._OPTIONS_WIDGET

    ## Execute

    def _preview_fill(self):
        layers = self.doc.model.layer_stack
        current = layers.current
        cur_path = layers.current_path
        top_path = layers.path_above(cur_path, insert=False)
        toplayer = layers.deepget(top_path)

        opts = self.get_options_widget()
        self.app.show_transient_message(_("Processing, Please wait..."))
        info = grabcutfill(
            current,
            toplayer,
            self._get_fg_color(),
            opts.dilation_size,
            opts.preview_method,
            opts.remove_lineart
        )

        if info is None:
            self.app.show_transient_message(_("Grabcutfill is not executed. lineart layer might be empty?"))
        else:
            self.app.show_transient_message(_("Grabcutfill Preview has completed."))

        self._preview_info = info
        self._queue_draw_preview(None)

    def _execute_fill(self):
        layers = self.doc.model.layer_stack
        current = layers.current
        cur_path = layers.current_path
        top_path = layers.path_above(cur_path, insert=False)
        toplayer = layers.deepget(top_path)

        if toplayer is None:
            self.app.message_dialog(
                _("You need a line-art layer above the current color-hint layer."),
                Gtk.MessageType.ERROR)
            return

        opts = self.get_options_widget()
        self.app.show_transient_message(_("Processing, Please wait..."))
        layer_src_info = grabcutfill(
            current,
            toplayer,
            self._get_fg_color(),
            opts.dilation_size,
            PreviewMethod.NO_PREVIEW,
            opts.remove_lineart
        )

        if layer_src_info:
            cpixbuf, ox, oy = layer_src_info
            layers = []
            cl = lib.layer.PaintingLayer(name=_("Grabcutfill Result"))
            cl.load_surface_from_pixbuf(cpixbuf, int(ox), int(oy))
            layers.append(cl)
            self.doc.model.grabcut_fill(layers, cur_path)
            self.app.show_transient_message(_("Grabcutfill has completed."))
        else:
            self.app.show_transient_message(_("Grabcutfill is not executed. lineart layer might be empty?"))

        if self._preview_info is not None:
            self._queue_draw_preview(None)
            self._preview_info = None

    def clear_preview(self):
        self._queue_draw_preview(None)
        self._preview_info = None

    ## Overlays
    def _generate_overlay(self, tdw):
        return _Overlay_Grabcut(self, tdw)

    def _queue_draw_preview(self, tdw):
        if tdw is None:
            for tdw in self._overlays.keys():
                self._queue_draw_preview(tdw)
            return

        if self._preview_info:
            surf, x, y, w, h = self._preview_info
            x, y, w, h = gui.ui_utils.model_to_display_area(
                        tdw, x, y, w, h)
            print((x,y,w,h))
            tdw.queue_draw_area(x, y, w, h)

    ## Others
    def _get_fg_color(self):
        color = self.app.brush_color_manager.get_color()
        return (int(color.r * 255),
                int(color.g * 255),
                int(color.b * 255))

class _Overlay_Grabcut(gui.overlays.Overlay):
    """Overlay for grabcut_fill mode """

    def __init__(self, mode, tdw):
        super(_Overlay_Grabcut, self).__init__()
        self._mode_ref = weakref.ref(mode)
        self._tdw = weakref.proxy(tdw)

    def paint(self, cr):
        """Draw Grabcut sample"""
        mode = self._mode_ref()
        tdw = self._tdw
        if (mode is not None
                and mode._preview_info is not None):
            surf, ox, oy, ow, oh = mode._preview_info
           #tdw = self._tdw
           #sx = x
           #sy = y
           #ex = x + w - 1
           #ey = y + h - 1
           #sx, sy = tdw.model_to_display(sx, sy)
           #ex, ey = tdw.model_to_display(ex, ey)
           #
           #sx , ex = min(sx, ex), max(sx, ex)
           #sy , ey = min(sy, ey), max(sy, ey)
           #w = ex - sx + 1
           #h = ey - sy + 1

            x, y, w, h = gui.ui_utils.model_to_display_area(
                            tdw, ox, oy, ow, oh)

            cr.save()
            cr.translate(x, y)
            sw = surf.get_width()
            sh = surf.get_height()
            if sw != ow:
                cr.scale(float(ow) / sw,
                         float(oh) / sh)

            # XXX cr.set_source_surface must be placed
            # after cr.scale called!!!
            cr.set_source_surface(
                surf, 0, 0)
            cr.rectangle(0, 0, sw, sh)
            cr.clip()
            cr.paint()
            cr.restore()

class GrabFillOptionsWidget (Gtk.Grid):
    """Configuration widget for the flood fill tool"""

    REMOVE_LINEART_PREF = 'grabcut_fill.remove_lineart'
    PREVIEW_FAST_PREF = 'grabcut_fill.preview_fast'
    DILATION_SIZE_PREF = 'grabcut_fill.dilate_size'

    DEFAULT_REMOVE_LINEART = False
    DEFAULT_PREVIEW_FAST = False
    DEFAULT_DILATION_SIZE = 0

    def __init__(self):
        Gtk.Grid.__init__(self)

        self.set_row_spacing(6)
        self.set_column_spacing(6)
        from application import get_app
        self.app = get_app()
        self._mode_ref = None
        prefs = self.app.preferences
        row = -1

        row += 1
        label = Gtk.Label()
        label.set_markup(_("Preview method:"))
        label.set_tooltip_text(_("To select Preview method."))
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        self.attach(label, 0, row, 1, 1)

        text = _("Faster")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            _("Generate preview in faster but not accurate method."))
        self.attach(checkbut, 1, row, 1, 1)
        active = prefs.get(self.PREVIEW_FAST_PREF,
                           self.DEFAULT_PREVIEW_FAST)
        checkbut.set_active(active)
        checkbut.connect("toggled", self._preview_fast_toggled_cb)
        self._preview_fast_toggle = checkbut

        row += 1
        label = Gtk.Label()
        label.set_markup(_("Output:"))
        label.set_tooltip_text(_("About output results"))
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        self.attach(label, 0, row, 1, 1)

        text = _("Remove lineart")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            _("Remove lineart opaque area from filled area."))
        self.attach(checkbut, 1, row, 1, 1)
        active = prefs.get(self.REMOVE_LINEART_PREF ,
                           self.DEFAULT_REMOVE_LINEART)
        checkbut.set_active(active)
        checkbut.connect("toggled", self._remove_lineart_toggled_cb)
        self._remove_lineart_toggle = checkbut

        row += 1
        label = Gtk.Label()
        label.set_markup(_("Dilation Size:"))
        label.set_tooltip_text(_("How many pixels the filled area to be dilated."))
        label.set_alignment(1.0, 0.5)
        label.set_hexpand(False)
        self.attach(label, 0, row, 1, 1)
        value = prefs.get(self.DILATION_SIZE_PREF, self.DEFAULT_DILATION_SIZE)
        value = float(value)
        # Theorically 'dilation fill' would accepts maximum
        # mypaintlib.TILE_SIZE-1 pixel as dilation size.
        # but it would be too large (even in 4K),
        adj = Gtk.Adjustment(value=value, lower=0.0,
                             upper=lib.mypaintlib.TILE_SIZE / 2 - 1,
                             step_increment=1, page_increment=4,
                             page_size=0)
        adj.connect("value-changed", self._dilation_size_changed_cb)
        self._dilation_size_adj = adj
        spinbtn = Gtk.SpinButton()
        spinbtn.set_hexpand(True)
        spinbtn.set_adjustment(adj)
        self.attach(spinbtn, 1, row, 1, 1)

        row += 1
        btn = Gtk.Button()
        btn.set_label(_("Preview"))
        btn.connect("clicked", self._preview_clicked_cb)
        btn.set_hexpand(True)
        self.attach(btn, 0, row, 2, 1)

        row += 1
        btn = Gtk.Button()
        btn.set_label(_("Execute"))
        btn.connect("clicked", self._execute_clicked_cb)
        btn.set_hexpand(True)
        self.attach(btn, 0, row, 2, 1)

    @property
    def target_mode(self):
        if self._mode_ref is not None:
            return self._mode_ref()

    @target_mode.setter
    def target_mode(self, mode):
        self._mode_ref = weakref.ref(mode)

    @property
    def preview_method(self):
        if self._preview_fast_toggle.get_active():
            return PreviewMethod.QUARTER
        else:
            return PreviewMethod.EXACT

    @property
    def remove_lineart(self):
        return self._remove_lineart_toggle.get_active()

    @property
    def dilation_size(self):
        return math.floor(self._dilation_size_adj.get_value())

    def _reset_clicked_cb(self, button):
        self._remove_lineart_toggle._set_active(self.DEFAULT_REMOVE_LINEART)
        self._preview_fast_toggle._set_active(self.DEFAULT_PREVIEW_FAST)

    def _execute_clicked_cb(self, button):
        mode = self.target_mode
        if mode is not None:
            mode._execute_fill()

    def _preview_clicked_cb(self, button):
        mode = self.target_mode
        if mode is not None:
            mode._preview_fill()

    def _dilation_size_changed_cb(self, adj):
        self.app.preferences[self.DILATION_SIZE_PREF] = self.dilation_size

    def _remove_lineart_toggled_cb(self, btn):
        self.app.preferences[self.REMOVE_LINEART_PREF] = self.remove_lineart

    def _preview_fast_toggled_cb(self, btn):
        # property preview_method returns dedicated enum class value,
        # so use get_active() of the widget.
        self.app.preferences[self.PREVIEW_FAST_PREF] = \
                self._preview_fast_toggle.get_active()


