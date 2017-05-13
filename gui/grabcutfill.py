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
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib, GObject, Pango
from gettext import gettext as _
import cairo

import gui.mode
import gui.cursor
import lib.mypaintlib as mypaintlib
import lib.surface
import lib.pixbufsurface
import gui.ui_utils
import gui.freehand
import lib.helpers
import drawwindow
import drawutils
import layers

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

class _Prefs:
    """Enumeration of preferences constants
    Mainly used at Optionspresenter """
    REMOVE_LINEART_PREF = 'grabcut_fill.remove_lineart_pixel'
    REMOVE_HINT_PREF = 'grabcut_fill.remove_hint'
    DILATION_SIZE_PREF = 'grabcut_fill.dilate_size'
    FILL_AREA_PREF = 'grabcut_fill.fill_area'

    DEFAULT_REMOVE_LINEART = True
    DEFAULT_REMOVE_HINT = True
    DEFAULT_DILATION_SIZE = 0
    DEFAULT_FILL_AREA = False

## Function defs

#def model_to_display_area(tdw, x, y, w, h):
#    """ Convert model-coodinate area into display one.
#    """
#    sx = x
#    sy = y
#    ex = x + w - 1
#    ey = y + h - 1
#    sx, sy = tdw.model_to_display(sx, sy)
#    ex, ey = tdw.model_to_display(ex, ey)
#
#    sx , ex = min(sx, ex), max(sx, ex)
#    sy , ey = min(sy, ey), max(sy, ey)
#    w = ex - sx + 1
#    h = ey - sy + 1
#    return (sx, sy, w, h)

def grabcutfill(sample_layer, lineart_layer,
                fg_color, dilation_size,
                remove_lineart_pixel,
                fill_contour = True,
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
                          remove_lineart_pixel option.
    :param remove_lineart_pixel: In some case, lineart itself
                           recognized as foreground.
                           It produces unintentional color
                           pixels around lineart.
                           If remove_lineart_pixel is true,
                           opaque pixels of lineart removed from
                           filled area.
    :param fill_contour: With removing lineart pixels,
                         it make small holes in grabcut filled area.
                         If this option is true, to surpass such glitch, 
                         entire grubcut area would be detected as 'contour' 
                         by cv2.findContours function 
                         and filled by cv2.drawContours function.

    :return: when failed, None.
             when success for execute, it is a new layer.
             when success for preview, it is a tuple of
            (a cairo surface,
                x position of lineart layer,
                y position of lineart layer,
                width of lineart layer,
                height of lineart layer)

            all dimension values are in model coordinate.

    """

    def get_tile_source(layer):
        # Wrapper inner function to get surface from the layer,
        # even the layer is actually layergroup 
        # (or something layer-like object).
        if hasattr(layer, "_surface"):
            return layer._surface
        else:
            return lib.surface.TileRequestWrapper(layer)

    # XXX DEBUGGING CODES
   #def save_mask_image(mask, fname='/tmp/maskimg.png'):
   #    import scipy.misc
   #    import Image
   #    mask2 = np.where((mask==1),255,0).astype('uint8')
   #    np.putmask(mask2, mask==2, 160)
   #    np.putmask(mask2, mask==3, 80)
   #    mask2 = np.c_[mask2, mask2, mask2]
   #    scipy.misc.imsave(fname, mask2)
   #
   #def save_cv_image(cvimage, fname='/tmp/cvimg.png'):
   #    import scipy.misc
   #    scipy.misc.imsave(fname, cvimage)

    # Import cv2 here,to delay import until this function used.
    # Otherwise, this customized Mypaint cannot run
    # when Python-OpenCV2 is not installed in system
    # even never use this function.
    try:
        import cv2
    except ImportError:
        from application import get_app
        app = get_app()
        app.message_dialog("You need to install python binding of OpenCV2 "
                           "for GrabcutFill.", Gtk.MessageType.ERROR)
        return None

    # Original shape (prefixed with 'o') might not be same as
    # current target image(img_art). that image might be preview one.
    # Also, target image would be added margin pixel border in future.
    lineart_bbox = lineart_layer.get_bbox()
    ox, oy, ow, oh = lineart_bbox
    if ow == 0 or oh == 0:
        return None

    sample_bbox = sample_layer.get_bbox()
    sx, sy, sw, sh = sample_bbox

    margin = 8
    N = mypaintlib.TILE_SIZE

    # Adjust sx and sy of bbox into tile-based position.
    sx = int(sx // N)
    sy = int(sy // N)

    # Get margin added dimension for Opencv image array.
    w = sw + margin * 2
    h = sh + margin * 2

    # XXX Background color in img_art array
    # In future, this should be configurable from optionspresenter,
    # to support the case of white(bright)-colored lineart.
    # grabCut function does not recognize alpha pixels,
    # so we need background color anyway.
    br, bg, bb = 1.0, 1.0, 1.0

    img_art = np.zeros((h, w, 3), dtype = np.uint8)

    mypaintlib.grabcututil_setup_cvimg(
        img_art,
        br, bg, bb,
        margin)

    # Create Opencv2 lineart image from the surface of the lineart layer. 
    src = get_tile_source(lineart_layer)

    for by in xrange(0, sh, N):
        ty = int(by // N) + sy
        for bx in xrange(0, sw, N):
            tx = int(bx // N) + sx
            with src.tile_request(tx, ty, readonly=True) as src_tile:
                # Currently, complete alpha in lineart layer
                # would be converted into pure white.
                mypaintlib.grabcututil_convert_tile_to_image(
                    img_art, src_tile,
                    bx, by,
                    br, bg, bb,
                    margin)

    mask = np.zeros((h, w, 1), dtype = np.uint8)

    # Initialize Opencv grabCut and mask image.
    bgdmodel = np.zeros((1,65),np.float64)
    fgdmodel = np.zeros((1,65),np.float64)
    rect = (1, 1, w-1, h-1)

    cv2.grabCut(
        img_art, mask, rect,
        bgdmodel, fgdmodel, 1,
        cv2.GC_INIT_WITH_RECT
    )

    # Then, Write filling hints into mask image.
    targ_r, targ_g, targ_b = fg_color
    src = get_tile_source(sample_layer)

    # TODO We need better labeling function for them.
    # Or, Use completely other way to indicate where is
    # fg or bg, such as placing nodes.
    # Currently using some sort of trick.

    for by in xrange(0, sh, N):
        ty = int(by // N) + sy
        for bx in xrange(0, sw, N):
            tx = int(bx // N) + sx
            with src.tile_request(tx, ty, readonly=True) as src_tile:

                mypaintlib.grabcututil_convert_tile_to_binary(
                    mask, src_tile,
                    bx, by,
                    targ_r, targ_g, targ_b,
                    0,# surely background hint
                    margin, 1,
                    0.05 # BG alpha tolerance should be lower
                         # than FG one.
                         # 0.05 is practical value,
                         # 0.2 produced failed result in many case.
                )

                mypaintlib.grabcututil_convert_tile_to_binary(
                    mask, src_tile,
                    bx, by,
                    targ_r, targ_g, targ_b,
                    1,# surely foreground hint
                    margin, 0,
                    0.7 # FG hints alpha tolerance.
                        # This MUST be higher than BG one.
                        # If BG one is same (or less just a bit),
                        # BG hint pixels would surround FG hints.
                        # so grabcut does not work.
                )


    # Finally, execute grabCut and get image area as mask.
    cv2.grabCut(
        img_art, mask, rect,
        bgdmodel, fgdmodel, iter_cnt,
        cv2.GC_INIT_WITH_MASK
       #cv2.GC_EVAL # <- does not work with this?
    )

    cl = lib.layer.PaintingLayer(name=_("Grabcutfill Result"))

    # Finalizing grabcut mask.
    # which is , remove detected lineart contour
    # and convert 'probable foreground' as foreground.
    mypaintlib.grabcututil_finalize_cvmask(
        mask, img_art,
        br, bg, bb,
        int(remove_lineart_pixel))


    # Dilating the output mask, if needed.
    if dilation_size > 0:
        dilation_size = int(dilation_size)
        neiborhood4 = np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]],
                                np.uint8)
        mask = cv2.dilate(mask, neiborhood4, iterations=dilation_size)

    # Detect 'grubcutted' and removed linearts area, 
    # and fill entire that area, to eliminate pixel holes.
    if remove_lineart_pixel and fill_contour:
        contours, hierarchy = cv2.findContours( 
                                mask,  
                                cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE)
        
        # We need BGR image to use drawcontour function.
        new_mask = np.zeros((h, w, 3), dtype = np.uint8)
        cv2.drawContours(
            new_mask, 
            contours, 
            -1, 
            (255, 255, 255),
            -1)
        mask = new_mask
        # Utilize new_mask BGR image as 'mask'
        # so target value would be changed.
        target_value = 255
    else:
        # mask is Binary image, so Target value must be 1.
        target_value = 1

    dst = cl._surface
    for by in xrange(0, sh, N):
        ty = int(by // N) + sy
        for bx in xrange(0, sw, N):
            tx = int(bx // N) + sx
            with dst.tile_request(tx, ty, readonly=False) as dst_tile:
                # Currently, complete alpha in lineart layer
                # would be converted into pure white.
                # This function accepts mask image of different 
                # two pixel formats.
                # It is binary image(array of 1byte mask, 0 or 1) 
                # or BGR image(array of 3bytes pixels).
                if mypaintlib.grabcututil_convert_binary_to_tile(
                        dst_tile, mask,
                        bx, by,
                        targ_r, targ_g, targ_b,
                        target_value,
                        margin):
                    dst._mark_mipmap_dirty(tx, ty)
    bbox = (sx * N, sy * N, sw, sh)
    dst.notify_observers(*bbox)

    return cl


## Class defs

class GrabcutFillMode (gui.freehand.FreehandMode,
                       gui.mode.OverlayMixin):

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

    _lineart_layer_ref = None

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

    @property
    def lineart_layer(self):
        """Lineart layer property.
        Internally, this prop use weak reference to ensure & follow
        layer existance.

        Furthermore, mode object would be generated each time 
        when grabcutfill tool is activated, even 'undo' operation
        executed. 
        Under this circumstance, self members are frequently initialized, 
        so we MUST use class attribute to store lineart layer reference.
        """
        cls = self.__class__
        if cls._lineart_layer_ref is not None:
            return cls._lineart_layer_ref()

    @lineart_layer.setter
    def lineart_layer(self, layer):
        cls = self.__class__
        if layer is not None:
            cls._lineart_layer_ref = weakref.ref(layer)
        else:
            cls._lineart_layer_ref = None

    ## Execute
    @drawwindow.with_wait_cursor
    def _execute_fill(self):
        app = self.app
        prefs = app.preferences
        layers = self.doc.model.layer_stack
        current = layers.current
        cur_path = layers.current_path

        lineart = self.lineart_layer
        if lineart is None:
            app.message_dialog(
                _("You need to select a line-art layer(or group)"
                  "at Options presenter."),
                Gtk.MessageType.ERROR)
            return

        # Getting pref options
        remove_lineart_pixel = prefs.get(_Prefs.REMOVE_LINEART_PREF,
                                         _Prefs.DEFAULT_REMOVE_LINEART)

        remove_hint_layer_flag = prefs.get(_Prefs.REMOVE_LINEART_PREF,
                                           _Prefs.DEFAULT_REMOVE_LINEART)

        fill_contour_flag = prefs.get(_Prefs.FILL_AREA_PREF,
                                      _Prefs.DEFAULT_FILL_AREA)

        if remove_hint_layer_flag:
            removed_hint_layer = current
        else:
            removed_hint_layer = None

        dilation_size = prefs.get(_Prefs.DILATION_SIZE_PREF,
                                  _Prefs.DEFAULT_DILATION_SIZE)

        # Executing grabcutfill
        app.show_transient_message(_("Processing, Please wait..."))
        new_layer = grabcutfill(
            current,
            lineart,
            self._get_fg_color(),
            dilation_size,
            remove_lineart_pixel,
            fill_contour = fill_contour_flag
        )

        if new_layer:
            # Actually, this 'grabcut_fill' (i.e. Command.grabcutfill) does
            # 'inserting the result layer into layerstack'.
            self.doc.model.grabcut_fill(
                new_layer,
                cur_path,
                removed_hint_layer
            )
            app.show_transient_message(_("Grabcutfill has completed."))

            if removed_hint_layer is not None:
                current = new_layer
            # DON'T FORGET TO Seleting current(hint) layer again.
            # Otherwise, when delete hint layer right after grabcut fill executed
            # exception would be raised.
            self.doc.model.select_layer(layer=current)


        else:
            app.show_transient_message(
                _("Grabcutfill is not executed."
                  "lineart layer might be empty?")
            )


    ## Overlays
    def _generate_overlay(self, tdw):
        return _Overlay_Grabcut(self, tdw)

    ## Others
    def _get_fg_color(self):
        color = self.app.brush_color_manager.get_color()
        return (color.r,
                color.g,
                color.b)

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

class LayerComboRenderer(Gtk.CellRenderer):
    """renderer used from layer combobox"""

    layer = GObject.property(type=GObject.TYPE_PYOBJECT, default=None)
    _FONT_SIZE = None
    MARGIN = 4
    icon_pixbufs = {}

    def __init__(self, targwidget):
        super(LayerComboRenderer, self).__init__()
        self.targwidget = targwidget

    @property 
    def FONT_SIZE(self):
        cls = self.__class__
        if cls._FONT_SIZE is None:
            st = Gtk.ComboBox.get_default_style()
            desc = st.font_desc
            size_base = desc.get_size() / Pango.SCALE
            if desc.get_size_is_absolute():
                # True == size is in device unit
                pass
            else:
                # False == size is in Point.
                # pixel = Point * (DPI / 72.0)
                scr = Gdk.Screen.get_default()
                dpi = scr.get_resolution()
                if dpi < 0:
                    dpi = 96.0
                size_base = math.floor(size_base * (dpi / 72.0))

            cls._FONT_SIZE = size_base
                
        return cls._FONT_SIZE

    # Gtk Property related:
    # These are something like a idiom.
    # DO NOT EDIT
    def do_set_property(self, pspec, value):
        setattr(self, pspec.name, value)

    def do_get_property(self, pspec):
        return getattr(self, pspec.name)

    # Rendering
    def do_render(self, cr, widget, bg_area, cell_area, flags):
        """
        :param cell_area: RectangleInt class
        """
        cr.save()
        layer = self.layer
        if layer is not None:
            cx, cy = cell_area.x, cell_area.y 
            cw, ch = cell_area.width, cell_area.height 
            cr.translate(cx, cy)
            h = self.FONT_SIZE
            top = (ch - h) / 2
            cr.translate(self.MARGIN, top)

            icon_name = layer.get_icon_name()
            icon = self.icon_pixbufs.get(icon_name, None)
            if icon is None:
                icon = drawutils.load_symbolic_icon(
                        icon_name,
                        self.FONT_SIZE,
                        fg=(0, 0, 0, 1),
                       )
                self.icon_pixbufs[icon_name] = icon

            # Drawing icon
            Gdk.cairo_set_source_pixbuf(cr, icon, 0 , 0)
            cr.rectangle(0, 0, h, h)
            cr.fill()
            cr.translate(h, 0)

            # Drawing layer name
            cr.move_to(self.MARGIN,
                       h)
            cr.set_font_size(h)
            cr.set_source_rgb(0.0, 0.0, 0.0)
            cr.show_text(layer.name)
        else:
            cr.rectangle(
                bg_area.x, bg_area.y, 
                bg_area.width, bg_area.height 
            )
            cr.fill()

        cr.restore()

    def do_get_preferred_height(self, widget):
        height = self.MARGIN + self.FONT_SIZE
        return (height, height)
    
    def do_get_preferred_width(self, widget):
        r = self.targwidget.get_allocation()
        # Set natural width as 1, to avoid resizing combobox itself.
        return (1, r.width)

class GrabFillOptionsWidget (Gtk.Grid):
    """Configuration widget for the flood fill tool"""

    def __init__(self):
        Gtk.Grid.__init__(self)
        self._update_ui = True

        self.set_row_spacing(6)
        self.set_column_spacing(6)
        from application import get_app
        self.app = get_app()
        self._mode_ref = None
        prefs = self.app.preferences

        def generate_label(text, tooltip):
            label = Gtk.Label()
            label.set_markup(text)
            label.set_tooltip_text(tooltip)
            label.set_alignment(1.0, 0.5)
            label.set_hexpand(False)
            self.attach(label, 0, row, 1, 1)
            return label

        row = 0
        label = generate_label(
                    _("Lineart:"),
                    _("Select Lineart layer"))

        combo = self._init_layer_combo()
        self.attach(combo, 1, row, 1, 1)
        self._lineart_combo = combo

        row += 1
        btn = Gtk.Button()
        btn.set_label(_("Set current layer as Lineart"))
        btn.connect("clicked", self._execute_clicked_cb)
        btn.set_hexpand(True)
        self.attach(btn, 0, row, 2, 1)

        row += 1
        label = generate_label(
                    _("Output:"),
                    _("About output results"))

        # 'Remove layer' related
        text = _("Remove lineart")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            _("Remove lineart opaque area from filled area."))
        self.attach(checkbut, 1, row, 1, 1)
        active = prefs.get(_Prefs.REMOVE_LINEART_PREF ,
                           _Prefs.DEFAULT_REMOVE_LINEART)
        checkbut.set_active(active)
        checkbut.connect("toggled", self._remove_lineart_pixel_toggled_cb)
        self._remove_lineart_pixel_toggle = checkbut

        row += 1
        text = _("Fill extracted area")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            _("Fill entire grabcut extracted area."))
        self.attach(checkbut, 1, row, 1, 1)
        active = prefs.get(_Prefs.FILL_AREA_PREF,
                           _Prefs.DEFAULT_FILL_AREA)
        checkbut.set_active(active)
        checkbut.connect("toggled", self._fill_grabcut_area_toggled_cb)
        self._fill_grabcut_area_toggle = checkbut

        row += 1
        text = _("Remove hint layer")
        checkbut = Gtk.CheckButton.new_with_label(text)
        checkbut.set_tooltip_text(
            _("Remove current hint layer,"
              "right after grabcut-fill is completed."))
        self.attach(checkbut, 1, row, 1, 1)
        active = prefs.get(_Prefs.REMOVE_HINT_PREF ,
                           _Prefs.DEFAULT_REMOVE_HINT)
        checkbut.set_active(active)
        checkbut.connect("toggled", self._remove_hint_toggled_cb)
        self._remove_hint_toggle = checkbut

        # Post process morphology operation related.
        row += 1
        label = generate_label(
                    _("Dilation Size:"),
                    _("How many pixels the filled area to be dilated."))

        value = prefs.get(_Prefs.DILATION_SIZE_PREF,
                          _Prefs.DEFAULT_DILATION_SIZE)
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
        btn.set_label(_("Execute"))
        btn.connect("clicked", self._execute_clicked_cb)
        btn.set_hexpand(True)
        self.attach(btn, 0, row, 2, 1)

        self._update_ui = False

    def _init_layer_combo(self):
        """Creating combobox for selecting lineart layer.
        """
        docmodel = self.app.doc.model
        treemodel = layers.RootStackTreeModelWrapper(docmodel)
        self._treemodel = treemodel
        combo = Gtk.ComboBox()
        combo.set_model(treemodel)
        combo.set_hexpand(True)

        cell = LayerComboRenderer(combo)
        combo.pack_start(cell, True)
        combo.add_attribute(cell, 'layer', 0)

        combo.connect("changed", self._on_layer_combo_changed)
        return combo

    @property
    def target_mode(self):
        if self._mode_ref is not None and self._update_ui==False:
            return self._mode_ref()

    @target_mode.setter
    def target_mode(self, mode):
        self._mode_ref = weakref.ref(mode)

    @property
    def dilation_size(self):
        return math.floor(self._dilation_size_adj.get_value())

    def _reset_clicked_cb(self, button):
        self._remove_lineart_pixel_toggle._set_active(
            self.DEFAULT_REMOVE_LINEART)

        self._preview_fast_toggle._set_active(
                self.DEFAULT_PREVIEW_FAST)

    def _execute_clicked_cb(self, button):
        mode = self.target_mode
        if mode is not None:
            mode._execute_fill()

    def _dilation_size_changed_cb(self, adj):
        self.app.preferences[_Prefs.DILATION_SIZE_PREF] = self.dilation_size

    def _remove_lineart_pixel_toggled_cb(self, btn):
        self.app.preferences[_Prefs.REMOVE_LINEART_PREF] = btn.get_active()

    def _remove_hint_toggled_cb(self, btn):
        self.app.preferences[_Prefs.REMOVE_HINT_PREF] = btn.get_active()

    def _fill_grabcut_area_toggled_cb(self, btn):
        self.app.preferences[_Prefs.FILL_AREA_PREF] = btn.get_active()

    def _on_layer_combo_changed(self, cmb):
        mode = self.target_mode
        if mode is not None:
            iter = cmb.get_active_iter()
            if iter is not None:
                layer = self._treemodel.get_value(iter, 0)
                mode.lineart_layer = layer
            else:
                mode.lineart_layer = None

