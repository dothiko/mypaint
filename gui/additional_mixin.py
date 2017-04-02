#!/usr/bin/env python
# -*- coding: UTF-8 -*-

class OverlayMixin(object):
    """ Overlay drawing mixin, for some modes.
    """

    def __init__(self):
        self._overlays = {}  # keyed by tdw

    def _generate_overlay(self, tdw):
        raise NotImplementedError("You need to implement _generate_overlay,to use overlay")

    #  FIXME: mostly copied from gui/inktool.py
    #  so inktool.py should inherit this class too...

    def _ensure_overlay_for_tdw(self, tdw):
        overlay = self._overlays.get(tdw)
        if not overlay:
            overlay = self._generate_overlay(tdw)
            tdw.display_overlays.append(overlay)
            self._overlays[tdw] = overlay
        return overlay

    def _discard_overlays(self):
        for tdw, overlay in self._overlays.items():
            tdw.display_overlays.remove(overlay)
            tdw.queue_draw()
        self._overlays.clear()

if __name__ == '__main__':

    pass


