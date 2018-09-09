#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import struct

class Infotype:
    """Module constants, to recognize information type
    in strokemap.
    """
    TUPLE = 0
    BEZIER = 1
    RULER = 2
    POINT = 3

def load_info(f):
    """Load base information from file stream.
    Called at 
    data.StrokemappedPaintingLayer._load_strokemap_from_file
    """
    node_type, node_length, tlx, tly = struct.unpack('>IIii', f.read(16))
    body = f.read(node_length)
    return (body, node_type, tlx, tly)

def pack_info(infotype, infobody, translate_x, translate_y):
    """Pack infomation, compatible with load_info method.
    Place this function here to ease maintainance.
    """
    data = struct.pack(
        '>IIii', 
        infotype, len(infobody),
        int(translate_x), int(translate_y)
    )
    data += infobody
    return data

class PickableInfoMixin(object):
    """For context-pick feature, pick additional information
    from strokemap.
    """

    def _apply_info(self, info, offset):
        raise NotImplementedError("You must implement _apply_info")

    def _match_info(self, info_type_id):
        raise NotImplementedError("You must implement _apply_info")

    def _erase_old_stroke(si):
        """Utility method for tools which is node-based stroke.
        This would be called from self._apply_info()
        """

        # Erase old strokemap,
        # Because it is drawn right after nodes recovered!
        # If we dont remove that strokemap, unused strokemap
        # remained in surface.
        model = self.doc.model
        cl = model.layer_stack.current
        si.remove_from_surface(cl._surface)
        assert hasattr(cl, "remove_stroke_info")

        # Also, Remove stroke-node information from layer. 
        # Without this, re-edited stroke still exist as older shape.
        cl.remove_stroke_info(si) 

    def restore_from_stroke_info(self, si): 
        """Restore nodes from stroke info(StrokeNode class).
        Almost same as ExperimentInktool, but Node class is different.

        :return : unpacked raw nodes data string.

        CAUTION: Returned object is raw string node datas.
                 You must de-serialize them into node objects and
                 call inject_nodes method of this mixin.
        """
        assert isinstance(si, lib.strokemap.StrokeInfo) 

        if not self._match_info(si.get_info_type()):
            self.app.show_transient_message(
                _("This stroke is not drawn by %s" % self.name)
            )
            return False

        with si.get_offset() as offset:
            self._apply_info(si.get_info(), offset)

        return True

if __name__ == '__main__':

    pass


