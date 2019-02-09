#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2018 by Dothiko <a.t.dothiko@gmail.com>
# Most part of this file is transplanted from
# original gui/inktool.py, re-organized it as Mixin.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.


import struct
from gettext import gettext as _

import lib.strokemap
import lib.command

class Infotype:
    """Module constants, to recognize information type
    in strokemap.
    This is the first byte of regularized `info` bytestring.
    """
    TUPLE = 0
    BEZIER = 1
    RULER = 2
    CENTER = 3

def load_from_filestream(f):
    """Load base information from file stream.
    Called at 
    data.StrokemappedPaintingLayer._load_strokemap_from_file
    """
    info_length, tlx, tly = struct.unpack('>Iii', f.read(12))
    body = f.read(info_length)
    return (body, tlx, tly)

def pack_for_filestream(infobody, translate_x, translate_y):
    """Pack infomation for file stream, compatible with load_info method.
    Place this function here to ease maintainance.

    :param infobody: A bytestring, which contains infotype at the first byte
                     of it, by regularize_info() function.
    """
    data = struct.pack(
        '>Iii', 
        len(infobody),
        int(translate_x), int(translate_y)
    )
    data += infobody
    return data

def regularize_info(infobody, infotype):
    """Utility function to add infotype, at the head of info bytestring.
    This is to regularize info data format over all derived class.
    Use this when PickableInfoMixin._pack_info returns bytestring.
    """
    assert infotype <= 255
    t = struct.pack('>B', infotype)
    return t + infobody

def extract_infotype(info):
    """Utility function to extract infotype from regularized infobody.
    """
    return ord(info[0])

def extract_info(info):
    """Utility function to extract info itself from regularized infobody.
    """
    return info[1:]

class PickableInfoMixin(object):
    """For context-pick feature, pick additional information
    from strokemap.
    """

    def _apply_info(self, strokeinfo, offset):
        """
        :param strokeinfo: Instance of lib.strokemap.StrokeInfo
        :param offset: A tuple of (x, y). This cannot be None.
                       If there is no offeet, use tuple (0, 0)
        """
        raise NotImplementedError("You must implement _apply_info")

    def _match_info(self, info_type_id):
        raise NotImplementedError("You must implement _match_info")

    def _pack_info(self):
        """Pack nodes as bytestring datas.
        Compressing it with zlib is optional.

        See regularize_info() function. Use that method to 
        return bytestring info.

        :return : nodes as bytestring data.
        """
        raise NotImplementedError("You must implement _pack_info")

    def _unpack_info(self, nodesinfo):
        """Unpack nodes from bytestring datas.
        :param noesinfo: a bytestring, generate from _pack_info
        """
        raise NotImplementedError("You must implement _unpack_info")

    def restore_from_stroke_info(self, si): 
        """Restore nodes from stroke info(StrokeNode class).
        Almost same as ExperimentInktool, but Node class is different.

        :param si: Picked stroke information.
                   This param can be either of StrokeInfo or StrokeShape.
                   And this method only accepts StrokeInfo.
        :return : unpacked raw nodes data string.

        CAUTION: Returned object is raw string node datas.
                 You must de-serialize them into node objects and
                 call inject_nodes method of this mixin.
        """
        if not (isinstance(si, lib.strokemap.StrokeInfo)
                    and self._match_info(si.get_info_type())):
            self.app.show_transient_message(
                _("This stroke is not drawn by %s" % self.get_name())
            )
            return False

        self._apply_info(si, si.get_offset())
        return True

