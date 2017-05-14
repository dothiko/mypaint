#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# This file is part of MyPaint.
# Copyright (C) 2017 by dothiko <dothiko@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# To implementing Projectsave functionality, files below are rewritten.
# 
# gui/filehandling.py (for save dialog, menu GUI handling) 
# gui/application.py (to initilize project-save MRU menu.)
# lib/document.py  (to invoke load/save project functionality)
# lib/layer/core.py
# lib/layer/data.py 
# lib/layer/tree.py
# lib/layer/group.py 
# (to add actual project load/save functinality for each layer classes)
# 
# basically, project-save functionality for layer classes are
# implemented as Mixin(Projectsaveable). 
# But some codes are added to existing methods,
# such as load_from_openraster_dir()


# ABOUT VERSION BACKUP:
#
# VERSION BACKUP functionality would work along this flow.
#
# 1. 'Save Current Project as New Version' menu action has start.
# 2. copy current stack.xml into backup directory with new version number, 
#    such as 'stack.xml.2'.
# 3. Walk all layers (Versioninfo.queue_backup_layers)
# 4. if modified timestamp or filesize of png file is not same as
#    recorded information at versioninfo.json (or record does not
#    exist), queue copy task
# 5. copy task runs when mypaint become idle,
#    it would copy the layer file with renaming,
#    such as, from "data/uuid-of-that-layer.png" to 
#    "backup/uuid-of-that-layer/2.png". 
# 6. if another project-save action triggered before queued task end, 
#    mypaint would flush all queued tasks before new project-save executed, 
#    so no problem.

from __future__ import absolute_import

import abc
import os
import shutil
import logging
logger = logging.getLogger(__name__)
import glob
import sys
import weakref
import uuid
import json
# We need __future__.absolte_import 
# for importing xml.etree.ElementTree
# Because the xml module in the lib/ directory is preferentially 
# loaded, in python 2.x series.
import xml.etree.ElementTree as ET

import lib.autosave
import lib.xml
import lib.layer

class Projectsaveable(lib.autosave.Autosaveable):
    """Mixin and abstract base for projectsaveable structures"""

    __metaclass__ = abc.ABCMeta

    ORA_LAYERID_ATTR = "{%s}layer-id" % (lib.xml.OPENRASTER_MYPAINT_NS,)

    @property
    def autosave_dirty(self):
        # To call superclass property
        return lib.autosave.Autosaveable.autosave_dirty.fget(self)

    @autosave_dirty.setter
    def autosave_dirty(self, value):
        """
        Setter for the dirty flag
        
        if autosave_dirty set as dirty,
        also project_dirty flag set as dirty.
        but when autosave_dirty is cleared,
        project_dirty is remained.
        """
        value = bool(value)
        self.__autosave_dirty = value
        if value == True:
            self.__project_dirty = True
        
    def save_to_project(self, projdir, path,
                           canvas_bbox, frame_bbox, force_write, **kwargs):
        """Saves the layer(or stack) data into an project directory

        This project-save method writes the each layers file on 
        the raw local host computer's filesystem, 
        not in orazip or other container file.
        The filesave is only done when the dirty flag of the layer
        is True, or force_write flag is True.

        Thus, file-saving time drastically shorten
        even all files are marked as dirty,
        because there is no read and write I/O time to unpack/re-pack the 
        container file.

        This is mostly same as (and base on) autosave, 
        the difference is that we can assign arbitrary directory 
        location and trigger save action at arbitrary timing.

        :param projdir: the project root directory.
        :param path: the filepath.
        :param canvas_bbox: the boundary box of canvas.
        :param frame_bbox: the boundary box of frame. this can be None.
        :param boolean force_write: if this flag is True, the layer(or stack)
            should be written even it is not dirty.                                    
        :param \*\*kwargs: To be passed to underlying save routines.
        """

    # load_from_project_dir is obsoluted, because it might makes
    # this class more complicated.
    # So simply modified load_from_oradir(or something like that)
    # to 'project' entry in kwargs.

    def clear_project_dirty(self):
        """
        clear dirty flag for project.

        setting dirty flag should be done from
        autosave_dirty.
        """
        self.__project_dirty = False

    @property
    def project_dirty(self):
        """
        Dirty flag for project-save feature.

        CAUTION: Project_dirty flag cannot be set alone.
        this is nearly read-only flag,
        we can only clear this.
        """
        try:
            return self.__project_dirty
        except AttributeError:
            self.__project_dirty = True
            return self.__project_dirty

    @property
    def filename(self):
        """This property shows the 'src' attribute of
        XML attr element.
        This property contains relative directory components.

        If that attribute does not exist (probably it should be
        first save of unsaved layer), self.unique_id used.

        NOTE:
        This property is fundamental one, just to retrieve unique
        picture filename.
        But, some type of layer would be consist from multiple files
        such as strokemap.
        So when you want to do some filesystem operation,
        do not use this property, use enum_filenames method instead.
        """

        if not hasattr(self, '_filename'):
            self._filename = os.path.join('data', 
                                          "%s.png" % self.unique_id)
        return self._filename

    @property
    def unique_id(self):
        """This property is ORA_LAYERID_ATTR('mypaint:layer-id') 
        attribute of XML attr element.
        """
        if not hasattr(self, '_unique_id') or self._unique_id == None:
            self._unique_id = str(uuid.uuid4())
        return self._unique_id

    def enum_filenames(self):
        """ Get a list of filenames which consists this layer. 
        Some type of layer might have additional stroke data or
        working file for vector graphics, etc.
        So we need enumlate all of them, to backup this layer.

        This is a generator function.
        """
        # XXX Should use NotimplementError and trap it ?
        #
        # Currently, use dummy yield statement and
        # it return 'empty' generator function as a default.
        # With this , we does not need try-catch statements
        # to use this method.
        # 
        # But, this codes cannot detect a bug such as 
        # forgetting to implement this method...
        raise StopIteration  
        yield None # Dummy yield, to return generator function.


class Versionsave(object):
    """Version save class, manage project version infomation.
    Very important class for version saving & reverting.
    """

    def __init__(self, path):
        """Constructor of Versionsave class.
        :param path: Project directory path. it should has 'backup'
                     subdirectory.
        """
        assert os.path.exists(path)
        self.dirbase = path
        self.backupdir = os.path.join(path, "backup")

        jsonpath = os.path.join(self.backupdir, 'versioninfo.json')
        if os.path.exists(jsonpath):
            with open(jsonpath, 'rt') as ifp:
                self.info = json.load(ifp)
        else:
            self.info = { 'version_number' : 0, }

        # Dirty flag, to avoid unnecessary storage writing.
        self._dirty = False

        # self.info is a dictionary object, contains various infomation.
        # We use layer 'unique-id' property as a key for self.info 
        # to get layer information, such as st_mtime, st_size and 
        # latest version number for that layer.

    def _init_version(self):
        """Init new backup version
        """
        for cpath in ('stack.xml', 
                os.path.join('Thumbnails', 'thumbnail.png')):
            srcpath = os.path.join(self.dirbase, cpath)
            if os.path.exists(srcpath):
                cpath = os.path.basename(cpath)
                # We MUST use shutil.copy here, not copy2.
                # to record the date of the version
                # created as file timestamp.
                shutil.copy(srcpath, 
                            os.path.join(self.backupdir, 
                            "%s.%d" % (cpath, self.version_num))
                           )
            else:
                logger.warning('Essential file %s does not exist.' % cpath)

    def proceed(self):
        self.info['version_number'] = self.version_num + 1
        self._init_version()

    def finalize(self):
        if self._dirty:
            jsonpath = os.path.join(self.backupdir, 'versioninfo.json')
            with open(jsonpath, 'wt') as ofp:
                json.dump(self.info, ofp)
            self._dirty = False

    # Version number related
    @property
    def version_num(self):
        return self.info['version_number']

    # File/Layer management,backup related
    def _get_srcpath(self, src):
        """Generate layer sourcefile path from project base directory. 
        if src is absolute path (it means that layer should be filebacked one)
        this method return it immidiately.
        """
        if os.path.isabs(src):
            return src
        return os.path.join(self.dirbase, src)

    def _is_already_backedup(self, layer_filename, layer_unique_id):
        """Tells whether the layer is already backed up or not.
        This method only look png file of layer.
        additional files(for example, stroke dat file) are ignored.

        Currently this method checks file modified date and size.
        To strict check, we will need hash to detect difference of files
        but it (might) takes too much processing time...?

        :rtype tuple:
        :return : a tuple of (
                boolean flag of 'backup file is same as original file',
                os.stat information of original file)
        """

        if layer_filename == None or layer_unique_id == None:
            return False

        srcpath = self._get_srcpath(layer_filename)

        # This method is called from _queue_backup_layers(),
        # and that method should be called after 
        # all current project layers saved.
        # so, every srcpath MUST exist.
        assert os.path.exists(srcpath)

        latest_bkup_info = self.info.get(layer_unique_id, None)
        statinfo = os.stat(srcpath)

        if latest_bkup_info:
            mtime, size, junk = latest_bkup_info
            return (statinfo.st_mtime == mtime and 
                    statinfo.st_size == size, statinfo)

        return (False, statinfo)

    def register_layer_info(self, id, statinfo):
        self.info[id] = (statinfo.st_mtime,
                statinfo.st_size,
                self.version_num)
        self._dirty = True

    def queue_backup_layers(self, processor, layer_stack):

        for path, cl in layer_stack.walk():
            self._queue_backup_single_layer(processor, cl)

        # Background layer is not included layer_stack.walk()
        # So call method for it.
        self._queue_backup_single_layer(
                processor, 
                layer_stack.background_layer
                )

        self.finalize()

    def _queue_backup_single_layer(self, processor, layer):
        id = layer.unique_id

        if layer.filename is not None:
            # This means 'layer has its own file instance to save'
            # (Not only FileBackedLayer, because BackgroundLayer 
            # has also backedup file in project-save)
            backed_up, statinfo = self._is_already_backedup(
                    layer.filename,
                    layer.unique_id
                    )

            if not backed_up:
                self.register_layer_info(id, statinfo)

                for sf in layer.enum_filenames():
                    targdir = os.path.join(self.backupdir, id)
                    junk, ext = os.path.splitext(sf)
                    if not os.path.exists(targdir):
                        os.mkdir(targdir)
                    # In contrast to the _init_version(),
                    # We MUST use shutil.copy2 here, not shutil.copy.
                    # It is because to ensure timestamp of the copied file is 
                    # same as original one.
                    # That timestamp(modified date) used to detect whether 
                    # a layer file has changed or not.
                    processor.add_work(
                            shutil.copy2,
                            os.path.join(self.dirbase, sf),
                            os.path.join(targdir, 
                                         "%d%s" % (self.version_num, ext)
                                        )
                            )


    def is_current_document_backedup(self):
        """Tells whether current document(in data/) is 
        same as last backup version or not.

        We need not only test 'unsaved_painting_time' property
        of model document, but also test whether current document
        on disk storage is different from last backuped one or not.

        Because, when we revert a document to some other version,
        we might overwrite data/ directory with that reverted one.
        If this happen, the document before revert would be lost
        forever.
        I think many people would not intend to lost pre-revert work.
        And it can be done easily and mistakenly.

        So we would need automatically save current document as 
        new version before revert it, but there is a problem.
        If create new version each time we revert without ask it 
        to user,there would be excessive number of pre-revert version.
        It would be annoying,confusing.
        
        At a glance, it seems to be good to ask the user every time, 
        but there are apperent cases where it is not necessary to ask. 

        It is, right after backup current document manually
        or right after reverted.

        but 'right after reverted' can be easily tested by
        referring model.project_version > 0 or not.

        Therefore, we need this method, to check whether the latest
        backedup files are same as files of data/ directory.

        see also revert_button_clicked_cb() of 
        gui.projectmanager.Versionsave.
        """

        xmlpath = os.path.join(self.dirbase, 'stack.xml')
        if not os.path.exists(xmlpath):
            return False

        doc = ET.parse(xmlpath)
        image_elem = doc.getroot()
        root_stack_elem = image_elem.find("stack")

        changed_layers = []
        Versionsave.walk_stack_xml(
                root_stack_elem,
                self.__check_single_layer_backedup,
                changed_layers
                )
        return len(changed_layers) == 0

    def __check_single_layer_backedup(self, elem, changed_layers):
        attr = elem.attrib
        backed_up, junk = self._is_already_backedup(
                attr.get('src', None),
                attr.get(Projectsaveable.ORA_LAYERID_ATTR, None)
                )

        if not backed_up:
            changed_layers.append(elem)
            return True # Exit the walk loop.Currently it is enough 
                        # only one layer is not backedup.

    def get_description(self, ver_num):
        """Get version specific user document(description)
        :param ver_num: version number
        :rtype string:
        :return : description document string for the version, 
                  it is written by end-user.
        """
        desc = self.info.get('desc', None)
        # desc should be a dictionary, whose keys are 
        # version number.
        if desc:
            # only string key allowed in json(javascript)
            ver_num = str(ver_num)
            if ver_num in desc:
                return desc[ver_num]

    def set_description(self, ver_num, doc):
        desc = self.info.get('desc', {})
        # only string key allowed in json(javascript)
        desc[str(ver_num)] = doc
        self.info['desc'] = desc
        self._dirty = True
        
    def get_visible_status(self, ver_num):
        """Get version visible status.
        This is used in ProjectManager window.
        :param ver_num: version number
        :rtype boolean:
        :return : visible state of the version.if True it is visible.
        """
        status = self.info.get('visible', {})
        if ver_num in status:
            return status[ver_num]
        else:
            return True # every version is visible as a default.

    def set_visible_status(self, ver_num, status):
        status = self.info.get('visible', {})
        # Actually, only false(hidden) flag is stored.
        # otherwise, when the version is visible, remove entry
        # from the dictionary. Then it would become visible as default behavior.
        if status == False:
            status[ver_num] = False
        elif ver_num in status:
            del status[ver_num]
        self.info['visible'] = status
        self._dirty = True

    ## Getting backedup source, stack.xml, etc from version number

    def get_version_src(self, unique_id, target_ver):
        """ Get the layer source path of assigned version
        or , if it does not exist, earlier one.
        
        :param unique_id: a string of unique uuid which is set
                          for a layer in project-save.
        :param target_ver: a integer number of version, to be reverted.
        :return : the ABSOLUTE path to suitable version file.
                  if there is no suitable one, return None.
        """
        if unique_id == None:
            return None

        basedir = os.path.join(self.backupdir, unique_id)
        if not os.path.exists(basedir):
            return None

        targfile = os.path.join(basedir,
                "%d.png" % target_ver)

        if os.path.exists(targfile):
            return targfile

        # The exactly matched file is not found.
        # but, it means we need earlier one.
        files = glob.glob(os.path.join(basedir, "*.png"))
        files.sort(reverse=True)
        # Versioned files are sorted from larger number,
        # and exactly same version as target_ver file does not exist.
        # So, the first versioned file which is lower than
        # target_ver is the file we searching.

        for cf in files:
            curbasename = os.path.basename(cf)
            try:
                cver = int(os.path.splitext(curbasename)[0])
                if cver < target_ver:
                    return cf
            except ValueError:
                pass

        return None

    def get_version_xml(self, target_ver):
        targfile = os.path.join(self.backupdir, "stack.xml.%d" % target_ver)
        if os.path.exists(targfile):
            return targfile

        return None

    def convert_xml_as_version(self, target_ver):
        """Load and modify backedup elementtree object,
        with backedup layer image.
        With this, we can revert from backup directory
        with minimum modifying lib/layer/*.py modules.

        :rtype : ElementTree document object
        :return: Modified ElementTree object.
                 If assigned version does not exist,
                 return None.
        """
        xmlfile = self.get_version_xml(target_ver)
        if not xmlfile:
            return None

        supported_strokemap_attrs = (
            lib.layer.PaintingLayer._ORA_STROKEMAP_ATTR,
            lib.layer.PaintingLayer._ORA_STROKEMAP_LEGACY_ATTR,
        )

        doc = ET.parse(self.get_version_xml(target_ver))
        image_elem = doc.getroot()
        root_stack_elem = image_elem.find("stack")

        Versionsave.walk_stack_xml(
                root_stack_elem,
                self.__convert_single_layer_element_cb,
                (supported_strokemap_attrs, target_ver)
                )

        return doc

    def __convert_single_layer_element_cb(self, elem, data):
        """Callback for converting xml 
        """
        attr = elem.attrib
        cur_uuid = attr.get(Projectsaveable.ORA_LAYERID_ATTR,
                            None)

        supported_strokemap_attrs, target_ver = data

        src = self.get_version_src(cur_uuid, target_ver)
        if src:
            # get_version_src() returns absolute path.
            # so it MUST converted into relative one.
            src = os.path.relpath(src, self.dirbase)
            attr['src'] = src
        else:
            logger.error("cannot find version %d file for layer %s" % 
                         (target_ver,
                          attr.get('name', 'unknown layer'))
                        )
            return 

        # Strokemap processing
        for atname in supported_strokemap_attrs:
            stmap = attr.get(atname, None)

            if stmap != None:
                basename, ext = os.path.splitext(src)
                attr[stmap] = '%s-strokemap.dat' % basename

    @staticmethod
    def walk_stack_xml(elem, callback, data=None):
        """Walk ElementTree and call callback for each
        element recursively.

        :param elem: the ElementTree element object.
        :param callback: callback
        :param data: user-defined data.None as default.
        """
        if elem.tag == 'stack':
            for child in elem:
                assert child != elem
                if Versionsave.walk_stack_xml(child, callback, data):
                    return True
        elif elem.tag == 'layer':
            # Exit entire walk recursive loop
            # when callback returns True
            if callback(elem, data):
                return True

if __name__ == '__main__':

    pass


