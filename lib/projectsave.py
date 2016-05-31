#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import abc
import os
import shutil
import logging
logger = logging.getLogger(__name__)

#: A name of timestamp attribute for stack.xml
#: This is for project-save functionality
PROJECT_OLD_TIMESTAMP_ATTR = "mypaint:project-old-timestamp"
PROJECT_CURRENT_TIMESTAMP_ATTR = "mypaint:project-current-timestamp"

class Projectsaveable:
    """Mixin and abstract base for projectsaveable structures"""

    __metaclass__ = abc.ABCMeta

    def save_to_project(self, projdir, path,
                           canvas_bbox, frame_bbox, force_write, **kwargs):
        """Saves the layer(or stack) data into an project directory

        This kind of mothod saves the each layers file on 
        the raw local host computer's filesystem, 
        not in orazip or other container file.
        the file-save is done when the dirty flag of the layer
        is True, or force_write flag is True.

        With this, file-saving time drastically fasten
        even all files are marked as dirty,
        because there is no read and write I/O to pack the 
        container file.

        :param unicode projdir: the project root directory.
        :param path: the filepath.
        :param canvas_bbox: the boundary box of canvas.
        :param frame_bbox: the boundary box of frame. this can be None.
        :param boolean force_write: if this flag is True, the layer(or stack)
            should be written even it is not dirty.                                    
        :param \*\*kwargs: To be passed to underlying save routines.
        """

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
    def src(self):
        """Read-only property.This is previously recorded filename, in data/ dir.
        This property is for referring from project-save related functionality.
        
        self._src is set at derived class internally.
        so, there is no setter property.
        
        This property is copy of 'src' value of layer tag in stack.xml.
        It should be related path,something like
        'data/foobar-blablabla-bla.png'
        """
        try:
            return self._src
        except AttributeError:
            return None

    def get_filename_for_project(self, ext=u".png", formatstr=None, 
            path_prefix=None):
        """
        Get a unique filename in a project.
        :param ext: default file extension, used when uuid exists.
        :param formatstr: format strings, in unicode.
        :param path_prefix: a tuple of path components, 
                    to be added the filename with os.path.join().
        """
        retfname = None
        if self.src != None:
            basename = os.path.basename(self.src)
            if formatstr:
                basename, ext = os.path.splitext(basename)
                retfname = formatstr % (basename,)
            else:
                retfname = basename
        else:
            if formatstr:
                retfname = formatstr % (self.autosave_uuid,)
            else:
                retfname = self.autosave_uuid + ext

        if path_prefix:
            return os.path.join(os.path.join(*path_prefix), retfname)
        else:
            return retfname
    
    def _retract_old_file(self, elem, proj_dir, file_name, force_write):
        """ Retract a file(to be overwritten after this method called)
        and set a file timestamp to elementtree object. 
        This needs from project-save's version management functionality.

        All 'save_to_project' method MUST call this method
        PRIOR TO OVERWRITING ITS IMAGE FILE.

        With this file timestamp, project-save can do version-management
        for overwritten files.

        HOW WORK THE PROJECTSAVE IS ABOUT THIS METHOD:
        When a layer changed and save has done, the overwritten files are 
        retracted(moved) into the specific directory for backup.
        With this method, we mark that old layer file's timestamp into elementtree.
        
        And when reverting the picture project, the version management functionality 
        searches the old picture file from the timestamp.
        
        :param elem: the elementtree object, to be added the timestamp attribute.
        :param file_name: layer file name. this is only basename, NOT fullpath. 
        :param force_write: a flag, to tell the current writing is forced
                            (ignore dirty flag, every layer is written) or not.
        """
        if self.project_dirty or force_write:
            name = PROJECT_OLD_TIMESTAMP_ATTR
        else:
            # does not overwritten.
            name = PROJECT_CURRENT_TIMESTAMP_ATTR

        file_path = os.path.join(proj_dir, 'data', file_name)


        if os.path.exists(file_path):
            st = os.stat(file_path)
            elem.attrib[name] = str(int(st.st_mtime))

            if self.project_dirty or force_write:
                backup_path = get_project_backup_filename(
                        proj_dir, file_name, file_path)
                shutil.move(file_path, backup_path)
        else:
            logger.warning("We need a timestamp of file %s, but that file does not found.(or newly created project?)" % file_path)



def get_project_backup_filename(projdir, basename, origpath=None):
    """ Get 'prefixed' project backup filename.

    :param projdir: the existing project directory
    :param basename: the base filename of target file.
                     if this is None, it is generated from origpath.
    :param origpath: the original file absolute path (optional)

    Either basename or origpath should be assigned valid one.
    """

    assert not (origpath == basename == None)

    if origpath == None:
        origpath = os.path.join(projdir, 'data', basename) 
    assert os.path.exists(origpath)
    if basename == None:
        basename = os.path.basename(origpath)
    st = os.stat(origpath)
    return os.path.join(projdir, 'backup',
            u"%d-%s" % (int(st.st_mtime), basename))

if __name__ == '__main__':

    pass


