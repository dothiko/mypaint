#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import abc
import os
import shutil
import logging
logger = logging.getLogger(__name__)
import glob
import sys

import lib.autosave

if sys.platform.startswith('linux'):
    HARDLINK_USEABLE = True
else:
    logger.warning('Projectsave cannot utilize hard link in this system.File-copy used instead of hard link.')
    HARDLINK_USEABLE = False

class Projectsaveable(lib.autosave.Autosaveable):
    """Mixin and abstract base for projectsaveable structures"""

    __metaclass__ = abc.ABCMeta

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
        
    def save_to_project(self, projdir, backupdir, path,
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

        :param projdir: the project root directory.
        :param backupdir : the backup directory.
        :param path: the filepath.
        :param canvas_bbox: the boundary box of canvas.
        :param frame_bbox: the boundary box of frame. this can be None.
        :param boolean force_write: if this flag is True, the layer(or stack)
            should be written even it is not dirty.                                    
        :param \*\*kwargs: To be passed to underlying save routines.
        """

    def load_from_project_dir(self, projdir, elem, cache_dir, feedback_cb,
                                 x=0, y=0, **kwargs):
        """
        Load layer(or stack) from a directory.

        This mainly utilize load_from_openraster_dir internally.
        This method is created for future use, currently merely
        call load_from_openraster_dir.
        """
        self.load_from_openraster_dir(projdir, elem, cache_dir, feedback_cb,
                                x=x, y=y, **kwargs)

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

    def get_filenames_for_project(self):
        """
        Get a list of filenames which consists this layer as a
        generator function.
        """
        yield self._get_source_filename()

        if hasattr(self,'workfilename'):
            yield self.workfilename
        elif hasattr(self,'_ORA_STROKEMAP_ATTR'):
            yield self._get_source_filename(
                    ext=None,
                    formatstr=u"%s-strokemap.dat")

        raise StopIteration

    def _get_source_filename(self, ext=u".png", formatstr=None, 
            path_prefix=None):
        """
        Get a source filename of this layer.
        :param ext: default file extension, used when uuid exists.
        :param formatstr: format strings, in unicode. 
                          when this argument is used,
                          ext argument should be ignored.
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

    def backup(self, backupdir, sourcedir, move_file=False):
        """ Link or Copy layer entity(i.e. png file and strokemap or something) 
        into the backup directory.

        :param backupdir: The destination directory of backup
        :param sourcedir: The source directory, it is project directory.
        :param move_file: Flag to indicate to move file, not copy. 
                            This flag is used when the layer is dirty,
                            and old layer contents should be moved as 
                            a backup.
        """
        for cpath in self.get_filenames_for_project():
            if cpath:
                cpath = os.path.join(sourcedir, 'data', cpath)
                if os.path.exists(cpath):
                    if move_file:
                        logger.info('moving file %s.' % cpath)
                        shutil.move(cpath, backupdir)
                    else:
                        link_backup(cpath, backupdir)
                else:
                    logger.warning('file %s does not exist in lib.projectsave.do_backup()' % cpath)




## Functions

def do_backup(targets, backupdir, sourcedir):
    """ Copy layer files to backup directory.
    """
    for cl in targets:
        cl.backup(backupdir, sourcedir)

def init_backup(filepath, backupdir):
    """ Called prior to project-dirty (or forced) write.
    This function cuts hard-link between current file and backup one,
    to avoid overwrite hard-linked backups.

    When this function called on some environment which does not
    support hard link, this function do nothing.

    :param filepath: The absolute path of file.
    :param backupdir: The backup destination, currently unused.
    """
    if HARDLINK_USEABLE:
        if (os.path.exists(filepath) and
                os.stat(filepath).st_nlink > 1):
            os.unlink(filepath)
            # Checking link count ,for safety.

    # Otherwise(the system does not support hard-link in python), 
    # do nothing.

def link_backup(src, dst):
    """ Link file from src to dst.
    If the platform does not support hard-link,
    this function copy it instead of link.

    :param src: Path of the original file.
    :param dst: Destination directory.NOT INCLUDING FILENAME.
    """

    if HARDLINK_USEABLE:
        logger.info('linking file %s to %s.' % (src, dst))
        try:
            os.link(src, 
                    os.path.join(dst, 
                        os.path.basename(src))
                   )
            return

        except OSError:
            logger.warning('It is cross-device link.No hard link executed.')
            # Fallthrough.

    # Otherwise, copy the file.
    shutil.copy(src, dst)

if __name__ == '__main__':

    pass


