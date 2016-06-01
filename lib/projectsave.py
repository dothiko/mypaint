#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import abc
import os
import shutil
import logging
logger = logging.getLogger(__name__)
import glob

import lib.autosave

#: A name of timestamp attribute for stack.xml
#: This is for project-save functionality
PROJECT_REVISION_ATTR = "mypaint:project-revision"
PROJECT_FILE_REVISION_ATTR = "mypaint:project-file-revision"

class Projectsaveable(lib.autosave.Autosaveable):
    """Mixin and abstract base for projectsaveable structures"""

    __metaclass__ = abc.ABCMeta

    @property
    def autosave_dirty(self):
        return super(Projectsaveable, self).autosave_dirty()

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

    def load_from_project_dir(self, projdir, elem, cache_dir, feedback_cb,
                                 x=0, y=0, **kwargs):
        """
        Load layer(or stack) from a directory.

        This mainly utilize load_from_openraster_dir internally.
        """
        self.load_from_openraster_dir(projdir, elem, cache_dir, feedback_cb,
                                x=x, y=y, **kwargs)
        self.decode_revision(elem, kwargs['project_revision'])

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

    @property
    def revision(self):
        """ The revision of source file.
        """
        try:
            return self.__revision
        except AttributeError:
            return None

    def decode_revision(self, elem, project_rev):
        """ Decode revision number of source file
        from ElementTree.

        :param project_rev: the default project revision.
        """
        self.__revision = elem.attrib.get(PROJECT_FILE_REVISION_ATTR, project_rev)
        print self.__revision

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
    
    def _process_old_file(self, elem, proj_dir, backup_dir, 
            project_revision, file_name, force_write):
        """ Retract a 'old' file(to be overwritten after this method called)
        into backup directory if needed, and also set a file revision to 
        an attribute of elementtree object.
        This attribute needs from project-save's version management functionality.

        All the 'save_to_project' method, which (might) overwrites existing
        layer png, MUST call this method BEFORE OVERWRITING ITS IMAGE FILE.
        (actually, such method would call _save_rect_to_project(), 
         so _save_rect_to_project() do it instead of that methods automatically.)

        With this file revision string, project-save can do version management.

        when reverting the picture project, the version management functionality 
        searches the needed old picture file from the revision number.
        
        :param elem: the elementtree object, to be added the timestamp attribute.
                     this should be contains the attribute 'PROJECT_REVISION_ATTR'
        :param backup_dir: the backup directory, it must be prefixed by generation of 
                            backup. if this is None, this method exits immidiately.
        :param file_name: layer file name. this is only basename, NOT fullpath. 
        :param force_write: a flag, to tell the current writing is forced
                            (ignore dirty flag, every layer is written) or not.
        """


        if self.project_dirty or force_write:
            assert project_revision != None
            elem.attrib[PROJECT_FILE_REVISION_ATTR] = project_revision
            self.__revision = project_revision
        else:
            # does not overwritten.
            assert self.revision != None
            elem.attrib[PROJECT_FILE_REVISION_ATTR] = self.revision

        # When backup_dir is None, it means
        # 'This project is completely new one'
        # so nothing to be backuped, exit now.
        if backup_dir == None:
            return

        file_path = os.path.join(proj_dir, 'data', file_name)
        if os.path.exists(file_path):
            if self.project_dirty or force_write:
                backup_path = get_project_backup_filename(
                        backup_dir, file_name, 
                        orig_path=file_path, revision = self.revision)
                shutil.move(file_path, backup_path)
       #else:
       #    logger.warning("We need a timestamp of file %s, but that file does not found.(or newly created project?)" % file_path)

class BackupManager(object):

    """ Backup file cleanup manager class.
    This class manage the generations of file at backup file pool.
    If a file is not referred from any generation of stack.xml,
    (i.e not referred from even oldest stack.xml)
    that file is deleted.

    This class should runs as queued idling task, so processing
    workload should be minimum...
    """

    def __init__(self, backup_dir, max_gen):
        self._backup_dir = backup_dir
        self._max_gen = max_gen

    def _query_xml(self, stack_xml_path):
        """ Query a stack.xml and return the file revisions.
        """
        doc = ET.parse(stack_xml_path)
        elem = doc.getroot()
        files_rev = {}
        root_stack_elem = image_elem.find("stack")

        def _decode_walk(elem, files_rev):
            if elem.tag != "stack":
                name = elem.attrib['src']
                rev = elem.attrib.get(PROJECT_FILE_REVISION_ATTR, None)
                files_rev[name] = rev
                return 
            else:
                for child_elem in elem.findall("./*"):
                    _decode_recursive(elem, files_rev)

        _decode_walk(root_stack_elem, files_rev)
        return files_rev

    def _get_older_paths(self, filesrc, revision):
        basename = '-'.join(os.path.basename(filesrc).split('-')[1:])
        glob_basename = "*-%s" % basename
        files = sorted(
                glob.glob(os.path.join(self._backup_dir, glob_basename))
                )
        return files[:files.index(srcname)]

    def execute(self):
        logger.info('generation management starts.') 
        xmls = sorted(
                glob.glob(os.path.join(self._backup_dir, '*_stack.xml'))
                )

        if 2 <= len(xmls) < self._max_gen:
            target_gen_idx = len(xmls) - self._max_gen
            second_oldest_xml = xmls[target_gen_idx]
            files_rev = self._query_xml(second_oldest_xml)
            for ck in files_rev:
                logger.info('Against a backup file %s...' % ck)
                for cpath in self._get_older_paths(ck, files_rev[ck]):
                    if os.path.exists(cpath):
                       #os.unlink(cpath)
                        logger.info('older backup file %s is removed' % cpath)
                        
            for oldxml in xmls[:target_gen_idx]:
               #os.unlink(oldest_xml)
                logger.info('older xml file %s is removed' % oldxml)
        else:
            logger.info('there is no files to be removed at %s.' %
                    self._backup_dir) 

        logger.info('generation management end.') 


## Functions

def clear_old_backup(backupdir, max_gen=5):
    """ clear old backup generation.
    """
    bm = BackupManager(backupdir, max_gen)
    bm.execute()

def get_project_backup_filename(backupdir, basename, origpath=None, revision=None):
    """ Get 'prefixed' project backup filename.
    Either basename or origpath should be assigned valid one.

    CAUTION:
    * Either basename or origpath should be assigned valid one.
    * The path that generates with 
      os.path.join(projdir, 'backup', backup_timestamp) 
      MUST exist prior to call this function.

    :param projdir: the existing project directory
    :param basename: the base filename of target file.
                     if this is None, it is generated from origpath.
    :param str backup_timestamp: the backup timestamp, 
                     this indicates the generation of backup.
    :param origpath: the original file absolute path (optional)
    :param str revision: the revision of file (optional)
                          this is for when using a exactly same prefix
                          between multiple files.
    :rtype str: the fullpath of backup file.
    """

    assert not (origpath == basename == None)

    if origpath == None:
        origpath = os.path.join(projdir, 'data', basename) 
    assert os.path.exists(origpath)
    if basename == None:
        basename = os.path.basename(origpath)
    return os.path.join(backupdir, 
            u"%s-%s" % (revision, basename))

if __name__ == '__main__':

    pass


