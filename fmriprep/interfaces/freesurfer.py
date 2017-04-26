#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
FreeSurfer tools interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division, absolute_import, unicode_literals

from nipype.interfaces import freesurfer as fs


class StructuralReference(fs.RobustTemplate):
    """ Variation on RobustTemplate that simply copies the source if a single
    volume is provided. """
    @property
    def cmdline(self):
        import nibabel as nb
        from nipype.utils.filemanip import copyfile
        cmd = super(StructuralReference, self).cmdline
        if len(self.inputs.in_files) > 1:
            return cmd

        img = nb.load(self.inputs.in_files[0])
        if len(img.shape) > 3 and img.shape[3] > 1:
            return cmd

        out_file = self._list_outputs()['out_file']
        copyfile(self.inputs.in_files[0], out_file)
        return "echo Only one time point!"
