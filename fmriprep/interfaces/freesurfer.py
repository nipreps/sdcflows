#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
FreeSurfer tools interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os.path as op
import nibabel as nb
from niworkflows.nipype.interfaces.base import isdefined, InputMultiPath
from niworkflows.nipype.interfaces import freesurfer as fs
from niworkflows.nipype.utils.filemanip import copyfile


class StructuralReference(fs.RobustTemplate):
    """ Variation on RobustTemplate that simply copies the source if a single
    volume is provided. """
    @property
    def cmdline(self):
        cmd = super(StructuralReference, self).cmdline
        if len(self.inputs.in_files) > 1:
            return cmd

        img = nb.load(self.inputs.in_files[0])
        if len(img.shape) > 3 and img.shape[3] > 1:
            return cmd

        out_file = self._list_outputs()['out_file']
        copyfile(self.inputs.in_files[0], out_file)
        return "echo Only one time point!"


class MakeMidthicknessInputSpec(fs.utils.MRIsExpandInputSpec):
    graymid = InputMultiPath(desc='Existing graymid/midthickness file')


class MakeMidthickness(fs.MRIsExpand):
    """ Variation on RobustTemplate that simply copies the source if a single
    volume is provided. """
    input_spec = MakeMidthicknessInputSpec

    @property
    def cmdline(self):
        cmd = super(MakeMidthickness, self).cmdline
        if not isdefined(self.inputs.graymid) or len(self.inputs.graymid) < 1:
            return cmd

        # Possible graymid values inclue {l,r}h.{graymid,midthickness}
        # Prefer midthickness to graymid, require to be of the same hemisphere
        # as input
        source = None
        in_base = op.basename(self.inputs.in_file)
        mt = self._associated_file(in_base, 'midthickness')
        gm = self._associated_file(in_base, 'graymid')

        for surf in self.inputs.graymid:
            if op.basename(surf) == mt:
                source = surf
                break
            if op.basename(surf) == gm:
                source = surf

        if source is None:
            return cmd

        return "cp {} {}".format(source, self._list_outputs()['out_file'])
