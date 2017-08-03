#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Miscelaneous utilities
"""


def fix_multi_T1w_source_name(in_files):
    """
    Make up a generic source name when there are multiple T1s
    """
    import os
    if not isinstance(in_files, list):
        return in_files
    subject_label = in_files[0].split(os.sep)[-1].split("_")[0].split("-")[-1]
    base, _ = os.path.split(in_files[0])
    return os.path.join(base, "sub-%s_T1w.nii.gz" % subject_label)


def add_suffix(in_files, suffix):
    """
    Wrap nipype's fname_presuffix to conveniently just add a prefix
    """
    import os.path as op
    from niworkflows.nipype.utils.filemanip import fname_presuffix, filename_to_list
    return op.basename(fname_presuffix(filename_to_list(in_files)[0],
                                       suffix=suffix))


if __name__ == '__main__':
    pass
