#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:29:40
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
# @Last Modified time: 2016-05-25 12:06:54
"""
Data grabbers
"""

from .utils import _get_dataset_dir, _fetch_file

def get_mni_template(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the necessary files from the mni template


    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.

    """
    if url is None:
        url = "http://googledrive.com/host/0BxI12kyv2olZdzRDUnBPYWZGZk0"

    dataset_name = 'mni_template'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)

    if _fetch_file(url, data_dir, filetype='tar', resume=resume, verbose=verbose,
                   md5sum='debfa882b8c301cd6d75dd769e73f727'):
        return data_dir
    else:
        return None

def get_ants_oasis_template(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the necessary files from the ANTs template of the OASIS dataset.


    :param str data_dir: path of the data directory. Used to force data storage
        in a non-standard location.
    :param str url: download URL of the dataset. Overwrite the default URL.

    """
    if url is None:
        url = "http://googledrive.com/host/0BxI12kyv2olZUXRWNU9aTlRvUkk"

    dataset_name = 'ants_oasis_template'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)

    if _fetch_file(url, data_dir, filetype='tar', resume=resume, verbose=verbose,
                   md5sum='34d39070b541c416333cc8b6c2fe993c'):
        return data_dir
    else:
        return None
