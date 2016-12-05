#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Fieldmap-processing workflows.

"""
from fmriprep.workflows.fieldmap.utils import create_encoding_file, mcflirt2topup
from fmriprep.workflows.fieldmap.phase_diff_and_magnitudes import phase_diff_and_magnitudes
from fmriprep.workflows.fieldmap.unwarp import sdc_unwarp