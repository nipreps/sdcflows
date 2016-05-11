#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The fmriprep reporting engine for visual assessment
"""

from .pdf_compose import generate_report
from .pipeline_reports import (anatomical_overlay, parcel_overlay,
                               stripped_brain_overlay, generate_report_workflow)
