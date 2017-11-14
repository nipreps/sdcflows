#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""

.. figure:: _static/unwarping.svg
    :scale: 100%

    Applying field correction warp using ANTs.

One of the major problems that affects :abbr:`EPI (echo planar imaging)` data
is the spatial distortion caused by the inhomogeneity of the field inside
the scanner.
The are four broad families of methodologies for mapping the field:

  * **Phase-difference techniques**: to estimate the fieldmap, these methods
    measure the phase evolution in time between two close
    :abbr:`GRE (Gradient Recall Echo)` acquisitions. Corresponds to the sections
    8.9.1 and 8.9.2 of the BIDS specification.
  * **Direct field-mapping**: some sequences (such as :abbr:`SE (spiral echo)`)
    are able to measure the fieldmap directly. Corresponds to section 8.9.3 of BIDS.
  * **Blip-up/blip-down or phase encoding polarity**:
    :abbr:`pepolar (Phase Encoding POLARity)`: techniques acquire at least two images
    distorted due to the inhomogeneity of the field, but varying in
    :abbr:`PE (phase-encoding)` direction.
    Corresponds to 8.9.4 of BIDS.
  * **Point-spread function acquisition**: Not supported by FMRIPREP.


Once the field-map is estimated, the distortion can be accounted for.
Fieldmap processing in FMRIPREP is structured as follows:

  1. :ref:`sdc-base`: the input BIDS dataset is queried to find the available field-mapping
     techniques and the appropriate processing workflows are set-up (applies to phase-difference
     and direct field-mapping techniques).

  2. :ref:`sdc-estimation`: all the estimation workflows produce a displacement field
     ready to be used in the correction step (applies to phase-difference
     and direct field-mapping techniques).

  3. :ref:`sdc-unwarp`: the correction step is applied (for phase encoding polarity
     this step also involves distortion correction displacement field estimation).

If the dataset metadata indicate tha more than one field map acquisition is
``IntendedFor`` (see BIDS Specification section 8.9) the following priority will
be used:

  1. Blip-up/blip-down

  2. Direct field-mapping

  3. Phase-difference techniques


Additionally, FMRIPREP now experimentally supports displacement field estimation
in the absence of fieldmaps. See :ref:`fieldmapless_estimation` for
further details.

Calculating the effective echo-spacing and total-readout time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Displacement along the phase-encoding direction (:math:`d_y(x, y, z)`) is
proportional to the slice readout time (:math:`T_\\text{ro}`)
and the field inhomogeneity (:math:`\\Delta B_0(x, y, z)`)
as follows [Hutton2002]_:

  .. math ::

      d_y(x, y, z) = \\gamma \\Delta B_0(x, y, z) T_\\text{ro}

where :math:`\\gamma` is the gyromagnetic ratio.


FMRIPREP extracts :math:`T_\\text{ro}` and :math:`t_\\text{ees}` using
the following two functions:

.. autofunction:: fmriprep.interfaces.fmap.get_ees

.. autofunction:: fmriprep.interfaces.fmap.get_trt

"""

from .base import init_fmap_estimator_wf, init_fmap_unwarp_report_wf
from .unwarp import init_sdc_unwarp_wf, init_pepolar_unwarp_wf
from .syn import init_nonlinear_sdc_wf
