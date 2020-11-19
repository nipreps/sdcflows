# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utilities to manipulate phase and phase difference maps."""


def au2rads(in_file, newpath=None):
    """Convert the input phase map in arbitrary units (a.u.) to rads."""
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    im = nb.load(in_file)
    data = im.get_fdata(caching="unchanged")  # Read as float64 for safety
    hdr = im.header.copy()

    # Rescale to [0, 2*pi]
    data = (data - data.min()) * (2 * np.pi / (data.max() - data.min()))

    # Round to float32 and clip
    data = np.clip(np.float32(data), 0.0, 2 * np.pi)

    hdr.set_data_dtype(np.float32)
    hdr.set_xyzt_units("mm")
    out_file = fname_presuffix(in_file, suffix="_rads", newpath=newpath)
    nb.Nifti1Image(data, None, hdr).to_filename(out_file)
    return out_file


def subtract_phases(in_phases, in_meta, newpath=None):
    """Calculate the phase-difference map, given two input phase maps."""
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    echo_times = tuple([m.pop("EchoTime", None) for m in in_meta])
    if not all(echo_times):
        raise ValueError(
            "One or more missing EchoTime metadata parameter "
            "associated to one or more phase map(s)."
        )

    if echo_times[0] > echo_times[1]:
        in_phases = (in_phases[1], in_phases[0])
        in_meta = (in_meta[1], in_meta[0])
        echo_times = (echo_times[1], echo_times[0])

    in_phases_nii = [nb.load(ph) for ph in in_phases]
    sub_data = (
        in_phases_nii[1].get_fdata(dtype="float32")
        - in_phases_nii[0].get_fdata(dtype="float32")
    )

    # wrap negative radians back to [0, 2pi]
    sub_data[sub_data < 0] += 2 * np.pi
    sub_data = np.clip(sub_data, 0.0, 2 * np.pi)

    new_meta = in_meta[1].copy()
    new_meta.update(in_meta[0])
    new_meta["EchoTime1"] = echo_times[0]
    new_meta["EchoTime2"] = echo_times[1]

    hdr = in_phases_nii[0].header.copy()
    hdr.set_data_dtype(np.float32)
    hdr.set_xyzt_units("mm")
    nii = nb.Nifti1Image(sub_data, in_phases_nii[0].affine, hdr)
    out_phdiff = fname_presuffix(in_phases[0], suffix="_phdiff", newpath=newpath)
    nii.to_filename(out_phdiff)
    return out_phdiff, new_meta


def phdiff2fmap(in_file, delta_te, newpath=None):
    r"""
    Convert the input phase-difference map into a fieldmap in Hz.

    Uses eq. (1) of [Hutton2002]_:

    .. math::

        \Delta B_0 (i, j, k) = \frac{\Delta \Theta (i, j, k)}{2\pi\gamma \, \Delta\text{TE}}

    where :math:`\Delta B_0 (i, j, k)` is the *fieldmap* in Hz,
    :math:`\Delta \Theta (i, j, k)` is the phase-difference map in rad,
    :math:`\gamma` is the gyromagnetic ratio of the H proton,
    and :math:`\Delta\text{TE}` is the elapsed time between the two GRE echoes.


    We can obtain a voxel displacement map following eq. (2) of the same paper:

    .. math::

        d_\text{PE} (i, j, k) = \gamma \, \Delta B_0 (i, j, k) \, T_\text{ro}

    where :math:`T_\text{ro}` is the readout time of one slice of the EPI dataset
    we want to correct for distortions, and
    :math:`\Delta_\text{PE} (i, j, k)` is the *voxel-shift map* (VSM) along the *PE*
    direction.

    Replacing (1) into (2), and eliminating the scaling effect of :math:`T_\text{ro}`,
    we obtain the *voxel-shift-velocity map* (voxels/ms) which can be then used to
    recover the actual displacement field of the target EPI dataset.

    .. math::

        v(i, j, k) = \frac{\Delta \Theta (i, j, k)}{2\pi \, \Delta\text{TE}}

    References
    ----------
    .. [Hutton2002] Hutton et al., Image Distortion Correction in fMRI: A Quantitative
      Evaluation, NeuroImage 16(1):217-240, 2002. doi:`10.1006/nimg.2001.1054
      <https://doi.org/10.1006/nimg.2001.1054>`_.


    """
    import math
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    #  GYROMAG_RATIO_H_PROTON_MHZ = 42.576

    out_file = fname_presuffix(in_file, suffix="_fmap", newpath=newpath)
    image = nb.load(in_file)
    data = image.get_fdata(dtype="float32") / (2.0 * math.pi * delta_te)
    nii = nb.Nifti1Image(data, image.affine, image.header)
    nii.set_data_dtype(np.float32)
    nii.to_filename(out_file)
    return out_file
