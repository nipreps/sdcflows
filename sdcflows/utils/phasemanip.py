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
    """Convert the input phase-difference map into a *fieldmap* in Hz."""
    import math
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    out_file = fname_presuffix(in_file, suffix="_fmap", newpath=newpath)
    image = nb.load(in_file)
    data = image.get_fdata(dtype="float32") / (2.0 * math.pi * delta_te)
    nii = nb.Nifti1Image(data, image.affine, image.header)
    nii.set_data_dtype(np.float32)
    nii.to_filename(out_file)
    return out_file
