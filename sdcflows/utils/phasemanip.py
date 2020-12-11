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
    out_file = fname_presuffix(str(in_file), suffix="_rads", newpath=newpath)
    nb.Nifti1Image(data, None, hdr).to_filename(out_file)
    return out_file


def subtract_phases(in_phases, in_meta, newpath=None):
    """Calculate the phase-difference map, given two input phase maps."""
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    echo_times = tuple([m.pop("EchoTime", None) for m in in_meta])
    if echo_times[0] > echo_times[1]:
        in_phases = (in_phases[1], in_phases[0])
        in_meta = (in_meta[1], in_meta[0])
        echo_times = (echo_times[1], echo_times[0])

    in_phases_nii = [nb.load(ph) for ph in in_phases]
    sub_data = in_phases_nii[1].get_fdata(dtype="float32") - in_phases_nii[0].get_fdata(
        dtype="float32"
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
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    out_file = fname_presuffix(str(in_file), suffix="_fmap", newpath=newpath)
    image = nb.load(in_file)
    data = image.get_fdata(dtype="float32") / (2.0 * np.pi * delta_te)
    nii = nb.Nifti1Image(data, image.affine, image.header)
    nii.set_data_dtype(np.float32)
    nii.to_filename(out_file)
    return out_file


def delta_te(in_values):
    r"""
    Read :math:`\Delta_\text{TE}` from BIDS metadata dict.

    Examples
    --------
    >>> t = delta_te({"EchoTime1": 0.00522, "EchoTime2": 0.00768})
    >>> f"{t:.5f}"
    '0.00246'

    >>> t = delta_te({'EchoTimeDifference': 0.00246})
    >>> f"{t:.5f}"
    '0.00246'

    >>> delta_te({"EchoTime1": "a"})  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError:

    >>> delta_te({"EchoTime2": "a"})  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError:

    >>> delta_te({})  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError:

    >>> delta_te({"EchoTimeDifference": "a"})  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError:

    """
    te2 = in_values.get("EchoTime2")
    te1 = in_values.get("EchoTime1")

    if te2 is None and te1 is None:
        try:
            te2 = float(in_values.get("EchoTimeDifference"))
            return abs(te2)
        except TypeError:
            raise ValueError(
                "Phase/phase-difference fieldmaps: no echo-times information."
            )
        except ValueError:
            raise ValueError(
                f"Could not interpret metadata <EchoTimeDifference={te2}>."
            )
    try:
        te2 = float(te2 or "unknown")
        te1 = float(te1 or "unknown")
    except ValueError:
        raise ValueError(f"Could not interpret metadata <EchoTime(1,2)={(te1, te2)}>.")

    return abs(te2 - te1)
