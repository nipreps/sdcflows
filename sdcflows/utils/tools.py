# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Image processing tools."""
import nibabel as nb


def deoblique_and_zooms(
    in_reference: nb.spatialimages.SpatialImage,
    oblique: nb.spatialimages.SpatialImage,
    factor: int = 4,
    padding: int = 1,
    factor_tol: float = 1e-4,
):
    """
    Generate a sampling reference aligned with in_reference fully covering oblique.

    Parameters
    ----------
    in_reference : :obj:`~nibabel.spatialimages.SpatialImage`
        The sampling reference.
    oblique : :obj:`~nibabel.spatialimages.SpatialImage`
        The rotated coordinate system whose extent should be fully covered by the
        generated reference.
    factor : :obj:`int`
        A factor to increase the resolution of the generated reference.
    padding : :obj:`int`
        Number of additional voxels around the most extreme positions of the projection of
        oblique on to the reference.
    factor_tol : :obj:`float`
        Absolute tolerance to determine whether factor is one.

    """
    from itertools import product
    import numpy as np
    from nibabel.affines import apply_affine, rescale_affine

    # Reference space metadata
    hdr = in_reference.header.copy()
    affine = in_reference.affine.copy()
    ref_shape = np.array(in_reference.shape[:3])
    ref_zooms = np.array(hdr.get_zooms()[:3])
    _, scode = in_reference.get_sform(coded=True)
    _, qcode = in_reference.get_qform(coded=True)

    # Calculate the 8 most extreme coordinates of oblique in in_reference space
    corners = np.array(list(product((0, 1), repeat=3))) * (
        np.array(oblique.shape[:3]) - 1
    )
    extent_ijk = apply_affine(np.linalg.inv(affine) @ oblique.affine, corners)

    underflow = np.clip(extent_ijk.min(0) - padding, None, 0).astype(int)
    overflow = np.ceil(
        np.clip(extent_ijk.max(0) + padding + 1 - ref_shape, 0, None)
    ).astype(int)
    if np.any(underflow < 0) or np.any(overflow > 0):
        # Add under/overflow voxels
        ref_shape += overflow - underflow
        # Consistently update origin
        affine[:-1, -1] = apply_affine(affine, underflow)

    # Make grid denser
    if abs(1.0 - factor) > factor_tol:
        new_shape = np.rint(ref_shape * factor)
        affine = rescale_affine(affine, ref_shape, ref_zooms / factor, new_shape)
        ref_shape = new_shape

    # Generate new reference
    hdr.set_sform(affine, scode)
    hdr.set_qform(affine, qcode)

    return in_reference.__class__(
        nb.fileslice.strided_scalar(ref_shape.astype(int)),
        affine,
        hdr,
    )


def resample_to_zooms(in_file, zooms, order=3, prefilter=True):
    """Resample the input data to a new grid with the requested zooms."""
    from pathlib import Path
    import numpy as np
    import nibabel as nb
    from nibabel.affines import rescale_affine
    from nitransforms.linear import Affine

    if isinstance(in_file, (str, Path)):
        in_file = nb.load(in_file)

    # Prepare output x-forms
    sform, scode = in_file.get_sform(coded=True)
    qform, qcode = in_file.get_qform(coded=True)

    hdr = in_file.header.copy()
    zooms = np.array(zooms)

    pre_zooms = np.array(in_file.header.get_zooms()[:3])
    # Could use `np.ceil` if we prefer
    new_shape = np.rint(np.array(in_file.shape[:3]) * pre_zooms / zooms)
    affine = rescale_affine(in_file.affine, in_file.shape[:3], zooms, new_shape)

    # Generate new reference
    hdr.set_sform(affine, scode)
    hdr.set_qform(affine, qcode)
    newref = in_file.__class__(
        np.zeros(new_shape.astype(int), dtype=hdr.get_data_dtype()),
        affine,
        hdr,
    )

    # Resample via identity transform
    return Affine(reference=newref).apply(in_file, order=order, prefilter=prefilter)


def ensure_positive_cosines(img):
    """
    Reorient axes polarity to have all positive direction cosines.

    In other words, this interface will reorient the image polarities to be all
    *positive*, respecting the axes ordering.
    For instance, *LAS* -> *RAS* or *ALS* -> *ARS*.

    """
    import nibabel as nb

    in_ornt = nb.io_orientation(img.affine)
    img_axcodes = nb.orientations.ornt2axcodes(in_ornt)
    out_ornt = in_ornt.copy()
    out_ornt[:, 1] = 1
    ornt_xfm = nb.orientations.ornt_transform(in_ornt, out_ornt)

    return img.as_reoriented(ornt_xfm), img_axcodes


def brain_masker(in_file, out_file=None, padding=5):
    """Use grayscale morphological operations to obtain a quick mask of EPI data."""
    from pathlib import Path
    import re
    import nibabel as nb
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import ball
    from skimage.filters import threshold_otsu
    from skimage.segmentation import random_walker

    # Load data
    img = nb.load(in_file)
    data = np.pad(img.get_fdata(dtype="float32"), padding)
    hdr = img.header.copy()

    # Cleanup background and invert intensity
    data[data < np.percentile(data[data > 0], 15)] = 0
    data[data > 0] -= data[data > 0].min()
    datainv = -data.copy()
    datainv -= datainv.min()
    datainv /= datainv.max()

    # Grayscale closing to enhance CSF layer surrounding the brain
    closed = ndimage.grey_closing(datainv, structure=ball(1))
    denoised = ndimage.median_filter(closed, footprint=ball(3))
    th = threshold_otsu(denoised)

    # Rough binary mask
    closedbin = np.zeros_like(closed)
    closedbin[closed < th] = 1
    closedbin = ndimage.binary_opening(closedbin, ball(3)).astype("uint8")

    label_im, nb_labels = ndimage.label(closedbin)
    sizes = ndimage.sum(closedbin, label_im, range(nb_labels + 1))
    mask = sizes == sizes.max()
    closedbin = mask[label_im]
    closedbin = ndimage.binary_closing(closedbin, ball(5)).astype("uint8")

    # Prepare markers
    markers = np.ones_like(closed, dtype="int8") * 2
    markers[1:-1, 1:-1, 1:-1] = 0
    closedbin_dil = ndimage.binary_dilation(closedbin, ball(5))
    markers[closedbin_dil] = 0
    closed_eroded = ndimage.binary_erosion(closedbin, structure=ball(5))
    markers[closed_eroded] = 1

    # Run random walker
    closed[closed > 0.0] -= closed[closed > 0.0].min()
    segtarget = (2 * closed / closed.max()) - 1.0
    labels = random_walker(
        segtarget, markers, spacing=img.header.get_zooms()[:3], return_full_prob=True
    )[..., padding:-padding, padding:-padding, padding:-padding]

    out_mask = Path(out_file or "brain_mask.nii.gz").absolute()

    hdr.set_data_dtype("uint8")
    img.__class__((labels[0, ...] >= 0.5).astype("uint8"), img.affine, hdr).to_filename(
        out_mask
    )

    out_probseg = re.sub(
        r"\.nii(\.gz)$", r"_probseg.nii\1", str(out_mask).replace("_mask.", ".")
    )
    hdr.set_data_dtype("float32")
    img.__class__((labels[0, ...]), img.affine, hdr).to_filename(out_probseg)

    out_brain = re.sub(
        r"\.nii(\.gz)$", r"_brainmasked.nii\1", str(out_mask).replace("_mask.", ".")
    )
    data = np.asanyarray(img.dataobj)
    data[labels[0, ...] < 0.5] = 0
    img.__class__(data, img.affine, img.header).to_filename(out_brain)

    return str(out_brain), str(out_probseg), str(out_mask)


def reorient_pedir(pe_dir, source_ornt, target_ornt=None):
    """Return updated PhaseEncodingDirection to account for an image data array rotation

    Orientations form a natural group with identity, product and inverse.
    This function thus has the following properties (here ``e`` is the identity,
    ``*`` the product and ``-`` the inverse; ``a`` and ``b`` are arbitrary orientations):

        reorient(pe_dir, e, e) == pe_dir
        reorient(pe_dir, a, e) == reorient(pe_dir, a)
        reorient(pe_dir, e, b) == reorient(pe_dir, -b)
        reorient(pe_dir, a, b) == reorient(pe_dir, a * -b, e)

    Therefore, this function accepts one or two orientations, and precomputed transforms
    from a to b can be passed as the "source" orientation.

    Passing only a source orientation will rotate into RAS:

    >>> from nibabel.orientations import axcodes2ornt
    >>> reorient_pedir("j", axcodes2ornt("RAS"))
    'j'
    >>> reorient_pedir("i", axcodes2ornt("PSL"))
    'j-'

    Passing only a target_ornt will rotate from RAS:

    >>> reorient_pedir("j", source_ornt=None, target_ornt=axcodes2ornt("RAS"))
    'j'
    >>> reorient_pedir("i", source_ornt=None, target_ornt=axcodes2ornt("PSL"))
    'k-'

    Passing both will rotate from source to target.

    >>> reorient_pedir("j", axcodes2ornt("LPS"), axcodes2ornt("AIR"))
    'i-'

    Passing a transform orientation as source_ornt will perform the transform,
    and passing it as ``target_ornt`` will invert the transform:

    >>> from nibabel.orientations import ornt_transform
    >>> xfm = ornt_transform(axcodes2ornt("LPS"), axcodes2ornt("AIR"))
    >>> reorient_pedir("j", xfm)
    'i-'
    >>> reorient_pedir("j", source_ornt=None, target_ornt=xfm)
    'k-'
    """
    from nibabel.orientations import ornt_transform

    if source_ornt is None:
        source_ornt = [[0, 1], [1, 1], [2, 1]]
    if target_ornt is None:
        target_ornt = [[0, 1], [1, 1], [2, 1]]

    xfm = ornt_transform(source_ornt, target_ornt).astype(int)  # shape: (3, 2)

    directions = "ijk" if pe_dir[0] in "ijk" else "xyz"

    source_axis = directions.index(pe_dir[0])
    source_flip = pe_dir[1:] == "-"

    axis_xfm = xfm[source_axis, :]  # shape: (2,)

    target_axis = directions[axis_xfm[0]]
    target_flip = source_flip ^ (axis_xfm[1] == -1)

    return f"{target_axis}-" if target_flip else target_axis
