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

    img_axcodes = nb.aff2axcodes(img.affine)
    in_ornt = nb.orientations.axcodes2ornt(img_axcodes)
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
