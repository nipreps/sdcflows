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
"""Basic miscellaneous utilities."""
import logging
from scipy.stats import mode
from scipy.ndimage import gaussian_filter
from typing import Tuple, cast

import numpy as np
import numpy.typing as npt


def front(inlist):
    """
    Pop from a list or tuple, otherwise return untouched.

    Examples
    --------
    >>> front([1, 0])
    1

    >>> front("/path/somewhere")
    '/path/somewhere'

    """
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist


def last(inlist):
    """
    Return the last element from a list or tuple, otherwise return untouched.

    Examples
    --------
    >>> last([1, 0])
    0

    >>> last("/path/somewhere")
    '/path/somewhere'

    """
    if isinstance(inlist, (list, tuple)):
        return inlist[-1]
    return inlist


def get_free_mem():
    """Probe the free memory right now."""
    try:
        from psutil import virtual_memory

        return round(virtual_memory().free, 1)
    except Exception:
        return None


def create_logger(name: str, level: int = 40) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # clear any existing handlers
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setLevel(level)
    # formatter = logging.Formatter('[%(name)s %(asctime)s] - %(levelname)s: %(message)s')
    formatter = logging.Formatter('[%(name)s - %(levelname)s]: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def get_largest_connected_component(mask_data: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Get the largest connected component of a mask

    Parameters
    ----------
    mask_data : npt.NDArray[np.bool_]
        Mask to get the largest connected component of

    Returns
    -------
    npt.NDArray[np.bool_]
        Mask with only the largest connected component
    """
    from skimage.measure import label, regionprops

    # get the largest connected component
    labelled_mask = label(mask_data)
    props = regionprops(labelled_mask)
    sorted_props = sorted(props, key=lambda x: x.area, reverse=True)
    mask_data = labelled_mask == sorted_props[0].label

    # return the mask
    return mask_data


def medic_automask(mag_file, voxel_quality, echo_times, automask_dilation, out_file="mask.nii.gz"):
    import os
    from typing import cast

    import nibabel as nb
    import numpy as np
    import numpy.typing as npt
    from scipy.ndimage import (
        binary_dilation,
        binary_erosion,
        binary_fill_holes,
        generate_binary_structure,
    )
    from skimage.filters import threshold_otsu

    from sdcflows.utils.misc import create_brain_mask, get_largest_connected_component

    out_file = os.path.abspath(out_file)

    mag_img = nb.load(mag_file)

    vq = nb.load(voxel_quality).get_fdata()
    vq[np.isnan(vq)] = 0  # I (TS) am seeing NaNs in this file, which breaks things.
    vq_mask = vq > threshold_otsu(vq)
    strel = generate_binary_structure(3, 2)
    vq_mask = binary_fill_holes(vq_mask, strel).astype(bool)
    # get largest connected component
    vq_mask = get_largest_connected_component(vq_mask)

    # combine masks
    brain_mask_file = create_brain_mask(mag_file)
    brain_mask = nb.load(brain_mask_file).get_fdata().astype(bool)
    combined_mask = brain_mask | vq_mask
    combined_mask = get_largest_connected_component(combined_mask)

    # erode then dilate
    combined_mask = cast(npt.NDArray[np.bool_], binary_erosion(combined_mask, strel, iterations=2))
    combined_mask = get_largest_connected_component(combined_mask)
    combined_mask = cast(
        npt.NDArray[np.bool_],
        binary_dilation(combined_mask, strel, iterations=2),
    )

    # get a dilated verision of the mask
    combined_mask_dilated = cast(
        npt.NDArray[np.bool_],
        binary_dilation(combined_mask, strel, iterations=automask_dilation),
    )

    # get sum of masks (we can select dilated vs original version by indexing)
    mask_data_select = combined_mask.astype(np.int8) + combined_mask_dilated.astype(np.int8)

    # let mask_data be the dilated version
    mask_data = mask_data_select > 0

    mask_img = nb.Nifti1Image(mask_data, mag_img.affine, mag_img.header)
    mask_img.to_filename(out_file)

    return out_file, None


def calculate_diffs(mag_file, phase_file):
    """Calculate magnitude and phase differences between first two echoes.

    Parameters
    ----------
    mag_file : str
        Magnitude image file, with shape (x, y, z, n_echoes).
    phase_file : str
        Phase image file, with shape (x, y, z, n_echoes).

    Returns
    -------
    mag_diff_file : str
        Magnitude difference between echoes 2 and 1, with shape (x, y, z).
    phase_diff_file : str
        Phase difference between echoes 2 and 1, with shape (x, y, z).

    Notes
    -----
    This calculates a signal difference map between the first two echoes using both the
    magnitude and phase data,
    then separates the magnitude and phase differences out.
    """
    import os

    import nibabel as nb
    import numpy as np

    mag_img = nb.load(mag_file)
    phase_img = nb.load(phase_file)
    mag_data = mag_img.get_fdata()
    phase_data = phase_img.get_fdata()
    signal_diff = (
        mag_data[..., 0]
        * mag_data[..., 1]
        * np.exp(1j * (phase_data[..., 1] - phase_data[..., 0]))
    )
    mag_diff = np.abs(signal_diff)
    phase_diff = np.angle(signal_diff)
    mag_diff_img = nb.Nifti1Image(mag_diff, mag_img.affine, mag_img.header)
    phase_diff_img = nb.Nifti1Image(phase_diff, phase_img.affine, phase_img.header)
    mag_diff_file = os.path.abspath("mag_diff.nii.gz")
    phase_diff_file = os.path.abspath("phase_diff.nii.gz")
    mag_diff_img.to_filename(mag_diff_file)
    phase_diff_img.to_filename(phase_diff_file)
    return mag_diff_file, phase_diff_file


def compute_offset(echo_ind: int, W: npt.NDArray, X: npt.NDArray, Y: npt.NDArray) -> int:
    """Method for computing the global mode offset for echoes > 1.

    TODO: Rename this and calculate_offset.

    Parameters
    ----------
    echo_ind : int
        Echo index
    W : npt.NDArray
        Weights
    X : npt.NDArray
        TEs in 2d matrix
    Y : npt.NDArray
        Masked unwrapped data weighted by magnitude

    Returns
    -------
    best_offset: int
    """
    # fit the model to the up to previous echo
    coefficients, _ = weighted_regression(X[:echo_ind], Y[:echo_ind], W[:echo_ind])

    # compute the predicted phase for the current echo
    Y_pred = X[echo_ind] * coefficients

    # compute the difference between the predicted phase and the unwrapped phase
    Y_diff = Y_pred - Y[echo_ind]

    # compute closest multiple of 2pi to the difference
    int_map = np.round(Y_diff / (2 * np.pi)).astype(int)

    # compute the most often occuring multiple
    best_offset = mode(int_map, axis=0, keepdims=False).mode
    best_offset = cast(int, best_offset)

    return best_offset


def weighted_regression(
    X: npt.NDArray,
    Y: npt.NDArray,
    W: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Single parameter weighted regression.

    This function does a single parameter weighted regression using elementwise matrix operations.

    Parameters
    ----------
    X : npt.NDArray
        Design matrix, rows are each echo, columns are voxels.
        If column is size 1, this array will be broadcasted across all voxels.
    Y : npt.NDArray
        Ordinate matrix, rows are each echo, columns are voxels.
        (This is meant for phase values)
    W : npt.NDArray
        Weight matrix (usually the magnitude image) echos x voxels

    Returns
    -------
    npt.NDArray
        model weights
    npt.NDArray
        residuals
    """
    # compute weighted X and Y
    WY = W * Y
    WX = W * X

    # now compute the weighted squared magnitude of the X
    sq_WX = np.sum(WX**2, axis=0, keepdims=True)

    # get the inverse X
    inv_WX = np.zeros(WX.shape)
    np.divide(WX, sq_WX, out=inv_WX, where=(sq_WX != 0))

    # compute the weights
    model_weights = (inv_WX * WY).sum(axis=0)

    # now compute residuals (on unweighted data)
    residuals = np.sum((Y - model_weights * X) ** 2, axis=0)

    # return model weights and residuals
    return model_weights, residuals


def calculate_diffs2(magnitude, phase):
    """Calculate the magnitude and phase differences between two complex-valued images.

    Parameters
    ----------
    magnitude : :obj:`str`
        The path to the magnitude image (concatenated across echoes).
    phase : :obj:`str`
        The path to the phase image (concatenated across echoes).

    Returns
    -------
    mag_diff_file : :obj:`str`
        The path to the magnitude difference image.
    phase_diff_file : :obj:`str`
        The path to the phase difference image.
    """
    import os

    import nibabel as nb
    import numpy as np

    mag_diff_file = os.path.abspath("magnitude_diff.nii.gz")
    phase_diff_file = os.path.abspath("phase_diff.nii.gz")

    magnitude_img = nb.load(magnitude)
    phase_img = nb.load(phase)
    magnitude_data = magnitude_img.get_fdata()
    phase_data = phase_img.get_fdata()

    signal_diff = (
        magnitude_data[..., 0]
        * magnitude_data[..., 1]
        * np.exp(1j * (phase_data[..., 1] - phase_data[..., 0]))
    )
    mag_diff = np.abs(signal_diff)
    phase_diff = np.angle(signal_diff)
    mag_diff_img = nb.Nifti1Image(mag_diff, magnitude_img.affine, magnitude_img.header)
    phase_diff_img = nb.Nifti1Image(phase_diff, phase_img.affine, phase_img.header)
    mag_diff_img.to_filename(mag_diff_file)
    phase_diff_img.to_filename(phase_diff_file)

    return mag_diff_file, phase_diff_file


def create_brain_mask(magnitude, extra_dilation=0):
    """Create a quick brain mask for a single frame.

    Parameters
    ----------
    magnitude : npt.NDArray[np.float32]
        Magnitude data with the shortest echo time
    extra_dilation : int
        Number of extra dilations (or erosions if negative) to perform, by default 0

    Returns
    -------
    npt.NDArray[np.bool_]
        Mask of voxels to use for unwrapping
    """
    import os

    import nibabel as nb
    import numpy as np
    from scipy.ndimage import (
        binary_dilation,
        binary_fill_holes,
        binary_erosion,
        generate_binary_structure,
    )
    from skimage.filters import threshold_otsu

    from sdcflows.utils.misc import get_largest_connected_component

    mag_img = nb.load(magnitude)
    mag_data = mag_img.get_fdata()
    mag_data = mag_data[..., 0]

    mask_file = os.path.abspath("mask.nii.gz")

    # create structuring element
    strel = generate_binary_structure(3, 2)

    # get the otsu threshold
    threshold = threshold_otsu(mag_data)
    mask_data = mag_data > threshold
    if mask_data.ndim != strel.ndim:
        raise ValueError(f"{mask_data.shape} ({mask_data.ndim}) != {strel.shape} ({strel.ndim})")
    mask_data = binary_fill_holes(mask_data, strel).astype(np.float32)

    # erode mask
    mask_data = binary_erosion(
        mask_data,
        structure=strel,
        iterations=2,
        border_value=1,
    ).astype(bool)

    # get largest connected component
    mask_data = get_largest_connected_component(mask_data)

    # dilate the mask
    mask_data = binary_dilation(mask_data, structure=strel, iterations=2)

    # extra dilation to get areas on the edge of the brain
    if extra_dilation > 0:
        mask_data = binary_dilation(mask_data, structure=strel, iterations=extra_dilation)
    # if negative, erode instead
    if extra_dilation < 0:
        mask_data = binary_erosion(mask_data, structure=strel, iterations=abs(extra_dilation))

    # since we can't have a completely empty mask, set all zeros to ones
    # if the mask is all empty
    if np.all(np.isclose(mask_data, 0)):
        mask_data = np.ones(mask_data.shape)

    mask_data = mask_data.astype(np.bool_).astype(np.uint8)
    mask_img = nb.Nifti1Image(mask_data, mag_img.affine, mag_img.header)
    mask_img.to_filename(mask_file)
    return mask_file


def calculate_offset(phase, unwrapped_diff, echo_times):
    """Calculate the phase offset between two echoes.

    TODO: Rename this and compute_offset.

    Parameters
    ----------
    phase : :obj:`str`
        The path to the phase image (concatenated across echoes).
    unwrapped_diff : :obj:`str`
        The path to the unwrapped phase difference image from the first two echoes.
    echo_times : :obj:`list` of :obj:`float`
        The echo times.

    Returns
    -------
    offset : :obj:`str`
        The path to the phase offset image.
    """
    import os

    import nibabel as nb
    import numpy as np

    offset_file = os.path.abspath("offset.nii.gz")

    phase_img = nb.load(phase)
    unwrapped_diff_img = nb.load(unwrapped_diff)
    phase_data = phase_img.get_fdata()
    unwrapped_diff_data = unwrapped_diff_img.get_fdata()

    proposed_offset = np.angle(
        np.exp(
            1j
            * (
                phase_data[..., 0]
                - ((echo_times[0] * unwrapped_diff_data) / (echo_times[1] - echo_times[0]))
            )
        )
    )
    proposed_offset_img = nb.Nifti1Image(proposed_offset, phase_img.affine, phase_img.header)
    proposed_offset_img.to_filename(offset_file)
    return offset_file


def subtract_offset(phase, offset):
    """Subtract the offset from the phase image.

    Parameters
    ----------
    phase : :obj:`str`
        The path to the phase image.
    offset : :obj:`str`
        The path to the offset image.

    Returns
    -------
    updated_phase : :obj:`str`
        The path to the updated phase image.
    """
    import os

    import nibabel as nb
    import numpy as np

    updated_phase_file = os.path.abspath("updated_phase.nii.gz")

    phase_img = nb.load(phase)
    offset_img = nb.load(offset)
    phase_data = phase_img.get_fdata()
    offset_data = offset_img.get_fdata()
    updated_phase = phase_data - offset_data[..., np.newaxis]
    updated_phase_img = nb.Nifti1Image(updated_phase, phase_img.affine, phase_img.header)
    updated_phase_img.to_filename(updated_phase_file)
    return updated_phase_file


def calculate_dual_echo_fieldmap(unwrapped_phase, echo_times):
    """Calculate the fieldmap from the unwrapped phase difference of the first two echoes.

    Parameters
    ----------
    unwrapped_phase : :obj:`str`
        The path to the unwrapped phase difference image.
    echo_times : :obj:`list` of :obj:`float`
        The echo times.

    Returns
    -------
    fieldmap : :obj:`str`
        The path to the fieldmap image.
    """
    import os

    import nibabel as nb
    import numpy as np

    fieldmap_file = os.path.abspath("fieldmap.nii.gz")

    unwrapped_phase_img = nb.load(unwrapped_phase)
    unwrapped_phase_data = unwrapped_phase_img.get_fdata()

    phase_diff = unwrapped_phase_data[..., 1] - unwrapped_phase_data[..., 0]
    fieldmap = (1000 / (2 * np.pi)) * phase_diff / (echo_times[1] - echo_times[0])
    fieldmap_img = nb.Nifti1Image(
        fieldmap,
        unwrapped_phase_img.affine,
        unwrapped_phase_img.header,
    )
    fieldmap_img.to_filename(fieldmap_file)

    return fieldmap_file


def modify_unwrapped_diff(phase, unwrapped_diff, echo_times):
    """Add 2*pi to unwrapped difference data and recalculate the offset."""
    import os

    import nibabel as nb
    import numpy as np

    updated_phase_file = os.path.abspath("updated_phase.nii.gz")

    phase_img = nb.load(phase)
    unwrapped_diff_img = nb.load(unwrapped_diff)
    phase_data = phase_img.get_fdata()
    unwrapped_diff_data = unwrapped_diff_img.get_fdata()

    new_proposed_offset = np.angle(
        np.exp(
            1j
            * (
                phase_data[..., 0]
                - (
                    (echo_times[0] * (unwrapped_diff_data + 2 * np.pi))
                    / (echo_times[1] - echo_times[0])
                )
            )
        )
    )
    new_proposed_phase = phase_data - new_proposed_offset[..., np.newaxis]
    new_proposed_phase_img = nb.Nifti1Image(
        new_proposed_phase,
        phase_img.affine,
        phase_img.header,
    )
    new_proposed_phase_img.to_filename(updated_phase_file)

    return updated_phase_file


def select_fieldmap(
    original_fieldmap,
    original_unwrapped_phase,
    original_offset,
    modified_fieldmap,
    modified_unwrapped_phase,
    unwrapped_diff,
    voxel_mask,
    echo_times,
    wrap_limit,
):
    import os

    import nibabel as nb
    import numpy as np

    FMAP_PROPORTION_HEURISTIC = 0.25
    FMAP_AMBIGUIOUS_HEURISTIC = 0.5

    new_unwrapped_diff_file = os.path.abspath("new_unwrapped_diff.nii.gz")

    orig_fmap_img = nb.load(original_fieldmap)
    orig_unwrapped_phase_img = nb.load(original_unwrapped_phase)
    orig_offset_img = nb.load(original_offset)
    mod_fmap_img = nb.load(modified_fieldmap)
    mod_unwrapped_phase_img = nb.load(modified_unwrapped_phase)
    unwrapped_diff_img = nb.load(unwrapped_diff)
    voxel_mask_img = nb.load(voxel_mask)

    orig_fmap_data = orig_fmap_img.get_fdata()
    orig_unwrapped_phase_data = orig_unwrapped_phase_img.get_fdata()
    orig_offset_data = orig_offset_img.get_fdata()
    mod_fmap_data = mod_fmap_img.get_fdata()
    mod_unwrapped_phase_data = mod_unwrapped_phase_img.get_fdata()
    unwrapped_diff_data = unwrapped_diff_img.get_fdata()
    voxel_mask_data = voxel_mask_img.get_fdata().astype(bool)

    masked_orig_fmap = orig_fmap_data[voxel_mask_data]
    if masked_orig_fmap.mean() < -10:
        # check if the proposed fieldmap is below -10
        unwrapped_diff_data += 2 * np.pi

    elif masked_orig_fmap.mean() < 0 and not wrap_limit:
        # check if the proposed fieldmap is between -10 and 0
        # look at proportion of voxels that are positive
        voxel_prop = np.count_nonzero(masked_orig_fmap > 0) / masked_orig_fmap.shape[0]

        # if the proportion of positive voxels is less than 0.25, then add 2pi
        if voxel_prop < FMAP_PROPORTION_HEURISTIC:
            unwrapped_diff_data += 2 * np.pi
        elif voxel_prop < FMAP_AMBIGUIOUS_HEURISTIC:
            # compute mean of phase offset
            mean_phase_offset = orig_offset_data[voxel_mask_data].mean()
            if mean_phase_offset < -1:
                phase_fits = np.concatenate(
                    (
                        np.zeros((*orig_unwrapped_phase_data.shape[:-1], 1)),
                        orig_unwrapped_phase_data,
                    ),
                    axis=-1,
                )
                _, residuals_1, _, _, _ = np.polyfit(
                    echo_times,
                    phase_fits[voxel_mask_data, :].T,
                    1,
                    full=True,
                )

                masked_mod_fmap = mod_fmap_data[voxel_mask_data]
                # fit linear model to the proposed phase
                new_phase_fits = np.concatenate(
                    (
                        np.zeros((*mod_unwrapped_phase_data.shape[:-1], 1)),
                        mod_unwrapped_phase_data,
                    ),
                    axis=-1,
                )
                _, residuals_2, _, _, _ = np.polyfit(
                    echo_times,
                    new_phase_fits[voxel_mask_data, :].T,
                    1,
                    full=True,
                )

                if (
                    np.isclose(residuals_1.mean(), residuals_2.mean(), atol=1e-3, rtol=1e-3)
                    and masked_mod_fmap.mean() > 0
                ):
                    unwrapped_diff_data += 2 * np.pi
                else:
                    unwrapped_diff_data -= 2 * np.pi

    new_unwrapped_diff_img = nb.Nifti1Image(
        unwrapped_diff_data,
        unwrapped_diff_img.affine,
        unwrapped_diff_img.header,
    )
    new_unwrapped_diff_img.to_filename(new_unwrapped_diff_file)
    return new_unwrapped_diff_file


def global_mode_correction(magnitude, unwrapped, mask, echo_times):
    """Compute the global mode offset for the first echo and apply it to the subsequent echoes.

    Parameters
    ----------
    magnitude : :obj:`str`
    unwrapped : :obj:`str`
    mask : :obj:`str`
    echo_times : :obj:`list` of :obj:`float`

    Returns
    -------
    new_unwrapped_file : :obj:`str`

    Notes
    -----
    This computes the global mode offset for the first echo then tries to find the offset
    that minimizes the residuals for each subsequent echo
    """
    import os

    import nibabel as nb
    import numpy as np

    from sdcflows.utils.misc import compute_offset, create_brain_mask

    new_unwrapped_file = os.path.abspath("unwrapped.nii.gz")

    mag_img = nb.load(magnitude)
    unwrapped_img = nb.load(unwrapped)
    mask_img = nb.load(mask)
    mag_data = mag_img.get_fdata()
    unwrapped_data = unwrapped_img.get_fdata()
    mask_data = mask_img.get_fdata().astype(bool)
    echo_times = np.array(echo_times)

    brain_mask = create_brain_mask(magnitude)
    brain_mask_img = nb.load(brain_mask)
    brain_mask_data = brain_mask_img.get_fdata().astype(bool)

    # for each of these matrices TEs are on rows, voxels are columns
    # get design matrix
    X = echo_times[:, np.newaxis]

    # get magnitude weight matrix
    W = mag_data[brain_mask_data, :].T

    # loop over each index past 1st echo
    for i in range(1, echo_times.shape[0]):
        # get matrix with the masked unwrapped data (weighted by magnitude)
        Y = unwrapped_data[brain_mask_data, :].T

        # Compute offset through linear regression method
        best_offset = compute_offset(i, W, X, Y)

        # apply the offset
        unwrapped_data[..., i] += 2 * np.pi * best_offset

    # set anything outside of mask_data to 0
    unwrapped_data[~mask_data] = 0

    new_unwrapped_img = nb.Nifti1Image(unwrapped_data, mag_img.affine, mag_img.header)
    new_unwrapped_img.to_filename(new_unwrapped_file)

    return new_unwrapped_file


def corr2_coeff(A, B):
    """Efficiently calculates correlation coefficient between the columns of two 2D arrays

    Parameters
    ----------
    A : npt.NDArray
        1st array to correlate
    B : npt.NDArray
        2nd array to correlate

    Returns
    -------
    npt.NDArray
        array of correlation coefficients
    """
    # Transpose A and B
    A = A.T
    B = B.T

    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(axis=1, keepdims=True)
    B_mB = B - B.mean(axis=1, keepdims=True)

    # Sum of squares across rows
    ssA = (A_mA**2).sum(axis=1, keepdims=True)
    ssB = (B_mB**2).sum(axis=1, keepdims=True).T

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA, ssB))


def check_temporal_consistency_corr(
    unwrapped_data,
    unwrapped_echo_1,
    TEs,
    mag,
    t,
    frame_idx,
    masks,
    threshold=0.98,
):
    """Ensures phase unwrapping solutions are temporally consistent

    This uses correlation as a similarity metric between frames to enforce temporal consistency.

    Parameters
    ----------
    unwrapped_data : npt.NDArray
        unwrapped phase data, where last column is time, and second to last column are the echoes
    TEs : npt.NDArray
        echo times
    mag : List[nib.Nifti1Image]
        magnitude images
    frames : List[int]
        list of frames that are being processed
    threshold : float
        threshold for correlation similarity. By default 0.98
    """
    from sdcflows.utils.misc import corr2_coeff

    logging.info(f"Computing temporal consistency check for frame: {t}")

    # generate brain mask (with 1 voxel erosion)
    echo_idx = np.argmin(TEs)
    mag_shortest = mag[echo_idx].dataobj[..., frame_idx]
    brain_mask = create_brain_mask(mag_shortest, -1)

    # get the current frame phase
    current_frame_data = unwrapped_echo_1[brain_mask, t][:, np.newaxis]

    # get the correlation between the current frame and all other frames
    corr = corr2_coeff(current_frame_data, unwrapped_echo_1[brain_mask, :]).ravel()

    # threhold the RD
    tmask = corr > threshold

    # get indices of mask
    indices = np.where(tmask)[0]

    # get mask for frame
    mask = masks[..., t] > 0

    # for each frame compute the mean value along the time axis (masked by indices and mask)
    mean_voxels = np.mean(unwrapped_echo_1[mask][:, indices], axis=-1)

    # for this frame figure out the integer multiple that minimizes the value to the mean voxel
    int_map = np.round((mean_voxels - unwrapped_echo_1[mask, t]) / (2 * np.pi)).astype(int)

    # correct the data using the integer map
    unwrapped_data[mask, 0, t] += 2 * np.pi * int_map

    # format weight matrix
    weights_mat = np.stack([m.dataobj[..., frame_idx] for m in mag], axis=-1)[mask].T

    # form design matrix
    X = TEs[:, np.newaxis]

    # fit subsequent echos to the weighted linear regression from the first echo
    for echo in range(1, unwrapped_data.shape[-2]):
        # form response matrix
        Y = unwrapped_data[mask, :echo, t].T

        # fit model to data
        coefficients, _ = weighted_regression(X[:echo], Y, weights_mat[:echo])

        # get the predicted values for this echo
        Y_pred = coefficients * TEs[echo]

        # compute the difference and get the integer multiple map
        int_map = np.round((Y_pred - unwrapped_data[mask, echo, t]) / (2 * np.pi)).astype(int)

        # correct the data using the integer map
        unwrapped_data[mask, echo, t] += 2 * np.pi * int_map
