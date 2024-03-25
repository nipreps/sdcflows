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


def create_brain_mask(
    mag_shortest: npt.NDArray[np.float32],
    extra_dilation: int = 0,
) -> npt.NDArray[np.bool_]:
    """Create a quick brain mask for a single frame.

    Parameters
    ----------
    mag_shortest : npt.NDArray[np.float32]
        Magnitude data with the shortest echo time
    extra_dilation : int
        Number of extra dilations (or erosions if negative) to perform, by default 0

    Returns
    -------
    npt.NDArray[np.bool_]
        Mask of voxels to use for unwrapping
    """
    from typing import cast

    import numpy as np
    import numpy.typing as npt
    from scipy.ndimage import (
        binary_dilation,
        binary_erosion,
        binary_fill_holes,
        generate_binary_structure,
    )
    from skimage.filters import threshold_otsu

    # create structuring element
    strel = generate_binary_structure(3, 2)

    # get the otsu threshold
    threshold = threshold_otsu(mag_shortest)
    mask_data = mag_shortest > threshold
    mask_data = cast(npt.NDArray[np.float32], binary_fill_holes(mask_data, strel))

    # erode mask
    mask_data = cast(
        npt.NDArray[np.bool_],
        binary_erosion(mask_data, structure=strel, iterations=2, border_value=1),
    )

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

    # return the mask
    return mask_data.astype(np.bool_)


def medic_automask(mag_file, voxel_quality, echo_times, automask_dilation, out_file="mask.nii.gz"):
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

    mag_img = nb.load(mag_file)
    mag_data = mag_img.get_fdata()

    vq = nb.load(voxel_quality).get_fdata()
    vq_mask = vq > threshold_otsu(vq)
    strel = generate_binary_structure(3, 2)
    vq_mask = cast(npt.NDArray[np.bool_], binary_fill_holes(vq_mask, strel))
    # get largest connected component
    vq_mask = get_largest_connected_component(vq_mask)

    # combine masks
    echo_idx = np.argmin(echo_times)
    mag_shortest = mag_data[..., echo_idx]
    brain_mask = create_brain_mask(mag_shortest)
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

    return out_file


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
    mag_diff_file = "mag_diff.nii.gz"
    phase_diff_file = "phase_diff.nii.gz"
    mag_diff_img.to_filename(mag_diff_file)
    phase_diff_img.to_filename(phase_diff_file)
    return mag_diff_file, phase_diff_file


def mcpc_3d_s(mag_file, phase_file, echo_times, unwrapped_diff_file, mask_file, wrap_limit):
    """Estimate and remove phase offset with MCPC-3D-S algorithm.

    Parameters
    ----------
    mag_file : str
        Magnitude image file
    phase_file : str
        Phase image file
    echo_times : :obj:`numpy.ndarray`
        Echo times
    unwrapped_diff_file : str
        Unwrapped phase difference file
    mask_file : str
        Mask file
    wrap_limit : bool
        If True, this turns off some heuristics for phase unwrapping.

    Returns
    -------
    offset_file : str
        Phase offset file
    new_unwrapped_diff_file : str
        Unwrapped phase difference file

    Notes
    -----
    "MCPC-3D-S estimates the phase offset by computing the unwrapped phase difference between
    the first and second echoes of the data,
    then estimating the phase offset by assuming linear phase accumulation between the first
    and second echoes." (from Van et al.)

    The MCPC-3D-S algorithm is described in https://doi.org/10.1002/mrm.26963.
    """
    import nibabel as nb
    import numpy as np

    FMAP_PROPORTION_HEURISTIC = 0.25
    FMAP_AMBIGUOUS_HEURISTIC = 0.5

    mag_img = nb.load(mag_file)
    phase_img = nb.load(phase_file)
    unwrapped_diff = nb.load(unwrapped_diff_file).get_fdata()
    mask = nb.load(mask_file).get_fdata().astype(bool)
    mag_data = mag_img.get_fdata()
    phase_data = phase_img.get_fdata()
    TE0, TE1 = echo_times[:2]
    phase0 = phase_data[..., 0]
    phase1 = phase_data[..., 1]
    mag0 = mag_data[..., 0]
    mag1 = mag_data[..., 1]
    voxel_mask = create_brain_mask(mag0, -2)
    phases = np.stack([phase0, phase1], axis=-1)
    mags = np.stack([mag0, mag1], axis=-1)
    TEs = np.array([TE0, TE1])
    all_TEs = np.array([0.0, TE0, TE1])
    proposed_offset = np.angle(np.exp(1j * (phase0 - ((TE0 * unwrapped_diff) / (TE1 - TE0)))))

    # get the new phases
    proposed_phases = phases - proposed_offset[..., np.newaxis]

    # compute the fieldmap
    # TODO: Move this step into a separate interface, since it calls ROMEO
    proposed_fieldmap, proposed_unwrapped_phases = get_dual_echo_fieldmap(
        proposed_phases, TEs, mags, mask
    )

    # check if the proposed fieldmap is below 10
    # print(f"proposed_fieldmap: {proposed_fieldmap[voxel_mask].mean()}")
    if proposed_fieldmap[voxel_mask].mean() < -10:
        unwrapped_diff += 2 * np.pi
    # check if the propossed fieldmap is between -10 and 0
    elif proposed_fieldmap[voxel_mask].mean() < 0 and not wrap_limit:
        # look at proportion of voxels that are positive
        voxel_prop = (
            np.count_nonzero(proposed_fieldmap[voxel_mask] > 0)
            / proposed_fieldmap[voxel_mask].shape[0]
        )

        # if the proportion of positive voxels is less than 0.25, then add 2pi
        if voxel_prop < FMAP_PROPORTION_HEURISTIC:
            unwrapped_diff += 2 * np.pi
        elif voxel_prop < FMAP_AMBIGUOUS_HEURISTIC:
            # compute mean of phase offset
            mean_phase_offset = proposed_offset[voxel_mask].mean()
            # print(f"mean_phase_offset: {mean_phase_offset}")
            # if less than -1 then
            if mean_phase_offset < -1:
                phase_fits = np.concatenate(
                    (
                        np.zeros((*proposed_unwrapped_phases.shape[:-1], 1)),
                        proposed_unwrapped_phases,
                    ),
                    axis=-1,
                )
                _, residuals_1, _, _, _ = np.polyfit(
                    all_TEs,
                    phase_fits[voxel_mask, :].T,
                    1,
                    full=True,
                )

                # check if adding 2pi makes it better
                new_proposed_offset = np.angle(
                    np.exp(1j * (phase0 - ((TE0 * (unwrapped_diff + 2 * np.pi)) / (TE1 - TE0))))
                )
                new_proposed_phases = phases - new_proposed_offset[..., np.newaxis]
                new_proposed_fieldmap, new_proposed_unwrapped_phases = get_dual_echo_fieldmap(
                    new_proposed_phases, TEs, mags, mask
                )

                # fit linear model to the proposed phases
                new_phase_fits = np.concatenate(
                    (
                        np.zeros((*new_proposed_unwrapped_phases.shape[:-1], 1)),
                        new_proposed_unwrapped_phases,
                    ),
                    axis=-1,
                )
                _, residuals_2, _, _, _ = np.polyfit(
                    echo_times, new_phase_fits[voxel_mask, :].T, 1, full=True
                )

                if (
                    np.isclose(residuals_1.mean(), residuals_2.mean(), atol=1e-3, rtol=1e-3)
                    and new_proposed_fieldmap[voxel_mask].mean() > 0
                ):
                    unwrapped_diff += 2 * np.pi
                else:
                    unwrapped_diff -= 2 * np.pi

    # compute the phase offset
    offset = np.angle(np.exp(1j * (phase0 - ((TE0 * unwrapped_diff) / (TE1 - TE0)))))

    new_unwrapped_diff_file = "unwrapped_diff.nii.gz"
    offset_file = "offset.nii.gz"
    nb.Nifti1Image(offset, mag_img.affine, mag_img.header).to_filename(offset_file)
    nb.Nifti1Image(unwrapped_diff, mag_img.affine, mag_img.header).to_filename(
        new_unwrapped_diff_file
    )
    return offset_file, new_unwrapped_diff_file


def global_mode_correction(
    echo_times,
    magnitude_file,
    phase_file,
    mask_file,
    masksum_file,
    automask=True,
):
    """Apply global mode correction.

    Compute the global mode offset for the first echo then try to find the offset
    that minimizes the residuals for each subsequent echo.
    """
    import nibabel as nb
    import numpy as np

    # global mode correction
    # use auto mask to get brain mask
    mag_data = nb.load(magnitude_file).get_fdata()
    phase_img = nb.load(phase_file)
    phase_data = phase_img.get_fdata()
    mask_data = nb.load(mask_file).get_fdata().astype(np.bool)

    echo_idx = np.argmin(echo_times)
    mag_shortest = mag_data[..., echo_idx]
    brain_mask = create_brain_mask(mag_shortest)

    # for each of these matrices TEs are on rows, voxels are columns
    # get design matrix
    X = echo_times[:, np.newaxis]

    # get magnitude weight matrix
    W = mag_data[brain_mask, :].T

    # loop over each index past 1st echo
    for i in range(1, echo_times.shape[0]):
        # get matrix with the masked unwrapped data (weighted by magnitude)
        Y = phase_data[brain_mask, :].T

        # Compute offset through linear regression method
        best_offset = compute_offset(i, W, X, Y)

        # apply the offset
        phase_data[..., i] += 2 * np.pi * best_offset

    # set anything outside of mask_data to 0
    phase_data[~mask_data] = 0

    unwrapped_file = "unwrapped_phase.nii.gz"
    nb.Nifti1Image(phase_data, phase_img.affine, phase_img.header).to_filename(unwrapped_file)

    if not automask:
        final_mask_file = "final_mask.nii.gz"
        nb.Nifti1Image(
            mask_data.astype(np.int8),
            phase_img.affine,
            phase_img.header,
        ).to_filename(final_mask_file)
    else:
        final_mask_file = masksum_file

    return final_mask_file, unwrapped_file


def compute_offset(echo_ind: int, W: npt.NDArray, X: npt.NDArray, Y: npt.NDArray) -> int:
    """Method for computing the global mode offset for echoes > 1.

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


def svd_filtering(
    field_maps: npt.NDArray,
    new_masks: npt.NDArray,
    voxel_size: float,
    n_frames: int,
    border_filt: Tuple[int, int],
    svd_filt: int,
):
    if new_masks.max() == 2 and n_frames >= np.max(border_filt):
        logging.info("Performing spatial/temporal filtering of border voxels...")
        smoothed_field_maps = np.zeros(field_maps.shape, dtype=np.float32)
        # smooth by 4 mm kernel
        sigma = (4 / voxel_size) / 2.355
        for i in range(field_maps.shape[-1]):
            smoothed_field_maps[..., i] = gaussian_filter(field_maps[..., i], sigma=sigma)
        # compute the union of all the masks
        union_mask = np.sum(new_masks, axis=-1) > 0
        # do temporal filtering of border voxels with SVD
        U, S, VT = np.linalg.svd(smoothed_field_maps[union_mask], full_matrices=False)
        # first pass of SVD filtering
        recon = np.dot(U[:, : border_filt[0]] * S[: border_filt[0]], VT[: border_filt[0], :])
        recon_img = np.zeros(field_maps.shape, dtype=np.float32)
        recon_img[union_mask] = recon
        # set the border voxels in the field map to the recon values
        for i in range(field_maps.shape[-1]):
            field_maps[new_masks[..., i] == 1, i] = recon_img[new_masks[..., i] == 1, i]
        # do second SVD filtering pass
        U, S, VT = np.linalg.svd(field_maps[union_mask], full_matrices=False)
        # second pass of SVD filtering
        recon = np.dot(U[:, : border_filt[1]] * S[: border_filt[1]], VT[: border_filt[1], :])
        recon_img = np.zeros(field_maps.shape, dtype=np.float32)
        recon_img[union_mask] = recon
        # set the border voxels in the field map to the recon values
        for i in range(field_maps.shape[-1]):
            field_maps[new_masks[..., i] == 1, i] = recon_img[new_masks[..., i] == 1, i]

    # use svd filter to denoise the field maps
    if n_frames >= svd_filt:
        logging.info("Denoising field maps with SVD...")
        logging.info(f"Keeping {svd_filt} components...")
        # compute the union of all the masks
        union_mask = np.sum(new_masks, axis=-1) > 0
        # compute SVD
        U, S, VT = np.linalg.svd(field_maps[union_mask], full_matrices=False)
        # only keep the first n_components components
        recon = np.dot(U[:, :svd_filt] * S[:svd_filt], VT[:svd_filt, :])
        recon_img = np.zeros(field_maps.shape, dtype=np.float32)
        recon_img[union_mask] = recon
        # set the voxel values in the mask to the recon values
        for i in range(field_maps.shape[-1]):
            field_maps[new_masks[..., i] > 0, i] = recon_img[new_masks[..., i] > 0, i]

    return field_maps


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
