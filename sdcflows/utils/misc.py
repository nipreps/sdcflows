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


def mcpc_3d_s(mag_file, phase_file, echo_times, unwrapped_diff_file, mask_file):
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
