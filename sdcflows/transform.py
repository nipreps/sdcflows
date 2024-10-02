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
r"""
The :math:`B_0` unwarping transform formalism.

This module implements a data structure to represent the displacements field associated
to the deformations caused by susceptibility-derived distortions.
This implementation attempts to provide a single representation of such distortions independently
of the estimation strategy (see :doc:`/methods`).

.. _bspline-interpolation:

That is achieved by implementing multi-level B-Spline cubic interpolators.
For one given level, a function :math:`f(\mathbf{s})` can be represented as a linear combination
of tensor-product cubic B-Spline basis (:math:`\Psi^3(\mathbf{k}, \mathbf{s})`;
see Eq. :math:`\eqref{eq:2}`).


.. math::

    f(\mathbf{s}) =
    \sum_{k_1} \sum_{k_2} \sum_{k_3} c(\mathbf{k}) \Psi^3(\mathbf{k}, \mathbf{s}).
    \label{eq:1}\tag{1}


"""
from __future__ import annotations

import os
from functools import partial
import asyncio
from pathlib import Path
from typing import Callable, Sequence, Tuple

import attr
import numpy as np
from warnings import warn
from scipy import ndimage as ndi
from scipy.interpolate import BSpline
from scipy.sparse import hstack as sparse_hstack, kron, lil_array

import nibabel as nb
import nitransforms as nt
from bids.utils import listify

from niworkflows.interfaces.nibabel import reorient_image

from sdcflows.utils.tools import ensure_positive_cosines


def _sdc_unwarp(
    data: np.ndarray,
    coordinates: np.ndarray,
    pe_info: Tuple[int, float],
    hmc_xfm: np.ndarray | None,
    jacobian: bool,
    fmap_hz: np.ndarray,
    output_dtype: str | np.dtype | None = None,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
) -> np.ndarray:
    """Resample one volume, moving through a head motion correction affine if provided."""

    if hmc_xfm is not None:
        # Move image with the head
        coords_shape = coordinates.shape
        coordinates = nb.affines.apply_affine(
            hmc_xfm, coordinates.reshape(coords_shape[0], -1).T
        ).T.reshape(coords_shape)

    # Map voxel coordinates applying the VSM
    # The VSM is just the displacements field given in index coordinates
    # voxcoords is the deformation field, i.e., the target position of each voxel
    vsm = fmap_hz * pe_info[1]
    coordinates[pe_info[0], ...] += vsm

    resampled = ndi.map_coordinates(
        data,
        coordinates,
        output=output_dtype,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=prefilter,
    )

    # The Jacobian determinant image is the amount of stretching in the PE direction.
    # Using central differences accounts for the shift in neighboring voxels.
    # The full Jacobian at each voxel would be a 3x3 matrix, but because there is
    # only warping in one direction, we end up with a diagonal matrix with two 1s.
    # The following is the other entry at each voxel, and hence the determinant.
    if jacobian:
        resampled *= 1 + np.gradient(vsm, axis=pe_info[0])

    return resampled


async def worker(
    data: np.ndarray,
    coordinates: np.ndarray,
    pe_info: Tuple[int, float],
    hmc_xfm: np.ndarray,
    func: Callable,
    semaphore: asyncio.Semaphore,
) -> np.ndarray:
    """Create one worker and attach it to the execution loop."""
    async with semaphore:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, func, data, coordinates, pe_info, hmc_xfm
        )
        return result


async def unwarp_parallel(
    fulldataset: np.ndarray,
    coordinates: np.ndarray,
    fmap_hz: np.ndarray,
    pe_info: Sequence[Tuple[int, float]],
    xfms: Sequence[np.ndarray],
    jacobian: bool,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
    output_dtype: str | np.dtype | None = None,
    max_concurrent: int = min(os.cpu_count(), 12),
) -> np.ndarray:
    r"""
    Unwarp an EPI dataset parallelizing across volumes.

    Parameters
    ----------
    fulldataset : :obj:`~numpy.ndarray`
        An array of shape (I, J, K, T), where I, J, K are the dimensions of spatial axes
        and T is the number of volumes.
        The full data array of the EPI that are wanted after correction.
    coordinates : :obj:`~numpy.ndarray`
        An array of shape (3, I, J, K) array providing the voxel (index) coordinates of
        the reference image (i.e., interpolated points) before SDC/HMC.
    fmap_hz : :obj:`~numpy.ndarray`
        An array of shape (I, J, K) containing the displacement of each voxel in voxel units.
    pe_info : :obj:`tuple` of (:obj:`int`, :obj:`float`)
        A tuple containing the index of the phase-encoding axis in the data array and
        the readout time (including sign, if displacements must be reversed)
    jacobian : :class:`bool`
        If :obj:`True`, apply Jacobian determinant correction after unwarping.
    xfms : :obj:`list` of obj:`~numpy.ndarray`
        A list of 4×4 matrices, each one formalizing
        the estimated head motion alignment to the scan's reference.
        Therefore, each of these matrices express the transform of every
        voxel's RAS (physical) coordinates in the image used as reference
        for realignment into the coordinates of each of the EPI series volume.
    order : :obj:`int`, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    mode : {'constant', 'reflect', 'nearest', 'mirror', 'wrap'}, optional
        Determines how the input image is extended when the resamplings overflows
        a border. Default is 'constant'.
    cval : :obj:`float`, optional
        Constant value for ``mode='constant'``. Default is 0.0.
    prefilter : :obj:`bool`, optional
        Determines if the image's data array is prefiltered with
        a spline filter before interpolation. The default is ``True``,
        which will create a temporary *float64* array of filtered values
        if *order > 1*. If setting this to ``False``, the output will be
        slightly blurred if *order > 1*, unless the input is prefiltered,
        i.e. it is the result of calling the spline filter on the original
        input.
    output_dtype : :obj:`str` or :obj:`~numpy.dtype`
        Override the output data type, instead of propagating it from the
        moving image.
    max_concurrent : :obj:`int`
        The maximum number of parallel resamplings at any given time of execution.
        Use this parameter to set an upper bound to memory utilization.

    """

    semaphore = asyncio.Semaphore(max_concurrent)

    if fulldataset.ndim == 3:
        fulldataset = fulldataset[..., np.newaxis]

    func = partial(
        _sdc_unwarp,
        jacobian=jacobian,
        fmap_hz=fmap_hz,
        output_dtype=output_dtype,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=prefilter,
    )

    # Create a worker task for each chunk
    tasks = []
    for volid, volume in enumerate(np.rollaxis(fulldataset, -1, 0)):
        xfm = None if xfms is None else xfms[volid]

        # IMPORTANT - the coordinates array must be copied every time anew per thread
        task = asyncio.create_task(
            worker(
                volume,
                coordinates.copy(),
                pe_info[volid],
                xfm,
                func,
                semaphore,
            )
        )
        tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

    # Collect the results and stack along last dimension
    results = np.stack([task.result() for task in tasks], -1)

    return results


@attr.s(slots=True)
class B0FieldTransform:
    """Represents and applies the transform to correct for susceptibility distortions."""

    coeffs = attr.ib(default=None)
    """B-Spline coefficients (one value per control point)."""
    mapped = attr.ib(default=None, init=False)
    """
    A cache of the interpolated field in Hz (i.e., the fieldmap *mapped* on to the
    target image we want to correct).
    """

    def fit(
        self,
        target_reference: nb.spatialimages.SpatialImage,
        xfm_data2fmap: np.ndarray | None = None,
        approx: bool = True,
    ) -> bool:
        r"""
        Generate the interpolation matrix (and the VSM with it).

        Implements Eq. :math:`\eqref{eq:1}`, interpolating :math:`f(\mathbf{s})`
        for all voxels in the target-image's extent.

        Parameters
        ----------
        target_reference : :obj:`~nibabel.spatialimages.SpatialImage`
            The image object containing a reference grid (same as that of the data
            to be resampled). If a 4D dataset is provided, then the fourth dimension
            will be dropped.
        xfm_data2fmap : :obj:`numpy.ndarray`
            Transform that maps coordinates on the `target_reference` onto the
            fieldmap reference (that is, the linear transform through which the fieldmap can
            be resampled in register with the `target_reference`).
            In other words, `xfm_data2fmap` is the result of calling a registration tool
            such as ANTs configured for a linear transform with at most 12 degrees of freedom
            and called with the image carrying `target_affine` as reference and the fieldmap
            reference as moving.
            The result of such a registration framework is an affine (our `xfm_data2fmap` here)
            that maps coordinates in reference (target) RAS onto the fieldmap RAS.
        approx : :obj:`bool`
            If ``True``, do not reconstruct the B-Spline field directly on the target
            (which will likely not be aligned with the fieldmap's grid), but rather use
            the fieldmap's grid and then use just regular interpolation.

        Returns
        -------
        updated : :obj:`bool`
            ``True`` if the internal field representation was fit,
            ``False`` if cache was valid and will be reused.

        """
        # Calculate the physical coordinates of target grid
        if isinstance(target_reference, (str, bytes, Path)):
            target_reference = nb.load(target_reference)

        approx &= xfm_data2fmap is not None  # Approximate iff xfm_data2fmap is defined
        xfm_data2fmap = xfm_data2fmap if xfm_data2fmap is not None else np.eye(4)
        # Project the reference's grid onto the fieldmap's
        projected_reference = target_reference.__class__(
            target_reference.dataobj,
            xfm_data2fmap @ target_reference.affine,
            target_reference.header,
        )

        # Make sure the data array has all cosines positive (i.e., no axes are flipped)
        projected_reference, _ = ensure_positive_cosines(projected_reference)

        # Approximate only if the coordinate systems are not aligned
        coeffs = listify(self.coeffs)
        approx &= not np.allclose(
            np.linalg.norm(
                np.cross(
                    coeffs[-1].affine[:-1, :-1].T,
                    target_reference.affine[:-1, :-1].T,
                ),
                axis=1,
            ),
            0,
            atol=1e-3,
        )

        if approx:
            from sdcflows.utils.tools import deoblique_and_zooms

            # Generate a sampling reference on the fieldmap's space that fully covers
            # the target_reference's grid.
            projected_reference = deoblique_and_zooms(
                coeffs[-1],
                target_reference,
            )

        # Generate tensor-product B-Spline weights
        colmat = sparse_hstack(
            [grid_bspline_weights(projected_reference, level) for level in coeffs]
        ).tocsr()
        coefficients = np.hstack(
            [level.get_fdata(dtype="float32").reshape(-1) for level in coeffs]
        )

        # Reconstruct the fieldmap (in Hz) from coefficients
        fmap = np.reshape(colmat @ coefficients, projected_reference.shape[:3])

        # Generate a NIfTI object
        hdr = target_reference.header.copy()
        hdr.set_intent("estimate", name="fieldmap Hz")
        hdr.set_data_dtype("float32")
        hdr["cal_max"] = max((abs(fmap.min()), fmap.max()))
        hdr["cal_min"] = -hdr["cal_max"]

        # Cache
        self.mapped = nb.Nifti1Image(fmap, projected_reference.affine, hdr)

        if approx:
            from nitransforms.linear import Affine

            _tmp_reference = nb.Nifti1Image(
                np.zeros(
                    target_reference.shape[:3], dtype=target_reference.get_data_dtype()
                ),
                target_reference.affine,
                target_reference.header,
            )
            # Interpolate fmap given on target_reference in the original target_reference
            # voxel locations (overwrite fmap)
            self.mapped = Affine(reference=_tmp_reference).apply(self.mapped)

        return True

    def apply(
        self,
        moving: nb.spatialimages.SpatialImage,
        pe_dir: str | Sequence[str],
        ro_time: float | Sequence[float],
        xfms: Sequence[np.ndarray] | None = None,
        jacobian: bool = True,
        xfm_data2fmap: np.ndarray | None = None,
        approx: bool = True,
        order: int = 3,
        mode: str = "constant",
        cval: float = 0.0,
        prefilter: bool = True,
        output_dtype: str | np.dtype | None = None,
        num_threads: int = None,
        allow_negative: bool = False,
    ):
        r"""
        Apply a transformation to an image, resampling on the reference spatial object.

        Handles parallelization to resample 4D images.

        Parameters
        ----------
        moving : :obj:`~nibabel.spatialimages.SpatialImage`
            The image object containing the data to be resampled in reference
            space
        pe_dir : :obj:`str` or :obj:`list` of :obj:`str`
            A valid ``PhaseEncodingDirection`` metadata value.
        ro_time : :obj:`float` or :obj:`list` of :obj:`float`
            The total readout time in seconds.
        jacobian : :class:`bool`
            If :obj:`True`, apply Jacobian determinant correction after unwarping.
        xfms : :obj:`None` or :obj:`list`
            A list of 4×4 matrices, each one formalizing
            the estimated head motion alignment to the scan's reference.
            Therefore, each of these matrices express the transform of every
            voxel's RAS (physical) coordinates in the image used as reference
            for realignment into the coordinates of each of the EPI series volume.
        xfm_data2fmap : :obj:`numpy.ndarray`
            Transform that maps coordinates on the ``target_reference`` onto the
            fieldmap reference (that is, the linear transform through which the fieldmap can
            be resampled in register with the ``target_reference``).
            In other words, ``xfm_data2fmap`` is the result of calling a registration tool
            such as ANTs configured for a linear transform with at most 12 degrees of freedom
            and called with the image carrying ``target_affine`` as reference and the fieldmap
            reference as moving.
            The result of such a registration framework is an affine (our ``xfm_data2fmap`` here)
            that maps coordinates in reference (target) RAS onto the fieldmap RAS.
        approx : :obj:`bool`
            If ``True``, do not reconstruct the B-Spline field directly on the target
            (which will likely not be aligned with the fieldmap's grid), but rather use
            the fieldmap's grid and then use just regular interpolation.
        order : :obj:`int`, optional
            The order of the spline interpolation, default is 3.
            The order has to be in the range 0-5.
        mode : {'constant', 'reflect', 'nearest', 'mirror', 'wrap'}, optional
            Determines how the input image is extended when the resamplings overflows
            a border. Default is 'constant'.
        cval : float, optional
            Constant value for ``mode='constant'``. Default is 0.0.
        prefilter : :obj:`bool`, optional
            Determines if the image's data array is prefiltered with
            a spline filter before interpolation. The default is ``True``,
            which will create a temporary *float64* array of filtered values
            if *order > 1*. If setting this to ``False``, the output will be
            slightly blurred if *order > 1*, unless the input is prefiltered,
            i.e. it is the result of calling the spline filter on the original
            input.
        output_dtype : :obj:`str` or :obj:`~numpy.dtype`
            Override the output data type, instead of propagating it from the
            moving image.
        num_threads : :obj:`int`
            The maximum number of parallel resamplings at any given time of execution.
            Use this parameter to set an upper bound to memory utilization.
        allow_negative : :obj:`bool`
            Remove negative values introduced in interpolation (may happen for nonnegative data
            when order :math:`\gt` 3). Set this value to `True` if your `moving` image does
            have negative values.

        Returns
        -------
        resampled : :obj:`~nibabel.spatialimages.SpatialImage`
            The data imaged after resampling to reference space.

        """

        # Ensure the fmap has been computed
        if isinstance(moving, (str, bytes, Path)):
            moving = nb.load(moving)

        # Make sure the data array has all cosines positive (i.e., no axes are flipped)
        moving, axcodes = ensure_positive_cosines(moving)

        if self.mapped is not None:
            warn(
                "The fieldmap has been already fit, the user is responsible for "
                "ensuring the parameters of the EPI target are consistent."
            )
        else:
            # Generate warp field (before ensuring positive cosines)
            self.fit(moving, xfm_data2fmap=xfm_data2fmap, approx=approx)

        # Squeeze non-spatial dimensions
        newshape = moving.shape[:3] + tuple(dim for dim in moving.shape[3:] if dim > 1)
        data = nb.arrayproxy.reshape_dataobj(moving.dataobj, newshape)
        ndim = min(data.ndim, 3)
        n_volumes = data.shape[3] if data.ndim == 4 else 1
        output_dtype = output_dtype or moving.header.get_data_dtype()

        # Prepare input parameters
        if isinstance(pe_dir, str):
            pe_dir = [pe_dir]

        if isinstance(ro_time, float):
            ro_time = [ro_time]

        if n_volumes > 1 and len(pe_dir) == 1:
            pe_dir *= n_volumes

        if n_volumes > 1 and len(ro_time) == 1:
            ro_time *= n_volumes

        pe_info = []
        for volid in range(n_volumes):
            pe_axis = "ijk".index(pe_dir[volid][0])
            axis_flip = axcodes[pe_axis] in ("LPI")
            pe_flip = pe_dir[volid].endswith("-")

            pe_info.append((
                pe_axis,
                # Displacements are reversed if either is true (after ensuring positive cosines)
                -ro_time[volid] if (axis_flip ^ pe_flip) else ro_time[volid],
            ))

        # Reference image's voxel coordinates (in voxel units)
        voxcoords = (
            nt.linear.Affine(reference=moving)
            .reference.ndindex.reshape((ndim, *data.shape[:ndim]))
            .astype("float32")
        )

        # Convert head-motion transforms to voxel-to-voxel:
        if xfms is not None:
            # if len(xfms) != n_volumes:
            #     raise RuntimeError(
            #         f"Number of head-motion estimates ({len(xfms)}) does not match the "
            #         f"number of volumes ({n_volumes})"
            #     )
            # vox2ras = moving.affine.copy()
            # ras2vox = np.linalg.inv(vox2ras)
            # xfms = [ras2vox @ xfm @ vox2ras for xfm in xfms]
            xfms = None
            warn(
                "Head-motion compensating (realignment) transforms are ignored when applying "
                "the unwarp with SDCFlows. This feature will be enabled as soon as unit tests "
                "are implemented for its quality assurance."
            )

        # Resample
        resampled = asyncio.run(
            unwarp_parallel(
                data,
                voxcoords,
                self.mapped.get_fdata(dtype="float32"),  # fieldmap in Hz
                pe_info,
                xfms,
                jacobian,
                output_dtype='float32',
                order=order,
                mode=mode,
                cval=cval,
                prefilter=prefilter,
                max_concurrent=num_threads or min(os.cpu_count(), 12),
            )
        )

        if not allow_negative:
            resampled[resampled < 0] = cval

        moved = moving.__class__(resampled, moving.affine, moving.header)
        moved.header.set_data_dtype(output_dtype)
        return reorient_image(moved, axcodes)

    def to_displacements(self, ro_time, pe_dir, itk_format=True):
        """
        Generate a NIfTI file containing a displacements field transform compatible with ITK/ANTs.

        The displacements field can be calculated following
        `Eq. (2) in the fieldmap fitting section
        <sdcflows.workflows.fit.fieldmap.html#mjx-eqn-eq%3Afieldmap-2>`__.

        Parameters
        ----------
        ro_time : :obj:`float`
            The total readout time in seconds (only if ``vsm=False``).
        pe_dir : :obj:`str`
            The ``PhaseEncodingDirection`` metadata value (only if ``vsm=False``).

        Returns
        -------
        spatialimage : :obj:`nibabel.nifti.Nifti1Image`
            A NIfTI 1.0 object containing the distortion.

        """
        return fmap_to_disp(self.mapped, ro_time, pe_dir, itk_format=itk_format)


def fmap_to_disp(fmap_nii, ro_time, pe_dir, itk_format=True):
    """
    Convert a fieldmap in Hz into an ITK/ANTs-compatible displacements field.

    The displacements field can be calculated following
    `Eq. (2) in the fieldmap fitting section
    <sdcflows.workflows.fit.fieldmap.html#mjx-eqn-eq%3Afieldmap-2>`__.

    Parameters
    ----------
    fmap_nii : :obj:`os.pathlike`
        Path to a voxel-shift-map (VSM) in NIfTI format
    ro_time : :obj:`float`
        The total readout time in seconds
    pe_dir : :obj:`str`
        The ``PhaseEncodingDirection`` metadata value

    Returns
    -------
    spatialimage : :obj:`nibabel.nifti.Nifti1Image`
        A NIfTI 1.0 object containing the distortion.

    """
    # Set polarity & scale VSM (voxel-shift-map) by readout time
    vsm = fmap_nii.get_fdata().copy() * (-ro_time if pe_dir.endswith("-") else ro_time)

    # Shape of displacements field
    # Note that ITK NIfTI fields are 5D (have an empty 4th dimension)
    fieldshape = vsm.shape[:3] + (1, 3)

    # Convert VSM to voxel displacements
    pe_axis = "ijk".index(pe_dir[0])
    ijk_deltas = np.zeros((vsm.size, 3), dtype="float32")
    ijk_deltas[:, pe_axis] = vsm.reshape(-1)

    # To convert from VSM to RAS field we just apply the affine
    aff = fmap_nii.affine.copy()
    aff[:3, 3] = 0  # Translations MUST NOT be applied, though.
    xyz_deltas = nb.affines.apply_affine(aff, ijk_deltas)
    if itk_format:
        # ITK displacement vectors are in LPS orientation
        xyz_deltas[..., (0, 1)] *= -1.0

    xyz_nii = nb.Nifti1Image(xyz_deltas.reshape(fieldshape), fmap_nii.affine)
    xyz_nii.header.set_intent("vector", name="SDC")
    xyz_nii.header.set_xyzt_units("mm")
    return xyz_nii


def disp_to_fmap(xyz_nii, epi_nii, ro_time, pe_dir, itk_format=True):
    """
    Convert a displacements field into a fieldmap in Hz.

    This is the inverse operation to the previous function.

    Parameters
    ----------
    xyz_nii : :class:`nibabel.Nifti1Image`
        Displacements field in NIfTI format.
    epi_nii : :class:`nibabel.Nifti1Image`
        Source EPI image.
    ro_time : :obj:`float`
        The total readout time of the EPI image in seconds.
    pe_dir : :obj:`str`
        The ``PhaseEncodingDirection`` metadata value of the EPI image.

    Returns
    -------
    spatialimage : :obj:`nibabel.nifti.Nifti1Image`
        A NIfTI 1.0 object containing the field in Hz.

    """
    xyz_deltas = np.squeeze(xyz_nii.get_fdata(dtype="float32")).reshape((-1, 3))

    if itk_format:
        # ITK displacement vectors are in LPS orientation
        xyz_deltas[:, (0, 1)] *= -1

    # Use the EPI's affine to determine voxel spacing, axis ordering and flips
    inv_aff = np.linalg.inv(epi_nii.affine)
    inv_mat = inv_aff[:3, :3]

    # Convert displacements from mm to voxel units
    # Using the inverse affine accounts for reordering of axes, etc.
    ijk_deltas = (xyz_deltas @ inv_mat.T).astype("float32")
    pe_axis = "ijk".index(pe_dir[0])
    vsm = ijk_deltas[:, pe_axis].reshape(xyz_nii.shape[:3])
    scale_factor = -ro_time if pe_dir.endswith("-") else ro_time

    fmap_nii = nb.Nifti1Image(vsm / scale_factor, xyz_nii.affine)
    fmap_nii.header.set_intent("estimate", name="Delta_B0 [Hz]")
    fmap_nii.header.set_xyzt_units("mm")
    fmap_nii.header["cal_max"] = max(
        (
            abs(np.asanyarray(fmap_nii.dataobj).min()),
            np.asanyarray(fmap_nii.dataobj).max(),
        )
    )
    fmap_nii.header["cal_min"] = -fmap_nii.header["cal_max"]
    return fmap_nii


def grid_bspline_weights(target_nii, ctrl_nii, dtype="float32"):
    r"""
    Evaluate tensor-product B-Spline weights on a grid.

    .. _bspline-tensor:

    For each of the *N* input samples :math:`(s_1, s_2, s_3)` and *K* control
    points or *knots* :math:`\mathbf{k} =(k_1, k_2, k_3)`, the tensor-product
    cubic B-Spline kernel weights are calculated:

    .. math::

        \Psi^3(\mathbf{k}, \mathbf{s}) =
        \beta^3(s_1 - k_1) \cdot \beta^3(s_2 - k_2) \cdot \beta^3(s_3 - k_3),
        \label{eq:2}\tag{2}

    where each :math:`\beta^3` represents the cubic B-Spline for one dimension.
    The 1D B-Spline kernel implementation uses :obj:`numpy.piecewise`, and is based on the
    closed-form given by Eq. (6) of [Unser1999]_.

    By iterating over dimensions, the data samples that fall outside of the compact
    support of the tensor-product kernel associated to each control point can be filtered
    out and dismissed to lighten computation.

    Finally, the resulting weights matrix :math:`\Psi^3(\mathbf{k}, \mathbf{s})` can easily be
    identified in `Eq. (1) <sdcflows.interfaces.bspline.html#bspline-interpolation>`_,
    and used as the design matrix for approximation of data.

    Parameters
    ----------
    target_nii :  :obj:`nibabel.spatialimages`
        An spatial image object (typically, a :obj:`~nibabel.nifti1.Nifti1Image`)
        embedding the target EPI image to be corrected.
        Provides the location of the *N* samples (total number of voxels) in the space.
    ctrl_nii : :obj:`nibabel.spatialimages`
        An spatial image object (typically, a :obj:`~nibabel.nifti1.Nifti1Image`)
        embedding the location of the control points of the B-Spline grid.
        The data array should contain a total of :math:`K` knots (control points).

    Returns
    -------
    weights : :obj:`numpy.ndarray` (:math:`N \times K`)
        A sparse matrix of interpolating weights :math:`\Psi^3(\mathbf{k}, \mathbf{s})`
        for the *N* voxels of the target EPI, for each of the total *K* knots.
        This sparse matrix can be directly used as design matrix for the fitting
        step of approximation/extrapolation.

    """
    sample_shape = target_nii.shape[:3]
    knots_shape = ctrl_nii.shape[:3]

    # Ensure the cross-product of affines is near zero (i.e., both coordinate systems are aligned)
    if not np.allclose(
        np.linalg.norm(
            np.cross(ctrl_nii.affine[:-1, :-1].T, target_nii.affine[:-1, :-1].T),
            axis=1,
        ),
        0,
        atol=1e-3,
    ):
        warn("Image's and B-Spline's grids are not aligned.")

    target_to_grid = np.linalg.inv(ctrl_nii.affine) @ target_nii.affine
    wd = []
    for axis in range(3):
        # 3D ijk coordinates of current axis
        coords = np.zeros((3, sample_shape[axis]), dtype=dtype)
        coords[axis] = np.arange(sample_shape[axis], dtype=dtype)

        # Calculate the index component of samples w.r.t. B-Spline knots along current axis
        # Size of locations is L
        locs = nb.affines.apply_affine(target_to_grid, coords.T)[:, axis]

        # Size of knots is K + 6 so that all locations are fully covered by basis
        knots = np.arange(-3, knots_shape[axis] + 3, dtype=dtype)

        bspl = BSpline(knots, np.eye(len(knots) - 3 - 1), 3)

        # Construct a sparse design matrix (L, K)
        distance = np.abs(locs[..., np.newaxis] - knots[np.newaxis, 3:-3])
        within_support = distance < 2.0

        colloc_ax = lil_array(distance.shape, dtype=dtype)
        colloc_ax[within_support] = bspl(locs)[:, 1:-1][within_support]

        # Convert to CSR for efficient multiplication
        wd.append(colloc_ax.tocsr())

    # Calculate the tensor product of the three design matrices
    return kron(kron(wd[0], wd[1]), wd[2])


def _move_coeff(in_coeff, fmap_ref, transform, fmap_target=None):
    """Read in a rigid transform from ANTs, and update the coefficients field affine."""
    xfm = nt.linear.Affine(
        nt.io.itk.ITKLinearTransform.from_filename(transform).to_ras(),
        reference=fmap_ref,
    )
    coeff = nb.load(in_coeff)
    hdr = coeff.header.copy()

    if fmap_target is not None:  # Debug mode: generate fieldmap reference
        nii_target = nb.load(fmap_target)
        debug_ref = (~xfm).apply(fmap_ref, reference=nii_target)
        debug_ref.header.set_qform(nii_target.affine, code=1)
        debug_ref.header.set_sform(nii_target.affine, code=1)
        debug_ref.to_filename(Path() / "debug_fmapref.nii.gz")

    # Generate a new transform
    newaff = np.linalg.inv(np.linalg.inv(coeff.affine) @ (~xfm).matrix)

    # Prepare output file
    hdr.set_qform(newaff, code=1)
    hdr.set_sform(newaff, code=1)

    # Make it easy on viz software to render proper range
    hdr["cal_max"] = max(
        (
            abs(np.asanyarray(coeff.dataobj).min()),
            np.asanyarray(coeff.dataobj).max(),
        )
    )
    hdr["cal_min"] = -hdr["cal_max"]
    return coeff.__class__(coeff.dataobj, newaff, hdr)
