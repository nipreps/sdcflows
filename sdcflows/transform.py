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
"""The :math:`B_0` unwarping transform formalism."""
from pathlib import Path

import attr
import numpy as np
from warnings import warn
from scipy import ndimage as ndi
from scipy.signal import cubic
from scipy.sparse import vstack as sparse_vstack, kron, lil_array

import nibabel as nb
import nitransforms as nt
from nitransforms.base import _as_homogeneous
from bids.utils import listify

from niworkflows.interfaces.nibabel import reorient_image


def _clear_mapped(instance, attribute, value):
    instance.mapped = None
    return value


@attr.s(slots=True)
class B0FieldTransform:
    """Represents and applies the transform to correct for susceptibility distortions."""

    coeffs = attr.ib(default=None)
    """B-Spline coefficients (one value per control point)."""
    xfm = attr.ib(default=None, on_setattr=_clear_mapped)
    """A rigid-body transform to prepend to the unwarping displacements field."""
    mapped = attr.ib(default=None, init=False)
    """
    A cache of the interpolated field in Hz (i.e., the fieldmap *mapped* on to the
    target image we want to correct).
    """

    def fit(self, spatialimage):
        r"""
        Generate the interpolation matrix (and the VSM with it).

        Implements Eq. :math:`\eqref{eq:1}`, interpolating :math:`f(\mathbf{s})`
        for all voxels in the target-image's extent.

        Returns
        -------
        updated : :obj:`bool`
            ``True`` if the internal field representation was fit,
            ``False`` if cache was valid and will be reused.

        """
        # Calculate the physical coordinates of target grid
        if isinstance(spatialimage, (str, bytes, Path)):
            spatialimage = nb.load(spatialimage)

        # Initialize xform (or set identity)
        xfm = self.xfm if self.xfm is not None else nt.linear.Affine()

        if self.mapped is not None:
            newaff = spatialimage.affine
            newshape = spatialimage.shape

            if np.all(newshape == self.mapped.shape) and np.allclose(
                newaff, self.mapped.affine
            ):
                return False

        weights = []
        coeffs = []

        # Generate tensor-product B-Spline weights
        for level in listify(self.coeffs):
            xfm.reference = spatialimage
            moved_cs = level.__class__(
                level.dataobj, xfm.matrix @ level.affine, level.header
            )
            wmat = grid_bspline_weights(spatialimage, moved_cs)
            weights.append(wmat)
            coeffs.append(level.get_fdata(dtype="float32").reshape(-1))

        # Interpolate the VSM (voxel-shift map)
        vsm = np.zeros(spatialimage.shape[:3], dtype="float32")
        vsm = (np.squeeze(np.hstack(coeffs).T) @ sparse_vstack(weights)).reshape(
            vsm.shape
        )

        # Cache
        hdr = spatialimage.header.copy()
        hdr.set_intent("estimate", name="Voxel shift")
        hdr.set_data_dtype("float32")
        hdr["cal_max"] = max((abs(vsm.min()), vsm.max()))
        hdr["cal_min"] = - hdr["cal_max"]
        self.mapped = nb.Nifti1Image(vsm, spatialimage.affine, hdr)
        return True

    def apply(
        self,
        spatialimage,
        pe_dir,
        ro_time,
        order=3,
        mode="constant",
        cval=0.0,
        prefilter=True,
        output_dtype=None,
    ):
        """
        Apply a transformation to an image, resampling on the reference spatial object.

        Parameters
        ----------
        spatialimage : `spatialimage`
            The image object containing the data to be resampled in reference
            space
        reference : spatial object, optional
            The image, surface, or combination thereof containing the coordinates
            of samples that will be sampled.
        order : int, optional
            The order of the spline interpolation, default is 3.
            The order has to be in the range 0-5.
        mode : {'constant', 'reflect', 'nearest', 'mirror', 'wrap'}, optional
            Determines how the input image is extended when the resamplings overflows
            a border. Default is 'constant'.
        cval : float, optional
            Constant value for ``mode='constant'``. Default is 0.0.
        prefilter: bool, optional
            Determines if the image's data array is prefiltered with
            a spline filter before interpolation. The default is ``True``,
            which will create a temporary *float64* array of filtered values
            if *order > 1*. If setting this to ``False``, the output will be
            slightly blurred if *order > 1*, unless the input is prefiltered,
            i.e. it is the result of calling the spline filter on the original
            input.

        Returns
        -------
        resampled : `spatialimage` or ndarray
            The data imaged after resampling to reference space.

        """
        from sdcflows.utils.tools import ensure_positive_cosines

        # Ensure the fmap has been computed
        if isinstance(spatialimage, (str, bytes, Path)):
            spatialimage = nb.load(spatialimage)

        spatialimage, axcodes = ensure_positive_cosines(spatialimage)

        self.fit(spatialimage)
        fmap = self.mapped.get_fdata().copy()

        # Reverse mapped if reversed blips
        if pe_dir.endswith("-"):
            fmap *= -1.0

        # Generate warp field
        pe_axis = "ijk".index(pe_dir[0])

        # Map voxel coordinates applying the VSM
        if self.xfm is None:
            ijk_axis = tuple([np.arange(s) for s in fmap.shape])
            voxcoords = np.array(
                np.meshgrid(*ijk_axis, indexing="ij"), dtype="float32"
            ).reshape(3, -1)
        else:
            # Map coordinates from reference to time-step
            self.xfm.reference = spatialimage
            hmc_xyz = self.xfm.map(self.xfm.reference.ndcoords.T)
            # Convert from RAS to voxel coordinates
            voxcoords = (
                np.linalg.inv(self.xfm.reference.affine)
                @ _as_homogeneous(np.vstack(hmc_xyz), dim=self.xfm.reference.ndim).T
            )[:3, ...]

        # fmap * ro_time is the voxel-shift map (VSM)
        # The VSM is just the displacements field given in index coordinates
        # voxcoords is the deformation field, i.e., the target position of each voxel
        voxcoords[pe_axis, ...] += fmap.reshape(-1) * ro_time

        # Prepare data
        data = np.squeeze(np.asanyarray(spatialimage.dataobj))
        output_dtype = output_dtype or data.dtype

        # Resample
        resampled = ndi.map_coordinates(
            data,
            voxcoords,
            output=output_dtype,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
        ).reshape(spatialimage.shape)

        moved = spatialimage.__class__(
            resampled, spatialimage.affine, spatialimage.header
        )
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


def disp_to_fmap(xyz_nii, ro_time, pe_dir, itk_format=True):
    """
    Convert a displacements field into a fieldmap in Hz.

    This is the inverse operation to the previous function.

    Parameters
    ----------
    xyz_nii : :obj:`os.pathlike`
        Path to a displacements field in NIfTI format.
    ro_time : :obj:`float`
        The total readout time in seconds.
    pe_dir : :obj:`str`
        The ``PhaseEncodingDirection`` metadata value.

    Returns
    -------
    spatialimage : :obj:`nibabel.nifti.Nifti1Image`
        A NIfTI 1.0 object containing the field in Hz.

    """
    xyz_deltas = np.squeeze(xyz_nii.get_fdata(dtype="float32")).reshape((-1, 3))

    if itk_format:
        # ITK displacement vectors are in LPS orientation
        xyz_deltas[:, (0, 1)] *= -1

    inv_aff = np.linalg.inv(xyz_nii.affine)
    inv_aff[:3, 3] = 0  # Translations MUST NOT be applied.

    # Convert displacements from mm to voxel units
    # Using the inverse affine accounts for reordering of axes, etc.
    ijk_deltas = nb.affines.apply_affine(inv_aff, xyz_deltas).astype("float32")
    pe_axis = "ijk".index(pe_dir[0])
    vsm = ijk_deltas[:, pe_axis].reshape(xyz_nii.shape[:3])
    scale_factor = -ro_time if pe_dir.endswith("-") else ro_time

    fmap_nii = nb.Nifti1Image(vsm / scale_factor, xyz_nii.affine)
    fmap_nii.header.set_intent("estimate", name="Delta_B0 [Hz]")
    fmap_nii.header.set_xyzt_units("mm")
    fmap_nii.header["cal_max"] = max((
        abs(np.asanyarray(fmap_nii.dataobj).min()),
        np.asanyarray(fmap_nii.dataobj).max(),
    ))
    fmap_nii.header["cal_min"] = - fmap_nii.header["cal_max"]
    return fmap_nii


def grid_bspline_weights(target_nii, ctrl_nii, dtype="float32"):
    r"""
    Evaluate tensor-product B-Spline weights on a grid.

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

    Finally, the resulting weights matrix :math:`\Psi^3(\mathbf{k}, \mathbf{s})`
    can be easily identified in Eq. :math:`\eqref{eq:1}` and used as the design matrix
    for approximation of data.

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
    weights : :obj:`numpy.ndarray` (:math:`K \times N`)
        A sparse matrix of interpolating weights :math:`\Psi^3(\mathbf{k}, \mathbf{s})`
        for the *N* voxels of the target EPI, for each of the total *K* knots.
        This sparse matrix can be directly used as design matrix for the fitting
        step of approximation/extrapolation.

    """
    sample_shape = target_nii.shape[:3]
    knots_shape = ctrl_nii.shape[:3]

    # Ensure the cross-product of affines is near zero (i.e., both coordinate systems are aligned)
    if not np.allclose(np.linalg.norm(
        np.cross(ctrl_nii.affine[:-1, :-1].T, target_nii.affine[:-1, :-1].T),
        axis=1,
    ), 0, atol=1e-3):
        warn("Image's and B-Spline's grids are not aligned.")

    target_to_grid = np.linalg.inv(ctrl_nii.affine) @ target_nii.affine
    wd = []
    for axis in range(3):
        # 3D ijk coordinates of current axis
        coords = np.zeros((3, sample_shape[axis]), dtype=dtype)
        coords[axis] = np.arange(sample_shape[axis], dtype=dtype)

        # Calculate the index component of samples w.r.t. B-Spline knots along current axis
        locs = nb.affines.apply_affine(target_to_grid, coords.T)[:, axis]
        knots = np.arange(knots_shape[axis], dtype=dtype)

        distance = np.abs(locs[np.newaxis, ...] - knots[..., np.newaxis])
        within_support = distance < 2.0
        d_vals, d_idxs = np.unique(distance[within_support], return_inverse=True)
        bs_w = cubic(d_vals)

        colloc_ax = lil_array((knots_shape[axis], sample_shape[axis]), dtype=dtype)
        colloc_ax[within_support] = bs_w[d_idxs]

        wd.append(colloc_ax)

    # Calculate the tensor product of the three design matrices
    return kron(kron(wd[0], wd[1]), wd[2]).astype(dtype)


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
    hdr["cal_max"] = max((
        abs(np.asanyarray(coeff.dataobj).min()),
        np.asanyarray(coeff.dataobj).max(),
    ))
    hdr["cal_min"] = - hdr["cal_max"]
    return coeff.__class__(coeff.dataobj, newaff, hdr)
