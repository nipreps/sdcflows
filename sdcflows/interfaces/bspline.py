# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Filtering of :math:`B_0` field mappings with B-Splines."""
from pathlib import Path
import numpy as np
import nibabel as nb
from nibabel.affines import apply_affine

from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    SimpleInterface,
    InputMultiObject,
    OutputMultiObject,
)


LOW_MEM_BLOCK_SIZE = 1000
DEFAULT_ZOOMS_MM = (40.0, 40.0, 20.0)       # For human adults (mid-frequency), in mm
DEFAULT_LF_ZOOMS_MM = (100.0, 100.0, 40.0)  # For human adults (low-frequency), in mm
DEFAULT_HF_ZOOMS_MM = (16.0, 16.0, 10.0)    # For human adults (high-frequency), in mm


class _BSplineApproxInputSpec(BaseInterfaceInputSpec):
    in_data = File(exists=True, mandatory=True, desc="path to a fieldmap")
    in_mask = File(exists=True, mandatory=True, desc="path to a brain mask")
    bs_spacing = InputMultiObject(
        [DEFAULT_ZOOMS_MM],
        traits.Tuple(traits.Float, traits.Float, traits.Float),
        usedefault=True,
        desc="spacing between B-Spline control points",
    )
    ridge_alpha = traits.Float(
        0.01, usedefault=True, desc="controls the regularization"
    )
    recenter = traits.Enum(
        "mode",
        "median",
        "mean",
        False,
        usedefault=True,
        desc="strategy to recenter the distribution of the input fieldmap",
    )
    extrapolate = traits.Bool(
        True,
        usedefault=True,
        desc="generate a field, extrapolated outside the brain mask",
    )


class _BSplineApproxOutputSpec(TraitedSpec):
    out_field = File(exists=True)
    out_coeff = OutputMultiObject(File(exists=True))
    out_error = File(exists=True)
    out_extrapolated = File()


class BSplineApprox(SimpleInterface):
    r"""
    Approximate the :math:`B_0` field using tensor-product B-Splines.

    The approximation effectively smooths the data, removing spikes and other
    sources of noise, as well as enables the extrapolation of the :math:`B_0` field
    beyond the brain mask, which alleviates boundary effects in correction.

    This interface resolves the optimization problem of obtaining the B-Spline coefficients
    :math:`c(\mathbf{k})` that best approximate the data samples within the
    brain mask :math:`f(\mathbf{s})`, following Eq. (17) -- in that case for 2D --
    of [Unser1999]_.
    Here, and adapted to 3D:

    .. math::

        f(\mathbf{s}) =
        \sum_{k_1} \sum_{k_2} \sum_{k_3} c(\mathbf{k}) \Psi^3(\mathbf{k}, \mathbf{s}).
        \label{eq:1}\tag{1}

    References
    ----------
    .. [Unser1999] M. Unser, "`Splines: A Perfect Fit for Signal and Image Processing
        <http://bigwww.epfl.ch/publications/unser9902.pdf>`__," IEEE Signal Processing
        Magazine 16(6):22-38, 1999.

    See Also
    --------
    :py:func:`bspline_weights` - for Eq. :math:`\eqref{eq:2}` and the evaluation of
    the tri-cubic B-Splines :math:`\Psi^3(\mathbf{k}, \mathbf{s})`.

    """

    input_spec = _BSplineApproxInputSpec
    output_spec = _BSplineApproxOutputSpec

    def _run_interface(self, runtime):
        from sklearn import linear_model as lm

        # Load in the fieldmap
        fmapnii = nb.load(self.inputs.in_data)
        data = fmapnii.get_fdata(dtype="float32")
        mask = nb.load(self.inputs.in_mask).get_fdata() > 0
        bs_spacing = [np.array(sp, dtype="float32") for sp in self.inputs.bs_spacing]

        # Recenter the fieldmap
        if self.inputs.recenter == "mode":
            from scipy.stats import mode

            data -= mode(data[mask], axis=None)[0][0]
        elif self.inputs.recenter == "median":
            data -= np.median(data[mask])
        elif self.inputs.recenter == "mean":
            data -= np.mean(data[mask])

        # Calculate spatial location of voxels, and normalize per B-Spline grid
        mask_indices = np.argwhere(mask)
        fmap_points = apply_affine(fmapnii.affine.astype("float32"), mask_indices)

        # Calculate the spatial location of control points
        bs_levels = []
        w_l = []
        ncoeff = []
        for sp in bs_spacing:
            level = bspline_grid(fmapnii, control_zooms_mm=sp)
            bs_levels.append(level)
            ncoeff.append(level.dataobj.size)
            w_l.append(bspline_weights(fmap_points, level))

        # Compose the interpolation matrix
        regressors = np.vstack(w_l)

        # Fit the model
        model = lm.Ridge(alpha=self.inputs.ridge_alpha, fit_intercept=False)
        model.fit(regressors.T, data[mask])

        interp_data = np.zeros_like(data)
        interp_data[mask] = np.array(model.coef_) @ regressors  # Interpolation

        # Store outputs
        out_name = fname_presuffix(
            self.inputs.in_data, suffix="_field", newpath=runtime.cwd
        )
        hdr = fmapnii.header.copy()
        hdr.set_data_dtype("float32")
        fmapnii.__class__(interp_data, fmapnii.affine, hdr).to_filename(out_name)
        self._results["out_field"] = out_name

        index = 0
        self._results["out_coeff"] = []
        for i, (n, bsl) in enumerate(zip(ncoeff, bs_levels)):
            out_level = out_name.replace("_field.", f"_coeff{i:03}.")
            bsl.__class__(
                np.array(model.coef_, dtype="float32")[index:index + n].reshape(
                    bsl.shape
                ),
                bsl.affine,
                bsl.header,
            ).to_filename(out_level)
            index += n
            self._results["out_coeff"].append(out_level)

        # Write out fitting-error map
        self._results["out_error"] = out_name.replace("_field.", "_error.")
        fmapnii.__class__(
            data * mask - interp_data, fmapnii.affine, fmapnii.header
        ).to_filename(self._results["out_error"])

        if not self.inputs.extrapolate:
            return runtime

        bg_indices = np.argwhere(~mask)
        bg_points = apply_affine(fmapnii.affine.astype("float32"), bg_indices)

        extrapolators = np.vstack(
            [bspline_weights(bg_points, level) for level in bs_levels]
        )
        interp_data[~mask] = np.array(model.coef_) @ extrapolators  # Extrapolation
        self._results["out_extrapolated"] = out_name.replace("_field.", "_extra.")
        fmapnii.__class__(interp_data, fmapnii.affine, hdr).to_filename(
            self._results["out_extrapolated"]
        )
        return runtime


class _Coefficients2WarpInputSpec(BaseInterfaceInputSpec):
    in_target = File(exist=True, mandatory=True, desc="input EPI data to be corrected")
    in_coeff = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc="input coefficients, after alignment to the EPI data",
    )
    ro_time = traits.Float(mandatory=True, desc="EPI readout time (s).")
    pe_dir = traits.Enum(
        "i",
        "i-",
        "j",
        "j-",
        "k",
        "k-",
        mandatory=True,
        desc="the phase-encoding direction corresponding to in_target",
    )
    low_mem = traits.Bool(
        False, usedefault=True, desc="perform on low-mem fingerprint regime"
    )


class _Coefficients2WarpOutputSpec(TraitedSpec):
    out_field = File(exists=True)
    out_warp = File(exists=True)


class Coefficients2Warp(SimpleInterface):
    r"""
    Convert a set of B-Spline coefficients to a full displacements map.

    Implements Eq. :math:`\eqref{eq:1}`, interpolating :math:`f(\mathbf{s})`
    for all voxels in the target-image's extent.
    When the readout time is known, the displacements field can be calculated
    following `Eq. (2) in the fieldmap fitting section
    <sdcflows.workflows.fit.fieldmap.html#mjx-eqn-eq%3Afieldmap-2>`__.

    """

    input_spec = _Coefficients2WarpInputSpec
    output_spec = _Coefficients2WarpOutputSpec

    def _run_interface(self, runtime):
        # Calculate the physical coordinates of target grid
        targetnii = nb.load(self.inputs.in_target)
        targetaff = targetnii.affine
        allmask = np.ones_like(targetnii.dataobj, dtype="uint8")
        voxels = np.argwhere(allmask == 1).astype("float32")
        points = apply_affine(targetaff.astype("float32"), voxels)

        weights = []
        coeffs = []
        blocksize = LOW_MEM_BLOCK_SIZE if self.inputs.low_mem else len(points)
        for cname in self.inputs.in_coeff:
            cnii = nb.load(cname)
            cdata = cnii.get_fdata(dtype="float32")
            coeffs.append(cdata.reshape(-1))

            idx = 0
            block_w = []
            while True:
                end = idx + blocksize
                subsample = points[idx:end, ...]
                if subsample.shape[0] == 0:
                    break

                idx = end
                block_w.append(bspline_weights(subsample, cnii))

            weights.append(np.hstack(block_w))

        data = np.zeros(targetnii.shape, dtype="float32")
        data[allmask == 1] = np.squeeze(np.vstack(coeffs).T) @ np.vstack(weights)

        hdr = targetnii.header.copy()
        hdr.set_data_dtype("float32")
        self._results["out_field"] = fname_presuffix(
            self.inputs.in_target, suffix="_field", newpath=runtime.cwd
        )
        targetnii.__class__(data, targetnii.affine, hdr).to_filename(
            self._results["out_field"]
        )

        # Generate warp field
        phaseEncDim = "ijk".index(self.inputs.pe_dir[0])
        phaseEncSign = [1.0, -1.0][len(self.inputs.pe_dir) != 2]

        data *= phaseEncSign * self.inputs.ro_time

        fieldshape = tuple(list(data.shape[:3]) + [3])
        self._results["out_warp"] = fname_presuffix(
            self.inputs.in_target, suffix="_xfm", newpath=runtime.cwd
        )
        # Compose a vector field
        field = np.zeros((data.size, 3), dtype="float32")
        field[..., phaseEncDim] = data.reshape(-1)
        aff = targetnii.affine.copy()
        aff[:3, 3] = 0.0
        field = nb.affines.apply_affine(aff, field).reshape(fieldshape)
        warpnii = targetnii.__class__(
            field[:, :, :, np.newaxis, :].astype("float32"), targetnii.affine, None
        )
        warpnii.header.set_intent("vector", (), "")
        warpnii.header.set_xyzt_units("mm")
        warpnii.to_filename(self._results["out_warp"])
        return runtime


class _TransformCoefficientsInputSpec(BaseInterfaceInputSpec):
    in_coeff = InputMultiObject(
        File(exist=True), mandatory=True, desc="input coefficients file(s)"
    )
    fmap_ref = File(exists=True, mandatory=True, desc="the fieldmap reference")
    transform = File(exists=True, mandatory=True, desc="rigid-body transform file")


class _TransformCoefficientsOutputSpec(TraitedSpec):
    out_coeff = OutputMultiObject(File(exists=True), desc="moved coefficients")


class TransformCoefficients(SimpleInterface):
    """Project coefficients files to another space through a rigid-body transform."""

    input_spec = _TransformCoefficientsInputSpec
    output_spec = _TransformCoefficientsOutputSpec

    def _run_interface(self, runtime):
        self._results["out_coeff"] = _move_coeff(
            self.inputs.in_coeff, self.inputs.fmap_ref, self.inputs.transform,
        )
        return runtime


def bspline_grid(img, control_zooms_mm=DEFAULT_ZOOMS_MM):
    """Create a :obj:`~nibabel.nifti1.Nifti1Image` embedding the location of control points."""
    if isinstance(img, (str, Path)):
        img = nb.load(img)

    im_zooms = np.array(img.header.get_zooms())
    im_shape = np.array(img.shape[:3])

    # Calculate the direction cosines of the target image
    dir_cos = img.affine[:3, :3] / im_zooms

    # Initialize the affine of the B-Spline grid
    bs_affine = np.eye(4)
    bs_affine[:3, :3] = np.array(control_zooms_mm) * dir_cos
    bs_zooms = nb.affines.voxel_sizes(bs_affine)

    # Calculate the shape of the B-Spline grid
    im_extent = im_zooms * (im_shape - 1)
    bs_shape = (im_extent // bs_zooms + 3).astype(int)

    # Center both images
    bs_affine[:3, 3] = apply_affine(img.affine, 0.5 * (im_shape - 1)) - apply_affine(
        bs_affine, 0.5 * (bs_shape - 1)
    )

    return img.__class__(np.zeros(bs_shape, dtype="float32"), bs_affine)


def bspline_weights(points, ctrl_nii):
    r"""
    Calculate the tensor-product cubic B-Spline kernel weights for a list of 3D points.

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
    points : :obj:`numpy.ndarray`; :math:`N \times 3`
        Array of 3D coordinates of samples from the data to be approximated,
        in index (i,j,k) coordinates with respect to the control points grid.
    ctrl_nii : :obj:`nibabel.spatialimages`
        An spatial image object (typically, a :obj:`~nibabel.nifti1.Nifti1Image`)
        embedding the location of the control points of the B-Spline grid.
        The data array should contain a total of :math:`K` knots (control points).

    Returns
    -------
    weights : :obj:`numpy.ndarray` (:math:`K \times N`)
        A sparse matrix of interpolating weights :math:`\Psi^3(\mathbf{k}, \mathbf{s})`
        for the *N* samples in ``points``, for each of the total *K* knots.
        This sparse matrix can be directly used as design matrix for the fitting
        step of approximation/extrapolation.

    """
    ncoeff = np.prod(ctrl_nii.shape[:3])
    knots = np.argwhere(np.ones(ctrl_nii.shape[:3], dtype="uint8") == 1)
    ctl_points = apply_affine(np.linalg.inv(ctrl_nii.affine).astype("float32"), points)

    weights = np.ones((ncoeff, points.shape[0]), dtype="float32")
    for i in range(3):
        d = np.abs(
            (knots[:, np.newaxis, i].astype("float32") - ctl_points[np.newaxis, :, i])[
                weights > 1e-6
            ]
        )
        weights[weights > 1e-6] *= np.piecewise(
            d,
            [d >= 2.0, d < 1.0, (d >= 1.0) & (d < 2)],
            [
                0.0,
                lambda d: (4.0 - 6.0 * d ** 2 + 3.0 * d ** 3) / 6.0,
                lambda d: (2.0 - d) ** 3 / 6.0,
            ],
        )

    weights[weights < 1e-6] = 0.0
    return weights


def _move_coeff(in_coeff, fmap_ref, transform):
    """Read in a rigid transform from ANTs, and update the coefficients field affine."""
    from pathlib import Path
    import nibabel as nb
    import nitransforms as nt

    if isinstance(in_coeff, str):
        in_coeff = [in_coeff]

    xfm = nt.linear.Affine(
        nt.io.itk.ITKLinearTransform.from_filename(transform).to_ras(),
        reference=fmap_ref,
    )

    out = []
    for i, c in enumerate(in_coeff):
        out.append(str(Path(f"moved_coeff_{i:03d}.nii.gz").absolute()))
        img = nb.load(c)
        newaff = xfm.matrix @ img.affine
        img.__class__(img.dataobj, newaff, img.header).to_filename(out[-1])

    return out
