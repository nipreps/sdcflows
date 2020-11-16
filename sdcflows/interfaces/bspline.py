# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
B-Spline filtering.

    .. testsetup::

        >>> tmpdir = getfixture('tmpdir')
        >>> tmp = tmpdir.chdir() # changing to a temporary directory
        >>> nb.Nifti1Image(np.zeros((90, 90, 60)), None, None).to_filename(
        ...     tmpdir.join('epi.nii.gz').strpath)

"""
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
        1e-4, usedefault=True, desc="controls the regularization"
    )
    recenter = traits.Enum(
        "mode",
        "median",
        "mean",
        "no",
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
    """
    Approximate the field to smooth it removing spikes and extrapolating beyond the brain mask.

    Examples
    --------

    """

    input_spec = _BSplineApproxInputSpec
    output_spec = _BSplineApproxOutputSpec

    def _run_interface(self, runtime):
        from sklearn import linear_model as lm

        # Load in the fieldmap
        fmapnii = nb.load(self.inputs.in_data)
        data = fmapnii.get_fdata(dtype="float32")
        oriented_nii = canonical_orientation(fmapnii)
        oriented_nii.to_filename("data.nii.gz")
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
        fmap_points = apply_affine(
            oriented_nii.affine.astype("float32"), mask_indices
        )

        # Calculate the spatial location of control points
        bs_levels = []
        w_l = []
        ncoeff = []
        for sp in bs_spacing:
            level = bspline_grid(oriented_nii, control_zooms_mm=sp)
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
        nb.Nifti1Image(interp_data, fmapnii.affine, hdr).to_filename(out_name)
        self._results["out_field"] = out_name

        index = 0
        self._results["out_coeff"] = []
        for i, (n, bsl) in enumerate(zip(ncoeff, bs_levels)):
            out_level = out_name.replace("_field.", f"_coeff{i:03}.")
            nb.Nifti1Image(
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
        nb.Nifti1Image(
            data * mask - interp_data, fmapnii.affine, fmapnii.header
        ).to_filename(self._results["out_error"])

        if not self.inputs.extrapolate:
            return runtime

        bg_indices = np.argwhere(~mask)
        bg_points = apply_affine(
            oriented_nii.affine.astype("float32"), bg_indices
        )

        extrapolators = np.vstack(
            [bspline_weights(bg_points, level) for level in bs_levels]
        )
        interp_data[~mask] = np.array(model.coef_) @ extrapolators  # Extrapolation
        self._results["out_extrapolated"] = out_name.replace("_field.", "_extra.")
        nb.Nifti1Image(interp_data, fmapnii.affine, hdr).to_filename(
            self._results["out_extrapolated"]
        )
        return runtime


def canonical_orientation(img):
    """Generate an alternative image aligned with the array axes."""
    if isinstance(img, (str, Path)):
        img = nb.load(img)

    shape = np.array(img.shape[:3])
    affine = np.diag(np.hstack((img.header.get_zooms()[:3], 1)))
    affine[:3, 3] -= affine[:3, :3] @ (0.5 * (shape - 1))
    nii = nb.Nifti1Image(img.dataobj, affine)
    nii.header.set_xyzt_units(*img.header.get_xyzt_units())
    return nii


def bspline_grid(img, control_zooms_mm=DEFAULT_ZOOMS_MM):
    """Calculate a Nifti1Image object encoding the location of control points."""
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
    im_center = img.affine @ np.hstack((0.5 * (im_shape - 1), 1))
    bs_center = bs_affine @ np.hstack((0.5 * (bs_shape - 1), 1))
    bs_affine[:3, 3] = im_center[:3] - bs_center[:3]

    return nb.Nifti1Image(np.zeros(bs_shape, dtype="float32"), bs_affine)


def bspline_weights(points, level):
    """Calculate the tensor-product cubic B-Spline weights for a list of 3D points."""
    ctl_spacings = [float(sp) for sp in level.header.get_zooms()[:3]]
    ncoeff = level.dataobj.size
    ctl_points = apply_affine(
        level.affine.astype("float32"), np.argwhere(np.isclose(level.dataobj, 0))
    )

    weights = np.ones((ncoeff, points.shape[0]), dtype="float32")
    for i in range(3):
        d = (
            np.abs(
                (ctl_points[:, np.newaxis, i] - points[np.newaxis, :, i])[
                    weights > 1e-6
                ]
            )
            / ctl_spacings[i]
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
