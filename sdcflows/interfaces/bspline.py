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


class _BSplineApproxOutputSpec(TraitedSpec):
    out_field = File(exists=True)
    out_coeff = OutputMultiObject(File(exists=True))


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
        data = fmapnii.get_fdata()
        nsamples = data.size
        mask = nb.load(self.inputs.in_mask).get_fdata() > 0
        bs_spacing = [np.array(sp, dtype="float32") for sp in self.inputs.bs_spacing]

        # Calculate B-Splines grid(s)
        bs_levels = []
        for sp in bs_spacing:
            bs_levels.append(bspline_grid(fmapnii, control_zooms_mm=sp))

        # Calculate spatial location of voxels, and normalize per B-Spline grid
        fmap_points = grid_coords(fmapnii)
        sample_points = []
        for sp in bs_spacing:
            sample_points.append((fmap_points / sp).astype("float32"))

        # Calculate the spatial location of control points
        w_l = []
        ncoeff = []
        for sp, level, points in zip(bs_spacing, bs_levels, sample_points):
            ncoeff.append(level.dataobj.size)
            _w = np.ones((ncoeff[-1], nsamples), dtype="float32")

            _gc = grid_coords(level, control_zooms_mm=sp)

            for i in range(3):
                d = np.abs((_gc[:, np.newaxis, i] - points[np.newaxis, :, i])[_w > 1e-6])
                _w[_w > 1e-6] *= np.piecewise(
                    d,
                    [d >= 2.0, d < 1.0, (d >= 1.0) & (d < 2)],
                    [0.,
                     lambda d: (4. - 6. * d ** 2 + 3. * d ** 3) / 6.,
                     lambda d: (2. - d) ** 3 / 6.]
                )

            _w[_w < 1e-6] = 0.0
            w_l.append(_w)

        # Calculate the cubic spline weights per dimension and tensor-product
        weights = np.vstack(w_l)
        dist_support = weights > 0.0

        # Compose the interpolation matrix
        interp_mat = np.zeros((np.sum(ncoeff), nsamples))
        interp_mat[dist_support] = weights[dist_support]

        # Fit the model
        model = lm.Ridge(alpha=self.inputs.ridge_alpha, fit_intercept=False)
        model.fit(
            interp_mat[..., mask.reshape(-1)].T,  # Regress only within brainmask
            data[mask],
        )

        # Store outputs
        out_name = str(
            Path(
                fname_presuffix(
                    self.inputs.in_data, suffix="_field", newpath=runtime.cwd
                )
            ).absolute()
        )
        hdr = fmapnii.header.copy()
        hdr.set_data_dtype("float32")
        nb.Nifti1Image(
            (model.intercept_ + np.array(model.coef_) @ interp_mat)
            .astype("float32")  # Interpolation
            .reshape(data.shape),
            fmapnii.affine,
            hdr,
        ).to_filename(out_name)
        self._results["out_field"] = out_name

        index = 0
        self._results["out_coeff"] = []
        for i, (n, bsl) in enumerate(zip(ncoeff, bs_levels)):
            out_level = out_name.replace("_field.", f"_coeff{i:03}.")
            nb.Nifti1Image(
                np.array(model.coef_, dtype="float32")[index : index + n].reshape(
                    bsl.shape
                ),
                bsl.affine,
                bsl.header,
            ).to_filename(out_level)
            index += n
            self._results["out_coeff"].append(out_level)
        return runtime


def bspline_grid(img, control_zooms_mm=DEFAULT_ZOOMS_MM):
    """Calculate a Nifti1Image object encoding the location of control points."""
    if isinstance(img, (str, Path)):
        img = nb.load(img)

    im_zooms = np.array(img.header.get_zooms())
    im_shape = np.array(img.shape[:3])

    # Calculate the direction cosines of the target image
    dir_cos = img.affine[:3, :3] / im_zooms

    # Initialize the affine of the B-Spline grid
    bs_affine = np.diag(np.hstack((np.array(control_zooms_mm) @ dir_cos, 1)))
    bs_zooms = nb.affines.voxel_sizes(bs_affine)

    # Calculate the shape of the B-Spline grid
    im_extent = im_zooms * (im_shape - 1)
    bs_shape = (im_extent // bs_zooms + 3).astype(int)

    # Center both images
    im_center = img.affine @ np.hstack((0.5 * (im_shape - 1), 1))
    bs_center = bs_affine @ np.hstack((0.5 * (bs_shape - 1), 1))
    bs_affine[:3, 3] = im_center[:3] - bs_center[:3]

    return nb.Nifti1Image(np.zeros(bs_shape, dtype="float32"), bs_affine)


def grid_coords(img, control_zooms_mm=None, dtype="float32"):
    """Create a linear space of physical coordinates."""
    if isinstance(img, (str, Path)):
        img = nb.load(img)

    grid = np.array(
        np.meshgrid(*[range(s) for s in img.shape[:3]]), dtype=dtype
    ).reshape(3, -1)
    coords = (img.affine @ np.vstack((grid, np.ones(grid.shape[-1])))).T[..., :3]

    if control_zooms_mm is not None:
        coords /= np.array(control_zooms_mm)

    return coords.astype(dtype)
