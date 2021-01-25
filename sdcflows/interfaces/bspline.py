# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Filtering of :math:`B_0` field mappings with B-Splines."""
from pathlib import Path
import numpy as np
import nibabel as nb
from nibabel.affines import apply_affine

from nipype import logging
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
DEFAULT_ZOOMS_MM = (40.0, 40.0, 20.0)  # For human adults (mid-frequency), in mm
DEFAULT_LF_ZOOMS_MM = (100.0, 100.0, 40.0)  # For human adults (low-frequency), in mm
DEFAULT_HF_ZOOMS_MM = (16.0, 16.0, 10.0)  # For human adults (high-frequency), in mm
BSPLINE_SUPPORT = 2 - 1.82e-3  # Disallows weights < 1e-9
LOGGER = logging.getLogger("nipype.interface")


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
        from scipy.sparse import vstack as sparse_vstack

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
        ncoeff = []
        regressors = None
        for sp in bs_spacing:
            level = bspline_grid(fmapnii, control_zooms_mm=sp)
            bs_levels.append(level)
            ncoeff.append(level.dataobj.size)

            regressors = (
                bspline_weights(fmap_points, level)
                if regressors is None
                else sparse_vstack((regressors, bspline_weights(fmap_points, level)))
            )

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
                np.array(model.coef_, dtype="float32")[index : index + n].reshape(
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
        if not bg_indices.size:
            self._results["out_extrapolated"] = self._results["out_field"]
            return runtime

        bg_points = apply_affine(fmapnii.affine.astype("float32"), bg_indices)
        extrapolators = sparse_vstack(
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
        from scipy.sparse import vstack as sparse_vstack

        # Calculate the physical coordinates of target grid
        targetnii = nb.load(self.inputs.in_target)
        allmask = np.ones_like(targetnii.dataobj, dtype="uint8")

        weights = []
        coeffs = []

        for cname in self.inputs.in_coeff:
            coeff_nii = nb.load(cname)
            wmat = grid_bspline_weights(targetnii, coeff_nii)
            weights.append(wmat)
            coeffs.append(coeff_nii.get_fdata(dtype="float32").reshape(-1))

        data = np.zeros(targetnii.shape, dtype="float32")
        data[allmask == 1] = np.squeeze(np.vstack(coeffs).T) @ sparse_vstack(weights)

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
        # Multiplying by the affine implicitly applies the voxel size to the shift map
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
            self.inputs.in_coeff,
            self.inputs.fmap_ref,
            self.inputs.transform,
        )
        return runtime


class _TOPUPCoeffReorientInputSpec(BaseInterfaceInputSpec):
    in_coeff = InputMultiObject(
        File(exist=True), mandatory=True, desc="input coefficients file(s) from TOPUP"
    )
    fmap_ref = File(exists=True, mandatory=True, desc="the fieldmap reference")


class _TOPUPCoeffReorientOutputSpec(TraitedSpec):
    out_coeff = OutputMultiObject(File(exists=True), desc="patched coefficients")


class TOPUPCoeffReorient(SimpleInterface):
    """
    Revise the orientation of TOPUP-generated B-Spline coefficients.

    TOPUP-generated "fieldcoeff" files are just B-Spline fields, where the shape
    of the field is fixated to be a decimated grid of the original image by an
    integer factor and added 3 pixels on each dimension.
    This is one root reason why TOPUP errors (FSL 6) or segfaults (FSL 5), when the
    input image has odd number of voxels along one or more directions.

    These "fieldcoeff" are fixated to be zero-centered, and have "plumb" orientation
    (as in, aligned with cardinal/imaging axes).
    The q-form of these NIfTI files is always diagonal, with the decimation factors
    set on the diagonal (and hence, the voxel zooms).
    The origin of the q-form is set to the reference image's shape.

    This interface modifies these coefficient files to be fully-fledged NIfTI images
    aligned with the reference image.
    Therefore, the s-form header of the coefficients file is updated to match that
    of the reference file.
    The s-form header is used because the imaging axes may be oblique.

    The q-form retains the original header and is marked with code 0.

    """

    input_spec = _TOPUPCoeffReorientInputSpec
    output_spec = _TOPUPCoeffReorientOutputSpec

    def _run_interface(self, runtime):
        self._results["out_coeff"] = [
            str(
                _fix_topup_fieldcoeff(
                    in_coeff,
                    self.inputs.fmap_ref,
                    fname_presuffix(in_coeff, suffix="_fixed", newpath=runtime.cwd),
                )
            )
            for in_coeff in self.inputs.in_coeff
        ]
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


def grid_bspline_weights(target_nii, ctrl_nii):
    """Fast, gridded evaluation."""
    from scipy.sparse import csr_matrix, vstack

    if isinstance(target_nii, (str, bytes, Path)):
        target_nii = nb.load(target_nii)
    if isinstance(ctrl_nii, (str, bytes, Path)):
        ctrl_nii = nb.load(ctrl_nii)

    shape = target_nii.shape[:3]
    ctrl_sp = ctrl_nii.header.get_zooms()[:3]
    ras2ijk = np.linalg.inv(ctrl_nii.affine)
    origin = apply_affine(ras2ijk, [tuple(target_nii.affine[:3, 3])])[0]

    wd = []
    for i, (o, n, sp) in enumerate(
        zip(origin, shape, target_nii.header.get_zooms()[:3])
    ):
        locations = np.arange(0, n, dtype="float32") * sp / ctrl_sp[i] + o
        knots = np.arange(0, ctrl_nii.shape[i], dtype="float32")
        distance = (locations[np.newaxis, ...] - knots[..., np.newaxis]).astype(
            "float32"
        )
        weights = np.zeros_like(distance, dtype="float32")
        within_support = np.abs(distance) < 2.0
        d = np.abs(distance[within_support])
        weights[within_support] = np.piecewise(
            d,
            [d < 1.0, d >= 1.0],
            [
                lambda d: (4.0 - 6.0 * d ** 2 + 3.0 * d ** 3) / 6.0,
                lambda d: (2.0 - d) ** 3 / 6.0,
            ],
        )
        wd.append(weights)

    ctrl_shape = ctrl_nii.shape[:3]
    data_size = np.prod(shape)
    wmat = None
    for i in range(ctrl_shape[0]):
        sparse_mat = (
            wd[0][i, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
            * wd[1][np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis]
            * wd[2][np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
        ).reshape((-1, data_size))
        sparse_mat[sparse_mat < 1e-9] = 0

        if wmat is None:
            wmat = csr_matrix(sparse_mat)
        else:
            wmat = vstack((wmat, csr_matrix(sparse_mat)))

    return wmat


def bspline_weights(points, ctrl_nii, blocksize=None, mem_percent=None):
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
    from scipy.sparse import csc_matrix, hstack
    from ..utils.misc import get_free_mem

    if isinstance(ctrl_nii, (str, bytes, Path)):
        ctrl_nii = nb.load(ctrl_nii)
    ncoeff = np.prod(ctrl_nii.shape[:3])
    knots = np.argwhere(np.ones(ctrl_nii.shape[:3], dtype="uint8") == 1)
    ras2ijk = np.linalg.inv(ctrl_nii.affine).astype("float32")

    if blocksize is None:
        blocksize = len(points)

    # Try to probe the free memory
    _free_mem = get_free_mem()
    suggested_blocksize = (
        int(np.round((_free_mem * (mem_percent or 0.9)) / (3 * 4 * ncoeff)))
        if _free_mem
        else blocksize
    )
    blocksize = min(blocksize, suggested_blocksize)
    LOGGER.debug(
        f"Determined a block size of {blocksize}, for interpolating "
        f"an image of {len(points)} voxels with a grid of {ncoeff} "
        f"coefficients ({_free_mem / 1024**3:.2f} GiB free memory)."
    )

    idx = 0
    wmatrix = None
    while True:
        end = idx + blocksize
        subsample = points[idx:end, ...]
        if subsample.shape[0] == 0:
            break

        ctl_points = apply_affine(ras2ijk, subsample)
        weights = np.ones((ncoeff, len(subsample)), dtype="float32")
        for i in range(3):
            nonzeros = weights > 1e-6
            distance = np.squeeze(
                np.abs(
                    (
                        knots[:, np.newaxis, i].astype("float32")
                        - ctl_points[np.newaxis, :, i]
                    )[nonzeros]
                )
            )
            within_support = distance < BSPLINE_SUPPORT
            d = distance[within_support]
            distance[~within_support] = 0
            distance[within_support] = np.piecewise(
                d,
                [d < 1.0, d >= 1.0],
                [
                    lambda d: (4.0 - 6.0 * d ** 2 + 3.0 * d ** 3) / 6.0,
                    lambda d: (2.0 - d) ** 3 / 6.0,
                ],
            )
            weights[nonzeros] *= distance

        weights[weights < 1e-6] = 0.0

        wmatrix = (
            csc_matrix(weights)
            if wmatrix is None
            else hstack((wmatrix, csc_matrix(weights)))
        )
        idx = end
    return wmatrix.tocsr()


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


def _fix_topup_fieldcoeff(in_coeff, fmap_ref, out_file=None):
    """Read in a coefficients file generated by TOPUP and fix x-form headers."""
    from pathlib import Path
    import numpy as np
    import nibabel as nb

    if out_file is None:
        out_file = Path("coefficients.nii.gz").absolute()

    coeffnii = nb.load(in_coeff)
    refnii = nb.load(fmap_ref)

    coeff_shape = np.array(coeffnii.shape[:3])
    ref_shape = np.array(refnii.shape[:3])
    factors = coeffnii.header.get_zooms()[:3]
    if not np.all(coeff_shape == ref_shape // factors + 3):
        raise ValueError(
            f"Shape of coefficients file {coeff_shape} does not meet the "
            f"expectation given the reference's shape {ref_shape}."
        )
    newaff = np.eye(4)
    newaff[:3, :3] = refnii.affine[:3, :3] * factors
    c_ref = nb.affines.apply_affine(refnii.affine, 0.5 * (ref_shape - 1))
    c_coeff = nb.affines.apply_affine(newaff, 0.5 * (coeff_shape - 1))
    newaff[:3, 3] = c_ref - c_coeff
    header = coeffnii.header.copy()
    coeffnii.header.set_qform(coeffnii.header.get_qform(coded=False), code=0)
    coeffnii.header.set_sform(newaff, code=1)

    coeffnii.__class__(coeffnii.dataobj, newaff, header).to_filename(out_file)
    return out_file
