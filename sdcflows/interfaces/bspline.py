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
"""Filtering of :math:`B_0` field mappings with B-Splines."""
from itertools import product
from pathlib import Path
import numpy as np
import nibabel as nb
from nibabel.affines import apply_affine
from nitransforms.linear import Affine

from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    isdefined,
    SimpleInterface,
    InputMultiObject,
    OutputMultiObject,
)

from sdcflows.transform import grid_bspline_weights
from sdcflows.utils.tools import reorient_pedir


LOW_MEM_BLOCK_SIZE = 1000
DEFAULT_ZOOMS_MM = (40.0, 40.0, 20.0)  # For human adults (mid-frequency), in mm
DEFAULT_LF_ZOOMS_MM = (100.0, 100.0, 40.0)  # For human adults (low-frequency), in mm
DEFAULT_HF_ZOOMS_MM = (16.0, 16.0, 10.0)  # For human adults (high-frequency), in mm
LOGGER = logging.getLogger("nipype.interface")


class _BSplineApproxInputSpec(BaseInterfaceInputSpec):
    in_data = File(exists=True, mandatory=True, desc="path to a fieldmap")
    in_mask = File(exists=True, desc="path to a brain mask")
    bs_spacing = InputMultiObject(
        [DEFAULT_HF_ZOOMS_MM],
        traits.Tuple(traits.Float, traits.Float, traits.Float),
        usedefault=True,
        desc="spacing between B-Spline control points",
    )
    ridge_alpha = traits.Float(
        1e-4, usedefault=True, desc="controls the regularization"
    )
    recenter = traits.Enum(
        "median",
        "mode",
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
    zooms_min = traits.Union(
        traits.Float,
        traits.Tuple(traits.Float, traits.Float, traits.Float),
        default_value=1.0,
        usedefault=True,
        desc="limit minimum image zooms, set 0.0 to use the original image",
    )
    debug = traits.Bool(
        False, usedefault=True, desc="generate extra assets for debugging"
    )


class _BSplineApproxOutputSpec(TraitedSpec):
    out_intercept = traits.Float
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
    of [Unser1999]_. Here, and for the case of 3D, the formulism is adapted in
    `Eq. (1) of the transform module <sdcflows.transform.html#bspline-interpolation>`_.

    References
    ----------
    .. [Unser1999] M. Unser, "`Splines: A Perfect Fit for Signal and Image Processing
        <http://bigwww.epfl.ch/publications/unser9902.pdf>`__," IEEE Signal Processing
        Magazine 16(6):22-38, 1999.

    See Also
    --------
    :py:func:`~sdcflows.transform.grid_bspline_weights` - for the evaluation of
    the tensor-product, cubic B-Splines (:math:`\Psi^3(\mathbf{k}, \mathbf{s})`)
    formalized in
    `Eq. (2) of the transform module <sdcflows.transform.html#bspline-tensor>`_.

    """

    input_spec = _BSplineApproxInputSpec
    output_spec = _BSplineApproxOutputSpec

    def _run_interface(self, runtime):
        from sklearn import linear_model as lm
        from scipy.sparse import hstack as sparse_hstack

        # Output name baseline
        out_name = fname_presuffix(
            self.inputs.in_data, suffix="_field", newpath=runtime.cwd
        )

        # Load in the fieldmap
        fmapnii = nb.load(self.inputs.in_data)
        # fmapnii = nb.as_closest_canonical(fmapnii)
        zooms = fmapnii.header.get_zooms()

        # Get a mask (or define on the spot to cover the full extent)
        masknii = (
            nb.load(self.inputs.in_mask) if isdefined(self.inputs.in_mask) else None
        )
        # if masknii is not None:
        #     masknii = nb.as_closest_canonical(masknii)

        # Determine the shape of bspline coefficients
        # This should not change with resizing, so do it first
        bs_grids = [
            bspline_grid(fmapnii, control_zooms_mm=sp) for sp in self.inputs.bs_spacing
        ]

        need_resize = np.any(np.array(zooms) < self.inputs.zooms_min)
        if need_resize:
            from sdcflows.utils.tools import resample_to_zooms

            target_zooms = np.maximum(zooms, self.inputs.zooms_min)

            LOGGER.info(
                "Resampling image with resolution exceeding 'zooms_min' "
                f"({'x'.join(str(s) for s in zooms)} → "
                f"{'x'.join(str(s) for s in target_zooms)})."
            )
            fmapnii = resample_to_zooms(fmapnii, target_zooms)

            if masknii is not None:
                masknii = resample_to_zooms(masknii, target_zooms)

        data = fmapnii.get_fdata(dtype="float32")

        # Generate a numpy array with the mask
        mask = (
            np.ones(fmapnii.shape, dtype=bool)
            if masknii is None
            else np.asanyarray(masknii.dataobj) > 1e-4
        )

        # Recenter the fieldmap
        center = 0
        if self.inputs.recenter == "mode":
            from scipy.stats import mode

            # Handle pre- and post-1.9 mode behavior.
            # squeeze can be dropped when the minimum version reaches 1.9
            # Will become: data -= mode(data[mask], keepdims=False).mode
            center = np.squeeze(mode(data[mask]).mode)
        elif self.inputs.recenter == "median":
            center = np.median(data[mask])
        elif self.inputs.recenter == "mean":
            center = np.mean(data[mask])

        data -= center
        data[~mask] = 0

        # Calculate collocation matrix from (possibly resized) image and knot grids
        colmat = sparse_hstack(
            [grid_bspline_weights(fmapnii, grid) for grid in bs_grids]
        ).tocsr()

        bs_grids_str = ["x".join(str(s) for s in grid.shape) for grid in bs_grids]
        bs_grids_str[-1] = f"and {bs_grids_str[-1]}"
        LOGGER.info(
            f"Approximating B-Splines grids ({', '.join(bs_grids_str)} [knots]) on a grid of "
            f"{'x'.join(str(s) for s in fmapnii.shape)} ({np.prod(fmapnii.shape)}) voxels,"
            f" of which {mask.sum()} fall within the mask."
        )

        # Fit the model
        model = lm.Ridge(alpha=self.inputs.ridge_alpha, fit_intercept=False)
        for attempt in range(3):
            model.fit(colmat, data.reshape(-1))
            extreme = np.abs(model.coef_).max()
            LOGGER.debug(f"Model fit attempt {attempt}: max(|coeffs|) = {extreme}")
            # Normal values seem to be ~1e2, bad ~1e8. May want to tweak this if
            # these distributions are wider than I think.
            if extreme < 1e4:
                break
        else:
            raise RuntimeError(
                f"Spline fit of input file {self.inputs.in_data} failed. "
                f"Extreme value {extreme:.2e} detected in spline coefficients."
            )

        self._results["out_intercept"] = model.intercept_

        # Store coefficients
        index = 0
        self._results["out_coeff"] = []
        for i, bsl in enumerate(bs_grids):
            n = bsl.dataobj.size
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

        # Interpolating in the original grid will require a new collocation matrix
        if need_resize:
            fmapnii = nb.load(self.inputs.in_data)
            # fmapnii = nb.as_closest_canonical(fmapnii)
            data = fmapnii.get_fdata(dtype="float32") - center
            if masknii is not None:
                masknii = nb.load(self.inputs.in_mask)
                # masknii = nb.as_closest_canonical(masknii)
                mask = np.asanyarray(masknii.dataobj) > 1e-4
            else:
                mask = np.ones_like(fmapnii.dataobj, dtype=bool)
            colmat = sparse_hstack(
                [grid_bspline_weights(fmapnii, grid) for grid in bs_grids]
            ).tocsr()

        regressors = colmat[mask.reshape(-1), :]
        interp_data = np.zeros_like(data)
        # Interpolate the field from the coefficients just calculated
        interp_data[mask] = regressors @ model.coef_

        # Store interpolated field
        hdr = fmapnii.header.copy()
        hdr.set_data_dtype("float32")
        outnii = fmapnii.__class__(interp_data, fmapnii.affine, hdr)
        outnii.header["cal_max"] = np.abs(outnii.dataobj).max()
        outnii.header["cal_min"] = -outnii.header["cal_max"]
        outnii.to_filename(out_name)
        self._results["out_field"] = out_name

        # Write out fitting-error map
        self._results["out_error"] = out_name.replace("_field.", "_error.")
        errornii = fmapnii.__class__(
            (data - interp_data) * mask, fmapnii.affine, fmapnii.header
        )
        errornii.header["cal_min"] = 0
        errornii.header["cal_max"] = np.max(errornii.dataobj)
        errornii.to_filename(self._results["out_error"])

        if not self.inputs.extrapolate:
            return runtime

        if np.all(mask):
            self._results["out_extrapolated"] = self._results["out_field"]
            return runtime

        extrapolators = colmat[~mask.reshape(-1), :]
        interp_data[~mask] = extrapolators @ model.coef_  # Extrapolation
        self._results["out_extrapolated"] = out_name.replace("_field.", "_extra.")
        fmapnii.__class__(interp_data, fmapnii.affine, hdr).to_filename(
            self._results["out_extrapolated"]
        )
        return runtime


class _ApplyCoeffsFieldInputSpec(BaseInterfaceInputSpec):
    in_data = File(exist=True, mandatory=True, desc="input EPI data to be corrected")
    in_coeff = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc="input coefficients as calculated in the estimation stage",
    )
    fmap2data_xfm = InputMultiObject(
        File(exists=True),
        desc="the transform by which the target EPI can be resampled on the fieldmap's grid.",
        xor=["data2fmap_xfm"],
    )
    data2fmap_xfm = InputMultiObject(
        File(exists=True),
        desc="the transform by which the fieldmap can be resampled on the target EPI's grid.",
        xor=["fmap2data_xfm"],
    )
    in_xfms = traits.List(
        traits.List(traits.List(traits.Float)),
        desc="list of head-motion correction matrices",
    )
    ro_time = InputMultiObject(
        traits.Float(), mandatory=True, desc="EPI readout time (s)."
    )
    pe_dir = InputMultiObject(
        traits.Enum("i", "i-", "j", "j-", "k", "k-"),
        mandatory=True,
        desc="the phase-encoding direction corresponding to in_data",
    )
    jacobian = traits.Bool(
        False,
        usedefault=True,
        desc="apply Jacobian determinant correction after unwarping",
    )
    num_threads = traits.Int(nohash=True, desc="number of threads")
    approx = traits.Bool(
        True,
        usedefault=True,
        desc=(
            "reconstruct the fieldmap on its original grid and then interpolate on the "
            "rotated grid, rather than reconstructing directly on the rotated grid."
        ),
    )


class _ApplyCoeffsFieldOutputSpec(TraitedSpec):
    out_corrected = OutputMultiObject(File(exists=True))
    out_field = OutputMultiObject(File(exists=True))


class ApplyCoeffsField(SimpleInterface):
    """
    Undistort a target, distorted image with a fieldmap, formalized by its B-Spline coefficients.

    Preconditions:

    * We have a "target EPI" - a BOLD or DWI dataset (or even MPRAGE, same principle),
      without having gone through HMC or SDC.
    * We have also the list of HMC matrices that *accounts for* head-motion, so after resampling
      the dataset through this list of transforms *the head does not move anymore*.
    * We have estimated the fieldmap's coefficients
    * We have the "fieldmap-to-data" affine transform that aligns the target dataset (e.g., EPI)
      and the fieldmap's "magnitude" (phasediff et al.) or "reference" (pepolar, syn).

    The algorithm is implemented in the :obj:`~sdcflows.transform.B0FieldTransform` object.
    First, we will call :obj:`~sdcflows.transform.B0FieldTransform.fit`, which
    results in:

    1. The reference grid of the target dataset is projected onto the fieldmap space
    2. The B-Spline coefficients are applied to reconstruct the field on the grid resulting
       above.

    After which, we can then call :obj:`~sdcflows.transform.B0FieldTransform.apply`.
    This second step will:

    3. Find the location of every voxel on each timepoint (meaning, after the head moved)
       and progress (or recede) along the phase-encoding axis to find the actual (voxel)
       coordinates of each voxel.
       With those coordinates known, interpolation is trivial.
    4. Generate a spatial image with the new data.

    Example
    -------

    >>> from sdcflows.interfaces.bspline import ApplyCoeffsField
    >>> unwarp = ApplyCoeffsField(pe_dir='j', ro_time=0.03125)
    >>> unwarp.inputs.in_data = str(data_dir / 'epi.nii.gz')
    >>> unwarp.inputs.in_coeff = str(data_dir / 'topup-coeff.nii.gz')
    >>> unwarp.inputs.data2fmap_xfm = str(data_dir / 'epi2fmap_xfm.txt')
    >>> result = unwarp.run()  # doctest: +SKIP

    Inverse transforms may be used instead:

    >>> unwarp = ApplyCoeffsField(pe_dir='j', ro_time=0.03125)
    >>> unwarp.inputs.in_data = str(data_dir / 'epi.nii.gz')
    >>> unwarp.inputs.in_coeff = str(data_dir / 'topup-coeff.nii.gz')
    >>> unwarp.inputs.fmap2data_xfm = str(data_dir / 'fmap2epi_xfm.txt')
    >>> result = unwarp.run()  # doctest: +SKIP

    """

    input_spec = _ApplyCoeffsFieldInputSpec
    output_spec = _ApplyCoeffsFieldOutputSpec

    def _run_interface(self, runtime):
        from sdcflows.transform import B0FieldTransform

        data2fmap_xfm = None

        if isdefined(self.inputs.data2fmap_xfm):
            data2fmap_xfm = Affine.from_filename(
                self.inputs.data2fmap_xfm if not isinstance(self.inputs.data2fmap_xfm, list)
                else self.inputs.data2fmap_xfm[0],
                fmt="itk"
            ).matrix
        elif isdefined(self.inputs.fmap2data_xfm):
            # Same, but inverting direction
            data2fmap_xfm = (~Affine.from_filename(
                self.inputs.fmap2data_xfm if not isinstance(self.inputs.fmap2data_xfm, list)
                else self.inputs.fmap2data_xfm[0],
                fmt="itk"
            )).matrix

        # Pre-cached interpolator object
        unwarp = B0FieldTransform(
            coeffs=[nb.load(cname) for cname in self.inputs.in_coeff]
        )

        # We can now write out the fieldmap
        self._results["out_field"] = fname_presuffix(
            self.inputs.in_data,
            suffix="_field",
            newpath=runtime.cwd,
        )

        self._results["out_corrected"] = fname_presuffix(
            self.inputs.in_data,
            suffix="_sdc",
            newpath=runtime.cwd,
        )

        unwarp.apply(
            self.inputs.in_data,
            self.inputs.pe_dir,
            self.inputs.ro_time,
            jacobian=self.inputs.jacobian,
            xfms=self.inputs.in_xfms or None,
            xfm_data2fmap=data2fmap_xfm,
            approx=self.inputs.approx,
            num_threads=self.inputs.num_threads or None,
        ).to_filename(self._results["out_corrected"])
        unwarp.mapped.to_filename(self._results["out_field"])
        return runtime


class _TransformCoefficientsInputSpec(BaseInterfaceInputSpec):
    in_coeff = InputMultiObject(
        File(exist=True), mandatory=True, desc="input coefficients file(s)"
    )
    fmap_ref = File(exists=True, mandatory=True, desc="the fieldmap reference")
    transform = File(exists=True, mandatory=True, desc="rigid-body transform file")
    fmap_target = File(
        exists=True, desc="the distorted EPI target (feed to set debug mode on)"
    )


class _TransformCoefficientsOutputSpec(TraitedSpec):
    out_coeff = OutputMultiObject(File(exists=True), desc="moved coefficients")


class TransformCoefficients(SimpleInterface):
    """Project coefficients files to another space through a rigid-body transform."""

    input_spec = _TransformCoefficientsInputSpec
    output_spec = _TransformCoefficientsOutputSpec

    def _run_interface(self, runtime):
        from sdcflows.transform import _move_coeff

        self._results["out_coeff"] = []

        for level in self.inputs.in_coeff:
            movednii = _move_coeff(
                level,
                self.inputs.fmap_ref,
                self.inputs.transform,
                fmap_target=(
                    self.inputs.fmap_target or None
                ),
            )
            out_file = fname_presuffix(
                level, suffix="_space-target", newpath=runtime.cwd
            )
            movednii.to_filename(out_file)
            self._results["out_coeff"].append(out_file)
        return runtime


class _TOPUPCoeffReorientInputSpec(BaseInterfaceInputSpec):
    in_coeff = InputMultiObject(
        File(exist=True), mandatory=True, desc="input coefficients file(s) from TOPUP"
    )
    fmap_ref = File(exists=True, mandatory=True, desc="the fieldmap reference")
    pe_dir = traits.Enum(
        *["".join(p) for p in product("ijkxyz", ("", "-"))],
        mandatory=True,
        desc="phase encoding direction",
    )


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
    All the director cosines of the output coefficients will be positive.
    In other words, the output orientation is either RAS, ARS, ASR, SAR, or SRA.

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
                    self.inputs.pe_dir,
                    out_file=fname_presuffix(
                        in_coeff, suffix="_fixed", newpath=runtime.cwd
                    ),
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


def _fix_topup_fieldcoeff(in_coeff, fmap_ref, pe_dir, out_file=None):
    """Read in a coefficients file generated by TOPUP and fix x-form headers."""
    from pathlib import Path
    import numpy as np
    import nibabel as nb

    if out_file is None:
        out_file = Path("coefficients.nii.gz").absolute()

    refnii = nb.load(fmap_ref)
    coeffnii = nb.load(in_coeff)

    # Coefficients generated by TOPUP are in LAS space, and we will convert to RAS.
    # Reorient the reference image and phase-encoding direction to RAS
    ref_ornt = nb.io_orientation(refnii.affine)
    refnii_ras = refnii.as_reoriented(ref_ornt)
    coeff_pe_dir = reorient_pedir(pe_dir, ref_ornt)

    # Coefficients - flip LR and overwrite coeffnii variable
    # Internal data orientation of FSL is LAS, so coefficients will be LR flipped,
    # and because the affine does not encode orientation (factors instead), this flip
    # always is implicit.
    # If the PE direction is x/i, the flip in the axis direction causes that the
    # fieldmap estimation must also be inverted in direction (multiply by -1.0)
    reverse_pe = -1.0 if coeff_pe_dir[0] == "i" else 1.0
    coeffnii = coeffnii.__class__(
        reverse_pe * np.flip(np.asanyarray(coeffnii.dataobj), axis=0),
        coeffnii.affine,
        coeffnii.header,
    )

    # Get matrix of B-Spline control knots
    coeff_shape = np.array(coeffnii.shape[:3])
    # Get factors (w.r.t. reference's pixel sizes) to calculate separation btw control points
    factors = np.array(coeffnii.header.get_zooms()[:3])
    # Shape checking
    ref_shape = np.array(refnii_ras.shape[:3])
    exp_shape = ref_shape // factors + 3 * (factors > 1)
    if not np.all(coeff_shape == exp_shape):
        raise ValueError(
            f"Shape of coefficients file {coeff_shape} does not meet the "
            f"expected shape {exp_shape} (toupup factors are {factors})."
        )

    # Contextualize the control points in space with a proper NIfTI affine
    newaff = np.eye(4)
    newaff[:3, :3] = refnii_ras.affine[:3, :3] * factors
    c_ref = nb.affines.apply_affine(refnii_ras.affine, 0.5 * (ref_shape - 1))
    c_coeff = nb.affines.apply_affine(newaff, 0.5 * (coeff_shape - 1))
    newaff[:3, 3] = c_ref - c_coeff

    # Edit coefficient's header
    header = coeffnii.header.copy()
    header.set_qform(newaff, code=1)
    header.set_sform(newaff, code=1)
    header["cal_max"] = max(
        (
            abs(np.asanyarray(coeffnii.dataobj).min()),
            np.asanyarray(coeffnii.dataobj).max(),
        )
    )
    header["cal_min"] = -header["cal_max"]
    header.set_intent("estimate", tuple(), name="B-Spline coefficients")

    # Write out fixed (generalized) coefficients
    coeffnii.__class__(coeffnii.dataobj, newaff, header).to_filename(out_file)
    return out_file


def _split_itk_file(in_file):
    from pathlib import Path

    lines = Path(in_file).read_text().splitlines()
    header = lines.pop(0)

    def _chunks(inlist, chunksize):
        for i in range(0, len(inlist), chunksize):
            yield "\n".join([header] + inlist[i : i + chunksize])

    for i, xfm in enumerate(_chunks(lines, 4)):
        p = Path(f"{i:05}")
        p.write_text(xfm)
        yield str(p)


def _b0_resampler(in_file, coeffs, pe, ro, hmc_xfm=None, unwarp=None, newpath=None):
    """Outsource the resampler into a separate callable function to allow parallelization."""
    from functools import partial

    # Prepare output names
    filename = partial(fname_presuffix, newpath=newpath)
    retval = [filename(in_file, suffix=s) for s in ("_unwarped", "_xfm", "_field")]

    if unwarp is None:
        from sdcflows.transform import B0FieldTransform

        # Create a new unwarp object
        unwarp = B0FieldTransform(
            coeffs=[nb.load(cname) for cname in coeffs],
        )

    if hmc_xfm is not None:
        from nitransforms.linear import Affine
        from nitransforms.io.itk import ITKLinearTransform as XFMLoader

        unwarp.xfm = Affine(XFMLoader.from_filename(hmc_xfm).to_ras())

    # Load distorted image
    distorted_img = nb.load(in_file)

    # Reorient to RAS to ensure consistency with coefficients
    # The b-spline weight matrix is sensitive to orientation
    ornt = nb.io_orientation(distorted_img.affine)
    distorted_ras = distorted_img.as_reoriented(ornt)
    pe_ras = reorient_pedir(pe, ornt)

    if unwarp.fit(distorted_ras):
        unwarp.mapped.to_filename(retval[2])
    else:
        retval[2] = None

    # Unwarp
    unwarped_img = unwarp.apply(distorted_ras, ro_time=ro, pe_dir=pe_ras)

    # Write out to disk
    unwarped_img.to_filename(retval[0])

    # Store the corresponding spatial transformation
    unwarp.to_displacements(ro_time=ro, pe_dir=pe_ras).to_filename(retval[1])

    return retval
