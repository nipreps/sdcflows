# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utilities."""
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    SimpleInterface,
    InputMultiObject,
    OutputMultiObject,
)

OBLIQUE_THRESHOLD_DEG = 0.5


class _FlattenInputSpec(BaseInterfaceInputSpec):
    in_data = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc="list of input data",
    )
    in_meta = InputMultiObject(
        traits.DictStrAny,
        mandatory=True,
        desc="list of metadata",
    )
    max_trs = traits.Int(50, usedefault=True, desc="only pick first TRs")


class _FlattenOutputSpec(TraitedSpec):
    out_list = OutputMultiObject(
        traits.Tuple(
            File(exists=True),
            traits.DictStrAny,
        ),
        desc="list of output files",
    )
    out_data = OutputMultiObject(File(exists=True))
    out_meta = OutputMultiObject(traits.DictStrAny)


class Flatten(SimpleInterface):
    """Flatten a list of 3D and 4D files (and metadata)."""

    input_spec = _FlattenInputSpec
    output_spec = _FlattenOutputSpec

    def _run_interface(self, runtime):
        self._results["out_list"] = _flatten(
            zip(self.inputs.in_data, self.inputs.in_meta),
            max_trs=self.inputs.max_trs,
            out_dir=runtime.cwd,
        )

        # Unzip out_data, out_meta outputs.
        self._results["out_data"], self._results["out_meta"] = zip(
            *self._results["out_list"]
        )
        return runtime


class _ConvertWarpInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="output of 3dQwarp")


class _ConvertWarpOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the warp converted into ANTs")


class ConvertWarp(SimpleInterface):
    """Convert a displacements field from ``3dQwarp`` to ANTS-compatible."""

    input_spec = _ConvertWarpInputSpec
    output_spec = _ConvertWarpOutputSpec

    def _run_interface(self, runtime):
        self._results["out_file"] = _qwarp2ants(
            self.inputs.in_file, newpath=runtime.cwd
        )
        return runtime


class _DeobliqueInputSpec(BaseInterfaceInputSpec):
    in_epi = File(
        exists=True, mandatory=True, desc="the EPI dataset potentially oblique"
    )
    mask_epi = File(
        exists=True,
        mandatory=True,
        desc="a binary mask corresponding to the EPI dataset",
    )
    in_anat = File(
        exists=True,
        mandatory=True,
        desc="the corresponging anatomical dataset potentially oblique",
    )
    mask_anat = File(
        exists=True,
        mandatory=True,
        desc="a binary mask corresponding to the anatomical dataset",
    )


class _DeobliqueOutputSpec(TraitedSpec):
    out_epi = File(exists=True, desc="the plumb, EPI dataset")
    out_anat = File(exists=True, desc="the plumb, anatomical dataset")
    mask_epi = File(
        exists=True,
        mandatory=True,
        desc="a binary mask corresponding to the EPI dataset",
    )
    mask_anat = File(
        exists=True,
        mandatory=True,
        desc="a binary mask corresponding to the anatomical dataset",
    )


class Deoblique(SimpleInterface):
    """Make a dataset plumb."""

    input_spec = _DeobliqueInputSpec
    output_spec = _DeobliqueOutputSpec

    def _run_interface(self, runtime):
        (
            self._results["out_epi"],
            self._results["out_anat"],
            self._results["mask_epi"],
            self._results["mask_anat"],
        ) = _deoblique(
            self.inputs.in_epi,
            self.inputs.in_anat,
            self.inputs.mask_epi,
            self.inputs.mask_anat,
            newpath=runtime.cwd,
        )
        return runtime


class _ReobliqueInputSpec(BaseInterfaceInputSpec):
    in_plumb = File(exists=True, mandatory=True, desc="the plumb EPI image")
    in_field = File(
        exists=True,
        mandatory=True,
        desc="the plumb field map, extracted from the displacements field estimated by SyN",
    )
    in_epi = File(
        exists=True, mandatory=True, desc="the original, potentially oblique EPI image"
    )


class _ReobliqueOutputSpec(TraitedSpec):
    out_epi = File(exists=True, desc="the reoblique'd EPI image")
    out_field = File(exists=True, desc="the reoblique'd EPI image")


class Reoblique(SimpleInterface):
    """Make a dataset plumb."""

    input_spec = _ReobliqueInputSpec
    output_spec = _ReobliqueOutputSpec

    def _run_interface(self, runtime):
        (self._results["out_epi"], self._results["out_field"],) = _reoblique(
            self.inputs.in_epi,
            self.inputs.in_plumb,
            self.inputs.in_field,
            newpath=runtime.cwd,
        )
        return runtime


class _IntensityClipInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True, mandatory=True, desc="file which intensity will be clipped"
    )
    p_min = traits.Float(35.0, usedefault=True, desc="percentile for the lower bound")
    p_max = traits.Float(99.98, usedefault=True, desc="percentile for the upper bound")
    nonnegative = traits.Bool(
        True, usedefault=True, desc="whether input intensities must be positive"
    )
    dtype = traits.Enum(
        "int16", "float32", "uint8", usedefault=True, desc="output datatype"
    )


class _IntensityClipOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="file after clipping")


class IntensityClip(SimpleInterface):
    """Clip the intensity range as prescribed by the percentiles."""

    input_spec = _IntensityClipInputSpec
    output_spec = _IntensityClipOutputSpec

    def _run_interface(self, runtime):
        self._results["out_file"] = _advanced_clip(
            self.inputs.in_file,
            p_min=self.inputs.p_min,
            p_max=self.inputs.p_max,
            nonnegative=self.inputs.nonnegative,
            dtype=self.inputs.dtype,
            newpath=runtime.cwd,
        )
        return runtime


def _flatten(inlist, max_trs=50, out_dir=None):
    """
    Split the input EPIs and generate a flattened list with corresponding metadata.

    Inputs
    ------
    inlist : :obj:`list` of :obj:`tuple`
        List of pairs (filepath, metadata)
    max_trs : :obj:`int`
        Index of frame after which all volumes will be discarded
        from the input images.

    """
    from pathlib import Path
    import nibabel as nb

    out_dir = Path(out_dir) if out_dir is not None else Path()

    output = []
    for i, (path, meta) in enumerate(inlist):
        img = nb.load(path)
        if len(img.shape) == 3:
            output.append((path, meta))
        else:
            splitnii = nb.four_to_three(img.slicer[:, :, :, :max_trs])
            stem = Path(path).name.rpartition(".nii")[0]

            for j, nii in enumerate(splitnii):
                out_name = (out_dir / f"{stem}_idx-{j:03}.nii.gz").absolute()
                nii.to_filename(out_name)
                output.append((str(out_name), meta))

    return output


def _qwarp2ants(in_file, newpath=None):
    """Ensure the data type and intent of a warp is acceptable by ITK-based tools."""
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    nii = nb.load(in_file)
    hdr = nii.header.copy()
    hdr.set_data_dtype("<f4")
    hdr.set_intent("vector", (), "")
    out_file = fname_presuffix(in_file, "_warpfield", newpath=newpath)
    data = np.squeeze(nii.get_fdata(dtype="float32"))[..., np.newaxis, :]
    nb.Nifti1Image(data, nii.affine, hdr).to_filename(out_file)
    return out_file


def _deoblique(in_epi, in_anat, mask_epi, mask_anat, newpath=None):
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    epinii = nb.load(in_epi)
    if np.all(nb.affines.obliquity(epinii.affine)) < OBLIQUE_THRESHOLD_DEG:
        return in_epi, in_anat, mask_epi, mask_anat

    newaff = np.eye(4)
    newaff[:3, :3] = np.diag(np.array(epinii.header.get_zooms()[:3]))
    newaff[:3, 3] -= (np.array(epinii.shape[:3]) - 1) * 0.5

    hdr = epinii.header.copy()
    hdr.set_qform(newaff, code=1)
    hdr.set_sform(newaff, code=1)
    newepi = epinii.__class__(epinii.dataobj, newaff, hdr)
    out_epi = fname_presuffix(in_epi, suffix="_plumb", newpath=newpath)
    newepi.to_filename(out_epi)

    out_files = [out_epi]
    for fname in (in_anat, mask_epi, mask_anat):
        nii = nb.load(fname)
        hdr = nii.header.copy()
        hdr.set_qform(newaff, code=1)
        hdr.set_sform(newaff, code=1)
        newnii = nii.__class__(np.asanyarray(nii.dataobj), newaff, hdr)
        out = fname_presuffix(fname, suffix="_plumb", newpath=newpath)
        newnii.to_filename(out)
        out_files.append(out)

    return tuple(out_files)


def _reoblique(in_epi, in_plumb, in_field, newpath=None):
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    epinii = nb.load(in_epi)
    if np.all(nb.affines.obliquity(epinii.affine)) < OBLIQUE_THRESHOLD_DEG:
        return in_plumb, in_field

    out_files = [
        fname_presuffix(f, suffix="_oriented", newpath=newpath)
        for f in (in_plumb, in_field)
    ]
    plumbnii = nb.load(in_plumb)
    plumbnii.__class__(plumbnii.dataobj, epinii.affine, epinii.header).to_filename(
        out_files[0]
    )

    fmapnii = nb.load(in_field)
    hdr = fmapnii.header.copy()
    hdr.set_qform(*epinii.header.get_qform(coded=True))
    hdr.set_sform(*epinii.header.get_sform(coded=True))
    fmapnii.__class__(fmapnii.dataobj, epinii.affine, hdr).to_filename(out_files[1])
    return out_files


def _advanced_clip(
    in_file, p_min=35, p_max=99.98, nonnegative=True, dtype="int16", newpath=None
):
    from pathlib import Path
    import nibabel as nb
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import ball

    out_file = (Path(newpath or "") / "clipped.nii.gz").absolute()

    # Load data
    img = nb.load(in_file)
    data = img.get_fdata(dtype="float32")

    # Calculate stats on denoised version, to preempt outliers from biasing
    denoised = ndimage.median_filter(data, footprint=ball(3))

    # Clip and cast
    a_min = np.percentile(denoised[denoised > 0], p_min)
    a_max = np.percentile(denoised[denoised > 0], p_max)
    if nonnegative:
        a_min = max(a_min, 0.0)

    data = np.clip(data, a_min=a_min, a_max=a_max)
    data -= data.min()
    data /= data.max()

    if dtype in ("uint8", "int16"):
        data = np.round(255 * data).astype(dtype)

    hdr = img.header.copy()
    hdr.set_data_dtype(dtype)
    img.__class__(data, img.affine, hdr).to_filename(out_file)

    return str(out_file)
