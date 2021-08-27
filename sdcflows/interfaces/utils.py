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
"""Utilities."""
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    SimpleInterface,
    InputMultiObject,
    OutputMultiObject,
    isdefined,
)
from nipype.interfaces.ants.segmentation import (
    DenoiseImage as _DenoiseImageBase,
    DenoiseImageInputSpec as _DenoiseImageInputSpecBase,
)
from nipype.interfaces.mixins import CopyHeaderInterface as _CopyHeaderInterface

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


class _UniformGridInputSpec(BaseInterfaceInputSpec):
    in_data = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc="list of input data",
    )
    reference = traits.Int(0, usedefault=True, desc="reference index")


class _UniformGridOutputSpec(TraitedSpec):
    out_data = OutputMultiObject(File(exists=True))
    reference = File(exists=True)


class UniformGrid(SimpleInterface):
    """Ensure all images in input have the same spatial parameters."""

    input_spec = _UniformGridInputSpec
    output_spec = _UniformGridOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb
        import numpy as np
        from nitransforms.linear import Affine
        from nipype.utils.filemanip import fname_presuffix

        retval = [None] * len(self.inputs.in_data)
        self._results["reference"] = self.inputs.in_data[self.inputs.reference]
        retval[self.inputs.reference] = self._results["reference"]

        refnii = nb.load(self._results["reference"])
        refshape = refnii.shape[:3]
        refaff = refnii.affine

        resampler = Affine(reference=refnii)
        for i, fname in enumerate(self.inputs.in_data):
            if retval[i] is not None:
                continue

            nii = nb.load(fname)
            retval[i] = fname_presuffix(
                fname, suffix=f"_regrid{i:03d}", newpath=runtime.cwd
            )

            if np.allclose(nii.shape[:3], refshape) and np.allclose(nii.affine, refaff):
                if np.all(nii.affine == refaff):
                    retval[i] = fname
                else:
                    # Set reference's affine if difference is small
                    nii.__class__(nii.dataobj, refaff, nii.header).to_filename(
                        retval[i]
                    )
                continue

            resampler.apply(nii).to_filename(retval[i])

        self._results["out_data"] = retval

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
    in_file = File(
        exists=True, mandatory=True, desc="the input dataset potentially oblique"
    )
    in_mask = File(
        exists=True,
        desc="a binary mask corresponding to the input dataset",
    )


class _DeobliqueOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the input dataset, after correcting obliquity")
    out_mask = File(exists=True, desc="the input mask, after correcting obliquity")


class Deoblique(SimpleInterface):
    """Make a dataset plumb."""

    input_spec = _DeobliqueInputSpec
    output_spec = _DeobliqueOutputSpec

    def _run_interface(self, runtime):
        self._results["out_file"] = _deoblique(
            self.inputs.in_file,
            newpath=runtime.cwd,
        )

        if isdefined(self.inputs.in_mask):
            self._results["out_mask"] = _deoblique(
                self.inputs.in_mask,
                in_affine=self._results["out_file"],
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
    in_mask = File(
        exists=True,
        desc="a binary mask corresponding to the input dataset",
    )


class _ReobliqueOutputSpec(TraitedSpec):
    out_epi = File(exists=True, desc="the reoblique'd EPI image")
    out_field = File(exists=True, desc="the reoblique'd EPI image")
    out_mask = File(exists=True, desc="the input mask, after correcting obliquity")


class Reoblique(SimpleInterface):
    """Make a dataset plumb."""

    input_spec = _ReobliqueInputSpec
    output_spec = _ReobliqueOutputSpec

    def _run_interface(self, runtime):
        in_mask = self.inputs.in_mask if isdefined(self.inputs.in_mask) else None
        (
            self._results["out_epi"],
            self._results["out_field"],
            self._results["out_mask"],
        ) = _reoblique(
            self.inputs.in_epi,
            self.inputs.in_plumb,
            self.inputs.in_field,
            in_mask=in_mask,
            newpath=runtime.cwd,
        )
        if not in_mask:
            self._results.pop("out_mask")

        return runtime


class _DenoiseImageInputSpec(_DenoiseImageInputSpecBase):
    copy_header = traits.Bool(
        True,
        usedefault=True,
        desc="copy headers of the original image into the output (corrected) file",
    )


class DenoiseImage(_DenoiseImageBase, _CopyHeaderInterface):
    """Add copy_header capability to DenoiseImage from nipype."""

    input_spec = _DenoiseImageInputSpec
    _copy_header_map = {"output_image": "input_image"}


class _PadSlicesInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="3D or 4D NIfTI image")
    axis = traits.Int(
        2,
        usedefault=True,
        desc="The axis through which slices are stacked in the input data"
    )


class _PadSlicesOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="The output file with even number of slices")
    padded = traits.Bool(desc="Indicator if the input image was padded")


class PadSlices(SimpleInterface):
    """
    Check an image for uneven slices, and add an empty slice if necessary

    This intends to avoid TOPUP's segfault without changing the standard configuration
    """
    input_spec = _PadSlicesInputSpec
    output_spec = _PadSlicesOutputSpec

    def _run_interface(self, runtime):
        self._results["out_file"], self._results["padded"] = _pad_num_slices(
            self.inputs.in_file, self.inputs.axis, runtime.cwd,
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


def _deoblique(in_file, in_affine=None, newpath=None):
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    nii = nb.load(in_file)
    if np.all(nb.affines.obliquity(nii.affine)) < OBLIQUE_THRESHOLD_DEG:
        return in_file

    if in_affine is None:
        orientation = nb.aff2axcodes(nii.affine)
        directions = (
            np.array(
                [int(l1 == l2) for l1, l2 in zip(orientation, "RAS")], dtype="float32"
            )
            * 2
            - 1
        )
        newaff = np.eye(4)
        newaff[:3, :3] = np.diag(directions * np.array(nii.header.get_zooms()[:3]))
        newaff[:3, 3] -= (np.array(nii.shape[:3]) - 1) * 0.5
    else:
        newaff = nb.load(in_affine).affine.copy()

    hdr = nii.header.copy()
    hdr.set_qform(newaff, code=1)
    hdr.set_sform(newaff, code=1)
    newnii = nii.__class__(nii.dataobj, newaff, hdr)
    out_file = fname_presuffix(in_file, suffix="_plumb", newpath=newpath)
    newnii.to_filename(out_file)
    return out_file


def _reoblique(in_epi, in_plumb, in_field, in_mask=None, newpath=None):
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    epinii = nb.load(in_epi)
    if np.all(nb.affines.obliquity(epinii.affine)) < OBLIQUE_THRESHOLD_DEG:
        return in_plumb, in_field, in_mask

    out_files = [
        fname_presuffix(f, suffix="_oriented", newpath=newpath)
        for f in (in_plumb, in_field)
    ] + [None]
    plumbnii = nb.load(in_plumb)
    plumbnii.__class__(plumbnii.dataobj, epinii.affine, epinii.header).to_filename(
        out_files[0]
    )

    fmapnii = nb.load(in_field)
    hdr = fmapnii.header.copy()
    hdr.set_qform(*epinii.header.get_qform(coded=True))
    hdr.set_sform(*epinii.header.get_sform(coded=True))
    fmapnii.__class__(fmapnii.dataobj, epinii.affine, hdr).to_filename(out_files[1])

    if in_mask:
        out_files[2] = fname_presuffix(in_mask, suffix="_oriented", newpath=newpath)
        masknii = nb.load(in_mask)
        hdr = masknii.header.copy()
        hdr.set_qform(*epinii.header.get_qform(coded=True))
        hdr.set_sform(*epinii.header.get_sform(coded=True))
        masknii.__class__(masknii.dataobj, epinii.affine, hdr).to_filename(out_files[2])

    return out_files


def _pad_num_slices(in_file, ax=2, newpath=None):
    """
    Ensure the image has even number of slices to avert TOPUP's segfault.

    Check if image has an even number of slices.
    If it does, return the image unaltered.
    Otherwise, return the image with an empty slice added.

    Parameters
    ----------
    img : :obj:`str` or :py:class:`~nibabel.spatialimages.SpatialImage`
        3D or 4D NIfTI image
    ax : :obj:`int`
        The axis through which slices are stacked in the input data.

    Returns
    -------
    file : :obj:`str`
        The output file with even number of slices
    padded : :obj:`bool`
        Indicator if the input image was padded.

    """
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix
    import numpy as np

    img = nb.load(in_file)
    if img.shape[ax] % 2 == 0:
        return in_file, False

    pwidth = [(0,0)] * len(img.shape)
    pwidth[ax] = (0, 1)
    padded = np.pad(img.dataobj, pwidth)
    hdr = img.header
    hdr.set_data_shape(padded.shape)
    out_file = fname_presuffix(in_file, suffix="_padded", newpath=newpath)
    img.__class__(padded, img.affine, header=hdr).to_filename(out_file)
    return out_file, True
