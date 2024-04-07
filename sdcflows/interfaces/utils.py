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
import os
from itertools import product

import nibabel as nb
from nilearn import image
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    DynamicTraitedSpec,
    TraitedSpec,
    File,
    traits,
    SimpleInterface,
    InputMultiObject,
    OutputMultiObject,
    Undefined,
    isdefined,
)
from nipype.interfaces.ants.segmentation import (
    DenoiseImage as _DenoiseImageBase,
    DenoiseImageInputSpec as _DenoiseImageInputSpecBase,
)
from nipype.interfaces.io import add_traits
from nipype.interfaces.mixins import CopyHeaderInterface as _CopyHeaderInterface

from sdcflows.utils.tools import reorient_pedir

OBLIQUE_THRESHOLD_DEG = 0.5
LOGGER = logging.getLogger("nipype.interface")


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
        self._results["out_data"], self._results["out_meta"] = zip(*self._results["out_list"])
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
            retval[i] = fname_presuffix(fname, suffix=f"_regrid{i:03d}", newpath=runtime.cwd)

            if np.allclose(nii.shape[:3], refshape) and np.allclose(nii.affine, refaff):
                if np.all(nii.affine == refaff):
                    retval[i] = fname
                else:
                    # Set reference's affine if difference is small
                    nii.__class__(nii.dataobj, refaff, nii.header).to_filename(retval[i])
                continue

            # Hack around nitransforms' unsafe cast by dropping get_data_dtype that conflicts
            # with effective dtype
            # NT23_0_1: Isssue in nitransforms.base.TransformBase.apply
            regridded_img = resampler.apply(nii.__class__(np.asanyarray(nii.dataobj), nii.affine))
            # Restore the original on-disk data type
            nii.__class__(regridded_img.dataobj, refaff, nii.header).to_filename(retval[i])

        self._results["out_data"] = retval

        return runtime


class _ReorientImageAndMetadataInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc="Input 3- or 4D image")
    target_orientation = traits.Str(desc="Axis codes of coordinate system to reorient to")
    pe_dir = InputMultiObject(
        traits.Enum(
            *["".join(p) for p in product("ijkxyz", ("", "-"))],
            mandatory=True,
            desc="Phase encoding direction",
        )
    )


class _ReorientImageAndMetadataOutputSpec(TraitedSpec):
    out_file = File(desc="Reoriented image")
    pe_dir = OutputMultiObject(
        traits.Enum(
            *["".join(p) for p in product("ijkxyz", ("", "-"))],
            desc="Phase encoding direction in reoriented image",
        )
    )


class ReorientImageAndMetadata(SimpleInterface):
    input_spec = _ReorientImageAndMetadataInputSpec
    output_spec = _ReorientImageAndMetadataOutputSpec

    def _run_interface(self, runtime):
        import numpy as np
        import nibabel as nb
        from nipype.utils.filemanip import fname_presuffix

        target = self.inputs.target_orientation.upper()
        if not all(code in "RASLPI" for code in target):
            raise ValueError(f"Invalid orientation code {self.inputs.target_orientation}")

        img = nb.load(self.inputs.in_file)
        img2target = nb.orientations.ornt_transform(
            nb.io_orientation(img.affine),
            nb.orientations.axcodes2ornt(self.inputs.target_orientation),
        ).astype(int)

        # Identity transform
        if np.array_equal(img2target, [[0, 1], [1, 1], [2, 1]]):
            self._results = dict(
                out_file=self.inputs.in_file,
                pe_dir=self.inputs.pe_dir,
            )
            return runtime

        reoriented = img.as_reoriented(img2target)

        pe_dirs = [reorient_pedir(pe_dir, img2target) for pe_dir in self.inputs.pe_dir]

        self._results = dict(
            out_file=fname_presuffix(self.inputs.in_file, suffix=target, newpath=runtime.cwd),
            pe_dir=pe_dirs,
        )

        reoriented.to_filename(self._results["out_file"])

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
        self._results["out_file"] = _qwarp2ants(self.inputs.in_file, newpath=runtime.cwd)
        return runtime


class _DeobliqueInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="the input dataset potentially oblique")
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
    in_epi = File(exists=True, mandatory=True, desc="the original, potentially oblique EPI image")
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
        2, usedefault=True, desc="The axis through which slices are stacked in the input data"
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
            self.inputs.in_file,
            self.inputs.axis,
            runtime.cwd,
        )
        return runtime


class _PositiveDirectionCosinesInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input image")


class _PositiveDirectionCosinesOutputSpec(TraitedSpec):
    out_file = File()
    in_orientation = traits.Str()


class PositiveDirectionCosines(SimpleInterface):
    """Reorient axes polarity to have all positive direction cosines."""

    input_spec = _PositiveDirectionCosinesInputSpec
    output_spec = _PositiveDirectionCosinesOutputSpec

    def _run_interface(self, runtime):
        (self._results["out_file"], self._results["in_orientation"]) = _ensure_positive_cosines(
            self.inputs.in_file,
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


def _deoblique(in_file, in_affine=None, newpath=None):
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    nii = nb.load(in_file)
    if np.all(nb.affines.obliquity(nii.affine) < OBLIQUE_THRESHOLD_DEG):
        return in_file

    if in_affine is None:
        orientation = nb.aff2axcodes(nii.affine)
        directions = (
            np.array([int(l1 == l2) for l1, l2 in zip(orientation, "RAS")], dtype="float32") * 2
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
    if np.all(nb.affines.obliquity(epinii.affine) < OBLIQUE_THRESHOLD_DEG):
        return in_plumb, in_field, in_mask

    out_files = [
        fname_presuffix(f, suffix="_oriented", newpath=newpath) for f in (in_plumb, in_field)
    ] + [None]
    plumbnii = nb.load(in_plumb)
    plumbnii.__class__(plumbnii.dataobj, epinii.affine, epinii.header).to_filename(out_files[0])

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

    pwidth = [(0, 0)] * len(img.shape)
    pwidth[ax] = (0, 1)
    padded = np.pad(img.dataobj, pwidth)
    hdr = img.header
    hdr.set_data_shape(padded.shape)
    out_file = fname_presuffix(in_file, suffix="_padded", newpath=newpath)
    img.__class__(padded, img.affine, header=hdr).to_filename(out_file)
    return out_file, True


def _ensure_positive_cosines(in_file: str, newpath: str = None):
    """
    Reorient axes polarity to have all positive direction cosines.

    In other words, this interface will reorient the image polarities to be all
    *positive*, respecting the axes ordering.
    For instance, *LAS* -> *RAS* or *ALS* -> *ARS*.

    """
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix
    from sdcflows.utils.tools import ensure_positive_cosines

    out_file = fname_presuffix(in_file, suffix="_flipfree", newpath=newpath)
    reoriented, axcodes = ensure_positive_cosines(nb.load(in_file))
    reoriented.to_filename(out_file)
    return out_file, "".join(axcodes)


class _TransposeImagesInputSpec(DynamicTraitedSpec):
    pass


class TransposeImages(SimpleInterface):
    """Convert input filenames to BIDS URIs, based on links in the dataset.

    This interface can combine multiple lists of inputs.
    """

    input_spec = _TransposeImagesInputSpec
    output_spec = DynamicTraitedSpec

    def __init__(self, numinputs=0, numoutputs=0, **inputs):
        super().__init__(**inputs)
        self._numinputs = numinputs
        self._numoutputs = numoutputs
        if numinputs >= 1:
            input_names = [f"in{i + 1}" for i in range(numinputs)]
        else:
            input_names = []
        add_traits(self.inputs, input_names)

    def _outputs(self):
        # Mostly copied from nipype.interfaces.dcmstack.LookupMeta.
        outputs = super()._outputs()

        undefined_traits = {}
        if self._numoutputs >= 1:
            output_names = [f"out{i + 1}" for i in range(self._numoutputs)]
        else:
            output_names = []

        for output_name in output_names:
            outputs.add_trait(output_name, traits.Any)
            undefined_traits[output_name] = Undefined

        outputs.trait_set(trait_change_notify=False, **undefined_traits)

        return outputs

    def _run_interface(self, runtime):
        inputs = [getattr(self.inputs, f"in{i + 1}") for i in range(self._numinputs)]

        for i_vol in range(self._numoutputs):
            vol_imgs = []
            for file_ in inputs:
                vol_img = image.index_img(file_, i_vol)
                vol_imgs.append(vol_img)
            concat_vol_img = image.concat_imgs(vol_imgs)

            vol_file = os.path.abspath(f"vol{i_vol + 1}.nii.gz")
            concat_vol_img.to_filename(vol_file)
            self._results[f"out{i_vol + 1}"] = vol_file

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs.update(self._results)
        return outputs


class _SVDFilterInputSpec(TraitedSpec):
    fieldmap = File(exists=True, mandatory=True, desc="Fieldmap image")
    mask = File(exists=True, desc="Mask image")
    border_filter = traits.List(
        traits.Float,
        desc="Border filter parameters",
        minlen=2,
        maxlen=2,
    )
    svd_filter = traits.Int(desc="SVD filter parameter")


class _SVDFilterOutputSpec(TraitedSpec):
    fieldmap = File(exists=True, desc="Filtered fieldmap image")


class SVDFilter(SimpleInterface):

    input_spec = _SVDFilterInputSpec
    output_spec = _SVDFilterOutputSpec

    def _run_interface(self, runtime):
        import numpy as np
        from nipype.utils.filemanip import fname_presuffix
        from scipy.ndimage import gaussian_filter

        border_filt = self.inputs.border_filter
        svd_filt = self.inputs.svd_filter

        fieldmap_img = nb.load(self.inputs.fieldmap)
        mask_img = nb.load(self.inputs.mask)

        n_frames = fieldmap_img.shape[3]
        voxel_size = fieldmap_img.header.get_zooms()[0]  # assuming isotropic voxels I guess

        mask_data = mask_img.get_fdata().astype(np.uint8)
        fieldmap_data = fieldmap_img.get_fdata()

        if mask_data.max() == 2 and n_frames >= np.max(border_filt):
            LOGGER.info("Performing spatial/temporal filtering of border voxels...")
            smoothed_field_maps = np.zeros(fieldmap_data.shape, dtype=np.float32)
            # smooth by 4 mm kernel
            sigma = (4 / voxel_size) / 2.355
            for i_vol in range(n_frames):
                smoothed_field_maps[..., i_vol] = gaussian_filter(
                    fieldmap_data[..., i_vol],
                    sigma=sigma,
                )

            # compute the union of all the masks
            union_mask = np.sum(mask_data, axis=-1) > 0

            # do temporal filtering of border voxels with SVD
            U, S, VT = np.linalg.svd(smoothed_field_maps[union_mask], full_matrices=False)

            # first pass of SVD filtering
            recon = np.dot(U[:, : border_filt[0]] * S[: border_filt[0]], VT[: border_filt[0], :])
            recon_img = np.zeros(fieldmap_data.shape, dtype=np.float32)
            recon_img[union_mask] = recon

            # set the border voxels in the field map to the recon values
            for i_vol in range(n_frames):
                fieldmap_data[mask_data[..., i_vol] == 1, i_vol] = recon_img[
                    mask_data[..., i_vol] == 1,
                    i_vol,
                ]

            # do second SVD filtering pass
            U, S, VT = np.linalg.svd(fieldmap_data[union_mask], full_matrices=False)

            # second pass of SVD filtering
            recon = np.dot(U[:, : border_filt[1]] * S[: border_filt[1]], VT[: border_filt[1], :])
            recon_img = np.zeros(fieldmap_data.shape, dtype=np.float32)
            recon_img[union_mask] = recon

            # set the border voxels in the field map to the recon values
            for i_vol in range(n_frames):
                fieldmap_data[mask_data[..., i_vol] == 1, i_vol] = recon_img[
                    mask_data[..., i_vol] == 1,
                    i_vol,
                ]

        # use svd filter to denoise the field maps
        if n_frames >= svd_filt:
            LOGGER.info("Denoising field maps with SVD...")
            LOGGER.info(f"Keeping {svd_filt} components...")

            # compute the union of all the masks
            union_mask = np.sum(mask_data, axis=-1) > 0

            # compute SVD
            U, S, VT = np.linalg.svd(fieldmap_data[union_mask], full_matrices=False)

            # only keep the first n_components components
            recon = np.dot(U[:, :svd_filt] * S[:svd_filt], VT[:svd_filt, :])
            recon_img = np.zeros(fieldmap_data.shape, dtype=np.float32)
            recon_img[union_mask] = recon

            # set the voxel values in the mask to the recon values
            for i_vol in range(n_frames):
                fieldmap_data[mask_data[..., i_vol] > 0, i_vol] = recon_img[
                    mask_data[..., i_vol] > 0,
                    i_vol,
                ]

        out_file = fname_presuffix(self.inputs.fieldmap, suffix="_filtered", newpath=runtime.cwd)
        nb.Nifti1Image(
            fieldmap_data,
            fieldmap_img.affine,
            fieldmap_img.header,
        ).to_filename(out_file)
        self._results["fieldmap"] = out_file

        return runtime


class _EnforceTemporalConsistencyInputSpec(TraitedSpec):
    phase_unwrapped = traits.List(
        File(exists=True),
        mandatory=True,
        desc="Unwrapped phase data",
    )
    echo_times = traits.List(
        traits.Float,
        mandatory=True,
        desc="Echo times",
    )
    magnitude = traits.List(
        File(exists=True),
        mandatory=True,
        desc="Magnitude images",
    )
    mask = File(exists=True, mandatory=True, desc="Brain mask")
    threshold = traits.Float(
        0.98,
        usedefault=True,
        desc="Threshold for correlation similarity",
    )


class _EnforceTemporalConsistencyOutputSpec(TraitedSpec):
    phase_unwrapped = traits.List(
        File(exists=True),
        desc="Unwrapped phase data after temporal consistency is enforced.",
    )


class EnforceTemporalConsistency(SimpleInterface):
    """Ensure phase unwrapping solutions are temporally consistent.

    This uses correlation as a similarity metric between frames to enforce temporal consistency.

    This is derived from ``warpkit.unwrap.check_temporal_consistency_corr``.

    XXX: Small change from warpkit:
    I used ``weights_mat[:j_echo, i_vol, :]`` instead of ``weights_mat[:j_echo]``
    Otherwise, ``coefficients`` is voxels x time instead of just voxels
    Not sure what the source of the problem is yet.
    """

    input_spec = _EnforceTemporalConsistencyInputSpec
    output_spec = _EnforceTemporalConsistencyOutputSpec

    def _run_interface(self, runtime):
        import numpy as np
        from nipype.utils.filemanip import fname_presuffix

        from sdcflows.utils.misc import corr2_coeff, create_brain_mask, weighted_regression

        echo_times = np.array(self.inputs.echo_times)
        n_echoes = echo_times.size
        mag_shortest = self.inputs.magnitude[0]

        # generate brain mask (with 1 voxel erosion)
        brain_mask_file = create_brain_mask(mag_shortest, -1)
        brain_mask = nb.load(brain_mask_file).get_fdata().astype(bool)

        # Concate the unwrapped phase data into a 5D array
        phase_imgs = [nb.load(phase) for phase in self.inputs.phase_unwrapped]
        unwrapped_echo_1 = phase_imgs[0].get_fdata()
        phase_unwrapped = np.stack([img.get_fdata() for img in phase_imgs], axis=3)
        magnitude_imgs = [nb.load(mag) for mag in self.inputs.magnitude]
        n_volumes = phase_imgs[0].shape[3]
        mask_img = nb.load(self.inputs.mask)
        mask_data = mask_img.get_fdata().astype(bool)

        for i_vol in range(n_volumes):
            LOGGER.info(f"Computing temporal consistency check for frame: {i_vol}")

            # get the current frame phase
            current_frame_data = unwrapped_echo_1[brain_mask, i_vol][:, np.newaxis]

            # get the correlation between the current frame and all other frames
            corr = corr2_coeff(current_frame_data, unwrapped_echo_1[brain_mask, :]).ravel()

            # threhold the RD
            tmask = corr > self.inputs.threshold

            # get indices of mask
            indices = np.where(tmask)[0]

            # get mask for frame
            volume_mask = mask_data[..., i_vol] > 0

            # for each frame compute the mean value along the time axis
            # (masked by indices and mask)
            mean_voxels = np.mean(unwrapped_echo_1[volume_mask][:, indices], axis=-1)

            # for this frame figure out the integer multiple that minimizes the value to the
            # mean voxel
            int_map = np.round(
                (mean_voxels - unwrapped_echo_1[volume_mask, i_vol]) / (2 * np.pi)
            ).astype(int)

            # correct the first echo's data using the integer map
            phase_unwrapped[volume_mask, 0, i_vol] += 2 * np.pi * int_map

            # format weight matrix
            weights_mat = np.stack([m.dataobj for m in magnitude_imgs], axis=-1)[volume_mask].T

            # form design matrix
            X = echo_times[:, np.newaxis]

            # fit subsequent echos to the weighted linear regression from the first echo
            for j_echo in range(1, n_echoes):
                # form response matrix
                Y = phase_unwrapped[volume_mask, :j_echo, i_vol].T

                # fit model to data
                # XXX: Small change from warpkit:
                # weights_mat[:j_echo, i_vol, :] instead of weights_mat[:j_echo]
                # Otherwise, coefficients is voxels x time instead of just voxels
                # Not sure what the issue is.
                coefficients, _ = weighted_regression(
                    X[:j_echo],
                    Y,
                    weights_mat[:j_echo, i_vol, :],
                )

                # get the predicted values for this echo
                Y_pred = coefficients * echo_times[j_echo]

                # compute the difference and get the integer multiple map
                int_map = np.round(
                    (Y_pred - phase_unwrapped[volume_mask, j_echo, i_vol]) / (2 * np.pi)
                ).astype(int)

                # correct the data using the integer map
                phase_unwrapped[volume_mask, j_echo, i_vol] += 2 * np.pi * int_map

        out_files = []
        for i_echo in range(n_echoes):
            phase_unwrapped_echo = phase_unwrapped[:, :, :, i_echo, :]
            phase_unwrapped_echo_img = nb.Nifti1Image(
                phase_unwrapped_echo,
                phase_imgs[0].affine,
                phase_imgs[0].header,
            )
            out_file = fname_presuffix(
                self.inputs.phase_unwrapped[i_echo],
                suffix="_temporallyconsistent",
                newpath=runtime.cwd,
            )
            phase_unwrapped_echo_img.to_filename(out_file)
            out_files.append(out_file)

        self._results["phase_unwrapped"] = out_files

        return runtime
