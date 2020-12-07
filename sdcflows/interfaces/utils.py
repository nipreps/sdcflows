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


class _FlattenInputSpec(BaseInterfaceInputSpec):
    in_data = InputMultiObject(
        File(exists=True), mandatory=True, desc="list of input data",
    )
    in_meta = InputMultiObject(
        traits.DictStrAny, mandatory=True, desc="list of metadata",
    )
    max_trs = traits.Int(50, usedefault=True, desc="only pick first TRs")


class _FlattenOutputSpec(TraitedSpec):
    out_list = OutputMultiObject(
        traits.Tuple(File(exists=True), traits.DictStrAny,), desc="list of output files"
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
