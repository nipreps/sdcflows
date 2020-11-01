# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities.

    .. testsetup::

        >>> tmpdir = getfixture('tmpdir')
        >>> tmp = tmpdir.chdir() # changing to a temporary directory
        >>> nb.Nifti1Image(np.zeros((90, 90, 60)), None, None).to_filename(
        ...     tmpdir.join('epi.nii.gz').strpath)

"""

from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    SimpleInterface,
    InputMultiObject,
    OutputMultiObject,
)

LOGGER = logging.getLogger("nipype.interface")


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
            zip(self.inputs.inlist, self.inputs.in_meta),
            max_trs=self.inputs.max_trs,
            out_dir=runtime.cwd,
        )

        # Unzip out_data, out_meta outputs.
        self._results["out_data"], self._results["out_meta"] = zip(
            *self._results["out_list"]
        )
        return runtime


def _flatten(inlist, max_trs=50, out_dir=None):
    """
    Split the input EPIs and generate a flattened list with corresponding metadata.

    Inputs
    ------
    inlist : :obj:`list` of :obj:`tuple`
        List of pairs (filepath, metadata)
    max_trs : int
        Index of frame after which all volumes will be discarded
        from the input EPI images.

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
