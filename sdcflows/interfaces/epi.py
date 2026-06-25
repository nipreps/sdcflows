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
"""Interfaces to deal with the various types of fieldmap sources."""

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)


class _GetReadoutTimeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='EPI image corresponding to the metadata')
    metadata = traits.Dict(mandatory=True, desc='metadata corresponding to the inputs')
    use_estimate = traits.Bool(
        False, usedefault=True, desc='Use "Estimated*" fields to calculate TotalReadoutTime'
    )
    fallback = traits.Float(desc='A fallback value, in seconds.')


class _GetReadoutTimeOutputSpec(TraitedSpec):
    readout_time = traits.Float
    pe_direction = traits.Enum('i', 'i-', 'j', 'j-', 'k', 'k-')
    pe_dir_fsl = traits.Enum('x', 'x-', 'y', 'y-', 'z', 'z-')


class GetReadoutTime(SimpleInterface):
    """Calculate the readout time from available metadata."""

    input_spec = _GetReadoutTimeInputSpec
    output_spec = _GetReadoutTimeOutputSpec

    def _run_interface(self, runtime):
        from ..utils.epimanip import get_trt

        self._results['readout_time'] = get_trt(
            self.inputs.metadata,
            self.inputs.in_file if isdefined(self.inputs.in_file) else None,
            use_estimate=self.inputs.use_estimate,
            fallback=self.inputs.fallback or None,
        )
        self._results['pe_direction'] = self.inputs.metadata['PhaseEncodingDirection']
        self._results['pe_dir_fsl'] = (
            self.inputs.metadata['PhaseEncodingDirection']
            .replace('i', 'x')
            .replace('j', 'y')
            .replace('k', 'z')
        )
        return runtime


class _SelectPEVolumesInputSpec(BaseInterfaceInputSpec):
    in_data = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='list of input EPI files (3D or 4D), in the order they were listed',
    )
    pe_dirs_fsl = InputMultiObject(
        traits.Enum('x', 'x-', 'y', 'y-', 'z', 'z-'),
        mandatory=True,
        desc="per-file PE directions, in FSL's conventions",
    )
    readout_times = InputMultiObject(
        traits.Float, mandatory=True, desc='per-file total readout times'
    )
    max_vols_per_pe = traits.Int(
        3,
        usedefault=True,
        desc='maximum number of volumes to keep for each PE direction',
    )


class _SelectPEVolumesOutputSpec(TraitedSpec):
    out_data = OutputMultiObject(File(exists=True), desc='selected 3D volumes')
    pe_dirs_fsl = OutputMultiObject(
        traits.Enum('x', 'x-', 'y', 'y-', 'z', 'z-'),
        desc="per-volume PE directions, in FSL's conventions",
    )
    readout_times = OutputMultiObject(traits.Float, desc='per-volume total readout times')


class SelectPEVolumes(SimpleInterface):
    """
    Limit the data fed into TOPUP to a few volumes per phase-encoding direction.

    FSL's ``topup`` documentation recommends acquiring (and using) only a handful
    of volumes per PE direction, kept as "spares" against intra-volume motion that
    ``topup``'s own movement model then resolves. It also notes that passing more than a
    single opposing pair *"doesn't seem to make any difference, and only means that
    topup takes longer to run"*.

    This interface walks the input runs in the order they are listed, splits any 4D
    runs into their constituent 3D volumes, and keeps at most ``max_vols_per_pe``
    volumes for each PE direction (e.g. 3 blip-up and 3 blip-down). Once a
    direction's budget is full, any remaining volumes — and any additional runs of
    the same polarity — are dropped. Per-volume PE directions and readout times are
    propagated so that ``topup`` receives one acquisition-parameters row per volume.

    """

    input_spec = _SelectPEVolumesInputSpec
    output_spec = _SelectPEVolumesOutputSpec

    def _run_interface(self, runtime):
        from collections import defaultdict

        import nibabel as nb
        from nipype.utils.filemanip import fname_presuffix

        kept = defaultdict(int)
        out_data, out_pe_dirs, out_readout_times = [], [], []

        for in_file, pe_dir, readout_time in zip(
            self.inputs.in_data,
            self.inputs.pe_dirs_fsl,
            self.inputs.readout_times,
            strict=True,
        ):
            remaining = self.inputs.max_vols_per_pe - kept[pe_dir]
            if remaining <= 0:
                continue

            img = nb.squeeze_image(nb.load(in_file))
            volumes = nb.four_to_three(img) if img.ndim == 4 else [img]

            for i, volume in enumerate(volumes[:remaining]):
                out_file = fname_presuffix(in_file, suffix=f'_vol{i}', newpath=runtime.cwd)
                volume.to_filename(out_file)
                out_data.append(out_file)
                out_pe_dirs.append(pe_dir)
                out_readout_times.append(readout_time)

            kept[pe_dir] += min(remaining, len(volumes))

        self._results['out_data'] = out_data
        self._results['pe_dirs_fsl'] = out_pe_dirs
        self._results['readout_times'] = out_readout_times
        return runtime


class _SortPEBlipsInputSpec(BaseInterfaceInputSpec):
    in_data = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='list of input data',
    )
    pe_dirs_fsl = InputMultiObject(
        traits.Enum('x', 'x-', 'y', 'y-', 'z', 'z-'),
        mandatory=True,
        desc="list of PE directions, in FSL's conventions",
    )
    readout_times = InputMultiObject(
        traits.Float, mandatory=True, desc='list of total readout times'
    )


class _SortPEBlipsOutputSpec(TraitedSpec):
    out_data = OutputMultiObject(
        File(),
        desc='list of input data',
    )
    pe_dirs = OutputMultiObject(
        traits.Enum('i', 'i-', 'j', 'j-', 'k', 'k-'),
        desc="list of PE directions, in BIDS's conventions",
    )
    pe_dirs_fsl = OutputMultiObject(
        traits.Enum('x', 'x-', 'y', 'y-', 'z', 'z-'),
        desc="list of PE directions, in FSL's conventions",
    )
    readout_times = OutputMultiObject(traits.Float, desc='list of total readout times')


class SortPEBlips(SimpleInterface):
    """Sort PE blips so they are consistently fed into TOPUP."""

    input_spec = _SortPEBlipsInputSpec
    output_spec = _SortPEBlipsOutputSpec

    def _run_interface(self, runtime):
        # Put sign first
        blips = [f'+{pe[0]}' if len(pe) == 1 else f'-{pe[0]}' for pe in self.inputs.pe_dirs_fsl]
        sorted_inputs = sorted(
            zip(
                blips,
                self.inputs.readout_times,
                self.inputs.in_data,
                strict=False,
            )
        )

        (
            self._results['pe_dirs_fsl'],
            self._results['readout_times'],
            self._results['out_data'],
        ) = zip(*sorted_inputs, strict=False)

        # Put sign back last
        self._results['pe_dirs_fsl'] = [
            pe[1] if pe.startswith('+') else f'{pe[1]}-' for pe in self._results['pe_dirs_fsl']
        ]
        self._results['pe_dirs'] = [
            pe.replace('x', 'i').replace('y', 'j').replace('z', 'k')
            for pe in self._results['pe_dirs_fsl']
        ]
        return runtime
