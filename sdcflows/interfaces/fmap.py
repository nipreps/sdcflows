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
import os
import numpy as np
import nibabel as nb
import nitransforms as nt
from nipype.utils.filemanip import fname_presuffix
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
from nipype.interfaces import freesurfer as fs

LOGGER = logging.getLogger("nipype.interface")


class _PhaseMap2radsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input (wrapped) phase map")


class _PhaseMap2radsOutputSpec(TraitedSpec):
    out_file = File(desc="the phase map in the range 0 - 6.28")


class PhaseMap2rads(SimpleInterface):
    """Convert a phase map given in a.u. (e.g., 0-4096) to radians."""

    input_spec = _PhaseMap2radsInputSpec
    output_spec = _PhaseMap2radsOutputSpec

    def _run_interface(self, runtime):
        from ..utils.phasemanip import au2rads

        self._results["out_file"] = au2rads(self.inputs.in_file, newpath=runtime.cwd)
        return runtime


class _SubtractPhasesInputSpec(BaseInterfaceInputSpec):
    in_phases = traits.List(File(exists=True), min=1, max=2, desc="input phase maps")
    in_meta = traits.List(
        traits.Dict(), min=1, max=2, desc="metadata corresponding to the inputs"
    )


class _SubtractPhasesOutputSpec(TraitedSpec):
    phase_diff = File(exists=True, desc="phase difference map")
    metadata = traits.Dict(desc="output metadata")


class SubtractPhases(SimpleInterface):
    """Calculate a phase difference map."""

    input_spec = _SubtractPhasesInputSpec
    output_spec = _SubtractPhasesOutputSpec

    def _run_interface(self, runtime):
        if len(self.inputs.in_phases) != len(self.inputs.in_meta):
            raise ValueError(
                "Length of input phase-difference maps and metadata files "
                "should match."
            )

        if len(self.inputs.in_phases) == 1:
            self._results["phase_diff"] = self.inputs.in_phases[0]
            self._results["metadata"] = self.inputs.in_meta[0]
            return runtime

        from ..utils.phasemanip import subtract_phases as _subtract_phases

        # Discard in_meta traits with copy(), so that pop() works.
        self._results["phase_diff"], self._results["metadata"] = _subtract_phases(
            self.inputs.in_phases,
            (self.inputs.in_meta[0].copy(), self.inputs.in_meta[1].copy()),
            newpath=runtime.cwd,
        )

        return runtime


class _Phasediff2FieldmapInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input fieldmap")
    metadata = traits.Dict(mandatory=True, desc="BIDS metadata dictionary")


class _Phasediff2FieldmapOutputSpec(TraitedSpec):
    out_file = File(desc="the output fieldmap")


class Phasediff2Fieldmap(SimpleInterface):
    """Convert a phase difference map into a fieldmap in Hz."""

    input_spec = _Phasediff2FieldmapInputSpec
    output_spec = _Phasediff2FieldmapOutputSpec

    def _run_interface(self, runtime):
        from ..utils.phasemanip import phdiff2fmap, delta_te as _delta_te

        self._results["out_file"] = phdiff2fmap(
            self.inputs.in_file, _delta_te(self.inputs.metadata), newpath=runtime.cwd
        )
        return runtime


class _CheckB0UnitsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input fieldmap")
    units = traits.Enum("Hz", "rad/s", mandatory=True, desc="fieldmap units")


class _CheckB0UnitsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output fieldmap in Hz")


class CheckB0Units(SimpleInterface):
    """Ensure the input fieldmap is given in Hz."""

    input_spec = _CheckB0UnitsInputSpec
    output_spec = _CheckB0UnitsOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.units == "Hz":
            self._results["out_file"] = self.inputs.in_file
            return runtime

        self._results["out_file"] = fname_presuffix(
            self.inputs.in_file, suffix="_Hz", newpath=runtime.cwd
        )
        img = nb.load(self.inputs.in_file)
        data = np.asanyarray(img.dataobj) / (2.0 * np.pi)
        img.__class__(data, img.affine, img.header).to_filename(
            self._results["out_file"]
        )
        return runtime


class _DisplacementsField2FieldmapInputSpec(BaseInterfaceInputSpec):
    transform = File(exists=True, mandatory=True, desc="input displacements field")
    epi = File(exists=True, mandatory=True, desc="source EPI image")
    ro_time = traits.Float(mandatory=True, desc="total readout time")
    pe_dir = traits.Enum(
        "j-", "j", "i", "i-", "k", "k-", mandatory=True, desc="phase encoding direction"
    )
    demean = traits.Bool(False, usedefault=True, desc="regress field to the mean")
    itk_transform = traits.Bool(
        True, usedefault=True, desc="whether this is an ITK/ANTs transform"
    )


class _DisplacementsField2FieldmapOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output fieldmap in Hz")


class DisplacementsField2Fieldmap(SimpleInterface):
    """Convert from a transform to a B0 fieldmap in Hz."""

    input_spec = _DisplacementsField2FieldmapInputSpec
    output_spec = _DisplacementsField2FieldmapOutputSpec

    def _run_interface(self, runtime):
        from sdcflows.transform import disp_to_fmap

        self._results["out_file"] = fname_presuffix(
            self.inputs.transform, suffix="_Hz", newpath=runtime.cwd
        )
        fmapnii = disp_to_fmap(
            nb.load(self.inputs.transform),
            nb.load(self.inputs.epi),
            ro_time=self.inputs.ro_time,
            pe_dir=self.inputs.pe_dir,
            itk_format=self.inputs.itk_transform,
        )

        if self.inputs.demean:
            data = np.asanyarray(fmapnii.dataobj)
            data -= np.median(data)

            fmapnii = fmapnii.__class__(
                data.astype("float32"),
                fmapnii.affine,
                fmapnii.header,
            )

        fmapnii.to_filename(self._results["out_file"])
        return runtime


class _CheckRegisterInputSpec(TraitedSpec):
    mag_files = InputMultiObject(
        File(exists=True),
        mandatory=True,
        minlen=1,
        maxlen=2,
        desc="Magnitude image(s) to verify registration",
    )
    fmap_files = InputMultiObject(
        File(exists=True),
        mandatory=True,
        minlen=1,
        maxlen=2,
        desc="Phase(diff) or fieldmap image(s) to update affines",
    )
    rot_thresh = traits.Float(
        0.02, usedefault=True, mandatory=True, desc="rotation threshold in radians"
    )
    trans_thresh = traits.Float(
        1., usedefault=True, mandatory=True, desc="translation threshold in mm"
    )


class _CheckRegisterOutputSpec(TraitedSpec):
    mag_files = OutputMultiObject(
        File,
        desc="Magnitude image(s) verified to be in register, with consistent affines",
    )
    fmap_files = OutputMultiObject(
        File,
        desc="Fieldmap files with consistent affines",
    )


class CheckRegister(SimpleInterface):
    """Check registration of one or more images and paired files

    Use cases:

    Phase1 + Phase2:

        >>> fmap_files = ['*_phase1.nii.gz', '*_phase2.nii.gz']
        >>> mag_files = ['*_magnitude1.nii.gz', '*_magnitude2.nii.gz']

    Phasediff (2 magnitude):

        >>> fmap_files = ['*_phasediff.nii.gz']
        >>> mag_files = ['*_magnitude1.nii.gz', '*_magnitude2.nii.gz']

    Phasediff (1 magnitude):

        >>> fmap_files = ['*_phasediff.nii.gz']
        >>> mag_files = ['*_magnitude1.nii.gz']

    Fieldmap (1 magnitude):

        >>> fmap_files = ['*_fieldmap.nii.gz']
        >>> mag_files = ['*_magnitude.nii.gz']

    In general, we expect all files to have the same affine and shape,
    and this will be a pass-through interface. If there are two magnitude
    files where the affine differs, then we will register them and inspect
    the registration parameters for evidence of significantly different FoV.
    The default values of 0.02rad and 1mm both correspond to shifts of 1mm
    (assuming 50mm radius brain).

    This may run ``mri_robust_register``, so should not be run without
    submitting.
    """

    input_spec = _CheckRegisterInputSpec
    output_spec = _CheckRegisterOutputSpec

    def _run_interface(self, runtime):
        mag_files = self.inputs.mag_files
        fmap_files = self.inputs.fmap_files

        mag_imgs = [nb.load(fname) for fname in mag_files]
        fmap_imgs = [nb.load(fname) for fname in fmap_files]

        # Baseline check: paired magnitude/phase maps are basically the same
        for mag, fmap in zip(mag_imgs, fmap_imgs):
            msg = _check_gross_geometry(mag, fmap)
            if msg is not None:
                LOGGER.critical(msg)
                raise ValueError(msg)

        # Verify images are in register before conforming affines
        if len(mag_files) == 2:
            msg = _check_gross_geometry(mag_imgs[0], mag_imgs[1])
            if msg is not None:
                LOGGER.critical(msg)
                raise ValueError(msg)

            # If affines match, do not attempt to register
            # Treat this as an assertion by the scanner or curator that they are aligned
            if not np.allclose(mag_imgs[0].affine, mag_imgs[1].affine):
                reg = fs.RobustRegister(
                    target_file=mag_files[0],
                    source_file=mag_files[1],
                    auto_sens=True,
                )
                result = reg.run()
                lta = nt.io.lta.FSLinearTransform.from_filename(result.outputs.out_reg_file)
                mat, vec = nb.affines.to_matvec(lta.to_ras())
                angles = np.abs(nb.eulerangles.mat2euler(mat))
                rot_thresh, trans_thresh = (
                    self.inputs.rot_thresh,
                    self.inputs.trans_thresh,
                )

                if np.any(angles > rot_thresh) or np.any(vec > trans_thresh):
                    LOGGER.critical(
                        "Magnitude files {mag_files} are not in register with rotation "
                        f"threshold {self.inputs.rot_thresh} and translation threshold "
                        f"{self.inputs.trans_thresh}. Please manually verify images "
                        "are in register and update the image headers before running SDC."
                    )
                    raise ValueError(
                        "Magnitude 1/2 orientation mismatch too big to ignore."
                    )

        # Probably redundant, but we could hit this error
        # with phase1/magnitude1 + wonky phase2 + no magnitude2
        if len(fmap_files) == 2:
            msg = _check_gross_geometry(fmap_imgs[0], fmap_imgs[1])
            if msg is not None:
                LOGGER.critical(msg)
                raise ValueError(msg)

        # Check/copy affines
        out_mags = [_conform_img(mag, mag_imgs[0], runtime.cwd) for mag in mag_imgs]
        out_fmaps = [_conform_img(fmap, mag_imgs[0], runtime.cwd) for fmap in fmap_imgs]

        self._results = {
            "mag_files": out_mags,
            "fmap_files": out_fmaps,
        }

        return runtime


def _conform_img(
    img: nb.spatialimages.SpatialImage,
    target_img: nb.spatialimages.SpatialImage,
    cwd: str,
) -> str:
    """Return path to image matching target_img geometry

    Copy target_affine to a new image if necessary.
    """
    if np.allclose(img.affine, target_img.affine):
        return img.get_filename()

    basename = os.path.basename(img.get_filename())
    out_file = os.path.join(cwd, basename)

    LOGGER.info(f"Copying affine to {basename}")
    new_img = img.__class__(img.dataobj, target_img.affine, img.header)
    new_img.to_filename(out_file)

    return out_file


def _check_gross_geometry(
    img1: nb.spatialimages.SpatialImage,
    img2: nb.spatialimages.SpatialImage,
):
    if img1.shape[:3] != img2.shape[:3]:
        return (
            "Images have shape mismatch: "
            f"{img1.get_filename()} {img1.shape}, "
            f"{img2.get_filename()} {img2.shape}"
        )
    if nb.aff2axcodes(img1.affine) != nb.aff2axcodes(img2.affine):
        return (
            "Images have orientation mismatch: "
            f"{img1.get_filename()} {''.join(nb.aff2axcodes(img1.affine))}, "
            f"{img2.get_filename()} {''.join(nb.aff2axcodes(img2.affine))}"
        )
