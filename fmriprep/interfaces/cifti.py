#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Handling connectivity
~~~~~~~~~~~~~~~~~~~~~
Combines FreeSurfer surfaces with subcortical volumes

"""
import os
from glob import glob

import nibabel as nb
from nibabel import cifti2 as ci
import numpy as np

from niworkflows.nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec, File, traits, isdefined,
    SimpleInterface, Directory
)

from niworkflows.nipype.utils.filemanip import split_filename

# CITFI structures with corresponding FS labels
CIFTI_STRUCT_WITH_LABELS = {
    # SURFACES
    'CIFTI_STRUCTURE_CORTEX_LEFT': [],
    'CIFTI_STRUCTURE_CORTEX_RIGHT': [],
    # 'CIFTI_STRUCTURE_CORTEX': [],
    # 'CIFTI_STRUCTURE_CEREBRAL_WHITE_MATTER_LEFT': [],
    # 'CIFTI_STRUCTURE_CEREBRAL_WHITE_MATTER_RIGHT': [],

    # SUBCORTICAL
    'CIFTI_STRUCTURE_ACCUMBENS_LEFT': [26],
    'CIFTI_STRUCTURE_ACCUMBENS_RIGHT': [58],
    # 'CIFTI_STRUCTURE_ALL_WHITE_MATTER': [],
    # 'CIFTI_STRUCTURE_ALL_GREY_MATTER': [],
    'CIFTI_STRUCTURE_AMYGDALA_LEFT': [18],
    'CIFTI_STRUCTURE_AMYGDALA_RIGHT': [54],
    'CIFTI_STRUCTURE_BRAIN_STEM': [16],
    'CIFTI_STRUCTURE_CAUDATE_LEFT': [11],
    'CIFTI_STRUCTURE_CAUDATE_RIGHT': [50],
    'CIFTI_STRUCTURE_CEREBELLAR_WHITE_MATTER_LEFT': [7],
    'CIFTI_STRUCTURE_CEREBELLAR_WHITE_MATTER_RIGHT': [46],
    # 'CIFTI_STRUCTURE_CEREBELLUM': [],
    'CIFTI_STRUCTURE_CEREBELLUM_LEFT': [6],
    'CIFTI_STRUCTURE_CEREBELLUM_RIGHT': [45],
    'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT': [28],
    'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT': [60],
    'CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT': [17],
    'CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT': [53],
    # 'CIFTI_STRUCTURE_OTHER': [],
    # 'CIFTI_STRUCTURE_OTHER_GREY_MATTER': [],
    # 'CIFTI_STRUCTURE_OTHER_WHITE_MATTER': [],
    'CIFTI_STRUCTURE_PALLIDUM_LEFT': [13],
    'CIFTI_STRUCTURE_PALLIDUM_RIGHT': [52],
    'CIFTI_STRUCTURE_PUTAMEN_LEFT': [12],
    'CIFTI_STRUCTURE_PUTAMEN_RIGHT': [51],
    'CIFTI_STRUCTURE_THALAMUS_LEFT': [10],
    'CIFTI_STRUCTURE_THALAMUS_RIGHT': [49],
}


class GenerateCiftiInputSpec(BaseInterfaceInputSpec):
    bold_file = File(mandatory=True, exists=True, desc="input BOLD file")
    volume_target = traits.Enum("MNI152NLin2009cAsym", mandatory=True, usedefault=True,
                                desc="CIFTI volumetric output space")
    surface_target = traits.Enum("fsaverage5", "fsaverage6", "fsaverage", mandatory=True,
                                 usedefault=True, desc="CIFTI surface target space")
    subjects_dir = Directory(mandatory=True, desc="FreeSurfer SUBJECTS_DIR")
    TR = traits.Float(mandatory=True, desc="repetition time")
    gifti_files = traits.List(File(exists=True), mandatory=True,
                              desc="surface geometry files")


class GenerateCiftiOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="generated CIFTI file")


class GenerateCifti(SimpleInterface):
    """
    Generate CIFTI image from BOLD file in target space
    """
    input_spec = GenerateCiftiInputSpec
    output_spec = GenerateCiftiOutputSpec

    def _run_interface(self, runtime):
        self.annotation_files, self.label_file = self._fetch_data()
        self._results["out_file"] = create_cifti_image(
            self.bold_file,
            self.label_file,
            self.annotation_files,
            self.gifti_files,
            self.volume_target,
            self.surface_target,
            self.TR)

        return runtime

    def _fetch_data(self):
        """Converts inputspec to files"""
        if self.surface_target == "fsnative":
            raise NotImplementedError

        annotation_files = sorted(glob(os.path.join(self.subjects_dir,
                                                    self.surface_target,
                                                    'label',
                                                    '*h.aparc.annot')))
        # TODO: fetch label_file
        return annotation_files


def create_cifti_image(bold_file, label_file, annotation_files,
                       gii_files, volume_target, surface_target, tr):
    """
    Generate CIFTI image in target space
    """

    # grab image information
    bold_img = nb.load(bold_file)
    bold_data = bold_img.get_data()
    timepoints = bold_img.shape[3]
    label_data = nb.load(label_file).get_data()

    # set up CIFTI information
    model_type = "CIFTI_MODEL_TYPE_VOXELS"
    series_map = ci.Cifti2MatrixIndicesMap((0, ),
                                           'CIFTI_INDEX_TYPE_SERIES',
                                           number_of_series_points=timepoints,
                                           series_exponent=0,
                                           series_start=0.0,
                                           series_step=tr,
                                           series_unit='SECOND')
    # Create CIFTI brain models
    idx_offset = 0
    brainmodels = []
    bm_ts = np.empty((timepoints, 0))

    for structure, labels in CIFTI_STRUCT_WITH_LABELS.items():
        if not labels:  # surface model
            model_type = "CIFTI_MODEL_TYPE_SURFACE"
            # use the corresponding annotation
            hemi = structure.split('_')[-1][0]
            annots = [fl for fl in annotation_files if ".aparc.annot" in fl]
            annot = (nb.freesurfer.read_annot(annots[0]) if hemi == "LEFT"
                     else nb.freesurfer.read_annot(annots[1]))
            # currently only supports L/R cortex
            gii = (nb.load(gii_files[0]) if hemi == "LEFT"
                   else nb.load(gii_files[1]))
            # calculate total number of vertices
            surf_verts = len(annot[0])
            # remove medial wall for CIFTI format
            vert_idx = np.nonzero(annot[0] != annot[2].index(b'unknown'))[0]
            # extract values across volumes
            ts = np.array([tsarr.data[vert_idx] for tsarr in gii.darrays])

            vert_idx = ci.Cifti2VertexIndices(vert_idx)
            bm = ci.Cifti2BrainModel(index_offset=idx_offset,
                                     index_count=len(vert_idx),
                                     model_type=model_type,
                                     brain_structure=structure,
                                     vertex_indices=vert_idx)
            bm.surface_number_of_vertices = surf_verts

            bm_ts = np.column_stack((bm_ts, ts))
            idx_offset += len(vert_idx)
            brainmodels.append(bm)
        else:
            vox = []
            ts = []
            for label in labels:
                ijk = np.nonzero(label_data == label)
                ts.append(bold_data[ijk])
                vox += [[ijk[0][ix], ijk[1][ix], ijk[2][ix]]
                        for ix, row in enumerate(ts)]

            # ATM will not total ts across multiple labels
            bm_ts = np.column_stack((bm_ts, ts.T))

            vox = ci.Cifti2VoxelIndicesIJK(vox)
            bm = ci.Cifti2BrainModel(index_offset=idx_offset,
                                     index_count=len(vox),
                                     model_type=model_type,
                                     brain_structure=structure,
                                     voxel_indices_ijk=vox)
            idx_offset += len(vox)
            brainmodels.append(bm)

    volume = ci.Cifti2Volume(
        bold_img.shape[:3],
        ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, bold_img.affine)
    )
    brainmodels.append(volume)

    # create CIFTI geometry based on brainmodels
    geometry_map = ci.Cifti2MatrixIndicesMap((1, ),
                                             'CIFTI_INDEX_TYPE_BRAIN_MODELS',
                                             maps=brainmodels)
    # provide some metadata to CIFTI matrix
    meta = {
        "target_surface": surface_target,
        "target_volume": volume_target,
        "download_links": [
            "TODO: link",
            "TODO: link",
        ]}

    # generate and save CIFTI image
    matrix = ci.Cifti2Matrix()
    matrix.append(series_map)
    matrix.append(geometry_map)
    matrix.metadata = ci.Cifti2MetaData(meta)
    hdr = ci.Cifti2Header(matrix)
    img = ci.Cifti2Image(bm_ts, hdr)

    _, out_file, _ = split_filename(bold_file)
    ci.save(img, "{}.dtseries.nii".format(out_file))
    return out_file
