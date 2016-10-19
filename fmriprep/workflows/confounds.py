'''
Workflow for discovering confounds.
Calculates frame displacement, segment regressors, global regressor, dvars, aCompCor, tCompCor
'''
from nipype.interfaces import utility, nilearn, ants, c3
from nipype.algorithms import confounds
from nipype.pipeline import engine as pe

from fmriprep.interfaces import mask
from fmriprep import interfaces

FAST_DEFAULT_SEGS = ['white matter', 'gray matter', 'CSF']

def discover_wf(settings, name="ConfoundDiscoverer"):
    ''' All input fields are required.

    Calculates global regressor and tCompCor
        from motion-corrected fMRI ('inputnode.fmri_file').
    Calculates DVARS from the fMRI and an EPI brain mask ('inputnode.epi_mask')
    Calculates frame displacement from MCFLIRT movement parameters ('inputnode.movpar_file')
    Calculates segment regressors and aCompCor
        from the fMRI and a white matter/gray matter/CSF segmentation ('inputnode.t1_seg'), after
        applying the transforms to the images ('inputnode.t1_transform', 'inputnode.epi_transform').
        Transforms are assumed to be ITK/ANTs formatted
        If no transform is needed, the transform fields may be set to 'identity'.

    Saves the confounds in a file ('outputnode.confounds_file')'''

    inputnode = pe.Node(utility.IdentityInterface(fields=['fmri_file', 'movpar_file', 't1_seg',
                                                          'epi_mask', 't1_transform',
                                                          'epi_transform', 'reference_image']),
                        name='inputnode')
    outputnode = pe.Node(utility.IdentityInterface(fields=['confounds_file']),
                         name='outputnode')

    # registration using ANTs
    t1_registration = pe.Node(ants.ApplyTransforms(interpolation='MultiLabel'), name='TransformT1')
    epi_registration = pe.Node(ants.WarpTimeSeriesImageMultiTransform(), name='TransformEPI')

    # Global and segment regressors
    signals = pe.Node(nilearn.SignalExtraction(include_global=True, detrend=True,
                                               class_labels=FAST_DEFAULT_SEGS),
                      name="SignalExtraction")
    # DVARS
    dvars = pe.Node(confounds.ComputeDVARS(save_all=True, remove_zerovariance=True),
                    name="ComputeDVARS")
    # Frame displacement
    frame_displace = pe.Node(confounds.FramewiseDisplacement(), name="FramewiseDisplacement")
    # CompCor
    tcompcor = pe.Node(confounds.TCompCor(components_file='tcompcor.tsv'), name="tCompCor")
    acompcor_roi = pe.Node(mask.BinarizeSegmentation(
        false_values=[FAST_DEFAULT_SEGS.index('gray matter') + 1, 0]), # 0 denotes background
                           name="CalcaCompCorROI")
    acompcor = pe.Node(confounds.ACompCor(components_file='acompcor.tsv'), name="aCompCor")

    # misc utilities
    concat = pe.Node(utility.Function(function=_gather_confounds, input_names=['signals', 'dvars',
                                                                               'frame_displace',
                                                                               'tcompcor',
                                                                               'acompcor'],
                                      output_names=['combined_out']),
                     name="ConcatConfounds")
    ds_confounds = pe.Node(interfaces.DerivativesDataSink(base_directory=settings['output_dir'],
                                                          suffix='confounds'),
                           name="DerivConfounds")

    workflow = pe.Workflow(name=name)
    workflow.connect([
        # connect inputnode to each non-anatomical confound node
        (inputnode, dvars, [('fmri_file', 'in_file'),
                            ('epi_mask', 'in_mask')]),
        (inputnode, frame_displace, [('movpar_file', 'in_plots')]),
        (inputnode, tcompcor, [('fmri_file', 'realigned_file')]),

        # anatomically-based confound computation requires coregistration
        (inputnode, t1_registration, [('reference_image', 'reference_image'),
                                      ('t1_seg', 'input_image'),
                                      ('t1_transform', 'transforms')]),
        (inputnode, epi_registration, [('fmri_file', 'input_image'),
                                       ('reference_image', 'reference_image'),
                                       ('epi_transform', 'transformation_series')]),

        # anatomical confound: signal extraction
        (t1_registration, signals, [('output_image', 'label_files')]),
        (epi_registration, signals, [('output_image', 'in_file')]),
        # anatomical confound: aCompCor
        (epi_registration, acompcor, [('output_image', 'realigned_file')]),
        (t1_registration, acompcor_roi, [('output_image', 'in_segments')]),
        (acompcor_roi, acompcor, [('out_mask', 'mask_file')]),

        # connect the confound nodes to the concatenate node
        (signals, concat, [('out_file', 'signals')]),
        (dvars, concat, [('out_all', 'dvars')]),
        (frame_displace, concat, [('out_file', 'frame_displace')]),
        (tcompcor, concat, [('components_file', 'tcompcor')]),
        (acompcor, concat, [('components_file', 'acompcor')]),

        (concat, outputnode, [('combined_out', 'confounds_file')]),

        # print stuff in derivatives
        (concat, ds_confounds, [('combined_out', 'in_file')]),
        (inputnode, ds_confounds, [('fmri_file', 'source_file')])
    ])

    return workflow

def _gather_confounds(signals=None, dvars=None, frame_displace=None, tcompcor=None, acompcor=None):
    ''' load confounds from the filenames, concatenate together horizontally, and re-save '''
    import pandas as pd
    import os.path as op

    all_files = [confound for confound in [signals, dvars, frame_displace, tcompcor, acompcor]
                 if confound != None]

    confounds = pd.DataFrame()
    for file_name in all_files: # assumes they all have headings already
        new = pd.read_csv(file_name, sep="\t")
        confounds = pd.concat((confounds, new), axis=1)

    combined_out = op.abspath('confounds.tsv')
    confounds.to_csv(combined_out, sep=str("\t"))

    return combined_out
