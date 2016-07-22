import re
from future.utils import raise_from

from nipype.pipeline import Workflow

from fmriprep.utils import misc
from fmriprep.workflows.fieldmap.fieldmap_to_phasediff import fieldmap_to_phasediff
from fmriprep.workflows.fieldmap.se_pair_workflow import se_pair_workflow

class FieldmapDecider(Workflow):
    ''' Initialize FieldmapDecider to automatically find a
    Fieldmap preprocessing workflow '''

    # POSSIBLE FILES ACCORDING TO BIDS 1.0.0
    # 8.9.1 one phase diff image, at least one magnitude image
    # 8.9.2 two phase images, two magnitude images
    # 8.9.3 fieldmap image (and one magnitude image)
    # 8.9.4 multiple phase encoded directions (topup)

    # inputs/outputs could be specified with TraitedSpecs?
    inputs = { 'inputnode.fieldmaps' = None }
    outputs = { 'outputnode.mag_brain' = None, # fsl mcflirt aligns mags, fsl something (see epi.py for mean epi) average of magnitudes
                'outputnode.fmap_mask' = None, 
                'outputnode.fmap_fieldcoef' = None,
                'outputnode.fmap_movpar' = None]

    def __init__(settings):
        subj_data = misc.get_subject(settings['bids_root'], 
                                settings['subject_id'])

        try:
            a_filename = subj_data[next(iter(dictionary))]['fieldmaps'][0]
        except IndexError as e:
            raise_from(NotImplementedError("No fieldmap data found"), e)

        for filename in subj_data[next(iter(subj_data))]['fieldmaps']:
            if is_fmap_type('phasediff', filename): # 8.9.1
                return PhaseDiffAndMagnitudes()
            elif is_fmap_type('phase', filename): # 8.9.2
                raise NotImplementedError("No workflow for phase fieldmap data")
            elif is_fmap_type('fieldmap', filename): # 8.9.3
                return fieldmap_to_phasediff()
            elif is_fmap_type('topup', filename): #8.9.4
                return se_pair_workflow()

        raise IOError("Unrecognized fieldmap structure")

    def sort_fmaps(files):
        fmaps = {}
        for type in fieldmap_suffixes.keys():
            fmaps[type] = [doc for doc in files if is_fmap_type(type, doc)]
        # funky return statement so sort_fmaps can be a Function interface
        return (fmaps[key] for key in fieldmap_suffixes.keys().sort())

    def is_fmap_type(fmap_type, filename):
        return re.search(misc.fieldmap_suffixes[fmap_type], filename)
