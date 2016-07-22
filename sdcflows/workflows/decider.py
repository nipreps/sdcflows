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

    inputs = { 'fieldmaps': None }
    outputs = { 'fieldmaps': None,
                'outputnode.mag_brain': None,
                'outputnode.fmap_mask': None,
                'outputnode.fmap_fieldcoef': None,
                'outputnode.fmap_movpar': None}

    def __init__(self, subject_data, settings):

        try:
            a_filename = subject_data['fieldmaps'][0]
        except IndexError as e:
            raise_from(NotImplementedError("No fieldmap data found"), e)

        for filename in subject_data['fieldmaps']:
            if re.search(misc.fieldmap_suffixes['phasediff'], a_filename): # 8.9.1
                self = PhaseDiffAndMagnitudes(settings=settings)
                return
            elif re.search(misc.fieldmap_suffixes['phase'], a_filename): # 8.9.2
                raise NotImplementedError("No workflow for phase fieldmap data")
                return
            elif re.search(misc.fieldmap_suffixes['fieldmap'], a_filename): # 8.9.3
                self = fieldmap_to_phasediff(settings=settings) # ???
                return
            elif re.search(misc.fieldmap_suffixes['topup'], a_filename): #8.0.4
                self = se_pair_workflow(settings=settings) # ???
                return

        raise IOError("Unrecognized fieldmap structure")

    def sort_fmaps(files):
        fmaps = {}
        for type in fieldmap_suffixes.keys():
            fmaps[type] = [doc for doc in files if is_fmap_type(type, doc)]
        # funky return statement so sort_fmaps can be a Function interface
        return (fmaps[key] for key in fieldmap_suffixes.keys().sort())

    def is_fmap_type(fmap_type, filename):
        return re.search(misc.fieldmap_suffixes[fmap_type], filename)

def fieldmap_decider(subject_data, settings):
    ''' Initialize FieldmapDecider to automatically find a
    Fieldmap preprocessing workflow '''

    # POSSIBLE FILES ACCORDING TO BIDS 1.0.0
    # 8.9.1 one phase diff image, at least one magnitude image
    # 8.9.2 two phase images, two magnitude images
    # 8.9.3 fieldmap image (and one magnitude image)
    # 8.9.4 multiple phase encoded directions (topup)

    inputs = { 'fieldmaps': None }
    outputs = { 'fieldmaps': None,
                'outputnode.mag_brain': None,
                'outputnode.fmap_mask': None,
                'outputnode.fmap_fieldcoef': None,
                'outputnode.fmap_movpar': None}

    try:
        a_filename = subject_data['fieldmaps'][0]
    except IndexError as e:
        raise_from(NotImplementedError("No fieldmap data found"), e)

    for filename in subject_data['fieldmaps']:
        if re.search(misc.fieldmap_suffixes['phasediff'], a_filename): # 8.9.1
            return PhaseDiffAndMagnitudes(settings=settings)
        elif re.search(misc.fieldmap_suffixes['phase'], a_filename): # 8.9.2
            raise NotImplementedError("No workflow for phase fieldmap data")
        elif re.search(misc.fieldmap_suffixes['fieldmap'], a_filename): # 8.9.3
            return fieldmap_to_phasediff(settings=settings) # ???
        elif re.search(misc.fieldmap_suffixes['topup'], a_filename): #8.0.4
            return se_pair_workflow(settings=settings) # ???

    raise IOError("Unrecognized fieldmap structure")
