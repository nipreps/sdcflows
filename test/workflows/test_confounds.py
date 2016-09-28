''' Testing module for fmriprep.workflows.confounds '''
import unittest

from fmriprep.workflows.confounds import discover_wf

from test.workflows.utilities import TestWorkflow

class TestConfounds(TestWorkflow):
    ''' Testing class for fmriprep.workflows.confounds '''
    @unittest.skip
    def test_discover_wf(self):
        # set up

        # run
        workflow = discover_wf()
        workflow.write_graph()

        # assert

        # check some key paths
        self.assert_circular(workflow, [
            ('outputnode', 'inputnode', [('confounds_file', 'fmri_file')])
        ])

        # Make sure mandatory inputs are set
        self.assert_inputs_set(workflow, {
            'SignalExtraction': ['in_file', 'label_files', 'class_labels'],
            'ComputeDVARS': ['in_file', 'in_mask']
        })
