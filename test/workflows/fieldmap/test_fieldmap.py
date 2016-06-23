import json
from fmriprep.workflows.fieldmap import (se_pair_workflow,
                                         fieldmap_to_phasediff)
import re
import unittest
import mock
from nipype.pipeline import engine as pe

class TestFieldMap(unittest.TestCase):

    SOME_INT = 3

    def test_se_pair_workflow(self):
        # SET UP INPUTS
        mock_settings = {
            'work_dir': '.'
        }

        # RUN
        result = se_pair_workflow.se_pair_workflow(settings=mock_settings)

        # ASSERT
        self.assertEqual(result.name,
                         se_pair_workflow.SE_PAIR_WORKFLOW_NAME)
        self.assertIsInstance(result, pe.Workflow)

        # in lieu of a good way to check equality of DAGs
        result_nodes = [result.get_node(name).interface.__class__.__name__
                        for name in result.list_node_names()]
        self.assertItemsEqual(result_nodes,
                              ['Function', 'N4BiasFieldCorrection', 'BET',
                               'MCFLIRT', 'Merge', 'Split', 'TOPUP',
                               'ApplyTOPUP', 'Function', 'DataSink',
                               'IdentityInterface', 'ReadSidecarJSON',
                               'IdentityInterface'])

    @mock.patch('nibabel.load')
    @mock.patch('numpy.savetxt')
    def test_create_encoding_file(self, mock_savetxt, mock_load):
        # SETUP INPUTS
        fieldmaps = 'some_file.nii.gz'
        in_dict = { 'TotalReadoutTime': 'a_time',
                    'PhaseEncodingDirection': ['i']
        }

        # SET UP EXPECTATIONS
        mock_load(fieldmaps).shape = ['', '', '', self.SOME_INT]
        expected_enc_table = ([[1, 0, 0, in_dict['TotalReadoutTime']]] *
                              self.SOME_INT)

        # RUN
        out_file = se_pair_workflow.create_encoding_file(fieldmaps, in_dict)

        # ASSERT
        # the output file is called parameters.txt
        self.assertRegexpMatches(out_file, '/parameters.txt$')
        # nibabel.load was called with fieldmaps
        mock_load.assert_called_with(fieldmaps)
        # numpy.savetxt was called once. It was called with expected_enc_table
        mock_savetxt.assert_called_once_with(mock.ANY, expected_enc_table, 
                                             fmt=mock.ANY)

    def test_fieldmap_to_phasediff(self):
        pass
