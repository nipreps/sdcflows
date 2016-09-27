''' Testing class for fmriprep.workflows.base '''
import mock
from traits.trait_base import _Undefined as trait_undefined

from fmriprep.workflows.base import wf_ds054_type, wf_ds005_type
from test.workflows.utilities import TestWorkflow

@mock.patch('fmriprep.interfaces.BIDSDataGrabber') # no actual BIDS dir necessary
class TestBase(TestWorkflow):

    def test_wf_ds054_type(self, _):
        # set up
        mock_subject_data = {'t1w': ['um'], 'sbref': ['um'], 'func': 'um'}
        mock_settings = {'output_dir': '.', 'work_dir': '.'}

        # run
        wf054 = wf_ds054_type(mock_subject_data, mock_settings)
        wf054.write_graph()

        # assert

        # check some key paths
        self.assert_circular(wf054, [
            ('SBrefSpatialNormalization', 'BIDSDatasource',
             [('outputnode.mat_sbr_to_t1', 'subject_data')]),
            ('EPIUnwarpWorkflow', 'BIDSDatasource', [('outputnode.epi_mean', 'subject_data')]),
            ('ConfoundDiscoverer', 'BIDSDatasource',
             [('outputnode.confounds_file', 'subject_data')]),
            ('EPI_SBrefRegistration', 'BIDSDatasource', [('outputnode.out_mat', 'subject_data')]),
            ('EPIUnwarpWorkflow', 'EPI_HMC', [('outputnode.epi_mean', 'inputnode.epi')]),
            ('ConfoundDiscoverer', 'EPI_HMC', [('outputnode.confounds_file', 'inputnode.epi')]),
            ('EPI_SBrefRegistration', 'EPI_HMC', [('outputnode.out_mat', 'inputnode.epi')])
        ])

        # Make sure mandatory input is set
        self.assertNotEqual(wf054.get_node('BIDSDatasource').inputs.subject_data.__class__,
                            trait_undefined)

    def test_wf_ds005_type(self, _):
        # set up
        mock_subject_data = {'func': ''}
        mock_settings = {'output_dir': '.'}

        # run
        wf005 = wf_ds005_type(mock_subject_data, mock_settings)
        wf005.write_graph()

        # assert
        self.assert_circular(wf005, [
            ('EPIMNITransformation', 'BIDSDatasource',
             [('DerivativesHMCMNI.out_file', 'subject_data')]),
            ('EPIMNITransformation', 'EPI_HMC', [('DerivativesHMCMNI.out_file', 'inputnode.epi')]),
            ('EPIMeanNormalization', 'EPI_HMC', [('outputnode.mat_epi_to_t1', 'inputnode.epi')]),
        ])

        self.assertNotEqual(wf005.get_node('BIDSDatasource').inputs.subject_data.__class__,
                            trait_undefined)
