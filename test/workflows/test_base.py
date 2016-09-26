from unittest import skip

from fmriprep.workflows.base import (wf_ds054_type, wf_ds005_type)
import mock
from test.workflows.utilities import (TestWorkflow, stub_node_factory)

@skip
@mock.patch('nipype.pipeline.engine.Node', side_effect=stub_node_factory)
class TestBase(TestWorkflow):

    def test_wf_ds054_type(self, mock_node):
        # set up
        mock_subject_data = ""
        mock_settings = {}

        # run
        wf054 = wf_ds054_type(mock_subject_data, mock_settings)
        wf054.run()

        # assert

    def test_wf_ds005_type(self, mock_node):
        # set up
        mock_subject_data = ""
        mock_settings = {}

        # run
        wf005 = wf_ds005_type(mock_subject_data, mock_settings)
        wf005.run()

        # assert
