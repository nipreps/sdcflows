''' Testing module for fmriprep.workflows.confounds '''
import mock
import pandas as pd

from fmriprep.workflows.confounds import discover_wf, _gather_confounds

from test.workflows.utilities import TestWorkflow
from test.workflows import stub

class TestConfounds(TestWorkflow):
    ''' Testing class for fmriprep.workflows.confounds '''

    def test_discover_wf(self):
        # set up

        # run
        workflow = discover_wf(stub.settings())
        workflow.write_graph()

        # assert

        # check some key paths
        self.assert_circular(workflow, [
            ('outputnode', 'inputnode', [('confounds_file', 'fmri_file')]),
            ('DerivConfounds', 'inputnode', [('out_file', 'fmri_file')])
        ])

        # Make sure mandatory inputs are set
        self.assert_inputs_set(workflow, {'outputnode': ['confounds_file'],
                                          'ConcatConfounds': ['signals', 'dvars', 'frame_displace',
                                                              'tcompcor', 'acompcor'],
                                          'tCompCor': ['components_file'],
                                          'aCompCor': ['components_file', 'mask_file'], })

    def test_gather_confounds_err(self):
        # set up
        signals = "signals.tsv"
        dvars = "signals.tsv"

        # run & assert
        with self.assertRaisesRegex(RuntimeError, "confound"):
            _gather_confounds(signals, dvars)

    @mock.patch('pandas.read_csv')
    @mock.patch.object(pd.DataFrame, 'to_csv', autospec=True)
    @mock.patch.object(pd.DataFrame, '__eq__', autospec=True,
                       side_effect=lambda me, them: me.equals(them))
    def test_gather_confounds(self, df_equality, mock_df, mock_csv_reader):
        # set up
        signals = "signals.tsv"
        dvars = "dvars.tsv"

        mock_csv_reader.side_effect = [pd.DataFrame({'a': [0.1]}), pd.DataFrame({'b': [0.2]})]

        # run
        _gather_confounds(signals, dvars)

        # assert
        calls = [mock.call(confounds, sep="\t") for confounds in [signals, dvars]]
        mock_csv_reader.assert_has_calls(calls)

        confounds = pd.DataFrame({'a': [0.1], 'b': [0.2]})
        mock_df.assert_called_once_with(confounds, "confounds.tsv", sep="\t")
