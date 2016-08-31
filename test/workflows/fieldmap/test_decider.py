import logging
import unittest

from fmriprep.workflows.fieldmap.base import fieldmap_decider
from fmriprep.utils import misc

from test.workflows.utilities import TestWorkflow

class TestDecider(TestWorkflow):
    # ideally this would be a true unit test but in the interests of
    # time it is more like a smoke test
    # also it won't run in circle ci or anywhere w/o some set up
    @unittest.skip("see above; uncomment for development")
    def test_decider_phasediff_and_mags(self):
        logging.basicConfig(level=logging.DEBUG)
        # SET UP INPUTS
        subject_data = misc.get_subject(c.DATASET, '100003')

        # RUN
        wf = fieldmap_decider(subject_data, {})

        # Unit tests of the decider should check different BIDS-valid
        # structures and see if the decision is correct. There's not
        # much of a point in running smoke test wrapped as unit tests.
        # If we also test some BIDS-invalid inputs... that'd be great!
