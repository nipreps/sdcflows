import json
import fmriprep.workflows.base as base
import re
import unittest
import mock

class TestBase(unittest.TestCase):

    def test_fmri_preprocess_single(self):
        ''' Tests that it runs without errors '''
        # NOT a test for correctness
        # SET UP INPUTS
        test_settings = {
            'output_dir': '.',
            'work_dir': '.'
        }

        # SET UP EXPECTATIONS

        # RUN
        base.fmri_preprocess_single(settings=test_settings)

        # ASSERT
