#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "templateflow",
# ]
# ///
from templateflow import api as tfapi

tfapi.get("MNI152NLin2009cAsym", resolution=2, desc="brain", suffix="mask")
tfapi.get("MNI152NLin2009cAsym", resolution=1, label="brain", suffix="probseg")
tfapi.get("MNI152NLin2009cAsym", resolution=2, desc="fMRIPrep", suffix="boldref")
