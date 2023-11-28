# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Standalone command line executable for estimation of fieldmaps."""
import argparse
from pathlib import Path


def _drop_sub(value):
    return value[4:] if value.startswith("sub-") else value


def _parser():
    parser = argparse.ArgumentParser(
        description="Parse a BIDS directory and show what estimators are available",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bids_dir", type=Path, help="The input BIDS directory to parse")
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        default=False,
        help="only find estimable fieldmaps (that is, estimation is not triggered)",
    )
    parser.add_argument(
        "-s",
        "--subjects",
        type=_drop_sub,
        nargs="+",
        help="One or more subject identifiers",
    )
    parser.add_argument(
        "--fmapless",
        action="store_true",
        default=False,
        help="Allow fieldmap-less estimation",
    )
    parser.add_argument(
        "--bids-database-dir",
        metavar="PATH",
        type=Path,
        help="Path to a PyBIDS database folder, for faster indexing (especially "
        "useful for large datasets). Will be created if not present.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print information while finding estimators (Useful for debugging)",
    )
    return parser


def gen_layout(bids_dir, database_dir=None):
    import re
    from bids.layout import BIDSLayout, BIDSLayoutIndexer

    _indexer = BIDSLayoutIndexer(
        validate=False,
        ignore=(
            "code",
            "stimuli",
            "sourcedata",
            "models",
            "derivatives",
            re.compile(r"^\."),
            re.compile(
                r"sub-[a-zA-Z0-9]+(/ses-[a-zA-Z0-9]+)?/(beh|eeg|ieeg|meg|micr|perf)"
            ),
        ),
    )

    layout_kwargs = {"indexer": _indexer}

    if database_dir:
        layout_kwargs["database_path"] = database_dir

    layout = BIDSLayout(bids_dir, **layout_kwargs)
    return layout


def main(argv=None):
    """
    Parse a BIDS directory and print subject estimators
    """
    from niworkflows.utils.bids import collect_participants
    from sdcflows.utils.wrangler import find_estimators
    from sdcflows.utils.misc import create_logger

    pargs = _parser().parse_args(argv)

    bids_dir = pargs.bids_dir.resolve(strict=True)
    layout = gen_layout(bids_dir, pargs.bids_database_dir)
    subjects = collect_participants(layout, pargs.subjects)
    logger = create_logger("sdcflow.wrangler", level=10 if pargs.verbose else 40)
    estimators_record = {}
    for subject in subjects:
        estimators_record[subject] = find_estimators(
            layout=layout,
            subject=subject,
            fmapless=pargs.fmapless,
            logger=logger,
        )

    # pretty print results
    if pargs.dry_run:
        print(f"Estimation for <{str(bids_dir)}> complete. Found:")
        for subject, estimators in estimators_record.items():
            print(f"\tsub-{subject}")
            if not estimators:
                print("\t\tNo estimators found")
                continue
            for estimator in estimators:
                print(f"\t\t{estimator}")
                for fl in estimator.sources:
                    fl_relpath = fl.path.relative_to(str(bids_dir / f"sub-{subject}"))
                    pe_dir = fl.metadata.get("PhaseEncodingDirection")
                    print(f"\t\t\t{pe_dir}\t{fl_relpath}")
        return


if __name__ == "__main__":
    main()
