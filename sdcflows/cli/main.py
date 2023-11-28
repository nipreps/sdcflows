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
import re
from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
from pathlib import Path

from sdcflows import __version__ as thisversion


def _parse_participant_labels(value):
    """
    Drop ``sub-`` prefix of participant labels.

    >>> _parse_participant_labels("s060")
    ['s060']
    >>> _parse_participant_labels("sub-s060")
    ['s060']
    >>> _parse_participant_labels("s060 sub-s050")
    ['s050', 's060']
    >>> _parse_participant_labels("s060 sub-s060")
    ['s060']
    >>> _parse_participant_labels("s060\tsub-s060")
    ['s060']

    """
    return sorted(
        set(
            re.sub(r"^sub-", "", item.strip())
            for item in re.split(r"\s+", f"{value}".strip())
        )
    )


def _parser():
    class ParticipantLabelAction(Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, _parse_participant_labels(" ".join(values)))

    def _path_exists(path, parser):
        """Ensure a given path exists."""
        if path is None or not Path(path).exists():
            raise parser.error(f"Path does not exist: <{path}>.")
        return Path(path).expanduser().absolute()

    def _min_one(value, parser):
        """Ensure an argument is not lower than 1."""
        value = int(value)
        if value < 1:
            raise parser.error("Argument can't be less than one.")
        return value

    def _to_gb(value):
        scale = {"G": 1, "T": 10**3, "M": 1e-3, "K": 1e-6, "B": 1e-9}
        digits = "".join([c for c in value if c.isdigit()])
        n_digits = len(digits)
        units = value[n_digits:] or "G"
        return int(digits) * scale[units[0]]

    def _bids_filter(value):
        from json import loads

        if value and Path(value).exists():
            return loads(Path(value).read_text())

    parser = ArgumentParser(
        description=f"""\
SDCFlows {thisversion}

Estimate fieldmaps available in a BIDS-compliant MRI dataset.""",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    PathExists = partial(_path_exists, parser=parser)
    PositiveInt = partial(_min_one, parser=parser)

    parser.add_argument(
        "bids_dir",
        action="store",
        type=PathExists,
        help="The root folder of a BIDS valid dataset (sub-XXXXX folders should "
        "be found at the top level in this folder).",
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help="The directory where the output files "
        "should be stored. If you are running group level analysis "
        "this folder should be prepopulated with the results of the "
        "participant level analysis.",
    )
    parser.add_argument(
        "analysis_level",
        action="store",
        nargs="+",
        help="Level of the analysis that will be performed. "
        "Multiple participant level analyses can be run independently "
        "(in parallel) using the same output_dir.",
        choices=["participant", "group"],
    )

    # optional arguments
    parser.add_argument(
        "--version", action="version", version=f"SDCFlows {thisversion}"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose_count",
        action="count",
        default=0,
        help="Increases log verbosity for each occurrence, debug level is -vvv.",
    )

    # main options
    g_bids = parser.add_argument_group("Options related to BIDS")
    g_bids.add_argument(
        "--participant-label",
        "--participant_label",
        "--participant-labels",
        "--participant_labels",
        dest="participant_label",
        action=ParticipantLabelAction,
        nargs="+",
        help="A space delimited list of participant identifiers or a single "
        "identifier (the sub- prefix can be removed).",
    )
    g_bids.add_argument(
        "--bids-filter-file",
        action="store",
        type=Path,
        metavar="PATH",
        help="a JSON file describing custom BIDS input filter using pybids "
        "{<suffix>:{<entity>:<filter>,...},...} "
        "(https://github.com/bids-standard/pybids/blob/master/bids/layout/config/bids.json)",
    )
    g_bids.add_argument(
        "--bids-database-dir",
        metavar="PATH",
        type=PathExists,
        help="Path to an existing PyBIDS database folder, for faster indexing "
        "(especially useful for large datasets).",
    )

    # General performance
    g_perfm = parser.add_argument_group("Options to handle performance")
    g_perfm.add_argument(
        "--nprocs",
        "--n_procs",
        "--n_cpus",
        "-n-cpus",
        action="store",
        type=PositiveInt,
        help="""\
Maximum number of simultaneously running parallel processes executed by *MRIQC* \
(e.g., several instances of ANTs' registration). \
However, when ``--nprocs`` is greater or equal to the ``--omp-nthreads`` option, \
it also sets the maximum number of threads that simultaneously running processes \
may aggregate (meaning, with ``--nprocs 16 --omp-nthreads 8`` a maximum of two \
8-CPU-threaded processes will be running at a given time). \
Under this mode of operation, ``--nprocs`` sets the maximum number of processors \
that can be assigned work within an *MRIQC* job, which includes all the processors \
used by currently running single- and multi-threaded processes. \
If ``None``, the number of CPUs available will be automatically assigned (which may \
not be what you want in, e.g., shared systems like a HPC cluster.""",
    )
    g_perfm.add_argument(
        "--omp-nthreads",
        "--ants-nthreads",
        action="store",
        type=PositiveInt,
        help="""\
Maximum number of threads that multi-threaded processes executed by *MRIQC* \
(e.g., ANTs' registration) can use. \
If ``None``, the number of CPUs available will be automatically assigned (which may \
not be what you want in, e.g., shared systems like a HPC cluster.""",
    )
    g_perfm.add_argument(
        "--mem",
        "--mem_gb",
        "--mem-gb",
        dest="memory_gb",
        action="store",
        type=_to_gb,
        help="Upper bound memory limit for MRIQC processes.",
    )
    g_perfm.add_argument(
        "--testing",
        dest="debug",
        action="store_true",
        default=False,
        help="Use testing settings for a minimal footprint.",
    )

    g_perfm.add_argument(
        "--pdb",
        dest="pdb",
        action="store_true",
        default=False,
        help="Open Python debugger (pdb) on exceptions.",
    )

    # Control instruments
    g_outputs = parser.add_argument_group("Instrumental options")
    g_outputs.add_argument(
        "-w",
        "--work-dir",
        action="store",
        type=Path,
        default=Path("work").absolute(),
        help="Path where intermediate results should be stored.",
    )
    g_outputs.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        default=False,
        help="only find estimable fieldmaps (that is, estimation is not triggered)",
    )
    g_outputs.add_argument(
        "--fmapless",
        action="store_true",
        default=False,
        help="Allow fieldmap-less estimation",
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
    from logging import DEBUG
    from niworkflows.utils.bids import collect_participants
    from sdcflows.utils.wrangler import find_estimators
    from sdcflows.utils.misc import create_logger

    pargs = _parser().parse_args(argv)

    bids_dir = pargs.bids_dir.resolve(strict=True)
    layout = gen_layout(bids_dir, pargs.bids_database_dir)
    subjects = collect_participants(layout, pargs.participant_label)
    logger = create_logger(
        "sdcflow.wrangler", int(max(25 - 5 * pargs.verbose_count, DEBUG))
    )
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
