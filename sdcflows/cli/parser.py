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

from sdcflows import config


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
SDCFlows {config.environment.version}

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
        "--version", action="version", version=f"SDCFlows {config.environment.version}"
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
        action=ParticipantLabelAction,
        nargs="+",
        help="A space delimited list of participant identifiers or a single "
        "identifier (the sub- prefix can be removed).",
    )
    g_bids.add_argument(
        "--session-label",
        action="store",
        nargs="*",
        type=str,
        help="Filter input dataset by session label.",
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
    g_bids.add_argument(
        "--bids-database-wipe",
        action="store_true",
        default=False,
        help="Wipe out previously existing BIDS indexing caches, forcing re-indexing.",
    )

    # General performance
    g_perfm = parser.add_argument_group("Options to handle performance")
    g_perfm.add_argument(
        "--nprocs",
        action="store",
        type=PositiveInt,
        help="""\
Maximum number of simultaneously running parallel processes executed by *SDCFlows* \
(e.g., several instances of ANTs' registration). \
However, when ``--nprocs`` is greater or equal to the ``--omp-nthreads`` option, \
it also sets the maximum number of threads that simultaneously running processes \
may aggregate (meaning, with ``--nprocs 16 --omp-nthreads 8`` a maximum of two \
8-CPU-threaded processes will be running at a given time). \
Under this mode of operation, ``--nprocs`` sets the maximum number of processors \
that can be assigned work within an *SDCFlows* job, which includes all the processors \
used by currently running single- and multi-threaded processes. \
If ``None``, the number of CPUs available will be automatically assigned (which may \
not be what you want in, e.g., shared systems like a HPC cluster.""",
    )
    g_perfm.add_argument(
        "--omp-nthreads",
        action="store",
        type=PositiveInt,
        help="""\
Maximum number of threads that multi-threaded processes executed by *SDCFlows* \
(e.g., ANTs' registration) can use. \
If ``None``, the number of CPUs available will be automatically assigned (which may \
not be what you want in, e.g., shared systems like a HPC cluster.""",
    )
    g_perfm.add_argument(
        "--mem-gb",
        dest="memory_gb",
        action="store",
        type=_to_gb,
        help="Upper bound memory limit for SDCFlows processes.",
    )
    g_perfm.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable changes to processing to aid in debugging",
    )
    g_perfm.add_argument(
        "--pdb",
        dest="pdb",
        action="store_true",
        default=False,
        help="Open Python debugger (pdb) on exceptions.",
    )
    g_perfm.add_argument(
        "--sloppy",
        action="store_true",
        default=False,
        help="Use sloppy mode for minimal footprint.",
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
        "--no-fmapless",
        action="store_false",
        dest="fmapless",
        default=True,
        help="Allow fieldmap-less estimation",
    )
    g_outputs.add_argument(
        "--use-plugin",
        action="store",
        default=None,
        type=Path,
        help="Nipype plugin configuration file.",
    )
    g_outputs.add_argument(
        "--notrack",
        action="store_true",
        help="Opt-out of sending tracking information of this run to the NiPreps developers. "
        "This information helps to improve SDCFlows and provides an indicator of "
        "real world usage for obtaining funding.",
    )

    return parser


def parse_args(args=None, namespace=None):
    """Parse args and run further checks on the command line."""
    from logging import DEBUG
    from json import loads

    parser = _parser()
    opts = parser.parse_args(args, namespace)
    config.execution.log_level = int(max(25 - 5 * opts.verbose_count, DEBUG))
    config.from_dict(vars(opts))

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        from yaml import load as loadyml

        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)
        _plugin = plugin_settings.get("plugin")
        if _plugin:
            config.nipype.plugin = _plugin
            config.nipype.plugin_args = plugin_settings.get("plugin_args", {})
            config.nipype.nprocs = config.nipype.plugin_args.get(
                "nprocs", config.nipype.nprocs
            )

    # Load BIDS filters
    if opts.bids_filter_file:
        config.execution.bids_filters = loads(opts.bids_filter_file.read_text())

    bids_dir = config.execution.bids_dir
    output_dir = config.execution.output_dir
    work_dir = config.execution.work_dir
    version = config.environment.version

    # Ensure input and output folders are not the same
    if output_dir == bids_dir:
        parser.error(
            "The selected output folder is the same as the input BIDS folder. "
            "Please modify the output path (suggestion: %s)."
            % bids_dir
            / "derivatives"
            / ("sdcflows_%s" % version.split("+")[0])
        )

    if bids_dir in work_dir.parents:
        parser.error(
            "The selected working directory is a subdirectory of the input BIDS folder. "
            "Please modify the output path."
        )

    # Setup directories
    config.execution.log_dir = output_dir / "logs"
    # Check and create output and working directories
    config.execution.log_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Force initialization of the BIDSLayout
    config.execution.init()

    participant_label = config.execution.layout.get_subjects()
    if config.execution.participant_label is not None:
        selected_label = set(config.execution.participant_label)
        missing_subjects = selected_label - set(participant_label)
        if missing_subjects:
            parser.error(
                "One or more participant labels were not found in the BIDS directory: "
                f"{', '.join(missing_subjects)}."
            )
        participant_label = selected_label

    config.execution.participant_label = sorted(participant_label)

    # Handle analysis_level
    analysis_level = set(config.workflow.analysis_level)
    config.workflow.analysis_level = list(analysis_level)
