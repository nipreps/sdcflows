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


def main(argv=None):
    """Entry point for SDCFlows' CLI."""
    import gc
    import os
    import sys
    from tempfile import mktemp
    import atexit
    from sdcflows import config
    from sdcflows.cli.parser import parse_args

    atexit.register(config.restore_env)

    # Run parser
    parse_args(argv)

    if config.execution.pdb:
        from niworkflows.utils.debug import setup_exceptionhook

        setup_exceptionhook()
        config.nipype.plugin = "Linear"

    # CRITICAL Save the config to a file. This is necessary because the execution graph
    # is built as a separate process to keep the memory footprint low. The most
    # straightforward way to communicate with the child process is via the filesystem.
    # The config file name needs to be unique, otherwise multiple sdcflows instances
    # will create write conflicts.
    config_file = mktemp(
        dir=config.execution.work_dir, prefix=".sdcflows.", suffix=".toml"
    )
    config.to_filename(config_file)
    config.file_path = config_file
    exitcode = 0

    if config.workflow.analysis_level != ["participant"]:
        raise ValueError("Analysis level can only be 'participant'")

    if config.execution.dry_run:  # --dry-run: pretty print results
        from niworkflows.utils.bids import collect_participants
        from sdcflows.utils.wrangler import find_estimators

        subjects = collect_participants(
            config.execution.layout,
            config.execution.participant_label,
        )
        estimators_record = {}
        for subject in subjects:
            estimators_record[subject] = find_estimators(
                layout=config.execution.layout,
                subject=subject,
                fmapless=config.workflow.fmapless,
                logger=config.loggers.cli,
            )

        print(f"Estimation for <{config.execution.bids_dir}> complete. Found:")
        for subject, estimators in estimators_record.items():
            print(f"\tsub-{subject}")
            if not estimators:
                print("\t\tNo estimators found")
                continue
            for estimator in estimators:
                print(f"\t\t{estimator}")
                for fl in estimator.sources:
                    fl_relpath = fl.path.relative_to(config.execution.bids_dir / f"sub-{subject}")
                    pe_dir = fl.metadata.get("PhaseEncodingDirection")
                    print(f"\t\t\t{pe_dir}\t{fl_relpath}")
        sys.exit(exitcode)

    # Initialize process pool if multiprocessing
    _pool = None
    if config.nipype.plugin in ("MultiProc", "LegacyMultiProc"):
        from contextlib import suppress
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        os.environ["OMP_NUM_THREADS"] = "1"

        with suppress(RuntimeError):
            mp.set_start_method("fork")
        gc.collect()

        _pool = ProcessPoolExecutor(
            max_workers=config.nipype.nprocs,
            initializer=config._process_initializer,
            initargs=(config.file_path,),
        )

    if not config.execution.notrack:
        from sdcflows.utils.telemetry import setup_migas

        setup_migas()

    # CRITICAL Call build_workflow(config_file, retval) in a subprocess.
    # Because Python on Linux does not ever free virtual memory (VM), running the
    # workflow construction jailed within a process preempts excessive VM buildup.
    from multiprocessing import Manager, Process

    with Manager() as mgr:
        from sdcflows.cli.workflow import build_workflow

        retval = mgr.dict()
        p = Process(target=build_workflow, args=(str(config_file), retval))
        p.start()
        p.join()

        sdcflows_wf = retval.get("workflow", None)
        exitcode = p.exitcode or retval.get("return_code", 0)

    # CRITICAL Load the config from the file. This is necessary because the ``build_workflow``
    # function executed constrained in a process may change the config (and thus the global
    # state of SDCFlows).
    config.load(config_file)

    exitcode = exitcode or (sdcflows_wf is None) * os.EX_SOFTWARE
    if exitcode != 0:
        sys.exit(exitcode)

    if len(sdcflows_wf.list_node_names()) == 0:
        config.loggers.cli.critical(
            'Workflow did not generate any jobs. Please check your inputs are valid.'
        )
        sys.exit(os.EX_USAGE)

    # Initialize nipype config
    config.nipype.init()
    # Make sure loggers are started
    config.loggers.init()

    # Resource management options
    if config.nipype.plugin in ("MultiProc", "LegacyMultiProc") and (
        1 < config.nipype.nprocs < config.nipype.omp_nthreads
    ):
        config.loggers.cli.warning(
            "Per-process threads (--omp-nthreads=%d) exceed total "
            "threads (--nthreads/--n_cpus=%d)",
            config.nipype.omp_nthreads,
            config.nipype.nprocs,
        )

    if sdcflows_wf is None:
        sys.exit(os.EX_SOFTWARE)

    if sdcflows_wf and config.execution.write_graph:
        sdcflows_wf.write_graph(graph2use="colored", format="svg", simple_form=True)

    # Clean up master process before running workflow, which may create forks
    gc.collect()
    # run SDCFlows
    _plugin = config.nipype.get_plugin()
    if _pool:
        from niworkflows.engine.plugin import MultiProcPlugin

        _plugin = {
            "plugin": MultiProcPlugin(
                pool=_pool, plugin_args=config.nipype.plugin_args
            ),
        }
    sdcflows_wf.run(**_plugin)
    config.loggers.cli.log(25, "Finished all workload")


if __name__ == "__main__":
    main()
