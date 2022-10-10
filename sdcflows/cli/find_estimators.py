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
             "useful for large datasets). Will be created if not present."
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
            re.compile(r"sub-[a-zA-Z0-9]+(/ses-[a-zA-Z0-9]+)?/(beh|eeg|ieeg|meg|micr|perf)"),
        ),
    )

    layout_kwargs = {'indexer': _indexer}

    if database_dir:
        layout_kwargs['database_path'] = database_dir

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
    logger = create_logger('sdcflow.wrangler', level=10 if pargs.verbose else 40)
    estimators_record = {}
    for subject in subjects:
        estimators_record[subject] = find_estimators(
            layout=layout,
            subject=subject,
            fmapless=pargs.fmapless,
            logger=logger,
        )

    # pretty print results
    print(f"Estimation for <{str(bids_dir)}> complete. Found:")
    for subject, estimators in estimators_record.items():
        print(f"\tsub-{subject}")
        if not estimators:
            print("\t\tNo estimators found")
            continue
        for estimator in estimators:
            print(f"\t\t{estimator}")
            for fl in estimator.sources:
                fl_relpath = fl.path.relative_to(str(bids_dir / f'sub-{subject}'))
                pe_dir = fl.metadata.get("PhaseEncodingDirection")
                print(f"\t\t\t{pe_dir}\t{fl_relpath}")


if __name__ == "__main__":
    main()
