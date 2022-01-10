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
    return parser


def gen_layout(bids_dir):
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
            re.compile(r"sub-[a-zA-Z0-9]+(/ses-[a-zA-Z0-9]+)?/(beh|dwi|eeg|ieeg|meg|perf)"),
        ),
    )
    layout = BIDSLayout(bids_dir, indexer=_indexer)
    return layout


def main(argv=None):
    """
    Parse a BIDS directory and print subject estimators
    """
    from niworkflows.utils.bids import collect_participants
    from sdcflows.utils.wrangler import find_estimators

    pargs = _parser().parse_args(argv)

    bids_dir = pargs.bids_dir.resolve(strict=True)
    layout = gen_layout(bids_dir)
    subjects = collect_participants(layout, pargs.subjects)
    estimators_record = {}
    for subject in subjects:
        estimators_record[subject] = find_estimators(
            layout=layout, subject=subject, fmapless=pargs.fmapless
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


if __name__ == "__main__":
    main()
