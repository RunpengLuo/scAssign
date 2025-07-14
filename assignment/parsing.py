import os
import argparse


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        prog="scAlign assignment",
        description="assigning single-cell data",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--sample",
        required=True,
        type=str,
        help="sample name, used for output prefix",
    )
    parser.add_argument(
        "--prep_dir",
        required=True,
        type=str,
        help="prep-ed directory",
    )
    parser.add_argument(
        "--modality",
        required=True,
        type=str,
        choices=["ATAC", "GEX", "BOTH"],
        default="BOTH",
        help="data modality",
    )
    parser.add_argument(
        "--mode",
        required=False,
        type=str,
        choices=["pure", "het"],
        default="pure",
        help="mode",
    )

    # TODO tumor proportion file input for het mode
    parser.add_argument(
        "--level",
        required=True,
        type=str,
        choices=["bbc", "seg"],
        default="bbc",
        help="bin level",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        required=True,
        type=str,
        help="output directory",
    )

    parser.add_argument(
        "--min_posterior",
        required=True,
        type=float,
        default=0.70,
        help="assignment min posterior threshold",
    )

    args = parser.parse_args()

    # sanity check
    return {
        "sample": args.sample,
        "prep_dir": args.prep_dir,
        "out_dir": args.outdir,
        "mode": args.mode,
        "modality": args.modality,
        "bin_level": args.level,
        "min_posterior": args.min_posterior
    }
