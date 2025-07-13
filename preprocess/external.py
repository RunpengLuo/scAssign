import os
import sys
import subprocess

import numpy as np
import pandas as pd


def filter_vcf2bed(in_vcf_file: str, seg_bed_file: str, out_file: str):
    """
    only keep variants resides in seg_bed_file
    convert to BED format
    """
    if not os.path.exists(f"{in_vcf_file}.tbi"):
        subprocess.run(["tabix", in_vcf_file], check=True)

    view_proc = subprocess.run(
        ["bcftools", "view", "-R", seg_bed_file, in_vcf_file],
        stdout=subprocess.PIPE,
        check=True,
    )
    subprocess.run(
        ["bcftools", "query", "-f", "%CHROM\t%POS0\t%POS\n"],
        input=view_proc.stdout,
        stdout=open(out_file, "w"),
        check=True,
    )
    return


def sort_bed(in_bed_file: str, out_file):
    subprocess.run(
        ["sort", "-k1,1", "-k2,2n", in_bed_file],
        stdout=open(out_file, "w"),
        check=True,
    )
    return


def run_bedtools_closest(
    seg_bed_file: str,
    in_vcf_file: str,
    in_bed_file: str,
    out_file: str,
    tmp_dir: str,
    gid_col=4,
    gname_col=7,
):
    print("run bedtools closest")
    tmp_bed_file = os.path.join(tmp_dir, "variants.filtered.bed")
    filter_vcf2bed(in_vcf_file, seg_bed_file, tmp_bed_file)

    tmp_sbed_file = os.path.join(tmp_dir, "variants.sorted.bed")
    sort_bed(tmp_bed_file, tmp_sbed_file)

    tmp_gbed_file = os.path.join(tmp_dir, "features.sorted.bed")
    sort_bed(in_bed_file, tmp_gbed_file)

    dist_proc = subprocess.run(
        [
            "bedtools",
            "closest",
            "-a",
            tmp_sbed_file,
            "-b",
            tmp_gbed_file,
            "-D",
            "ref",
            "-t",
            "all",
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    gid_col += 3
    gname_col += 3
    format_str = "{print $1, $2, " + f"${gid_col}, ${gname_col}, " + "$(NF)}"
    subprocess.run(
        ["awk", "-F\t", format_str, "OFS=\t"],
        input=dist_proc.stdout,
        stdout=open(out_file, "w"),
        check=True,
    )

    os.remove(tmp_bed_file)
    os.remove(tmp_sbed_file)
    os.remove(tmp_gbed_file)

    out_df = pd.read_table(
        out_file, sep="\t", names=["#CHR", "POS", "geneID", "geneType", "dist"]
    )
    return out_df


def run_bedtools_merge(
    seg_bed_file: str,
    in_vcf_file: str,
    out_file: str,
    tmp_dir: str,
    max_dist=500,
):
    """
    form clamp of SNPs to simulate pseudo-peak
    report #CHR\tSTART\tEND\t#SNP
    """
    print("run bedtools merge")
    tmp_bed_file = os.path.join(tmp_dir, "variants.filtered.bed")
    filter_vcf2bed(in_vcf_file, seg_bed_file, tmp_bed_file)

    tmp_sbed_file = os.path.join(tmp_dir, "variants.sorted.bed")
    sort_bed(tmp_bed_file, tmp_sbed_file)

    subprocess.run(
        [
            "bedtools",
            "merge",
            "-i",
            tmp_sbed_file,
            "-d",
            str(max_dist),
            "-c",
            str(1),
            "-o",
            "count",
        ],
        stdout=open(out_file, "w"),
        check=True,
    )
    os.remove(tmp_bed_file)

    out_df = pd.read_table(out_file, sep="\t", names=["#CHR", "START", "END", "#SNP"])
    return out_df


# def run_bedtools_merge(in_bed: str, out_bed: str, max_dist=0):
#     subprocess.run(
#         [
#             "bedtools",
#             "merge",
#             "-i",
#             in_bed,
#             "-d",
#             str(max_dist),
#             "-c",
#             str(1),
#             "-o",
#             "count",
#         ],
#         stdout=open(out_bed, "w"),
#         check=True,
#     )
