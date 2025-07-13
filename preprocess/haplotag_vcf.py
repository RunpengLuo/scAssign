import os
import sys
import gzip

import numpy as np
import pandas as pd
import subprocess
from cyvcf2 import VCF, Writer
from intervaltree import IntervalTree


# TODO improve the speed
def haplotag_VCF(
    seg_file: str, phase_file: str, vcf_file: str, out_file: str, tmp_dir: str
):
    """
    use phased SNPs to tag SNPs from cellsnp-lite VCF file with PS and GT
    """
    tmp_region_bed = os.path.join(tmp_dir, "region.tsv.gz")
    tmp_fil_vcf = os.path.join(tmp_dir, "filtered.vcf.gz")

    phased_df = pd.read_table(phase_file, sep="\t")
    b_phase = np.round(phased_df["PHASE"]).astype(np.int8)
    a_phase = (1 - b_phase).astype(np.int8)
    phased_df["GT-A"] = a_phase
    phased_df["GT-B"] = b_phase

    phased_df.to_csv(
        tmp_region_bed, sep="\t", header=False, index=False, columns=["#CHR", "POS"]
    )

    if not os.path.exists(f"{vcf_file}.csi"):
        subprocess.run(["bcftools", "index", "-f", vcf_file], check=True)

    # filter unphased SNPs
    subprocess.run(
        ["bcftools", "view", "-R", tmp_region_bed, vcf_file, "-Oz", "-o", tmp_fil_vcf],
        check=True,
    )
    subprocess.run(["bcftools", "index", "-f", tmp_fil_vcf], check=True)

    # update GT tag and add PS tag
    phased_df = phased_df.set_index(keys=["#CHR", "POS"])
    vcf = VCF(tmp_fil_vcf)
    vcf.add_format_to_header(
        {"ID": "PS", "Number": "1", "Type": "Integer", "Description": "Phase Set"}
    )

    writer = Writer(out_file, vcf)  # Writes headers and meta-information

    print(f"start writing to {out_file}")
    # Replace GT fields for matching sites
    for record in vcf:
        key = (record.CHROM, record.POS)
        row = phased_df.loc[key]
        gt = [row["GT-A"], row["GT-B"], True]
        ps = np.array([row["PS"]], dtype=np.int32)
        record.genotypes = [gt]
        record.set_format("PS", ps)
        writer.write_record(record)
    vcf.close()
    writer.close()

    os.remove(tmp_region_bed)
    os.remove(tmp_fil_vcf)
    os.remove(f"{tmp_fil_vcf}.csi")
    return
