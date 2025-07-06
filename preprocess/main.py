import os
import sys

import numpy as np
import pandas as pd

from parsing import *
from utils import *
from phase_snps import phase_snps
from format_cn import format_cn_profile
from aggregation import aggregate_counts
from format_features import format_feature_matrix

if __name__ == "__main__":
    args = parse_arguments()
    sample = args["sample"]
    seg_ucn = args["seg"]
    bbc_ucn = args["bbc"]
    normal_1bed = args["nbed"]
    tumor_1bed = args["tbed"]

    barcode_file = args["barcodes"]
    vcf_file = args["vcf"]
    hair_file = args["hairs"]

    bvcf_atac = args["atac_vcf"]
    dp_mtx_atac = args["atac_dp"]
    ad_mtx_atac = args["atac_ad"]
    bvcf_gex = args["gex_vcf"]
    dp_mtx_gex = args["gex_dp"]
    ad_mtx_gex = args["gex_ad"]

    cr_barcode_file = args["raw_barcodes"]
    cr_feature_file = args["raw_features"]
    cr_mtx_file = args["feature_mtx"]

    laplace = args["laplace"]

    out_dir = args["outdir"]
    os.makedirs(out_dir, exist_ok=True)

    ##################################################
    print(f"extract copy-number profile and block/segment information")
    proc_seg_file = os.path.join(out_dir, "Position.seg.tsv")
    proc_seg_acopy_file = os.path.join(out_dir, "Acopy.seg.tsv")
    proc_seg_bcopy_file = os.path.join(out_dir, "Bcopy.seg.tsv")
    proc_seg_cbaf_file = os.path.join(out_dir, "BAF.seg.tsv")
    proc_bbc_file = os.path.join(out_dir, "Position.bbc.tsv")
    proc_bbc_acopy_file = os.path.join(out_dir, "Acopy.bbc.tsv")
    proc_bbc_bcopy_file = os.path.join(out_dir, "Bcopy.bbc.tsv")
    proc_bbc_cbaf_file = os.path.join(out_dir, "BAF.bbc.tsv")
    format_cn_profile(
        seg_ucn,
        bbc_ucn,
        proc_seg_file,
        proc_seg_acopy_file,
        proc_seg_bcopy_file,
        proc_seg_cbaf_file,
        proc_bbc_file,
        proc_bbc_acopy_file,
        proc_bbc_bcopy_file,
        proc_bbc_cbaf_file,
        laplace_alpha=laplace,
    )

    ##################################################
    print(f"load cell barcodes")
    barcodes = read_barcodes(barcode_file)
    pd.DataFrame({"BARCODE": barcodes}).to_csv(
        os.path.join(out_dir, "Barcodes.tsv"), sep="\t", 
        index=False, header=True)
    ncells = len(barcodes)
    print(f"#barcodes={ncells}")

    ##################################################
    phase_mode = "hmm-b"
    phase_file = os.path.join(out_dir, f"phased_snps.tsv.gz")
    # phase_file = os.path.join(out_dir, f"phased_snps.{phase_mode}.tsv")
    phase_snps(
        proc_seg_file,
        proc_bbc_file,
        vcf_file,
        hair_file,
        normal_1bed,
        tumor_1bed,
        sample,
        phase_mode,
        phase_file,
    )

    ##################################################
    has_atac = False
    if bvcf_atac != None and dp_mtx_atac != None and ad_mtx_atac != None:
        print("aggregate SNP counts for scATAC-seq")
        has_atac = True
        atac_bbc_Aallele_file = os.path.join(out_dir, "ATAC_Aallele.bbc.npz")
        atac_bbc_Ballele_file = os.path.join(out_dir, "ATAC_Ballele.bbc.npz")
        atac_bbc_Tallele_file = os.path.join(out_dir, "ATAC_Tallele.bbc.npz")
        atac_bbc_nSNP_file = os.path.join(out_dir, "ATAC_nsnps.bbc.npz")
        if not os.path.exists(atac_bbc_nSNP_file):
            aggregate_counts(
                proc_seg_file,
                barcode_file,
                bvcf_atac,
                dp_mtx_atac,
                ad_mtx_atac,
                phase_file,
                "ATAC",
                atac_bbc_Aallele_file,
                atac_bbc_Ballele_file,
                atac_bbc_Tallele_file,
                atac_bbc_nSNP_file,
                "PHASE",
            )

        atac_seg_Aallele_file = os.path.join(out_dir, "ATAC_Aallele.seg.npz")
        atac_seg_Ballele_file = os.path.join(out_dir, "ATAC_Ballele.seg.npz")
        atac_seg_Tallele_file = os.path.join(out_dir, "ATAC_Tallele.seg.npz")
        atac_seg_nSNP_file = os.path.join(out_dir, "ATAC_nsnps.seg.npz")
        if not os.path.exists(atac_seg_nSNP_file):
            aggregate_counts(
                proc_seg_file,
                barcode_file,
                bvcf_atac,
                dp_mtx_atac,
                ad_mtx_atac,
                phase_file,
                "ATAC",
                atac_seg_Aallele_file,
                atac_seg_Ballele_file,
                atac_seg_Tallele_file,
                atac_seg_nSNP_file,
                "PHASE",
            )

    ##################################################
    has_gex = False
    if bvcf_gex != None and dp_mtx_gex != None and ad_mtx_gex != None:
        print("aggregate SNP counts for scRNA-seq")
        has_gex = True
        gex_bbc_Aallele_file = os.path.join(out_dir, "GEX_Aallele.bbc.npz")
        gex_bbc_Ballele_file = os.path.join(out_dir, "GEX_Ballele.bbc.npz")
        gex_bbc_Tallele_file = os.path.join(out_dir, "GEX_Tallele.bbc.npz")
        gex_bbc_nSNP_file = os.path.join(out_dir, "GEX_nsnps.bbc.npz")
        if not os.path.exists(gex_bbc_nSNP_file):
            aggregate_counts(
                proc_seg_file,
                barcode_file,
                bvcf_gex,
                dp_mtx_gex,
                ad_mtx_gex,
                phase_file,
                "GEX",
                gex_bbc_Aallele_file,
                gex_bbc_Ballele_file,
                gex_bbc_Tallele_file,
                gex_bbc_nSNP_file,
                "PHASE",
            )

        gex_seg_Aallele_file = os.path.join(out_dir, "GEX_Aallele.seg.npz")
        gex_seg_Ballele_file = os.path.join(out_dir, "GEX_Ballele.seg.npz")
        gex_seg_Tallele_file = os.path.join(out_dir, "GEX_Tallele.seg.npz")
        gex_seg_nSNP_file = os.path.join(out_dir, "GEX_nsnps.seg.npz")
        if not os.path.exists(gex_seg_nSNP_file):
            aggregate_counts(
                proc_seg_file,
                barcode_file,
                bvcf_gex,
                dp_mtx_gex,
                ad_mtx_gex,
                phase_file,
                "GEX",
                gex_seg_Aallele_file,
                gex_seg_Ballele_file,
                gex_seg_Tallele_file,
                gex_seg_nSNP_file,
                "PHASE",
            )

    ##################################################
    print("process feature matrix")
    gex_mat_file = os.path.join(out_dir, "gene_expression.npz")
    atac_mat_file = os.path.join(out_dir, "peak_signals.npz")
    feature_file = os.path.join(out_dir, "features.tsv.gz")
    
    format_feature_matrix(proc_seg_file, barcode_file, cr_barcode_file,
                          cr_feature_file, cr_mtx_file, gex_mat_file, 
                          atac_mat_file, feature_file, has_gex, has_atac)

