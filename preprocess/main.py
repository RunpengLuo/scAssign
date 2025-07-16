import os
import sys

import numpy as np
import pandas as pd

from parsing import *
from utils import *
from phase_snps import phase_snps
from haplotag_vcf import haplotag_VCF
from format_cn import format_cn_profile
from binning import binning_features
from aggregation import aggregate_counts
from format_features import format_features, union_snp_features, format_feature_matrix

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

    # used to annotate gene location
    # since cellranger features only record TSS site for genes
    gene_ann_file = args["gene_bed"]

    laplace_alpha = args["laplace"]
    exclude_baf_eps = args["baf_eps"]
    exclude_baf_tol = args["baf_tol"]
    exclude_seg_len = args["seg_len"]
    min_gex_count = args["min_gex_count"]
    min_atac_count = args["min_atac_count"]

    has_gex = bvcf_gex != None
    has_atac = bvcf_atac != None
    assert has_gex or has_atac

    out_dir = args["outdir"]
    tmp_dir = os.path.join(out_dir, "tmp")
    seg_dir = os.path.join(out_dir, "seg")
    bbc_dir = os.path.join(out_dir, "bbc")
    fet_dir = os.path.join(out_dir, "feature")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(bbc_dir, exist_ok=True)
    os.makedirs(fet_dir, exist_ok=True)

    tag_vcf = True

    ##################################################
    print(f"extract copy-number profile and block/segment information")
    proc_seg_file = os.path.join(seg_dir, "Position.seg.tsv")
    proc_seg_bed_file = os.path.join(seg_dir, "Position.seg.bed")
    proc_seg_acopy_file = os.path.join(seg_dir, "Acopy.seg.tsv")
    proc_seg_bcopy_file = os.path.join(seg_dir, "Bcopy.seg.tsv")
    proc_seg_cbaf_file = os.path.join(seg_dir, "BAF.seg.tsv")
    # exclude_cns = ["1|1;2|2;2|1;4|2", "1|1;2|2;5|2;2|2"] #HT941
    # exclude_cns = ["1|1;1|1;1|0"]
    exclude_cns = []
    format_cn_profile(
        seg_ucn,
        bbc_ucn,
        proc_seg_file,
        proc_seg_bed_file,
        proc_seg_acopy_file,
        proc_seg_bcopy_file,
        proc_seg_cbaf_file,
        exclude_baf_eps,
        exclude_baf_tol,
        exclude_seg_len,
        laplace_alpha,
        exclude_cns
    )

    ##################################################
    print(f"load cell barcodes")
    barcodes = read_barcodes(barcode_file)
    pd.DataFrame({"BARCODE": barcodes}).to_csv(
        os.path.join(out_dir, "Barcodes.tsv"), sep="\t", index=False, header=True
    )
    ncells = len(barcodes)
    print(f"#barcodes={ncells}")

    ##################################################
    print("phase SNPs per copy-number segment")
    phase_mode = "hmm-b"
    phase_file = os.path.join(out_dir, f"phased_snps.tsv.gz")
    phase_snps(
        proc_seg_file,
        vcf_file,
        hair_file,
        normal_1bed,
        tumor_1bed,
        sample,
        phase_mode,
        phase_file,
    )

    ##################################################
    print("tag VCF files with phasing information")
    tagged_vcf_file = os.path.join(out_dir, "phased.vcf.gz")
    if not os.path.exists(tagged_vcf_file):
        haplotag_VCF(proc_seg_file, phase_file, vcf_file, tagged_vcf_file, tmp_dir)

    ##################################################
    print("extract copy-number related features")
    proc_gene_bed_file = os.path.join(fet_dir, "genes.raw.bed")
    proc_atac_bed_file = os.path.join(fet_dir, "peaks.raw.bed")
    format_features(
        proc_seg_file,
        gene_ann_file,
        cr_feature_file,
        proc_gene_bed_file,
        proc_atac_bed_file,
        has_gex,
        has_atac)

    ##################################################
    if has_gex:
        print(f"perform scRNA-seq binning on segments")
        proc_gene_ext_bed_file = os.path.join(fet_dir, "genes.ext.bed")
        union_snp_features(
            proc_seg_bed_file,
            bvcf_gex,
            proc_gene_bed_file,
            proc_gene_ext_bed_file,
            tmp_dir,
            max_dist=5
            )

        proc_gex_id_file = os.path.join(bbc_dir, "GEX_ids.bbc.tsv")
        proc_gex_bin_file = os.path.join(bbc_dir, "GEX_Position.bbc.tsv")
        proc_gex_bin_acopy_file = os.path.join(bbc_dir, "GEX_Acopy.bbc.tsv")
        proc_gex_bin_bcopy_file = os.path.join(bbc_dir, "GEX_Bcopy.bbc.tsv")
        proc_gex_bin_cbaf_file = os.path.join(bbc_dir, "GEX_BAF.bbc.tsv")
        if not os.path.exists(proc_gex_id_file):
            binning_features(
                proc_seg_file,
                proc_seg_acopy_file,
                proc_seg_bcopy_file,
                proc_seg_cbaf_file,
                bvcf_gex,
                proc_gene_ext_bed_file,
                proc_gex_id_file,
                proc_gex_bin_file,
                proc_gex_bin_acopy_file,
                proc_gex_bin_bcopy_file,
                proc_gex_bin_cbaf_file,
                min_dp_count=min_gex_count
            )
        
        print(f"aggregate total counts for scRNA-seq")
        proc_gex_mat_file = os.path.join(bbc_dir, "GEX_total.bbc.npz")
        if not os.path.exists(proc_gex_mat_file):
            format_feature_matrix(
                proc_gex_id_file,
                barcode_file,
                cr_barcode_file,
                cr_mtx_file,
                proc_gex_mat_file
            )
        
        print("aggregate SNP counts for scRNA-seq")
        gex_bin_Aallele_file = os.path.join(bbc_dir, "GEX_Aallele.bbc.npz")
        gex_bin_Ballele_file = os.path.join(bbc_dir, "GEX_Ballele.bbc.npz")
        gex_bin_Tallele_file = os.path.join(bbc_dir, "GEX_Tallele.bbc.npz")
        gex_bin_nSNP_file = os.path.join(bbc_dir, "GEX_nsnps.bbc.npz")
        if not os.path.exists(gex_bin_nSNP_file):
            aggregate_counts(
                proc_gex_bin_file,
                barcode_file,
                bvcf_gex,
                dp_mtx_gex,
                ad_mtx_gex,
                phase_file,
                "GEX",
                gex_bin_Aallele_file,
                gex_bin_Ballele_file,
                gex_bin_Tallele_file,
                gex_bin_nSNP_file,
                "PHASE",
            )

        gex_seg_Aallele_file = os.path.join(seg_dir, "GEX_Aallele.seg.npz")
        gex_seg_Ballele_file = os.path.join(seg_dir, "GEX_Ballele.seg.npz")
        gex_seg_Tallele_file = os.path.join(seg_dir, "GEX_Tallele.seg.npz")
        gex_seg_nSNP_file = os.path.join(seg_dir, "GEX_nsnps.seg.npz")
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

    if has_atac:
        print(f"perform scATAC-seq binning on segments")
        proc_atac_ext_bed_file = os.path.join(fet_dir, "peaks.ext.bed")
        union_snp_features(
            proc_seg_bed_file,
            bvcf_atac,
            proc_atac_bed_file,
            proc_atac_ext_bed_file,
            tmp_dir,
            max_dist=2000
            )

        proc_atac_id_file = os.path.join(bbc_dir, "ATAC_ids.bbc.tsv")
        proc_atac_bin_file = os.path.join(bbc_dir, "ATAC_Position.bbc.tsv")
        proc_atac_bin_acopy_file = os.path.join(bbc_dir, "ATAC_Acopy.bbc.tsv")
        proc_atac_bin_bcopy_file = os.path.join(bbc_dir, "ATAC_Bcopy.bbc.tsv")
        proc_atac_bin_cbaf_file = os.path.join(bbc_dir, "ATAC_BAF.bbc.tsv")
        if not os.path.exists(proc_atac_id_file):
            binning_features(
                proc_seg_file,
                proc_seg_acopy_file,
                proc_seg_bcopy_file,
                proc_seg_cbaf_file,
                bvcf_atac,
                proc_atac_ext_bed_file,
                proc_atac_id_file,
                proc_atac_bin_file,
                proc_atac_bin_acopy_file,
                proc_atac_bin_bcopy_file,
                proc_atac_bin_cbaf_file,
                min_dp_count=min_atac_count
            )
        
        print(f"aggregate total counts for scATAC-seq")
        proc_atac_mat_file = os.path.join(bbc_dir, "ATAC_total.bbc.npz")
        if not os.path.exists(proc_atac_mat_file):
            format_feature_matrix(
                proc_atac_id_file,
                barcode_file,
                cr_barcode_file,
                cr_mtx_file,
                proc_atac_mat_file
            )
        
        print("aggregate SNP counts for scATAC-seq")
        atac_bin_Aallele_file = os.path.join(bbc_dir, "ATAC_Aallele.bbc.npz")
        atac_bin_Ballele_file = os.path.join(bbc_dir, "ATAC_Ballele.bbc.npz")
        atac_bin_Tallele_file = os.path.join(bbc_dir, "ATAC_Tallele.bbc.npz")
        atac_bin_nSNP_file = os.path.join(bbc_dir, "ATAC_nsnps.bbc.npz")
        if not os.path.exists(atac_bin_nSNP_file):
            aggregate_counts(
                proc_atac_bin_file,
                barcode_file,
                bvcf_atac,
                dp_mtx_atac,
                ad_mtx_atac,
                phase_file,
                "ATAC",
                atac_bin_Aallele_file,
                atac_bin_Ballele_file,
                atac_bin_Tallele_file,
                atac_bin_nSNP_file,
                "PHASE",
            )

        atac_seg_Aallele_file = os.path.join(seg_dir, "ATAC_Aallele.seg.npz")
        atac_seg_Ballele_file = os.path.join(seg_dir, "ATAC_Ballele.seg.npz")
        atac_seg_Tallele_file = os.path.join(seg_dir, "ATAC_Tallele.seg.npz")
        atac_seg_nSNP_file = os.path.join(seg_dir, "ATAC_nsnps.seg.npz")
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
