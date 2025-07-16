import os
import sys
import gzip
import numpy as np
import pandas as pd

import scipy.io
from scipy.io import mmread
from scipy import sparse

from utils import *
from external import run_bedtools_closest, run_bedtools_merge


def annotate_feature_segID(features: pd.DataFrame, segs: pd.DataFrame):
    """
    annotate feature with segment index (analogous to segID), if it (partial)-overlaps with segment
    """

    def locate_gene(gene, starts, ends):
        start, end = gene["START"], gene["END"]
        mask = (end > starts) & (start < ends)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return pd.NA
        return indices[0]

    for ch in segs["#CHR"].unique():
        segs_ch = segs[segs["#CHR"] == ch]
        starts = segs_ch["START"].to_numpy()
        ends = segs_ch["END"].to_numpy()
        features_ch = features.loc[features["#CHR"] == ch, :]
        if len(features_ch) == 0:
            continue
        seg_indices = features_ch.apply(
            func=lambda gene: locate_gene(gene, starts, ends), axis=1
        )
        seg_indices_isna = seg_indices.isna().to_numpy()
        features_ch = features_ch.loc[~seg_indices_isna, :]
        if len(features_ch) == 0:
            continue
        seg_indices = seg_indices[~seg_indices_isna]
        features.loc[features_ch.index, "segID"] = segs_ch.iloc[seg_indices].index
    return features


def format_features(
    seg_file: str,
    ann_file: str,
    cr_feature_file: str,
    out_gex_bed_file: str,
    out_atac_bed_file: str,
    has_gex: bool,
    has_atac: bool,
):
    """
    1. For gene feature, annotate with gene transcript region from gene annotation file
    2. For either modality, separately
        1. mark excluded if feature not reside in copy-number segment or doesn't have position information
    output
    for each modality, 0-bed format file
    3. #CHR, START, END, featureID, featureName, global-matrix-index
    """
    segs = pd.read_table(seg_file, sep="\t")
    segs.index = segs.index.astype(int)

    features = pd.read_csv(
        cr_feature_file,
        header=None,
        sep="\t",
        names=["feature_id", "feature_name", "modality", "#CHR", "START", "END"],
    )
    features = features.reset_index(drop=False).rename(columns={"index": "mat_index"})
    features.loc[:, "segID"] = pd.NA

    if has_gex:
        annotations = pd.read_csv(
            ann_file,
            header=None,
            sep="\t",
            comment="#",
            names=[
                "#CHR",
                "START",
                "END",
                "feature_id",
                "feature_num",
                "strand",
                "feature_type",
            ],
            dtype={"START": np.uint32, "END": np.uint32},
        )
        annotations["START"] = annotations["START"] - 1  # covnert to 0-bed
        annotations = annotations[["#CHR", "START", "END", "feature_id"]]
        genes = features.loc[
            features["modality"] == "Gene Expression",
            ["feature_id", "mat_index", "segID"],
        ]
        ann_genes = pd.merge(left=genes, right=annotations, on="feature_id", how="left")
        print(f"#total genes={len(ann_genes)}")

        ann_genes = annotate_feature_segID(ann_genes, segs)
        ann_genes = ann_genes.dropna(subset=["#CHR", "START", "END", "segID"]).copy(
            deep=True
        )
        ann_genes["START"] = ann_genes["START"].astype(np.int32)
        ann_genes["END"] = ann_genes["END"].astype(np.int32)
        ann_genes.to_csv(
            out_gex_bed_file,
            sep="\t",
            header=False,
            index=False,
            columns=["#CHR", "START", "END", "feature_id", "mat_index"],
        )
        print(f"#cn-related genes={len(ann_genes)}")

    if has_atac:
        peaks = features.loc[features["modality"] == "Peaks", :]
        ann_peaks = annotate_feature_segID(peaks, segs)
        print(f"#total peaks={len(ann_peaks)}")

        ann_peaks = ann_peaks.dropna(subset=["#CHR", "START", "END", "segID"]).copy(
            deep=True
        )
        ann_peaks["START"] = ann_peaks["START"].astype(np.int32)
        ann_peaks["END"] = ann_peaks["END"].astype(np.int32)
        ann_peaks.to_csv(
            out_atac_bed_file,
            sep="\t",
            header=False,
            index=False,
            columns=["#CHR", "START", "END", "feature_id", "mat_index"],
        )
        print(f"#cn-related peaks={len(ann_peaks)}")

    return


def union_snp_features(
    seg_bed_file: str,
    snp_vcf_file: str,
    feature_bed_file: str,
    out_ext_file: str,
    tmp_dir: str,
    max_dist=5,
):
    """
    Given a list of cn-related features and Het SNP vcf,
    1. assign SNPs to features based on closest distance via bedtools.
    2. extend feature's region to cover related SNPs with <max_dist> threshold
    3. exclude features that doesn't contain Het SNPs.
    Output an updated feature BED file.
    #CHR START END feature_id mat_index #SNP DP cluster_id
    """
    tmp_dist_file = os.path.join(tmp_dir, "snp_gene_distance.0pos.tsv")
    dist_df = run_bedtools_closest(
        seg_bed_file,
        snp_vcf_file,
        feature_bed_file,
        tmp_dist_file,
        tmp_dir,
        load_df=True,
    )
    print(f"#snp-feature links={len(dist_df)}")
    os.remove(tmp_dist_file)

    snp_df = read_VCF_cellsnp_err_header(snp_vcf_file)
    snp_df.loc[:, "POS"] -= 1
    dist_df = pd.merge(left=dist_df, right=snp_df, on=["#CHR", "POS"], how="left")
    nsnps_old = dist_df[["#CHR", "POS"]].drop_duplicates().shape[0]

    dist_df = dist_df.loc[dist_df["dist"].abs() <= max_dist, :]
    nsnps_new = dist_df[["#CHR", "POS"]].drop_duplicates().shape[0]
    nsnps_ratio = round(100 * nsnps_new / nsnps_old, 3)
    print(f"#Het-covered-SNPs={nsnps_new}/{nsnps_old}, {nsnps_ratio}%")
    assert len(dist_df) > 0, "no Het SNPs being covered by any features, invalid data"

    feature_grps = dist_df.groupby(by=["feature_id"], sort=False, as_index=True)
    feature_df = feature_grps.agg(
        **{
            "#CHR": ("#CHR", "first"),
            "START": ("POS", "min"),
            "END": ("POS", "max"),
            "mat_index": ("mat_index", "first"),
        }
    ).reset_index()
    feature_df.loc[:, "END"] += 1
    feature_df.loc[:, "#SNP"] = feature_grps.size().reset_index(drop=True)

    tmp_ext_file = os.path.join(tmp_dir, "tmp_features.ext.bed")
    feature_df.to_csv(
        tmp_ext_file,
        sep="\t",
        header=False,
        index=False,
        columns=["#CHR", "START", "END", "feature_id", "mat_index"],
    )

    print("local clustering over overlapping feature")
    run_bedtools_merge(tmp_ext_file, out_ext_file, tmp_dir, 0)
    os.remove(tmp_ext_file)

    return

def format_feature_matrix(
    bin_ids_file: str,
    barcode_file: str,
    cr_barcode_file: str,
    cr_mtx_file: str,
    out_mat_file: str
):
    # load barcodes
    barcodes = read_barcodes(barcode_file)
    raw_barcodes = pd.read_csv(cr_barcode_file, header=None)[0].tolist()
    barcode_idx = [raw_barcodes.index(b) for b in barcodes]

    bin_ids = pd.read_table(bin_ids_file, sep="\t")
    bin_mat = np.zeros((len(bin_ids), len(barcodes)), dtype=np.uint32)
    mat_indicies = bin_ids.apply(func=lambda r: list(map(int, r["mat_index"].split(","))), axis=1)

    # load matrix
    feature_matrix = mmread(cr_mtx_file).tocsr()
    feature_matrix = feature_matrix[:, barcode_idx]

    for i in range(len(mat_indicies)):
        mat_index = mat_indicies[i]
        bin_mat[i, :] = np.sum(feature_matrix[mat_index, :], axis=0)

    sparse.save_npz(out_mat_file, csr_matrix(bin_mat))
    return
