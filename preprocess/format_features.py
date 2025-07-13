import os
import sys
import numpy as np
import pandas as pd

from scipy.io import mmread
import scipy.io
import gzip
from scipy import sparse

from utils import *


def annotate_feature_segID(features: pd.DataFrame, segs: pd.DataFrame):
    """
    annotate feature with segment index (analogous to segID), if it (partial)-overlaps with segment
    """
    features.loc[:, "segID"] = pd.NA

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
        if len(seg_indices) == 0:
            continue
        features_ch = features_ch.loc[seg_indices != pd.NA, :]
        seg_indices = seg_indices[seg_indices != pd.NA]
        features.loc[features_ch.index, "segID"] = segs_ch.iloc[seg_indices].index
    return features


def format_feature_matrix(
    seg_file: str,
    barcode_file: str,
    cr_barcode_file: str,
    cr_feature_file: str,
    cr_mtx_file: str,
    out_gex_mat_file: str,
    out_atac_mat_file: str,
    out_feature_file: str,
    out_feature_bed_file: str,
    has_gex: bool,
    has_atac: bool,
):
    segs = pd.read_table(seg_file, sep="\t")
    segs.index = segs.index.astype(int)

    features = pd.read_csv(
        cr_feature_file,
        header=None,
        sep="\t",
        names=["feature_id", "feature_name", "feature_type", "#CHR", "START", "END"],
    )
    features = features.dropna(subset=["#CHR", "START", "END"])

    # drop features that not belong to copy number segments
    features = annotate_feature_segID(features, segs)
    features = features.dropna(subset=["segID"])
    if len(features) == 0:
        print(f"error, no feature is found for given segments")
        return 1

    # load barcodes
    barcodes = read_barcodes(barcode_file)
    raw_barcodes = pd.read_csv(cr_barcode_file, header=None)[0].tolist()

    # load matrix
    feature_matrix_raw = mmread(cr_mtx_file).tocsr()[features.index]
    feature_matrix: pd.DataFrame = pd.DataFrame.sparse.from_spmatrix(
        feature_matrix_raw, columns=raw_barcodes
    )
    feature_matrix = feature_matrix[barcodes]

    features = features.reset_index(drop=True)

    if has_gex:
        gex_matrix = feature_matrix.loc[
            features["feature_type"] == "Gene Expression", :
        ]
        if len(gex_matrix) == 0:
            print("no gene in copy-number segments")
        else:
            print(f"#genes={len(gex_matrix)}")
            sparse.save_npz(out_gex_mat_file, csr_matrix(gex_matrix.to_numpy()))

    if has_atac:
        peak_matrix = feature_matrix.loc[features["feature_type"] == "Peaks", :]
        if len(peak_matrix) == 0:
            print("no peaks in copy-number segments")
        else:
            print(f"#peaks={len(peak_matrix)}")
            sparse.save_npz(out_atac_mat_file, csr_matrix(peak_matrix.to_numpy()))

    features.to_csv(out_feature_file, header=True, index=False, sep="\t")
    features.to_csv(
        out_feature_bed_file,
        header=False,
        index=False,
        sep="\t",
        columns=["#CHR", "START", "END", "feature_id"],
    )
    return
