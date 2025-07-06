import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse

def read_barcodes(prep_dir: str):
    return pd.read_csv(os.path.join(prep_dir, "Barcodes.tsv"), sep="\t")

def read_copy_number_data(prep_dir: str, bin_level: str):
    assert bin_level in ["seg", "bbc"]
    assert os.path.isdir(prep_dir)
    bin_file = os.path.join(prep_dir, f"Position.{bin_level}.tsv")
    acopy_file = os.path.join(prep_dir, f"Acopy.{bin_level}.tsv")
    bcopy_file = os.path.join(prep_dir, f"Bcopy.{bin_level}.tsv")
    baf_file = os.path.join(prep_dir, f"BAF.{bin_level}.tsv")
    bins = pd.read_csv(bin_file, sep="\t")
    acopies = pd.read_table(acopy_file, sep="\t").to_numpy()
    bcopies = pd.read_table(bcopy_file, sep="\t").to_numpy()
    ccopies = acopies + bcopies
    bafs_df = pd.read_table(baf_file, sep="\t")
    clones = bafs_df.columns.tolist()
    bafs = bafs_df.to_numpy()
    return clones, bins, acopies, bcopies, ccopies, bafs


def read_single_cell_data(prep_dir: str, modality: str, bin_level: str):
    assert modality in ["GEX", "ATAC"]
    assert bin_level in ["seg", "bbc"]
    assert os.path.isdir(prep_dir)

    Aallele_file = os.path.join(prep_dir, f"{modality}_Aallele.{bin_level}.npz")
    Ballele_file = os.path.join(prep_dir, f"{modality}_Ballele.{bin_level}.npz")
    Tallele_file = os.path.join(prep_dir, f"{modality}_Tallele.{bin_level}.npz")
    nSNP_file = os.path.join(prep_dir, f"{modality}_nsnps.{bin_level}.npz")

    counts_Aallele: np.ndarray = sparse.load_npz(Aallele_file).toarray()
    counts_Ballele: np.ndarray = sparse.load_npz(Ballele_file).toarray()
    counts_Tallele: np.ndarray = sparse.load_npz(Tallele_file).toarray()
    counts_Nsnp: np.ndarray = sparse.load_npz(nSNP_file).toarray()
    return counts_Aallele, counts_Ballele, counts_Tallele, counts_Nsnp


def read_single_cell_features(prep_dir: str):
    assert os.path.isdir(prep_dir)
    gex_mat_file = os.path.join(prep_dir, "gene_expression.npz")
    atac_mat_file = os.path.join(prep_dir, "peak_signals.npz")
    feature_file = os.path.join(prep_dir, "features.tsv.gz")

    features = pd.read_csv(feature_file, sep="\t")
    
    gex_mat = None
    gene_ids = None
    gene_segIDs = None
    if os.path.exists(gex_mat_file):
        gex_mat: np.ndarray = sparse.load_npz(gex_mat_file).toarray()
        gene_ids = features.loc[features["feature_type"] == "Gene Expression", "feature_id"].tolist()
        gene_segIDs = features.loc[features["feature_type"] == "Gene Expression", "segID"].tolist()
        assert gex_mat.shape[0] == len(gene_ids) and len(gene_ids) > 0

    peak_mat = None
    peak_ids = None
    peak_segIDs = None
    if os.path.exists(atac_mat_file):
        peak_mat: np.ndarray = sparse.load_npz(atac_mat_file).toarray()
        peak_ids = features.loc[features["feature_type"] == "Peaks", "feature_id"].tolist()
        peak_segIDs = features.loc[features["feature_type"] == "Peaks", "segID"].tolist()
        assert peak_mat.shape[0] == len(peak_ids) and len(peak_ids) > 0

    return features, gex_mat, gene_ids, gene_segIDs, peak_mat, peak_ids, peak_segIDs

def encode_feature_cn(features: pd.DataFrame, ccopies: np.ndarray, clones: list):
    for i in range(len(clones)):
        features[f"cn_{clones[i]}"] = ccopies[features["segID"].to_numpy(), i]
    return features

def get_feature_cn_matrix(features: pd.DataFrame, clones: list, modality: str):
    assert modality in ["GEX", "ATAC"]
    columns = [f"cn_{clone}" for clone in clones]
    feat_cn_mat = None
    if modality == "GEX":
        feat_cn_mat = features.loc[features["feature_type"] == "Gene Expression", columns].to_numpy()
    else:
        feat_cn_mat = features.loc[features["feature_type"] == "Peaks", columns].to_numpy()
    return feat_cn_mat
