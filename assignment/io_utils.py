import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse

def read_barcodes(prep_dir: str):
    return pd.read_csv(os.path.join(prep_dir, "Barcodes.tsv"), sep="\t")

def read_copy_number_data(prep_dir: str, bin_level: str, modality=""):
    assert bin_level in ["seg", "bbc"]
    assert os.path.isdir(prep_dir)

    if bin_level == "seg":
        bin_file = os.path.join(prep_dir, f"{bin_level}/Position.{bin_level}.tsv")
        acopy_file = os.path.join(prep_dir, f"{bin_level}/Acopy.{bin_level}.tsv")
        bcopy_file = os.path.join(prep_dir, f"{bin_level}/Bcopy.{bin_level}.tsv")
        baf_file = os.path.join(prep_dir, f"{bin_level}/BAF.{bin_level}.tsv")
    else:
        bin_file = os.path.join(prep_dir, f"{bin_level}/{modality}_Position.{bin_level}.tsv")
        acopy_file = os.path.join(prep_dir, f"{bin_level}/{modality}_Acopy.{bin_level}.tsv")
        bcopy_file = os.path.join(prep_dir, f"{bin_level}/{modality}_Bcopy.{bin_level}.tsv")
        baf_file = os.path.join(prep_dir, f"{bin_level}/{modality}_BAF.{bin_level}.tsv")

    acopies = pd.read_table(acopy_file, sep="\t").to_numpy(dtype=np.int32)
    bcopies = pd.read_table(bcopy_file, sep="\t").to_numpy(dtype=np.int32)
    ccopies = acopies + bcopies
    bafs_df = pd.read_table(baf_file, sep="\t").astype(dtype=np.float32)
    clones = bafs_df.columns.tolist()
    bafs = bafs_df.to_numpy(dtype=np.float32)

    bins = pd.read_csv(bin_file, sep="\t")
    bins["cn-state"] = "unknown"
    for bid in range(len(bins)):
        states = []
        for cid in range(len(clones)):
            states.append(f"{acopies[bid, cid]}|{bcopies[bid, cid]}")
        bins.loc[bid, "cn-state"] = ";".join(states)
    return clones, bins, acopies, bcopies, ccopies, bafs


def read_single_cell_data(prep_dir: str, modality: str, bin_level: str):
    assert modality in ["GEX", "ATAC"]
    assert bin_level in ["seg", "bbc"]
    assert os.path.isdir(prep_dir)

    Aallele_file = os.path.join(prep_dir, f"{bin_level}/{modality}_Aallele.{bin_level}.npz")
    Ballele_file = os.path.join(prep_dir, f"{bin_level}/{modality}_Ballele.{bin_level}.npz")
    Tallele_file = os.path.join(prep_dir, f"{bin_level}/{modality}_Tallele.{bin_level}.npz")
    nSNP_file = os.path.join(prep_dir, f"{bin_level}/{modality}_nsnps.{bin_level}.npz")

    counts_Aallele: np.ndarray = sparse.load_npz(Aallele_file).toarray().astype(dtype=np.int32)
    counts_Ballele: np.ndarray = sparse.load_npz(Ballele_file).toarray().astype(dtype=np.int32)
    counts_Tallele: np.ndarray = sparse.load_npz(Tallele_file).toarray().astype(dtype=np.int32)
    counts_Nsnp: np.ndarray = sparse.load_npz(nSNP_file).toarray().astype(dtype=np.int32)
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
        gex_mat: np.ndarray = sparse.load_npz(gex_mat_file).toarray().astype(dtype=np.int32)
        gene_ids = features.loc[features["feature_type"] == "Gene Expression", "feature_id"].tolist()
        gene_segIDs = features.loc[features["feature_type"] == "Gene Expression", "segID"].tolist()
        assert gex_mat.shape[0] == len(gene_ids) and len(gene_ids) > 0

    peak_mat = None
    peak_ids = None
    peak_segIDs = None
    if os.path.exists(atac_mat_file):
        peak_mat: np.ndarray = sparse.load_npz(atac_mat_file).toarray().astype(dtype=np.int32)
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
        feat_cn_mat = features.loc[features["feature_type"] == "Gene Expression", columns].to_numpy(dtype=np.int32)
    else:
        feat_cn_mat = features.loc[features["feature_type"] == "Peaks", columns].to_numpy(dtype=np.int32)
    return feat_cn_mat

def read_verify_file(verify_file: str):
    def simp_type(t: str):
        return t if "tumor" in t.lower() else "normal"
    def simp_type2(t: str):
        return "tumor" if "tumor" in t.lower() else "normal"
    # cell_id cell_types final_type
    verify_df = pd.read_table(verify_file)
    verify_df = verify_df.rename(columns={"cell_id": "BARCODE"})
    if "met_subcluster" in verify_df.columns.tolist():
        print("use column met_subcluster as final_type")
        verify_df["final_type"] = verify_df["met_subcluster"]

    if "final_type" not in verify_df.columns.tolist():
        assert "cell_types" in verify_df.columns.tolist(), (
            "cell_types column does not exist"
        )
        print("use column cell_types as final_type")
        verify_df["final_type"] = verify_df["cell_types"]
    if "simp_type" not in verify_df.columns.tolist():
        verify_df["simp_type"] = verify_df.apply(
            func=lambda r: simp_type(r["final_type"]), axis=1
        )
    if "simp_type2" not in verify_df.columns.tolist():
        verify_df["simp_type2"] = verify_df.apply(
            func=lambda r: simp_type2(r["final_type"]), axis=1
        )
    return verify_df
