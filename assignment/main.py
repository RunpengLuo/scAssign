import os
import sys

import numpy as np
import pandas as pd

from sc_data import SC_Data
from sc_models import sc_model_GEX, sc_model_multiome
from sc_runner import *
from sc_visual import *

from allele_model import allele_model_multiome

"""
ATAC_Aallele.seg.npz    ATAC_nsnps.seg.npz      BAF.bbc.tsv             Bcopy.seg.tsv           GEX_Tallele.seg.npz     Position.seg.tsv        peak_signals.npz
ATAC_Ballele.seg.npz    Acopy.bbc.tsv           BAF.seg.tsv             GEX_Aallele.seg.npz     GEX_nsnps.seg.npz       features.tsv.gz         phased_snps.tsv.gz
ATAC_Tallele.seg.npz    Acopy.seg.tsv           Bcopy.bbc.tsv           GEX_Ballele.seg.npz     Position.bbc.tsv        gene_expression.npz
"""

_, sample, prep_dir, out_dir, mode = sys.argv
os.makedirs(out_dir, exist_ok=True)

seg_data = SC_Data(prep_dir, mode, "seg")
seg_data.transform_data()

# model = {"GEX": sc_model_GEX, "ATAC": None, "BOTH": sc_model_multiome}[mode]
model = allele_model_multiome
validate_model(seg_data, model=model, out_dir=out_dir)

rep_dir = os.path.join(out_dir, f"rep_1")
os.makedirs(rep_dir, exist_ok=True)
assign_file = os.path.join(rep_dir, "expose_pi.tsv")
if not os.path.exists(assign_file):
    run_model(seg_data, model=model, curr_repeat=1, out_dir=rep_dir)
    assert os.path.exists(assign_file)

final_dir = rep_dir
clones = seg_data.clones
# compute final decision with MAP estimate
def decision(r):
    probs = [r[clone] for clone in clones]
    (best_clone, best_prob) = max(zip(clones, probs), key=lambda t: t[1])
    if best_prob > 0.70:
        return best_clone
    else:
        return "unassigned"

assign_df = pd.read_table(os.path.join(final_dir, "expose_pi.tsv"), sep="\t")
assign_df = assign_df.rename(columns={str(i): clone for i, clone in enumerate(clones)})
assign_df["BARCODE"] = seg_data.barcodes.BARCODE
assign_df["Decision"] = assign_df.apply(func=lambda r: decision(r), axis=1)
assign_df = assign_df[["BARCODE"] + clones + ["Decision"]]
assign_df.to_csv(os.path.join(out_dir, "final_assignment.tsv"), sep="\t", header=True, index=False)

for dec_name in assign_df["Decision"].unique():
    assigned = assign_df.loc[assign_df["Decision"] == dec_name, :]
    print(f"#{dec_name}={len(assigned)}")


out_baf = os.path.join(out_dir, "baf_heatmap.subclonal.pdf")
out_snp = os.path.join(out_dir, "snp_heatmap.subclonal.pdf")
plot_heatmap(sample, seg_data, assign_df, out_baf, out_snp, ignore_clonal=True, min_clen_clustering=15e6)

out_baf = os.path.join(out_dir, "baf_heatmap.pdf")
out_snp = os.path.join(out_dir, "snp_heatmap.pdf")
plot_heatmap(sample, seg_data, assign_df, out_baf, out_snp)

out_1d = os.path.join(out_dir, "baf_violin.pdf")
plot_violin(sample, seg_data, assign_df, out_1d)
# plot_feature_distribution(seg_data.gene_mat, "gene_expression", out_dir)
# plot_feature_distribution(seg_data.peak_mat, "peak_signal", out_dir)

