import os
import sys

import numpy as np
import pandas as pd

from parsing import parse_arguments

from sc_data import SC_Data
from sc_runner import *
from sc_model import *

"""
ATAC_Aallele.seg.npz    ATAC_nsnps.seg.npz      BAF.bbc.tsv             Bcopy.seg.tsv           GEX_Tallele.seg.npz     Position.seg.tsv        peak_signals.npz
ATAC_Ballele.seg.npz    Acopy.bbc.tsv           BAF.seg.tsv             GEX_Aallele.seg.npz     GEX_nsnps.seg.npz       features.tsv.gz         phased_snps.tsv.gz
ATAC_Tallele.seg.npz    Acopy.seg.tsv           Bcopy.bbc.tsv           GEX_Ballele.seg.npz     Position.bbc.tsv        gene_expression.npz
"""

if __name__ == "__main__":
    args = parse_arguments()
    sample = args["sample"]
    prep_dir = args["prep_dir"]
    out_dir = args["out_dir"]
    mode = args["mode"]
    modality = args["modality"]
    bin_level = args["bin_level"]
    min_posterior = args["min_posterior"]

    assert mode == "pure", f"other mode haven't implemented {mode}"

    os.makedirs(out_dir, exist_ok=True)

    ##################################################
    sc_data = SC_Data(prep_dir, modality, bin_level)
    sc_data.transform_data()
    sc_params = SC_Params(use_betabinom=True, fix_tau=False)
    print(sc_params)

    model = allele_model
    validate_model(sc_data, sc_params, model=model, out_dir=out_dir)

    ##################################################
    # separate normal and tumor cells via simple allele model TODO
    # and model gene base expression

    rep_dir = os.path.join(out_dir, f"rep_1")
    os.makedirs(rep_dir, exist_ok=True)
    assign_file = os.path.join(rep_dir, "expose_pi.tsv")
    if not os.path.exists(assign_file):
        run_model(sc_data, sc_params, model=model, curr_repeat=1, out_dir=rep_dir)

    ##################################################
    final_dir = rep_dir
    clones = sc_data.clones

    # compute final decision with MAP estimate
    def decision(r):
        probs = [r[clone] for clone in clones]
        (best_clone, best_prob) = max(zip(clones, probs), key=lambda t: t[1])
        if best_prob > min_posterior:
            return best_clone
        else:
            return "unassigned"

    assign_df = pd.read_table(os.path.join(final_dir, "expose_pi.tsv"), sep="\t")
    assign_df = assign_df.rename(
        columns={str(i): clone for i, clone in enumerate(clones)}
    )
    assign_df["BARCODE"] = sc_data.barcodes.BARCODE
    assign_df["Decision"] = assign_df.apply(func=lambda r: decision(r), axis=1)
    assign_df = assign_df[["BARCODE"] + clones + ["Decision"]]
    assign_df.to_csv(
        os.path.join(out_dir, "final_assignment.tsv"),
        sep="\t",
        header=True,
        index=False,
    )

    for dec_name in assign_df["Decision"].unique():
        assigned = assign_df.loc[assign_df["Decision"] == dec_name, :]
        print(f"#{dec_name}={len(assigned)}")
    sys.exit(0)
