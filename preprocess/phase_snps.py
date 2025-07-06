import os
import sys

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import binom
from scipy.special import betaln, gammaln, logsumexp, psi
from scipy.optimize import minimize

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils import *
from hmm_utils import *


def phase_snps(
    seg_file: str,
    bbc_file: str,
    snp_file: str,
    hair_file: str,
    nbed_file: str,
    tbed_file: str,
    sample: str,
    mode: str,
    out_file: str,
    tol=10e-6,
):
    """
    phase SNPs per segment, store phasing information to PHASE column
    REF~Binom(TOTAL, p); p=baf * PHASE + (1-baf) * (1 - PHASE)
    """
    if os.path.exists(out_file):
        print("phasing exists, skip")
        return
    print(f"compute phasing per CN segment using {mode} phasing")
    assert mode in ["em", "hmm-b"]
    assert os.path.exists(hair_file) or mode != "em"

    segs = pd.read_table(seg_file, sep="\t")
    snps = read_VCF(snp_file)
    snp_counts = read_baf_file(tbed_file)
    tumor_snps = pd.merge(
        left=snps, right=snp_counts, on=["CHR", "POS"], sort=False, how="left"
    )
    assert np.all(~tumor_snps.REF.isna()) and (np.all(~tumor_snps.ALT.isna()))
    tumor_snps.loc[:, "POS"] -= 1  # convert to 0-based indexing
    tumor_snps["TOTAL"] = tumor_snps["ALT"] + tumor_snps["REF"]
    tumor_snps["PHASE"] = np.nan

    hairs = None
    if mode.startswith("hmm"):
        hairs = load_hairs(hair_file, smoothing=True)
        assert len(tumor_snps) == len(hairs)

    for ch in segs["#CHR"].unique():
        segs_ch = segs[segs["#CHR"] == ch]
        num_segs_ch = len(segs_ch)
        tumor_snps_ch = tumor_snps[tumor_snps["CHR"] == ch]
        for si in range(num_segs_ch):
            seg = segs_ch.iloc[si]
            seg_s, seg_t = seg["START"], seg["END"]
            seg_baf = seg["mhBAF"]

            tsnps_seg = subset_baf(tumor_snps_ch, ch, seg_s, seg_t)
            nobs_seg = len(tsnps_seg)
            tsnps_idx = tsnps_seg.index
            refs = tsnps_seg["REF"].to_numpy()
            alts = tsnps_seg["ALT"].to_numpy()
            totals = tsnps_seg["TOTAL"].to_numpy()

            if mode.startswith("hmm"):
                hairs_seg = hairs[tsnps_idx[0] : tsnps_idx[-1] + 1, :]
                sitewise_transmat = build_sitewise_transmat(
                    tsnps_seg, hairs_seg, norm=True, log=True
                )
                new_baf, phases, ll = binom_hmm(
                    nobs_seg, refs, alts, totals, seg_baf, sitewise_transmat
                )
            else:
                assert mode == "em"
                log_aaf = np.log(1 - seg_baf)
                log_baf = np.log(seg_baf)
                phases = baf_posterior(refs, alts, log_aaf, log_baf)
            tumor_snps.loc[tsnps_idx, "PHASE"] = phases
            print(
                f"{ch}:{seg_s}-{seg_t}\t#SNPs={nobs_seg}\tBAF={round(seg_baf, 2)}"
            )

    tumor_snps = tumor_snps[~tumor_snps["PHASE"].isna()]
    tumor_snps.loc[:, "POS"] += 1  # convert back to 1-based indexing
    tumor_snps.to_csv(
        out_file,
        sep="\t",
        header=True,
        index=False,
        columns=["CHR", "POS", "REF", "ALT", "PHASE"],
    )
    return
