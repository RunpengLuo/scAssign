import os
import sys

import numpy as np
import pandas as pd

from sc_data import SC_Data
from sc_models import sc_model_GEX, sc_model_multiome
from sc_runner import validate_model, run_model

"""
ATAC_Aallele.seg.npz    ATAC_nsnps.seg.npz      BAF.bbc.tsv             Bcopy.seg.tsv           GEX_Tallele.seg.npz     Position.seg.tsv        peak_signals.npz
ATAC_Ballele.seg.npz    Acopy.bbc.tsv           BAF.seg.tsv             GEX_Aallele.seg.npz     GEX_nsnps.seg.npz       features.tsv.gz         phased_snps.tsv.gz
ATAC_Tallele.seg.npz    Acopy.seg.tsv           Bcopy.bbc.tsv           GEX_Ballele.seg.npz     Position.bbc.tsv        gene_expression.npz
"""

_, prep_dir, out_dir, out_prefix, mode, bin_level = sys.argv
os.makedirs(out_dir, exist_ok=True)

data_obj = SC_Data(prep_dir, mode, bin_level, out_dir)
data_obj.transform_data()

model = {"GEX": sc_model_GEX, "ATAC": None, "BOTH": sc_model_multiome}[mode]

validate_model(data_obj, model)

run_model(data_obj, model=model, curr_repeat=1)

