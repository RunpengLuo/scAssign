import os
import sys

import numpy as np
import pandas as pd

from sc_assign import SC_Assign

"""
ATAC_Aallele.seg.npz    ATAC_nsnps.seg.npz      BAF.bbc.tsv             Bcopy.seg.tsv           GEX_Tallele.seg.npz     Position.seg.tsv        peak_signals.npz
ATAC_Ballele.seg.npz    Acopy.bbc.tsv           BAF.seg.tsv             GEX_Aallele.seg.npz     GEX_nsnps.seg.npz       features.tsv.gz         phased_snps.tsv.gz
ATAC_Tallele.seg.npz    Acopy.seg.tsv           Bcopy.bbc.tsv           GEX_Ballele.seg.npz     Position.bbc.tsv        gene_expression.npz
"""

_, prep_dir, out_dir, out_prefix = sys.argv
os.makedirs(out_dir, exist_ok=True)

obj = SC_Assign(prep_dir, "BOTH", "seg", out_dir)

obj.transform_data()

obj.validate_model()

obj.run_model()
