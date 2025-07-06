# import os
# import sys


# import pandas as pd
# import numpy as np

# from utils import *

# _, prep_dir, gex_dir, atac_dir = sys.argv

# # assign SNPs to features, check how many SNPs didn't assigned
# feature_file = os.path.join(prep_dir, "features.tsv.gz")
# atac_vcf = os.path.join(atac_dir, "cellSNP.base.vcf.gz")
# gex_vcf = os.path.join(gex_dir, "cellSNP.base.vcf.gz")


# features = pd.read_csv(feature_file, sep="\t")
# genes = features.loc[features["feature_type"] == "Gene Expression"]
# peaks = features.loc[features["feature_type"] == "Peaks"]

# atac_snps = read_VCF(atac_vcf)
# atac_snps.loc[:, "POS"] -= 1
# gex_snps = read_VCF(gex_vcf)
# gex_snps.loc[:, "POS"] -= 1

# def locate_gene(snp):
#     ch, pos = snp["POS"], snp["POS"]
#     gene = genes.loc[(features["#CHR"] == ch) & (features["START"] <= pos) & (pos < features["END"]), "feature_id"]
#     if len(gene) == 0:
#         gene = pd.NA
#     return gene

# def locate_peak(snp):
#     ch, pos = snp["POS"], snp["POS"]
#     peak = peaks.loc[(features["#CHR"] == ch) & (features["START"] <= pos) & (pos < features["END"]), "feature_id"]
#     if len(peak) == 0:
#         peak = pd.NA
#     return peak

# snp_genes = gex_snps.apply(func=lambda snp: locate_gene(snp), axis=1)
# snp_peaks = atac_snps.apply(func=lambda snp: locate_peak(snp), axis=1)

# print(type(snp_genes))

# TODO to be removed


import os
import sys
import pandas as pd
import numpy as np
import pyranges as pr
from utils import *
from scipy.io import mmread

_, prep_dir, gex_dir, atac_dir = sys.argv

# File paths
feature_file = os.path.join(prep_dir, "features.tsv.gz")
atac_vcf = os.path.join(atac_dir, "cellSNP.base.vcf.gz")
atac_tmat = os.path.join(atac_dir, "cellSNP.tag.DP.mtx")
atac_mtx: csr_matrix = mmread(atac_tmat).tocsr()

gex_vcf = os.path.join(gex_dir, "cellSNP.base.vcf.gz")
gex_tmat = os.path.join(gex_dir, "cellSNP.tag.DP.mtx")
gex_mtx: csr_matrix = mmread(gex_tmat).tocsr()

# Read feature matrix
features = pd.read_csv(feature_file, sep="\t")
genes = features.loc[features["feature_type"] == "Gene Expression"].copy()
peaks = features.loc[features["feature_type"] == "Peaks"].copy()

# Read SNPs
atac_snps = read_VCF(atac_vcf)
atac_snps["POS"] -= 1
atac_snps["NZ"] = np.diff(atac_mtx.indptr)
print(atac_snps)
atac_snps = atac_snps.loc[atac_snps["NZ"] > 0, :].reset_index(drop=True)
gex_snps = read_VCF(gex_vcf)
gex_snps["POS"] -= 1
gex_snps["NZ"] = np.diff(gex_mtx.indptr)
gex_snps = gex_snps.loc[gex_snps["NZ"] > 0, :].reset_index(drop=True)

# Convert to PyRanges
def to_pyranges(df, chrom_col="CHR", start_col="POS", name_col=None):
    pr_df = df[[chrom_col, start_col]].copy()
    pr_df.columns = ["Chromosome", "Start"]
    pr_df["End"] = pr_df["Start"] + 1
    if name_col:
        pr_df[name_col] = df[name_col].values
    return pr.PyRanges(pr_df)

def feature_to_pyranges(feature_df):
    pr_df = feature_df[["#CHR", "START", "END", "feature_id"]].copy()
    pr_df.columns = ["Chromosome", "Start", "End", "feature_id"]
    return pr.PyRanges(pr_df)

# Convert SNPs and features to ranges
gex_snp_ranges = to_pyranges(gex_snps)
atac_snp_ranges = to_pyranges(atac_snps)
gene_ranges = feature_to_pyranges(genes)
peak_ranges = feature_to_pyranges(peaks)

# Join to find overlaps
gene_overlap = gex_snp_ranges.join(gene_ranges)
peak_overlap = atac_snp_ranges.join(peak_ranges)

# You can merge results back to original SNPs if needed:
gex_snps["feature_id"] = pd.NA
atac_snps["feature_id"] = pd.NA

if not gene_overlap.df.empty:
    gex_snps.loc[gene_overlap.df.index, "feature_id"] = gene_overlap.df["feature_id"].values

if not peak_overlap.df.empty:
    atac_snps.loc[peak_overlap.df.index, "feature_id"] = peak_overlap.df["feature_id"].values

print("Mapped GEX SNPs:", gex_snps["feature_id"].notna().sum())
print(f"total GEX SNPs: {len(gex_snps)}")
print("Mapped ATAC SNPs:", atac_snps["feature_id"].notna().sum())
print(f"total ATAC SNPs: {len(atac_snps)}")
