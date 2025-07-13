import numpy as np
import pandas as pd
from utils import *

from external import run_bedtools_closest, run_bedtools_merge
from format_features import annotate_feature_segID


def adaptive_binning_seg(
    feature_df: pd.DataFrame,
    snp_df: pd.DataFrame,
    chrom=None,
    seg_id=None,
    min_dp_count=10,
):
    """bin features such that it has sufficient dp_count per bin, same chrom by default"""
    blocks = []
    dp_count = 0
    block_start = -1
    block_end = -1
    for _, seg in feature_df.iterrows():
        curr_start = seg["START"]
        block_end = seg["END"]
        if block_start == -1:
            block_start = curr_start
        snps = snp_df.loc[
            (snp_df["POS"] >= block_start) & (snp_df["POS"] < block_end), :
        ]
        dp_count = snps["DP"].sum()
        if dp_count >= min_dp_count:
            blocks.append([chrom, block_start, block_end, len(snps), dp_count, seg_id])
            block_start = -1
            dp_count = 0
    if dp_count > 0:
        blocks.append([chrom, block_start, block_end, len(snps), dp_count, seg_id])
        block_start = -1
        dp_count = 0

    return blocks


def adaptive_binning(
    segs: pd.DataFrame, feature_df: pd.DataFrame, snp_df: pd.DataFrame, min_dp_count=10
):
    """
    each segment consists multiple features
    aggregate to form meta-feature
    """
    feature_df = annotate_feature_segID(feature_df, segs)
    snps_grps = snp_df.groupby("#CHR", sort=False)
    blocks = []
    for seg_ch in segs["#CHR"].unique():
        snps_ch = snps_grps.get_group(seg_ch)
        segs_ch = segs.loc[segs["#CHR"] == seg_ch, :]
        for si, seg in segs_ch.iterrows():
            seg_start = seg["START"]
            seg_end = seg["END"]
            features = feature_df.loc[feature_df["segID"] == si, :]
            if len(features) == 0:
                print(f"warning, no feature found in {seg_ch}:{seg_start}-{seg_end}")
                continue
            snps = snps_ch.loc[
                (snps_ch["POS"] >= seg_start) & (snps_ch["POS"] < seg_end), :
            ]
            sub_blocks = adaptive_binning_seg(features, snps, seg_ch, si, min_dp_count)
            blocks.extend(sub_blocks)
    blocks_df = pd.DataFrame(
        blocks, columns=["#CHR", "START", "END", "#SNP", "DP", "segID"]
    )
    return blocks_df


def post_binning(
    blocks_df: pd.DataFrame,
    segs: pd.DataFrame,
    seg_acopy_file: str,
    seg_bcopy_file: str,
    segs_cbaf_file: str,
    out_bin_file: str,
    out_bin_acopy_file: str,
    out_bin_bcopy_file: str,
    out_bin_cbaf_file: str,
):
    seg_ids = blocks_df["segID"].tolist()
    blocks_df = blocks_df.join(
        segs[["mhBAF", "BBC_mhBAF"]], on="segID", sort=False, how="left"
    )
    blocks_df.to_csv(
        out_bin_file,
        index=False,
        header=True,
        sep="\t",
        columns=[
            "#CHR",
            "START",
            "END",
            "#SNP",
            "DP",
            "mhBAF",
            "BBC_mhBAF",
        ],
    )

    pd.read_table(seg_acopy_file, sep="\t").iloc[seg_ids].to_csv(
        out_bin_acopy_file, index=False, header=True, sep="\t"
    )
    pd.read_table(seg_bcopy_file, sep="\t").iloc[seg_ids].to_csv(
        out_bin_bcopy_file, index=False, header=True, sep="\t"
    )
    pd.read_table(segs_cbaf_file, sep="\t").iloc[seg_ids].to_csv(
        out_bin_cbaf_file, index=False, header=True, sep="\t"
    )
    return


# def merge_overlap(
#     df: pd.DataFrame
# ):
#     """ assume df is sorted """
#     df.loc[:, "ovlp_id"] = 0
#     ovlp_id = 0
#     for ch in df["#CHR"].unique():
#         df_ch = df.loc[df["#CHR"] == ch, :]
#         curr_end = df_ch.iloc[0]["END"]
#         group.loc[0, "START"]
#         curr_end = group.loc[0, "END"]

#         for _, row in df_ch.iterrows():
#             seg_start = row["START"]
#             seg_end = row["END"]
#             if prev_end == -1:


#     return


def binning_gex(
    seg_file: str,
    seg_acopy_file: str,
    seg_bcopy_file: str,
    segs_cbaf_file: str,
    gex_vcf_file: str,
    seg_bed_file: str,
    gene_bed_file: str,
    out_feature_file: str,
    out_bin_file: str,
    out_bin_acopy_file: str,
    out_bin_bcopy_file: str,
    out_bin_cbaf_file: str,
    tmp_dir: str,
    min_gex_count=10,
    max_dist=5,
    merge_overlap=True,
):
    """
    1) align SNPs to gene annotation via closest metric bedtools closest
    2) filter promoter alignment
    3) record per-gene #SNP, aggregated-DP
    """
    tmp_dist_file = os.path.join(tmp_dir, "snp_gene_distance.0pos.tsv")
    dist_df = run_bedtools_closest(
        seg_bed_file, gex_vcf_file, gene_bed_file, tmp_dist_file, tmp_dir
    )
    os.remove(tmp_dist_file)

    dist_df = dist_df.loc[dist_df["dist"] <= max_dist, :]
    dist_df = dist_df.loc[~dist_df["geneID"].str.lower().str.endswith("promoter"), :]
    print(f"#snp-gene mapping={len(dist_df)}")
    assert len(dist_df) > 0, "incompatiable gene annotation"

    snp_df = read_VCF_cellsnp_err_header(gex_vcf_file)
    snp_df.loc[:, "POS"] -= 1

    print("aggregate SNPs per gene")
    dist_df = pd.merge(left=dist_df, right=snp_df, on=["#CHR", "POS"], how="left")
    gene_grps = dist_df.groupby(by=["geneID", "geneType"], sort=False, as_index=True)
    gene_df = gene_grps.agg(
        **{
            "#CHR": ("#CHR", "first"),
            "START": ("POS", "min"),
            "END": ("POS", "max"),
            "DP": ("DP", "sum"),
        }
    ).reset_index()
    gene_df.loc[:, "END"] += 1
    gene_df.loc[:, "#SNP"] = gene_grps.size().reset_index(drop=True)
    gene_df["#CHR"] = pd.Categorical(
        gene_df["#CHR"], categories=get_ord2chr(), ordered=True
    )
    gene_df.sort_values(by=["#CHR", "START"], inplace=True, ignore_index=True)
    gene_df.to_csv(out_feature_file, sep="\t", header=True, index=False)

    # if merge_overlap: # TODO
    # merge overlapping genes

    segs = pd.read_table(seg_file, sep="\t")
    gene_df = annotate_feature_segID(gene_df, segs)
    print(f"#genes={len(gene_df)}")

    print("adaptive binning over genes")
    blocks_df = adaptive_binning(segs, gene_df, snp_df, min_gex_count)
    print(f"#blocks={len(blocks_df)}")
    post_binning(
        blocks_df,
        segs,
        seg_acopy_file,
        seg_bcopy_file,
        segs_cbaf_file,
        out_bin_file,
        out_bin_acopy_file,
        out_bin_bcopy_file,
        out_bin_cbaf_file,
    )
    return


def binning_atac(
    seg_file: str,
    seg_acopy_file: str,
    seg_bcopy_file: str,
    segs_cbaf_file: str,
    seg_bed_file: str,
    atac_vcf_file: str,
    out_feature_file: str,
    out_bin_file: str,
    out_bin_acopy_file: str,
    out_bin_bcopy_file: str,
    out_bin_cbaf_file: str,
    tmp_dir: str,
    min_atac_count=10,
    max_dist=500,
):
    peak_df = run_bedtools_merge(
        seg_bed_file, atac_vcf_file, out_feature_file, tmp_dir, max_dist
    )
    print(f"#peaks={len(peak_df)}")

    peak_df["#CHR"] = pd.Categorical(
        peak_df["#CHR"], categories=get_ord2chr(), ordered=True
    )
    peak_df.sort_values(by=["#CHR", "START"], inplace=True, ignore_index=True)

    snp_df = read_VCF_cellsnp_err_header(atac_vcf_file)
    snp_df.loc[:, "POS"] -= 1
    segs = pd.read_table(seg_file, sep="\t")
    peak_df = annotate_feature_segID(peak_df, segs)

    print("adaptive binning over peaks")
    blocks_df = adaptive_binning(segs, peak_df, snp_df, min_atac_count)
    print(f"#blocks={len(blocks_df)}")
    post_binning(
        blocks_df,
        segs,
        seg_acopy_file,
        seg_bcopy_file,
        segs_cbaf_file,
        out_bin_file,
        out_bin_acopy_file,
        out_bin_bcopy_file,
        out_bin_cbaf_file,
    )
    return
