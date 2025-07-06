import gzip
import time

from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy import sparse
import pandas as pd

from utils import *

def aggregate_counts(
    seg_file: str,
    barcode_file: str,
    base_vcf_file: str,
    dp_mtx_file: str,
    ad_mtx_file: str,
    phase_file: str,
    data_type: str,
    out_agg_a_file: str,
    out_agg_b_file: str,
    out_agg_t_file: str,
    out_agg_n_file: str,
    phase_col: str,
    v=1,
):
    """
    aggregate b-allele/total allele SNP counts per segment per cell
    output:
        1. segment by cell a-allele matrix file
        2. segment by cell b-allele matrix file
        3. segment by cell total-allele matrix file
        4. segment by cell #SNP matrix file
    """
    assert data_type in ["ATAC", "GEX"], f"invalid data type: {data_type}"

    barcodes = read_barcodes(barcode_file)
    print("load copy-number profile")

    segs = pd.read_table(seg_file, sep="\t")
    segs.index = segs.index.astype(int)

    print("load phase information")
    phase_df = pd.read_table(phase_file, sep="\t")
    base_snps = read_VCF(base_vcf_file)
    base_snps = pd.merge(base_snps, phase_df, on=["CHR", "POS"], how="left")
    base_snps.loc[:, "POS"] -= 1 # convert to 0-based indexing
    useref_arr = base_snps[phase_col].to_numpy()
    phased_arr = (~base_snps[phase_col].isna()).to_numpy()
    
    print("load DP and AD sparse matrix")
    dp_mtx: csr_matrix = mmread(dp_mtx_file).tocsr()
    ad_mtx: csr_matrix = mmread(ad_mtx_file).tocsr()

    # outputs
    b_allele_mat = np.zeros((len(segs), len(barcodes)), dtype=np.int32)
    t_allele_mat = np.zeros((len(segs), len(barcodes)), dtype=np.int32)
    snp_count_mat = np.zeros((len(segs), len(barcodes)), dtype=np.int32)

    for ch in segs["#CHR"].unique():
        segs_ch = segs[segs["#CHR"] == ch]
        num_segs_ch = len(segs_ch)
        base_snps_ch = base_snps[base_snps["CHR"] == ch]
        for si in range(num_segs_ch):
            seg = segs_ch.iloc[si]
            seg_s, seg_t = seg["START"], seg["END"]
            snps_seg = subset_baf(base_snps_ch, ch, seg_s, seg_t)
            snp_indices = snps_seg.index.to_numpy()
            nsnps_seg = len(snps_seg)
            print(f"{ch}:{seg_s}-{seg_t}\t#Het-SNP={nsnps_seg}")
            if nsnps_seg == 0:
                print("\twarning, no SNPs found at this region")
                continue

            # access allele-count matrix
            gsnp_s, gsnp_t = snp_indices[0], snp_indices[-1] + 1
            rows_dp = dp_mtx[gsnp_s:gsnp_t, :].toarray()
            rows_alt = ad_mtx[gsnp_s:gsnp_t, :].toarray()
            rows_ref = rows_dp - rows_alt

            snp_phased = phased_arr[snp_indices]
            snp_useref = useref_arr[snp_indices]

            # filter unphased SNPs
            rows_dp = rows_dp[snp_phased]
            rows_alt = rows_alt[snp_phased]
            rows_ref = rows_ref[snp_phased]
            snp_useref = snp_useref[snp_phased][:, np.newaxis]  # (n, 1)
            # aggregate phased counts
            rows_beta = rows_alt * (1 - snp_useref) + rows_ref * snp_useref

            seg_index = segs_ch.index[si]
            b_allele_mat[seg_index, :] = np.round(np.sum(rows_beta, axis=0))
            t_allele_mat[seg_index, :] = np.sum(rows_dp, axis=0)
            snp_count_mat[seg_index, :] = np.sum(rows_dp > 0, axis=0)
    
    a_allele_mat = (t_allele_mat - b_allele_mat).astype(np.int32)

    sparse.save_npz(out_agg_a_file, csr_matrix(a_allele_mat))
    sparse.save_npz(out_agg_b_file, csr_matrix(b_allele_mat))
    sparse.save_npz(out_agg_t_file, csr_matrix(t_allele_mat))
    sparse.save_npz(out_agg_n_file, csr_matrix(snp_count_mat))
    return
