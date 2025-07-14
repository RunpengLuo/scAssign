import os
import sys
import time

import numpy as np
import pandas as pd

import logging

import torch
import torch.nn.functional as F
from io_utils import *


class SC_Data:
    def __init__(
        self, prep_dir: str, data_type: str, bin_level: str
    ) -> None:
        assert data_type in ["GEX", "ATAC", "BOTH"]
        assert bin_level in ["seg", "bbc"]
        assert os.path.isdir(prep_dir)

        self.has_atac = data_type in ["ATAC", "BOTH"]
        self.has_gex = data_type in ["GEX", "BOTH"]
        self.bin_level = bin_level

        ##################################################
        self.barcodes = read_barcodes(prep_dir)
        self.num_cells = len(self.barcodes)
        print(f"#cells={self.num_cells}")

        ##################################################
        # copy-number information
        clones, bins, acopies, bcopies, ccopies, bafs = read_copy_number_data(
            prep_dir, "seg"
        )
        self.clones = clones
        self.bins = bins
        self.acopies = acopies
        self.bcopies = bcopies
        self.ccopies = ccopies
        self.bafs = bafs

        self.num_clones = len(self.clones)
        self.num_bins = len(self.bins)
        print(f"clones={clones}")
        print(f"#bins={self.num_bins}")

        ##################################################
        # scATACseq information allele-level, bin by cell
        if self.has_atac:
            _, bins, acopies, bcopies, ccopies, bafs = (
                read_copy_number_data(prep_dir, bin_level, "ATAC")
            )
            self.atac_bins = bins
            self.atac_acopies = acopies
            self.atac_bcopies = bcopies
            self.atac_ccopies = ccopies
            self.atac_bafs = bafs
            self.atac_num_bins = len(bins)

            counts_Aallele, counts_Ballele, counts_Tallele, counts_Nsnp = (
                read_single_cell_data(prep_dir, "ATAC", bin_level)
            )
            self.atac_acounts = counts_Aallele
            self.atac_bcounts = counts_Ballele
            self.atac_tcounts = counts_Tallele
            self.atac_nsnps = counts_Nsnp

        ##################################################
        # scRNAseq information, allele-level, bin by cell
        if self.has_gex:
            _, bins, acopies, bcopies, ccopies, bafs = (
                read_copy_number_data(prep_dir, bin_level, "GEX")
            )
            self.gex_bins = bins
            self.gex_acopies = acopies
            self.gex_bcopies = bcopies
            self.gex_ccopies = ccopies
            self.gex_bafs = bafs
            self.gex_num_bins = len(bins)

            counts_Aallele, counts_Ballele, counts_Tallele, counts_Nsnp = (
                read_single_cell_data(prep_dir, "GEX", bin_level)
            )
            self.gex_acounts = counts_Aallele
            self.gex_bcounts = counts_Ballele
            self.gex_tcounts = counts_Tallele
            self.gex_nsnps = counts_Nsnp

        ##################################################
        # features, total-level, feature by cell
        # cell-ranger matrix is not good, skip this for now
        # if os.path.exists(os.path.join(prep_dir, "features.tsv.gz")):
        #     features, gene_mat, gene_ids, gene_segIDs, peak_mat, peak_ids, peak_segIDs = (
        #         read_single_cell_features(prep_dir)
        #     )

        #     self.gene_mat = gene_mat
        #     self.gene_ids = gene_ids
        #     self.gene_segIDs = gene_segIDs
        #     self.peak_mat = peak_mat
        #     self.peak_ids = peak_ids
        #     self.peak_segIDs = peak_segIDs

        #     self.features = encode_feature_cn(features, self.ccopies, self.clones)
        #     if self.has_gex:
        #         self.gene_cns = get_feature_cn_matrix(self.features, self.clones, "GEX")
        #         self.num_genes = len(self.gene_ids)
        #         print(f"#genes={self.num_genes}")

        #     if self.has_atac:
        #         self.peak_cns = get_feature_cn_matrix(self.features, self.clones, "ATAC")
        #         self.num_peaks = len(self.peak_ids)
        #         print(f"#peaks={self.num_peaks}")
        #     assert not self.has_gex or len(self.gene_ids) > 0
        #     assert not self.has_atac or len(self.peak_ids) > 0

        assert self.num_cells > 0
        assert self.num_bins > 0
        print("SC_Data initialized")
        return

    def transform_data(self, epsilon=1e-4):
        """
        convert data to tensors
        """
        if self.has_gex:
            self.gex_acounts_tensor = torch.from_numpy(self.gex_acounts)
            self.gex_acounts_tensor_T = self.gex_acounts_tensor.T
            self.gex_bcounts_tensor = torch.from_numpy(self.gex_bcounts)
            self.gex_bcounts_tensor_T = self.gex_bcounts_tensor.T
            self.gex_tcounts_tensor = torch.from_numpy(self.gex_tcounts)  # bin by cell
            self.gex_tcounts_tensor_T = self.gex_tcounts_tensor.T  # cell by bin

            self.gex_acopies_tensor = torch.from_numpy(self.gex_acopies)
            self.gex_acopies_tensor_T = self.gex_acopies_tensor.T
            self.gex_bcopies_tensor = torch.from_numpy(self.gex_bcopies)
            self.gex_bcopies_tensor_T = self.gex_bcopies_tensor.T
            self.gex_ccopies_tensor = torch.from_numpy(self.gex_ccopies)
            self.gex_ccopies_tensor_T = self.gex_ccopies_tensor.T
            self.gex_bafs_tensor = torch.from_numpy(self.gex_bafs).clamp(
                min=epsilon
            )  # bin by clone
            self.gex_bafs_tensor_T = self.gex_bafs_tensor.T  # clone by bin

        if self.has_atac:
            self.atac_acounts_tensor = torch.from_numpy(self.atac_acounts)
            self.atac_acounts_tensor_T = self.atac_acounts_tensor.T
            self.atac_bcounts_tensor = torch.from_numpy(self.atac_bcounts)
            self.atac_bcounts_tensor_T = self.atac_bcounts_tensor.T
            self.atac_tcounts_tensor = torch.from_numpy(
                self.atac_tcounts
            )  # bin by cell
            self.atac_tcounts_tensor_T = self.atac_tcounts_tensor.T  # cell by bin

            self.atac_acopies_tensor = torch.from_numpy(self.atac_acopies)
            self.atac_acopies_tensor_T = self.atac_acopies_tensor.T
            self.atac_bcopies_tensor = torch.from_numpy(self.atac_bcopies)
            self.atac_bcopies_tensor_T = self.atac_bcopies_tensor.T
            self.atac_ccopies_tensor = torch.from_numpy(self.atac_ccopies)
            self.atac_ccopies_tensor_T = self.atac_ccopies_tensor.T
            self.atac_bafs_tensor = torch.from_numpy(self.atac_bafs).clamp(
                min=epsilon
            )  # bin by clone
            self.atac_bafs_tensor_T = self.atac_bafs_tensor.T  # clone by bin
        return
