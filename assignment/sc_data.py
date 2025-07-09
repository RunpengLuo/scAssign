import os
import sys
import time

from io_utils import *

import numpy as np
import pandas as pd

import logging

import torch
import torch.nn.functional as F


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
            prep_dir, bin_level
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
            counts_Aallele, counts_Ballele, counts_Tallele, counts_Nsnp = (
                read_single_cell_data(prep_dir, "ATAC", "seg")
            )
            self.atac_acounts = counts_Aallele
            self.atac_bcounts = counts_Ballele
            self.atac_tcounts = counts_Tallele
            self.atac_nsnps = counts_Nsnp

        ##################################################
        # scRNAseq information, allele-level, bin by cell
        if self.has_gex:
            counts_Aallele, counts_Ballele, counts_Tallele, counts_Nsnp = (
                read_single_cell_data(prep_dir, "GEX", "seg")
            )
            self.gex_acounts = counts_Aallele
            self.gex_bcounts = counts_Ballele
            self.gex_tcounts = counts_Tallele
            self.gex_nsnps = counts_Nsnp

        ##################################################
        # features, total-level, feature by cell
        features, gene_mat, gene_ids, gene_segIDs, peak_mat, peak_ids, peak_segIDs = (
            read_single_cell_features(prep_dir)
        )

        self.gene_mat = gene_mat
        self.gene_ids = gene_ids
        self.gene_segIDs = gene_segIDs
        self.peak_mat = peak_mat
        self.peak_ids = peak_ids
        self.peak_segIDs = peak_segIDs

        self.features = encode_feature_cn(features, self.ccopies, self.clones)
        if self.has_gex:
            self.gene_cns = get_feature_cn_matrix(self.features, self.clones, "GEX")
            self.num_genes = len(self.gene_ids)
            print(f"#genes={self.num_genes}")

        if self.has_atac:
            self.peak_cns = get_feature_cn_matrix(self.features, self.clones, "ATAC")
            self.num_peaks = len(self.peak_ids)
            print(f"#peaks={self.num_peaks}")

        assert self.num_cells > 0
        assert self.num_bins > 0
        assert not self.has_gex or len(self.gene_ids) > 0
        assert not self.has_atac or len(self.peak_ids) > 0
        # print("SC_Data initialized")

    def transform_data(self, epsilon=1e-4):
        """
        convert data to tensors
        """
        if self.has_gex:
            self.gene_mat_tensor = torch.from_numpy(
                self.gene_mat
            ).float()  # gene by cell
            self.gene_mat_tensor_T = self.gene_mat_tensor.T  # cell by gene
            self.gene_cns_tensor = torch.from_numpy(self.gene_cns)  # gene by clone
            self.gene_cns_tensor_T = self.gene_cns_tensor.T  # clone by gene
            # self.gene_total_tensor = torch.from_numpy(self.gene_mat.sum(axis=0))

            self.mu_g0_hat = torch.mean(self.gene_mat_tensor, dim=1).clamp(
                min=epsilon
            )  # (ngenes, )

            self.gene_bin_ids, gene_bin_inverse = np.unique(
                self.gene_segIDs, return_inverse=True
            )
            bin2gene_ids = [[] for _ in range(self.num_bins)]
            for gene_id, bin_id in enumerate(gene_bin_inverse):
                bin2gene_ids[bin_id].append(gene_id)
            for bin_id in range(self.num_bins):
                bin2gene_ids[bin_id] = torch.tensor(bin2gene_ids[bin_id]).long()
            self.bin2gene_ids = bin2gene_ids
            self.padded_bin2gene_ids = torch.stack(
                [
                    F.pad(t, (0, self.num_genes - t.shape[0]), value=-1)
                    for t in bin2gene_ids
                ]
            )

            self.gex_acounts_tensor = torch.from_numpy(self.gex_acounts)
            self.gex_acounts_tensor_T = self.gex_acounts_tensor.T
            self.gex_bcounts_tensor = torch.from_numpy(self.gex_bcounts)
            self.gex_bcounts_tensor_T = self.gex_bcounts_tensor.T
            self.gex_tcounts_tensor = torch.from_numpy(self.gex_tcounts)  # bin by cell
            self.gex_tcounts_tensor_T = self.gex_tcounts_tensor.T  # cell by bin

        if self.has_atac:
            self.peak_mat_tensor = torch.from_numpy(
                self.peak_mat
            ).float()  # peak by cell
            self.peak_mat_tensor_T = self.peak_mat_tensor.T  # cell by peak
            self.peak_cns_tensor = torch.from_numpy(self.peak_cns)  # peak by clone
            self.peak_cns_tensor_T = self.peak_cns_tensor.T  # clone by peak
            # self.peak_total_tensor = torch.from_numpy(self.peak_mat.sum(axis=0))

            self.mu_p0_hat = torch.mean(self.peak_mat_tensor, dim=1).clamp(
                min=epsilon
            )  # (npeaks, )

            self.peak_bin_ids, peak_bin_inverse = np.unique(
                self.peak_segIDs, return_inverse=True
            )
            bin2peak_ids = [[] for _ in range(self.num_bins)]
            for peak_id, bin_id in enumerate(peak_bin_inverse):
                bin2peak_ids[bin_id].append(peak_id)
            for bin_id in range(self.num_bins):
                bin2peak_ids[bin_id] = torch.tensor(bin2peak_ids[bin_id]).long()
            self.bin2peak_ids = bin2peak_ids
            self.padded_bin2peak_ids = torch.stack(
                [
                    F.pad(t, (0, self.num_peaks - t.shape[0]), value=-1)
                    for t in bin2peak_ids
                ]
            )

            self.atac_acounts_tensor = torch.from_numpy(self.atac_acounts)
            self.atac_acounts_tensor_T = self.atac_acounts_tensor.T
            self.atac_bcounts_tensor = torch.from_numpy(self.atac_bcounts)
            self.atac_bcounts_tensor_T = self.atac_bcounts_tensor.T
            self.atac_tcounts_tensor = torch.from_numpy(
                self.atac_tcounts
            )  # bin by cell
            self.atac_tcounts_tensor_T = self.atac_tcounts_tensor.T  # cell by bin

        self.acopies_tensor = torch.from_numpy(self.acopies)
        self.acopies_tensor_T = self.acopies_tensor.T
        self.bcopies_tensor = torch.from_numpy(self.bcopies)
        self.bcopies_tensor_T = self.bcopies_tensor.T
        self.ccopies_tensor = torch.from_numpy(self.ccopies)
        self.ccopies_tensor_T = self.ccopies_tensor.T
        self.bafs_tensor = torch.from_numpy(self.bafs).clamp(
            min=epsilon
        )  # bin by clone
        self.bafs_tensor_T = self.bafs_tensor.T  # clone by bin
        return
