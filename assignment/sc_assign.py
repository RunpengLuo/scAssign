import os
import sys
import time

from io_utils import *
from model_utils import *

import numpy as np
import pandas as pd

import logging

import pyro
import pyro.optim
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.infer.autoguide.initialization import init_to_sample
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.ops.indexing import Vindex

import torch
from torch.nn import Softplus
import torch.nn.functional as F


class SC_Assign:
    def __init__(
        self, prep_dir: str, data_type: str, bin_level: str, out_dir: str
    ) -> None:
        assert data_type in ["GEX", "ATAC", "BOTH"]
        assert bin_level in ["seg", "bbc"]
        assert os.path.isdir(prep_dir)

        self.data_type = data_type
        self.bin_level = bin_level
        self.out_dir = out_dir

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
        self.atac_acounts = None
        self.atac_bcounts = None
        self.atac_tcounts = None
        self.atac_nsnps = None
        if data_type in ["ATAC", "BOTH"]:
            counts_Aallele, counts_Ballele, counts_Tallele, counts_Nsnp = (
                read_single_cell_data(prep_dir, "ATAC", "seg")
            )
            self.atac_acounts = counts_Aallele
            self.atac_bcounts = counts_Ballele
            self.atac_tcounts = counts_Tallele
            self.atac_nsnps = counts_Nsnp

        ##################################################
        # scRNAseq information, allele-level, bin by cell
        self.gex_acounts = None
        self.gex_bcounts = None
        self.gex_tcounts = None
        self.gex_nsnps = None
        if data_type in ["GEX", "BOTH"]:
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
        self.gene_cns = None
        if data_type in ["GEX", "BOTH"]:
            self.gene_cns = get_feature_cn_matrix(self.features, self.clones, "GEX")
            self.num_genes = len(self.gene_ids)
            print(f"#genes={self.num_genes}")

        self.peak_cns = None
        if data_type in ["ATAC", "BOTH"]:
            self.peak_cns = get_feature_cn_matrix(self.features, self.clones, "ATAC")
            self.num_peaks = len(self.peak_ids)
            print(f"#peaks={self.num_peaks}")

        assert self.num_cells > 0
        assert self.num_bins > 0
        assert self.gene_ids is None or len(self.gene_ids) > 0
        assert self.peak_ids is None or len(self.peak_ids) > 0
        # print("SC_Assign initialized")

    def transform_data(
        self,
    ):
        """
        convert data to tensors
        """
        if self.data_type in ["GEX", "BOTH"]:
            self.gene_mat_tensor = torch.from_numpy(self.gene_mat)  # gene by cell
            self.gene_mat_tensor_T = self.gene_mat_tensor.T  # cell by gene
            self.gene_cns_tensor = torch.from_numpy(self.gene_cns)  # gene by clone
            self.gene_cns_tensor_T = self.gene_cns_tensor.T  # clone by gene
            # self.gene_total_tensor = torch.from_numpy(self.gene_mat.sum(axis=0))

            self.gex_acounts_tensor = torch.from_numpy(self.gex_acounts)
            self.gex_acounts_tensor_T = self.gex_acounts_tensor.T
            self.gex_bcounts_tensor = torch.from_numpy(self.gex_bcounts)
            self.gex_bcounts_tensor_T = self.gex_bcounts_tensor.T
            self.gex_tcounts_tensor = torch.from_numpy(self.gex_tcounts)  # bin by cell
            self.gex_tcounts_tensor_T = self.gex_tcounts_tensor.T  # cell by bin
        
        if self.data_type in ["ATAC", "BOTH"]:
            self.peak_mat_tensor = torch.from_numpy(self.peak_mat)  # peak by cell
            self.peak_mat_tensor_T = self.peak_mat_tensor.T  # cell by peak
            self.peak_cns_tensor = torch.from_numpy(self.peak_cns)  # peak by clone
            self.peak_cns_tensor_T = self.peak_cns_tensor.T  # clone by peak
            # self.peak_total_tensor = torch.from_numpy(self.peak_mat.sum(axis=0))

            self.atac_acounts_tensor = torch.from_numpy(self.atac_acounts)
            self.atac_acounts_tensor_T = self.atac_acounts_tensor.T
            self.atac_bcounts_tensor = torch.from_numpy(self.atac_bcounts)
            self.atac_bcounts_tensor_T = self.atac_bcounts_tensor.T
            self.atac_tcounts_tensor = torch.from_numpy(self.atac_tcounts)  # bin by cell
            self.atac_tcounts_tensor_T = self.atac_tcounts_tensor.T  # cell by bin

        self.acopies_tensor = torch.from_numpy(self.acopies)  # bin by clone
        self.acopies_tensor_T = self.acopies_tensor.T  # clone by bin
        self.bcopies_tensor = torch.from_numpy(self.bcopies)  # bin by clone
        self.bcopies_tensor_T = self.bcopies_tensor.T  # clone by bin
        self.ccopies_tensor = torch.from_numpy(self.ccopies)  # bin by clone
        self.ccopies_tensor_T = self.ccopies_tensor.T  # clone by bin

    @config_enumerate
    def sc_model(self, temperature=0.5):
        K = 6
        chi_alpha = 2
        chi_rate = 1
        pk_betas = torch.tensor([1, 1]).float()
        h_betas = torch.tensor([1, 1]).float()
        sigma = 10
        dirichlet_alpha = 1
        softplus = Softplus()

        epsilon = 1e-4  # avoid log(0)

        ##################################################
        mu_g0_hat = torch.mean(self.gene_mat_tensor.float(), dim=1)  # (ngenes, )
        mu_g0_hat = mu_g0_hat.clamp(min=epsilon)
        # gene_totals = torch.tensor(self.gene_mat.sum(axis=0))  # (ncells,)
        # gex_totals = torch.tensor(self.gex_tcounts.sum(axis=0))  # (ncells,)
        # gene_bin_ids: sorted bin IDs covering genes; gene_bin_ids[gene_bin_inverse] == gene_segIDs
        gene_bin_ids, gene_bin_inverse = np.unique(
            self.gene_segIDs, return_inverse=True
        )
        bin2gene_ids = [[] for _ in range(self.num_bins)]
        for gene_id, bin_id in enumerate(gene_bin_inverse):
            bin2gene_ids[bin_id].append(gene_id)
        for bin_id in range(self.num_bins):
            bin2gene_ids[bin_id] = torch.tensor(bin2gene_ids[bin_id]).long()

        padded_bin2gene_ids = torch.stack(
            [F.pad(t, (0, self.num_genes - t.shape[0]), value=-1) for t in bin2gene_ids]
        )

        ##################################################
        chi = pyro.sample(
            "expose_chi",
            dist.Gamma(torch.ones(K) * chi_alpha, torch.ones(K) * chi_rate).to_event(1),
        )
        with pyro.plate("GENE", self.num_genes):
            w_g = pyro.sample(
                "expose_w",
                dist.Normal(torch.zeros(K), 1.0 / torch.sqrt(chi)).to_event(1),
            )
            pk_g = pyro.sample("expose_pk", dist.Dirichlet(pk_betas))
            k_g = pyro.sample(
                "expose_k",
                dist.RelaxedOneHotCategorical(
                    temperature=torch.tensor(temperature), probs=pk_g
                ),
            )  # [ngenes, 2], first column for dependent, second column for independent

            mu_g0 = pyro.sample(
                "expose_mu0",
                dist.Normal(
                    inverse_softplus(mu_g0_hat),
                    scale=torch.ones(self.num_genes) * sigma,
                ),
            )
            mu_g0 = softplus(mu_g0)
            mu_g1 = mu_g0  # TODO

        ##################################################
        ave_w_g = torch.zeros(self.num_bins, K)
        for bin_id in gene_bin_ids:
            gene_ids = bin2gene_ids[bin_id]
            ave_w_g[bin_id] = w_g[gene_ids].mean(dim=0)

        ##################################################
        plate_cell = pyro.plate("CELL", self.num_cells)
        with plate_cell:
            psi_n = pyro.sample(
                "expose_psi", dist.Normal(torch.zeros(K), torch.ones(K)).to_event(1)
            )

            pi = pyro.sample(
                "expose_pi",
                dist.Dirichlet(torch.ones(self.num_clones) * dirichlet_alpha),
            )
            z_n = pyro.sample("z", dist.Categorical(pi))

            mu_x_n = (
                mu_g0 * Vindex(self.gene_cns_tensor_T)[z_n] * k_g[:, 0]
                + mu_g1 * 2 * k_g[:, 1]
            ) * torch.exp(torch.matmul(psi_n, torch.transpose(w_g, 0, 1)))
            x_n = pyro.sample(
                "x",
                dist.Multinomial(total_count=1, probs=mu_x_n, validate_args=False),
                obs=self.gene_mat_tensor_T,
            )

            mu_d_n = (
                Vindex(self.ccopies_tensor_T)[z_n] * torch.dot(k_g[:, 0], mu_g0)
                + 2 * torch.dot(k_g[:, 1], mu_g1)
            ) * torch.exp(torch.matmul(psi_n, torch.transpose(ave_w_g, 0, 1)))
            d_n = pyro.sample(
                "d",
                dist.Multinomial(total_count=1, probs=mu_d_n, validate_args=False),
                obs=self.gex_tcounts_tensor_T,
            )

        ##################################################
        with pyro.plate("SEGMENT", self.num_bins) as bin_id:
            # ph_s = pyro.sample("expose_ph", dist.Dirichlet(h_betas))
            # h_s = pyro.sample(
            #     "expose_h",
            #     dist.RelaxedOneHotCategorical(
            #         temperature=torch.tensor(temperature), probs=ph_s
            #     ),
            # )

            # (nbins, nclones)
            gene_idx = padded_bin2gene_ids[bin_id]
            gene_idx = gene_idx[gene_idx != -1]
            bcopy_s = Vindex(self.bcopies_tensor)[bin_id] * torch.sum(
                k_g[gene_idx, 0]
            ) + 1 * torch.sum(k_g[gene_idx, 1])
            ccopy_s = Vindex(self.ccopies_tensor)[bin_id] * torch.sum(
                k_g[gene_idx, 0]
            ) + 2 * torch.sum(k_g[gene_idx, 1])
            baf_s = bcopy_s / ccopy_s  # (nbins, nclones)
            aaf_s = 1 - baf_s

        ##################################################
        with plate_cell:
            # bp_s = (
            #     Vindex(baf_s)[:, z_n] * h_s[:, 0] + Vindex(aaf_s)[:, z_n] * h_s[:, 1]
            # )  # (ncells, nbins)
            bp_s = Vindex(baf_s)[:, z_n]  # (ncells, nbins)
            probs_s = torch.stack([1 - bp_s, bp_s], dim=-1)  # (ncells, nbins, 2)
            obs_s = torch.stack(
                [self.gex_acounts_tensor_T, self.gex_bcounts_tensor_T], dim=-1
            )  # (ncells, nbins, 2)
            y_n = pyro.sample(
                "y",
                dist.Multinomial(
                    total_count=1, probs=probs_s, validate_args=False
                ).to_event(1),
                obs=obs_s,
            )

    def validate_model(self, plot_model=True):
        pyro.enable_validation(False)
        pyro.clear_param_store()
        model = self.sc_model
        optim = pyro.optim.Adam({"lr": 0.1, "betas": [0.8, 0.99]})
        elbo = TraceEnum_ELBO(max_plate_nesting=1)
        global_guide = AutoDelta(
            poutine.block(
                model, expose_fn=lambda msg: msg["name"].startswith("expose_")
            ),
            init_loc_fn=init_to_sample,
        )
        svi = SVI(model, global_guide, optim, loss=elbo)
        loss = svi.loss(model, global_guide)
        print(f"validate-loss={loss}")

        guide_trace = poutine.trace(global_guide).get_trace()
        model_trace = poutine.trace(poutine.replay(model, guide_trace)).get_trace()
        model_trace.compute_log_prob()
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                value = site["value"]
                print(f"{name}: shape={site['value'].shape}")
                if torch.isnan(value).any():
                    print(f"value NaN detected at site: {name}, value={value}")
                    sys.exit(1)
                if torch.isnan(site["log_prob"]).any():
                    print(f"log_prob NaN detected at site: {name}, value={value}")
                    sys.exit(1)
        if plot_model:
            dot = pyro.render_model(
                model, model_args=(), render_distributions=True, render_params=True
            )
            dot.graph_attr.update(
                # rankdir="TB",       # top-to-bottom layout
                ranksep="1.0",  # more space between ranks
                nodesep="1.0",  # more space between nodes
                fontsize="12",  # smaller font
                dpi="300",
            )
            dot.node_attr.update(fontsize="12")
            dot.render(os.path.join(self.out_dir, "model"), format="png", cleanup=True)
        return

    def run_model(self, curr_repeat=1):
        start_time = round(time.time() * 1000)
        learning_rate = 0.1
        anneal_rate = 0.01
        min_temp = 0.5
        max_temp = 1.0
        min_iter = 10
        max_iter = 400
        rel_tol = 5e-5

        np_temp = max_temp
        losses = []

        init_seed = True
        init_niter = 1

        optim = pyro.optim.Adam({"lr": learning_rate, "betas": [0.8, 0.99]})
        elbo = TraceEnum_ELBO(max_plate_nesting=1)
        model = self.sc_model

        ##################################################
        def init_model(seed):
            pyro.set_rng_seed(seed)
            torch.manual_seed(seed)
            pyro.enable_validation(False)
            pyro.clear_param_store()

            global_guide = AutoDelta(
                poutine.block(
                    model, expose_fn=lambda msg: msg["name"].startswith("expose_")
                ),
                init_loc_fn=init_to_sample,
            )
            svi = SVI(model, global_guide, optim, loss=elbo)
            print(f"compute loss {seed}")
            loss = svi.loss(model, global_guide, np_temp)
            print(f"init model seed={seed} loss={loss}")
            return loss

        if init_seed:
            init_seeds = list(
                range(curr_repeat * init_niter, (curr_repeat + 1) * init_niter)
            )
            loss, seed = min((init_model(seed), seed) for seed in init_seeds)
            losses.append(loss)
            print(f"seed={seed}, init-loss={loss}")
        else:
            seed = curr_repeat * init_niter
            print(f"seed={seed}")

        ##################################################
        pyro.set_rng_seed(seed)
        torch.manual_seed(seed)
        pyro.clear_param_store()

        global_guide = AutoDelta(
            poutine.block(
                model, expose_fn=lambda msg: msg["name"].startswith("expose_")
            ),
            init_loc_fn=init_to_sample,
        )

        svi = SVI(model, global_guide, optim, loss=elbo)

        ##################################################
        for curr_iter in range(max_iter):
            if curr_iter % 100 == 1:
                # temperature annealing for Gumbel-softmax (RelaxedOneHotCategorical)
                np_temp = np.maximum(
                    max_temp * np.exp(-anneal_rate * curr_iter), min_temp
                )
                print(f"iteration={curr_iter}, drop temperature to {np_temp}")
            curr_loss = svi.step(np_temp)
            losses.append(curr_loss)
            print(f"iteration={curr_iter}\tloss={curr_loss}")
            if curr_iter >= min_iter:
                prev_loss = losses[-2]
                loss_diff = abs((curr_loss - prev_loss) / prev_loss)
                if loss_diff < rel_tol:
                    print(
                        f"ELBO converged at iteration {curr_iter}, loss-diff={loss_diff}"
                    )
                    break

        ##################################################
        map_estimates = global_guide()

        # store results to LOG file
        rep_dir = os.path.join(self.out_dir, f"rep_{curr_repeat}")
        os.makedirs(rep_dir, exist_ok=True)
        for name in map_estimates:
            out_file = os.path.join(rep_dir, f"{name}.tsv")
            data = map_estimates[name].detach().cpu().numpy()
            pd.DataFrame(data).to_csv(out_file, sep="\t", index=False)

        loss_df = pd.DataFrame({"loss": losses})
        loss_df.to_csv(
            os.path.join(rep_dir, "loss.tsv"), sep="\t", index=False, header=True
        )

        end_time = round(time.time() * 1000)
        print(f"total-elapsed={end_time - start_time}ms for rep={curr_repeat}")
        return
