import pyro
import pyro.optim
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.ops.indexing import Vindex

import torch
from torch.nn import Softplus
import torch.nn.functional as F

from sc_data import SC_Data
from model_utils import *

##################################################
def sample_gene(
    data: SC_Data,
    chi_alpha: int,
    chi_rate: int,
    pk_betas: torch.Tensor,
    temp_tensor: torch.Tensor,
    sigma: int,
    K: int,
    softplus,
    epsilon,
):
    chi_alpha = torch.ones(K) * chi_alpha
    chi_rate = torch.ones(K) * chi_rate
    chi_gene = pyro.sample(
        "expose_chi_gene",
        dist.Gamma(chi_alpha, chi_rate).to_event(1),
    )
    std_gene = 1.0 / torch.sqrt(chi_gene.clamp(min=epsilon))
    mu_gene = torch.zeros(K)

    with pyro.plate("GENE", data.num_genes):
        pk_g = pyro.sample("expose_pk_gene", dist.Dirichlet(pk_betas))
        k_g = pyro.sample(
            "expose_k_gene",
            dist.RelaxedOneHotCategorical(temperature=temp_tensor, probs=pk_g),
        )  # [ngenes, 2], first column for dependent, second column for independent

        w_g = pyro.sample(
            "expose_w_gene",
            dist.Normal(mu_gene, std_gene).to_event(1),
        )

        mu_g0 = pyro.sample(
            "expose_mu0_gene",
            dist.Normal(
                inverse_softplus(data.mu_g0_hat),
                scale=torch.ones(data.num_genes) * sigma,
            ),
        )
        mu_g0 = softplus(mu_g0)
        mu_g1 = mu_g0  # TODO

    ave_w_g = torch.zeros(data.num_bins, K)
    for bin_id in data.gene_bin_ids:
        gene_ids = data.bin2gene_ids[bin_id]
        ave_w_g[bin_id] = w_g[gene_ids].mean(dim=0)
    return k_g, w_g, mu_g0, mu_g1, ave_w_g

##################################################
def sample_peak(
    data: SC_Data,
    chi: torch.Tensor,
    pk_betas: torch.Tensor,
    temp_tensor: torch.Tensor,
    sigma: int,
    K: int,
    softplus,
):
    with pyro.plate("PEAK", data.num_peaks):
        # TODO do not model gene dosage effect for ATACseq data
        pk_p = pyro.sample("expose_pk_peak", dist.Dirichlet(pk_betas))
        k_p = pyro.sample(
            "expose_k_peak",
            dist.RelaxedOneHotCategorical(
                temperature=temp_tensor, probs=pk_p
            ),
        )  # [ngenes, 2], first column for dependent, second column for independent

        w_p = pyro.sample(
            "expose_w_peak",
            dist.Normal(torch.zeros(K), 1.0 / torch.sqrt(chi)).to_event(1),
        )

        mu_p0 = pyro.sample(
            "expose_mu0_peak",
            dist.Normal(
                inverse_softplus(data.mu_p0_hat),
                scale=torch.ones(data.num_peaks) * sigma,
            ),
        )
        mu_p0 = softplus(mu_p0)
        mu_p1 = mu_p0  # TODO

    ave_w_p = torch.zeros(data.num_bins, K)
    for bin_id in data.peak_bin_ids:
        peak_ids = data.bin2peak_ids[bin_id]
        ave_w_p[bin_id] = w_p[peak_ids].mean(dim=0)
    return k_p, w_p, mu_p0, mu_p1, ave_w_p

##################################################
@config_enumerate
def sc_model_GEX(data: SC_Data, temperature=0.5):
    assert data.has_gex
    K = 6  # structural noise latent dim
    chi_alpha = 2
    chi_rate = 1
    pk_betas = torch.tensor([1, 1]).float()
    # h_betas = torch.tensor([1, 1]).float() # marginal phasing prior
    sigma = 10
    dirichlet_alpha = 1
    softplus = Softplus()

    epsilon = 1e-4  # avoid log(0)
    temp_tensor = torch.tensor(temperature)

    ##################################################
    chi = pyro.sample(
        "expose_chi",
        dist.Gamma(torch.ones(K) * chi_alpha, torch.ones(K) * chi_rate).to_event(1),
    )

    ##################################################
    k_g, w_g, mu_g0, mu_g1, ave_w_g = sample_gene(
        data, chi, pk_betas, temp_tensor, sigma, K, softplus
    )

    ##################################################
    with pyro.plate("SEGMENT", data.num_bins) as bin_id:
        # TODO do we marginalize the phasing? not necessary
        # ph_s = pyro.sample("expose_ph", dist.Dirichlet(h_betas))
        # h_s = pyro.sample(
        #     "expose_h",
        #     dist.RelaxedOneHotCategorical(
        #         temperature=torch.tensor(temperature), probs=ph_s
        #     ),
        # )
        gene_idx = data.padded_bin2gene_ids[bin_id]
        gene_idx = gene_idx[gene_idx != -1]
        bcopy_s_gene = Vindex(data.bcopies_tensor)[bin_id] * torch.sum(
            k_g[gene_idx, 0]
        ) + 1 * torch.sum(k_g[gene_idx, 1])
        ccopy_s_gene = Vindex(data.ccopies_tensor)[bin_id] * torch.sum(
            k_g[gene_idx, 0]
        ) + 2 * torch.sum(k_g[gene_idx, 1])
        baf_s_gene = bcopy_s_gene / ccopy_s_gene  # (nbins, nclones)
        baf_s_gene = baf_s_gene.clamp(min=epsilon)

    ##################################################
    with pyro.plate("CELL", data.num_cells):
        pi = pyro.sample(
            "expose_pi",
            dist.Dirichlet(torch.ones(data.num_clones) * dirichlet_alpha),
        )
        z_n = pyro.sample("z", dist.Categorical(pi), infer={"enumerate": "parallel"})

        psi_n = pyro.sample(
            "expose_psi", dist.Normal(torch.zeros(K), torch.ones(K)).to_event(1)
        )

        mu_x_n_gene = (
            mu_g0 * Vindex(data.gene_cns_tensor_T)[z_n] * k_g[:, 0]
            + mu_g1 * 2 * k_g[:, 1]
        ) * torch.exp(torch.matmul(psi_n, torch.transpose(w_g, 0, 1)))

        pyro.sample(
            "x_gene",
            dist.Multinomial(
                total_count=1, probs=mu_x_n_gene.clamp(min=epsilon), validate_args=False
            ),
            obs=data.gene_mat_tensor_T,
        )

        mu_d_n_gene = (
            Vindex(data.ccopies_tensor_T)[z_n] * torch.dot(k_g[:, 0], mu_g0)
            + 2 * torch.dot(k_g[:, 1], mu_g1)
        ) * torch.exp(torch.matmul(psi_n, torch.transpose(ave_w_g, 0, 1)))

        pyro.sample(
            "d_gene",
            dist.Multinomial(
                total_count=1, probs=mu_d_n_gene.clamp(min=epsilon), validate_args=False
            ),
            obs=data.gex_tcounts_tensor_T,
        )

        ##################################################
        # TODO do we marginalize the phasing? not necessary
        # bp_s = (
        #     Vindex(baf_s)[:, z_n] * h_s[:, 0] + Vindex(aaf_s)[:, z_n] * h_s[:, 1]
        # )  # (ncells, nbins)
        bp_s_gene = Vindex(baf_s_gene)[:, z_n]  # (ncells, nbins)
        probs_s_gene = torch.stack(
            [1 - bp_s_gene, bp_s_gene], dim=-1
        )  # (ncells, nbins, 2)
        obs_s_gene = torch.stack(
            [data.gex_acounts_tensor_T, data.gex_bcounts_tensor_T], dim=-1
        )  # (ncells, nbins, 2)
        pyro.sample(
            "y_gene",
            dist.Multinomial(
                total_count=1, probs=probs_s_gene.clamp(min=epsilon), validate_args=False
            ).to_event(1),
            obs=obs_s_gene,
        )
    return

##################################################
@config_enumerate
def sc_model_multiome(data: SC_Data, temperature=0.5):
    assert data.has_gex and data.has_atac
    K = 6  # structural noise latent dim
    chi_alpha = 2
    chi_rate = 1
    pk_betas = torch.tensor([1, 1]).float()
    # h_betas = torch.tensor([1, 1]).float() # marginal phasing prior
    sigma = 10
    dirichlet_alpha = 1
    softplus = Softplus()

    epsilon = 1e-4  # avoid log(0)
    temp_tensor = torch.tensor(temperature)

    ##################################################
    chi = pyro.sample(
        "expose_chi",
        dist.Gamma(torch.ones(K) * chi_alpha, torch.ones(K) * chi_rate).to_event(1),
    )

    ##################################################
    k_g, w_g, mu_g0, mu_g1, ave_w_g = sample_gene(
        data, chi, pk_betas, temp_tensor, sigma, K, softplus
    )

    ##################################################
    k_p, w_p, mu_p0, mu_p1, ave_w_p = sample_peak(
        data, chi, pk_betas, temp_tensor, sigma, K, softplus
    )

    ##################################################
    plate_cell = pyro.plate("CELL", data.num_cells)
    with plate_cell:
        pi = pyro.sample(
            "expose_pi",
            dist.Dirichlet(torch.ones(data.num_clones) * dirichlet_alpha),
        )
        z_n = pyro.sample("z", dist.Categorical(pi))

        psi_n = pyro.sample(
            "expose_psi", dist.Normal(torch.zeros(K), torch.ones(K)).to_event(1)
        )

        mu_x_n_gene = (
            mu_g0 * Vindex(data.gene_cns_tensor_T)[z_n] * k_g[:, 0]
            + mu_g1 * 2 * k_g[:, 1]
        ) * torch.exp(torch.matmul(psi_n, torch.transpose(w_g, 0, 1)))

        pyro.sample(
            "x_gene",
            dist.Multinomial(
                total_count=1, probs=mu_x_n_gene.clamp(min=epsilon), validate_args=False
            ),
            obs=data.gene_mat_tensor_T,
        )

        mu_d_n_gene = (
            Vindex(data.ccopies_tensor_T)[z_n] * torch.dot(k_g[:, 0], mu_g0)
            + 2 * torch.dot(k_g[:, 1], mu_g1)
        ) * torch.exp(torch.matmul(psi_n, torch.transpose(ave_w_g, 0, 1)))

        pyro.sample(
            "d_gene",
            dist.Multinomial(
                total_count=1, probs=mu_d_n_gene.clamp(min=epsilon), validate_args=False
            ),
            obs=data.gex_tcounts_tensor_T,
        )

        mu_x_n_peak = (
            mu_p0 * Vindex(data.peak_cns_tensor_T)[z_n] * k_p[:, 0]
            + mu_p1 * 2 * k_p[:, 1]
        ) * torch.exp(torch.matmul(psi_n, torch.transpose(w_p, 0, 1)))
        # mu_x_n_peak = (mu_p0 * Vindex(data.peak_cns_tensor_T)[z_n]) * torch.exp(
        #     torch.matmul(psi_n, torch.transpose(w_p, 0, 1))
        # )
        pyro.sample(
            "x_peak",
            dist.Multinomial(total_count=1, probs=mu_x_n_peak.clamp(min=epsilon), validate_args=False),
            obs=data.peak_mat_tensor_T,
        )

        mu_d_n_peak = (
            Vindex(data.ccopies_tensor_T)[z_n] * torch.dot(k_p[:, 0], mu_p0)
            + 2 * torch.dot(k_p[:, 1], mu_p1)
        ) * torch.exp(torch.matmul(psi_n, torch.transpose(ave_w_p, 0, 1)))

        pyro.sample(
            "d_peak",
            dist.Multinomial(total_count=1, probs=mu_d_n_peak.clamp(min=epsilon), validate_args=False),
            obs=data.atac_tcounts_tensor_T,
        )

    ##################################################
    with pyro.plate("SEGMENT", data.num_bins) as bin_id:
        # TODO do we marginalize the phasing? not necessary
        # ph_s = pyro.sample("expose_ph", dist.Dirichlet(h_betas))
        # h_s = pyro.sample(
        #     "expose_h",
        #     dist.RelaxedOneHotCategorical(
        #         temperature=torch.tensor(temperature), probs=ph_s
        #     ),
        # )
        gene_idx = data.padded_bin2gene_ids[bin_id]
        gene_idx = gene_idx[gene_idx != -1]
        bcopy_s_gene = Vindex(data.bcopies_tensor)[bin_id] * torch.sum(
            k_g[gene_idx, 0]
        ) + 1 * torch.sum(k_g[gene_idx, 1])
        ccopy_s_gene = Vindex(data.ccopies_tensor)[bin_id] * torch.sum(
            k_g[gene_idx, 0]
        ) + 2 * torch.sum(k_g[gene_idx, 1])
        baf_s_gene = bcopy_s_gene / ccopy_s_gene  # (nbins, nclones)
        baf_s_gene = baf_s_gene.clamp(min=epsilon)

        peak_idx = data.padded_bin2peak_ids[bin_id]
        peak_idx = peak_idx[peak_idx != -1]
        bcopy_s_peak = Vindex(data.bcopies_tensor)[bin_id] * torch.sum(
            k_p[peak_idx, 0]
        ) + 1 * torch.sum(k_p[peak_idx, 1])
        ccopy_s_peak = Vindex(data.ccopies_tensor)[bin_id] * torch.sum(
            k_p[peak_idx, 0]
        ) + 2 * torch.sum(k_p[peak_idx, 1])
        baf_s_peak = bcopy_s_peak / ccopy_s_peak  # (nbins, nclones)
        baf_s_peak = baf_s_peak.clamp(min=epsilon)

    ##################################################
    with plate_cell:
        # TODO do we marginalize the phasing? not necessary
        # bp_s = (
        #     Vindex(baf_s)[:, z_n] * h_s[:, 0] + Vindex(aaf_s)[:, z_n] * h_s[:, 1]
        # )  # (ncells, nbins)
        bp_s_gene = Vindex(baf_s_gene)[:, z_n]  # (ncells, nbins)
        probs_s_gene = torch.stack(
            [1 - bp_s_gene, bp_s_gene], dim=-1
        )  # (ncells, nbins, 2)
        obs_s_gene = torch.stack(
            [data.gex_acounts_tensor_T, data.gex_bcounts_tensor_T], dim=-1
        )  # (ncells, nbins, 2)
        pyro.sample(
            "y_gene",
            dist.Multinomial(
                total_count=1, probs=probs_s_gene.clamp(min=epsilon), validate_args=False
            ).to_event(1),
            obs=obs_s_gene,
        )

        bp_s_peak = Vindex(data.bafs_tensor)[:, z_n]  # (ncells, nbins)
        probs_s_peak = torch.stack(
            [1 - bp_s_peak, bp_s_peak], dim=-1
        )  # (ncells, nbins, 2)
        obs_s_peak = torch.stack(
            [data.atac_acounts_tensor_T, data.atac_bcounts_tensor_T], dim=-1
        )  # (ncells, nbins, 2)
        pyro.sample(
            "y_peak",
            dist.Multinomial(
                total_count=1, probs=probs_s_peak.clamp(min=epsilon), validate_args=False
            ).to_event(1),
            obs=obs_s_peak,
        )
    return
