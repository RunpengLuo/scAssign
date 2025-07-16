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


class SC_Params:
    def __init__(
        self, use_betabinom=True, fix_tau=False, dirichlet_alpha=1, epsilon=1e-4
    ) -> None:
        self.use_betabinom = use_betabinom
        self.fix_tau = fix_tau

        # dirichlet input parameter
        self.dirichlet_alpha = dirichlet_alpha
        # avoid log(0), numerical stability constant
        self.epsilon = epsilon

    def update(self):
        pass

    def __str__(self) -> str:
        ret = "====================parameters====================\n"
        for k, v in vars(self).items():
            ret += f"{k}: {v}\n"
        return ret


@config_enumerate
def allele_model(sc_data: SC_Data, sc_params: SC_Params):
    """
    allele-specific model, scRNAseq + scATACseq
    baf modelling only, noise free
    """
    dirichlet_prior = torch.ones(sc_data.num_clones) * sc_params.dirichlet_alpha
    if sc_params.use_betabinom:
        if sc_params.fix_tau:
            tau_gex = torch.tensor(1.0)
            tau_atac = torch.tensor(1.0)
        else:
            # scalar tau shared across all data points
            if sc_data.has_gex:
                tau_gex = pyro.param(
                    "tau", torch.tensor(1.0), constraint=dist.constraints.positive
                )
            if sc_data.has_atac:
                tau_atac = pyro.param(
                    "tau", torch.tensor(1.0), constraint=dist.constraints.positive
                )

    cell_plate = pyro.plate("CELL", sc_data.num_cells)
    with cell_plate:
        pi = pyro.sample(
            "expose_pi",
            dist.Dirichlet(dirichlet_prior),
        )
        z_n = pyro.sample("z", dist.Categorical(pi), infer={"enumerate": "parallel"})
    if sc_data.has_gex:
        if sc_params.use_betabinom:
            with cell_plate:
                bafs_n_gex = Vindex(sc_data.gex_bafs_tensor)[:, z_n]  # (ncells, nbins)
                alpha = tau_gex * bafs_n_gex
                beta = tau_gex * (1 - bafs_n_gex)
                log_probs_gex = beta_binomial_log_prob(
                    sc_data.gex_bcounts_tensor_T,
                    sc_data.gex_tcounts_tensor_T,
                    alpha,
                    beta,
                ).sum(dim=-1)
                pyro.factor("Y_gene_BBin", log_probs_gex)
        else:
            with cell_plate:
                bafs_n_gex = Vindex(sc_data.gex_bafs_tensor)[:, z_n]  # (ncells, nbins)
                log_probs_gex = binomial_log_prob(
                    sc_data.gex_bcounts_tensor_T,
                    sc_data.gex_tcounts_tensor_T,
                    bafs_n_gex
                ).sum(dim=-1)
                pyro.factor("Y_gene_Bin", log_probs_gex)
        
    if sc_data.has_atac:
        if sc_params.use_betabinom:
            with cell_plate:
                bafs_n_atac = Vindex(sc_data.atac_bafs_tensor)[
                    :, z_n
                ]  # (ncells, nbins)
                alpha = tau_atac * bafs_n_atac
                beta = tau_atac * (1 - bafs_n_atac)
                log_probs_atac = beta_binomial_log_prob(
                        sc_data.atac_bcounts_tensor_T,
                        sc_data.atac_tcounts_tensor_T,
                        alpha,
                        beta,
                    ).sum(dim=-1)
                pyro.factor("Y_atac_BBin", log_probs_atac)
        else:
            with cell_plate:
                bafs_n_atac = Vindex(sc_data.atac_bafs_tensor)[:, z_n]  # (ncells, nbins)
                log_probs_atac = binomial_log_prob(
                    sc_data.atac_bcounts_tensor_T,
                    sc_data.atac_tcounts_tensor_T,
                    bafs_n_atac
                ).sum(dim=-1)
                pyro.factor("Y_atac_Bin", log_probs_atac)
    return


# @config_enumerate
# def allele_model_multiome(sc_data: SC_Data, temp=None):
#     """
#     allele-specific model, scRNAseq + scATACseq
#     """
#     K = 6  # structural noise latent dim
#     chi_alpha = 2
#     chi_alpha = torch.ones(K) * chi_alpha
#     chi_rate = 1
#     chi_rate = torch.ones(K) * chi_rate
#     # pk_betas = torch.tensor([1, 1]).float()
#     # h_betas = torch.tensor([1, 1]).float() # marginal phasing prior
#     sigma = 10
#     dirichlet_alpha = 1
#     softplus = Softplus()

#     epsilon = 1e-4  # avoid log(0)
#     # temp_tensor = torch.tensor(temperature)

#     ##################################################
#     chi_gene = pyro.sample(
#         "expose_chi_gene",
#         dist.Gamma(chi_alpha, chi_rate).to_event(1),
#     )
#     std_gene = 1.0 / torch.sqrt(chi_gene.clamp(min=epsilon))
#     mu_gene = torch.zeros(K)
#     with pyro.plate("GEX", sc_data.gex_num_bins):
#         w_g = pyro.sample(
#             "expose_w_gene",
#             dist.Normal(mu_gene, std_gene).to_event(1),
#         )

#     chi_atac = pyro.sample(
#         "expose_chi_atac",
#         dist.Gamma(chi_alpha, chi_rate).to_event(1),
#     )
#     std_atac = 1.0 / torch.sqrt(chi_atac.clamp(min=epsilon))
#     mu_atac = torch.zeros(K)
#     with pyro.plate("ATAC", sc_data.atac_num_bins):
#         w_p = pyro.sample(
#             "expose_w_peak",
#             dist.Normal(mu_atac, std_atac).to_event(1),
#         )

#     max_gex_tcounts = int(sc_data.gex_tcounts_tensor_T.sum(dim=1).max().item())
#     max_atac_tcounts = int(sc_data.atac_tcounts_tensor_T.sum(dim=1).max().item())

#     mu_psi = torch.zeros(K)
#     std_psi = torch.ones(K)
#     dirichlet_prior = torch.ones(sc_data.num_clones) * dirichlet_alpha
#     with pyro.plate("CELL", sc_data.num_cells):
#         pi = pyro.sample(
#             "expose_pi",
#             dist.Dirichlet(dirichlet_prior),
#         )
#         z_n = pyro.sample("z", dist.Categorical(pi), infer={"enumerate": "parallel"})
#         psi_n = pyro.sample(
#             "expose_psi", dist.Normal(mu_psi, std_psi).to_event(1)
#         )

#         mu_d_n_gene = ((
#             Vindex(sc_data.gex_ccopies_tensor_T)[z_n]
#         ) * torch.exp(torch.matmul(psi_n, torch.transpose(w_g, 0, 1)).clamp(max=10))).clamp(min=epsilon)
#         probs_gene = mu_d_n_gene / mu_d_n_gene.sum(dim=1, keepdim=True)
#         pyro.sample(
#             "Y_gene",
#             dist.Multinomial(
#                 total_count=1, probs=probs_n_gex, validate_args=False
#             ).to_event(1),
#             obs=obs_s_gene,
#         )

#         obs_s_atac = torch.stack(
#             [sc_data.atac_acounts_tensor_T, sc_data.atac_bcounts_tensor_T], dim=-1
#         )  # (ncells, nbins, 2)
#         pyro.sample(
#             "Y_atac",
#             dist.Multinomial(
#                 total_count=1, probs=probs_n_atac, validate_args=False
#             ).to_event(1),
#             obs=obs_s_atac,
#         )
