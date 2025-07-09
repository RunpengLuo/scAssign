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

@config_enumerate
def allele_model_multiome(data: SC_Data, temp=None):
    """
    allele-specific model, scRNAseq + scATACseq 
    """
    K = 6  # structural noise latent dim
    chi_alpha = 2
    chi_alpha = torch.ones(K) * chi_alpha
    chi_rate = 1
    chi_rate = torch.ones(K) * chi_rate
    # pk_betas = torch.tensor([1, 1]).float()
    # h_betas = torch.tensor([1, 1]).float() # marginal phasing prior
    sigma = 10
    dirichlet_alpha = 1
    softplus = Softplus()

    epsilon = 1e-4  # avoid log(0)
    # temp_tensor = torch.tensor(temperature)

    ##################################################
    chi_gene = pyro.sample(
        "expose_chi_gene",
        dist.Gamma(chi_alpha, chi_rate).to_event(1),
    )
    std_gene = 1.0 / torch.sqrt(chi_gene.clamp(min=epsilon))
    mu_gene = torch.zeros(K)
    
    chi_atac = pyro.sample(
        "expose_chi_atac",
        dist.Gamma(chi_alpha, chi_rate).to_event(1),
    )
    std_atac = 1.0 / torch.sqrt(chi_atac.clamp(min=epsilon))
    mu_atac = torch.zeros(K)

    with pyro.plate("SEGMENT", data.num_bins):
        w_g = pyro.sample(
            "expose_w_gene",
            dist.Normal(mu_gene, std_gene).to_event(1),
        )
        w_p = pyro.sample(
            "expose_w_atac",
            dist.Normal(mu_atac, std_atac).to_event(1),
        )

    max_gex_tcounts = int(data.gex_tcounts_tensor_T.sum(dim=1).max().item())
    max_atac_tcounts = int(data.atac_tcounts_tensor_T.sum(dim=1).max().item())

    mu_psi = torch.zeros(K)
    std_psi = torch.ones(K)
    dirichlet_prior = torch.ones(data.num_clones) * dirichlet_alpha
    with pyro.plate("CELL", data.num_cells):
        pi = pyro.sample(
            "expose_pi",
            dist.Dirichlet(dirichlet_prior),
        )
        z_n = pyro.sample("z", dist.Categorical(pi), infer={"enumerate": "parallel"})
        psi_n = pyro.sample(
            "expose_psi", dist.Normal(mu_psi, std_psi).to_event(1)
        )

        mu_d_n_gene = ((
            Vindex(data.ccopies_tensor_T)[z_n]
        ) * torch.exp(torch.matmul(psi_n, torch.transpose(w_g, 0, 1)).clamp(max=10))).clamp(min=epsilon)
        probs_gene = mu_d_n_gene / mu_d_n_gene.sum(dim=1, keepdim=True)
        pyro.sample(
            "D_gene",
            dist.Multinomial(
                total_count=max_gex_tcounts, probs=probs_gene, validate_args=False
            ),
            obs=data.gex_tcounts_tensor_T,
        )

        mu_d_n_atac = ((
            Vindex(data.ccopies_tensor_T)[z_n]
        ) * torch.exp(torch.matmul(psi_n, torch.transpose(w_p, 0, 1)).clamp(max=10))).clamp(min=epsilon)
        probs_atac = mu_d_n_atac / mu_d_n_atac.sum(dim=1, keepdim=True)
        pyro.sample(
            "D_atac",
            dist.Multinomial(
                total_count=max_atac_tcounts, probs=probs_atac, validate_args=False
            ),
            obs=data.atac_tcounts_tensor_T,
        )

        bafs_n = Vindex(data.bafs_tensor)[:, z_n]  # (ncells, nbins)
        probs_n = torch.stack(
            [1 - bafs_n, bafs_n], dim=-1
        ).clamp(min=epsilon)  # (ncells, nbins, 2)
        obs_s_gene = torch.stack(
            [data.gex_acounts_tensor_T, data.gex_bcounts_tensor_T], dim=-1
        )  # (ncells, nbins, 2)
        pyro.sample(
            "Y_gene",
            dist.Multinomial(
                total_count=1, probs=probs_n, validate_args=False
            ).to_event(1),
            obs=obs_s_gene,
        )

        obs_s_atac = torch.stack(
            [data.atac_acounts_tensor_T, data.atac_bcounts_tensor_T], dim=-1
        )  # (ncells, nbins, 2)
        pyro.sample(
            "Y_atac",
            dist.Multinomial(
                total_count=1, probs=probs_n, validate_args=False
            ).to_event(1),
            obs=obs_s_atac,
        )

# def 
