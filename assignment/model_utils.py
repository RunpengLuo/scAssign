import numpy as np
import torch


def inverse_softplus(x):
    """
    inverse the softplus function
    :param x: number matrix (torch.tensor)
    :return: number matrix (torch.tensor)
    """
    return x + torch.log(-torch.expm1(-x))

def beta_binomial_log_prob(y, n, alpha, beta):
    return (
        torch.lgamma(n + 1)
        - torch.lgamma(y + 1)
        - torch.lgamma(n - y + 1)
        + torch.lgamma(y + alpha)
        + torch.lgamma(n - y + beta)
        - torch.lgamma(n + alpha + beta)
        + torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
    )

def binomial_log_prob(y, n, p):
    return (
        torch.lgamma(n + 1)
        - torch.lgamma(y + 1)
        - torch.lgamma(n - y + 1)
        + y * torch.log(p)
        + (n - y) * torch.log(1 - p)
    )
