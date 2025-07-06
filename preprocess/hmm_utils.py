import numpy as np
import pandas as pd
from scipy.stats import binom
from scipy.special import betaln, gammaln, logsumexp, psi
from scipy.optimize import minimize
from scipy.special import softmax


def build_sitewise_transmat(bin_snps: pd.DataFrame, hairs: np.ndarray, 
                            alpha=1, beta=1, norm=True, log=True):
    """
    compute per-site transition probability, CM from population phasing and hair from read supports
    Each weighted by alpha and beta, respectively.
    
    GT is haplotype 1 binary string from population phasing
    CM is absolute genetic distance
    """
    nsnps = len(bin_snps)
    # if nsnps == 0:
    #     return None
    # assert "GT" in bin_snps.columns
    # assert "CM" in bin_snps.columns

    sitewise_transmat = np.zeros((nsnps, 4), dtype=np.float128)
    if not hairs is None:
        assert nsnps == len(hairs)
        sitewise_transmat += hairs * beta

    # confidence of population-phasing
    # cms = bin_snps.CM.to_numpy()
    # cms_diff = cms[:-1] - cms[1:]
    # switch_probs = 0.5 * (1 - np.exp(-2 * cms_diff))
    # conf_probs = 1 - switch_probs

    # gts = bin_snps.GT.to_numpy().astype(np.int8)
    # adj_gts = np.column_stack((gts[:-1], gts[1:]))
    # # 00->0, 01->1, 10->2, 11->3
    # adj_idx = 2 * adj_gts[:, 0] + adj_gts[:, 1]
    # adj_idx_neg = 2 * adj_gts[:, 0] - adj_gts[:, 1] + 1
    # sitewise_transmat[np.arange(nsnps)[1:], adj_idx] += conf_probs
    # sitewise_transmat[np.arange(nsnps)[1:], adj_idx_neg] += switch_probs

    # # 00->3, 01->2, 10->1, 11->0-----
    # adj_idx_rev = 3 - (2 * adj_gts[:, 0] + adj_gts[:, 1])
    # sitewise_transmat[np.arange(nsnps)[1:], adj_idx_rev] += conf_probs

    sitewise_transmat[sitewise_transmat == 0] = 1  # Avoid division by zero
    # print(sitewise_transmat[:10])
    if norm:
        sitewise_transmat = sitewise_transmat / sitewise_transmat.sum(axis=1, keepdims=True)
    if log:
        sitewise_transmat = np.log(sitewise_transmat)
    return sitewise_transmat


##########################################################################
def compute_log_alpha_beta(
    nobs: int,
    log_emissions: np.ndarray,
    log_transmat: np.ndarray,
    log_startprob: np.ndarray,
):
    log_alpha = np.zeros((nobs, 2), dtype=np.float128)  # alpha(z_n)
    log_alpha[0] = log_emissions[0] + log_startprob
    for obs in range(1, nobs):
        log_alpha[obs, 0] = log_emissions[obs, 0] + logsumexp(
            log_alpha[obs - 1] + log_transmat[obs, [0, 2]]
        )  # 00, 10
        log_alpha[obs, 1] = log_emissions[obs, 1] + logsumexp(
            log_alpha[obs - 1] + log_transmat[obs, [1, 3]]
        )  # 01, 11

    log_beta = np.zeros((nobs, 2), dtype=np.float128)  # beta(z_n)
    log_beta[-1] = 0
    for obs in reversed(range(nobs - 1)):
        log_beta[obs, 0] = logsumexp(
            log_beta[obs + 1] + log_transmat[obs, [0, 1]] + log_emissions[obs + 1]
        )  # 00, 01
        log_beta[obs, 1] = logsumexp(
            log_beta[obs + 1] + log_transmat[obs, [2, 3]] + log_emissions[obs + 1]
        )  # 10, 11
    return log_alpha, log_beta


def compute_log_gamma(log_alpha: np.ndarray, log_beta: np.ndarray):
    """
    compute gamma(z_n) = alpha(z_n)beta(z_n) / p(data)
    """
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    return log_gamma


def compute_log_xi(
    nobs: int,
    log_emissions: np.ndarray,
    log_transmat: np.ndarray,
    log_alpha: np.ndarray,
    log_beta: np.ndarray,
):
    """
    compute xi(z_n-1, z_n) = alpha(z_n-1) emission(x_n,z_n) transition(z_n-1, z_n) beta(z_n) / p(data)
    """
    log_xi = np.zeros((nobs, 4), dtype=np.float128)
    log_xi += log_transmat
    # 00, 01, 10, 11
    for obs in range(1, nobs):
        log_xi[obs, 0] = (
            log_alpha[obs - 1, 0] + log_beta[obs, 0] + log_emissions[obs, 0]
        )
        log_xi[obs, 1] = (
            log_alpha[obs - 1, 0] + log_beta[obs, 1] + log_emissions[obs, 1]
        )
        log_xi[obs, 2] = (
            log_alpha[obs - 1, 1] + log_beta[obs, 0] + log_emissions[obs, 0]
        )
        log_xi[obs, 3] = (
            log_alpha[obs - 1, 1] + log_beta[obs, 1] + log_emissions[obs, 1]
        )
    log_xi -= logsumexp(log_xi, axis=1, keepdims=True)
    return log_xi


def binom_hmm(
    nobs: int,
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    init_p: float,
    log_transmat: np.ndarray,
    fix_p=True,
    fix_transmat=True,
    max_iter=20,
    tol=10e-6,
):
    """
    hmm to infer latent phasing states
    """
    totals_sum = np.sum(D)
    p = init_p
    new_p = p
    ll = -np.inf
    log_startprob = np.log([0.5, 0.5])
    log_emissions = np.zeros((nobs, 2), dtype=np.float128)
    gamma = np.zeros((nobs, 2), dtype=np.float128)

    for iteration in range(max_iter):
        p = np.float64(p)
        # log emission
        log_emissions[:, 0] = binom.logpmf(X, D, p)
        log_emissions[:, 1] = binom.logpmf(X, D, 1 - p)

        # E-step
        log_alpha, log_beta = compute_log_alpha_beta(
            nobs, log_emissions, log_transmat, log_startprob
        )
        log_gamma = compute_log_gamma(log_alpha, log_beta)
        gamma[:, :] = np.exp(log_gamma) # phasing information

        # log_xi = compute_log_xi(nobs, log_emissions, log_transmat, log_alpha, log_beta)
        # xi = np.exp(log_xi)

        # M-step: optimize p
        if not fix_p:
            t1 = np.dot(X, gamma[:, 0]) + np.dot(Y, gamma[:, 1])
            new_p = t1 / totals_sum
        if abs(new_p - p) < tol:
            break
        p = new_p
    logp = np.log(p)
    logp_ = np.log(1-p)
    ll = np.dot(X, gamma[:, 0]) * logp + np.dot(Y, gamma[:, 0]) * logp_
    ll += np.dot(Y, gamma[:, 1]) * logp + np.dot(X, gamma[:, 1]) * logp_
    return p, gamma[:, 0], ll


##########################################################################
def viterbi_decode(hairs: np.ndarray, emissions_beta: np.ndarray, eta=1.0, beta=1.0):
    """
    refs, alts, allele counts
    hairs denote adjacent allele #supporting reads
    """
    nobs = emissions_beta.shape[0]

    # first row for prob of useref, second row for usealt
    emissions_alpha = 1 - emissions_beta
    emissions = np.row_stack([emissions_alpha, emissions_beta])

    prev = np.zeros((2, nobs), dtype=np.uint8)
    viterbi = np.zeros((2, nobs), dtype=np.float64)
    # viterbi[s,m]=v_m(s)
    viterbi[0, 0] = eta * (emissions[0, 0])
    viterbi[1, 0] = eta * (emissions[1, 0])  # USEREF

    for t in range(1, nobs):
        hair = hairs[t]  # 00 01 10 11
        # v_t(0) = max(v_t_(0) + h(0, 0) + e(x, 0), v_t_(1) + h(1, 0) + e(x, 0))
        vt00 = viterbi[0, t - 1] + beta * hair[0] + eta * emissions[0, t]
        vt10 = viterbi[1, t - 1] + beta * hair[2] + eta * emissions[0, t]
        if vt00 > vt10:
            viterbi[0, t] = vt00
            prev[0, t] = 0
        else:
            viterbi[0, t] = vt10
            prev[0, t] = 1

        # v_t(1) = max(v_t_(0) + h(0, 1) + e(x, 1), v_t_(1) + h(1, 1) + e(x, 1))
        vt01 = viterbi[0, t - 1] + beta * hair[1] + eta * emissions[1, t]
        vt11 = viterbi[1, t - 1] + beta * hair[3] + eta * emissions[1, t]
        if vt01 > vt11:
            viterbi[1, t] = vt01
            prev[1, t] = 0
        else:
            viterbi[1, t] = vt11
            prev[1, t] = 1

    path = np.zeros(nobs, dtype=np.uint8)
    if viterbi[0, -1] > viterbi[1, -1]:
        path[-1] = 0
    else:
        path[-1] = 1
    for t in range(nobs - 2, -1, -1):
        path[t] = prev[path[t + 1], t + 1]
    return path

def one_posterior(alpha: float, beta: float, log_aaf: float, log_baf: float):
    a = log_baf * beta + log_aaf * alpha
    b = log_aaf * beta + log_baf * alpha
    useref = softmax([a, b])[0]
    return useref

def baf_posterior(refs: np.ndarray, alts: np.ndarray, log_aaf: float, log_baf: float):
    # compute posterior of phasing given mhBAF
    nsnps = len(refs)
    e_arr = np.zeros((2, nsnps), dtype=np.float128)
    e_arr[0] = log_baf * refs + log_aaf * alts
    e_arr[1] = log_aaf * refs + log_baf * alts
    useref_frac = softmax(e_arr, axis=0)[0, :]
    return useref_frac

def revert_path(
    refs: np.ndarray, alts: np.ndarray, path: np.ndarray, log_aaf: float, log_baf: float
):
    beta = np.round(np.sum(path * refs + (1 - path) * alts)).astype(np.int32)
    alpha = np.sum(refs) + np.sum(alts) - beta
    useref = one_posterior(alpha, beta, log_aaf, log_baf)
    if useref < 0.5:
        path = 1 - path
    return path
