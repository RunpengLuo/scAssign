import numpy as np
import torch

def inverse_softplus(x):
    '''
    inverse the softplus function
    :param x: number matrix (torch.tensor)
    :return: number matrix (torch.tensor)
    '''
    return x + torch.log(-torch.expm1(-x))
