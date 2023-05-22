# from https://github.com/FreyrS/dMaSIF

import numpy as np
import torch
from pykeops.torch import LazyTensor

def initialize(device='cpu', seed=None):
    
    import pykeops
    import os
    import warnings

    warnings.filterwarnings("ignore")

    # Ensure reproducibility:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    if seed!=None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    torch.set_num_threads(4)
    if device!='cpu':
        if seed!=None:
            torch.cuda.manual_seed_all(seed)

    try:
        pykeops.set_bin_folder(f'.cache/pykeops-1.5-cpython-37/{os.uname().nodename}/')
    except AttributeError:
        pykeops.set_build_folder(f'.cache/keops2.1/{os.uname().nodename}/build/')

def pass_function():
    pass 

default_device=f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
inttensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
synchronize = torch.cuda.synchronize if torch.cuda.is_available() else pass_function

def numpy(x): 
    return x.detach().cpu().numpy()


def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    Ns = batch.bincount()
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = (
        torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
    )
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)

    return ranges, slices


def diagonal_ranges(batch_x=None, batch_y=None):
    """Encodes the block-diagonal structure associated to a batch vector."""

    if batch_x is None and batch_y is None:
        return None  # No batch processing
    elif batch_y is None:
        batch_y = batch_x  # "symmetric" case

    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x


def soft_dimension(features):
    """Continuous approximation of the rank of a (N, D) sample.

    Let "s" denote the (D,) vector of eigenvalues of Cov,
    the (D, D) covariance matrix of the sample "features".
    Then,
        R(features) = \sum_i sqrt(s_i) / \max_i sqrt(s_i)

    This quantity encodes the number of PCA components that would be
    required to describe the sample with a good precision.
    It is equal to D if the sample is isotropic, but is generally much lower.

    Up to the re-normalization by the largest eigenvalue,
    this continuous pseudo-rank is equal to the nuclear norm of the sample.
    """

    nfeat = features.shape[-1]
    features = features.view(-1, nfeat)
    x = features - torch.mean(features, dim=0, keepdim=True)
    cov = x.T @ x
    try:
        u, s, v = torch.svd(cov)
        R = s.sqrt().sum() / s.sqrt().max()
    except:
        return -1
    return R.item()
