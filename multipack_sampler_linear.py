from typing import Optional, List
import math

import torch.distributed as dist
from torch.utils.data import Sampler

import numpy as np
import numba


@numba.njit
def ffd_check(work: np.ndarray, a: np.ndarray, c: int, n: int, nbits: int):
    # First-fit-decreasing bin packing
    # Check if a[] could fit in n bins with capacity c
    # https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing

    a = np.sort(a)[::-1]

    valid_size = (1 << nbits) + n
    work[:valid_size] = c
    for size in a:
        # Find the branch
        u = 1
        for i in range(nbits):
            lch = u << 1
            rch = (u << 1) | 1
            if work[lch] >= size:
                u = lch
            else:
                u = rch

        if u >= valid_size or work[u] < size:
            return False

        # Update
        work[u] -= size
        for i in range(nbits - 1):  # Root not needed
            u = u >> 1

            work[u] = max(work[u << 1], work[(u << 1) | 1])

    return True


@numba.njit
def ffd_with_result(work: np.ndarray, a: np.ndarray, c: int, n: int, nbits: int, start_index: int):
    # First-fit-decreasing bin packing (with result return)
    indices = np.argsort(a)[::-1]
    a = a[indices]

    bins_result = []
    base = 1 << nbits

    valid_size = (1 << nbits) + n
    work[:valid_size] = c
    for idx, size in enumerate(a):
        # Find the branch
        u = 1
        for i in range(nbits):
            lch = u << 1
            rch = (u << 1) | 1
            if work[lch] >= size:
                u = lch
            else:
                u = rch

        bin_id = u - base
        if bin_id >= len(bins_result):
            bins_result.append([start_index + indices[idx]])
        else:
            bins_result[bin_id].append(start_index + indices[idx])

        # Update
        work[u] -= size
        for i in range(nbits - 1):  # Root not needed
            u = u >> 1

            work[u] = max(work[u << 1], work[(u << 1) | 1])

    return bins_result


@numba.njit
def allocate(lengths: np.ndarray, lengths_cumsum: np.ndarray, rank: int, c: int, n: int):
    # Dynamic batch allocator, similar to Multifit
    # https://en.wikipedia.org/wiki/Multifit_algorithm
    # ~99.5% efficiency on OpenChat training set (12 * 2048 ctx len)

    nbits = int(math.ceil(math.log2(n)))
    work = np.zeros((1 << (nbits + 1), ), dtype=lengths.dtype)

    s = 0
    start_index = 0
    result = []

    while True:
        # binary search [l, r)
        l = 1
        r = 1 + np.searchsorted(lengths_cumsum[start_index:], s + c * n, "right")

        while r - l > 1:
            m = (l + r) // 2
            if ffd_check(work, lengths[start_index: start_index + m], c, n, nbits):
                l = m
            else:
                r = m

        # use length l
        batch = ffd_with_result(work, lengths[start_index: start_index + l], c, n, nbits, start_index)
        assert len(batch) <= n
        if len(batch) < n:
            break

        start_index += l
        s = lengths_cumsum[start_index - 1]

        # add local rank
        result.append(batch[rank])

    return result, s, len(result) * c * n


class MultipackDistributedBatchSampler_LinearAttention(Sampler):
    """Unpadded length sampling using Multipack for models with linear attention complexity.
       WARNING: This algorithm might put most long sequences into one node, causing significant lag if attention complexity is quadratic.
       
       Approximate (at most 1.22x ?) the optimal solution of the identical-machines scheduling problem, which is NP-hard.
    """

    def __init__(
        self,
        batch_max_length: int,
        lengths: List[int],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        # Get rank
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.batch_max_length = batch_max_length
        self.lengths = lengths
        assert isinstance(self.lengths, np.ndarray)

        self.epoch = 0

        # statistics
        self.eff_total_used = 0
        self.eff_total_slots = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def generate_batches(self, set_stats=False):
        indices = np.random.default_rng(seed=self.seed + self.epoch).permutation(len(self.lengths))

        lengths = self.lengths[indices]
        lengths_cumsum = np.cumsum(lengths)

        batches, total_used, total_slots = allocate(lengths=lengths,
                                                    lengths_cumsum=lengths_cumsum,
                                                    rank=self.rank,
                                                    c=self.batch_max_length,
                                                    n=self.num_replicas)
        
        batches = [indices[batch] for batch in batches]

        # statistics
        if set_stats:
            self.eff_total_used += total_used
            self.eff_total_slots += total_slots
        
        return batches
    
    def __iter__(self):
        batches = self.generate_batches(set_stats=True)
        return iter(batches)

    def num_batches(self):
        batches = self.generate_batches()
        return len(batches)

    def efficiency(self):
        return self.eff_total_used / self.eff_total_slots
