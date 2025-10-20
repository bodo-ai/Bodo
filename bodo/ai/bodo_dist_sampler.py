from __future__ import annotations

import torch

from bodo.mpi4py import MPI

from .pandas_dataset import PandasDataset


class BodoDistributedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: PandasDataset, worker_ranks: list[int], shuffle=True):
        self.dataset = dataset
        self.worker_ranks = worker_ranks
        self.shuffle = shuffle
        world_group = MPI.COMM_WORLD.Get_group()
        self.worker_group = world_group.Incl(worker_ranks)
        world_group.Free()
        self.worker_subcomm = MPI.COMM_WORLD.Create(self.worker_group)
        self.seed = 0

    def __del__(self):
        if hasattr(self, "worker_group") and self.worker_group != MPI.GROUP_NULL:
            self.worker_group.Free()
        if hasattr(self, "worker_subcomm") and self.worker_subcomm != MPI.COMM_NULL:
            self.worker_subcomm.Free()

    def __iter__(self):
        # Create a list of all indices
        indices = list(range(len(self.dataset)))

        # Shuffle the indices if required
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
            self.seed += 1  # Change seed for next epoch

        # Ensure all ranks have the same number of samples
        max_sample_len = self.worker_subcomm.allreduce(len(indices), op=MPI.MAX)
        indices += indices[: (max_sample_len - len(indices))]
        return iter(indices)

    def __len__(self):
        return len(self.dataset)
