from __future__ import annotations

import torch
from mpi4py import MPI


class BodoDistributedSampler(torch.utils.data.Sampler):
    """
    A distributed sampler that works with Bodo's torch_train.
    It expects each worker to have it's own slice of the global dataset.
    It ensures that all workers have the same number of samples by padding
    the indices with duplicates if necessary.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        worker_ranks: list[int],
        shuffle=True,
        seed=0,
    ):
        """
        Args:
            dataset (torch.utils.data.Dataset): The dataset to sample from.
            worker_ranks (list[int]): The ranks of the workers that will be sampling.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
            seed (int, optional): The seed for shuffling. Defaults to 0.
        """
        self.dataset = dataset
        self.worker_ranks = worker_ranks
        self.shuffle = shuffle

        # Setup a subcomm of worker ranks for communicating
        # the number of samples on each worker rank
        world_group = MPI.COMM_WORLD.Get_group()
        self.worker_group = world_group.Incl(worker_ranks)
        world_group.Free()
        self.worker_subcomm = MPI.COMM_WORLD.Create(self.worker_group)
        self.seed = seed
        if self.worker_subcomm != MPI.COMM_NULL:
            self.max_sample_len = self.worker_subcomm.allreduce(
                len(self.dataset), op=MPI.MAX
            )
            self.worker_group.Free()
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
        indices += indices[: (self.max_sample_len - len(indices))]
        return iter(indices)

    def __len__(self):
        return self.max_sample_len
