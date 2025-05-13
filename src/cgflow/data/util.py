import math
import random

import torch.distributed
from torch.utils.data import BatchSampler, RandomSampler, Sampler


class BucketBatchSampler(BatchSampler):
    def __init__(
        self,
        bucket_limits: list[int],
        lengths: list[int],
        batch_cost: float,
        bucket_costs: list[float] | None = None,
        drop_last: bool = True,
        round_batch_to_8: bool = False,
    ):

        # Modern GPUs can be more efficient when data is provided as a multiple of 8 (for 16-bit training)
        self.round_batch_to_8 = round_batch_to_8
        self.drop_last = drop_last

        if bucket_costs is not None and len(bucket_costs) != len(bucket_limits):
            raise ValueError("The number of costs and buckets must be the same.")

        if max(lengths) > max(bucket_limits):
            raise ValueError("Largest length cannot be larger than largest bucket limit.")

        bucket_limits = sorted(bucket_limits)

        # Use a constant bucket cost by default
        if bucket_costs is None:
            bucket_costs = [1.0] * len(bucket_limits)

        # Add indices to correct bucket based on seq length
        buckets = [[] for _ in range(len(bucket_limits))]
        for data_idx, length in enumerate(lengths):
            for b_idx, limit in enumerate(bucket_limits):
                if limit >= length:
                    buckets[b_idx].append(data_idx)
                    break

        # TODO allow non-shuffled sampling
        samplers = [RandomSampler(idxs, replacement=False) if len(idxs) > 0 else None for idxs in buckets]
        bucket_batch_sizes = [self._round_batch_size(batch_cost / cost) for cost in bucket_costs]

        batches_per_bucket = []
        for bucket, batch_size in zip(buckets, bucket_batch_sizes, strict=False):
            n_batches = int(len(bucket) // batch_size)
            if not drop_last and n_batches * batch_size != len(bucket):
                n_batches += 1

            batches_per_bucket.append(n_batches)

        print()
        print("items per bucket", [len(idxs) for idxs in buckets])
        print("bucket batch sizes", bucket_batch_sizes)
        print("batches per bucket", batches_per_bucket)

        self.buckets = buckets
        self.samplers = samplers
        self.bucket_batch_sizes = bucket_batch_sizes
        self.batches_per_bucket = batches_per_bucket

    def __len__(self):
        return sum(self.batches_per_bucket)

    def __iter__(self):
        iters = [iter(sampler) if sampler is not None else None for sampler in self.samplers]
        remaining_batches = self.batches_per_bucket[:]
        remaining_items = [len(items) for items in self.buckets]

        while sum(remaining_batches) > 0:
            b_idx = random.choices(range(len(remaining_batches)), weights=remaining_batches, k=1)[0]
            if remaining_batches[b_idx] > 1 or self.drop_last:
                batch_size = self.bucket_batch_sizes[b_idx]
            else:
                batch_size = remaining_items[b_idx]

            batch_idxs = [next(iters[b_idx]) for _ in range(batch_size)]

            # Samplers will produce indices into the list, so look up dataset indices using sampled bucket indices
            batch = [self.buckets[b_idx][idx] for idx in batch_idxs]

            remaining_batches[b_idx] -= 1
            remaining_items[b_idx] -= batch_size

            yield batch

    def _round_batch_size(self, batch_size):
        if not self.round_batch_to_8:
            bs = math.floor(batch_size)
        else:
            bs = 8 * round(batch_size / 8)

        bs = 1 if bs == 0 else bs
        return bs


class _PartialBucketSampler(BatchSampler):
    def __init__(
        self,
        buckets: list[list[int]],
        bucket_batch_sizes: list[int],
        batches_per_bucket: list[int],
        drop_last: bool = True,
    ):

        self.drop_last = drop_last

        samplers = [RandomSampler(idxs, replacement=False) for idxs in buckets]
        self.buckets = buckets
        self.samplers = samplers
        self.bucket_batch_sizes = bucket_batch_sizes
        self.batches_per_bucket = batches_per_bucket

        print()
        print("items per bucket", [len(idxs) for idxs in buckets])
        print("bucket batch sizes", bucket_batch_sizes)
        print("batches per bucket", batches_per_bucket)

    def __len__(self):
        return sum(self.batches_per_bucket)

    def __iter__(self):
        iters = [iter(sampler) for sampler in self.samplers]
        remaining_batches = self.batches_per_bucket[:]
        remaining_items = [len(items) for items in self.buckets]

        while sum(remaining_batches) > 0:
            b_idx: int = random.choices(range(len(remaining_batches)), weights=remaining_batches, k=1)[0]
            if remaining_batches[b_idx] > 1 or self.drop_last:
                batch_size = self.bucket_batch_sizes[b_idx]
            else:
                batch_size = remaining_items[b_idx]
            batch_idxs = [next(iters[b_idx]) for _ in range(batch_size)]

            # Samplers will produce indices into the list, so look up dataset indices using sampled bucket indices
            batch = [self.buckets[b_idx][idx] for idx in batch_idxs]

            remaining_batches[b_idx] -= 1
            remaining_items[b_idx] -= batch_size

            yield batch


class BucketBatchSampler_DDP(Sampler):
    def __init__(
        self,
        bucket_limits: list[int],
        lengths: list[int],
        batch_cost: float,
        bucket_costs: list[float] | None = None,
        drop_last: bool = True,
        round_batch_to_8: bool = False,
        num_replicas: int | None = None,
        rank: int | None = None,
    ):
        if rank is None:
            rank = torch.distributed.get_rank()
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()

        # Modern GPUs can be more efficient when data is provided as a multiple of 8 (for 16-bit training)
        self.round_batch_to_8 = round_batch_to_8
        if bucket_costs is not None and len(bucket_costs) != len(bucket_limits):
            raise ValueError("The number of costs and buckets must be the same.")
        if max(lengths) > max(bucket_limits):
            raise ValueError("Largest length cannot be larger than largest bucket limit.")

        bucket_limits = sorted(bucket_limits)
        if bucket_costs is None:
            bucket_costs = [1.0] * len(bucket_limits)

        # Add indices to correct bucket based on seq length
        buckets = [[] for _ in range(len(bucket_limits))]
        for data_idx, length in enumerate(lengths):
            for b_idx, limit in enumerate(bucket_limits):
                if limit >= length:
                    buckets[b_idx].append(data_idx)
                    break
        buckets = [b for b in buckets if len(b) > 0]

        # distributed parallel
        num_buckets = len(buckets)
        bucket_per_rank = num_buckets / num_replicas
        buckets = buckets[int(rank * bucket_per_rank) : int((rank + 1) * bucket_per_rank)]
        bucket_costs = bucket_costs[int(rank * bucket_per_rank) : int((rank + 1) * bucket_per_rank)]

        # add status
        bucket_batch_sizes = [self._round_batch_size(batch_cost / cost) for cost in bucket_costs]
        batches_per_bucket = []
        for bucket, batch_size in zip(buckets, bucket_batch_sizes, strict=True):
            n_batches = int(len(bucket) // batch_size)
            if not drop_last and n_batches * batch_size != len(bucket):
                n_batches += 1
            batches_per_bucket.append(n_batches)

        self.sampler = _PartialBucketSampler(buckets, bucket_batch_sizes, batches_per_bucket, drop_last)

    def __len__(self):
        return sum(self.sampler.batches_per_bucket)

    def __iter__(self):
        yield from iter(self.sampler)

    def _round_batch_size(self, batch_size):
        if not self.round_batch_to_8:
            bs = math.floor(batch_size)
        else:
            bs = 8 * round(batch_size / 8)

        bs = 1 if bs == 0 else bs
        return bs


class DistributedBucketBatchSampler(Sampler):
    def __init__(
        self,
        bucket_limits: list[int],
        lengths: list[int],
        batch_cost: float,
        bucket_costs: list[float] | None = None,
        drop_last: bool = True,
        round_batch_to_8: bool = False,
        num_replicas: int | None = None,
        rank: int | None = None,
    ):
        if rank is None:
            rank = torch.distributed.get_rank()
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()
        self.rank = rank
        self.num_replicas = num_replicas

        self.sampler = BucketBatchSampler(bucket_limits, lengths, batch_cost, bucket_costs, drop_last, round_batch_to_8)

    def __len__(self):
        return sum(self.sampler.batches_per_bucket)

    def __iter__(self):
        for batch in self.sampler:
            # distribute batch
            nbatch = len(batch) / self.num_replicas
            minibatch = batch[int(self.rank * nbatch) : int((self.rank + 1) * nbatch)]
            if len(minibatch) == 0:  # prevent empty batch
                minibatch = [batch[0]]
            yield minibatch
