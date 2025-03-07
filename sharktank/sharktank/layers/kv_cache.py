# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Light-weight encapsulations for various forms of attention KV-caches.

These are not complete abstractions: they are primarily focused on making
tightly coupled transformer blocks a bit less "stringy" with loose tensors
and dims floating around everywhere.
"""

from typing import Optional, Tuple, Union, List

import abc
import math

import torch

from ..utils.debugging import trace_tensor
from ..types import SplitPrimitiveTensor, ReplicatedTensor
from .. import ops

__all__ = ["PagedKVCache"]


class PagedKVCache:
    """Implementation of a KV cache on top of a 'page table'.

    The page table slab is physically represented as a 2D tensor:
        [page_count, flattened_dims]

    Each "page" can be thought of as a 6D view onto:

    * transformer block
    * cache partition (K or V cache)
    * block sequence stride (number of sequence positions per block)
    * attention heads
    * attention dimensionality

    Note that the internal page structure matches the organization of the
    model, allowing contiguous individual local reads and writes at a sub-block
    granularity if indexing deeply into the structure.

    When `shard_count > 1`, it would split the `attn_head_count` dimension.
    The page slab is a 1D sharded split tensor.
    It is reinterpreted as a 6D tensor, by working around the lack of sharded
    block-cyclic sharded tensor type.
    """

    def __init__(
        self,
        *,
        transformer_block_count: int,
        attn_head_count: int,
        attn_head_dim: int,
        cache_partition_count: int = 2,
        block_seq_stride: int = 16,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        shard_count: int = 1,
        pipeline_to_device_lookup: Tuple[Tuple[int, ...], ...] = None,
        block_to_pipeline_lookup: Tuple[int, ...] = None,
    ):
        self.transformer_block_count = transformer_block_count
        self.attn_head_count = attn_head_count
        self.attn_head_dim = attn_head_dim
        self.cache_partition_count = cache_partition_count
        self.block_seq_stride = block_seq_stride
        self.shard_count = shard_count
        if pipeline_to_device_lookup is None:
            pipeline_to_device_lookup = (tuple(range(self.shard_count)),)
        self.pipeline_to_device_lookup = pipeline_to_device_lookup
        if block_to_pipeline_lookup is None:
            block_to_pipeline_lookup = tuple(0 for _ in range(transformer_block_count))
        assert all(table >= 0 for table in block_to_pipeline_lookup)
        self.block_to_pipeline_lookup = block_to_pipeline_lookup
        self.pipeline_count = len(pipeline_to_device_lookup)
        # TODO: Ensure that block_to_table_lookup and pipeline_count are consistent

        if attn_head_count % shard_count != 0:
            raise ValueError(
                f"The attention head count {attn_head_count} must be a multiple of the tensor parallelism size {shard_count}."
            )

        self.pipeline_to_block_count = tuple(
            sum(1 for block in block_to_pipeline_lookup if block == i)
            for i in range(self.pipeline_count)
        )
        # Some derived values based on attributes.
        self.sub_page_dims = [
            [
                self.pipeline_to_block_count[pipeline],
                self.cache_partition_count,
                self.block_seq_stride,
                self.attn_head_count // self.shard_count,
                self.attn_head_dim,
            ]
            for pipeline in range(self.pipeline_count)
        ]
        self.page_slab_flat_dims = [
            math.prod(sub_page_dim) for sub_page_dim in self.sub_page_dims
        ]
        self.device = device
        self.dtype = dtype

    def unflatten_page_tables(
        self, state: list[Union[torch.Tensor, SplitPrimitiveTensor]]
    ) -> list[Union[torch.Tensor, SplitPrimitiveTensor]]:
        """Unflattens the 2D page tables to 6D tensors."""
        assert (
            len(state) == self.pipeline_count
        ), f"Expected {self.pipeline_count}-element state. Got: {len(state)}"
        if self.shard_count == 1:
            assert all(
                not isinstance(page_slab, SplitPrimitiveTensor) for page_slab in state
            )
            return [
                page_slab.unflatten(1, self.sub_page_dims[pipeline])
                for pipeline, page_slab in enumerate(state)
            ]
        else:
            assert all(page_slab.shard_count == self.shard_count for page_slab in state)
            unflattened = []
            for pipeline, page_slab in enumerate(state):
                shards = [
                    shard.unflatten(1, self.sub_page_dims[pipeline])
                    for shard in page_slab.shards
                ]
                unflattened.append(
                    SplitPrimitiveTensor(
                        ts=shards,
                        shard_dim=4,
                        devices=page_slab.devices,
                        pinned=page_slab.pinned,
                    )
                )
            return unflattened

    def shard_state(
        self, state: List[torch.Tensor]
    ) -> List[Union[torch.Tensor, SplitPrimitiveTensor]]:
        """Shard an unsharded state.
        We can't just split the slab on the sub page dims.
        First it needs to be reinterpreted into the actual shape.
        The split the head dimension, then flatten each shard.
        This is a work-around for the lack of block-cyclic sharded tensor type."""
        if self.shard_count == 1:
            return state

        page_table = state[0].reshape(
            [
                -1,
                self.transformer_block_count,
                self.cache_partition_count,
                self.block_seq_stride,
                self.attn_head_count,
                self.attn_head_dim,
            ]
        )

        flat_sharded_page_tables = []
        for pipeline in range(self.pipeline_count):
            # TODO: Do I need to make copies here, or are views enough?
            # TODO: Handle 1 tensor per pipeline case. Currently that dim gets collapsed.
            i_min = sum(self.pipeline_to_block_count[:pipeline])
            i_max = i_min + self.pipeline_to_block_count[pipeline]
            sharded_page_table = ops.reshard_split(
                page_table[:, i_min:i_max, ...],
                dim=4,
                count=self.shard_count,
            )
            shards_flattened = [
                ops.flatten(shard, start_dim=1) for shard in sharded_page_table.shards
            ]
            flat_sharded_page_tables.append(
                SplitPrimitiveTensor(
                    ts=shards_flattened,
                    shard_dim=1,
                    devices=self.pipeline_to_device_lookup[pipeline],
                    pinned=True,
                )
            )
        return flat_sharded_page_tables

    @property
    def pad_sequence_stride(self) -> int:
        return self.block_seq_stride

    def allocate(
        self, page_count: int
    ) -> list[Union[torch.Tensor, SplitPrimitiveTensor]]:
        """Allocates tensor state for a page table for the given capacity in
        pages.
        """
        shards = [
            [
                torch.empty(
                    [page_count, self.page_slab_flat_dims[pipeline]],
                    dtype=self.dtype,
                    device=self.device,
                )
                for _ in range(self.shard_count)
            ]
            for pipeline in range(self.pipeline_count)
        ]

        if self.shard_count == 1:
            assert self.pipeline_count == 1
            return shards[0]

        return [
            SplitPrimitiveTensor(
                ts=shards[i], shard_dim=1, devices=devices, pinned=True
            )
            for i, devices in enumerate(self.pipeline_to_device_lookup)
        ]

    def read(
        self,
        state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        *,
        transformer_block_index: int,
        seq_len: int,
        page_ids: Optional[Union[torch.Tensor, ReplicatedTensor]] = None,
    ):
        """Reads K/V caches the page table for the given page_ids.

        Args:
        state: State struct as returned from allocate().
        transformer_block_index: The index of the transformer block accessing
            the cache.
        page_ids: Tensor of [bs, max_seqlen // block_pos_stride] of page ids
            to access.

        Returns the K/V cache partitions, linearized. Note that this reference
        approach to reading by materializing linearly may not be terribly
        efficient unless if the compiler can fuse the gather.
        """
        page_tables = self.unflatten_page_tables(state)  # 6D
        page_table = page_tables[self.block_to_pipeline_lookup[transformer_block_index]]

        bs, block_seq_len, *_ = page_ids.shape
        # Blocks dim 1,2 according to the configured block stride.
        blocked_shape = [
            bs,
            block_seq_len,
            self.cache_partition_count,
            self.block_seq_stride,
            self.attn_head_count // self.shard_count,
            self.attn_head_dim,
        ]

        # Gather both partitions and split post gather. This is more
        # computationally efficient without gather fusion:
        subblock_table = page_table.flatten(start_dim=0, end_dim=1)
        page_stride = self.pipeline_to_block_count[
            self.block_to_pipeline_lookup[transformer_block_index]
        ]

        transformer_block_index = torch.full(
            (bs, block_seq_len), transformer_block_index
        )
        subblock_ids = page_ids * page_stride + transformer_block_index
        selected = ops.index_select(subblock_table, 0, subblock_ids.flatten(0, 1))

        selected = selected.unflatten(0, blocked_shape[:2])
        key = selected[:, :, 0, :seq_len].flatten(1, 2)[:, :seq_len]
        value = selected[:, :, 1, :seq_len].flatten(1, 2)[:, :seq_len]

        return key, value

    def write_timestep(
        self,
        state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        # List of [bs, 1, attn_head_count, attn_head_dim]
        cache_partitions: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        *,
        transformer_block_index: int,
        # [bs]
        seq_positions: Union[torch.Tensor, ReplicatedTensor],
        # [bs, max_seqlen // block_pos_stride]
        page_ids: Union[torch.Tensor, ReplicatedTensor],
    ):
        """Writes a single batched timestep across all cache partitions.

        Note that this internally loops over the batch size, which cannot be
        dynamic.
        """
        device = self.device
        page_tables = self.unflatten_page_tables(state)  # 6D
        page_table = page_tables[self.block_to_pipeline_lookup[transformer_block_index]]
        page_table = page_table.flatten(0, 3)
        bs, *_ = seq_positions.shape
        assert len(cache_partitions) == self.cache_partition_count

        # [bs, 1, atten_head_count, attn_head_dim]
        for idx, cache_partition in enumerate(cache_partitions):
            # [bs, 1]
            page_index = seq_positions // self.block_seq_stride

            page_id = ops.gather(page_ids, dim=1, index=page_index.unsqueeze(1))
            page_offset = (seq_positions % self.block_seq_stride).unsqueeze(1)

            # [1, 1]
            if isinstance(seq_positions, ReplicatedTensor):
                partitions = [
                    torch.tensor(idx).unsqueeze(0)
                    for _ in range(seq_positions.shard_count)
                ]

                transformer_block = [
                    torch.full((bs, 1), transformer_block_index, device=device)
                    for _ in range(seq_positions.shard_count)
                ]

                partitions = ReplicatedTensor(ts=partitions)
                transformer_block = ReplicatedTensor(ts=transformer_block)
            else:
                partitions = torch.tensor(idx).unsqueeze(0)
                transformer_block = torch.full(
                    (bs, 1), transformer_block_index, device=device
                )

            partitions = partitions.repeat(bs, 1)

            transformer_block_count_in_pipeline = self.pipeline_to_block_count[
                self.block_to_pipeline_lookup[transformer_block_index]
            ]

            index = page_id
            index = index * transformer_block_count_in_pipeline + transformer_block
            index = index * self.cache_partition_count + partitions
            index = index * self.block_seq_stride + page_offset
            values = ops.to(cache_partition, dtype=page_table.dtype)
            if page_table.dtype == torch.float8_e4m3fnuz:
                # Workaround for Torch not supporting torch.Tensor.index_copy_ for f8.
                page_table_as_int8 = page_table.view(dtype=torch.int8)
                values_int8 = values.view(dtype=torch.int8)
                page_table_as_int8.index_put_(indices=(index,), values=values_int8)
            else:
                page_table.index_put_(indices=(index,), values=values)

        return

    def write(
        self,
        state: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        cache_partitions: list[Union[torch.Tensor, SplitPrimitiveTensor]],
        *,
        transformer_block_index: int,
        page_ids: Union[torch.Tensor, ReplicatedTensor],
    ):
        """Writes cache partitions from a linear layout to the page table.

        This is the inverse of the linear read. The same caveat applies if the
        in-place scatter cannot be fused.
        """
        page_tables = self.unflatten_page_tables(state)  # 6D
        page_table = page_tables[self.block_to_pipeline_lookup[transformer_block_index]]
        bs, block_seq_len, *_ = page_ids.shape

        # Reshape the page cache into sub-blocks so that we can index at the
        # granularity of the transformer_block and cache partition.
        # This requires us to recompute indices to the sub-block reference
        # frame.
        # The subblock slab is organized as:
        #   [page, attn_layer, cache_partition]
        # Where the cache line can be 0 (k) or 1 (v).
        subblock_table = page_table.flatten(start_dim=0, end_dim=2)
        transformer_block_count_in_pipeline = self.pipeline_to_block_count[
            self.block_to_pipeline_lookup[transformer_block_index]
        ]
        page_stride = transformer_block_count_in_pipeline * self.cache_partition_count
        transformer_block_stride = self.cache_partition_count
        base_subblock_ids = page_ids * page_stride + (
            transformer_block_index * transformer_block_stride
        )

        for index, partition in enumerate(cache_partitions):
            part_block_view = partition.unflatten(
                1, (block_seq_len, self.block_seq_stride)
            )
            part_block_view = part_block_view.flatten(0, 1)

            subblock_ids = (
                (base_subblock_ids + index) if index > 0 else base_subblock_ids
            ).flatten(0, 1)

            part_block = ops.to(part_block_view, dtype=subblock_table.dtype)
            if subblock_table.dtype == torch.float8_e4m3fnuz:
                # Workaround for Torch not supporting torch.Tensor.index_copy_ for f8.
                subblock_table_as_int8 = subblock_table.view(dtype=torch.int8)
                part_block_as_int8 = part_block.view(dtype=torch.int8)
                subblock_table_as_int8.index_copy_(0, subblock_ids, part_block_as_int8)
            else:
                subblock_table.index_copy_(0, subblock_ids, part_block)
