# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import pytest
import torch
import unittest
from parameterized import parameterized

from sharktank.utils.testing import assert_tensor_close
from sharktank.types import ReplicatedTensor
from sharktank.utils.attention import *


class AttentionMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)

    @parameterized.expand([(None,), (torch.tensor([0, 1, 2, 3, 4]),)])
    def test_base_version(self, start_positions: torch.Tensor | None = None):
        assert False


class AttentionMaskForDecodeTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)


class BooleanChunkedAttentionMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)

    def test_base_version(self):
        attention_chunk_size = 3
        start_index = 0
        end_index = 6
        expected_result = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1, 1],
            ]
        ).to(torch.bool)
        actual_result = create_boolean_chunked_attention_mask(
            attention_chunk_size, start_index, end_index
        )
        assert torch.all(actual_result == expected_result)


class CausalContextMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)

    @parameterized.expand([(None,), (torch.tensor([0, 1, 2, 3, 4]),)])
    def test_base_version(self, start_positions: torch.Tensor | None = None):
        bs = len(start_positions) if start_positions is not None else 1
        source_len = 6
        target_len = 5
        expected_shape = (bs, 1, target_len, source_len)
        expected_result = torch.zeros(expected_shape, dtype=torch.bool)

        _start_positions = start_positions if start_positions is not None else [0]
        for batch, start_position in enumerate(_start_positions):
            for i in range(start_position, target_len):
                for j in range(source_len):
                    expected_result[batch, :, i, j] = j >= i

        actual_result = create_causal_context_mask(
            source_len, target_len, start_positions
        )
        assert tuple(actual_result.shape) == expected_shape
        assert torch.all(actual_result == expected_result)


class ChunkedAttentionMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)


class InputMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)

    def test_base_version(self):
        batch_seqlen = 5
        seq_lens = torch.tensor([1, 2, 3, 4, 5])
        expected = torch.tensor(
            [
                [0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
        ).to(torch.bool)
        actual = create_input_mask(seq_lens, batch_seqlen)
        assert torch.all(actual == expected)

    @parameterized.expand([(1,), (2,), (8,)])
    def test_replicated_version(self, shard_count: int):
        batch_seqlen = 10
        _seq_lens = torch.randint(1, batch_seqlen, (8,))
        seq_lens = ReplicatedTensor(ts=_seq_lens, shard_count=shard_count)

        expected = create_input_mask(_seq_lens, batch_seqlen)
        assert isinstance(expected, torch.Tensor)

        actual = create_input_mask(seq_lens, batch_seqlen)
        assert isinstance(actual, ReplicatedTensor)

        for shard in actual.shards:
            assert_tensor_close(expected, shard, atol=0, rtol=0)
