# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import torch
import unittest
from parameterized import parameterized

from sharktank.utils.testing import assert_tensor_close
from sharktank.types import ReplicatedTensor
from sharktank.utils.attention import *


class AttentionMaskTest(unittest.TestCase):
    pass


class AttentionMaskForDecodeTest(unittest.TestCase):
    pass


class BooleanChunkedAttentionMaskTest(unittest.TestCase):
    pass


class CausalContextMaskTest(unittest.TestCase):
    pass


class ChunkedAttentionMaskTest(unittest.TestCase):
    pass


class InputMaskTest(unittest.TestCase):
    def test_base_version(self):
        batch_seqlen = 5
        seq_lens = torch.tensor([1, 2, 3, 4, 5])
        expected_result = torch.tensor(
            [
                [0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
        ).to(torch.bool)
        actual_result = create_input_mask(seq_lens, batch_seqlen)
        assert torch.all(actual_result == expected_result)

    @parameterized.expand([(1,), (2,), (8,)])
    def test_replicated_version(self, shard_count: int):
        batch_seqlen = 10
        _seq_lens = torch.randint(1, batch_seqlen, (8,))
        seq_lens = ReplicatedTensor(ts=_seq_lens, shard_count=shard_count)

        expected_mask = create_input_mask(_seq_lens, batch_seqlen)
        assert isinstance(expected_mask, torch.Tensor)

        actual_mask = create_input_mask(seq_lens, batch_seqlen)
        assert isinstance(actual_mask, ReplicatedTensor)

        for shard in actual_mask.shards:
            assert_tensor_close(expected_mask, shard, atol=0, rtol=0)
