# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from copy import deepcopy

import torch

from sharktank.layers.paged_llama_attention_block import PagedLlamaAttentionBlock
from sharktank.layers.rotary_embedding import RotaryEmbeddingLayer

from sharktank.models.deepseek.toy_deepseek import generate
from sharktank.models.llm import PagedLlmModelV1
from sharktank.types.theta import Theta, flat_to_nested_dict
from sharktank.types.sharding import shard_theta, LatentAttentionBlockSharding
from sharktank import ops
from sharktank.utils.evaluate import pad_tokens
from sharktank.utils.load_llm import TorchGenerator


class DeepseekShardedTest(unittest.TestCase):
    def test_deepseek(self):
        theta, config = generate(12345)
        theta = theta("blk", 0)

        sharding = 2
        spec = LatentAttentionBlockSharding(sharding)
        sharding_keys = {k for k in spec.theta_sharding().keys()}
        flattened = theta.flatten()

        t = {}
        for k in flattened:
            if k.split(".")[0] in sharding_keys:
                t[k] = flattened[k]
        theta = Theta(flat_to_nested_dict(t))

        sharded_config = deepcopy(config)
        sharded_config.tensor_parallelism_size = sharding

        sharded_theta = shard_theta(theta, sharded_config)

        hp = config.hp
        reference_model = PagedLlamaAttentionBlock(
            theta=theta,
            block_index=0,
            cache=None,
            head_count=hp.attention_head_count,
            head_dim=hp.attn_head_dim,
            head_count_kv=hp.attention_head_count_kv,
            rms_epsilon=hp.attention_layer_norm_rms_epsilon,
            rope_dimension_count=hp.rope_dimension_count,
            v_head_dim=hp.v_head_dim,
            model_arch=hp.model_arch,
        )

        sharded_model = PagedLlamaAttentionBlock(
            theta=sharded_theta,
            block_index=0,
            cache=None,
            head_count=hp.attention_head_count,
            head_dim=hp.attn_head_dim,
            head_count_kv=hp.attention_head_count_kv,
            rms_epsilon=hp.attention_layer_norm_rms_epsilon,
            rope_dimension_count=hp.rope_dimension_count,
            v_head_dim=hp.v_head_dim,
            model_arch=hp.model_arch,
        )

        bs = 1
        seq = 11
        embed = hp.embedding_length
        input = torch.rand((bs, seq, embed))

        embedding = RotaryEmbeddingLayer(
            rope_dimension_count=hp.rope_dimension_count,
            rope_freq_base=hp.rope_freq_base,
            max_seqlen=hp.context_length,
        )

        sharded_embedding = RotaryEmbeddingLayer(
            rope_dimension_count=hp.rope_dimension_count,
            rope_freq_base=hp.rope_freq_base,
            max_seqlen=hp.context_length,
            tensor_parallelism_size=sharding,
        )

        reference = reference_model.forward(embedding=embedding, h=input)
        sharded = sharded_model.forward(
            embedding=sharded_embedding, h=ops.replicate(input, count=sharding)
        )
        sharded = ops.unshard(sharded)
        assert torch.isclose(reference, sharded, atol=1e-5).all()

    def testTensorParallelToySizedModelEagerVsUnsharded(self):
        theta, config = generate(12345)
        tensor_parallelism_size = 2

        sharded_config = deepcopy(config)
        sharded_config.tensor_parallelism_size = tensor_parallelism_size

        sharded_theta = shard_theta(theta, sharded_config)

        reference_model = PagedLlmModelV1(theta=theta, config=config)
        target_model = PagedLlmModelV1(theta=sharded_theta, config=sharded_config)

        ids = [[3, 22, 13, 114, 90, 232, 61, 13, 244, 13, 212]]
        token_ids, seq_lens = pad_tokens(
            token_ids=ids,
            pad_to_multiple_of=config.block_seq_stride,
        )
        token_ids = torch.as_tensor(token_ids)
        seq_lens = torch.as_tensor(seq_lens)

        reference_generator = TorchGenerator(reference_model)
        reference_batch = reference_generator.begin_batch(
            token_ids=token_ids,
            seq_lens=seq_lens,
        )
        reference_batch.prefill()
        reference_logits = reference_batch.prefill_logits

        target_generator = TorchGenerator(target_model)
        target_batch = target_generator.begin_batch(
            token_ids=token_ids,
            seq_lens=seq_lens,
        )
        target_batch.prefill()
        target_logits = target_batch.prefill_logits

        torch.testing.assert_close(
            target_logits, reference_logits, atol=2e-4, rtol=2e-2
        )
