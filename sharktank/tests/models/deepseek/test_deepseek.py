# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import subprocess
import pytest
import unittest
import iree
from copy import deepcopy

import torch

from sharktank.models.llm import *
from sharktank.models.deepseek.toy_deepseek import generate
from sharktank.utils.create_cache import create_paged_kv_cache
from sharktank.utils.export_artifacts import ExportArtifacts, IreeBenchmarkException
from sharktank.utils.iree import (
    TorchLikeIreeModule,
    get_iree_devices,
    load_iree_module,
    make_hal_buffer_view_trace_default_callback,
    with_iree_device_context,
)
from sharktank.utils.load_llm import *
from sharktank.utils.evaluate import *
from sharktank.utils.testing import TempDirTestBase
from sharktank.utils import debugging
import os


# @pytest.mark.usefixtures("get_iree_flags")
class DeepseekTest(TempDirTestBase):
    def setUp(self):
        super().setUp()
        self.callback_stash = debugging.get_trace_tensor_callback()
        debugging.set_trace_tensor_callback(
            debugging.trace_tensor_to_safetensors_callback
        )

        self.enable_tensor_trace_stash = debugging.flags.enable_tensor_trace
        # debugging.flags.enable_tensor_trace = True

        self.trace_path_stash = debugging.flags.trace_path
        debugging.flags.trace_path = Path(
            "/home/alvasile/repos/shark-ai/sharktank/logits"
        )

    def tearDown(self):
        super().tearDown()
        debugging.set_trace_tensor_callback(self.callback_stash)
        debugging.flags.enable_tensor_trace = self.enable_tensor_trace_stash
        debugging.flags.trace_path = self.trace_path_stash

    def testCrossEntropy(self):
        theta, config = generate(12345)
        model = PagedLlmModelV1(theta=theta, config=config)

        ids = [[3, 22, 13, 114, 90, 232, 61, 13, 244, 13, 212]]

        token_ids, seq_lens = pad_tokens(
            token_ids=ids,
            pad_to_multiple_of=config.block_seq_stride,
        )
        token_ids = torch.as_tensor(token_ids)
        seq_lens = torch.as_tensor(seq_lens)

        generator = TorchGenerator(model)
        batch = generator.begin_batch(
            token_ids=token_ids,
            seq_lens=seq_lens,
        )

        batch.prefill()
        logits = batch.prefill_logits

        ids = token_ids[0, :-1]
        logits = logits[0, 1:]
        cross_entropy = torch.nn.functional.cross_entropy(logits, ids)

        assert pytest.approx(9.7477, 1e-4) == cross_entropy

    def testUnshardedToySizedModelIREEVsEager(self):
        work_dir = Path("/home/alvasile/repos/shark-ai/sharktank/ireevseager")
        theta, config = generate(12345)
        # config.device = "cuda:5"

        token_ids_path = work_dir / "token_ids.npy"
        seq_lens_path = work_dir / "seq_lens.npy"
        seq_block_ids_before_prefill_path = (
            work_dir / "seq_block_ids_before_prefill.npy"
        )
        iree_cache_state_path = work_dir / "iree_cache_state.npy"

        dataset_path = work_dir / "parameters.irpa"
        reference_results_path = work_dir / "results_reference.npy"

        ids = [
            [1, 2, 3, 4],
            [9, 8, 7, 6],
            [3, 5, 2, 1],
        ]
        token_ids, seq_lens = pad_tokens(
            token_ids=ids,
            pad_to_multiple_of=config.block_seq_stride,
        )
        token_ids = torch.as_tensor(token_ids)
        np.save(token_ids_path, token_ids.cpu().numpy())
        token_ids = torch.tensor(np.load(token_ids_path))

        seq_lens = torch.as_tensor(seq_lens)
        batch_size = token_ids.shape[0]

        dataset = Dataset(root_theta=theta, properties=config.to_properties())
        dataset.save(path=dataset_path)
        dataset = Dataset.load(dataset_path)

        reference_model = PagedLlmModelV1(theta=theta, config=config)
        reference_generator = TorchGenerator(reference_model)
        reference_batch = reference_generator.begin_batch(
            token_ids=token_ids,
            seq_lens=seq_lens,
        )
        cache_state_before_prefill = deepcopy(reference_batch.cache_state)
        iree_cache = create_paged_kv_cache(config)
        iree_cache_state = iree_cache.shard_state(deepcopy(cache_state_before_prefill))
        np.save(iree_cache_state_path, iree_cache_state[0].cpu().numpy())
        iree_cache_state = [torch.tensor(np.load(iree_cache_state_path))]

        seq_block_ids_before_prefill = reference_batch.pad_block_ids()
        np.save(
            seq_block_ids_before_prefill_path,
            seq_block_ids_before_prefill.cpu().numpy(),
        )
        seq_block_ids_before_prefill = torch.tensor(
            np.load(seq_block_ids_before_prefill_path)
        )

        reference_batch.prefill()
        result_reference = reference_batch.prefill_logits
        np.save(reference_results_path, result_reference.cpu().numpy())
        result_reference = torch.tensor(np.load(reference_results_path))

        case = "failing"
        mlir_path = work_dir / f"{case}.mlir"
        export_config_path = work_dir / f"{case}_export_config.json"
        export_artifacts = ExportArtifacts.from_config(
            config,
            irpa_path=dataset_path,
            batch_size=batch_size,
            iree_hip_target="gfx942",
            iree_hal_target_device="hip",
            # iree_hal_local_target_device_backends=self.iree_hal_local_target_device_backends,
        )
        export_artifacts.export_to_mlir(
            output_mlir=mlir_path,
            output_config=export_config_path,
            skip_decode=True,  # TODO: enable decode
        )

        iree_module_path = work_dir / f"{case}.vmfb"
        export_artifacts.compile_to_vmfb(
            output_mlir=mlir_path,
            output_vmfb=iree_module_path,
            args=[],
        )

        np.save(seq_lens_path, seq_lens.cpu().numpy())
        seq_lens = torch.tensor(np.load(seq_lens_path))

        run_args = [
            "iree-run-module",
            "--hip_use_streams=true",
            f"--parameters=model={dataset_path}",
            f"--module={iree_module_path}",
            "--device=hip://5",
            f"--function=prefill_bs{batch_size}",
            f"--input=@{token_ids_path}",
            f"--input=@{seq_lens_path}",
            f"--input=@{seq_block_ids_before_prefill_path}",
            f"--input=@{iree_cache_state_path}",
            f"--output=@results_{case}.npy",
        ]
        cmd = subprocess.list2cmdline(run_args)
        print(f" Launching run command:\n" f"cd {work_dir} && {cmd}")
        proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=work_dir)
        return_code = proc.returncode
        if return_code != 0:
            raise IreeBenchmarkException(proc, work_dir)

        result_iree = torch.tensor(np.load(work_dir / f"results_{case}.npy"))

        def f(x1: torch.Tensor, x2: torch.Tensor) -> None:
            diff = torch.abs(x1 - x2)
            i_max = diff.argmax()
            v_max = diff.flatten()[i_max].item()
            print(f"Max absolute diff: {v_max:.5e}")
            base_value = torch.abs(x1.flatten()[i_max]).item()
            if base_value:
                print(f"Max relative diff: {v_max / base_value:.5e}")
            else:
                print("Max relative diff: undefined (division by zero)")

        assert result_iree.shape == result_reference.shape
        attni, hi = result_iree[..., :32], result_iree[..., 32:]
        attnr, hr = result_reference[..., :32], result_reference[..., 32:]

        print("attn_output")
        f(attnr, attni)
        print("hidden_states")
        f(hr, hi)
        pass
