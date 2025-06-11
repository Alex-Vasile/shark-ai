import subprocess
import torch
import numpy as np

import torch.nn.functional as F
from iree.turbine.aot import *
from sharktank.layers.base import ThetaLayer
from sharktank import ops
from sharktank.types.tensors import DefaultPrimitiveTensor
from sharktank.types.theta import Theta
from sharktank.utils.export_artifacts import (
    IreeBenchmarkException,
    IreeCompileException,
)


class ScatterLayer(ThetaLayer):
    def generate_random_theta(self):
        return

    def forward(self, attn_output: torch.Tensor) -> torch.Tensor:
        """
        h: (batch_size * sequence_length, input_feature_dim)
        top_experts_index: (batch_size * sequence_length, num_top_experts)
        expert_gate: (batch_size * sequence_length, num_top_experts)
        """
        self.v_head_dim = 128

        # attn_output = attn_output[:, :, :, : self.v_head_dim]

        # attn_output = attn_output.transpose(1, 2)

        # attn_output = attn_output.flatten(2)

        return ops.linear(attn_output, attn_output[..., : attn_output.shape[-1]])


import os

name = "isolated"

cwd = os.path.realpath(os.curdir)
cwd = "/home/alvasile/repos/shark-ai/sharktank/denseffnmoe/"
mlir_path = cwd + f"{name}.mlir"
iree_module_path = cwd + f"{name}.vmfb"
iree_result_path = cwd + "iree_result.npy"

model = ScatterLayer(Theta([DefaultPrimitiveTensor(data=torch.tensor([1]))]))

attn_output_path = cwd + "attn_output.npy"
# b_path = cwd + "b.npy"
attn_output = torch.tensor(np.load(attn_output_path))

# Run locally
eager_result = model.forward(attn_output)
# assert torch.isclose(
# eager_result, expected_output, rtol=1.3e-6, atol=1e-5
# ).all(), "Eager result does not match expected output"


dynamic = torch.export.Dim("dynamic")

dynamic_shapes = {
    "attn_output": {2: dynamic},  # , 1: input_feature_dim},
}

# Run through IREE
fxb = FxProgramsBuilder(model)


@fxb.export_program(
    name="scatter",
    args=(attn_output,),
    dynamic_shapes=dynamic_shapes,
    strict=False,
)
def _(model, attn_output) -> torch.Tensor:
    return model(attn_output)


output = export(fxb, import_symbolic_shape_expressions=True)
output.save_mlir(mlir_path)

compile_args = [
    "iree-compile",
    mlir_path,
    f"-o={iree_module_path}",
    "--iree-hal-target-device=hip",
    "--iree-hip-target=gfx942",
    "--iree-opt-level=O3",
    "--iree-hal-indirect-command-buffers=true",
    "--iree-stream-resource-memory-model=discrete",
    "--iree-hal-memoization=true",
]
cmd = subprocess.list2cmdline(compile_args)
# print(f" Launching compile command:\n" f"cd {cwd} && {cmd}")
proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
return_code = proc.returncode
if return_code != 0:
    raise IreeCompileException(proc, cwd)

# Write run command
run_args = [
    "iree-run-module",
    "--hip_use_streams=true",
    f"--module={iree_module_path}",
    "--device=hip://0",
    "--function=scatter",
    f"--input=@{attn_output_path}",
    f"--output=@{iree_result_path}",
]
cmd = subprocess.list2cmdline(run_args)
# print(f" Launching compile command:\n" f"cd {cwd} && {cmd}")
proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=cwd)
return_code = proc.returncode
if return_code != 0:
    raise IreeBenchmarkException(proc, cwd)

iree_result = torch.tensor(np.load(iree_result_path))
print(torch.isclose(iree_result, eager_result, rtol=1.3e-6, atol=1e-5).all().item())
