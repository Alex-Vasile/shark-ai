from pathlib import Path
import subprocess
import numpy as np
import iree
import torch

"""
Returns come out of the attention block.
Usual operation is
def attn_block(b):
    x = f(b)
    ...
    a = ...(x)
    return a + b

Simplified by removing many of the intermediary steps:
def attn_block(b):
    x = f(b)
    ...
    a = ...(x)
    return torch.cat((a, b), dim=-1)
or
def attn_block(b):
    x = f(b)
    ...
    a = ...(x)
    return torch.cat((a, a), dim=-1)

In both cases, the value we're concatting should not affect the value of `a`.
**The addition also produces errors, but the concat was easier to track down.**
"""

work_dir = Path("ireevseager").absolute()

token_ids_path = work_dir / "token_ids.npy"
seq_lens_path = work_dir / "seq_lens.npy"
seq_block_ids_before_prefill_path = work_dir / "seq_block_ids_before_prefill.npy"
iree_cache_state_path = work_dir / "iree_cache_state.npy"


# Compile and run both cases
for case in ["passing"]:
    compile_args = [
        "iree-compile",
        f"{case}.mlir",
        f"-o={case}.vmfb",
        "--iree-hal-target-device=hip",
        "--iree-hip-target=gfx942",
        "--iree-opt-level=O3",
        "--iree-hal-indirect-command-buffers=true",
        "--iree-stream-resource-memory-model=discrete",
        "--iree-hal-memoization=true",
    ]
    cmd = subprocess.list2cmdline(compile_args)
    print(f" Launching run command:\n" f"cd {work_dir} && {cmd}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=work_dir)

    run_args = [
        "iree-run-module",
        "--hip_use_streams=true",
        f"--parameters=model=parameters.irpa",
        f"--module={case}.vmfb",
        "--device=hip://5",
        f"--function=prefill_bs3",
        f"--input=@{token_ids_path}",
        f"--input=@{seq_lens_path}",
        f"--input=@{seq_block_ids_before_prefill_path}",
        f"--input=@{iree_cache_state_path}",
        f"--output=@results_{case}.npy",
    ]
    cmd = subprocess.list2cmdline(run_args)
    print(f" Launching run command:\n" f"cd {work_dir} && {cmd}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=work_dir)


# Load reference results an passing & failing mlir results
def f(x1: torch.Tensor, x2: torch.Tensor) -> None:
    diff = torch.abs(x1 - x2)
    i_max = diff.argmax()
    v_max = diff.flatten()[i_max].item()
    print(f"\tMax absolute diff: {v_max:.5e}")
    base_value = torch.abs(x1.flatten()[i_max]).item()
    if base_value:
        print(f"\tMax relative diff: {v_max / base_value:.5e}")
    else:
        print("\tMax relative diff: undefined (division by zero)")


split_res = lambda res: (res[..., :32], res[..., 32:])
## .npy stores torch.cat((a, b), dim=-1)
a_ref, b_ref = split_res(torch.tensor(np.load(work_dir / "results_reference.npy")))
## .npy stores torch.cat((a, b), dim=-1)
a_fail, b_fail = split_res(torch.tensor(np.load(work_dir / "results_failing.npy")))
## .npy stores torch.cat((a, a), dim=-1)
a_pass1, a_pass2 = split_res(torch.tensor(np.load(work_dir / "results_passing.npy")))

print("Reference results a1")
f(a_ref, a_pass1)
print("Reference results a2")
f(a_ref, a_pass2)
print("Failing results a")
f(a_ref, a_fail)
print("Failing results b")
f(b_ref, b_fail)
# Compare results
## Passing case
torch.testing.assert_close(a_ref, a_pass1, rtol=0, atol=1e-3)
torch.testing.assert_close(a_ref, a_pass2, rtol=0, atol=1e-3)

## Failing case
torch.testing.assert_close(b_ref, b_fail, rtol=0, atol=1e-3)  # This one passes
torch.testing.assert_close(a_ref, a_fail, rtol=0, atol=1e-3)  # This one fails
