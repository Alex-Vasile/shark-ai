from pathlib import Path
import subprocess
import numpy as np
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

work_dir = Path(".").absolute()

token_ids_path = work_dir / "token_ids.npy"
seq_lens_path = work_dir / "seq_lens.npy"
seq_block_ids_before_prefill_path = work_dir / "seq_block_ids_before_prefill.npy"
iree_cache_state_path = work_dir / "iree_cache_state.npy"
results_reference_path = work_dir / "results_reference.npy"
results_failing_path = work_dir / "results_failing.npy"
results_passing_path = work_dir / "results_passing.npy"

# Compile and run both cases
for case in ["failing", "passing"]:
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
    print(f"Compile command:\n" f"cd {work_dir} && {cmd}")
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
        f"--output=@{work_dir / f'results_{case}'}.npy",
    ]
    cmd = subprocess.list2cmdline(run_args)
    print(f"Run command:\n" f"cd {work_dir} && {cmd}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, cwd=work_dir)
    print(proc)


# Load reference results an passing & failing mlir results
split_res = lambda res: (res[..., :32], res[..., 32:])
## .npy stores torch.cat((a, b), dim=-1)
a_ref, b_ref = split_res(torch.tensor(np.load(results_reference_path)))
## .npy stores torch.cat((a, b), dim=-1)
a_fail, b_fail = split_res(torch.tensor(np.load(results_failing_path)))
## .npy stores torch.cat((a, a), dim=-1)
a_pass1, a_pass2 = split_res(torch.tensor(np.load(results_passing_path)))

# Compare results
def f(x1: torch.Tensor, x2: torch.Tensor) -> None:
    diff = torch.abs(x1 - x2)
    i_max = diff.argmax()
    v_max = diff.flatten()[i_max].item()
    print(f"\tMax abs diff: {v_max:.3e}")
    base_value = torch.abs(x1.flatten()[i_max]).item()
    if base_value:
        print(f"\tMax rel diff: {v_max / base_value:.3e}")
    else:
        print("\tMax rel diff: undefined (division by zero)")


print("PASSING: a1")
f(a_ref, a_pass1)
print("PASSING: a2")
f(a_ref, a_pass2)
print("FAILING: a")
f(a_ref, a_fail)
print("FAILING: b")
f(b_ref, b_fail)
