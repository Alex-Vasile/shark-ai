import iree
from iree import runtime as iree_runtime
import gc
import time

vmfb_path = "deepseek_v3_f16_torch_pp8.vmfb"
irpa_path = "/shark-dev/weights/deepseek_v3/fp16/deepseek_v3_f16.irpa"
vm_instance = iree_runtime.VmInstance()


def run_iree_module() -> None:
    start = time.time()
    devices = [iree.runtime.get_device(f"hip://{i}") for i in range(8)]
    hal_module = iree.runtime.create_hal_module(instance=vm_instance, devices=devices)
    end = time.time()
    print(f"Start: {end - start:.2f} seconds")

    start = end
    parameter_index = iree.runtime.ParameterIndex()
    parameter_index.load(file_path=irpa_path)
    parameter_provider = parameter_index.create_provider(scope="model")
    parameters_module = iree.runtime.create_io_parameters_module(
        vm_instance, parameter_provider
    )
    end = time.time()
    print(f"Parameters: {end - start:.2f} seconds")

    start = end
    vm_module = iree.runtime.VmModule.mmap(vm_instance, vmfb_path)
    end = time.time()
    print(f"VM Module: {end - start:.2f} seconds")

    start = end
    iree.runtime.VmContext(
        instance=vm_instance,
        modules=(hal_module, parameters_module, vm_module),
    )
    end = time.time()
    print(f"Create VM Context: {end - start:.2f} seconds")

    start = end
    gc.collect()
    print(f"gc.collect(): {time.time() - start:.2f} seconds")


start = time.time()
run_iree_module()
runtime = time.time() - start
print(f"Runtime: {runtime:.2f} seconds")
