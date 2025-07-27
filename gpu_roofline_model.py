# chooper1/PerformanceModeling/tree/aditya-updated
import math


def get_gpu_info(gpu_name, precision):
    if gpu_name == "A6000":
        # specs are at: https://www.techpowerup.com/gpu-specs/rtx-a6000.c3686

        # clock speeds and memory bw
        dram_bw = 384
        clock_freq = 1.41 * 1E9
        mem_clock_freq = 2 * 1E9

        # number of SMs, number of tensor and cuda cores
        num_sm = 84
        cuda_cores_per_sm = 128
        tensor_cores_per_sm = 4

        # 3rd generation tensor cores
        # 128 for input_prec=32 [8,4,4], 256 for input_prec=16 [8,8,4]
        if precision == 16:
            tensor_core_macs_per_cycle = 256
        elif precision == 32:
            tensor_core_macs_per_cycle = 128
        else:
            assert (False)  # not implemented

    elif gpu_name == "A100":
        # specs are at: https://www.techpowerup.com/gpu-specs/a100-pcie-80-gb.c3821

        # clock speeds and memory bw
        dram_bw = 5120
        clock_freq = 1.065 * 1E9
        mem_clock_freq = 1.512 * 1E9

        # number of SMs, number of tensor and cuda cores
        num_sm = 108
        cuda_cores_per_sm = 64
        tensor_cores_per_sm = 4

        # 3rd generation tensor cores
        # 128 for input_prec=32 [8,4,4], 256 for input_prec=16 [8,8,4]
        if precision == 16:
            tensor_core_macs_per_cycle = 256
        elif precision == 32:
            tensor_core_macs_per_cycle = 128
        else:
            assert (False)  # not implemented

    else:
        assert (False)  # not implemented

    return dram_bw, clock_freq, mem_clock_freq, num_sm, cuda_cores_per_sm, tensor_cores_per_sm, tensor_core_macs_per_cycle


# initial GPU model for matmuls
def compute_mmm_runtime_gpu(
    A,
    B,
    gpu_name="A6000",  # options: A6000, A100
    precision=16,
    cuda_start_latency=0,
    use_tcores=True,
    noloadA=False,
    noloadB=False,
    nowrite=False,
    causal=False,
):
    dram_bw, clock_freq, mem_clock_freq, num_sm, cuda_cores_per_sm, tensor_cores_per_sm, tensor_core_macs_per_cycle = get_gpu_info(gpu_name, precision)

    N0 = A[0]
    N1 = A[1]
    N2 = B[1]
    assert (A[1] == B[0])

    N0_size = N0 * N1 * precision / 8
    N1_size = N1 * N2 * precision / 8
    out_size = N0 * N2 * precision / 8

    if causal:
        macs = (N0 * N1 * N2)
        min_dim = min(N0, N2)
        macs_masked = N1 * ((min_dim * (min_dim - 1)) // 2)
        macs -= macs_masked
    else:
        macs = (N0 * N1 * N2)

    if use_tcores:
        peak_flops = num_sm * tensor_cores_per_sm * tensor_core_macs_per_cycle
        mac_cycles = macs / peak_flops
    else:
        peak_flops = num_sm * cuda_cores_per_sm * 1  # 1 max per cycle per cuda core
        mac_cycles = macs / peak_flops

    # compute bw requirements
    bytes_loaded_dram_to_l2 = N0_size + N1_size
    bytes_out_l2_to_dram = out_size

    # compute bw requirements
    assert (not noloadB)  # not implemented
    if noloadA and nowrite:
        total_mops = N1_size
    elif nowrite:
        total_mops = bytes_loaded_dram_to_l2
    elif noloadA:
        total_mops = bytes_out_l2_to_dram + N1_size
    else:
        total_mops = bytes_out_l2_to_dram + bytes_loaded_dram_to_l2
    mops_cycles = total_mops / dram_bw

    # if operator fusion, report only compute runtime
    if total_mops == 0:
        compute_runtime = mac_cycles / clock_freq
        runtime = compute_runtime
        compute_bound = True
    else:
        # use arithmetic intensity + compute / memory runtime to determine whether compute or memory bound
        arithmetic_intensity = macs / total_mops
        memory_runtime = mops_cycles / mem_clock_freq
        compute_runtime = mac_cycles / clock_freq

        if (arithmetic_intensity * dram_bw * (mem_clock_freq / clock_freq) < peak_flops):
            runtime = memory_runtime
            compute_bound = False
        else:
            runtime = compute_runtime
            compute_bound = True

    runtime += cuda_start_latency

    return runtime, compute_bound, 2 * macs, total_mops


# initial GPU model for nonlinear operations
def compute_nonlinear_runtime_gpu(
    A,
    gpu_name="A6000",  # options: A6000, A100
    precision=16,
    operation='layernorm',
    cuda_start_latency=0,  # 2.5e-6
    noload=False,
    nowrite=False,
    causal=False
):
    dram_bw, clock_freq, mem_clock_freq, num_sm, cuda_cores_per_sm, _, _ = get_gpu_info(gpu_name, precision)

    N0 = A[0]
    N1 = A[1]
    N2 = 1

    N0_size = N0 * N1 * precision / 8
    out_size = N0 * N1 * precision / 8
    if (operation == 'elemadd' or operation == 'elemmul'):  # two operands
        N1_size = N0 * N1 * precision / 8
    elif (operation == 'layernorm'):  # bias term
        N1_size = 2 * N1 * precision / 8
    else:
        N1_size = 0
    input_size = N0_size + N1_size

    # get number of compute cycles
    # for now, assume all are the same (doesn't have a big effect)
    if causal:
        ops = N0 * N1
        min_dim = min(N0, N1)
        ops_masked = (min_dim * (min_dim - 1)) // 2
        ops -= ops_masked
    else:
        ops = N0 * N1

    # get number of compute cycles
    peak_flops = num_sm * cuda_cores_per_sm
    ops_cycles = ops / peak_flops

    # compute bw requirements
    if noload and nowrite:
        total_mops = 0
    elif nowrite:
        total_mops = input_size
    elif noload:
        total_mops = out_size
    else:
        total_mops = input_size + out_size
    mops_cycles = total_mops / dram_bw

    # if operator fusion, report only compute runtime
    if total_mops == 0:
        compute_runtime = ops_cycles / clock_freq
        runtime = compute_runtime
        compute_bound = True
    else:
        # use arithmetic intensity + compute / memory runtime to determine whether compute or memory bound
        arithmetic_intensity = ops / (total_mops)
        memory_runtime = mops_cycles / mem_clock_freq
        compute_runtime = ops_cycles / clock_freq

        if (arithmetic_intensity * dram_bw * (mem_clock_freq / clock_freq) < peak_flops):
            runtime = memory_runtime
            compute_bound = False
        else:
            runtime = compute_runtime
            compute_bound = True

    runtime += cuda_start_latency

    return runtime, compute_bound, ops, total_mops