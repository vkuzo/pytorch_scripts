"""
Debugging the output of PyTorch profiler for torch.compile GPU events
"""

import torch
import torch.nn as nn


def run():
    # simple toy model
    M, K = 1024, 2048
    m = nn.Sequential(nn.LayerNorm(M), nn.Linear(M, K)).cuda()
    m = torch.compile(m)
    x = torch.randn(8192, M, device="cuda")

    # warmup
    y = m(x)
    y.sum().backward()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        y = m(x)
        y.sum().backward()

    key_averages = p.key_averages()

    # reference of all aggregated events
    print(key_averages.table())

    # manually filter top-level CPU events with attributed CUDA time
    # example CPU event row:
    #                                               aten::addmm         0.83%      76.554us         0.98%      90.846us      90.846us       1.022ms        31.82%       1.022ms       1.022ms             1
    # and it maps to this CUDA event:
    #   sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize256x64...         0.00%       0.000us         0.00%       0.000us       0.000us       1.022ms        31.82%       1.022ms       1.022ms             1
    thresh = 1e-10
    cpu_events_with_cuda_time = [
        e
        for e in key_averages
        if e.self_cpu_time_total > thresh and e.self_device_time_total > thresh
    ]
    total_cuda_time = 0
    print("cpu events with cuda time")
    for e in cpu_events_with_cuda_time:
        print(e)
        total_cuda_time += e.self_device_time_total
    # the row below should match the CUDA time from the original table
    print("total_cuda_time", total_cuda_time)

    # dump logs for additional debugging - feel free to ignore
    trace_path = "/home/vasiliy/local/tmp/debug_trace.json"
    p.export_chrome_trace(trace_path)

    event_log = "/home/vasiliy/local/tmp/debug_events.txt"
    with open(event_log, "w") as f:
        for e in p.events():
            # print(e)
            f.write(str(e) + "\n")

    print("done")


if __name__ == "__main__":
    run()
