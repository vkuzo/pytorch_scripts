import csv
import scipy
import torch
from dataclasses import dataclass, fields, astuple

from quant_logger import get_default_stats, counter


def get_kurtosis_scipy(x):
    """
    Reference implementation for excess population kurtosis, runs on the CPU only
    """
    k = scipy.stats.kurtosis(
        x.cpu().numpy(),
        fisher=True,  # True → excess kurtosis (0 for Gaussian)
        bias=True,  # Biased estimator
    )
    return k


def get_kurtosis(x, dim=None, keepdim=False):
    """
    PyTorch implementation for excess population kurtosis, can run on GPU

    Source: https://en.wikipedia.org/wiki/Kurtosis
    """
    mean = torch.mean(x, dim=dim, keepdim=keepdim)
    var = torch.var(x, dim=dim, keepdim=keepdim, correction=0)
    mu_4 = torch.mean(torch.pow(x - mean, 4.0), dim=dim, keepdim=keepdim)
    s_4 = torch.pow(var, 2.0)
    eps = 1e-12
    k = (mu_4 / (s_4 + eps)) - 3.0
    return k


@dataclass
class TensorStats:
    max_abs: float
    avg: float
    std: float
    k_r_min: float
    k_r_p50: float
    k_r_max: float
    k_32_min: float
    k_32_p50: float
    k_32_max: float
    k_16_min: float
    k_16_p50: float
    k_16_max: float


def get_default_stats_with_kurtosis(x: torch.Tensor) -> TensorStats:
    max_abs, avg, std = get_default_stats(x)
    # distribution of rowwise kurtosis
    k_r = get_kurtosis(x.reshape(-1, x.shape[-1]), dim=1, keepdim=True)
    k_r_min, k_r_max = torch.aminmax(k_r)
    k_r_p50 = torch.median(k_r)
    # distribution of groups-of-32 kurtosis
    k_32 = get_kurtosis(x.reshape(-1, 32), dim=1, keepdim=True)
    k_32_min, k_32_max = torch.aminmax(k_32)
    k_32_p50 = torch.median(k_32)
    # distribution of groups-of-16 kurtosis
    k_16 = get_kurtosis(x.reshape(-1, 16), dim=1, keepdim=True)
    k_16_min, k_16_max = torch.aminmax(k_16)
    k_16_p50 = torch.median(k_16)
    return TensorStats(
        max_abs,
        avg,
        std,
        k_r_min.item(),
        k_r_p50.item(),
        k_r_max.item(),
        k_32_min.item(),
        k_32_p50.item(),
        k_32_max.item(),
        k_16_min.item(),
        k_16_p50.item(),
        k_16_max.item(),
    )


def enable_kurtosis_logging():
    @torch.library.custom_op("quant_logger::log_tensor", mutates_args=("x",))
    def log_tensor(x: torch.Tensor, fqn: str, op: str, tag: str) -> None:
        counter_val = counter[0]
        counter[0] += 1
        s = get_default_stats_with_kurtosis(x)
        k_r_s = f"k_r: min {s.k_r_min:.1f}, p50 {s.k_r_p50:.1f}, max {s.k_r_max:.1f}"
        k_32_s = (
            f"k_32: min {s.k_32_min:.1f}, p50 {s.k_32_p50:.1f}, max {s.k_32_max:.1f}"
        )
        k_16_s = (
            f"k_16: min {s.k_16_min:.1f}, p50 {s.k_16_p50:.1f}, max {s.k_16_max:.1f}"
        )
        print(
            f"t={tag}, c={counter_val}, {fqn=}, {op=}, max={s.max_abs:.2f}, avg={s.avg:.2f}, std={s.std:.2f}, {k_r_s}, {k_32_s}, {k_16_s}"
        )


def enable_log_stats_to_file(filename):
    headers = ["tag", "counter_val", "fqn", "op"]
    data_fields = [field.name for field in fields(TensorStats)]
    headers.extend(data_fields)
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    @torch.library.custom_op("quant_logger::log_tensor", mutates_args=("x",))
    def log_tensor(x: torch.Tensor, fqn: str, op: str, tag: str) -> None:
        counter_val = counter[0]
        counter[0] += 1
        s = get_default_stats_with_kurtosis(x)
        data = [tag, counter_val, fqn, op, *astuple(s)]
        with open(filename, "a") as f:
            writer = csv.writer(f)
            writer.writerow(data)
