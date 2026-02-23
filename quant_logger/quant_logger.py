import csv
import os
import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
import scipy
from dataclasses import dataclass, fields, astuple

from torchao.utils import TorchAOBaseTensor

aten = torch.ops.aten

counter = [0]


def reset_counter():
    counter[0] = 0


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


def get_default_stats(x: torch.Tensor) -> TensorStats:
    # max_abs
    max_abs = torch.max(torch.abs(x))
    # mean
    avg = torch.mean(x)
    # std
    std = torch.std(x, correction=0)
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
        max_abs.item(),
        avg.item(),
        std.item(),
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


@torch.library.custom_op("quant_logger::log_tensor", mutates_args=("x",))
def log_tensor(x: torch.Tensor, fqn: str, op: str, tag: str) -> None:
    """
    User can redefine this function in their code to customize logging (write to file,
    log custom stats, etc).
    """
    # counter
    counter_val = counter[0]
    counter[0] += 1

    s = get_default_stats(x)

    k_r_s = f"k_r: min {s.k_r_min:.1f}, p50 {s.k_r_p50:.1f}, max {s.k_r_max:.1f}"

    # distribution of groups-of-32 kurtosis
    k_32_s = f"k_32: min {s.k_32_min:.1f}, p50 {s.k_32_p50:.1f}, max {s.k_32_max:.1f}"

    # distribution of groups-of-16 kurtosis
    k_16_s = f"k_16: min {s.k_16_min:.1f}, p50 {s.k_16_p50:.1f}, max {s.k_16_max:.1f}"

    print(
        f"t={tag}, c={counter_val}, {fqn=}, {op=}, max={s.max_abs:.2f}, avg={s.avg:.2f}, std={s.std:.2f}, {k_r_s}, {k_32_s}, {k_16_s}"
    )


@log_tensor.register_fake
def _(x: torch.Tensor, fqn: str, op: str, tag: str) -> None:
    pass


# convenience overrides


# save entire tensors to disk
def enable_log_tensor_save_tensors_to_disk(save_dir):
    os.makedirs(save_dir, exist_ok=True)

    @torch.library.custom_op("quant_logger::log_tensor", mutates_args=("x",))
    def log_tensor(x: torch.Tensor, fqn: str, op: str, tag: str) -> None:
        filename = f"{fqn}_{op}_{tag}.pt"
        # Replace invalid path characters
        filename = filename.replace("/", "_").replace(":", "_")
        filepath = os.path.join(save_dir, filename)
        torch.save(x.clone(), filepath)
        print(f"Saved tensor to {filepath}")


# save defaults stats to csv file
def enable_log_stats_to_file(filename):
    # write the headers
    headers = ["tag", "counter_val", "fqn", "op"]
    data_fields = [field.name for field in fields(TensorStats)]
    headers.extend(data_fields)
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    @torch.library.custom_op("quant_logger::log_tensor", mutates_args=("x",))
    def log_tensor(x: torch.Tensor, fqn: str, op: str, tag: str) -> None:
        # counter
        counter_val = counter[0]
        counter[0] += 1
        s = get_default_stats(x)
        data = [tag, counter_val, fqn, op, *astuple(s)]
        with open(filename, "a") as f:
            writer = csv.writer(f)
            writer.writerow(data)


# kurtosis


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


class ActivationLoggingTensor(TorchAOBaseTensor):
    tensor_data_names = ["original_weight_tensor"]
    tensor_attribute_names = ["fqn"]

    def __new__(
        cls,
        original_weight_tensor: torch.Tensor,
        fqn: str,
    ):
        kwargs = {}
        dtype = original_weight_tensor.dtype
        kwargs["dtype"] = dtype
        kwargs["requires_grad"] = False
        kwargs["device"] = original_weight_tensor.device
        shape = original_weight_tensor.shape
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)  # type: ignore[attr-defined]

    def __init__(
        self,
        original_weight_tensor: torch.Tensor,
        fqn: str,
    ):
        self.original_weight_tensor = original_weight_tensor
        self.fqn = fqn

    @classmethod
    def from_float(
        cls,
        input_float: torch.Tensor,
    ):
        return cls(input_float)


implements = ActivationLoggingTensor.implements


@implements(aten.addmm.default)
def _(func, types, args, kwargs):
    bias, x, w = args
    torch.ops.quant_logger.log_tensor(x, w.fqn, str(func), "act")
    out = func(bias, x, w.original_weight_tensor, **kwargs)
    return out


@implements(aten.mm.default)
def _(func, types, args, kwargs):
    x, w = args
    torch.ops.quant_logger.log_tensor(x, w.fqn, str(func), "act")
    out = func(x, w.original_weight_tensor, **kwargs)
    return out


@implements(aten.t.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func, args, kwargs, args[0]._apply_fn_to_data(torch.t)
    )


def add_activation_loggers(model: torch.nn.Module):
    fqn_to_module = dict(model.named_modules())
    for fqn, parameter in model.named_parameters():
        parent_fqn = ".".join(fqn.split(".")[:-1])
        child_fqn = fqn.split(".")[-1]
        # TODO(future PR): support biases and embeddings, layer_norm, etc
        if (child_fqn == "bias") or ("embed" in fqn) or ("layer_norm" in fqn):
            continue
        parent_module = fqn_to_module[parent_fqn]
        # for now just handle linears
        if not isinstance(parent_module, torch.nn.Linear):
            continue
        new_parameter = torch.nn.Parameter(ActivationLoggingTensor(parameter, fqn))
        setattr(parent_module, child_fqn, new_parameter)


def log_parameter_info(model: torch.nn.Module):
    for fqn, parameter in model.named_parameters():
        torch.ops.quant_logger.log_tensor(parameter, fqn, "", "param")
