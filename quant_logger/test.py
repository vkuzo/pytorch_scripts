import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_logger import (
    add_activation_loggers,
    log_parameter_info,
    reset_counter,
    enable_log_tensor_save_tensors_to_disk,
    enable_log_stats_to_file,
    get_kurtosis_scipy,
    get_kurtosis,
)

from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig

from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)


def get_toy_model(dim1, dim2):
    return nn.Sequential(
        nn.Linear(dim1, dim2, bias=False),
        nn.ReLU(),
        nn.Linear(dim2, dim1, bias=False),
    )


class ModelWithLoop(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x0):
        x1 = self.fc(x0)
        x2 = F.relu(x1)
        x3 = self.fc(x2)
        return x3


def test_hello_world():
    M, K, N = 2, 32, 64
    x = torch.randn(M, K)
    m = get_toy_model(K, N)
    add_activation_loggers(m)
    m(x)


def test_loop():
    dim = 32
    x = torch.randn(dim, dim)
    m = ModelWithLoop(dim)

    log_parameter_info(m)
    reset_counter()

    add_activation_loggers(m)
    m(x)


def test_log_parameter_info():
    K, N = 4, 32
    m = get_toy_model(K, N)
    log_parameter_info(m)


def test_custom_logging_fn():
    @torch.library.custom_op("quant_logger::log_tensor", mutates_args=("x",))
    def log_tensor(x: torch.Tensor, fqn: str, op: str, tag: str) -> None:
        min_val = torch.min(x)
        print(f"{tag=}, {fqn=}, {min_val=}")

    M, K, N = 2, 4, 6
    x = torch.randn(M, K)
    m = get_toy_model(K, N)
    add_activation_loggers(m)
    m(x)


def test_custom_logging_fn_save_tensors():
    save_dir = os.path.expanduser("~/tmp/20260102_activations_test/")
    enable_log_tensor_save_tensors_to_disk(save_dir)

    M, K, N = 2, 4, 6
    x = torch.randn(M, K)
    m = get_toy_model(K, N)
    log_parameter_info(m)
    add_activation_loggers(m)
    m(x)


def test_custom_logging_fn_save_stats():
    filename = os.path.expanduser("~/tmp/20260106_test.csv")
    enable_log_stats_to_file(filename)

    M, K, N = 2, 32, 64
    x = torch.randn(M, K)
    m = get_toy_model(K, N)
    log_parameter_info(m)
    add_activation_loggers(m)
    m(x)

    with open(filename, "r") as f:
        print(f.readlines())


def test_quantized_model():
    M, K, N = 32, 64, 128
    x = torch.randn(M, K, device="cuda")
    m = get_toy_model(K, N).cuda()

    quantize_(m, Float8DynamicActivationFloat8WeightConfig())

    add_activation_loggers(m)
    m(x)


def test_opt_125m():
    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(model)

    filename = os.path.expanduser("~/tmp/20260106_test.csv")
    enable_log_stats_to_file(filename)

    log_parameter_info(model)
    add_activation_loggers(model)

    prompt = "Hello, world!"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=1)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def test_kurtosis_scipy():
    unit_normal = torch.randn(8192 * 8192, device="cuda")
    k = get_kurtosis_scipy(unit_normal)
    assert k < 0.001


def test_kurtosis_pt():
    # unit normal
    unit_normal = torch.randn(8192 * 8192, device="cuda")
    k = get_kurtosis(unit_normal)
    k_ref = get_kurtosis_scipy(unit_normal)
    assert k < 0.001

    # matches scipy on a tensor with an outlier
    x = torch.randn(16)
    x[0] = 100
    k_ref = get_kurtosis_scipy(x)
    k = get_kurtosis(x).item()
    assert abs(k_ref - k) < 0.001

    # calculating kurtosis over the second dim of a 2d tensor works correctly
    x2 = torch.randn(2, 16)
    x2[0][0] = 5.0
    x2[1][0] = 15.0
    k_ref = [get_kurtosis_scipy(x2[0]), get_kurtosis_scipy(x2[1])]
    k = get_kurtosis(x2, dim=1, keepdim=True)
    assert abs(k_ref[0] - k[0]) < 0.001
    assert abs(k_ref[1] - k[1]) < 0.001
