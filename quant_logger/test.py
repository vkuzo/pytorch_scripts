import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_logger import (
    add_activation_loggers,
    log_parameter_info,
    reset_counter,
    enable_log_tensor_save_tensors_to_disk,
    enable_log_stats_to_file,
)

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


def test_log_activations_simple(capsys):
    reset_counter()
    M, K, N = 2, 32, 64
    x = torch.randn(M, K)
    m = get_toy_model(K, N)
    add_activation_loggers(m)
    m(x)

    captured = capsys.readouterr()
    assert "t=act, c=0, fqn='0.weight', op='linear'" in captured.out


def test_log_parameter_info_simple(capsys):
    reset_counter()
    K, N = 4, 32
    m = get_toy_model(K, N)
    log_parameter_info(m)

    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")
    assert "t=param, c=0, fqn='0.weight', op=''," in lines[0]
    assert "t=param, c=1, fqn='2.weight', op=''," in lines[1]


def test_loop_simple(capsys):
    reset_counter()
    dim = 32
    x = torch.randn(dim, dim)
    m = ModelWithLoop(dim)

    log_parameter_info(m)
    reset_counter()

    add_activation_loggers(m)
    m(x)

    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")
    assert "t=param, c=0, fqn='fc.weight', op=''," in lines[0]
    assert "t=param, c=1, fqn='fc.bias', op=''," in lines[1]
    assert "t=act, c=0, fqn='fc.weight', op='linear'," in lines[2]
    assert "t=act, c=1, fqn='fc.weight', op='linear'," in lines[3]


def test_custom_logging_fn(capsys):
    @torch.library.custom_op("quant_logger::log_tensor", mutates_args=("x",))
    def log_tensor(x: torch.Tensor, fqn: str, op: str, tag: str) -> None:
        min_val = torch.min(x)
        print(f"custom {tag=}, {fqn=}, {min_val=}")

    reset_counter()
    M, K, N = 2, 4, 6
    x = torch.randn(M, K)
    m = get_toy_model(K, N)
    add_activation_loggers(m)
    m(x)

    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")
    assert "custom tag='act', fqn='0.weight'," in lines[0]
    assert "custom tag='act', fqn='2.weight'," in lines[1]


def test_custom_logging_fn_save_tensors(tmp_path):
    save_dir = tmp_path / "activations"
    enable_log_tensor_save_tensors_to_disk(str(save_dir))

    reset_counter()
    M, K, N = 2, 4, 6
    x = torch.randn(M, K)
    m = get_toy_model(K, N)
    log_parameter_info(m)
    add_activation_loggers(m)
    m(x)

    # Check saved tensors can be loaded from disk
    param_0 = torch.load(save_dir / "0.weight__param.pt")
    param_2 = torch.load(save_dir / "2.weight__param.pt")
    act_0 = torch.load(save_dir / "0.weight_linear_act.pt")
    act_2 = torch.load(save_dir / "2.weight_linear_act.pt")

    assert param_0.shape == (N, K)
    assert param_2.shape == (K, N)
    assert act_0.shape == (M, K)
    assert act_2.shape == (M, N)


def test_custom_logging_fn_save_stats(tmp_path):
    filename = tmp_path / "stats.csv"
    enable_log_stats_to_file(str(filename))

    reset_counter()
    M, K, N = 2, 32, 64
    x = torch.randn(M, K)
    m = get_toy_model(K, N)
    log_parameter_info(m)
    add_activation_loggers(m)
    m(x)

    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 4
    assert rows[0]["tag"] == "param" and rows[0]["fqn"] == "0.weight"
    assert rows[1]["tag"] == "param" and rows[1]["fqn"] == "2.weight"
    assert rows[2]["tag"] == "act" and rows[2]["fqn"] == "0.weight"
    assert rows[3]["tag"] == "act" and rows[3]["fqn"] == "2.weight"


def test_opt_125m(tmp_path):
    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    filename = tmp_path / "stats.csv"
    enable_log_stats_to_file(str(filename))

    reset_counter()
    log_parameter_info(model)
    add_activation_loggers(model)

    prompt = "Hello, world!"
    inputs = tokenizer(prompt, return_tensors="pt")
    _outputs = model.generate(**inputs, max_new_tokens=1)

    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check at least one parameter was logged
    param_rows = [r for r in rows if r["tag"] == "param"]
    assert len(param_rows) > 0
    assert param_rows[0]["fqn"] == "model.decoder.embed_tokens.weight"

    # Check at least one activation was logged
    act_rows = [r for r in rows if r["tag"] == "act"]
    assert len(act_rows) > 0
    assert act_rows[0]["fqn"] == "model.decoder.layers.0.self_attn.q_proj.weight"
