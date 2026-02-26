import torch

from quant_logger_kurtosis import get_kurtosis_scipy, get_kurtosis


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
