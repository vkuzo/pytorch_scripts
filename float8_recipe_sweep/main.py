import copy
import fire
import itertools

import pandas as pd

import torch
import torch.nn as nn
import torchao

from torchao.float8 import Float8LinearConfig, convert_to_float8_training
from torchao.float8.float8_utils import compute_error

torch.manual_seed(0)

@torch.no_grad()
def mean_absolute_error(x_ref, x):
    return torch.mean(torch.abs(x - x_ref))

@torch.no_grad()
def mean_absolute_percentage_error(x_ref, x):
    tmp = torch.abs(x_ref - x) / torch.clamp(torch.abs(x_ref), min=1e-9)
    # trim to avoid values close to 0 from 
    # significantly impacting the results
    tmp = torch.clamp(tmp, max=1e3)
    return torch.mean(tmp)

def run():

    # M, K, N = 1024, 1024, 1024
    # M, K, N = 16, 32, 64
    M, K, N = 1024, 179840, 1024

    Ks = [2048, 8192, 32768, 131072]
    outlier_vals = [1000, 10000, 100000]
    x_outlier_val = 1.0
    # w_outlier_val = 1000.0
    go_outlier_val = 1.0

    headers = ['M', 'K', 'N', 'w_outlier_val', 'recipe', 'o_mae', 'gi_mae', 'gw_mae', 'o_mape', 'gi_mape', 'gw_mape']
    results = []

    for K, w_outlier_val in itertools.product(Ks, outlier_vals):

        print(f"M {M}, K {K}, N {N}, x_outlier_val {x_outlier_val}, w_outlier_val {w_outlier_val}, go_outlier_val {go_outlier_val}")

        x_ref = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        # add an outlier
        x_ref[0][0] = x_outlier_val
        x_ref.requires_grad_()

        m_ref = nn.Linear(K, N, bias=False, dtype=torch.bfloat16).cuda()
        # add an outlier
        with torch.no_grad():
            m_ref.weight[0][0] = w_outlier_val
            pass

        go_ref = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        # add an outlier
        go_ref[0][0] = go_outlier_val

        x_ref_copy = copy.deepcopy(x_ref)
        m_ref_copy = copy.deepcopy(m_ref)
        y_ref = m_ref_copy(x_ref_copy)
        y_ref.backward(go_ref)
        # print('y_ref', y_ref[0])

        # for recipe_name in ('tensorwise', 'rowwise'):
        # for recipe_name in ('rowwise',):
        # for recipe_name in ('rowwise_with_gw_hp',):


        for recipe_name in ('tensorwise', 'rowwise', 'rowwise_with_gw_hp'):
            x = copy.deepcopy(x_ref)
            m = copy.deepcopy(m_ref)
            go = copy.deepcopy(go_ref)

            config = Float8LinearConfig.from_recipe_name(recipe_name)
            m = convert_to_float8_training(m, config=config)

            y = m(x)
            y.backward(go)
            # print('y', y[0])

            o_sqnr = compute_error(y_ref, y).item()
            gi_sqnr = compute_error(x_ref_copy.grad, x.grad).item()
            gw_sqnr = compute_error(m_ref_copy.weight.grad, m.weight.grad).item()

            o_mae = mean_absolute_error(y_ref, y).item()
            gi_mae = mean_absolute_error(x_ref_copy.grad, x.grad).item()
            gw_mae = mean_absolute_error(m_ref_copy.weight.grad, m.weight.grad).item()

            o_mape = mean_absolute_percentage_error(y_ref, y).item()
            gi_mape = mean_absolute_percentage_error(x_ref_copy.grad, x.grad).item()
            gw_mape = mean_absolute_percentage_error(m_ref_copy.weight.grad, m.weight.grad).item()

            print(recipe_name)
            print(f'o_mae: {o_mae}, gi_mae: {gi_mae}, gw_mae: {gw_mae}')
            print(f'o_mape: {o_mape}, gi_mape: {gi_mape}, gw_mape: {gw_mape}')

            results.append([M, K, N, w_outlier_val, recipe_name, o_mae, gi_mae, gw_mae, o_mape, gi_mape, gw_mape])

    print(results)

    df = pd.DataFrame(results, columns=headers)
    print(df)


if __name__ == "__main__":
    fire.Fire(run)
