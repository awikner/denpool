from statistics import median
from math import ceil, sqrt
import pandas as pd
import torch

def df_to_nprec(df, var_name, value_name):
    return (pd.melt(df, var_name = var_name, value_name = value_name).set_index(var_name).to_records())


def median_confidence(data, q = 0.5, z = 1.96):
    data.sort()
    n = len(data)
    j = ceil(n*q - z*sqrt(n*q*(1-q)))
    k = ceil(n*q + z*sqrt(n*q*(1-q)))
    med_val = median(data)
    low_bnd, up_bnd = data[j], data[k]
    confidence = max(abs(med_val - low_bnd), abs(med_val - up_bnd))
    return med_val, confidence

def interval_score(y_hat, y, alphas):
    l = y_hat[:, :, 1:alphas.size(-1) + 1]
    u = torch.flip(y_hat[:, :, alphas.size(-1) + 1:], [2])
    l_mask = y < l
    u_mask = y > u
    interval_scores = (u - l) + 2.0 / alphas * (l - y) * l_mask + 2.0 / alphas * (y - u) * u_mask
    return interval_scores

def weighted_interval_score(y_hat, y, alphas):
    median_scores = torch.abs(y - y_hat[:, :, 0].unsqueeze(-1))
    interval_scores = interval_score(y_hat, y, alphas)
    loss = 0.5 * median_scores + torch.sum(0.5 * alphas * interval_scores, dim=2, keepdim=True)
    return 1 / (alphas.size(-1) + 0.5) * loss.mean()


