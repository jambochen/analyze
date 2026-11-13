# 滑動視窗的核心計算供正式模擬與小樣本列舉共用。
import numpy as np


def window_sum(x, k):
    """按列順序回傳全部長度 k 視窗的和；跨區段的視窗由呼叫者排除。"""
    c = np.r_[0., np.cumsum(x)]
    return c[k:] - c[:-k]


def summarize(y, expected, variance, windows):
    """同一得分序列在各有效視窗、分組與權重下的連敗率及累積殘差平方。"""
    values = []
    for metric, k, valid, masks, weights in windows:
        if metric == "loss_run":
            z = (window_sum(y == 0, k)[valid] == k).astype(float)
        else:
            z = window_sum(y - expected, k)[valid] ** 2 / window_sum(variance, k)[valid]
        for mask, weight in zip(masks, weights):
            values.append(z[mask].mean() if mask.any() else np.nan)
            values.append(np.dot(z[mask], weight) if mask.any() else np.nan)
    return np.array(values)
