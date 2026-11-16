# 雙方共用訓練期節點，模型匯出與重建共用同一個自然三次樣條轉換。
import numpy as np
from patsy import cr


def basis(values, knots):
    """固定節點的五維自然三次樣條基底，邊界以外線性外推。"""
    return np.asarray(cr(np.asarray(values), knots=np.asarray(knots)[1:-1], lower_bound=knots[0], upper_bound=knots[-1]))


def design(d, knots, center):
    """雙方相同中心化轉換之差；移除最後一欄以排除常數造成的秩不足。"""
    return ((basis(d["rd"].to_numpy(), knots) - center) - (basis(d["opp_rd"].to_numpy(), knots) - center))[:, :-1]


def probabilities(d, model):
    """還原已含固定基準 eta/width 的樣本之局前機率；model 為匯出 JSON 的分支設定。"""
    import ordinal

    eta = d["eta"].to_numpy().copy()
    for name, value in zip(model["columns"], model["coef"]):
        eta += d[name].to_numpy() * value
    if model["spline_coef"]:
        eta += design(d, model["knots"], np.asarray(model["center"])) @ np.asarray(model["spline_coef"])
    return ordinal.probs(eta, d["width"].to_numpy())
