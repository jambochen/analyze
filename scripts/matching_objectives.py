# 固定白黑方的替代配法；目標與保護限制以可觀察的局前分差定義。
import numpy as np
from scipy.optimize import linear_sum_assignment


def assignments(gap, allowed):
    """平均、懸殊局數、批內最壞分差及個人惡化上限的精確指派。"""
    n = len(gap)
    ix = np.arange(n)
    original = np.diag(gap)
    result = {"rating": linear_sum_assignment(np.where(allowed, gap, np.inf))[1]}
    harm = (gap > original[:, None]).astype(int) + (gap > original[None, :]).astype(int)
    changed = 1 - np.eye(n)
    # 分差為整數，先保留最小總分差，再最少惡化玩家，最後最少換對手。
    guard_cost = gap * (2 * n + 1) * (n + 1) + harm * (n + 1) + changed
    result["rating_guard"] = linear_sum_assignment(np.where(allowed, guard_cost, np.inf))[1]
    # 整數權重依序保留超標局數、總分差、惡化人數與換對手數。
    for threshold in [200, 400]:
        cost = (gap >= threshold) * (n * guard_cost.max() + 1) + guard_cost
        assert n * cost.max() < 2**53
        result[f"tail{threshold}"] = linear_sum_assignment(np.where(allowed, cost, np.inf))[1]
    # 門檻可行性具單調性；原配法的最大分差是已知可行上界。
    levels = np.unique(gap[allowed & (gap <= original.max())])
    lo, hi = 0, len(levels) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        feasible = allowed & (gap <= levels[mid])
        trial = linear_sum_assignment((~feasible).astype(float))[1]
        if feasible[ix, trial].all():
            hi = mid
        else:
            lo = mid + 1
    result["minimax"] = linear_sum_assignment(np.where(allowed & (gap <= levels[lo]), guard_cost, np.inf))[1]
    # 同時保護邊的兩端，各自比較自己原對手的分差；原配法保證可行。
    for tolerance in [0, 50, 100]:
        protected = allowed & (gap <= original[:, None] + tolerance) & (gap <= original[None, :] + tolerance)
        result[f"cap{tolerance}"] = linear_sum_assignment(np.where(protected, guard_cost, np.inf))[1]
    return result
