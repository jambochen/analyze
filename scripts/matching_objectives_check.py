# 小型批次枚舉所有合法配法，獨立核對最佳化目標與雙端保護限制。
from itertools import permutations

import numpy as np
import polars as pl

import matching_objectives as mo

rng = np.random.default_rng(2609)
rows = []
for n in range(2, 7):
    ix = np.arange(n)
    for trial in range(20):
        w, b = rng.integers(800, 2500, size=(2, n))
        gap = abs(w[:, None] - b[None, :]).astype(float)
        if trial == 0:
            gap[:] = 0
        allowed = rng.uniform(size=(n, n)) > .35
        np.fill_diagonal(allowed, True)
        if trial == 1:
            allowed = np.eye(n, dtype=bool)
        perm = np.array([x for x in permutations(range(n)) if allowed[ix, x].all()])
        costs = gap[ix[None, :], perm]
        solutions = mo.assignments(gap, allowed)
        original = np.diag(gap)
        for policy, chosen in solutions.items():
            assert len(set(chosen)) == n and allowed[ix, chosen].all()
            selected = gap[ix, chosen]
            harm = (selected > original).sum() + (selected > original[chosen]).sum()
            harms = (costs > original).sum(axis=1) + (costs > original[perm]).sum(axis=1)
            changes = (perm != ix).sum(axis=1)
            if policy == "rating":
                actual, expected = (selected.sum(),), (costs.sum(axis=1).min(),)
            elif policy == "rating_guard":
                actual = (selected.sum(), (selected > original).sum() + (selected > original[chosen]).sum(), (chosen != ix).sum())
                expected = min(zip(costs.sum(axis=1), (costs > original).sum(axis=1) + (costs > original[perm]).sum(axis=1), (perm != ix).sum(axis=1)))
            elif policy.startswith("tail"):
                threshold = int(policy[4:])
                actual = ((selected >= threshold).sum(), selected.sum(), harm, (chosen != ix).sum())
                expected = min(zip((costs >= threshold).sum(axis=1), costs.sum(axis=1), harms, changes))
            elif policy == "minimax":
                actual = (selected.max(), selected.sum(), harm, (chosen != ix).sum())
                expected = min(zip(costs.max(axis=1), costs.sum(axis=1), harms, changes))
            else:
                tolerance = int(policy[3:])
                keep = ((costs <= original + tolerance) & (costs <= original[perm] + tolerance)).all(axis=1)
                assert (selected <= original + tolerance).all() and (selected <= original[chosen] + tolerance).all()
                actual = (selected.sum(), harm, (chosen != ix).sum())
                expected = min(zip(costs[keep].sum(axis=1), harms[keep], changes[keep]))
            assert actual == expected, (n, trial, policy, actual, expected)
            rows.append(dict(n=n, trial=trial, policy=policy, feasible_assignments=len(perm), exact=True))
pl.DataFrame(rows).write_csv("outputs/tables/matching_objectives_check.csv")
print(pl.DataFrame(rows).group_by("policy").agg(cases=pl.len(), exact=pl.col("exact").all()))
