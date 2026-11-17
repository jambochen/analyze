# 只彙整已保存的局前預測；合併區間保留跨時段共享玩家的相依。
import json
from pathlib import Path

import numpy as np
import polars as pl

import likelihood as lk

specs = json.loads(Path('outputs/tables/likelihood_selection_models.json').read_text())
labels = {'baseline': '評分基準', 'results': '上一局結果',
          'rating_change': '評分變動', 'recent': '上一局結果＋評分變動'}
pl.DataFrame([{'speed': s['speed'], 'fold': s['fold'],
               'model': s['aliases']['recent'], 'label': labels[s['aliases']['recent']]}
              for s in specs]).write_csv('outputs/tables/baseline_report_selection.csv')

rows = []
original = pl.read_csv('outputs/tables/likelihood_selection_prediction.csv')
pooled = pl.read_csv('outputs/tables/hierarchical_prediction_pooled.csv')
trueskill = pl.read_csv('outputs/tables/trueskill_report_comparison.csv')
for speed in ['Bullet', 'Blitz', 'Classical']:
    parts = []
    for fold in ['Jul-Sep', 'Oct-Dec']:
        d = pl.read_parquet(f'data/processed/hierarchical_{speed}_{fold}_slope_prediction.parquet')
        ts = pl.read_parquet(f'data/processed/trueskill_{speed}_{fold}_prediction.parquet')
        intercept = pl.read_parquet(f'data/processed/hierarchical_{speed}_{fold}_intercept_prediction.parquet')
        keys = ['game_id', 'a_id', 'b_id', 'y']
        assert d.select(keys).equals(ts.select(keys))
        assert d.select(keys).equals(intercept.select(keys))
        for model in ['baseline', 'recent']:
            assert np.allclose(d[f'{model}_score_mse'], ts[f'{model}_score_mse'], atol=1e-13, rtol=0)
        ref = original.filter(pl.col('speed') == speed, pl.col('fold') == fold,
                              pl.col('model') == 'recent', pl.col('vs') == 'baseline',
                              pl.col('metric') == 'score_mse', pl.col('group') == 'all').row(0, named=True)
        assert d.height == ref['n']
        assert np.isclose((d['recent_score_mse']-d['baseline_score_mse']).mean(), ref['diff'], atol=1e-13, rtol=0)
        parts.append(d.select('game_id', 'a_id', 'b_id', 'baseline_score_mse', 'recent_score_mse', 'slope_score_mse').with_columns(
            ts['trueskill_score_mse'], intercept['intercept_score_mse']))
    d = pl.concat(parts)
    assert d.select(pl.struct('game_id', 'a_id').n_unique()).item() == d.height
    for model, reference in [('recent', 'baseline'), ('slope', 'recent'),
                             ('trueskill', 'recent'), ('trueskill', 'slope'), ('slope', 'intercept')]:
        delta = (d[f'{model}_score_mse']-d[f'{reference}_score_mse']).to_numpy()
        diff = delta.mean()
        se = lk.dyad_se((delta-diff)/len(delta), d['a_id'].to_numpy(), d['b_id'].to_numpy())
        reference_mean = d[f'{reference}_score_mse'].mean()
        row = {'speed': speed, 'model': model, 'vs': reference, 'n': d.height,
               'mean': d[f'{model}_score_mse'].mean(), 'reference_mean': reference_mean,
               'diff': diff, 'se': se, 'lo': diff-1.96*se, 'hi': diff+1.96*se,
               'reduction_percent': -100*diff/reference_mean}
        if model == 'slope':
            check = pooled.filter(pl.col('speed') == speed, pl.col('kind') == model,
                                  pl.col('vs') == reference, pl.col('metric') == 'score_mse').row(0, named=True)
            assert all(np.isclose(row[k], check[k], atol=1e-13, rtol=0) for k in ['diff', 'se', 'lo', 'hi'])
        if model == 'trueskill' and reference == 'recent':
            check = trueskill.filter(pl.col('speed') == speed, pl.col('vs') == reference,
                                     pl.col('metric') == 'score_mse').row(0, named=True)
            assert all(np.isclose(row[k], check[k], atol=1e-13, rtol=0) for k in ['diff', 'se', 'lo', 'hi'])
        rows.append(row)
out = pl.DataFrame(rows)
out.write_csv('outputs/tables/baseline_report_comparison.csv')
print(out)
